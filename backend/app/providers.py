"""Registro dos alvos de geração (local / remoto) e qual está ativo no momento.

Local e remoto NÃO falam o mesmo protocolo:

- "local": Ollama de verdade (base_url + /api/chat + /api/tags), roda na máquina
  do backend.
- "remote": API própria do Mac mini em https://llm.smartdatatest.com (ver
  llm-api-referencia.md) — /v1/chat com auth Bearer, /v1/ready para liveness, sem
  endpoint de embeddings. `kind` em Provider é o que diferencia os dois em
  generation.py e em check_reachable().

Importante: **embeddings sempre usam o Ollama local**, nunca o provider ativo — a
API remota não expõe rota de embeddings (só usa nomic-embed-text internamente
para o RAG dela própria, via /v1/corpora), e misturar espaços de embedding entre
providers corromperia silenciosamente a busca por similaridade. Trocar de
provider troca só a geração. Ver app/rag/ingestion.py.

O provider "ativo" é estado em memória (thread-safe via lock, mesmo padrão de
app/metrics.py); reinicia a partir de ACTIVE_PROVIDER do .env a cada boot do
processo.
"""
from __future__ import annotations

from typing import Literal

import threading

import httpx
from pydantic import BaseModel

from app.config import get_settings

# Default documentado da API remota quando nenhum modelo é pedido explicitamente
# (ver GET /v1/models em llm-api-referencia.md).
REMOTE_DEFAULT_MODEL = "qwen3:4b"


class Provider(BaseModel):
    name: str
    label: str
    base_url: str
    generation_model: str
    embedding_model: str
    kind: Literal["ollama", "smartdatatest"] = "ollama"
    api_key: str | None = None


def _build_providers() -> dict[str, Provider]:
    settings = get_settings()
    providers: dict[str, Provider] = {
        "local": Provider(
            name="local",
            label="Local",
            base_url=settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            kind="ollama",
        )
    }
    if settings.remote_api_base_url:
        providers["remote"] = Provider(
            name="remote",
            label=settings.remote_label,
            base_url=settings.remote_api_base_url,
            generation_model=settings.remote_generation_model or REMOTE_DEFAULT_MODEL,
            # Embeddings sempre vêm do Ollama local — a API remota não expõe essa rota.
            # Mantido aqui só para exibição/log ("com que embedding o contexto foi montado").
            embedding_model=settings.embedding_model,
            kind="smartdatatest",
            api_key=settings.remote_api_key,
        )
    return providers


_lock = threading.Lock()
_active_name: str | None = None  # resolvido lazy na 1a chamada, a partir do .env


def list_providers() -> list[Provider]:
    return list(_build_providers().values())


def get_active_provider() -> Provider:
    global _active_name
    providers = _build_providers()
    with _lock:
        if _active_name is None or _active_name not in providers:
            settings = get_settings()
            _active_name = settings.active_provider if settings.active_provider in providers else "local"
        name = _active_name
    return providers[name]


def get_active_provider_name() -> str:
    return get_active_provider().name


def set_active_provider(name: str) -> Provider:
    global _active_name
    providers = _build_providers()
    if name not in providers:
        raise ValueError(f"Provider '{name}' não configurado.")
    with _lock:
        _active_name = name
    return providers[name]


async def check_reachable(provider: Provider) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if provider.kind == "smartdatatest":
                # /v1/ready é deliberadamente público (doc: "é o alvo do monitor
                # externo") — sem Bearer, ao contrário do resto da API.
                resp = await client.get(f"{provider.base_url}/v1/ready")
            else:
                resp = await client.get(f"{provider.base_url}/api/tags")
            resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False
