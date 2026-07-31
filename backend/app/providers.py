"""Registro dos alvos Ollama (local / remoto) e qual está ativo no momento.

Local e remoto são endpoints Ollama completos e independentes (base_url + modelo de
geração + modelo de embeddings) — trocar o ativo redireciona tanto /chat quanto
/ingest, sem precisar reiniciar o backend nem editar o .env. O provider "ativo" é
estado em memória (thread-safe via lock, mesmo padrão de app/metrics.py); reinicia
a partir de ACTIVE_PROVIDER do .env a cada boot do processo.
"""
from __future__ import annotations

import threading

import httpx
from pydantic import BaseModel

from app.config import get_settings


class Provider(BaseModel):
    name: str
    label: str
    base_url: str
    generation_model: str
    embedding_model: str


def _build_providers() -> dict[str, Provider]:
    settings = get_settings()
    providers: dict[str, Provider] = {
        "local": Provider(
            name="local",
            label="Local",
            base_url=settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
        )
    }
    if settings.remote_ollama_base_url:
        providers["remote"] = Provider(
            name="remote",
            label=settings.remote_label,
            base_url=settings.remote_ollama_base_url,
            generation_model=settings.remote_generation_model or settings.generation_model,
            embedding_model=settings.remote_embedding_model or settings.embedding_model,
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
            resp = await client.get(f"{provider.base_url}/api/tags")
            resp.raise_for_status()
        return True
    except httpx.HTTPError:
        return False
