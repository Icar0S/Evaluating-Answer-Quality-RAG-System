"""Fixtures dos testes de API.

Todos os testes daqui são herméticos: nenhuma chamada real, nenhuma GPU, nenhum
acesso à rede e nenhuma escrita em `logs/` ou `data/` do projeto. É isso que
permite rodá-los no CI do GitHub, onde nada disso existe.

Três isolamentos importam:

1. `Settings` é `@lru_cache` — sem limpar o cache, o primeiro teste congelaria a
   configuração para todos os outros.
2. O `.env` real do usuário (que tem o provider remoto configurado, com chave de
   API de verdade) venceria os defaults e faria o resultado do teste depender da
   máquina. Variáveis de ambiente têm precedência sobre o `.env` no
   pydantic-settings, então sobrescrevemos todas as que importam.
3. `providers` guarda o provider ativo em estado de módulo, que vazaria de um
   teste para o outro.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import metrics, providers
from app.config import get_settings
from app.rag import vector_store

# Hosts propositalmente inexistentes: se algum mock deixar passar uma chamada,
# o teste falha com erro de conexão em vez de acertar um serviço real.
LOCAL_BASE_URL = "http://ollama-local-de-teste:11434"
REMOTE_BASE_URL = "http://api-remota-de-teste"
LOCAL_MODEL = "modelo-local-de-teste"
LOCAL_EMBEDDING_MODEL = "embeddings-local-de-teste"
REMOTE_MODEL = "modelo-remoto-de-teste"
REMOTE_API_KEY = "sk-chave-de-teste"
REMOTE_LABEL = "Servidor de teste"


class LlmBackendMock:
    """Backend falso para os dois protocolos que o projeto fala (ver app/providers.py):

    - "ollama" (local): /api/tags, /api/chat, /api/embed, /api/embeddings
    - "smartdatatest" (remoto): /v1/ready, /v1/chat, com Bearer auth

    Por padrão o host remoto está *offline* (recusa conexão), reproduzindo o
    cenário real de um servidor de terceiros que pode não estar acessível.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.offline_hosts: set[str] = set()
        self.completion = "Resposta de teste do modelo."
        self.eval_count = 42
        self.eval_duration_ns = 2_000_000_000  # 2s -> 21 tok/s

    def set_offline(self, base_url: str) -> None:
        self.offline_hosts.add(httpx.URL(base_url).host)

    def set_online(self, base_url: str) -> None:
        self.offline_hosts.discard(httpx.URL(base_url).host)

    def requests_to(self, base_url: str) -> list[httpx.Request]:
        host = httpx.URL(base_url).host
        return [r for r in self.requests if r.url.host == host]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        if request.url.host in self.offline_hosts:
            raise httpx.ConnectError("conexao recusada (mock)", request=request)

        path = request.url.path

        # --- protocolo Ollama (local) ---
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": LOCAL_MODEL}]})

        if path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {"content": self.completion},
                    "prompt_eval_count": 100,
                    "eval_count": self.eval_count,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": self.eval_duration_ns,
                    "total_duration": 2_500_000_000,
                },
            )

        if path == "/api/embed":
            payload = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in payload["input"]]})

        if path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

        # --- protocolo smartdatatest (remoto, ver llm-api-referencia.md) ---
        if path == "/v1/ready":
            return httpx.Response(200, json={"status": "ready"})

        if path == "/v1/chat":
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": self.completion,
                    "thinking": None,
                    "model": payload.get("model") or providers.REMOTE_DEFAULT_MODEL,
                    "requested_model": payload.get("model") or providers.REMOTE_DEFAULT_MODEL,
                    "fallback_applied": False,
                    "max_tokens_adjusted": False,
                    "sources": [],
                    "tokens_in": 100,
                    "tokens_out": self.eval_count,
                    "duration_s": self.eval_duration_ns / 1e9,
                    "queue_wait_s": 0.0,
                },
            )

        return httpx.Response(404, json={"error": f"rota nao mockada: {path}"})


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Congela a configuração em valores de teste, independente do .env da máquina."""
    monkeypatch.setenv("OLLAMA_BASE_URL", LOCAL_BASE_URL)
    monkeypatch.setenv("GENERATION_MODEL", LOCAL_MODEL)
    monkeypatch.setenv("EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
    # Vazio = provider remoto não configurado (o default de quem não mexeu no .env).
    monkeypatch.setenv("REMOTE_API_BASE_URL", "")
    monkeypatch.setenv("REMOTE_API_KEY", "")
    monkeypatch.setenv("REMOTE_GENERATION_MODEL", "")
    monkeypatch.setenv("REMOTE_LABEL", REMOTE_LABEL)
    monkeypatch.setenv("ACTIVE_PROVIDER", "local")
    # Caminhos absolutos em tmp: nada é escrito em logs/ ou data/ do projeto.
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))

    get_settings.cache_clear()
    monkeypatch.setattr(providers, "_active_name", None)
    monkeypatch.setattr(metrics, "_cpu_history", [])
    monkeypatch.setattr(metrics, "_gpu_history", [])
    monkeypatch.setattr(metrics, "_last_generation", None)
    monkeypatch.setattr(metrics, "_active_generations", 0)

    yield

    get_settings.cache_clear()


@pytest.fixture
def remote_configured(monkeypatch):
    """Preenche as variáveis do provider remoto (equivale ao .env com a API do Mac mini)."""
    monkeypatch.setenv("REMOTE_API_BASE_URL", REMOTE_BASE_URL)
    monkeypatch.setenv("REMOTE_API_KEY", REMOTE_API_KEY)
    monkeypatch.setenv("REMOTE_GENERATION_MODEL", REMOTE_MODEL)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def ollama(monkeypatch) -> LlmBackendMock:
    """Intercepta todo tráfego httpx do backend e devolve respostas falsas.

    `autouse` de propósito: garante que nenhum teste da suíte consiga fazer uma
    chamada de rede real, mesmo por descuido. Testes que precisam inspecionar ou
    reconfigurar as respostas só pedem a fixture pelo nome. Nome mantido curto
    (`ollama`) por compatibilidade com os testes existentes, mesmo cobrindo os
    dois protocolos agora.
    """
    mock = LlmBackendMock()
    mock.set_offline(REMOTE_BASE_URL)

    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(mock.handle)
        return real_client(*args, **kwargs)

    def build_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(mock.handle)
        return real_async_client(*args, **kwargs)

    # TestClient herda de httpx.Client e resolve a classe base no import, então
    # continua usando a implementação real mesmo com estes patches ativos.
    monkeypatch.setattr(httpx, "Client", build_client)
    monkeypatch.setattr(httpx, "AsyncClient", build_async_client)
    return mock


@pytest.fixture(autouse=True)
def _stub_vector_store(monkeypatch):
    """Evita subir o ChromaDB: o alvo destes testes é o roteamento por provider."""
    monkeypatch.setattr(vector_store, "collection_count", lambda: 42)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda embedding, top_k: {
            "ids": [["doc.pdf::p1::c0"]],
            "documents": [["Trecho de teste sobre avaliacao de RAG."]],
            "metadatas": [[{"document": "doc.pdf", "page": 1}]],
            "distances": [[0.25]],
        },
    )


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)
