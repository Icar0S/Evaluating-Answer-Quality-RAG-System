"""Fixtures dos testes do benchmark: herméticos como os de tests/api (sem Ollama, sem Chroma, sem .env da máquina)."""
from __future__ import annotations

import pytest

from app.config import get_settings, set_settings_overrides

# Os campos que mudam o que a recuperação e a geração fazem. Variável de ambiente ganha do .env no
# pydantic-settings: com elas fixadas, o .env da máquina não muda o resultado de um teste.
_ENV = {
    "OLLAMA_BASE_URL": "http://ollama-local-de-teste:11434",
    "EMBEDDING_MODEL": "embeddings-local-de-teste",
    "GENERATION_MODEL": "modelo-local-de-teste",
    "ACTIVE_PROVIDER": "local",
    "REMOTE_API_BASE_URL": "",
    "REMOTE_API_KEY": "",
    "RAG_INDEX": "",
    "TOP_K": "4",
    "MIN_SIMILARITY_SCORE": "0.0",
    "RETRIEVAL_USE_MMR": "false",
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    set_settings_overrides({})
    yield
    set_settings_overrides({})
    get_settings.cache_clear()
