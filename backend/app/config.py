"""Configuração centralizada da aplicação, carregada de variáveis de ambiente/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (dois níveis acima de backend/app/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    generation_timeout_seconds: int = 120

    # Provider remoto (opcional — API própria do Mac mini, ver llm-api-referencia.md).
    # NÃO é Ollama-compatível: /v1/chat com Bearer, sem rota de embeddings (por
    # isso não há remote_embedding_model — embeddings são sempre locais).
    remote_api_base_url: str | None = None
    remote_api_key: str | None = None
    remote_generation_model: str | None = None
    remote_label: str = "Servidor (Mac mini)"

    # Provider ativo ao iniciar o backend (local | remote); trocável em runtime via /providers/active
    active_provider: str = "local"

    # Chunking
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    # Desloca todas as fronteiras de chunk em N tokens (o primeiro chunk da
    # pagina fica menor, os seguintes ficam alinhados N tokens adiante). 0 = sem
    # deslocamento, que e o comportamento historico. Existe para o operador de
    # mutacao K4 (tests/mutation/), que injeta "corte no meio de regra".
    chunk_boundary_offset_tokens: int = 0

    # Retrieval
    top_k: int = 4
    # Piso de similaridade (0..1) para um chunk entrar no contexto. 0.0 = sem
    # filtro. Acima de 0, uma pergunta fora do corpus tende a chegar na geracao
    # com contexto vazio, o que dispara a frase de abstencao em vez de uma
    # resposta apoiada em trecho irrelevante.
    min_similarity_score: float = 0.0
    # MMR (Maximal Marginal Relevance) na selecao final dos chunks: busca
    # top_k * retrieval_fetch_multiplier candidatos e escolhe top_k balanceando
    # relevancia e diversidade. Desligado por padrao para preservar o
    # comportamento historico do assistente; a configuracao do estudo de
    # mutacao (tests/mutation/config/study.yaml) liga.
    retrieval_use_mmr: bool = False
    retrieval_mmr_lambda: float = 0.7
    retrieval_fetch_multiplier: int = 3

    # Paths (relativos à raiz do projeto)
    source_pdfs_dir: str = "data/source_pdfs"
    vector_store_dir: str = "data/vector_store"
    vector_store_collection: str = "rag_test_specialist"
    logs_dir: str = "logs"

    # Geracao
    # None = nao envia `options` ao Ollama (mantem o default do modelo, que e o
    # comportamento historico). Fixar em 0.0 nao torna a geracao deterministica,
    # mas evita amplificar a variancia -- e o que o protocolo do estudo pede.
    generation_temperature: float | None = None
    generation_seed: int | None = None

    # Prompt: cada bloco de regra do system prompt e ligavel/desligavel, porque
    # os operadores de mutacao P1-P3 removem exatamente um bloco cada.
    # Os defaults reproduzem o prompt historico (v1).
    prompt_require_grounding: bool = True
    prompt_require_abstention: bool = True
    prompt_require_citation: bool = False

    # Prompt versioning
    prompt_version: str = "v1"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000"

    @property
    def source_pdfs_path(self) -> Path:
        return PROJECT_ROOT / self.source_pdfs_dir

    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_dir

    @property
    def logs_path(self) -> Path:
        return PROJECT_ROOT / self.logs_dir

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# Sobrescritas em memoria aplicadas por cima do .env/ambiente. Existem para o
# harness de mutacao (tests/mutation/runner/sut.py), que precisa reconfigurar o
# pipeline entre mutantes sem reescrever o .env do usuario nem subir um processo
# novo por mutante. Nao sao usadas pela API em producao: quem nao chamar
# set_settings_overrides() ve exatamente o comportamento de antes.
_overrides: dict[str, Any] = {}


def set_settings_overrides(values: Mapping[str, Any] | None = None) -> None:
    """Substitui o conjunto de sobrescritas e invalida o cache de Settings."""
    global _overrides
    _overrides = dict(values or {})
    get_settings.cache_clear()


def get_settings_overrides() -> dict[str, Any]:
    return dict(_overrides)


@lru_cache
def get_settings() -> Settings:
    # Argumentos de init tem precedencia sobre env e .env no pydantic-settings.
    return Settings(**_overrides)
