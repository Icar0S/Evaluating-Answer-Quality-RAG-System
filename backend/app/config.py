"""Configuração centralizada da aplicação, carregada de variáveis de ambiente/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # Chunking
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # Retrieval
    top_k: int = 4
    min_similarity_score: float = 0.0

    # Paths (relativos à raiz do projeto)
    source_pdfs_dir: str = "data/source_pdfs"
    vector_store_dir: str = "data/vector_store"
    vector_store_collection: str = "rag_test_specialist"
    logs_dir: str = "logs"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
