"""Braço do benchmark: a configuração efetiva de uma rodada, do jeito que vai das respostas ao relatório.

Hoje a identidade de uma rodada é só `rag_index` + idioma (plano-benchmark.md, §4, princípio 7). Um braço é
a versão do índice mais o que mudou na consulta e na geração (`--set CHAVE=VALOR`), e o que se grava é o
efetivo: o que o pipeline de fato usou depois do .env, da versão e das sobrescritas, com o manifesto do
índice que respondeu. Roda no venv do backend.
"""
from __future__ import annotations

import json


def parse_overrides(values: list[str]) -> dict:
    """--set CHAVE=VALOR (campos de app.config.Settings, em minúsculas); o pydantic converte o tipo."""
    overrides = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not key.strip() or not sep:
            raise SystemExit(f"--set espera CHAVE=VALOR, veio {value!r}")
        overrides[key.strip().lower()] = raw.strip()
    return overrides


def provenance(overrides: dict | None = None) -> dict:
    """A configuração efetiva do braço em uso (chamar com as sobrescritas já aplicadas)."""
    from app import providers
    from app.config import get_settings
    from app.rag import generation, ingestion

    settings = get_settings()
    path = ingestion.manifest_path()
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    registered = ingestion.registered_documents(settings.index_name)
    active = providers.get_active_provider()
    return {
        "index": settings.index_name,
        "collection": settings.vector_store_collection,
        "embedding_model": settings.embedding_model,
        "manifest": {
            "created_at": manifest.get("created_at"),
            "n_documents": manifest.get("n_documents"),
            "n_chunks": manifest.get("n_chunks"),
            "chunk_size_tokens": manifest.get("chunk_size_tokens"),
            "chunk_overlap_tokens": manifest.get("chunk_overlap_tokens"),
            "c0_version": (manifest.get("c0") or {}).get("version"),
        },
        "registered_documents": len(registered) if registered is not None else None,
        "retrieval": {
            "top_k": settings.top_k,
            "use_mmr": settings.retrieval_use_mmr,
            "mmr_lambda": settings.retrieval_mmr_lambda,
            "fetch_multiplier": settings.retrieval_fetch_multiplier,
            "min_similarity_score": settings.min_similarity_score,
        },
        "generation": {
            "provider": active.name,
            "model": active.generation_model,
            "temperature": settings.generation_temperature,
            "seed": settings.generation_seed,
            "num_ctx": generation.resolve_num_ctx(),
            "num_predict": settings.generation_num_predict,
            "think": settings.generation_think,
        },
        "prompt": {
            "version": settings.prompt_version,
            "require_grounding": settings.prompt_require_grounding,
            "require_abstention": settings.prompt_require_abstention,
            "require_citation": settings.prompt_require_citation,
        },
        "overrides": dict(overrides or {}),
    }
