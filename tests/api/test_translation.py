"""Tradução da pergunta só na busca (app/rag/translation.py, E1 do plano-benchmark.md)."""
from __future__ import annotations

import hashlib
import json

import pytest

from app.config import get_settings
from app.rag import retrieval, translation, vector_store

from conftest import LOCAL_BASE_URL, LOCAL_MODEL


def _chat_requests(ollama) -> list[dict]:
    return [json.loads(r.content) for r in ollama.requests_to(LOCAL_BASE_URL) if r.url.path == "/api/chat"]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "traducoes.json"
    path.write_text(json.dumps({"model": "gemma3:4b", "translations": {"O que é RAG?": "What is RAG?"},
                                "milliseconds": {"O que é RAG?": 812.5}}), encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_QUERY_TRANSLATION", "true")
    monkeypatch.setenv("RETRIEVAL_TRANSLATION_CACHE", str(path))
    get_settings.cache_clear()
    return path


def test_desligada_a_busca_usa_a_pergunta_original(ollama):
    assert translation.query_for_retrieval("O que é RAG?") == "O que é RAG?"
    assert translation.provenance() is None
    assert _chat_requests(ollama) == []


def test_com_o_cache_usa_a_traducao_gravada_sem_chamar_o_llm(cache, ollama):
    assert translation.query_for_retrieval("O que é RAG?") == "What is RAG?"
    assert _chat_requests(ollama) == []

    prov = translation.provenance()
    assert prov["cache_model"] == "gemma3:4b"
    assert prov["cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()


def test_fora_do_cache_traduz_no_ollama_local_sem_raciocinio(cache, ollama):
    ollama.completion = '"What does the retriever return?"'

    assert translation.query_for_retrieval("O que o recuperador devolve?") == "What does the retriever return?"
    [payload] = _chat_requests(ollama)
    assert (payload["model"], payload["think"], payload["options"]["temperature"]) == (LOCAL_MODEL, False, 0)


def test_a_busca_embeda_a_traducao(cache, ollama, monkeypatch):
    empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: empty)

    chunks, _ = retrieval.retrieve("O que é RAG?")

    assert chunks == []
    embeds = [json.loads(r.content) for r in ollama.requests_to(LOCAL_BASE_URL) if r.url.path == "/api/embed"]
    assert embeds[-1]["input"] == ["What is RAG?"]
