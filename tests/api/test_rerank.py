"""Reranker com o LLM local (E8 do plano-benchmark.md): a nota e a reordenação no retrieve()."""
from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.rag import rerank, retrieval, vector_store

from conftest import LOCAL_BASE_URL


@pytest.fixture(autouse=True)
def _empty_cache(monkeypatch):
    monkeypatch.setattr(rerank, "_SCORES", {})


@pytest.mark.parametrize("reply, expected", [("7", 7.0), ("Nota: 12", 10.0), ("nada a ver", 0.0)])
def test_nota_do_llm_vira_numero_de_0_a_10(ollama, reply, expected):
    ollama.completion = reply

    assert rerank.score("O que é RAGAS?", "RAGAS is a framework.") == expected


def test_a_mesma_pergunta_e_trecho_nao_pagam_duas_vezes(ollama):
    ollama.completion = "5"
    rerank.score("Q", "T")
    rerank.score("Q", "T")

    chats = [r for r in ollama.requests_to(LOCAL_BASE_URL) if r.url.path == "/api/chat"]
    assert len(chats) == 1
    assert json.loads(chats[0].content)["options"] == {"temperature": 0, "num_predict": 4}


def test_retrieve_reordena_pela_nota_e_fica_com_os_k(monkeypatch):
    texts = {"a::c0": "texto a", "b::c0": "texto b", "c::c0": "texto c"}
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: {
        "ids": [list(texts)], "documents": [list(texts.values())],
        "metadatas": [[{"document": f"{c[0]}.pdf", "page": 1} for c in texts]], "distances": [[0.1, 0.2, 0.3]]})
    notes = {"texto a": 2.0, "texto b": 9.0, "texto c": 9.0}
    monkeypatch.setattr(rerank, "score", lambda question, text, client=None: notes[text])
    monkeypatch.setenv("RETRIEVAL_RERANK", "true")
    get_settings.cache_clear()

    chunks, _ = retrieval.retrieve("pergunta", top_k=2)

    # Empate na nota fica com o maior cosseno (b, 0,8, antes de c, 0,7); o cosseno continua no similarity_score.
    assert [(c["chunk_id"], c["rerank_score"], c["similarity_score"]) for c in chunks] == [
        ("b::c0", 9.0, 0.8), ("c::c0", 9.0, 0.7)]


def test_erro_do_ollama_tenta_de_novo_e_so_na_ultima_sobe(ollama, monkeypatch):
    import httpx

    monkeypatch.setattr(rerank.time, "sleep", lambda seconds: None)
    real, failures = ollama.handle, {"left": 2}

    def flaky(request):
        if request.url.path == "/api/chat" and failures["left"] > 0:
            failures["left"] -= 1
            return httpx.Response(500, json={"error": "falha passageira"})
        return real(request)

    monkeypatch.setattr(ollama, "handle", flaky)
    ollama.completion = "6"
    assert rerank.score("Q", "T") == 6.0

    failures["left"] = rerank.ATTEMPTS
    with pytest.raises(httpx.HTTPStatusError):
        rerank.score("outra pergunta", "T")
