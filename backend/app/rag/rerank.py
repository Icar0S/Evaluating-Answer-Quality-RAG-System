"""Reranker com o LLM local (RETRIEVAL_RERANK; E8 do plano-benchmark.md).

O plano propõe um cross-encoder (bge-reranker-v2-m3) e, como "alternativa sem
dependência nova", o LLM como reranker via Ollama, mais lento: é esta. Cada
candidato vira uma chamada que pede uma nota de 0 a 10 para "o trecho ajuda a
responder a pergunta?", com temperatura 0, sem raciocínio e 4 tokens de saída.
A nota fica em `rerank_score`; o cosseno continua em `similarity_score`.
Resposta sem número vale 0. Sempre o Ollama local, como o embedding.
"""
from __future__ import annotations

import re

import httpx

from app.config import get_settings

PROMPT_VERSION = "rerank-v1"
SYSTEM_PROMPT = (
    "Você avalia se um trecho de documento ajuda a responder uma pergunta. Responda só com um número inteiro de "
    "0 a 10: 0 = o trecho não tem nada a ver com a pergunta; 10 = o trecho contém a resposta."
)
_NUMBER = re.compile(r"\d+")
# Nota já pedida, por (modelo, pergunta, trecho): no mesmo processo, a mesma pergunta não paga duas vezes (o
# diagnóstico da C1 busca a lista do braço e a de 10, com os mesmos candidatos).
_SCORES: dict[tuple[str, str, str], float] = {}
_MAX_CACHED = 20000


def score(question: str, text: str, client: httpx.Client | None = None) -> float:
    settings = get_settings()
    model = settings.retrieval_rerank_model or settings.generation_model
    key = (model, question, text)
    if key not in _SCORES:
        if len(_SCORES) >= _MAX_CACHED:
            _SCORES.clear()
        _SCORES[key] = _ask(model, question, text, client)
    return _SCORES[key]


def _ask(model: str, question: str, text: str, client: httpx.Client | None) -> float:
    settings = get_settings()
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"PERGUNTA: {question}\n\nTRECHO:\n{text}\n\nNOTA:"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 4},
    }
    own = client is None
    client = client or httpx.Client(timeout=settings.generation_timeout_seconds)
    try:
        resp = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        match = _NUMBER.search(resp.json().get("message", {}).get("content", ""))
    finally:
        if own:
            client.close()
    return float(min(int(match.group()), 10)) if match else 0.0


def rerank(question: str, candidates: list[dict]) -> list[dict]:
    """Os candidatos com `rerank_score`, do maior para o menor (empate: o maior cosseno, depois a ordem)."""
    settings = get_settings()
    with httpx.Client(timeout=settings.generation_timeout_seconds) as client:
        scored = [{**c, "rerank_score": score(question, c["text"], client)} for c in candidates]
    order = sorted(range(len(scored)), key=lambda i: (-scored[i]["rerank_score"], -scored[i]["similarity_score"], i))
    return [scored[i] for i in order]
