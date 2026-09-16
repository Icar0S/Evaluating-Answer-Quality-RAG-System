"""O1 — similaridade de cosseno contra a resposta de referência, com limiar τ*.

Custo baixo, roda em cada execução. O limiar não é escolhido a dedo: sai de
calibration/calibrate.py, que varre τ ∈ [0,60; 0,95] sobre 300 pares rotulados
por construção e escolhe o F1 máximo com IC bootstrap de 95% (§6.3).

Duas decisões que não são detalhe de implementação:

1. O modelo de embedding do oráculo é FIXO em config/study.yaml e independente da
   configuração do SUT. Sem isso, o operador E1 (trocar o embedding) mudaria o
   instrumento junto com o objeto medido e seria inobservável por construção.

2. Embeddings são memorizados por (modelo, texto). As 30 respostas de referência
   reaparecem em cada um dos 18 mutantes × 5 repetições; sem cache seriam ~2.700
   chamadas de embedding para 30 textos distintos.
"""
from __future__ import annotations

import math

from tests.mutation.oracles.base import Verdict
from tests.mutation.runner import sut  # noqa: F401  (garante backend no sys.path)
from app.rag.ingestion import embed_texts

_cache: dict[tuple[str, str], list[float]] = {}


def embed(text: str, model: str) -> list[float]:
    key = (model, text)
    if key not in _cache:
        [vector] = embed_texts([text], model=model)
        _cache[key] = vector
    return _cache[key]


def prime(texts: list[str], model: str) -> int:
    """Embeda em lote tudo que ainda não está em cache. Devolve quantos foram.

    Existe por causa da RQ3, não por desempenho em si. O oráculo O1 é
    classificado como "custo baixo" no §6.4, e a RQ3 compara o custo dos cinco
    oráculos entre si. Uma implementação que embeda um texto por chamada HTTP
    gastaria ~2.700 idas ao Ollama na campanha e faria O1 parecer caro — mas o
    custo seria do laço, não do oráculo. Medir a versão ingênua responderia a
    pergunta errada.

    O endpoint /api/embed aceita lista; quem chama agrupa o que já tem em mãos
    (as N repetições de um caso, mais a referência) antes de pontuar.
    """
    faltando = [t for t in dict.fromkeys(texts) if (model, t) not in _cache and t.strip()]
    if not faltando:
        return 0
    for texto, vetor in zip(faltando, embed_texts(faltando, model=model)):
        _cache[(model, texto)] = vetor
    return len(faltando)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity(answer: str, reference: str, model: str) -> float:
    if not answer.strip() or not reference.strip():
        return 0.0
    return cosine(embed(answer, model), embed(reference, model))


def evaluate(answer: str, reference: str, tau: float, model: str) -> Verdict:
    if tau is None:
        raise ValueError(
            "τ* não calibrado. Rode: python -m tests.mutation.calibration.calibrate --write"
        )
    try:
        score = similarity(answer, reference, model)
    except Exception as exc:  # noqa: BLE001 — sem embedding não há juízo a emitir
        return Verdict(oracle="O1", verdict="pass", evaluable=False, detail=f"embedding falhou: {exc}")

    return Verdict(
        oracle="O1",
        verdict="pass" if score >= tau else "fail",
        score=round(score, 4),
        detail=f"cosseno {score:.4f} vs tau {tau:.2f}",
        components={"tau": tau, "model": model},
    )
