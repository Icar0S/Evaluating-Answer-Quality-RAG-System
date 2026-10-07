"""Fixtures da suíte RAGAS.

Mesma regra de tests/deepeval: exercita a stack real (Ollama com geração,
embeddings e juiz, e o vector store populado), roda só localmente e nunca no
CI. Sem a stack pronta, a suíte inteira se auto-pula em vez de falhar.
"""
from __future__ import annotations

from typing import Callable

import pytest

from _ragas_shared import (
    Golden,
    OllamaJudge,
    RagasSettings,
    build_embeddings,
    build_judge,
    build_metrics,
    get_ragas_settings,
    run_pipeline,
    stack_status,
)


@pytest.fixture(scope="session", autouse=True)
def _skip_without_real_stack() -> None:
    reason = stack_status()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def ragas_settings() -> RagasSettings:
    return get_ragas_settings()


@pytest.fixture(scope="session")
def judge() -> OllamaJudge:
    return build_judge()


@pytest.fixture(scope="session")
def metrics(judge: OllamaJudge) -> dict:
    return build_metrics(judge, build_embeddings())


@pytest.fixture(scope="session")
def sut_output() -> Callable[[Golden], tuple[str, list[str]]]:
    """Resposta e trechos do SUT por golden, gerados UMA vez por sessão.

    Cada golden vira 5 testes (um por métrica); sem o cache o pipeline rodaria
    5 vezes por pergunta e, como a geração não é determinística, cada métrica
    poderia estar julgando uma resposta diferente. A falha também fica no
    cache: um SUT em timeout não deve custar 5 timeouts por golden.
    """
    cache: dict[str, tuple[str, list[str]] | Exception] = {}

    def get(golden: Golden) -> tuple[str, list[str]]:
        if golden.name not in cache:
            try:
                cache[golden.name] = run_pipeline(golden.input)
            except Exception as exc:  # noqa: BLE001 - relançada abaixo, em todo teste do golden
                cache[golden.name] = exc
        cached = cache[golden.name]
        if isinstance(cached, Exception):
            pytest.fail(f"o SUT falhou ao responder {golden.name!r}: {type(cached).__name__}: {cached}")
        return cached

    return get
