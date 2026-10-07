"""Fixtures da camada de avaliação DeepEval (Fase 2).

Diferente de tests/api/ (hermético, sem Ollama/GPU/rede — ver
tests/api/README.md), esta suíte exercita a stack real: Ollama de
geração+embeddings+juiz e o vector store populado. Roda só localmente, nunca
no CI — mesma regra de frontend/tests/e2e/chat-flow.spec.ts (ver
docs/ARCHITECTURE.md, "Testes determinísticos vs. testes que exercitam o
modelo"). Por isso o skip é o oposto do de tests/api: aqui a suíte inteira se
apaga quando a stack real NÃO está pronta, em vez de mockar tudo.
"""
from __future__ import annotations

import pytest

from _shared import build_judge_model, stack_status


@pytest.fixture(scope="session", autouse=True)
def _skip_without_real_stack() -> None:
    reason = stack_status()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="session")
def judge_model():
    return build_judge_model()
