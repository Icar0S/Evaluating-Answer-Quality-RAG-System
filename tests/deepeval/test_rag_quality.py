"""Avaliação de qualidade de resposta do RAG com DeepEval — Fase 2.

Roda o pipeline real (retrieval + generation — as mesmas funções que
POST /chat usa, backend/app/main.py) contra o golden dataset público em
goldens/dataset.json, e audita cada resposta com 5 métricas RAG julgadas por
um LLM local via Ollama (ver _shared.py). Isso é o LLM-as-judge que audita a
heurística `grounded` (regex sobre NOT_FOUND_MARKER) do backend com um
julgamento de fato — a lacuna que docs/ARCHITECTURE.md já registrava como
"a cargo da Fase 2".

Requer Ollama no ar com GENERATION_MODEL, EMBEDDING_MODEL e o modelo juiz
baixados, e o vector store já populado (`python scripts/ingest_documents.py`).
Sem isso, a suíte inteira se auto-pula — ver conftest.py.

Rodar (venv isolado, ver README.md deste diretório):
    tests\\deepeval\\.venv\\Scripts\\python.exe -m pytest tests/deepeval

Use pytest puro, não `deepeval test run`: a CLI liga o cache local do
deepeval (env var DEEPEVAL=1), que no Windows sem o extra `pywin32` quebra
com AttributeError depois que a avaliação já rodou de verdade (ver
README.md deste diretório, "Bug conhecido: deepeval test run no Windows").
"""
from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from _dataset import load_goldens
from _shared import THRESHOLD, run_pipeline


@pytest.mark.parametrize("golden", load_goldens(), ids=lambda g: g.name or g.input[:40])
def test_rag_quality(golden: Golden, judge_model) -> None:
    answer, retrieval_context = run_pipeline(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        context=golden.context,
        retrieval_context=retrieval_context,
        name=golden.name,
    )

    assert_test(
        test_case,
        [
            FaithfulnessMetric(model=judge_model, threshold=THRESHOLD),
            AnswerRelevancyMetric(model=judge_model, threshold=THRESHOLD),
            ContextualPrecisionMetric(model=judge_model, threshold=THRESHOLD),
            ContextualRecallMetric(model=judge_model, threshold=THRESHOLD),
            ContextualRelevancyMetric(model=judge_model, threshold=THRESHOLD),
        ],
    )
