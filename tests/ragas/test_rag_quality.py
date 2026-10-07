"""Qualidade de resposta do RAG com RAGAS: goldens x 5 métricas.

Roda o pipeline real (as mesmas funções que POST /chat usa) sobre os goldens
de tests/deepeval/goldens/dataset.json — os mesmos da suíte DeepEval, para as
duas poderem ser comparadas — e julga cada resposta com o juiz local (ver
_ragas_shared.py). Um teste por par golden x métrica: a saída do pytest já é a
matriz de quem passa onde.

Métrica não medida (erro do juiz, NaN) REPROVA com o motivo, em vez de passar
ou ser pulada: o estudo de mutação mostrou o custo de um oráculo que "passa"
porque não conseguiu medir.

Rodar da raiz do repo (venv próprio, ver README.md desta pasta):
    tests\\ragas\\.venv\\Scripts\\python.exe -m pytest tests/ragas
"""
from __future__ import annotations

import pytest

from _ragas_shared import METRIC_NAMES, CaseInputs, load_goldens, score_metric

CASES = [(golden, metric) for golden in load_goldens() for metric in METRIC_NAMES]


@pytest.mark.parametrize(("golden", "metric_name"), CASES, ids=[f"{g.name}::{m}" for g, m in CASES])
def test_rag_quality(golden, metric_name, metrics, sut_output, ragas_settings) -> None:
    answer, contexts = sut_output(golden)
    case = CaseInputs(
        user_input=golden.input,
        response=answer,
        retrieved_contexts=contexts,
        reference=golden.expected_output,
    )

    outcome = score_metric(metric_name, metrics[metric_name], case)

    if not outcome.measured:
        pytest.fail(f"{metric_name} não foi medida: {outcome.error}\n\nresposta do SUT: {answer[:400]}")
    assert outcome.score >= ragas_settings.ragas_threshold, (
        f"{metric_name}={outcome.score:.3f} < {ragas_settings.ragas_threshold}\n\nresposta do SUT: {answer[:400]}"
    )
