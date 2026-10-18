"""Recalibração do limiar (tests/benchmark/limiar.py): a regra do estudo de mutação, fora da amostra."""
from __future__ import annotations

import pytest

from tests.benchmark import limiar


def test_separacao_perfeita_fica_no_meio_da_folga():
    chosen = limiar.choose_threshold([0.70, 0.80], [0.40, 0.50])

    assert chosen == {"threshold": 0.6, "rule": "ponto médio da folga", "separable": True}


def test_sem_separacao_o_limiar_fica_com_margem_abaixo_do_menor_positivo():
    positives, negatives = [0.63, 0.70, 0.75], [0.50, 0.59, 0.66]

    chosen = limiar.choose_threshold(positives, negatives)

    assert chosen["separable"] is False
    assert chosen["threshold"] <= min(positives) - limiar.MARGIN
    assert limiar.evaluate(chosen["threshold"], positives, negatives)["false_block"] == 0.0


def test_avaliacao_e_validacao_cruzada_por_documento():
    result = limiar.evaluate(0.6, positives=[0.55, 0.7], negatives=[0.5, 0.65])
    assert (result["correct_abstention"], result["false_block"]) == (0.5, 0.5)

    inside = [("a", 0.70), ("a", 0.72), ("b", 0.61), ("c", 0.80)]
    outside = [0.40, 0.45, 0.62]
    cv = limiar.leave_one_document_out(inside, outside)
    # Sem o documento b, o menor positivo é 0,70 e o limiar sobe acima de 0,61: a pergunta de b é barrada.
    assert cv["false_block_count"] == 1
    with pytest.raises(ValueError):
        limiar.choose_threshold([], [0.1])
