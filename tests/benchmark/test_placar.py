"""Placar: Δ pareado com bootstrap por documento, McNemar exato, Holm e a regra de veredito provisória."""
from __future__ import annotations

import pytest

from tests.benchmark import placar


def test_delta_pareado_so_nas_perguntas_dos_dois_lados_e_por_documento():
    reference = {"q1": 0.5, "q2": 0.5, "q3": 1.0, "q4": 0.0}
    arm = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q5": 1.0}  # q4 sem medida no braço, q5 sem medida na referência
    clusters = {"q1": "a", "q2": "a", "q3": "b"}

    d = placar.paired_delta(reference, arm, clusters)

    assert (d["n"], d["clusters"], d["one_side_only"]) == (3, 2, 2)
    assert d["delta"] == pytest.approx(1 / 3, abs=1e-4)
    # Reamostrando documentos: o "a" dá Δ 0,5 e o "b" dá 0; o IC vai de 0 a 0,5.
    assert (d["ci_low"], d["ci_high"]) == (0.0, 0.5)
    assert d["abs_p95"] == 0.5
    assert placar.paired_delta({"q1": 1.0}, {"q2": 1.0}, {}) is None


def test_mcnemar_exato_e_holm():
    reference = {f"q{i}": True for i in range(10)}
    arm = {f"q{i}": i < 2 for i in range(10)}  # o braço erra 8 que a referência acerta

    result = placar.mcnemar_exact(reference, arm)

    assert (result["b"], result["c"]) == (8, 0)
    assert result["p_value"] == pytest.approx(2 / 2**8)
    assert placar.mcnemar_exact(reference, reference)["p_value"] == 1.0
    assert placar.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_veredito_provisorio():
    big_up = {"delta": 0.2, "ci_low": 0.05, "ci_high": 0.3}
    big_unsure = {"delta": 0.2, "ci_low": -0.05, "ci_high": 0.4}
    small = {"delta": 0.02, "ci_low": 0.01, "ci_high": 0.03}

    assert placar.verdict(big_up, margin=0.1) == "melhor"
    assert placar.verdict(big_up, margin=0.1, higher_is_better=False) == "pior"
    assert placar.verdict(big_unsure, margin=0.1) == "inconclusivo"
    # Diferença estatística, mas dentro do ruído das rodadas idênticas.
    assert placar.verdict(small, margin=0.05) == "sem diferença relevante"
    assert placar.verdict(None, margin=0.1) == "não medido"
    assert placar.verdict(small, margin=None) == "sem margem do E0"


def test_margem_e_o_maior_p95_entre_os_pares_aa():
    pairs = [{"faithfulness": {"abs_p95": 0.04}, "hit_at_k": None}, {"faithfulness": {"abs_p95": 0.07}}]

    assert placar.noise_margins(pairs) == {"faithfulness": 0.07}
