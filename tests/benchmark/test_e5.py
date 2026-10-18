"""E5: concentração das fontes (tests/benchmark/e5.py), num caso pequeno feito à mão.

Quatro perguntas, cinco documentos no índice (A, B, C, D e M). M tem um chunk só e aparece em todas as
perguntas, de outros documentos: é o ímã. A pergunta de C nunca traz C nos 10 primeiros: C é morto.
"""
from __future__ import annotations

import pytest

from tests.benchmark import e5


def question(name: str, source: str, documents: list[str]) -> dict:
    return {"name": name, "source": source, "documents": documents,
            "chunk_ids": [f"{d}::c0" for d in documents], "documents_at_10": documents}


QUESTIONS = [
    question("q1", "A", ["A", "A", "M", "M"]),
    question("q2", "B", ["M", "B", "M", "C"]),
    question("q3", "C", ["M", "M", "M", "M"]),
    question("q4", "A", ["A", "M", "A", "A"]),
]
CHUNKS = {"A": 10, "B": 10, "C": 10, "D": 10, "M": 1}


def test_participacao_observada_esperada_e_massa_fora_da_fonte():
    observed = e5.observed_shares(QUESTIONS)

    assert observed == pytest.approx({"A": 0.3125, "M": 0.5625, "B": 0.0625, "C": 0.0625})
    assert e5.expected_shares(QUESTIONS) == {"A": 0.5, "B": 0.25, "C": 0.25}
    assert e5.off_source_mass(QUESTIONS) == pytest.approx(0.625)
    assert e5.total_variation(observed, e5.expected_shares(QUESTIONS)) == pytest.approx(0.5625)


def test_n80_enunciado_gini_ima_e_morto():
    result = e5.analyze(QUESTIONS, n_documents=5, chunk_counts=CHUNKS, k=4)

    assert result["n80"] == {"observed": 2, "expected": 3, "of": 5}
    # Top 20% de 5 documentos = 1 documento: M no observado, A no esperado.
    assert result["statement"]["observed"] == {"all": 0.25, "any": 1.0}
    assert result["statement"]["expected"] == {"all": 0.5, "any": 0.5}
    assert [m["document"] for m in result["magnets"]] == ["M"]
    assert result["magnets"][0]["undue"] == 4
    assert result["dead"] == ["C"]
    assert [m["chunk_id"] for m in result["chunk_magnets"]] == ["M::c0"]
    assert result["gini"]["observed"] > result["gini"]["expected"]
    assert result["lorenz"]["observed"][0] == (0.0, 0.0) and result["lorenz"]["observed"][-1] == (1.0, 1.0)


def test_gini_e_n80_nos_extremos():
    assert e5.gini([1, 0, 0, 0]) == pytest.approx(0.75)
    assert e5.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert e5.gini([0, 0]) is None
    assert e5.n80({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == 4


def test_perguntas_saem_do_diagnostico_so_as_de_dentro_do_indice():
    diagnostic = {"rows": [
        {"name": "q1", "source_document": "A.md", "c1": {"pdf-v1": {"pt": {
            "in_index": True, "chunks": [{"chunk_id": "A::p1::c0", "document": "A.pdf"}],
            "documents_at_10": ["A.pdf", "B.pdf"]}}}},
        {"name": "q2", "source_document": "Z.md", "c1": {"pdf-v1": {"pt": {"in_index": False, "chunks": []}}}},
    ]}

    [q] = e5.questions_from_diagnostic(diagnostic, "pdf-v1", "pt")

    assert (q["source"], q["documents"], q["documents_at_10"]) == ("A", ["A"], ["A", "B"])
