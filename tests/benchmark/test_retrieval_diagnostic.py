"""tests/retrieval_diagnostic.py com a recuperação falsa: o formato que o relatório lê e o C1 de cada braço."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.config import get_settings
from tests.benchmark import c1

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    "FP1 Missing Content The first fail case is when asking a question that cannot be answered from the "
    "available documents."
)
GOLDENS = [
    {"name": "dentro", "input": "pergunta em português", "input_en": "question in english",
     "context": [EVIDENCE], "source_document": "seven.md", "conversion": "limpo"},
    {"name": "fora", "input": "receita de bolo", "input_en": "cake recipe", "source_document": None,
     "out_stratum": "escopo"},
    {"name": "sem-rotulo", "input": "golden antigo, sem source_document"},
]
POOL = {
    # PT: o documento certo vem em 2º; EN: em 1º, com a evidência.
    "pergunta em português": [("portugues.pdf", "texto em português", 0.62), ("seven.pdf", EVIDENCE, 0.58)],
    "question in english": [("seven.pdf", EVIDENCE, 0.71), ("other.pdf", "unrelated", 0.40)],
    "receita de bolo": [("portugues.pdf", "texto em português", 0.52)],
    "cake recipe": [("other.pdf", "unrelated", 0.30)],
}


@pytest.fixture
def diagnostic(monkeypatch):
    spec = importlib.util.spec_from_file_location("retrieval_diagnostic", ROOT / "tests" / "retrieval_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_retrieve(question: str, top_k: int | None = None):
        settings = get_settings()
        chunks = [
            {"chunk_id": f"{doc}::p1::c0", "document": doc, "page": 1, "text": text, "similarity_score": score}
            for doc, text, score in POOL[question]
            if score >= settings.min_similarity_score
        ]
        return chunks[: top_k or settings.top_k], 1.0

    monkeypatch.setattr(module.retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(module.ingestion, "registered_documents", lambda index: {"seven.pdf": "0" * 64})
    return module


def test_ranks_continuam_no_formato_do_relatorio_e_o_c1_vem_ao_lado(diagnostic):
    result = diagnostic.run(["pdf-v1"], [g for g in GOLDENS if "source_document" in g])

    [row] = result["rows"]
    assert row["ranks"] == {"pdf-v1": {"pt": 2, "en": 1}}
    assert row["c1"]["pdf-v1"]["en"]["evidence_at_k"] is True
    assert row["c1"]["pdf-v1"]["pt"]["in_index"] is True
    [out] = result["out_rows"]
    assert out["c1"]["pdf-v1"]["pt"]["top1_similarity"] == 0.52
    summary = result["c1"]["summary"]["pdf-v1"]
    assert summary["pt"]["hit_at_1"]["mean"] == 0.0 and summary["en"]["hit_at_1"]["mean"] == 1.0
    assert summary["en"]["top1_similarity_auc"]["auc"] == 1.0
    config = result["c1"]["arms"]["pdf-v1"]
    assert (config["index"], config["retrieval"]["top_k"], config["embedding_model"]) == ("pdf-v1", 4, "nomic-embed-text")
    assert result["c1"]["version"] == c1.C1_VERSION


def test_limiar_do_braco_bloqueia_e_vira_abstencao(diagnostic):
    goldens = [g for g in GOLDENS if "source_document" in g]

    result = diagnostic.run(["pdf-v1"], goldens, overrides={"min_similarity_score": "0.55"})

    summary = result["c1"]["summary"]["pdf-v1"]["pt"]
    assert result["c1"]["arms"]["pdf-v1"]["retrieval"]["min_similarity_score"] == 0.55
    assert result["c1"]["arms"]["pdf-v1"]["overrides"] == {"min_similarity_score": "0.55"}
    # A pergunta de fora (0,52) fica sem trecho; a de dentro (0,62) passa.
    assert (summary["threshold_abstention_rate"], summary["false_block_rate"]) == (1.0, 0.0)
    assert result["out_rows"][0]["c1"]["pdf-v1"]["pt"]["n_chunks"] == 0
    # As sobrescritas valem só durante o diagnóstico.
    assert get_settings().min_similarity_score == 0.0


def test_documento_que_a_versao_nao_tem_conta_como_de_fora(diagnostic, monkeypatch):
    monkeypatch.setattr(diagnostic.ingestion, "registered_documents", lambda index: {"outro.pdf": "0" * 64})
    goldens = [g for g in GOLDENS if "source_document" in g]

    summary = diagnostic.run(["pdf-v1"], goldens)["c1"]["summary"]["pdf-v1"]["pt"]

    assert (summary["n_inside"], summary["n_outside"]) == (0, 2)


def test_goldens_sem_source_document_nao_entram(diagnostic, tmp_path: Path):
    path = tmp_path / "goldens.json"
    path.write_text(__import__("json").dumps(GOLDENS), encoding="utf-8")

    assert [g["name"] for g in diagnostic.load_goldens([str(path)])] == ["dentro", "fora"]


def test_pergunta_ambigua_fica_fora_da_c1(diagnostic):
    ambiguous = {"name": "amb", "input": "receita de bolo", "source_document": None, "question_class": "ambiguo"}

    result = diagnostic.run(["pdf-v1"], [GOLDENS[0], ambiguous])

    assert result["out_rows"] == []
