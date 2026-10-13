"""Relatório comparando versões do índice (pdf-v1 x md-v1), com resultados sintéticos.

Herméticos. Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_report
from report_data import NOT_FOUND_MARKER, compare, load_run, origin_stats

DIMS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance")


def _ragas_run(path: Path, rag_index: str, lang: str, cases: list[tuple[str, str, str, float]], judge="qwen3:8b") -> Path:
    """cases: (nome, origem, resposta, nota usada em todas as dimensões)."""
    data = {
        "framework": "ragas",
        "generated_at": "2026-10-13T18:00:00+00:00",
        "generation_model": "gemma3:4b",
        "judge_model": judge,
        "threshold": 0.7,
        "rag_index": rag_index,
        "lang": lang,
        "n_cases": len(cases),
        "cases": [
            {
                "name": name,
                "origin": origin,
                "input": f"Pergunta {name}?",
                "actual_output": answer,
                "expected_output": "Referência.",
                "retrieval_context": [f"trecho de {rag_index}"],
                "success": score >= 0.7,
                "metrics": [{"name": d, "score": score, "success": score >= 0.7, "reason": None, "error": None} for d in DIMS],
            }
            for name, origin, answer, score in cases
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def pdf_run(tmp_path: Path) -> Path:
    return _ragas_run(
        tmp_path / "pdf.json", "pdf-v1", "pt",
        [("p1", "pdf", "Resposta do PDF.", 0.9), ("m1", "md", f"{NOT_FOUND_MARKER}.", 0.0)],
    )


@pytest.fixture
def md_run(tmp_path: Path) -> Path:
    return _ragas_run(
        tmp_path / "md.json", "md-v1", "pt",
        [("p1", "pdf", f"{NOT_FOUND_MARKER}.", 0.0), ("m1", "md", "Resposta do Markdown.", 0.6)],
    )


def test_rodada_se_identifica_pelo_indice_e_pelo_idioma(tmp_path: Path, md_run: Path):
    english = _ragas_run(tmp_path / "en.json", "md-v1", "en", [("m1", "md", "Answer.", 0.9)])

    assert load_run(md_run).name == "RAGAS · md-v1"
    assert load_run(english).name == "RAGAS · md-v1 · perguntas em inglês"


def test_entre_indices_respostas_diferentes_nao_sao_aviso_mas_juiz_diferente_e(tmp_path: Path, pdf_run: Path):
    other_judge = _ragas_run(
        tmp_path / "md2.json", "md-v1", "pt", [("p1", "pdf", "outra", 0.5), ("m1", "md", "outra", 0.5)],
        judge="gemma3:4b",
    )

    comp = compare(load_run(pdf_run), load_run(other_judge))

    assert comp.across_variants
    warnings = " ".join(comp.warnings)
    assert "Juízes diferentes" in warnings
    assert "respostas avaliadas são idênticas" not in warnings
    assert "trechos recuperados" not in warnings


def test_recorte_por_origem_separa_acervo_proprio_de_alheio(md_run: Path):
    stats = {o.origin: o for o in origin_stats(load_run(md_run))}

    assert stats["md"].means["faithfulness"] == pytest.approx(0.6)
    assert stats["md"].abstained == 0
    assert stats["pdf"].abstained == 1  # pergunta do acervo alheio: abstenção
    assert stats["pdf"].n_cases == 1


def test_relatorio_traz_origem_e_diagnostico_de_recuperacao(tmp_path: Path, pdf_run: Path, md_run: Path):
    execution = {
        "tool": "tests/compare_indices.py",
        "started_at": "2026-10-13T15:00:00-03:00",
        "duration_seconds": 60,
        "steps": [],
        "retrieval_diagnostic": {
            "top_k": 4,
            "indices": ["pdf-v1", "md-v1"],
            "langs": ["pt", "en"],
            "rows": [
                {"name": "m1", "conversion": "grave", "source_document": "a.md",
                 "ranks": {"pdf-v1": {"pt": None, "en": None}, "md-v1": {"pt": None, "en": 1}}},
            ],
        },
    }

    outputs = build_report.build([pdf_run, md_run], ["md", "xlsx"], tmp_path / "out", execution=execution)

    md = outputs["md"].read_text(encoding="utf-8")
    assert "Execução de `tests/compare_indices.py`" in md
    assert "Comparação entre as rodadas" in md
    assert "Por origem da pergunta" in md
    assert "Recuperação do documento certo" in md
    assert "**md-v1**, perguntas em EN: 1/1" in md
    assert "RAGAS · pdf-v1: média" in md


def test_historico_desenha_a_evolucao_de_cada_rodada(tmp_path: Path):
    # Duas rodadas da mesma variante em datas diferentes: é o caso em que o
    # gráfico de evolução de fato desenha linhas (um ponto só não vira linha).
    root = tmp_path / "out"
    first = _ragas_run(tmp_path / "a.json", "md-v1", "pt", [("m1", "md", "R.", 0.5)])
    second_data = json.loads(first.read_text(encoding="utf-8"))
    second_data["generated_at"] = "2026-10-20T18:00:00+00:00"
    second = tmp_path / "b.json"
    second.write_text(json.dumps(second_data), encoding="utf-8")

    build_report.build([first], ["md"], root / "20261013T180000", history_root=root)
    outputs = build_report.build([second], ["md"], root / "20261020T180000", history_root=root)

    assert (root / "20261020T180000" / "figuras" / "historico.png").is_file()
    assert "Evolução da média" in outputs["md"].read_text(encoding="utf-8")
