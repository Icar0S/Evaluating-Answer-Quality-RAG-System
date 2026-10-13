"""Conversões de tests/demo_e2e.py: o que o navegador gravou vira respostas, diagnóstico e relatório.

Herméticos: nenhum navegador, container ou juiz roda aqui. Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

import sys
from pathlib import Path

import build_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, para importar demo_e2e
import demo_e2e  # noqa: E402

from test_index_comparison import _ragas_run  # noqa: E402

GOLDENS = [
    ("pdf", {"name": "p1", "input": "Pergunta dos PDFs?", "expected_output": "Ref P.", "context": ["ctx"]}),
    ("md", {"name": "m1", "input": "Pergunta dos Markdown?", "expected_output": "Ref M1.",
            "source_document": "a.md", "conversion": "grave"}),
    ("md", {"name": "m2", "input": "Outra?", "expected_output": "Ref M2.",
            "source_document": "b.md", "conversion": "limpo"}),
]


def _record(name: str, answer: str, documents: list[str], index: str = "md-v1") -> dict:
    """Uma linha do respostas_<índice>.jsonl, como e2e/goldens.spec.ts grava."""
    return {
        "rag_index": index,
        "name": name,
        "origin": "?",
        "input": "?",
        "actual_output": answer,
        "retrieval_context": [f"trecho de {d}" for d in documents],
        "sources": documents,
        "source_documents": documents,
        "error": None,
        "seconds": 2.5,
        "metadata": {"model": "gemma3:4b", "embedding_model": "nomic-embed-text", "provider": "local",
                     "top_k": 4, "prompt_version": "v1", "rag_index": index},
        "finished_at": "2026-10-13T18:00:00+00:00",
    }


def test_respostas_seguem_a_ordem_dos_goldens_e_a_que_faltou_entra_com_erro():
    records = [_record("m1", "primeira", ["x.md"]), _record("p1", "do PDF", []), _record("m1", "repetida", ["a.md"])]

    data = demo_e2e.build_answers("md-v1", records, GOLDENS)

    assert [a["name"] for a in data["answers"]] == ["p1", "m1", "m2"]
    answers = {a["name"]: a for a in data["answers"]}
    assert answers["m1"]["actual_output"] == "repetida"  # teste repetido: vale a última resposta
    assert answers["m1"]["expected_output"] == "Ref M1."  # pergunta e referência vêm do golden
    assert answers["p1"]["origin"] == "pdf"
    # Sem resposta: entra com erro, para o RAGAS contar como não medida e os
    # dois RAGs ficarem com os mesmos casos.
    assert answers["m2"]["actual_output"] is None
    assert answers["m2"]["error"] == demo_e2e.MISSING
    assert (data["rag_index"], data["generation_model"], data["lang"], data["top_k"]) == ("md-v1", "gemma3:4b", "pt", 4)


def test_diagnostico_sai_dos_trechos_que_o_chat_devolveu():
    answer_sets = {
        "pdf-v1": demo_e2e.build_answers("pdf-v1", [_record("m1", "r", ["x.pdf", "y.pdf"], "pdf-v1")], GOLDENS),
        "md-v1": demo_e2e.build_answers("md-v1", [_record("m1", "r", ["z.md", "a.md"]), _record("m2", "r", ["b.md"])],
                                        GOLDENS),
    }

    diagnostic = demo_e2e.retrieval_diagnostic(answer_sets, GOLDENS)

    assert diagnostic["indices"] == ["pdf-v1", "md-v1"]
    assert diagnostic["langs"] == ["pt"]
    rows = {r["name"]: r for r in diagnostic["rows"]}
    assert set(rows) == {"m1", "m2"}  # só os goldens com documento-fonte conhecido
    assert rows["m1"]["ranks"] == {"pdf-v1": {"pt": None}, "md-v1": {"pt": 2}}
    assert rows["m2"]["ranks"]["md-v1"] == {"pt": 1}
    assert rows["m2"]["conversion"] == "limpo"


def test_conta_os_testes_de_cada_projeto_do_playwright():
    results = {
        "suites": [
            {
                "title": "goldens.spec.ts",
                "specs": [],
                "suites": [
                    {
                        "title": "Perguntas de referência",
                        "specs": [
                            {"tests": [
                                {"projectName": "RAG pdf-v1 (PDFs)", "status": "expected", "results": [{"duration": 1500}]},
                                {"projectName": "RAG md-v1 (Markdown)", "status": "unexpected",
                                 "results": [{"duration": 500}]},
                            ]},
                            {"tests": [{"projectName": "RAG md-v1 (Markdown)", "status": "skipped", "results": []}]},
                        ],
                    }
                ],
            }
        ]
    }

    counts = demo_e2e.playwright_counts(results)

    assert counts["RAG pdf-v1 (PDFs)"] == {"total": 1, "failed": 0, "errors": 0, "skipped": 0, "ms": 1500}
    assert counts["RAG md-v1 (Markdown)"] == {"total": 2, "failed": 1, "errors": 0, "skipped": 1, "ms": 500}


def test_relatorio_da_demonstracao_diz_de_onde_vieram_respostas_e_diagnostico(tmp_path: Path):
    pdf = _ragas_run(tmp_path / "pdf.json", "pdf-v1", "pt", [("m1", "md", "Resposta do PDF.", 0.4)])
    md = _ragas_run(tmp_path / "md.json", "md-v1", "pt", [("m1", "md", "Resposta do Markdown.", 0.9)])
    answer_sets = {
        "pdf-v1": demo_e2e.build_answers("pdf-v1", [_record("m1", "r", ["x.pdf"], "pdf-v1")], GOLDENS[1:2]),
        "md-v1": demo_e2e.build_answers("md-v1", [_record("m1", "r", ["a.md"])], GOLDENS[1:2]),
    }
    execution = {
        "tool": "scripts/demo_e2e.bat",
        "started_at": "2026-10-13T15:00:00-03:00",
        "duration_seconds": 60,
        "note": "As respostas avaliadas são as que o chat deu nos testes E2E ao vivo.",
        "retrieval_diagnostic": demo_e2e.retrieval_diagnostic(answer_sets, GOLDENS[1:2]),
        "steps": [],
    }

    outputs = build_report.build([pdf, md], ["md"], tmp_path / "out", execution=execution)

    text = outputs["md"].read_text(encoding="utf-8")
    assert "mesmos goldens (`tests/deepeval/goldens/dataset_md.json`)" in text
    assert "Execução de `scripts/demo_e2e.bat`" in text
    assert "o chat deu nos testes E2E ao vivo" in text
    assert "Tirado dos trechos que o chat devolveu" in text
    assert "**md-v1**, perguntas em PT: 1/1" in text
    assert "**pdf-v1**, perguntas em PT: 0/1" in text
