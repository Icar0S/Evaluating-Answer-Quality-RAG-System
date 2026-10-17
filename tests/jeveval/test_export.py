"""Testes herméticos do run_and_export.py e do contrato com o relatório consolidado.

O script roda inteiro, com o Jev falso injetado e as saídas numa pasta
temporária: o que se testa é o formato que o tests/reports lê e o tratamento de
recusa (N/A) e de geração que falhou (não medida). Sem rede e sem chave.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import _jev_shared as shared
import run_and_export
from _fake_jev import FakeJev, best, not_applicable_else, worst
from _jev_shared import METRIC_NAMES, read_not_found_marker

MARKER = read_not_found_marker()


def _answers_file(tmp_path: Path) -> Path:
    base = {"origin": "pdf", "expected_output": "RAGAS é um framework de avaliação.", "context": ["trecho"]}
    data = {
        "generated_at": "2026-10-16T12:00:00+00:00",
        "provider": "ollama-local",
        "generation_model": "gemma3:4b",
        "rag_index": "pdf-v1",
        "lang": "pt",
        "answers": [
            {**base, "name": "c1", "input": "O que é o RAGAS?", "actual_output": "Um framework de avaliação.",
             "retrieval_context": ["t1", "t2"], "error": None, "seconds": 3.0},
            {**base, "name": "c2-recusa", "input": "E o ARES?", "actual_output": f"{MARKER}.",
             "retrieval_context": ["t9"], "error": None, "seconds": 2.0},
            {**base, "name": "c3-falhou", "input": "E o MetaRAG?", "actual_output": None,
             "retrieval_context": [], "error": "ReadTimeout: sem resposta", "seconds": 60.0},
        ],
    }
    path = tmp_path / "respostas.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _rule(state: dict, key: str, question) -> object:
    """O que um bom instrumento faria: recusa vira N/A (e reprova no resto); resposta normal, nota máxima."""
    if MARKER in (state["test_case"].get("actual_output") or ""):
        return not_applicable_else(worst)(state, key, question)
    return best(state, key, question)


def _run(tmp_path: Path, *extra: str, jev: FakeJev | None = None) -> dict:
    out = tmp_path / "results"
    code = run_and_export.main(
        ["--answers", str(_answers_file(tmp_path)), "--out-dir", str(out), *extra], model=jev or FakeJev(_rule)
    )
    assert code == 0
    (json_path,) = out.glob("*.json")
    return json.loads(json_path.read_text(encoding="utf-8"))


def _metric(case: dict, name: str) -> dict:
    return next(m for m in case["metrics"] if m["name"] == name)


def test_export_no_formato_compartilhado(tmp_path: Path) -> None:
    data = _run(tmp_path)

    assert data["framework"] == "jeveval"
    assert data["judge_model"] == "fake-jev (teste)"
    assert data["generation_model"] == "gemma3:4b"
    assert data["instrument_sha"] == shared.instrument_sha()
    assert set(data["question_sets"]) == set(METRIC_NAMES)
    assert data["threshold"] == 0.7
    assert data["thresholds"]["correctness"] == 0.5
    assert data["n_cases"] == data["cases_done"] == 3
    for case in data["cases"]:
        assert {"name", "input", "actual_output", "expected_output", "retrieval_context", "metrics"} <= set(case)
        assert [m["name"] for m in case["metrics"]] == list(METRIC_NAMES)

    c1 = data["cases"][0]
    assert all(m["score"] == 1.0 and m["success"] and m["applicable"] for m in c1["metrics"])
    assert c1["success"] is True
    assert _metric(c1, "correctness")["cause"] == "none"


def test_recusa_sai_na_e_nao_entra_nas_contas(tmp_path: Path) -> None:
    data = _run(tmp_path)
    c2 = data["cases"][1]

    for name in ("faithfulness", "answer_relevancy"):
        m = _metric(c2, name)
        assert (m["score"], m["success"], m["applicable"], m["error"]) == (None, None, False, None)
    # A recuperação continua medida (o estado dessas métricas nem traz a resposta,
    # então a recusa não as alcança), e a correção diz que a abstenção foi errada.
    for name in ("context_precision", "context_recall", "context_relevance"):
        assert _metric(c2, name)["applicable"] and _metric(c2, name)["score"] == 1.0
    assert _metric(c2, "correctness")["score"] == 0.0
    assert _metric(c2, "correctness")["cause"] == "missing_info"

    assert data["metrics_not_applicable"]["faithfulness"] == 1
    assert data["metrics_measured"]["faithfulness"] == "1/2"  # c1 medida; c3 não; c2 fora (N/A)
    assert data["metrics_avg"]["faithfulness"] == 1.0  # N/A não puxa a média para baixo


def test_geracao_que_falhou_vira_nota_nao_medida_sem_chamar_o_jev(tmp_path: Path) -> None:
    jev = FakeJev(_rule)
    data = _run(tmp_path, jev=jev)
    c3 = data["cases"][2]

    for m in c3["metrics"]:
        assert m["score"] is None and m["applicable"] is True
        assert m["error"] == "geração falhou: ReadTimeout: sem resposta"
    assert c3["success"] is False
    assert data["jev_calls"] == len(jev.calls) == 2 * len(METRIC_NAMES)


def test_csv_uma_linha_por_caso_e_metrica(tmp_path: Path) -> None:
    _run(tmp_path)
    (csv_path,) = (tmp_path / "results").glob("*.csv")

    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert rows[0][:5] == ["golden", "metrica", "score", "passou", "aplicavel"]
    assert len(rows) == 1 + 3 * len(METRIC_NAMES)


def test_subconjunto_de_metricas(tmp_path: Path) -> None:
    data = _run(tmp_path, "--metrics", "faithfulness,correctness")

    assert [m["name"] for m in data["cases"][0]["metrics"]] == ["faithfulness", "correctness"]
    assert set(data["question_sets"]) == {"faithfulness", "correctness"}
    assert data["instrument_sha"] == shared.instrument_sha(["faithfulness", "correctness"])


def test_metrica_desconhecida_aborta(tmp_path: Path) -> None:
    argv, jev = ["--answers", str(_answers_file(tmp_path)), "--metrics", "bleu"], FakeJev()

    with pytest.raises(SystemExit):
        run_and_export.main(argv, model=jev)


def test_sem_jev_o_script_sai_com_erro_sem_chamar_nada(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "jev_status", lambda settings=None: "TYPESAFE_API_KEY não definida (teste)")
    monkeypatch.setattr(shared, "build_jev", lambda settings=None: pytest.fail("não devia construir o Jev"))

    assert run_and_export.main(["--answers", str(_answers_file(tmp_path)), "--out-dir", str(tmp_path)]) == 1


def test_arquivo_antigo_sem_referencia_cai_nos_goldens(tmp_path: Path) -> None:
    # Respostas geradas antes de o arquivo ser autocontido: só nome e resposta.
    path = tmp_path / "antigo.json"
    path.write_text(json.dumps({"answers": [{"name": "ragas-definicao", "actual_output": "x"}]}), encoding="utf-8")

    _, (case,) = shared.load_answers(path)

    assert case.input.startswith("O que é o RAGAS")
    assert case.expected_output.startswith("RAGAS (Retrieval Augmented Generation Assessment)")


def test_relatorio_consolidado_le_o_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # O contrato entre as camadas: o tests/reports lê o JSON daqui sem saber do
    # deepeval (report_data só usa a biblioteca padrão, por isso abre neste venv).
    data = _run(tmp_path)
    (json_path,) = (tmp_path / "results").glob("*.json")
    monkeypatch.syspath_prepend(str(shared.PROJECT_ROOT / "tests" / "reports"))
    import report_data

    run = report_data.load_run(json_path)

    assert run.framework == "JevEval"
    assert run.judge_model == data["judge_model"]
    c1, c2, c3 = run.cases
    assert set(c1.scores) == {d.key for d in report_data.DIMENSIONS}  # a correção fica fora das dimensões
    assert c2.abstained
    assert not c2.score("faithfulness").applicable
    stats = report_data.dimension_stats(run)["faithfulness"]
    assert (stats.n_cases, stats.n_measured, stats.n_not_applicable) == (2, 1, 1)
    assert [name for name, _ in stats.errors] == ["c3-falhou"]
