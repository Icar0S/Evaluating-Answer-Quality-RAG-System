"""Testes do orquestrador (tests/run_all.py) e do histórico de relatórios.

Herméticos: nenhuma suíte de verdade roda aqui. Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

import build_report
import report_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, para importar run_all
import run_all  # noqa: E402

from test_build_report import deepeval_file, jeveval_file, ragas_file  # noqa: E402,F401 - fixtures reaproveitadas

PYTEST_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="2" tests="10" time="1.2"/></testsuites>"""
PLAYWRIGHT_JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites id="" name="" tests="7" failures="0" skipped="1" errors="0" time="12.3">
  <testsuite name="chat-flow.spec.ts" tests="3" failures="0" skipped="1" errors="0"/>
  <testsuite name="monitor.spec.ts" tests="4" failures="0" skipped="0" errors="0"/>
</testsuites>"""


def test_le_junit_do_pytest(tmp_path: Path) -> None:
    path = tmp_path / "api.xml"
    path.write_text(PYTEST_JUNIT, encoding="utf-8")

    assert run_all.junit_counts(path) == {"total": 10, "failed": 1, "errors": 0, "skipped": 2}


def test_le_junit_do_playwright_somando_as_suites_sem_contar_o_total_duas_vezes(tmp_path: Path) -> None:
    path = tmp_path / "e2e.xml"
    path.write_text(PLAYWRIGHT_JUNIT, encoding="utf-8")

    assert run_all.junit_counts(path) == {"total": 7, "failed": 0, "errors": 0, "skipped": 1}


def test_junit_ausente_ou_quebrado_vira_none(tmp_path: Path) -> None:
    assert run_all.junit_counts(tmp_path / "nao_existe.xml") is None
    quebrado = tmp_path / "x.xml"
    quebrado.write_text("<testsuites", encoding="utf-8")
    assert run_all.junit_counts(quebrado) is None


@pytest.mark.parametrize(
    ("code", "tests", "status"),
    [
        (0, {"total": 5, "failed": 0, "errors": 0, "skipped": 0}, "passou"),
        (1, {"total": 5, "failed": 1, "errors": 0, "skipped": 0}, "falhou"),
        (0, {"total": 5, "failed": 0, "errors": 0, "skipped": 5}, "pulada"),
        (5, None, "pulada"),
        (None, None, "erro"),
        (2, None, "erro"),
    ],
)
def test_status_da_suite(code, tests, status) -> None:
    assert run_all.status_from(code, tests)[0] == status


def test_resumo_de_uma_avaliacao_de_qualidade(ragas_file: Path) -> None:
    detail, tests = run_all.quality_detail(ragas_file)

    # 2 casos x 5 métricas; c2 tem answer_relevancy 0,0 (medida, reprovada).
    assert tests == {"total": 10, "failed": 1, "errors": 0, "skipped": 0}
    assert "10/10 notas medidas" in detail


def test_resumo_conta_na_como_pulada_e_nao_como_erro(jeveval_file: Path) -> None:
    detail, tests = run_all.quality_detail(jeveval_file)

    # 2 casos x 6 métricas (5 dimensões + correção). No c2, uma recusa: fidelidade
    # e relevância saem N/A, e a correção 0,1 é medida e reprova.
    assert tests == {"total": 12, "failed": 1, "errors": 0, "skipped": 2}
    assert "10/10 notas medidas" in detail
    assert "9/10 ≥ limiar" in detail
    assert "2 N/A" in detail


def _runner(tmp_path: Path) -> run_all.Runner:
    args = argparse.Namespace(pular=[], rapido=False, limite=None, respostas_separadas=False,
                              qualidade_pytest=False, abrir=False)
    return run_all.Runner(args, tmp_path)


def test_jeveval_pula_sem_chave_da_typesafe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(run_all, "dotenv_value", lambda name: None)
    runner = _runner(tmp_path)
    runner.answers_file = tmp_path / "respostas.json"

    runner.jeveval()

    assert [(s.key, s.status) for s in runner.steps] == [("jeveval_controles", "pulada"), ("jeveval", "pulada")]
    assert "TYPESAFE_API_KEY" in runner.steps[0].detail


def test_jeveval_pula_sem_respostas_compartilhadas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Com --respostas-separadas não há o arquivo único: o JevEval não gera respostas.
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    runner = _runner(tmp_path)

    runner.jeveval()

    assert {s.status for s in runner.steps} == {"pulada"}
    assert "respostas compartilhadas" in runner.steps[0].detail


def test_rapido_roda_so_os_hermeticos_do_jeveval(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.args.rapido = True

    assert runner.wants("jeveval")
    assert not runner.wants_live("jeveval")


EXECUTION = {
    "started_at": "2026-10-07T11:30:00-03:00",
    "duration_seconds": 3000,
    "shared_answers": {"file": "respostas.json", "n": 10, "failures": 0, "generation_model": "gemma3:4b"},
    "reused_results": [],
    "steps": [
        {"key": "api", "name": "API (pytest)", "status": "passou", "detail": "22 passaram",
         "tests": {"total": 22, "failed": 0, "errors": 0, "skipped": 0}, "seconds": 3.0, "log": "logs/api.log"},
        {"key": "ragas_controles", "name": "RAGAS: controles do juiz", "status": "falhou", "detail": "1 falhou",
         "tests": {"total": 5, "failed": 1, "errors": 0, "skipped": 0}, "seconds": 60.0,
         "log": "logs/ragas_controles.log"},
        {"key": "ragas", "name": "RAGAS: avaliação", "status": "concluída", "detail": "50/50 notas medidas",
         "tests": {"total": 50, "failed": 30, "errors": 0, "skipped": 0}, "seconds": 900.0, "log": "logs/ragas.log"},
    ],
}


def test_relatorio_mostra_a_execucao_e_avisa_quando_os_controles_reprovam(ragas_file: Path, tmp_path: Path) -> None:
    outputs = build_report.build([ragas_file], ["md", "xlsx"], tmp_path / "out", execution=EXECUTION)

    md = outputs["md"].read_text(encoding="utf-8")
    assert "## 1. Execução dos testes" in md
    assert "✗ falhou" in md
    assert "✓ concluída" in md
    assert "mesmas respostas" in md
    assert "Controles do juiz RAGAS reprovaram" in md
    assert "Execução" in load_workbook(outputs["xlsx"]).sheetnames


def test_historico_guarda_cada_relatorio_e_conta_cada_rodada_uma_vez(
    ragas_file: Path, deepeval_file: Path, tmp_path: Path
) -> None:
    root = tmp_path / "out"
    build_report.build([ragas_file, deepeval_file], ["md"], root / "20261007T100000", history_root=root)
    # Segundo relatório a partir dos MESMOS resultados: nova entrada no índice,
    # mas as rodadas avaliadas não se duplicam na evolução.
    build_report.build([ragas_file, deepeval_file], ["md", "pdf"], root / "20261007T110000", history_root=root)

    entries = report_history.load_entries(root)
    assert [e["_folder"] for e in entries] == ["20261007T100000", "20261007T110000"]
    assert len(report_history.unique_runs(entries)) == 2

    index = (root / "HISTORICO.md").read_text(encoding="utf-8")
    assert index.index("20261007T110000") < index.index("20261007T100000")  # mais recente primeiro
    assert (root / "historico.csv").read_text(encoding="utf-8-sig").count("\n") == 1 + 2 * 5

    latest = root / report_history.LATEST_DIR
    assert (latest / "relatorio.pdf").is_file()
    assert json.loads((latest / "resumo.json").read_text(encoding="utf-8"))["files"]["pdf"] == "relatorio.pdf"


def test_relatorio_fora_da_pasta_de_historico_nao_mexe_no_indice(ragas_file: Path, tmp_path: Path) -> None:
    build_report.build([ragas_file], ["md"], tmp_path / "avulso")

    assert (tmp_path / "avulso" / "resumo.json").is_file()
    assert not (tmp_path / "HISTORICO.md").exists()


# ------------------------------------------------ modo --api (container de testes)
# No Docker, as respostas vêm da API do RAG no ar: o pipeline in-process leria o
# índice vazio da imagem e outro Ollama, e as notas seriam de outro RAG.


def test_ollama_url_segue_ambiente_depois_env_depois_padrao(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_all, "dotenv_value", lambda name: "http://do-dotenv:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    assert run_all.ollama_url() == "http://ollama:11434"  # o ambiente ganha, como no pydantic-settings

    monkeypatch.setenv("OLLAMA_BASE_URL", "  ")
    assert run_all.ollama_url() == "http://do-dotenv:11434"

    monkeypatch.delenv("OLLAMA_BASE_URL")
    monkeypatch.setattr(run_all, "dotenv_value", lambda name: None)
    assert run_all.ollama_url() == "http://localhost:11434"


def test_api_vem_de_RUN_ALL_API_URL_e_sem_ela_nada_muda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_ALL_API_URL", raising=False)
    assert run_all.build_parser().parse_args([]).api is None

    monkeypatch.setenv("RUN_ALL_API_URL", "http://localhost:8010")
    assert run_all.build_parser().parse_args([]).api == "http://localhost:8010"
    assert run_all.build_parser().parse_args(["--api", "http://outra:8010"]).api == "http://outra:8010"


@pytest.fixture
def fake_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """run_logged falso: grava um respostas.json como o generate_answers.py e guarda o comando."""
    calls: dict = {"cmds": [], "unloaded": []}
    python = tmp_path / "python-do-backend"
    python.write_text("", encoding="utf-8")
    monkeypatch.setitem(run_all.PY, "backend", python)

    def fake_run_logged(cmd, log_path, **kwargs):
        calls["cmds"].append([str(c) for c in cmd])
        target = Path(cmd[cmd.index("--out") + 1])
        target.write_text(json.dumps({"generation_model": "gemma3:4b", "answers": [{"name": "g1", "error": None}]}),
                          encoding="utf-8")
        return 0, 1.0

    monkeypatch.setattr(run_all, "run_logged", fake_run_logged)
    monkeypatch.setattr(run_all, "unload_model", lambda url, model: calls["unloaded"].append((url, model)) or True)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    calls["python"] = str(python)
    return calls


def test_sem_api_as_respostas_saem_do_pipeline_in_process_como_antes(tmp_path: Path, fake_generation: dict) -> None:
    runner = _runner(tmp_path)  # Namespace sem o campo api, como o demo_e2e.py monta

    runner.generate_answers()

    cmd = fake_generation["cmds"][0]
    assert cmd[0] == fake_generation["python"]
    assert cmd[1].endswith("generate_answers.py") and "--api" not in cmd
    assert runner.steps[0].name == "Respostas do assistente (geradas uma vez)"
    assert "api_url" not in runner.shared_answers
    assert fake_generation["unloaded"] == []  # na máquina, o gerador fica onde está


def test_com_api_as_respostas_vem_da_api_e_o_gerador_sai_da_memoria(tmp_path: Path, fake_generation: dict) -> None:
    runner = _runner(tmp_path)
    runner.args.api = "http://localhost:8010"
    runner.args.limite = 1

    runner.generate_answers()

    cmd = fake_generation["cmds"][0]
    assert cmd[cmd.index("--api") + 1] == "http://localhost:8010"
    assert cmd[cmd.index("--limit") + 1] == "1"
    step = runner.steps[0]
    assert (step.name, step.status) == ("Respostas do assistente (pela API em http://localhost:8010)", "concluída")
    assert runner.shared_answers["api_url"] == "http://localhost:8010"
    assert runner.answers_file == tmp_path / "respostas.json"
    # Os juízes não usam o gerador: sai da memória do Ollama da stack antes deles (o OOM de 13/10/2026).
    assert fake_generation["unloaded"] == [("http://ollama:11434", "gemma3:4b")]


def test_com_api_o_pytest_de_qualidade_sai_pulado_com_o_motivo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran = []
    monkeypatch.setattr(run_all.Runner, "pytest_step", lambda self, key, *a, **k: ran.append(key))
    runner = _runner(tmp_path)
    runner.args.api = "http://localhost:8010"

    runner.quality_pytest("ragas_pytest", "RAGAS: pytest de qualidade", "ragas", "tests/ragas/test_rag_quality.py")

    assert ran == []
    assert (runner.steps[0].key, runner.steps[0].status) == ("ragas_pytest", "pulada")
    assert "in-process" in runner.steps[0].detail and "índice local" in runner.steps[0].detail

    runner.args.api = None
    runner.quality_pytest("deepeval_pytest", "DeepEval: pytest de qualidade", "deepeval", "tests/deepeval")
    assert ran == ["deepeval_pytest"]  # sem --api, roda como antes


def test_unload_model_so_descarrega_o_que_esta_carregado(monkeypatch: pytest.MonkeyPatch) -> None:
    posted = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        posted.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(run_all.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(run_all, "http_get", lambda url, timeout=3: json.dumps({"models": [{"name": "gemma3:4b"}]}))

    assert run_all.unload_model("http://ollama:11434", "qwen3:8b") is False
    assert run_all.unload_model("http://ollama:11434", "gemma3:4b") is True
    assert posted == [{"model": "gemma3:4b", "keep_alive": 0}]
