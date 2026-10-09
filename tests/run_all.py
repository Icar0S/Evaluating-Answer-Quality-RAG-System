"""Roda todas as suítes de teste do projeto e, no fim, gera o relatório consolidado.

Uso, da raiz do repositório (ou pelo atalho scripts\\run_all_tests.bat):

    backend\\.venv\\Scripts\\python.exe tests\\run_all.py              # tudo (~40 min a horas: o DeepEval domina)
    backend\\.venv\\Scripts\\python.exe tests\\run_all.py --rapido     # só API, gerador e E2E (minutos)
    backend\\.venv\\Scripts\\python.exe tests\\run_all.py --limite 2   # qualidade em 2 goldens (fumaça)
    backend\\.venv\\Scripts\\python.exe tests\\run_all.py --pular e2e deepeval --abrir

Ordem, da mais rápida para a mais lenta:

 1. API (pytest, hermética)                       backend/.venv
 2. Gerador de relatório (pytest, hermética)      tests/reports/.venv
 3. E2E (Playwright)                              frontend/tests/node_modules
    sobe o backend sozinho se ele não estiver no ar, e derruba no fim
 4. Respostas do assistente, geradas UMA vez      backend/.venv
 5. RAGAS: controles do juiz (pytest)             tests/ragas/.venv
 6. RAGAS: avaliação dos goldens                  tests/ragas/.venv
 7. DeepEval: avaliação dos goldens               tests/deepeval/.venv
 8. Relatório consolidado                         tests/reports/.venv

As avaliações (6 e 7) julgam as MESMAS respostas (passo 4): é isso que faz a
comparação RAGAS x DeepEval do relatório ser uma comparação de métodos. Elas
rodam pelo run_and_export.py de cada suíte, que mede as mesmas métricas com o
mesmo limiar do pytest delas, sem gerar e julgar tudo duas vezes; para rodar
também o pytest de qualidade, use --qualidade-pytest.

Fica de fora: a campanha do estudo de mutação (tests/mutation), que leva horas
e tem protocolo próprio (ver o README de lá).

Tudo de uma execução fica em tests/reports/out/<data>/: o relatório, o
execucao.json, as respostas compartilhadas e o log de cada suíte (logs/). Os
relatórios anteriores continuam lá; o índice é tests/reports/out/HISTORICO.md
e o mais recente tem cópia fixa em tests/reports/out/ultimo/.

Só biblioteca padrão: roda com qualquer Python 3.10+.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "tests" / "reports" / "out"
BACKEND_PORT = 8010
FRONTEND_PORT = 5510
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
SUITES = ("api", "relatorio", "e2e", "ragas", "deepeval")

STATUS_TAG = {"passou": "[ OK  ]", "concluída": "[ OK  ]", "falhou": "[FALHA]", "pulada": "[PULOU]", "erro": "[ERRO ]"}


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


PY = {
    "backend": venv_python(ROOT / "backend" / ".venv"),
    "reports": venv_python(ROOT / "tests" / "reports" / ".venv"),
    "ragas": venv_python(ROOT / "tests" / "ragas" / ".venv"),
    "deepeval": venv_python(ROOT / "tests" / "deepeval" / ".venv"),
}
SETUP_HINT = {
    "backend": "crie com: python -m venv backend\\.venv && backend\\.venv\\Scripts\\pip install -r backend\\requirements.txt",
    "reports": "crie com: python -m venv tests\\reports\\.venv && tests\\reports\\.venv\\Scripts\\pip install -r tests\\reports\\requirements.txt",
    "ragas": "crie com: python -m venv tests\\ragas\\.venv && tests\\ragas\\.venv\\Scripts\\pip install -r tests\\ragas\\requirements.txt",
    "deepeval": "crie com: python -m venv tests\\deepeval\\.venv && tests\\deepeval\\.venv\\Scripts\\pip install -r backend\\requirements-eval.txt",
}


@dataclass
class Step:
    key: str
    name: str
    status: str = "pulada"
    detail: str = ""
    tests: dict | None = None
    seconds: float | None = None
    log: str | None = None
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ utilidades


def _child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    # Filhos Python escrevendo em pipe no Windows usariam cp1252 e quebrariam
    # num acento; UTF-8 explícito em todos.
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.update(extra or {})
    return env


def run_logged(cmd: list, log_path: Path, cwd: Path = ROOT, env: dict | None = None, timeout: float | None = None,
               echo: bool = False) -> tuple[int | None, float]:
    """Roda `cmd` gravando tudo em `log_path`; com echo, mostra o progresso no console.

    Devolve (código de saída ou None se estourou o tempo, segundos).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, env=_child_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )

        def pump() -> None:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if echo and line.strip():
                    print("        " + line.rstrip(), flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = None
        reader.join(timeout=10)
    return code, time.monotonic() - t0


def junit_counts(path: Path) -> dict | None:
    """total/failed/errors/skipped de um JUnit XML (pytest e Playwright gravam esse formato)."""
    if not path.is_file():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    suites = list(root.iter("testsuite")) or [root]
    count = {"total": 0, "failed": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        count["total"] += int(suite.get("tests", 0))
        count["failed"] += int(suite.get("failures", 0))
        count["errors"] += int(suite.get("errors", 0))
        count["skipped"] += int(suite.get("skipped", 0))
    return count


def status_from(code: int | None, tests: dict | None) -> tuple[str, str]:
    """Resultado de uma suíte de testes a partir do código de saída e da contagem."""
    if code is None:
        return "erro", "estourou o tempo limite"
    if tests and tests["total"]:
        broken = tests["failed"] + tests["errors"]
        passed = tests["total"] - broken - tests["skipped"]
        summary = f"{passed} passaram, {broken} falharam, {tests['skipped']} pulados"
        if broken:
            return "falhou", summary
        if tests["skipped"] == tests["total"]:
            return "pulada", "todos os testes se auto-pularam (pré-requisito ausente; ver o log)"
        return "passou", summary
    if code == 5:  # pytest: nenhum teste coletado
        return "pulada", "nenhum teste coletado"
    return ("passou", "") if code == 0 else ("erro", f"saiu com código {code} sem relatório de testes (ver o log)")


def http_get(url: str, timeout: float = 3) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - só localhost
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - qualquer falha = indisponível
        return None


def backend_is_ours() -> bool:
    body = http_get(f"{BACKEND_URL}/health")
    return bool(body) and "ollama_reachable" in body


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def frontend_port_blocked() -> bool:
    """5510 ocupada por algo que NÃO é o nosso frontend (ex.: Live Server do VS Code).

    O Playwright reaproveita qualquer servidor que já esteja na porta
    (reuseExistingServer) e os testes falhariam contra o site errado.
    """
    if not port_in_use(FRONTEND_PORT):
        return False
    body = http_get(f"http://localhost:{FRONTEND_PORT}/chat.html")
    return not (body and "components/chat.js" in body)


def ollama_url() -> str:
    """OLLAMA_BASE_URL do .env (sem depender de pydantic), ou o padrão."""
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("OLLAMA_BASE_URL="):
                return line.split("=", 1)[1].strip() or "http://localhost:11434"
    return "http://localhost:11434"


def results_snapshot(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.json")} if directory.is_dir() else set()


def quality_detail(path: Path) -> tuple[str, dict]:
    """Resumo de um JSON de avaliação: notas medidas e notas >= limiar."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = [m for c in data.get("cases", []) for m in c.get("metrics", [])]
    measured = sum(1 for m in metrics if m.get("score") is not None)
    passed = sum(1 for m in metrics if m.get("success"))
    cases = data.get("cases", [])
    all_ok = sum(1 for c in cases if c.get("success"))
    detail = (
        f"{measured}/{len(metrics)} notas medidas, {passed}/{len(metrics)} ≥ limiar, "
        f"{all_ok}/{len(cases)} casos com todas ≥ limiar"
    )
    return detail, {"total": len(metrics), "failed": len(metrics) - passed - (len(metrics) - measured),
                    "errors": len(metrics) - measured, "skipped": 0}


# ----------------------------------------------------------------------- passos


class Runner:
    def __init__(self, args: argparse.Namespace, out_dir: Path) -> None:
        self.args = args
        self.out_dir = out_dir
        self.logs = out_dir / "logs"
        self.junit = out_dir / "junit"
        self.steps: list[Step] = []
        self.new_results: dict[str, Path] = {}
        self.shared_answers: dict | None = None
        self.answers_file: Path | None = None

    def wants(self, suite: str) -> bool:
        if suite in self.args.pular:
            return False
        if self.args.rapido and suite in ("ragas", "deepeval"):
            return False
        return True

    def record(self, step: Step) -> Step:
        self.steps.append(step)
        extra = f"  {step.detail}" if step.detail else ""
        took = f"  ({step.seconds:.0f}s)" if step.seconds is not None else ""
        print(f"    {STATUS_TAG.get(step.status, step.status)} {step.name}{took}{extra}", flush=True)
        return step

    def announce(self, name: str) -> None:
        print(f"\n--> {name}", flush=True)

    def skip(self, key: str, name: str, why: str) -> Step:
        return self.record(Step(key=key, name=name, status="pulada", detail=why))

    def pytest_step(self, key: str, name: str, venv: str, target: str, timeout: float) -> Step:
        self.announce(name)
        python = PY[venv]
        if not python.exists():
            return self.skip(key, name, f"venv não encontrado ({SETUP_HINT[venv]})")
        junit = self.junit / f"{key}.xml"
        log = self.logs / f"{key}.log"
        code, secs = run_logged(
            [python, "-m", "pytest", target, "-p", "no:cacheprovider", f"--junitxml={junit}", "-rfE"],
            log, timeout=timeout,
        )
        tests = junit_counts(junit)
        if tests is None and code not in (None, 0, 5):
            # pytest nem chegou a rodar (ex.: pytest não instalado nesse venv).
            return self.record(Step(key, name, "erro", "pytest não rodou (ver o log)", None, secs, self._rel(log)))
        status, detail = status_from(code, tests)
        return self.record(Step(key, name, status, detail, tests, secs, self._rel(log)))

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.out_dir).as_posix()

    # 3. E2E -------------------------------------------------------------------
    def e2e(self) -> None:
        name = "E2E (Playwright)"
        self.announce(name)
        tests_dir = ROOT / "frontend" / "tests"
        npx = shutil.which("npx")
        if not (tests_dir / "node_modules").is_dir() or not npx:
            self.skip("e2e", name, "Playwright não instalado (cd frontend\\tests && npm install && npx playwright install chromium)")
            return
        if frontend_port_blocked():
            self.skip("e2e", name, f"porta {FRONTEND_PORT} ocupada por outro serviço (ex.: Live Server do VS Code); "
                                   "feche-o ou veja o README, \"Problemas comuns\"")
            return
        backend_proc = None
        backend_log = None
        backend_note = "backend já estava no ar"
        if not backend_is_ours():
            if port_in_use(BACKEND_PORT):
                backend_note = "porta 8010 ocupada por outro serviço: testes com geração real vão se auto-pular"
            elif PY["backend"].exists():
                print("    subindo o backend para os testes com geração real...", flush=True)
                backend_log = (self.logs / "backend.log").open("w", encoding="utf-8", errors="replace")
                backend_proc = subprocess.Popen(
                    [str(PY["backend"]), "-m", "uvicorn", "app.main:app", "--port", "8010"],
                    cwd=ROOT / "backend", env=_child_env(), stdout=backend_log, stderr=subprocess.STDOUT,
                )
                for _ in range(60):
                    if backend_is_ours():
                        break
                    time.sleep(1)
                backend_note = (
                    "backend iniciado pelo script e encerrado no fim" if backend_is_ours()
                    else "backend não subiu em 60 s (ver logs/backend.log): testes com geração real vão se auto-pular"
                )
            else:
                backend_note = "backend fora do ar e sem venv: testes com geração real vão se auto-pular"
        try:
            junit = self.junit / "e2e.xml"
            log = self.logs / "e2e.log"
            code, secs = run_logged(
                [npx, "playwright", "test", "--reporter=list,junit"], log, cwd=tests_dir,
                env={"PLAYWRIGHT_JUNIT_OUTPUT_NAME": str(junit)}, timeout=1800,
            )
            tests = junit_counts(junit)
            status, detail = status_from(code, tests)
            self.record(Step("e2e", name, status, f"{detail}; {backend_note}".strip("; "), tests, secs, self._rel(log)))
        finally:
            if backend_proc is not None:
                backend_proc.terminate()
                try:
                    backend_proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    backend_proc.kill()
            if backend_log is not None:
                backend_log.close()

    # 4-7. Qualidade ----------------------------------------------------------
    def quality(self) -> None:
        do_ragas, do_deepeval = self.wants("ragas"), self.wants("deepeval")
        if not (do_ragas or do_deepeval):
            return
        if http_get(f"{ollama_url()}/api/tags") is None:
            for key, name, wanted in (("ragas", "RAGAS", do_ragas), ("deepeval", "DeepEval", do_deepeval)):
                if wanted:
                    self.skip(key, f"{name}: avaliação", f"Ollama fora do ar em {ollama_url()}")
            return

        if not self.args.respostas_separadas:
            self.generate_answers()
            if self.answers_file is None:
                for key, name, wanted in (("ragas", "RAGAS", do_ragas), ("deepeval", "DeepEval", do_deepeval)):
                    if wanted:
                        self.skip(key, f"{name}: avaliação", "sem respostas para julgar (ver o passo anterior)")
                return

        if do_ragas:
            self.pytest_step("ragas_controles", "RAGAS: controles do juiz", "ragas",
                             "tests/ragas/test_judge_controls.py", timeout=1800)
            self.evaluation("ragas", "RAGAS: avaliação", "ragas", ROOT / "tests" / "ragas" / "run_and_export.py",
                            ROOT / "tests" / "ragas" / "results", [])
            if self.args.qualidade_pytest:
                self.pytest_step("ragas_pytest", "RAGAS: pytest de qualidade", "ragas",
                                 "tests/ragas/test_rag_quality.py", timeout=4 * 3600)
        if do_deepeval:
            self.evaluation("deepeval", "DeepEval: avaliação", "deepeval",
                            ROOT / "tests" / "deepeval" / "run_and_export.py", ROOT / "tests" / "deepeval" / "results",
                            ["--no-open"])
            if self.args.qualidade_pytest:
                self.pytest_step("deepeval_pytest", "DeepEval: pytest de qualidade", "deepeval",
                                 "tests/deepeval", timeout=4 * 3600)

    def generate_answers(self) -> None:
        name = "Respostas do assistente (geradas uma vez)"
        self.announce(name)
        if not PY["backend"].exists():
            self.skip("respostas", name, f"venv do backend não encontrado ({SETUP_HINT['backend']})")
            return
        target = self.out_dir / "respostas.json"
        cmd = [PY["backend"], ROOT / "tests" / "generate_answers.py", "--out", target]
        if self.args.limite:
            cmd += ["--limit", self.args.limite]
        log = self.logs / "respostas.log"
        code, secs = run_logged(cmd, log, timeout=3600, echo=True)
        if code != 0 or not target.is_file():
            self.record(Step("respostas", name, "erro", "não gerou as respostas (ver o log)", None, secs, self._rel(log)))
            return
        data = json.loads(target.read_text(encoding="utf-8"))
        failures = sum(1 for a in data["answers"] if a.get("error"))
        self.answers_file = target
        self.shared_answers = {
            "file": target.name,
            "n": len(data["answers"]),
            "failures": failures,
            "generation_model": data.get("generation_model"),
        }
        self.record(Step("respostas", name, "concluída",
                         f"{len(data['answers'])} resposta(s) com {data.get('generation_model')} ({failures} falha(s))",
                         None, secs, self._rel(log)))

    def evaluation(self, key: str, name: str, venv: str, script: Path, results_dir: Path, extra: list) -> None:
        self.announce(name)
        if not PY[venv].exists():
            self.skip(key, name, f"venv não encontrado ({SETUP_HINT[venv]})")
            return
        before = results_snapshot(results_dir)
        cmd = [PY[venv], script, *extra]
        if self.answers_file:
            cmd += ["--answers", self.answers_file]
        if self.args.limite:
            cmd += ["--limit", self.args.limite]
        log = self.logs / f"{key}.log"
        code, secs = run_logged(cmd, log, timeout=4 * 3600, echo=True)
        new = sorted(results_snapshot(results_dir) - before)
        if code is None:
            self.record(Step(key, name, "erro", "estourou o tempo limite", None, secs, self._rel(log)))
            return
        if code != 0 or not new:
            self.record(Step(key, name, "erro", f"saiu com código {code} sem resultado novo (ver o log)",
                             None, secs, self._rel(log)))
            return
        result = results_dir / new[-1]
        self.new_results[key] = result
        detail, tests = quality_detail(result)
        self.record(Step(key, name, "concluída", detail, tests, secs, self._rel(log),
                         extra={"result": str(result.relative_to(ROOT))}))

    # 8. Relatório ------------------------------------------------------------
    def report(self, started: datetime, t0: float) -> Path | None:
        self.announce("Relatório consolidado")
        reused = []
        inputs = list(self.new_results.values())
        produced = {("RAGAS" if k == "ragas" else "DeepEval") for k in self.new_results}
        for framework, directory in (("DeepEval", ROOT / "tests" / "deepeval" / "results"),
                                     ("RAGAS", ROOT / "tests" / "ragas" / "results")):
            if framework in produced:
                continue
            candidates = sorted(p for p in directory.glob("*.json") if p.stem[:8].isdigit()) if directory.is_dir() else []
            if candidates:
                inputs.append(candidates[-1])
                reused.append(str(candidates[-1].relative_to(ROOT)))

        execution = {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "duration_seconds": round(time.monotonic() - t0, 1),
            "options": {k: v for k, v in vars(self.args).items()},
            "shared_answers": self.shared_answers,
            "reused_results": reused,
            "steps": [asdict(s) for s in self.steps],
        }
        exec_path = self.out_dir / "execucao.json"
        exec_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        if not PY["reports"].exists():
            print(f"    [ERRO ] sem o venv do relatório ({SETUP_HINT['reports']})", flush=True)
            return None
        if not inputs:
            print("    [PULOU] nenhum resultado de avaliação (rode sem --rapido ao menos uma vez)", flush=True)
            return None
        if reused:
            print("    reaproveitando resultados anteriores para: " + ", ".join(reused), flush=True)
        cmd = [PY["reports"], ROOT / "tests" / "reports" / "build_report.py", "--out", self.out_dir,
               "--execution", exec_path]
        for path in inputs:
            cmd += ["--input", path]
        code, _ = run_logged(cmd, self.logs / "relatorio.log", timeout=900, echo=True)
        if code != 0:
            print("    [ERRO ] o relatório não foi gerado (ver logs/relatorio.log)", flush=True)
            return None
        return self.out_dir / "relatorio.pdf"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Roda todas as suítes de teste e gera o relatório consolidado.")
    parser.add_argument("--rapido", action="store_true", help="só API, gerador de relatório e E2E (sem os juízes)")
    parser.add_argument("--pular", nargs="+", default=[], choices=SUITES, help="suítes a pular")
    parser.add_argument("--limite", type=int, default=None, help="avaliações de qualidade só nos N primeiros goldens")
    parser.add_argument("--respostas-separadas", action="store_true",
                        help="cada framework gera as próprias respostas (comportamento antigo; compara pior)")
    parser.add_argument("--qualidade-pytest", action="store_true",
                        help="roda também o pytest de qualidade do RAGAS e do DeepEval (dobra o tempo)")
    parser.add_argument("--abrir", action="store_true", help="abre o PDF do relatório ao terminar")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).astimezone()
    t0 = time.monotonic()
    out_dir = OUT_ROOT / started.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args, out_dir)

    print(f"=== Testes do RAG — {started.strftime('%d/%m/%Y %H:%M')} ===")
    print(f"Saída: {out_dir}")

    if runner.wants("api"):
        runner.pytest_step("api", "API (pytest)", "backend", "tests/api", timeout=900)
    if runner.wants("relatorio"):
        runner.pytest_step("relatorio", "Gerador de relatório (pytest)", "reports", "tests/reports", timeout=900)
    if runner.wants("e2e"):
        runner.e2e()
    runner.quality()
    pdf = runner.report(started, t0)

    total = time.monotonic() - t0
    print(f"\n=== Resumo ({total / 60:.1f} min) ===")
    for step in runner.steps:
        print(f"  {STATUS_TAG.get(step.status, step.status)} {step.name}")
    if pdf and pdf.is_file():
        print(f"\nRelatório: {pdf}")
        print(f"Histórico: {OUT_ROOT / 'HISTORICO.md'}")
        print(f"Último relatório (caminho fixo): {OUT_ROOT / 'ultimo' / 'relatorio.pdf'}")
        if args.abrir and hasattr(os, "startfile"):
            os.startfile(pdf)  # noqa: S606 - abre no visualizador padrão do Windows
    broken = [s for s in runner.steps if s.status in ("falhou", "erro")]
    return 1 if broken or pdf is None else 0


if __name__ == "__main__":
    sys.exit(main())
