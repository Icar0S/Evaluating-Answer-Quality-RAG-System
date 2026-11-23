"""Roda a pergunta única de ponta a ponta (plano-escopo-final.md): as 5 respostas, os juízes repetidos e a análise.

Uso, da raiz do repositório:

    backend\\.venv\\Scripts\\python.exe tests\\pergunta_unica\\rodar.py                  # análise: 3 julgamentos
    backend\\.venv\\Scripts\\python.exe tests\\pergunta_unica\\rodar.py --julgamentos 1  # apresentação: 1
    backend\\.venv\\Scripts\\python.exe tests\\pergunta_unica\\rodar.py --pular deepeval jeveval
    backend\\.venv\\Scripts\\python.exe tests\\pergunta_unica\\rodar.py --retomar tests\\pergunta_unica\\out\\<pasta>
    python tests/pergunta_unica/rodar.py --api http://localhost:8010 --ollama http://127.0.0.1:11435   # Docker

Os passos:

1. **As respostas** (tests/generate_answers.py): a pergunta (`--golden`, padrão
   `md-mtbench-vieses`), uma resposta por semente (`--sementes`, padrão 1-5),
   cada uma com o gerador recarregado e todas com os mesmos trechos. In-process,
   no índice `--indice` (padrão `md-v1`) e com a temperatura 1 (a padrão do
   gemma3:4b, fixada); com `--api`, o RAG no ar.
2. **O gerador sai da memória** do Ollama: o juiz (qwen3:8b) e o gerador
   (gemma3:4b) não cabem juntos na GPU de 8 GB, e o Ollama trocaria um pelo outro
   no meio do julgamento.
3. **Os juízes, K vezes** (`--julgamentos`, padrão 3): para k = 1..K, RAGAS,
   DeepEval e JevEval julgam as mesmas respostas com `--repeticao k
   --ordem-semente k` (a ordem dos casos muda, com semente, porque o juiz também
   passa pelo cache do Ollama) e com os passos de cada nota (`--passos`). Por k,
   e não por juiz: uma rodada interrompida deixa julgamentos completos. Sem
   TYPESAFE_API_KEY, o JevEval sai como pulado; é a única etapa que manda
   conteúdo para fora da máquina (tests/jeveval/README.md).
4. **A análise** (analise.py): por juiz e métrica, a média, a faixa e quanto da
   variação é do gerador (entre as respostas) e quanto é do juiz (entre os
   julgamentos da mesma resposta).

Tudo fica em tests/pergunta_unica/out/<data>/: respostas.json, um export por
juiz e julgamento (ragas/, deepeval/, jeveval/), analise.json e analise.md, o
log de cada passo (logs/) e o execucao.json. Com `--retomar <pasta>`, o que já
tem arquivo completo não roda de novo.

Só biblioteca padrão: roda com qualquer Python 3.10+.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_ROOT = HERE / "out"
sys.path.insert(0, str(HERE))

import analise  # noqa: E402


def _load_run_all():
    """tests/run_all.py pelo caminho: os venvs (PY), a chave do Jev e o descarregar do Ollama são os dele."""
    spec = importlib.util.spec_from_file_location("run_all_pergunta_unica", ROOT / "tests" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # o @dataclass do run_all precisa do módulo registrado
    spec.loader.exec_module(module)
    return module


run_all = _load_run_all()

DEFAULT_GOLDEN = "md-mtbench-vieses"
DEFAULT_GOLDENS = "md=tests/deepeval/goldens/dataset_md.json"
DEFAULT_INDEX = "md-v1"
# A padrão do gemma3:4b (ollama show, 23/11/2026), fixada para ficar gravada (docs/escopo-final-status.md, §2).
GENERATION_TEMPERATURE = 1
JUDGES = ("ragas", "deepeval", "jeveval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A pergunta única: 5 respostas, os juízes repetidos e a análise.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help=f"a pergunta (padrão {DEFAULT_GOLDEN})")
    parser.add_argument("--goldens", default=DEFAULT_GOLDENS, help=f"o arquivo dela, tag=caminho ({DEFAULT_GOLDENS})")
    parser.add_argument("--indice", default=DEFAULT_INDEX, help=f"versão do índice, in-process (padrão {DEFAULT_INDEX})")
    parser.add_argument("--sementes", default="1-5", help="uma resposta por semente (padrão 1-5)")
    parser.add_argument("--julgamentos", type=int, default=3, help="julgamentos por resposta e por juiz (padrão 3)")
    parser.add_argument("--pular", nargs="*", default=[], choices=JUDGES, help="juízes que não rodam")
    parser.add_argument("--api", default=None, metavar="URL", help="as respostas vêm do RAG no ar (POST URL/chat)")
    parser.add_argument("--ollama", default=None, metavar="URL",
                        help="o Ollama do gerador (obrigatório com --api; no Docker, http://127.0.0.1:11435)")
    parser.add_argument("--retomar", type=Path, default=None, metavar="PASTA", help="continua uma rodada")
    return parser


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.dir = args.retomar or OUT_ROOT / stamp
        self.logs = self.dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.answers = self.dir / "respostas.json"
        self.steps: list[dict] = []
        self.env = dict(os.environ, PYTHONUTF8="1")
        # localhost custa ~2 s por chamada nesta máquina (o Ollama só escuta IPv4); o juiz faz dezenas de chamadas.
        if "OLLAMA_BASE_URL" not in self.env:
            self.env["OLLAMA_BASE_URL"] = run_all.ollama_url().replace("://localhost", "://127.0.0.1")

    # ---------------------------------------------------------------- execução

    def record(self, key: str, status: str, detail: str = "", seconds: float | None = None, **extra) -> None:
        self.steps.append({"passo": key, "estado": status, "detalhe": detail, "segundos": seconds, **extra})
        tag = run_all.STATUS_TAG.get(status, f"[{status}]")
        print(f"{tag} {key}: {detail}" + (f" ({seconds / 60:.1f} min)" if seconds else ""), flush=True)
        self.save()

    def save(self) -> None:
        (self.dir / "execucao.json").write_text(json.dumps({
            "plano": "plano-escopo-final.md",
            "pasta": str(self.dir),
            "argumentos": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
            "ollama_dos_juizes": self.env.get("OLLAMA_BASE_URL"),
            "passos": self.steps,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def call(self, key: str, cmd: list, log_name: str) -> tuple[int, float]:
        log = self.logs / f"{log_name}.log"
        print(f"\n==> {key}\n    {' '.join(str(c) for c in cmd)}\n    log: {log}", flush=True)
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=self.env, stdout=fh, stderr=subprocess.STDOUT,
                                  check=False)
        return proc.returncode, round(time.perf_counter() - t0, 1)

    # ------------------------------------------------------------------- passos

    def answers_ready(self) -> bool:
        if not self.answers.is_file():
            return False
        data = json.loads(self.answers.read_text(encoding="utf-8"))
        protocol = data.get("protocolo") or {}
        return (not protocol.get("interrompido") and len(data.get("answers", [])) == data.get("n_cases")
                and all(a.get("error") is None for a in data["answers"]))

    def generate(self) -> bool:
        if self.answers_ready():
            self.record("respostas", "concluída", f"já existiam em {self.answers.name}")
            return True
        a = self.args
        common = ["--out", self.answers, "--goldens", a.goldens, "--only", a.golden, "--sementes", a.sementes,
                  "--estado-limpo"]
        if a.api:
            if not a.ollama:
                self.record("respostas", "erro", "--api precisa de --ollama (o Ollama do backend, para o estado limpo)")
                return False
            cmd = [sys.executable, ROOT / "tests" / "generate_answers.py", *common, "--api", a.api, "--ollama", a.ollama]
        else:
            python = run_all.PY["backend"]
            if not python.exists():
                self.record("respostas", "erro", f"venv do backend não existe: {run_all.SETUP_HINT['backend']}")
                return False
            cmd = [python, ROOT / "tests" / "generate_answers.py", *common, "--index", a.indice, "--local",
                   "--set", f"generation_temperature={GENERATION_TEMPERATURE}"]
            if a.ollama:
                cmd += ["--ollama", a.ollama]
        code, seconds = self.call("respostas", cmd, "respostas")
        ok = code == 0 and self.answers_ready()
        data = json.loads(self.answers.read_text(encoding="utf-8")) if self.answers.is_file() else {}
        protocol = data.get("protocolo") or {}
        detail = (f"{len(data.get('answers', []))} respostas, trechos iguais: "
                  f"{all(t.get('iguais') for t in protocol.get('trechos', {}).values())}, temperatura "
                  f"{protocol.get('temperatura')}" if ok else
                  f"código {code}: {protocol.get('interrompido') or 'ver logs/respostas.log'}")
        self.record("respostas", "concluída" if ok else "falhou", detail, seconds)
        return ok

    def free_generator(self) -> None:
        data = json.loads(self.answers.read_text(encoding="utf-8"))
        model = data.get("generation_model")
        ollama = (self.args.ollama or (data.get("protocolo") or {}).get("ollama_url")
                  or self.env["OLLAMA_BASE_URL"]).rstrip("/")
        freed = bool(model) and run_all.unload_model(ollama, model)
        self.record("liberar o gerador", "concluída",
                    f"{model} {'tirado da memória' if freed else 'já fora da memória'} em {ollama}")

    def export_done(self, judge: str, k: int) -> Path | None:
        for path in sorted((self.dir / judge).glob(f"*-r{k}.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cases_done") == data.get("n_cases"):
                return path
        return None

    def judge(self, judge: str, k: int) -> None:
        key = f"{judge} (julgamento {k})"
        if judge in self.args.pular:
            self.record(key, "pulada", "--pular")
            return
        if judge == "jeveval" and not run_all.typesafe_key_configured():
            self.record(key, "pulada", "sem TYPESAFE_API_KEY no ambiente ou no .env (ver tests/jeveval/README.md)")
            return
        done = self.export_done(judge, k)
        if done:
            self.record(key, "concluída", f"já existia: {done.name}")
            return
        python = run_all.PY[judge]
        if not python.exists():
            self.record(key, "erro", f"venv não existe: {run_all.SETUP_HINT[judge]}")
            return
        cmd = [python, ROOT / "tests" / judge / "run_and_export.py", "--answers", self.answers,
               "--repeticao", k, "--ordem-semente", k, "--out-dir", self.dir / judge]
        if judge in ("ragas", "deepeval"):
            cmd.append("--passos")
        if judge == "deepeval":
            cmd.append("--no-open")
        code, seconds = self.call(key, cmd, f"{judge}-r{k}")
        done = self.export_done(judge, k)
        if code == 0 and done:
            data = json.loads(done.read_text(encoding="utf-8"))
            measured = [m for c in data["cases"] for m in c["metrics"] if m.get("score") is not None]
            total = sum(len(c["metrics"]) for c in data["cases"])
            self.record(key, "concluída", f"{len(measured)}/{total} notas medidas, {done.name}", seconds,
                        export=str(done))
        else:
            self.record(key, "falhou", f"código {code}, ver logs/{judge}-r{k}.log", seconds)

    def analyse(self) -> None:
        paths = analise.find_exports(self.dir)
        if not paths:
            self.record("análise", "pulada", "nenhum export de juiz")
            return
        report = analise.variance_report([analise.load_export(p) for p in paths])
        (self.dir / "analise.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        table = analise.markdown(report)
        (self.dir / "analise.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table, flush=True)
        self.record("análise", "concluída", f"{len(paths)} exports, analise.md")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run(args)
    print(f"Pergunta única: {args.golden}, sementes {args.sementes}, {args.julgamentos} julgamento(s) por juiz. "
          f"Pasta: {run.dir}", flush=True)
    t0 = time.perf_counter()
    if not run.generate():
        return 1
    run.free_generator()
    for k in range(1, args.julgamentos + 1):
        for judge in JUDGES:
            run.judge(judge, k)
    run.analyse()
    failed = [s for s in run.steps if s["estado"] in ("falhou", "erro")]
    print(f"\nPronto em {(time.perf_counter() - t0) / 60:.1f} min; {len(failed)} passo(s) com falha. {run.dir}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
