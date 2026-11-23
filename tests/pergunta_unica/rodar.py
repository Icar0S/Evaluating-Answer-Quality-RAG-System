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
   conteúdo para fora da máquina (tests/jeveval/README.md). Nota que o juiz não deu
   (erro, timeout) é um BURACO: o passo tenta de novo só os casos com buraco
   (`--tentativas`, padrão 2), sem trocar nota já medida. Se sobrar buraco, o passo
   sai "incompleta", e a rodada não termina com código 0.
4. **A análise** (analise.py): por juiz e métrica, a média, a faixa e quanto da
   variação é do gerador (entre as respostas) e quanto é do juiz (entre os
   julgamentos da mesma resposta).
5. **Os pares** (pares.py, plano §6): o ArenaGEval do DeepEval
   (tests/deepeval/pairwise.py) e a Choice do Jev (tests/jeveval/pairwise.py)
   julgam os 10 pares nas duas ordens (`--uma-ordem`: só AB, a rodada da
   apresentação); a preferência implícita sai das notas pontuais dos três. A
   análise mede o ranking, o viés de posição, o de verbosidade e a concordância.
   `--sem-pares` pula.

Tudo fica em tests/pergunta_unica/out/<data>/: respostas.json, um export por
juiz e julgamento (ragas/, deepeval/, jeveval/), os pares (pares/), analise.json
e analise.md, pares-analise.json e pares-analise.md, o log de cada passo (logs/)
e o execucao.json. Com `--retomar <pasta>`, o que já está completo não roda de
novo; o que tem buraco é completado; um export interrompido no meio vai para
descartados/ e o julgamento recomeça.

A máquina: a rodada de análise leva horas. Na bateria, o script não começa
(`--na-bateria` para insistir); enquanto roda, segura a suspensão por inatividade;
e, se a máquina dormir mesmo assim, o passo registra quando e por quanto tempo
(energia.py). Antes do DeepEval, o modelo de embeddings sai da GPU: em 23/11/2026
ele dividia os 8 GB com o juiz, e o Ollama ficou sem memória.

Só biblioteca padrão: roda com qualquer Python 3.10+.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
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
import energia  # noqa: E402
import mutacao  # noqa: E402
import pares  # noqa: E402


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
# Os operadores de mutantes.py (D5). A lista fica aqui para o argparse não importar o backend.
# N0 é o controle: o mutante que não muda nada.
MUTANTS = ("N0", "C1", "C2", "P1", "R1")
# O embedding da relevância da resposta do RAGAS (settings.embedding_model): fica 5 min na GPU depois do RAGAS.
EMBEDDING_MODEL = "nomic-embed-text"
# Entre uma tentativa e a próxima: dá tempo de a rede ou o Ollama voltarem.
RETRY_PAUSE_SECONDS = 30
EXIT_ON_BATTERY = 4
FAILED_STATES = ("falhou", "erro", "incompleta")


def log_name(label: str) -> str:
    """`mutante C1: ragas-r1` -> `mutante-C1-ragas-r1`: só letras, dígitos, `_` e `-` (o `:` não vale no Windows)."""
    return re.sub(r"[^\w-]+", "-", label).strip("-")


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
    parser.add_argument("--sem-pares", action="store_true", help="não roda a comparação em pares")
    parser.add_argument("--uma-ordem", action="store_true", help="os pares só na ordem AB (10 julgamentos, não 20)")
    parser.add_argument("--mutantes", nargs="*", default=[], choices=MUTANTS,
                        help="os mutantes a rodar (mutantes.py; só in-process): N0 (o controle) C1 C2 P1 R1")
    parser.add_argument("--mutantes-julgamentos", type=int, default=1, help="julgamentos por juiz em cada mutante")
    parser.add_argument("--ingles", action="store_true",
                        help="também as 5 respostas com a pergunta em inglês (o input_en; o item GER-3 do checklist)")
    parser.add_argument("--sem-relatorio", action="store_true", help="não gera o relatório no fim")
    parser.add_argument("--tentativas", type=int, default=2,
                        help="vezes que um passo tenta até não ter buraco; da 2.ª em diante, só o que falta (padrão 2)")
    parser.add_argument("--na-bateria", action="store_true", help="roda mesmo fora da tomada")
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
        self.machine: dict = {}
        self.watch = energia.SuspensionWatch()
        self.env = dict(os.environ, PYTHONUTF8="1")
        # localhost custa ~2 s por chamada nesta máquina (o Ollama só escuta IPv4); o juiz faz dezenas de chamadas.
        if "OLLAMA_BASE_URL" not in self.env:
            self.env["OLLAMA_BASE_URL"] = run_all.ollama_url().replace("://localhost", "://127.0.0.1")

    # ---------------------------------------------------------------- execução

    def record(self, key: str, status: str, detail: str = "", seconds: float | None = None, **extra) -> None:
        asleep = self.watch.take()
        if asleep:
            extra["suspensoes"] = asleep
        self.steps.append({"passo": key, "estado": status, "detalhe": detail, "segundos": seconds, **extra})
        tag = run_all.STATUS_TAG.get(status, f"[{status}]")
        print(f"{tag} {key}: {detail}" + (f" ({seconds / 60:.1f} min)" if seconds else ""), flush=True)
        for gap in asleep:
            print(f"        A MÁQUINA SUSPENDEU neste passo: {gap['de']} a {gap['ate']} "
                  f"({gap['segundos'] / 60:.0f} min). O tempo do passo não vale, e notas podem ter falhado.", flush=True)
        self.save()

    def save(self) -> None:
        (self.dir / "execucao.json").write_text(json.dumps({
            "plano": "plano-escopo-final.md",
            "pasta": str(self.dir),
            "argumentos": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
            "ollama_dos_juizes": self.env.get("OLLAMA_BASE_URL"),
            "maquina": self.machine,
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

    @staticmethod
    def answers_ready(path: Path) -> bool:
        if not path.is_file():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data.get("protocolo") or {}
        return (not protocol.get("interrompido") and len(data.get("answers", [])) == data.get("n_cases")
                and all(a.get("error") is None for a in data["answers"]))

    def generate(self, out: Path | None = None, overrides: dict[str, str] | None = None, use_index: bool = True,
                 label: str = "respostas", lang: str = "pt") -> bool:
        """As respostas do protocolo. `overrides` e `use_index`: as de um mutante (só in-process)."""
        out = out or self.answers
        if self.answers_ready(out):
            self.record(label, "concluída", f"já existiam em {out.name}")
            return True
        out.parent.mkdir(parents=True, exist_ok=True)
        a = self.args
        common = ["--out", out, "--goldens", a.goldens, "--only", a.golden, "--sementes", a.sementes,
                  "--estado-limpo", "--lang", lang]
        if a.api:
            if not a.ollama:
                self.record(label, "erro", "--api precisa de --ollama (o Ollama do backend, para o estado limpo)")
                return False
            cmd = [sys.executable, ROOT / "tests" / "generate_answers.py", *common, "--api", a.api, "--ollama", a.ollama]
        else:
            python = run_all.PY["backend"]
            if not python.exists():
                self.record(label, "erro", f"venv do backend não existe: {run_all.SETUP_HINT['backend']}")
                return False
            cmd = [python, ROOT / "tests" / "generate_answers.py", *common, "--local",
                   "--set", f"generation_temperature={GENERATION_TEMPERATURE}"]
            if use_index:
                cmd += ["--index", a.indice]
            for key, value in (overrides or {}).items():
                cmd += ["--set", f"{key}={value}"]
            if a.ollama:
                cmd += ["--ollama", a.ollama]
        log = log_name(label)
        code, seconds = self.call(label, cmd, log)
        ok = code == 0 and self.answers_ready(out)
        data = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        protocol = data.get("protocolo") or {}
        detail = (f"{len(data.get('answers', []))} respostas, trechos iguais: "
                  f"{all(t.get('iguais') for t in protocol.get('trechos', {}).values())}, temperatura "
                  f"{protocol.get('temperatura')}" if ok else
                  f"código {code}: {protocol.get('interrompido') or f'ver logs/{log}.log'}")
        self.record(label, "concluída" if ok else "falhou", detail, seconds)
        return ok

    def free_generator(self, answers: Path | None = None) -> None:
        data = json.loads((answers or self.answers).read_text(encoding="utf-8"))
        model = data.get("generation_model")
        ollama = (self.args.ollama or (data.get("protocolo") or {}).get("ollama_url")
                  or self.env["OLLAMA_BASE_URL"]).rstrip("/")
        freed = bool(model) and run_all.unload_model(ollama, model)
        self.record("liberar o gerador", "concluída",
                    f"{model} {'tirado da memória' if freed else 'já fora da memória'} em {ollama}")

    def free_embedding(self) -> None:
        """Antes do DeepEval: o embedding do RAGAS ainda está na GPU e, com o juiz, passa dos 8 GB."""
        if run_all.unload_model(self.env["OLLAMA_BASE_URL"].rstrip("/"), EMBEDDING_MODEL):
            print(f"    ({EMBEDDING_MODEL} tirado da memória antes do DeepEval)", flush=True)

    def export_done(self, judge: str, k: int, base: Path | None = None) -> Path | None:
        """O export do julgamento k com todos os casos. Pode ter buracos: quem chama olha `analise.unmeasured`."""
        for path in sorted(((base or self.dir) / judge).glob(f"*-r{k}.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cases_done") == data.get("n_cases"):
                return path
        return None

    def set_aside_partial(self, judge: str, k: int, base: Path) -> None:
        """Um export interrompido no meio sai da pasta que a análise lê (senão o julgamento entraria duas vezes)."""
        keep = self.export_done(judge, k, base)
        for path in sorted((base / judge).glob(f"*-r{k}.json")):
            if path != keep:
                aside = base / judge / "descartados"
                aside.mkdir(parents=True, exist_ok=True)
                for sibling in sorted(path.parent.glob(f"{path.stem}.*")):
                    shutil.move(str(sibling), str(aside / sibling.name))

    def judge_once(self, judge: str, k: int, attempt: int, base: Path, answers: Path, key: str,
                   log: str) -> tuple[int, float]:
        """Uma tentativa: o julgamento inteiro, se não há export completo; senão, só os casos com buraco."""
        self.set_aside_partial(judge, k, base)
        done = self.export_done(judge, k, base)
        cmd = [run_all.PY[judge], ROOT / "tests" / judge / "run_and_export.py", "--answers", answers,
               "--repeticao", k, "--ordem-semente", k]
        if judge in ("ragas", "deepeval"):
            cmd.append("--passos")
        if judge == "deepeval":
            cmd.append("--no-open")
            self.free_embedding()
        if done is None:
            suffix = "" if attempt == 1 else f"-tentativa{attempt}"
            return self.call(key, [*cmd, "--out-dir", base / judge], log + suffix)

        export = json.loads(done.read_text(encoding="utf-8"))
        number = 2 + len(export.get("tentativas", []))
        cases = sorted({name for name, _ in analise.unmeasured(export)})
        out = base / judge / "tentativas"
        before = set(out.glob(f"*-r{k}.json"))
        code, seconds = self.call(f"{key}: {number}.ª tentativa, {len(cases)} caso(s) com buraco",
                                  [*cmd, *(x for name in cases for x in ("--only", name)), "--out-dir", out],
                                  f"{log}-tentativa{number}")
        new = sorted(set(out.glob(f"*-r{k}.json")) - before)
        if new:
            retry = json.loads(new[-1].read_text(encoding="utf-8"))
            analise.merge_retry(export, retry, number, f"tentativas/{new[-1].name}")
            done.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
        return code, seconds

    def judge(self, judge: str, k: int, base: Path | None = None, answers: Path | None = None,
              label: str = "") -> None:
        """Um julgamento de um juiz. `base` e `answers`: os de um mutante (padrão: a rodada)."""
        base, answers = base or self.dir, answers or self.answers
        key = f"{label}{judge} (julgamento {k})"
        log = log_name(f"{label}{judge}-r{k}")
        if judge in self.args.pular:
            self.record(key, "pulada", "--pular")
            return
        if judge == "jeveval" and not run_all.typesafe_key_configured():
            self.record(key, "pulada", "sem TYPESAFE_API_KEY no ambiente ou no .env (ver tests/jeveval/README.md)")
            return

        def state() -> tuple[Path | None, list[tuple[str, str]], str]:
            path = self.export_done(judge, k, base)
            if path is None:
                return None, [], ""
            export = json.loads(path.read_text(encoding="utf-8"))
            gaps, total = analise.unmeasured(export), sum(len(c["metrics"]) for c in export["cases"])
            return path, gaps, f"{total - len(gaps)}/{total} notas (ou N/A), {path.name}"

        path, gaps, detail = state()
        if path and not gaps:
            self.record(key, "concluída", f"já existia: {detail}")
            return
        if not run_all.PY[judge].exists():
            self.record(key, "erro", f"venv não existe: {run_all.SETUP_HINT[judge]}")
            return
        code, total_seconds = 0, 0.0
        for attempt in range(1, max(1, self.args.tentativas) + 1):
            if attempt > 1:
                time.sleep(RETRY_PAUSE_SECONDS)
            code, seconds = self.judge_once(judge, k, attempt, base, answers, key, log)
            total_seconds += seconds
            path, gaps, detail = state()
            if path and not gaps:
                break
        if path is None:
            self.record(key, "falhou", f"código {code}, ver logs/{log}*.log", total_seconds)
        elif gaps:
            missing = ", ".join(f"#{name.split('#')[-1]} {metric}" for name, metric in gaps[:6])
            self.record(key, "incompleta", f"{detail}; sem nota: {missing}" + (" …" if len(gaps) > 6 else ""),
                        total_seconds, export=str(path), sem_nota=[list(g) for g in gaps])
        else:
            filled = sum(len(t["preenchidas"]) for t in
                         json.loads(path.read_text(encoding="utf-8")).get("tentativas", []))
            self.record(key, "concluída", detail + (f" ({filled} numa nova tentativa)" if filled else ""),
                        total_seconds, export=str(path))

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

    # -------------------------------------------------------------------- pares

    PAIR_JUDGES = {"deepeval": "deepeval-arena", "jeveval": "jeveval-par"}

    def pair_file(self, judge: str) -> Path:
        return self.dir / "pares" / f"{self.PAIR_JUDGES[judge]}.json"

    def pair_state(self, judge: str) -> tuple[int, int] | None:
        """(julgados, total) do arquivo de pares, ou None se não existe ou não é o plano pedido (uma ordem x duas)."""
        path = self.pair_file(judge)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("duas_ordens") != (not self.args.uma_ordem):
            return None
        return sum(1 for j in data.get("julgamentos", []) if j["vencedor"] is not None), data.get("n_julgamentos") or 0

    def pairs(self, judge: str) -> None:
        key = f"pares ({judge})"
        if judge in self.args.pular:
            self.record(key, "pulada", "--pular")
            return
        if judge == "jeveval" and not run_all.typesafe_key_configured():
            self.record(key, "pulada", "sem TYPESAFE_API_KEY no ambiente ou no .env (ver tests/jeveval/README.md)")
            return
        state = self.pair_state(judge)
        if state and state[0] == state[1]:
            self.record(key, "concluída", f"já existia: {self.pair_file(judge).name}")
            return
        # --completar: o que já foi julgado (com as mesmas respostas) fica; só o que falta vai ao juiz.
        cmd = [run_all.PY[judge], ROOT / "tests" / judge / "pairwise.py", "--answers", self.answers,
               "--out", self.pair_file(judge), "--completar"]
        if self.args.uma_ordem:
            cmd.append("--uma-ordem")
        code, total_seconds = 0, 0.0
        for attempt in range(1, max(1, self.args.tentativas) + 1):
            if attempt > 1:
                time.sleep(RETRY_PAUSE_SECONDS)
            if judge == "deepeval":
                self.free_embedding()
            code, seconds = self.call(key, cmd, f"pares-{judge}" + (f"-tentativa{attempt}" if attempt > 1 else ""))
            total_seconds += seconds
            state = self.pair_state(judge)
            if state and state[0] == state[1]:
                break
        if state is None:
            self.record(key, "falhou", f"código {code}, ver logs/pares-{judge}*.log", total_seconds)
        else:
            judged, total = state
            self.record(key, "concluída" if judged == total else "incompleta", f"{judged}/{total} julgados",
                        total_seconds)

    def analyse_pairs(self) -> None:
        judges = {}
        for judge, name in self.PAIR_JUDGES.items():
            path = self.pair_file(judge)
            if path.is_file():
                judges[name] = json.loads(path.read_text(encoding="utf-8"))["julgamentos"]
        _, answers = pares.answers_by_name(self.answers)
        names = sorted(answers)
        exports = [analise.load_export(p) for p in analise.find_exports(self.dir)]
        implicit = {}
        for fw in analise.FRAMEWORKS:
            means = pares.pointwise_means([e for e in exports if analise.framework_of(e) == fw])
            if means:
                implicit[fw] = pares.implicit_judgments(means, names)
        if not judges and not implicit:
            self.record("análise dos pares", "pulada", "nenhum julgamento de pares nem nota pontual")
            return
        report = pares.pairs_report(judges, answers, implicit)
        (self.dir / "pares-analise.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
        table = pares.markdown(report)
        (self.dir / "pares-analise.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table, flush=True)
        self.record("análise dos pares", "concluída",
                    f"{len(judges)} juiz(es) de pares e {len(implicit)} implícito(s), pares-analise.md")

    # ----------------------------------------------------------------- mutantes

    def mutant(self, op: str) -> None:
        """Um mutante (mutantes.py): prepara, gera as 5 respostas com as sobrescritas dele e julga."""
        base = self.dir / "mutantes" / op
        meta_file = base / "mutante.json"
        if self.args.api:
            self.record(f"mutante {op}", "pulada",
                        "os mutantes rodam in-process: o backend no ar não aceita sobrescritas por requisição")
            return
        if not meta_file.is_file():
            key = f"mutante {op} (preparar)"
            cmd = [run_all.PY["backend"], HERE / "mutantes.py", "preparar", "--op", op, "--dir", base]
            code, seconds = self.call(key, cmd, f"mutante-{op}-preparar")
            if code != 0 or not meta_file.is_file():
                self.record(key, "falhou", f"código {code}, ver logs/mutante-{op}-preparar.log", seconds)
                return
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.record(key, "concluída", f"{meta['nome']}: {', '.join(meta['artefatos'])}", seconds)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        answers = base / "respostas.json"
        if not self.generate(answers, meta["set"], meta["usa_indice_base"], label=f"mutante {op} (respostas)"):
            return
        self.free_generator(answers)
        for k in range(1, self.args.mutantes_julgamentos + 1):
            for judge in JUDGES:
                self.judge(judge, k, base=base, answers=answers, label=f"mutante {op}: ")

    def analyse_mutants(self, ops: list[str]) -> None:
        original = [analise.load_export(p) for p in analise.find_exports(self.dir)]
        mutants = {}
        for op in ops:
            base = self.dir / "mutantes" / op
            if (base / "respostas.json").is_file() and (base / "mutante.json").is_file():
                mutants[op] = {"meta": json.loads((base / "mutante.json").read_text(encoding="utf-8")),
                               "answers": json.loads((base / "respostas.json").read_text(encoding="utf-8")),
                               "exports": [analise.load_export(p) for p in analise.find_exports(base)]}
        if not mutants:
            self.record("análise dos mutantes", "pulada", "nenhum mutante com respostas")
            return
        report = mutacao.mutation_report(original, json.loads(self.answers.read_text(encoding="utf-8")), mutants)
        (self.dir / "mutantes-analise.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        table = mutacao.markdown(report)
        (self.dir / "mutantes-analise.md").write_text(table + "\n", encoding="utf-8")
        print("\n" + table, flush=True)
        self.record("análise dos mutantes", "concluída", f"{len(mutants)} mutante(s), mutantes-analise.md")

    # ---------------------------------------------------------------- relatório

    def report(self) -> None:
        """O relatório da rodada (tests/reports/pergunta_unica_report.py), com o checklist, em <pasta>/relatorio/."""
        python = run_all.PY["reports"]
        if not python.exists():
            self.record("relatório", "erro", f"venv do relatório não existe: {run_all.SETUP_HINT['reports']}")
            return
        cmd = [python, ROOT / "tests" / "reports" / "build_report.py", "--pergunta-unica", self.dir]
        code, seconds = self.call("relatório", cmd, "relatorio")
        out = self.dir / "relatorio"
        ok = code == 0 and (out / "relatorio.md").is_file()
        self.record("relatório", "concluída" if ok else "falhou",
                    f"{out / 'relatorio.pdf'}" if ok else f"código {code}, ver logs/relatorio.log", seconds)


def steps(run: Run) -> bool:
    """Os passos da rodada, na ordem. False só quando não há respostas: sem elas não há o que julgar."""
    args = run.args
    if not run.generate():
        return False
    if args.ingles:
        run.generate(run.dir / "ingles" / "respostas.json", label="respostas em inglês", lang="en")
    run.free_generator()
    for k in range(1, args.julgamentos + 1):
        for judge in JUDGES:
            run.judge(judge, k)
    run.analyse()
    if not args.sem_pares:
        for judge in Run.PAIR_JUDGES:
            run.pairs(judge)
        run.analyse_pairs()
    if args.mutantes:
        for op in args.mutantes:
            run.mutant(op)
        run.analyse_mutants(args.mutantes)
    if not args.sem_relatorio:
        run.report()
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    power = energia.power()
    if power["tomada"] is False and not args.na_bateria:
        print(f"[ERRO] A máquina está na bateria ({power['bateria']}%). A rodada leva horas, e na bateria a GPU cai de "
              "desempenho e a máquina pode suspender. Ligue na tomada, ou rode com --na-bateria.", file=sys.stderr)
        return EXIT_ON_BATTERY
    run = Run(args)
    print(f"Pergunta única: {args.golden}, sementes {args.sementes}, {args.julgamentos} julgamento(s) por juiz. "
          f"Pasta: {run.dir}", flush=True)
    t0 = time.perf_counter()
    with energia.keep_awake() as awake:
        run.machine = {"energia_no_inicio": power, "suspensao_por_inatividade_bloqueada": awake}
        run.watch.start()
        try:
            finished = steps(run)
        finally:
            run.watch.stop()
    failed = [step for step in run.steps if step["estado"] in FAILED_STATES]
    print(f"\nPronto em {(time.perf_counter() - t0) / 60:.1f} min; {len(failed)} passo(s) com falha ou buraco. "
          f"{run.dir}", flush=True)
    for step in failed:
        print(f"  {step['passo']}: {step['estado']} ({step['detalhe']})", flush=True)
    return 0 if finished and not failed else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
