"""Testes E2E ao vivo nos dois RAGs e o relatório que compara a qualidade deles.

É o que o scripts\\demo_e2e.bat roda depois de subir os containers (README da
raiz, "Demonstração E2E ao vivo"). Da raiz do repositório, com os containers
no ar:

    python tests\\demo_e2e.py --abrir
    python tests\\demo_e2e.py --goldens 3      # só 3 perguntas de cada acervo (ensaio)

Passos:

 1. Testes E2E com o navegador visível (frontend/tests,
    playwright.demo.config.ts). A interface roda uma vez. O chat, as perguntas
    de referência e o "quem responde não é quem julga" rodam em cada RAG: o dos
    PDFs (pdf-v1, porta 8010) e o dos Markdown (md-v1, porta 8011). As
    respostas às perguntas de referência, com os trechos que o chat devolveu,
    ficam em <saída>/e2e/.
 2. Com o que o navegador gravou, monta um arquivo de respostas por RAG, no
    formato de tests/generate_answers.py, e tira dele a posição do documento
    certo entre os trechos.
 3. RAGAS: os controles do juiz e a avaliação das respostas de cada RAG. O juiz
    é o do Ollama do container (RAG_JUDGE_OLLAMA_URL e RAG_JUDGE_MODEL, que o
    .bat define).
 4. Relatório consolidado em tests/reports/out/<data>/, com os dois RAGs lado a
    lado, o recorte por origem da pergunta e a recuperação do documento certo.

Só biblioteca padrão. As utilidades de execução vêm de tests/run_all.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all as ra  # noqa: E402

ROOT = ra.ROOT
E2E_DIR = ROOT / "frontend" / "tests"
# Os conjuntos de goldens, na ordem em que frontend/tests/e2e/goldens.spec.ts os pergunta.
GOLDEN_SETS = [
    ("pdf", ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"),
    ("md", ROOT / "tests" / "deepeval" / "goldens" / "dataset_md.json"),
]
# Os dois RAGs do docker-compose.yml (os mesmos de playwright.demo.config.ts).
RAGS = [
    {"index": "pdf-v1", "label": "PDFs", "url": os.environ.get("DEMO_PDF_API_URL", "http://localhost:8010")},
    {"index": "md-v1", "label": "Markdown", "url": os.environ.get("DEMO_MD_API_URL", "http://localhost:8011")},
]
MISSING = "sem resposta no E2E (o teste falhou ou foi interrompido antes de responder)"
DIAGNOSTIC_DESCRIPTION = (
    "Tirado dos trechos que o chat devolveu a cada pergunta nos testes E2E, ou seja, a mesma busca que "
    "alimentou a resposta. Só há perguntas em português."
)


# --------------------------------------------------------------- conversões


def load_goldens(per_set: int | None) -> list[tuple[str, dict]]:
    """(origem, golden) na ordem de goldens.spec.ts; per_set = só os N primeiros de cada conjunto."""
    out: list[tuple[str, dict]] = []
    for origin, path in GOLDEN_SETS:
        out += [(origin, g) for g in json.loads(path.read_text(encoding="utf-8"))[:per_set]]
    return out


def golden_name(golden: dict) -> str:
    return golden.get("name") or golden["input"][:40]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_answers(index: str, records: list[dict], goldens: list[tuple[str, dict]]) -> dict:
    """O que o navegador gravou, no formato de tests/generate_answers.py, na ordem dos goldens.

    Pergunta, referência e contexto vêm do golden; resposta e trechos, do chat.
    Um golden sem resposta (teste que falhou antes de gravar) entra com erro: o
    RAGAS o conta como não medido, e os dois RAGs ficam com os mesmos casos.
    """
    by_name = {r["name"]: r for r in records}  # teste repetido: vale a última resposta
    meta = next((r["metadata"] for r in reversed(records) if r.get("metadata")), {})
    answers = []
    for origin, golden in goldens:
        name = golden_name(golden)
        record = by_name.get(name)
        answers.append(
            {
                "name": name,
                "origin": origin,
                "input": golden["input"],
                "expected_output": golden.get("expected_output", ""),
                "context": golden.get("context") or [],
                "source_document": golden.get("source_document"),
                "actual_output": (record or {}).get("actual_output"),
                "retrieval_context": (record or {}).get("retrieval_context") or [],
                "sources": (record or {}).get("sources") or [],
                "source_documents": (record or {}).get("source_documents") or [],
                "error": record.get("error") if record else MISSING,
                "seconds": (record or {}).get("seconds"),
            }
        )
    finished = [r["finished_at"] for r in records if r.get("finished_at")]
    return {
        "generated_at": max(finished) if finished else datetime.now(timezone.utc).isoformat(),
        "collected_by": "testes E2E ao vivo (scripts/demo_e2e.bat): respostas e trechos devolvidos pelo chat",
        "provider": meta.get("provider"),
        "generation_model": meta.get("model") or os.environ.get("DEMO_GENERATION_MODEL"),
        "rag_index": index,
        "embedding_model": meta.get("embedding_model"),
        "lang": "pt",
        "top_k": meta.get("top_k"),
        "prompt_version": meta.get("prompt_version"),
        "goldens": [f"{origin}={path.relative_to(ROOT).as_posix()}" for origin, path in GOLDEN_SETS],
        "n_cases": len(answers),
        "answers": answers,
    }


def retrieval_diagnostic(answer_sets: dict[str, dict], goldens: list[tuple[str, dict]]) -> dict | None:
    """Posição do documento-fonte de cada golden entre os trechos que o chat devolveu (formato de
    tests/retrieval_diagnostic.py, que o relatório já sabe mostrar)."""
    rows = []
    for _origin, golden in goldens:
        source = golden.get("source_document")
        if not source:
            continue
        name = golden_name(golden)
        ranks = {}
        for index, data in answer_sets.items():
            answer = next((a for a in data["answers"] if a["name"] == name), {})
            documents = answer.get("source_documents") or []
            ranks[index] = {"pt": documents.index(source) + 1 if source in documents else None}
        rows.append(
            {"name": name, "conversion": golden.get("conversion"), "source_document": source, "ranks": ranks}
        )
    if not rows:
        return None
    top_k = next((d["top_k"] for d in answer_sets.values() if d.get("top_k")), 4)
    return {
        "top_k": top_k,
        "indices": list(answer_sets),
        "langs": ["pt"],
        "rows": rows,
        "description": DIAGNOSTIC_DESCRIPTION,
    }


def playwright_counts(results: dict) -> dict[str, dict]:
    """Testes por projeto do Playwright (reporter json): total, falhas, pulados e segundos."""
    counts: dict[str, dict] = {}

    def walk(suite: dict) -> None:
        for spec in suite.get("specs", []):
            for test in spec.get("tests", []):
                c = counts.setdefault(
                    test.get("projectName") or "?", {"total": 0, "failed": 0, "errors": 0, "skipped": 0, "ms": 0}
                )
                c["total"] += 1
                c["ms"] += sum(r.get("duration", 0) for r in test.get("results", []))
                if test.get("status") == "skipped":
                    c["skipped"] += 1
                elif test.get("status") == "unexpected":
                    c["failed"] += 1
        for child in suite.get("suites", []):
            walk(child)

    for suite in results.get("suites", []):
        walk(suite)
    return counts


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "_", text.lower()).strip("_")


def _health(url: str) -> dict | None:
    body = ra.http_get(f"{url}/health", timeout=15)
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError:
        return None


def unload_model(ollama_url: str, model: str) -> bool:
    """Tira o modelo da memória do Ollama (keep_alive 0), se estiver carregado. True se tirou."""
    body = ra.http_get(f"{ollama_url}/api/ps", timeout=5)
    try:
        loaded = {m["name"] for m in json.loads(body).get("models", [])} if body else set()
    except (json.JSONDecodeError, AttributeError):
        return False
    if not loaded & {model, f"{model}:latest"}:
        return False
    request = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60):  # noqa: S310 - Ollama local
            return True
    except OSError:
        return False


def _per_set(value: str | None) -> int | None:
    try:
        return int(value) if value and int(value) > 0 else None
    except ValueError:
        return None


# ------------------------------------------------------------------- passos


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Testes E2E ao vivo nos dois RAGs e o relatório que compara a qualidade deles."
    )
    parser.add_argument(
        "--goldens", type=int, default=_per_set(os.environ.get("DEMO_GOLDENS")),
        help="só as N primeiras perguntas de cada acervo (padrão: todas, ou DEMO_GOLDENS)",
    )
    parser.add_argument("--abrir", action="store_true", help="abre o PDF do relatório ao terminar")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).astimezone()
    t0 = time.monotonic()
    out_dir = ra.OUT_ROOT / started.strftime("%Y%m%dT%H%M%S")
    answers_dir = out_dir / "e2e"
    answers_dir.mkdir(parents=True, exist_ok=True)
    runner = ra.Runner(
        argparse.Namespace(pular=[], rapido=False, limite=None, respostas_separadas=False, qualidade_pytest=False),
        out_dir,
    )
    judge_url = os.environ.get("RAG_JUDGE_OLLAMA_URL", "http://localhost:11435")
    judge_model = os.environ.get("RAG_JUDGE_MODEL", "qwen3:8b")
    goldens = load_goldens(args.goldens)

    print(f"=== Demonstração E2E: RAG dos PDFs × RAG dos Markdown — {started.strftime('%d/%m/%Y %H:%M')} ===")
    print(f"Saída: {out_dir}")
    npx = shutil.which("npx")
    if not npx or not (E2E_DIR / "node_modules").is_dir():
        print("[ERRO] Playwright não instalado (cd frontend\\tests && npm install && npx playwright install chromium).",
              file=sys.stderr)
        return 1
    for rag in RAGS:
        health = _health(rag["url"])
        state = (
            f"{health.get('vector_store_documents', '?')} trechos no índice {health.get('rag_index', '?')}"
            if health else "FORA DO AR (os testes dele vão se auto-pular)"
        )
        print(f"  RAG dos {rag['label']} ({rag['index']}) em {rag['url']}: {state}")
    print(f"  Perguntas de referência: {len(goldens)} por RAG ({len(goldens) * len(RAGS)} no total). "
          f"Juiz: {judge_model} em {judge_url}.")

    # 1. Testes E2E ao vivo ----------------------------------------------------
    runner.announce("Testes E2E ao vivo (navegador visível)")
    results_json = answers_dir / "resultados.json"
    log = out_dir / "logs" / "e2e.log"
    code, secs = ra.run_logged(
        [npx, "playwright", "test", "-c", "playwright.demo.config.ts"], log, cwd=E2E_DIR,
        env={
            "DEMO_ANSWERS_DIR": str(answers_dir),
            "DEMO_RESULTS_JSON": str(results_json),
            "DEMO_GOLDENS": str(args.goldens or ""),
        },
        timeout=4 * 3600, echo=True,
    )
    if results_json.is_file():
        for project, tests in playwright_counts(json.loads(results_json.read_text(encoding="utf-8"))).items():
            seconds = round(tests.pop("ms") / 1000, 1)
            status, detail = ra.status_from(0 if code is not None else None, tests)
            runner.record(ra.Step(f"e2e_{_slug(project)}", f"E2E ao vivo: {project}", status, detail, tests,
                                  seconds, runner._rel(log)))
    else:
        runner.record(ra.Step("e2e", "E2E ao vivo (Playwright)", "erro",
                              f"o Playwright saiu com código {code} sem resultados (ver o log)", None, secs,
                              runner._rel(log)))

    # 2. Respostas coletadas pelo navegador ---------------------------------------
    answer_sets: dict[str, dict] = {}
    answer_files: dict[str, Path] = {}
    for rag in RAGS:
        index = rag["index"]
        name = f"Respostas do RAG {index} (coletadas no E2E)"
        records = read_jsonl(answers_dir / f"respostas_{index}.jsonl")
        if not records:
            runner.skip(f"respostas_{index}", name,
                        "nenhuma resposta gravada (RAG fora do ar, índice vazio ou testes pulados)")
            continue
        data = build_answers(index, records, goldens)
        target = out_dir / f"respostas_{index}_pt.json"
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        failures = sum(1 for a in data["answers"] if a["error"])
        runner.record(ra.Step(f"respostas_{index}", name, "concluída",
                              f"{len(data['answers'])} pergunta(s) a {data.get('generation_model')}, "
                              f"{failures} sem resposta"))
        answer_sets[index] = data
        answer_files[index] = target
    diagnostic = retrieval_diagnostic(answer_sets, goldens) if answer_sets else None

    # 3. RAGAS -------------------------------------------------------------------
    # O juiz e os embeddings da AnswerRelevancy são os do Ollama do container; o
    # gerador entra só para o aviso de self-grading do RAGAS.
    generator = next((d["generation_model"] for d in answer_sets.values() if d.get("generation_model")),
                     os.environ.get("DEMO_GENERATION_MODEL", "gemma3:4b"))
    os.environ.update(
        OLLAMA_BASE_URL=judge_url,
        RAGAS_JUDGE_MODEL=judge_model,
        GENERATION_MODEL=generator,
        EMBEDDING_MODEL=os.environ.get("DEMO_EMBEDDING_MODEL", "nomic-embed-text"),
        ACTIVE_PROVIDER="local",
        RAG_INDEX="",
    )
    if answer_sets:
        # As respostas já estão gravadas: o gerador não é mais usado. Medido no
        # ensaio de 13/10/2026: com gerador, juiz e embeddings carregados juntos,
        # a VM do Docker (7,6 GB) matou o processo do modelo duas vezes por falta
        # de memória, e o juiz devolveu erro 500 nessas chamadas.
        if unload_model(judge_url, generator):
            print(f"\n    {generator} descarregado do Ollama do juiz: daqui em diante só o juiz e os embeddings "
                  "usam a memória.", flush=True)
        print(f"\n    Julgar leva ~1 min por resposta com GPU: ~{len(goldens) * len(answer_sets)} min no total.",
              flush=True)
        runner.pytest_step("ragas_controles", "RAGAS: controles do juiz", "ragas",
                           "tests/ragas/test_judge_controls.py", timeout=1800)
        for index, path in answer_files.items():
            runner.answers_file = path
            runner.evaluation(f"ragas_{index}", f"RAGAS: RAG {index}", "ragas",
                              ROOT / "tests" / "ragas" / "run_and_export.py", ROOT / "tests" / "ragas" / "results", [])

    # 4. Relatório ---------------------------------------------------------------
    runner.announce("Relatório consolidado")
    execution = {
        "tool": "scripts/demo_e2e.bat",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "duration_seconds": round(time.monotonic() - t0, 1),
        "options": {"goldens_por_acervo": args.goldens, "juiz": judge_model, "ollama_do_juiz": judge_url},
        "shared_answers": None,
        "reused_results": [],
        "note": (
            "As respostas avaliadas são as que o chat deu nos testes E2E ao vivo, com os dois RAGs no Docker: "
            "mesmo gerador, mesmo juiz, mesmas perguntas e acervos diferentes."
        ),
        "retrieval_diagnostic": diagnostic,
        "steps": [vars(s) for s in runner.steps],
    }
    exec_path = out_dir / "execucao.json"
    exec_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pdf = None
    if not runner.new_results:
        print("    [ERRO ] nenhuma avaliação concluída; relatório não gerado.", flush=True)
    elif not ra.PY["reports"].exists():
        print(f"    [ERRO ] sem o venv do relatório ({ra.SETUP_HINT['reports']})", flush=True)
    else:
        cmd = [ra.PY["reports"], ROOT / "tests" / "reports" / "build_report.py", "--out", out_dir,
               "--execution", exec_path]
        for result in runner.new_results.values():
            cmd += ["--input", result]
        code, _ = ra.run_logged(cmd, out_dir / "logs" / "relatorio.log", timeout=900, echo=True)
        if code == 0 and (out_dir / "relatorio.pdf").is_file():
            pdf = out_dir / "relatorio.pdf"

    print(f"\n=== Resumo ({(time.monotonic() - t0) / 60:.1f} min) ===")
    for step in runner.steps:
        print(f"  {ra.STATUS_TAG.get(step.status, step.status)} {step.name}")
    print("\nVídeo e trace de cada teste: cd frontend\\tests && npm run demo:report")
    if pdf:
        print(f"Relatório comparando os dois RAGs: {pdf}")
        if args.abrir and hasattr(os, "startfile"):
            os.startfile(pdf)  # noqa: S606 - abre no visualizador padrão do Windows
    broken = [s for s in runner.steps if s.status in ("falhou", "erro")]
    return 1 if broken or pdf is None else 0


if __name__ == "__main__":
    sys.exit(main())
