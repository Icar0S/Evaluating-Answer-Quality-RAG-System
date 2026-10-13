"""Compara versões do índice do RAG (ex.: pdf-v1 x md-v1): mesmas perguntas, mesmo gerador, mesmo juiz.

Uso, da raiz do repositório:

    backend\\.venv\\Scripts\\python.exe tests\\compare_indices.py                 # pdf-v1 x md-v1, RAGAS
    backend\\.venv\\Scripts\\python.exe tests\\compare_indices.py --limite 2      # 2 goldens de cada conjunto (fumaça)
    backend\\.venv\\Scripts\\python.exe tests\\compare_indices.py --deepeval --abrir

Passos:

 1. Garante que cada versão do índice existe (ingere se estiver vazia).
 2. Diagnóstico de recuperação, só com embeddings (tests/retrieval_diagnostic.py):
    o documento certo vem entre os trechos? Pergunta em português e em inglês.
 3. Para cada versão (e para as de --en, também com as perguntas em inglês):
    gera as respostas uma vez, para os goldens dos DOIS acervos
    (tests/deepeval/goldens/dataset.json e dataset_md.json). Perguntas do
    acervo alheio medem se o RAG se abstém quando o assunto não está nele.
 4. RAGAS: controles do juiz uma vez e a avaliação de cada conjunto de
    respostas. DeepEval só com --deepeval (pode levar horas).
 5. Relatório consolidado em tests/reports/out/<data>/, com as rodadas lado a
    lado, o recorte por origem da pergunta e o diagnóstico de recuperação.

Só biblioteca padrão: as utilidades de execução vêm de tests/run_all.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all as ra  # noqa: E402

ROOT = ra.ROOT
GOLDENS = {
    "pdf": ROOT / "tests" / "deepeval" / "goldens" / "dataset.json",
    "md": ROOT / "tests" / "deepeval" / "goldens" / "dataset_md.json",
}


def goldens_with_english() -> list[str]:
    return [tag for tag, path in GOLDENS.items() if any(g.get("input_en") for g in json.loads(path.read_text(encoding="utf-8")))]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Compara versões do índice do RAG com as mesmas perguntas.")
    parser.add_argument("--indices", nargs="+", default=["pdf-v1", "md-v1"], help="versões a comparar")
    parser.add_argument(
        "--en", nargs="*", default=["md-v1"],
        help="versões que também recebem as perguntas em inglês (padrão: md-v1; vazio = nenhuma)",
    )
    parser.add_argument("--deepeval", action="store_true", help="avalia também com o DeepEval (lento)")
    parser.add_argument("--limite", type=int, default=None, help="só os N primeiros goldens de cada conjunto")
    parser.add_argument("--abrir", action="store_true", help="abre o PDF do relatório ao terminar")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).astimezone()
    t0 = time.monotonic()
    out_dir = ra.OUT_ROOT / started.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # O Runner do run_all dá o registro de etapas, os logs e a avaliação com --answers.
    # limite=None de propósito: o --limite já corta cada conjunto de goldens na
    # geração das respostas; repassado à avaliação, cortaria o arquivo inteiro
    # (dois conjuntos) e descartaria as perguntas do segundo acervo.
    runner = ra.Runner(
        argparse.Namespace(pular=[], rapido=False, limite=None, respostas_separadas=False,
                           qualidade_pytest=False),
        out_dir,
    )
    print(f"=== Comparação de índices do RAG — {started.strftime('%d/%m/%Y %H:%M')} ===")
    print(f"Índices: {', '.join(args.indices)} · saída: {out_dir}")

    if ra.http_get(f"{ra.ollama_url()}/api/tags") is None:
        print(f"[ERRO] Ollama fora do ar em {ra.ollama_url()}.", file=sys.stderr)
        return 1
    for venv in ("backend", "ragas", "reports"):
        if not ra.PY[venv].exists():
            print(f"[ERRO] venv '{venv}' não encontrado ({ra.SETUP_HINT[venv]}).", file=sys.stderr)
            return 1

    # 1. Índices -----------------------------------------------------------
    for index in args.indices:
        name = f"Índice {index}"
        runner.announce(name)
        log = out_dir / "logs" / f"indice_{index}.log"
        code, secs = ra.run_logged(
            [ra.PY["backend"], ROOT / "scripts" / "ingest_documents.py", "--index", index, "--if-empty"], log,
            timeout=3600, echo=True,
        )
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = next((l.strip() for l in reversed(lines) if "chunks" in l.lower() or "Chunks" in l), "")
        if code != 0:
            runner.record(ra.Step(f"indice_{index}", name, "erro", "falhou ao preparar o índice (ver o log)", None,
                                  secs, runner._rel(log)))
            return 1
        runner.record(ra.Step(f"indice_{index}", name, "concluída", detail, None, secs, runner._rel(log)))

    # 2. Diagnóstico de recuperação ------------------------------------------
    name = "Diagnóstico de recuperação (documento certo nos trechos)"
    runner.announce(name)
    diag_path = out_dir / "diagnostico_recuperacao.json"
    log = out_dir / "logs" / "diagnostico.log"
    code, secs = ra.run_logged(
        [ra.PY["backend"], ROOT / "tests" / "retrieval_diagnostic.py", "--out", diag_path, "--indices",
         *args.indices, "--goldens", GOLDENS["md"], "--goldens", GOLDENS["pdf"]],
        log, timeout=1800, echo=True,
    )
    diagnostic = json.loads(diag_path.read_text(encoding="utf-8")) if code == 0 and diag_path.is_file() else None
    summary = " | ".join(
        l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if ": PT " in l
    )
    runner.record(ra.Step("diagnostico", name, "concluída" if diagnostic else "erro", summary or "ver o log", None,
                          secs, runner._rel(log)))

    # 3. Respostas -----------------------------------------------------------
    english = goldens_with_english()
    answer_sets: list[tuple[str, str, Path]] = []
    for index in args.indices:
        for lang in ["pt"] + (["en"] if index in (args.en or []) else []):
            name = f"Respostas: {index} ({lang.upper()})"
            runner.announce(name)
            target = out_dir / f"respostas_{index}_{lang}.json"
            tags = list(GOLDENS) if lang == "pt" else english
            cmd = [ra.PY["backend"], ROOT / "tests" / "generate_answers.py", "--out", target, "--index", index,
                   "--lang", lang]
            for tag in tags:
                cmd += ["--goldens", f"{tag}={GOLDENS[tag].relative_to(ROOT).as_posix()}"]
            if args.limite:
                cmd += ["--limit", args.limite]
            log = out_dir / "logs" / f"respostas_{index}_{lang}.log"
            code, secs = ra.run_logged(cmd, log, timeout=3 * 3600, echo=True)
            if code != 0 or not target.is_file():
                runner.record(ra.Step(f"respostas_{index}_{lang}", name, "erro", "não gerou as respostas (ver o log)",
                                      None, secs, runner._rel(log)))
                continue
            data = json.loads(target.read_text(encoding="utf-8"))
            failures = sum(1 for a in data["answers"] if a.get("error"))
            runner.record(ra.Step(f"respostas_{index}_{lang}", name, "concluída",
                                  f"{len(data['answers'])} resposta(s) com {data.get('generation_model')} "
                                  f"({failures} falha(s)); goldens: {', '.join(tags)}", None, secs, runner._rel(log)))
            answer_sets.append((index, lang, target))

    # 4. Avaliação -----------------------------------------------------------
    runner.pytest_step("ragas_controles", "RAGAS: controles do juiz", "ragas",
                       "tests/ragas/test_judge_controls.py", timeout=1800)
    for index, lang, answers in answer_sets:
        runner.answers_file = answers
        runner.evaluation(f"ragas_{index}_{lang}", f"RAGAS: {index} ({lang.upper()})", "ragas",
                          ROOT / "tests" / "ragas" / "run_and_export.py", ROOT / "tests" / "ragas" / "results", [])
        if args.deepeval:
            runner.evaluation(f"deepeval_{index}_{lang}", f"DeepEval: {index} ({lang.upper()})", "deepeval",
                              ROOT / "tests" / "deepeval" / "run_and_export.py",
                              ROOT / "tests" / "deepeval" / "results", ["--no-open"])

    # 5. Relatório -----------------------------------------------------------
    runner.announce("Relatório consolidado")
    execution = {
        "tool": "tests/compare_indices.py",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "duration_seconds": round(time.monotonic() - t0, 1),
        "options": vars(args),
        "shared_answers": None,
        "reused_results": [],
        "retrieval_diagnostic": diagnostic,
        "steps": [vars(s) for s in runner.steps],
    }
    exec_path = out_dir / "execucao.json"
    exec_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not runner.new_results:
        print("    [ERRO ] nenhuma avaliação concluída; relatório não gerado.", flush=True)
        return 1
    cmd = [ra.PY["reports"], ROOT / "tests" / "reports" / "build_report.py", "--out", out_dir, "--execution", exec_path]
    for result in runner.new_results.values():
        cmd += ["--input", result]
    code, _ = ra.run_logged(cmd, out_dir / "logs" / "relatorio.log", timeout=900, echo=True)

    print(f"\n=== Resumo ({(time.monotonic() - t0) / 60:.1f} min) ===")
    for step in runner.steps:
        print(f"  {ra.STATUS_TAG.get(step.status, step.status)} {step.name}")
    pdf = out_dir / "relatorio.pdf"
    if code == 0 and pdf.is_file():
        print(f"\nRelatório: {pdf}")
        if args.abrir and hasattr(os, "startfile"):
            os.startfile(pdf)  # noqa: S606 - abre no visualizador padrão do Windows
    broken = [s for s in runner.steps if s.status in ("falhou", "erro")]
    return 1 if broken or code != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
