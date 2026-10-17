"""Julga as respostas compartilhadas com o JevEval e salva os números, não só pass/fail.

Esta camada não chama o assistente: julga o arquivo de respostas que
tests/generate_answers.py gera UMA vez por execução do tests/run_all.py — o
mesmo texto que o DeepEval e o RAGAS julgam. É o que faz a comparação entre os
três frameworks no relatório consolidado ser de método, e não de rodada. Fora do
run_all, gere as respostas com o venv do backend e passe o arquivo aqui.

Saídas em tests/jeveval/results/ (não versionadas, mesma regra das outras camadas):

- <timestamp>.json: o formato do export do DeepEval e do RAGAS (cases[] com
  name/input/actual_output/expected_output/retrieval_context/metrics[]), com o
  que é próprio daqui: `applicable` (N/A), `confidence` do Jev, o `breakdown`
  por pergunta, a causa provável na correção, as perguntas usadas
  (`question_sets`) e a impressão digital delas (`instrument_sha`);
- <timestamp>.csv: uma linha por caso x métrica (separador `;`, abre no Excel).

O JSON é regravado a cada caso: uma rodada interrompida deixa o que já mediu.

Rodar da raiz do repo (venv próprio, ver README.md desta pasta):
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\run_and_export.py --answers respostas.json
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\run_and_export.py --answers respostas.json --limit 2
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\run_and_export.py --answers r.json --metrics faithfulness,correctness
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _jev_shared as shared  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_metrics(value: str | None) -> list[str]:
    if not value:
        return list(shared.METRIC_NAMES)
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in shared.METRIC_SPECS]
    if unknown:
        raise SystemExit(f"[ERRO] Métrica(s) desconhecida(s): {', '.join(unknown)}. Válidas: {', '.join(shared.METRIC_NAMES)}")
    return names


def _version(module: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:  # noqa: BLE001 - pacote ausente só deixa o campo vazio
        return None


def main(argv: list[str] | None = None, model=None) -> int:
    """`model` existe para os testes herméticos injetarem o Jev falso; sem ele, usa o Jev de verdade."""
    parser = argparse.ArgumentParser(description="Avaliação JevEval das respostas compartilhadas, com export JSON/CSV.")
    parser.add_argument(
        "--answers", type=Path, required=True,
        help="respostas já geradas (tests/generate_answers.py); esta camada não chama o assistente",
    )
    parser.add_argument("--limit", type=int, default=None, help="avalia só os N primeiros casos")
    parser.add_argument("--only", action="append", default=[], help="nome de um golden (repetível)")
    parser.add_argument("--metrics", default=None, help=f"subconjunto, separado por vírgula ({','.join(shared.METRIC_NAMES)})")
    parser.add_argument("--out-dir", type=Path, default=shared.RESULTS_DIR, help="pasta de saída")
    args = parser.parse_args(argv)

    if not args.answers.is_file():
        _log(f"[ERRO] Arquivo de respostas não encontrado: {args.answers}")
        return 1
    names = _parse_metrics(args.metrics)
    settings = shared.get_settings()
    if model is None:
        reason = shared.jev_status(settings)
        if reason:
            _log(f"[ERRO] Jev indisponível: {reason}")
            return 1
        model = shared.build_jev(settings)

    meta, cases = shared.load_answers(args.answers)
    if args.only:
        cases = [c for c in cases if c.name in set(args.only)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        _log("[ERRO] Nenhum caso selecionado.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"{stamp}.json"
    csv_path = args.out_dir / f"{stamp}.csv"

    output: dict = {
        "framework": "jeveval",
        "deepeval_version": _version("deepeval"),
        "typesafe_sdk_version": _version("typesafe-sdk"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": meta.get("provider"),
        "generation_model": meta.get("generation_model"),
        "answers_source": str(args.answers),
        "rag_index": meta.get("rag_index"),
        "lang": meta.get("lang", "pt"),
        "judge_model": model.get_model_name(),
        "threshold": settings.jeveval_threshold,
        "thresholds": {name: shared.threshold_for(name, settings) for name in names},
        "instrument_sha": shared.instrument_sha(names),
        "question_sets": shared.question_sets(names),
        "question_sources": {name: shared.METRIC_SPECS[name].source for name in names},
        "n_cases": len(cases),
        "cases_done": 0,
        "cases": [],
    }

    _log(
        f"JevEval (deepeval {output['deepeval_version']}) | juiz: {output['judge_model']} | "
        f"respostas: {args.answers.name} ({output['generation_model']}) | {len(cases)} casos x {len(names)} métricas"
    )
    _log(f"instrumento {output['instrument_sha']} | progresso salvo a cada caso em {json_path.name}")
    started = time.perf_counter()
    calls = failures = 0
    cost = 0.0

    for i, case in enumerate(cases, start=1):
        _log(f"\n[{i}/{len(cases)}] {case.name}")
        records = []
        for name in names:
            threshold = shared.threshold_for(name, settings)
            if case.error:
                record = shared.not_measured_record(name, threshold, f"geração falhou: {case.error}")
            else:
                outcome = shared.measure(shared.METRIC_SPECS[name], model, case.inputs(), threshold)
                record = shared.metric_record(outcome, threshold)
                calls += 1
                failures += outcome.error is not None
                cost += outcome.cost_usd or 0.0
            records.append(record)
            if record["score"] is not None:
                shown = f"{record['score']:.3f}  confiança {record['confidence'] if record['confidence'] is not None else '-'}"
            elif not record["applicable"]:
                shown = "N/A (nenhuma pergunta se aplica)"
            else:
                shown = f"NÃO MEDIDA ({(record['error'] or '')[:120]})"
            _log(f"    {name:18s} {shown}")

        output["cases"].append(
            {
                "name": case.name,
                "origin": case.origin,
                "input": case.input,
                "actual_output": case.actual_output,
                "expected_output": case.expected_output,
                "retrieval_context": case.retrieval_context,
                "success": shared.case_success(records),
                "sut_seconds": case.seconds,
                "metrics": records,
            }
        )
        output.update(
            cases_done=i,
            **shared.metrics_summary(output["cases"], names),
            jev_calls=calls,
            jev_failures=failures,
            cost_usd=round(cost, 6),
            wall_seconds=round(time.perf_counter() - started, 1),
        )
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["golden", "metrica", "score", "passou", "aplicavel", "confianca", "causa", "erro", "segundos"])
        for case in output["cases"]:
            for m in case["metrics"]:
                writer.writerow(
                    [case["name"], m["name"], m["score"], m["success"], m["applicable"], m["confidence"],
                     m.get("cause") or "", m["error"] or "", m["seconds"]]
                )

    _log(f"\n{'métrica':18s} {'média':>7s} {'medidas':>8s} {'>= limiar':>10s} {'N/A':>4s}")
    for name in names:
        mean = output["metrics_avg"][name]
        _log(
            f"{name:18s} {'-' if mean is None else f'{mean:.3f}':>7s} {output['metrics_measured'][name]:>8s} "
            f"{output['metrics_passed'][name]:>10s} {output['metrics_not_applicable'][name]:>4d}"
        )
    _log(
        f"\nchamadas ao Jev: {calls} (falhas: {failures}) | custo: US$ {cost:.4f} | "
        f"tempo total: {output['wall_seconds'] / 60:.1f} min"
    )
    _log(f"JSON: {json_path}\nCSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    # Saída redirecionada para arquivo no Windows usa cp1252: um acento derrubaria o processo.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
