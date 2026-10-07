"""Roda a avaliação RAGAS sobre os goldens e salva os números, não só pass/fail.

Saídas em tests/ragas/results/ (não versionadas, mesma regra de
tests/deepeval/results/):

- <timestamp>.json: mesmo formato do export do DeepEval (generated_at,
  generation_model, judge_model, threshold, metrics_avg, cases[] com
  name/input/actual_output/expected_output/retrieval_context/metrics[]), para
  as duas avaliações poderem ser cruzadas por caso. Acrescenta o que é próprio
  daqui: versão do ragas, alterações de prompt, chamadas/falhas do juiz.
- <timestamp>.csv: uma linha por caso x métrica, para planilha ou pandas.

O JSON é regravado a cada caso concluído: uma rodada interrompida deixa o que
já mediu (cases_done < n_cases).

Rodar da raiz do repo:
    tests\\ragas\\.venv\\Scripts\\python.exe tests\\ragas\\run_and_export.py
    tests\\ragas\\.venv\\Scripts\\python.exe tests\\ragas\\run_and_export.py --limit 2
    tests\\ragas\\.venv\\Scripts\\python.exe tests\\ragas\\run_and_export.py --only ragas-definicao
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Rodado como script (não via pytest), o pythonpath do pytest.ini não vale:
# backend/ (para `app`) e esta pasta (para `_ragas_shared`) entram à mão.
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parents[1] / "backend")]

import _ragas_shared as shared  # noqa: E402
import ragas  # noqa: E402

from app import providers  # noqa: E402
from app.config import get_settings  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _metrics_summary(cases: list[dict], threshold: float) -> tuple[dict, dict, dict]:
    avg, measured, passed = {}, {}, {}
    for name in shared.METRIC_NAMES:
        scores = [m["score"] for c in cases for m in c["metrics"] if m["name"] == name and m["score"] is not None]
        avg[name] = round(statistics.fmean(scores), 4) if scores else None
        measured[name] = f"{len(scores)}/{len(cases)}"
        passed[name] = f"{sum(s >= threshold for s in scores)}/{len(cases)}"
    return avg, measured, passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliação RAGAS dos goldens, com export JSON/CSV.")
    parser.add_argument("--limit", type=int, default=None, help="avalia só os N primeiros goldens")
    parser.add_argument("--only", action="append", default=[], help="nome de um golden (repetível)")
    args = parser.parse_args()

    reason = shared.stack_status()
    if reason:
        _log(f"[ERRO] Stack não está pronta: {reason}")
        return 1

    settings, rs = get_settings(), shared.get_ragas_settings()
    goldens = shared.load_goldens()
    if args.only:
        goldens = [g for g in goldens if g.name in set(args.only)]
    if args.limit:
        goldens = goldens[: args.limit]
    if not goldens:
        _log("[ERRO] Nenhum golden selecionado.")
        return 1

    judge = shared.build_judge()
    metrics = shared.build_metrics(judge, shared.build_embeddings())
    active = providers.get_active_provider()

    shared.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = shared.RESULTS_DIR / f"{stamp}.json"
    csv_path = shared.RESULTS_DIR / f"{stamp}.csv"

    output: dict = {
        "framework": "ragas",
        "ragas_version": ragas.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": active.name,
        "generation_model": active.generation_model,
        "embedding_model": settings.embedding_model,
        "judge_model": rs.ragas_judge_model,
        "judge_num_ctx": rs.ragas_judge_num_ctx,
        "threshold": rs.ragas_threshold,
        "prompt_overrides": shared.PROMPT_OVERRIDES,
        "n_cases": len(goldens),
        "cases_done": 0,
        "cases": [],
    }

    _log(
        f"RAGAS {ragas.__version__} | geração: {active.generation_model} ({active.name}) | "
        f"juiz: {rs.ragas_judge_model} | {len(goldens)} goldens x {len(metrics)} métricas"
    )
    started = time.perf_counter()

    for i, golden in enumerate(goldens, start=1):
        _log(f"\n[{i}/{len(goldens)}] {golden.name}")
        t0 = time.perf_counter()
        try:
            answer, contexts = shared.run_pipeline(golden.input)
            gen_error = None
        except Exception as exc:  # noqa: BLE001 - um golden com SUT quebrado não derruba a rodada
            answer, contexts, gen_error = "", [], f"{type(exc).__name__}: {exc}"
        sut_seconds = round(time.perf_counter() - t0, 1)

        case_metrics = []
        for name, metric in metrics.items():
            t1 = time.perf_counter()
            if gen_error:
                outcome = shared.MetricOutcome(name=name, score=None, error=f"geração falhou: {gen_error}")
            else:
                case = shared.CaseInputs(golden.input, answer, contexts, golden.expected_output)
                outcome = shared.score_metric(name, metric, case)
            success = None if outcome.score is None else outcome.score >= rs.ragas_threshold
            case_metrics.append(
                {
                    "name": name,
                    "score": None if outcome.score is None else round(outcome.score, 4),
                    "success": success,
                    "reason": outcome.reason,
                    "error": outcome.error,
                    "seconds": round(time.perf_counter() - t1, 1),
                }
            )
            shown = f"{outcome.score:.3f}" if outcome.measured else f"NÃO MEDIDA ({outcome.error[:120]})"
            _log(f"    {name:18s} {shown}")

        output["cases"].append(
            {
                "name": golden.name,
                "input": golden.input,
                "actual_output": answer,
                "expected_output": golden.expected_output,
                "retrieval_context": contexts,
                "success": all(m["success"] for m in case_metrics),
                "sut_seconds": sut_seconds,
                "metrics": case_metrics,
            }
        )
        avg, measured, passed = _metrics_summary(output["cases"], rs.ragas_threshold)
        output.update(
            cases_done=i,
            metrics_avg=avg,
            metrics_measured=measured,
            metrics_passed=passed,
            judge_calls=judge.calls,
            judge_failures=judge.failures,
            wall_seconds=round(time.perf_counter() - started, 1),
        )
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["golden", "metrica", "score", "passou", "erro", "segundos"])
        for case in output["cases"]:
            for m in case["metrics"]:
                writer.writerow([case["name"], m["name"], m["score"], m["success"], m["error"] or "", m["seconds"]])

    _log(f"\n{'métrica':18s} {'média':>7s} {'medidas':>8s} {'>= limiar':>10s}")
    for name in shared.METRIC_NAMES:
        mean = output["metrics_avg"][name]
        _log(
            f"{name:18s} {'-' if mean is None else f'{mean:.3f}':>7s} "
            f"{output['metrics_measured'][name]:>8s} {output['metrics_passed'][name]:>10s}"
        )
    _log(
        f"\nchamadas ao juiz: {judge.calls} (falhas: {judge.failures}) | "
        f"tempo total: {output['wall_seconds'] / 60:.1f} min"
    )
    _log(f"JSON: {json_path}\nCSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
