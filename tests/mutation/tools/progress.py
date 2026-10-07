"""Quanto já rodou da campanha, a que ritmo, e quanto falta.

Lê só `results/runs.jsonl` — não toca no SUT nem na GPU, então pode rodar com a
campanha em andamento. A projeção usa o custo medido por mutante quando ele já
rodou. Para os que ainda não rodaram, interpola entre os
mutantes já medidos na curva (tokens de prompt -> segundos) e estima o tamanho
do prompt pela configuração de recuperação do mutante. Interpolação em vez de
uma curva fechada porque essa relação tem um joelho: até onde o modelo cabe na
VRAM o tempo é quase constante (manda a geração da resposta), e depois dele
cresce depressa (o prompt passa a ser processado em parte na CPU). A estimativa
melhora sozinha conforme a campanha avança.

    python -m tests.mutation.tools.progress
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.operators.apply import load_catalog  # noqa: E402
from tests.mutation.runner import jsonl, paths  # noqa: E402
from tests.mutation.runner.console import get_logger  # noqa: E402
from tests.mutation.runner.sut import load_study_config  # noqa: E402

logger = get_logger("mutation.progress")

BASELINE = "baseline"


def context_tokens(overrides: dict) -> float:
    """Proxy do tamanho do prompt: quantos tokens de contexto a configuração pede."""
    return float(overrides.get("top_k", 4)) * float(overrides.get("chunk_size_tokens", 1200))


def main() -> int:
    study = load_study_config()
    catalog = load_catalog()
    runs = jsonl.read(paths.RUNS_JSONL)
    if not runs:
        logger.error("results/runs.jsonl vazio.")
        return 1

    per_case = int(study.execution["campaign_repetitions"])
    cases = len({r["case_id"] for r in runs if r["mutant_id"] == BASELINE}) or 30
    target = cases * per_case

    done: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        done[run["mutant_id"]].append(run)

    def merged_overrides(operator) -> dict:
        overrides = dict(study.baseline_overrides)
        overrides.update(operator.action.get("overrides", {}))
        for key, spec in operator.action.get("overrides_relative", {}).items():
            overrides[key] = overrides[key] * spec["multiply"]
        return overrides

    # (tokens de prompt medidos, segundos medidos) de tudo que já rodou
    measured: dict[str, float] = {}
    samples: list[tuple[float, float]] = []

    def collect(runs_of: list[dict], key: str | None) -> None:
        usable = [r for r in runs_of if r.get("tokens_in")]
        if len(usable) < 10:
            return
        seconds = st.median(r["wall_ms"] / 1000 for r in usable)
        tokens = st.median(float(r["tokens_in"]) for r in usable)
        if key:
            measured[key] = seconds
        samples.append((tokens, seconds))

    baseline_runs = done.get(BASELINE, [])
    collect(baseline_runs, None)
    for operator in catalog.operators:
        collect(done.get(operator.id, []), operator.id)

    baseline_tokens = st.median(float(r["tokens_in"]) for r in baseline_runs if r.get("tokens_in")) \
        if baseline_runs else 4600.0
    baseline_ctx = context_tokens(study.baseline_overrides) or 1.0

    curve = sorted(samples)

    def seconds_at(tokens: float) -> float:
        """Interpola na curva medida; extrapola pela inclinação do último trecho."""
        if not curve:
            return 30.0
        if len(curve) == 1 or tokens <= curve[0][0]:
            return curve[0][1]
        for (t0, s0), (t1, s1) in zip(curve, curve[1:]):
            if tokens <= t1:
                return s0 + (s1 - s0) * (tokens - t0) / (t1 - t0)
        (t0, s0), (t1, s1) = curve[-2], curve[-1]
        slope = (s1 - s0) / (t1 - t0) if t1 != t0 else 0.0
        return s1 + slope * (tokens - t1)

    def project(operator) -> float:
        if operator.id in measured:
            return measured[operator.id]
        # O prompt de um mutante não medido cresce com a sua configuração de
        # recuperação, na mesma proporção que o do baseline.
        tokens = baseline_tokens * (context_tokens(merged_overrides(operator)) / baseline_ctx)
        return seconds_at(tokens)

    total_done = sum(len(v) for k, v in done.items() if k != BASELINE)
    total_target = len(catalog.operators) * target
    logger.info("Campanha: %d/%d invocações (%.1f%%), %d com erro.",
                total_done, total_target, 100 * total_done / total_target,
                sum(1 for k, v in done.items() if k != BASELINE for r in v if r.get("error")))

    remaining_s = 0.0
    lines = []
    for operator in catalog.operators:
        got = len(done.get(operator.id, []))
        seconds = project(operator)
        missing = max(0, target - got)
        remaining_s += missing * seconds
        state = "ok" if missing == 0 else ("parcial" if got else "—")
        lines.append(f"  {operator.id:<3} {got:>3}/{target:<3} {seconds:>6.0f} s/inv  "
                     f"{'medido' if operator.id in measured else 'estimado':<8} {state}")
    for line in lines:
        logger.info("%s", line)

    judge_pairs = len(catalog.operators) * cases
    baseline_pairs = cases
    judge_s = 0.0
    if baseline_runs:
        # a avaliação do baseline já rodou: usa o próprio ritmo dela se der,
        # senão 22 s por par (5 julgamentos do O4 + embeddings do O1).
        judge_s = 22.0
    logger.info("Curva medida (tokens de prompt -> s): %s.",
                ", ".join(f"{int(t)}->{sec:.0f}" for t, sec in curve))
    logger.info("Falta na campanha: %.1f h.", remaining_s / 3600)
    logger.info("Avaliação depois (%d pares x ~%.0f s): %.1f h.",
                judge_pairs, judge_s, judge_pairs * judge_s / 3600)
    logger.info("Total até os vereditos: %.1f h.", (remaining_s + judge_pairs * judge_s) / 3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
