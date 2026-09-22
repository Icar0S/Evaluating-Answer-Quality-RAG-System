"""Aplica os cinco oráculos sobre results/runs.jsonl e produz results/verdicts.jsonl.

Roda sobre o log, nunca sobre o SUT: rejulgar não custa GPU de geração. É o que
torna possível recalibrar τ*, trocar o backend de O3 ou reexecutar o juiz sem
repetir as ~3.000 invocações.

Regras que este módulo materializa (§7.1):

    assinatura(caso, mutante) = veredito modal das N execuções
    caso avaliável            = veredito único nas N execuções do BASELINE
    S                         = conjunto de casos avaliáveis (por oráculo)

O conjunto avaliável é calculado por oráculo, e não globalmente: um caso pode ser
estável sob assertivas determinísticas e instável sob cosseno. O protocolo já
conta as observações assim ("18 × |S| observações por oráculo").

Custo: O1/O2/O5 rodam em cada execução (baratos); O3/O4 rodam só sobre a resposta
modal (§6.4). O tempo, a GPU e a energia de cada julgamento ficam registrados na
própria linha do veredito, que é o insumo da RQ3.

Níveis de rigor (RQ3, §7.6): `--level` limita quantas execuções entram no
veredito — L1 usa a primeira, L2 as cinco, L3 as trinta. Os vereditos saem
etiquetados com o nível, então a mesma campanha responde "o que L1 teria
detectado?" sem reexecutar o SUT. Só o custo dos oráculos muda entre níveis, e
esse custo é medido, não estimado.

Uso:
    python -m tests.mutation.evaluate
    python -m tests.mutation.evaluate --level L1              # veredito pontual
    python -m tests.mutation.evaluate --oracles O1,O2,O5      # pula os caros
    python -m tests.mutation.evaluate --mutants baseline,R1
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.mutation.oracles import o1_cosine, o2_assertions, o3_ragas, o4_judge, o5_conjunctive
from tests.mutation.oracles.base import Verdict, modal_answer, normalize
from tests.mutation.runner import cost, jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import load_study_config, sut_configuration
from tests.mutation.suite.loader import load_assertions, load_cases

logger = get_logger("mutation.evaluate")

PER_RUN_ORACLES = ("O1", "O2", "O5")
MODAL_ORACLES = ("O3", "O4")
BASELINE_ID = "baseline"


def _hit_generation_cap(run: dict, cap: int | None) -> bool:
    """A execução foi cortada pelo teto de geração antes de produzir resposta?

    Sinal duplo de propósito: só conta quando a resposta está vazia E o número
    de tokens gerados bateu no teto. Resposta vazia com poucos tokens é outra
    coisa (erro de rede, recusa degenerada) e continua sendo julgada.
    """
    if cap is None or run.get("error"):
        return False
    tokens_out = run.get("tokens_out")
    return not (run.get("answer") or "").strip() and tokens_out is not None and tokens_out >= int(cap)


def group_runs(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["mutant_id"], run["case_id"])].append(run)
    for runs_of_pair in grouped.values():
        runs_of_pair.sort(key=lambda run: run["repetition"])
    return grouped


def modal_verdict(verdicts: list[Verdict]) -> tuple[str, bool, float | None]:
    """Veredito modal e se houve unanimidade nas N execuções."""
    usable = [v for v in verdicts if v.evaluable]
    if not usable:
        return "pass", False, None
    counts = Counter(v.verdict for v in usable)
    modal = counts.most_common(1)[0][0]
    unanimous = len(counts) == 1
    scores = [v.score for v in usable if v.score is not None]
    mean_score = round(sum(scores) / len(scores), 4) if scores else None
    return modal, unanimous, mean_score


def evaluate_per_run(
    run: dict,
    reference: str,
    case_assertions: list[dict[str, Any]],
    defaults: dict[str, Any],
    tau: float,
    embedding_model: str,
) -> dict[str, Verdict]:
    o1 = o1_cosine.evaluate(run["answer"], reference, tau, embedding_model)
    o2 = o2_assertions.evaluate(run["answer"], case_assertions, defaults)
    o5 = o5_conjunctive.evaluate(o1, o2)
    return {"O1": o1, "O2": o2, "O5": o5}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aplica os oráculos sobre runs.jsonl.")
    parser.add_argument("--level", default="L2", choices=["L1", "L2", "L3"], help="nível de rigor (RQ3)")
    parser.add_argument("--oracles", help="subconjunto, ex.: O1,O2,O5")
    parser.add_argument("--mutants", help="subconjunto de mutant_id")
    parser.add_argument("--force", action="store_true", help="recalcula vereditos já existentes")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    study = load_study_config()
    oracle_config = study.oracles
    enabled = [o.strip() for o in args.oracles.split(",")] if args.oracles else list(oracle_config["enabled"])

    tau = oracle_config["O1"].get("tau")
    if "O1" in enabled or "O5" in enabled:
        if tau is None:
            logger.error(
                "τ* não calibrado (oracles.O1.tau = null em config/study.yaml). "
                "Rode: python -m tests.mutation.calibration.calibrate --write"
            )
            return 1

    max_repetitions = int(study.execution["levels"][args.level])
    runs = [r for r in jsonl.read(paths.RUNS_JSONL) if r["repetition"] <= max_repetitions]
    if not runs:
        logger.error("results/runs.jsonl vazio — rode run_campaign.py antes.")
        return 1

    # Execuções cortadas pelo teto de geração NÃO viram veredito (22/09).
    # O qwen3 raciocina antes de responder; num caso difícil ele consome os
    # `num_predict` tokens no raciocínio e devolve resposta VAZIA. Reprovar isso
    # atribuiria ao mutante um efeito do nosso teto — o erro de atribuição que
    # o estudo inteiro existe para evitar. Sai do numerador e do denominador e
    # é reportado; a alternativa (subir o teto) muda `num_ctx` e o custo de
    # todas as invocações, e o teto é parte da configuração declarada.
    cap = study.baseline_overrides.get("generation_num_predict")
    capped = [r for r in runs if _hit_generation_cap(r, cap)]
    if capped:
        by_case: dict[str, int] = defaultdict(int)
        for record in capped:
            by_case[f"{record['mutant_id']}/{record['case_id']}"] += 1
        logger.warning(
            "%d execução(ões) atingiram o teto de geração (%s tokens) e ficam fora dos vereditos: %s",
            len(capped), cap, dict(sorted(by_case.items())),
        )
        runs = [r for r in runs if not _hit_generation_cap(r, cap)]

    cases = {case.case_id: case for case in load_cases()}
    assertions, defaults = load_assertions()
    grouped = group_runs(runs)

    if args.mutants:
        wanted = {m.strip() for m in args.mutants.split(",")}
        grouped = {key: value for key, value in grouped.items() if key[0] in wanted}

    done: set[tuple[str, str, str]] = set()
    if not args.force:
        done = {
            (record.get("level", "L2"), record["mutant_id"], record["case_id"], record["oracle"])
            for record in jsonl.read(paths.VERDICTS_JSONL)
        }
        if done:
            logger.info("Retomando: %d vereditos já calculados.", len(done))

    embedding_model = oracle_config["O1"]["embedding_model"]
    o3_config = o3_ragas.O3Config(
        backend=oracle_config["O3"]["backend"],
        judge_model=study.models["judge"],
        embedding_model=embedding_model,
        thresholds=oracle_config["O3"].get("thresholds"),
        aggregate=oracle_config["O3"].get("aggregate", "all"),
    )
    judge_model = study.models["judge"]
    judgments = int(oracle_config["O4"].get("judgments", 5))
    judge_temperature = float(oracle_config["O4"].get("temperature", 0.0))

    # 1ª passada: vereditos por execução e por (mutante, caso).
    per_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    # sut_configuration serve aqui só para fixar o Ollama e o modelo de embedding
    # do oráculo; nenhuma geração do SUT acontece nesta fase.
    with sut_configuration(study.baseline_overrides):
        for (mutant_id, case_id), pair_runs in sorted(grouped.items()):
            case = cases.get(case_id)
            if case is None:
                logger.warning("case_id %s não está em golden.jsonl — ignorado.", case_id)
                continue

            valid_runs = [run for run in pair_runs if not run.get("error")]
            if not valid_runs:
                logger.warning("%s/%s: todas as execuções falharam — sem veredito.", mutant_id, case_id)
                continue

            reference = case.expected_output
            case_assertions = assertions.get(case_id, [])
            results: dict[str, dict[str, Any]] = {}

            per_run_verdicts: dict[str, list[Verdict]] = defaultdict(list)
            oracle_cost: dict[str, dict[str, float]] = defaultdict(lambda: {"wall_ms": 0.0, "gpu_s": 0.0, "wh": 0.0})

            if any(o in enabled for o in PER_RUN_ORACLES):
                # Uma ida ao Ollama para as N repetições mais a referência, em vez
                # de uma por repetição. Ver o docstring de o1_cosine.prime: isto
                # é sobre medir o custo certo na RQ3, não sobre desempenho.
                if "O1" in enabled or "O5" in enabled:
                    o1_cosine.prime([reference] + [r["answer"] for r in valid_runs], embedding_model)
                for run in valid_runs:
                    started = time.perf_counter()
                    verdicts = evaluate_per_run(
                        run, reference, case_assertions, defaults, tau, embedding_model
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    for name, verdict in verdicts.items():
                        if name in enabled:
                            per_run_verdicts[name].append(verdict)
                            oracle_cost[name]["wall_ms"] += elapsed_ms / len(verdicts)

                for name, verdicts_list in per_run_verdicts.items():
                    verdict, unanimous, mean_score = modal_verdict(verdicts_list)
                    failures = [v.detail for v in verdicts_list if v.failed]
                    results[name] = {
                        "verdict": verdict,
                        "score": mean_score,
                        "per_run": [v.verdict if v.evaluable else "na" for v in verdicts_list],
                        "stable": unanimous,
                        "evaluable_oracle": any(v.evaluable for v in verdicts_list),
                        "detail": (failures[0] if failures else ""),
                        "runs": len(verdicts_list),
                        "cost": dict(oracle_cost[name]),
                    }

            if any(o in enabled for o in MODAL_ORACLES):
                answers = [run["answer"] for run in valid_runs]
                modal = modal_answer(answers)
                modal_run = next(
                    (run for run in valid_runs if normalize(run["answer"]) == normalize(modal)),
                    valid_runs[0],
                )

                if "O3" in enabled:
                    with cost.measure() as holder:
                        verdict = o3_ragas.evaluate(
                            case.question, modal, modal_run["retrieval_context"], reference, o3_config
                        )
                    sample = holder[0]
                    results["O3"] = {
                        "verdict": verdict.verdict,
                        "score": verdict.score,
                        "stable": True,
                        "evaluable_oracle": verdict.evaluable,
                        "detail": verdict.detail,
                        "runs": 1,
                        "components": verdict.components,
                        "cost": {"wall_ms": sample.wall_ms, "gpu_s": sample.gpu_s, "wh": sample.wh},
                    }

                if "O4" in enabled:
                    with cost.measure() as holder:
                        verdict = o4_judge.evaluate(
                            case.question, reference, modal, judge_model, judgments, judge_temperature
                        )
                    sample = holder[0]
                    results["O4"] = {
                        "verdict": verdict.verdict,
                        "score": verdict.score,
                        "stable": True,
                        "evaluable_oracle": verdict.evaluable,
                        "detail": verdict.detail,
                        "runs": judgments,
                        "components": verdict.components,
                        "cost": {"wall_ms": sample.wall_ms, "gpu_s": sample.gpu_s, "wh": sample.wh},
                    }

            per_pair[(mutant_id, case_id)] = results
            logger.info(
                "%s/%s: %s",
                mutant_id,
                case_id,
                ", ".join(f"{name}={data['verdict']}" for name, data in sorted(results.items())),
            )

    # 2ª passada: conjunto avaliável S, definido pelo baseline, por oráculo.
    requires_pass = bool(study.execution.get("evaluable_requires_baseline_pass", True))
    evaluable: dict[tuple[str, str], bool] = {}
    for (mutant_id, case_id), results in per_pair.items():
        if mutant_id != BASELINE_ID:
            continue
        for oracle, data in results.items():
            stable = bool(data["stable"]) and bool(data["evaluable_oracle"])
            if requires_pass:
                stable = stable and data["verdict"] == "pass"
            evaluable[(case_id, oracle)] = stable

    if not any(key[0] == BASELINE_ID for key in per_pair):
        logger.warning(
            "Sem execuções de baseline neste lote: o campo `evaluable` sai como null e "
            "analyze.py não vai conseguir montar S. Rode --baseline antes da análise final."
        )

    written = 0
    for (mutant_id, case_id), results in sorted(per_pair.items()):
        operator = next(
            (run["operator"] for run in grouped[(mutant_id, case_id)] if run.get("operator")), None
        )
        for oracle, data in sorted(results.items()):
            if not args.force and (args.level, mutant_id, case_id, oracle) in done:
                continue
            is_evaluable = evaluable.get((case_id, oracle))
            record = {
                "level": args.level,
                "mutant_id": mutant_id,
                "operator": operator,
                "case_id": case_id,
                "oracle": oracle,
                "verdict": data["verdict"],
                "score": data["score"],
                "modal": True,
                "stable": data["stable"],
                "evaluable": is_evaluable,
                "kill": bool(
                    mutant_id != BASELINE_ID
                    and data["verdict"] == "fail"
                    and data["evaluable_oracle"]
                    and (is_evaluable is not False)
                ),
                "detail": data.get("detail", "")[:500],
                "runs": data.get("runs"),
                "per_run": data.get("per_run", []),
                "components": data.get("components", {}),
                "cost": data.get("cost", {}),
                # Preenchidos por coherence.py (§7.4), nunca aqui.
                "coherent": None,
                "coherence_code": None,
            }
            jsonl.append(paths.VERDICTS_JSONL, record)
            written += 1

    logger.info("%d vereditos escritos em %s (nível %s, até %d execuções por caso)",
                written, paths.VERDICTS_JSONL, args.level, max_repetitions)
    sizes = Counter(oracle for (case_id, oracle), ok in evaluable.items() if ok)
    if sizes:
        logger.info("Conjunto avaliável |S| por oráculo: %s", dict(sizes))
    logger.info("Próximo passo: python -m tests.mutation.coherence --classify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
