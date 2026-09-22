"""Orquestra baseline, campanha de mutantes e o nível L3 da RQ3.

Separação deliberada entre **executar** e **julgar**: este script só produz
results/runs.jsonl (o que o SUT respondeu). Os vereditos saem de evaluate.py,
sobre o log. Três razões:

1. A campanha custa ~8,5 h de GPU; os oráculos custam mais e podem precisar de
   reexecução (τ* recalibrado, juiz instável, backend de O3 trocado). Rejulgar
   sobre o log não gasta uma hora de GPU a mais.
2. A RQ3 compara o custo dos oráculos entre si. Misturar o tempo de geração com
   o tempo de julgamento tornaria essa conta impossível de desembaraçar depois.
3. "Nunca analisar sobre estado volátil de execução" (§4.1).

Retomada: run_id já presente em runs.jsonl é pulado. Uma campanha interrompida na
hora 6 continua de onde parou.

Uso:
    python -m tests.mutation.run_campaign --baseline
    python -m tests.mutation.run_campaign --campaign
    python -m tests.mutation.run_campaign --campaign --exclude-cuttable   # regra de corte
    python -m tests.mutation.run_campaign --l3
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.mutation.operators import apply as operators
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import ensure_index, invoke, load_study_config, sut_configuration
from tests.mutation.suite.loader import Case, load_cases

logger = get_logger("mutation.campaign")

BASELINE_ID = "baseline"


def existing_run_ids() -> set[str]:
    return {record["run_id"] for record in jsonl.read(paths.RUNS_JSONL)}


def _runnable_cases(cases: list[Case]) -> list[Case]:
    ready = [case for case in cases if not case.is_draft]
    skipped = len(cases) - len(ready)
    if skipped:
        logger.warning(
            "%d caso(s) ainda em rascunho ficam de fora desta execução. "
            "A campanha final exige a suíte completa (validate_suite.py).",
            skipped,
        )
    return ready


def run_block(
    mutant_id: str,
    operator_id: str | None,
    cases: list[Case],
    repetitions: int,
    overrides: dict,
    done: set[str],
    seed_from_repetition: bool,
    shuffle_per_repetition: bool = True,
) -> int:
    """Roda cases x repetitions com a configuração já aplicada. Devolve quantas invocações fez."""
    executed = 0
    total = len(cases) * repetitions
    started = time.perf_counter()

    for repetition in range(1, repetitions + 1):
        # A seed entra na configuração, não na chamada: o SUT lê seed de Settings.
        run_overrides = dict(overrides)
        if seed_from_repetition:
            run_overrides["generation_seed"] = repetition

        # Ordem dos casos embaralhada por repetição, com a repetição como semente.
        # Descoberto no baseline v2 (21/09): a temperatura 0 a semente não muda
        # nada, mas a resposta depende do PROMPT ANTERIOR no slot do servidor
        # (reuso de prefixo do cache KV). Em ordem fixa, r2..r10 de um caso têm
        # sempre o mesmo predecessor e saem byte a byte iguais — 10 repetições
        # que amostram 2 estados. Embaralhar dá a cada repetição um predecessor
        # diferente; é o que "repetir" deveria significar num servidor com cache.
        order = list(cases)
        if shuffle_per_repetition:
            random.Random(repetition).shuffle(order)

        with sut_configuration(run_overrides) as resolved:
            ensure_index(resolved)
            for case in order:
                run_id = f"{mutant_id}-{case.case_id}-r{repetition}"
                if run_id in done:
                    continue
                invocation = invoke(
                    mutant_id=mutant_id,
                    operator=operator_id,
                    case_id=case.case_id,
                    question=case.question,
                    repetition=repetition,
                )
                jsonl.append(paths.RUNS_JSONL, invocation.to_record())
                done.add(run_id)
                executed += 1
                if executed % 10 == 0:
                    elapsed = time.perf_counter() - started
                    rate = elapsed / executed
                    logger.info(
                        "  %s: %d/%d invocações (%.1fs/inv, faltam ~%.0f min)",
                        mutant_id, executed, total, rate, rate * (total - executed) / 60,
                    )
    return executed


def run_baseline(study, cases: list[Case], done: set[str]) -> None:
    repetitions = int(study.execution["baseline_repetitions"])
    logger.info("BASELINE: %d casos x %d execuções", len(cases), repetitions)
    operators.revert()  # garante corpus limpo
    executed = run_block(
        BASELINE_ID, None, cases, repetitions, study.baseline_overrides, done,
        bool(study.execution.get("seed_from_repetition", True)),
        bool(study.execution.get("shuffle_cases_per_repetition", True)),
    )
    logger.info("BASELINE concluído: %d novas invocações.", executed)


def run_campaign(study, cases: list[Case], done: set[str], selected: list[str] | None, exclude_cuttable: bool) -> None:
    catalog = operators.load_catalog()
    chosen = catalog.active(include_cuttable=not exclude_cuttable)
    if selected:
        chosen = [op for op in chosen if op.id in selected]
    if not chosen:
        logger.error("Nenhum operador selecionado.")
        return

    repetitions = int(study.execution["campaign_repetitions"])
    baseline = study.baseline_overrides
    # Mutantes que não mexem no índice rodam primeiro: aproveitam o índice do
    # baseline e adiam o custo de reindexação para o fim do lote.
    ordered = sorted(chosen, key=lambda op: (op.kind == "corpus", op.id))

    logger.info(
        "CAMPANHA: %d mutantes x %d casos x %d execuções = %d invocações",
        len(ordered), len(cases), repetitions, len(ordered) * len(cases) * repetitions,
    )

    for position, operator in enumerate(ordered, start=1):
        logger.info("[%d/%d] %s — %s", position, len(ordered), operator.id, operator.name)
        mutant = operators.apply(operator.id, catalog, baseline, study.raw)
        merged = dict(baseline)
        merged.update(mutant.config_overrides)
        if mutant.config_overrides:
            logger.info("  overrides: %s", mutant.config_overrides)
        for entry in mutant.file_log:
            logger.info("  corpus: %s", entry)

        executed = run_block(
            operator.id, operator.id, cases, repetitions, merged, done,
            bool(study.execution.get("seed_from_repetition", True)),
            bool(study.execution.get("shuffle_cases_per_repetition", True)),
        )
        logger.info("  %s: %d novas invocações.", operator.id, executed)

    operators.revert()
    logger.info("CAMPANHA concluída; work/ revertido ao corpus base.")


def run_l3(study, cases: list[Case], done: set[str]) -> None:
    """Nível L3 da RQ3: 30 execuções sobre um subconjunto de 10 casos (§7.6)."""
    repetitions = int(study.execution["levels"]["L3"])
    sample_size = int(study.execution["l3_case_sample"])
    # Subconjunto determinístico: os 10 primeiros casos prontos em ordem de id.
    # Determinístico importa mais que aleatório aqui — o L3 precisa ser comparável
    # com L1 e L2 sobre EXATAMENTE os mesmos casos.
    sample = sorted(cases, key=lambda case: case.case_id)[:sample_size]
    logger.info("L3: %d casos x %d execuções (baseline)", len(sample), repetitions)
    operators.revert()
    executed = run_block(
        "l3-baseline", None, sample, repetitions, study.baseline_overrides, done,
        bool(study.execution.get("seed_from_repetition", True)),
        bool(study.execution.get("shuffle_cases_per_repetition", True)),
    )
    logger.info("L3 concluído: %d novas invocações.", executed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roda baseline e campanha de mutação.")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--l3", action="store_true")
    parser.add_argument("--operators", help="lista separada por vírgula (ex.: R1,P2) para rodar só alguns")
    parser.add_argument("--exclude-cuttable", action="store_true", help="regra de corte da semana 6 (12 operadores)")
    parser.add_argument("--repetitions", type=int, help="sobrepõe o número de execuções desta fase")
    parser.add_argument("--cases", help="lista de case_id separada por vírgula (piloto)")
    args = parser.parse_args(argv)

    if not (args.baseline or args.campaign or args.l3):
        parser.error("escolha ao menos uma fase: --baseline, --campaign ou --l3")

    paths.ensure_dirs()
    study = load_study_config()
    if args.repetitions:
        study.raw["execution"]["baseline_repetitions"] = args.repetitions
        study.raw["execution"]["campaign_repetitions"] = args.repetitions

    cases = _runnable_cases(load_cases())
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        cases = [case for case in cases if case.case_id in wanted]
    if not cases:
        logger.error("Nenhum caso executável. Escreva a suíte (semana 2) antes de rodar a campanha.")
        return 1

    done = existing_run_ids()
    if done:
        logger.info("Retomando: %d invocações já registradas em runs.jsonl.", len(done))

    started = time.perf_counter()
    if args.baseline:
        run_baseline(study, cases, done)
    if args.campaign:
        selected = [o.strip() for o in args.operators.split(",")] if args.operators else None
        run_campaign(study, cases, done, selected, args.exclude_cuttable)
    if args.l3:
        run_l3(study, cases, done)

    logger.info("Tempo total: %.1f min", (time.perf_counter() - started) / 60)
    logger.info("Próximo passo: python -m tests.mutation.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
