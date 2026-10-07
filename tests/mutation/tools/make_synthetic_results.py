"""Gera resultados SINTÉTICOS para exercitar analyze.py sem gastar GPU.

Para que serve: validar o caminho `run_campaign -> evaluate -> coherence ->
analyze` na semana 5, antes de a campanha de ~8,5 h começar na semana 6.
Descobrir na segunda-feira que a análise quebra num caso de borda é barato;
descobrir depois de queimar a noite de GPU não é.

**Estes dados não são resultado de nada.** Toda linha sai com `"synthetic":
true`, os arquivos vão para results/synthetic/ por padrão, e analyze.py aponta
para lá com --results-dir. Nenhum número daqui pode entrar no artigo.

Uso:
    python -m tests.mutation.tools.make_synthetic_results
    python -m tests.mutation.analyze --results-dir tests/mutation/results/synthetic
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.operators.apply import load_catalog
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.synthetic")

# Perfil de detecção por camada, escolhido para reproduzir a H1 (corpus e índice
# sobrevivem mais) — é um cenário de teste do código, não uma previsão.
LAYER_DETECTION = {
    "corpus": 0.12,
    "index": 0.15,
    "chunking": 0.30,
    "retrieval": 0.45,
    "prompt": 0.55,
}

# Sensibilidade relativa de cada oráculo, para o swing da RQ2 não sair zero.
ORACLE_SENSITIVITY = {"O1": 0.85, "O2": 1.00, "O3": 0.75, "O4": 1.15, "O5": 1.25}

CODES_BY_LAYER = {
    "corpus": ["RET-MISS", "RET-STALE"],
    "index": ["RET-MISS", "RET-STALE"],
    "chunking": ["RET-MISS", "RET-DILUTE"],
    "retrieval": ["RET-DILUTE", "GEN-NOABSTAIN"],
    "prompt": ["GEN-HALLUC", "GEN-NOABSTAIN", "FMT-BREAK"],
}


def _run_record(mutant_id, operator, case, repetition, rng, fails):
    answer = "Resposta sintética divergente." if fails else "Resposta sintética de referência."
    return {
        "run_id": f"{mutant_id}-{case.case_id}-r{repetition}",
        "mutant_id": mutant_id,
        "operator": operator,
        "case_id": case.case_id,
        "repetition": repetition,
        "question": case.question,
        "answer": answer,
        "retrieved_chunk_ids": [f"doc::p{rng.randint(1, 20)}::c0"],
        "retrieved_documents": ["doc.pdf"],
        "retrieval_context": ["trecho sintético"],
        "similarity_scores": [round(rng.uniform(0.4, 0.9), 4)],
        "grounded": not fails,
        "model": "qwen3:8b",
        "embed_model": "nomic-embed-text",
        "temperature": 0.0,
        "seed": repetition,
        "top_k": 4,
        "tokens_in": rng.randint(600, 1400),
        "tokens_out": rng.randint(80, 400),
        "wall_ms": round(rng.uniform(6000, 14000), 2),
        "retrieval_ms": round(rng.uniform(80, 400), 2),
        "generation_ms": round(rng.uniform(5000, 13000), 2),
        "gpu_s": round(rng.uniform(4.0, 11.0), 4),
        "wh": round(rng.uniform(0.04, 0.16), 6),
        "error": None,
        "ts": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
    }


def generate(results_dir: Path, seed: int, repetitions: int, baseline_repetitions: int) -> None:
    rng = random.Random(seed)
    cases = load_cases()
    catalog = load_catalog()

    runs_path = results_dir / "runs.jsonl"
    verdicts_path = results_dir / "verdicts.jsonl"
    coherence_path = results_dir / "coherence.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    verdicts: list[dict] = []
    coherence: list[dict] = []

    # ---- baseline: quase tudo estável; alguns casos instáveis viram flakiness.
    unstable = set(rng.sample([case.case_id for case in cases], k=max(1, len(cases) // 10)))
    for case in cases:
        for repetition in range(1, baseline_repetitions + 1):
            runs.append(_run_record("baseline", None, case, repetition, rng, fails=False))
        for oracle in ORACLE_SENSITIVITY:
            stable = case.case_id not in unstable
            verdicts.append(
                {
                    "level": "L2", "mutant_id": "baseline", "operator": None,
                    "case_id": case.case_id, "oracle": oracle, "verdict": "pass",
                    "score": round(rng.uniform(0.8, 0.98), 4), "modal": True, "stable": stable,
                    "evaluable": stable, "kill": False, "detail": "", "runs": repetitions,
                    "per_run": ["pass"] * repetitions, "components": {},
                    "cost": _oracle_cost(oracle, rng), "coherent": None, "coherence_code": None,
                    "synthetic": True,
                }
            )

    # ---- mutantes
    for operator in catalog.operators:
        base_rate = LAYER_DETECTION.get(operator.layer, 0.3)
        for case in cases:
            case_fails = {
                oracle: rng.random() < min(0.95, base_rate * sensitivity)
                for oracle, sensitivity in ORACLE_SENSITIVITY.items()
            }
            any_fail = any(case_fails.values())
            for repetition in range(1, repetitions + 1):
                runs.append(_run_record(operator.id, operator.id, case, repetition, rng, any_fail))

            for oracle, fails in case_fails.items():
                evaluable = case.case_id not in unstable
                verdicts.append(
                    {
                        "level": "L2", "mutant_id": operator.id, "operator": operator.id,
                        "case_id": case.case_id, "oracle": oracle,
                        "verdict": "fail" if fails else "pass",
                        "score": round(rng.uniform(0.2, 0.6) if fails else rng.uniform(0.75, 0.97), 4),
                        "modal": True, "stable": True, "evaluable": evaluable,
                        "kill": bool(fails and evaluable), "detail": "motivo sintético" if fails else "",
                        "runs": repetitions, "per_run": ["fail" if fails else "pass"] * repetitions,
                        "components": {}, "cost": _oracle_cost(oracle, rng),
                        "coherent": None, "coherence_code": None, "synthetic": True,
                    }
                )

            if any_fail and case.case_id not in unstable:
                codes = CODES_BY_LAYER.get(operator.layer, ["OTHER"])
                # 80% de mortes coerentes: o suficiente para MS_bruto e
                # MS_coerente saírem diferentes na tabela.
                coherent = rng.random() < 0.8
                coherence.append(
                    {
                        "mutant_id": operator.id, "operator": operator.id, "case_id": case.case_id,
                        "oracles": sorted(o for o, f in case_fails.items() if f),
                        "coherence_code": rng.choice(codes) if coherent else "OTHER",
                        "coherent": coherent,
                        "classified_at": datetime.now(timezone.utc).isoformat(),
                        "pass": "first", "synthetic": True,
                    }
                )

    # Nível L1: mesmo conjunto, uma execução só — detecta menos, custa menos.
    l1_verdicts = []
    for verdict in verdicts:
        if verdict["mutant_id"] == "baseline":
            l1_verdicts.append({**verdict, "level": "L1", "runs": 1, "per_run": verdict["per_run"][:1]})
            continue
        # Com uma execução só, parte das mortes se perde no ruído.
        keeps = verdict["verdict"] == "fail" and rng.random() < 0.8
        l1_verdicts.append(
            {
                **verdict, "level": "L1", "runs": 1,
                "per_run": verdict["per_run"][:1],
                "verdict": "fail" if keeps else "pass",
                "kill": bool(keeps and verdict["evaluable"]),
            }
        )
    verdicts.extend(l1_verdicts)

    jsonl.write_all(runs_path, runs)
    jsonl.write_all(verdicts_path, verdicts)
    jsonl.write_all(coherence_path, coherence)
    logger.info(
        "SINTÉTICO: %d invocações, %d vereditos, %d classificações em %s",
        len(runs), len(verdicts), len(coherence), results_dir,
    )
    logger.info("Analisar com: python -m tests.mutation.analyze --results-dir %s", results_dir)


def _oracle_cost(oracle: str, rng: random.Random) -> dict:
    """Custos que reproduzem a assimetria do §6.4: O4 caro, O1/O2/O5 baratos."""
    profile = {
        "O1": (40, 120), "O2": (1, 5), "O5": (40, 130),
        "O3": (4000, 12000), "O4": (12000, 30000),
    }[oracle]
    wall_ms = rng.uniform(*profile)
    return {
        "wall_ms": round(wall_ms, 2),
        "gpu_s": round(wall_ms / 1000 * rng.uniform(0.3, 0.8), 4),
        "wh": round(wall_ms / 1000 * rng.uniform(0.002, 0.008), 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera resultados sintéticos para testar a análise.")
    parser.add_argument("--results-dir", default=str(paths.RESULTS_DIR / "synthetic"))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--baseline-repetitions", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="sobrescreve resultados existentes")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = paths.PROJECT_ROOT / results_dir

    if results_dir.resolve() == paths.RESULTS_DIR.resolve() and not args.force:
        raise SystemExit(
            "Recusando escrever dados sintéticos por cima de results/. Use o padrão "
            "(results/synthetic/) ou --force se for mesmo isso que você quer."
        )
    if (results_dir / "runs.jsonl").exists() and not args.force:
        raise SystemExit(f"{results_dir}/runs.jsonl já existe. Use --force para sobrescrever.")

    generate(results_dir, args.seed, args.repetitions, args.baseline_repetitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
