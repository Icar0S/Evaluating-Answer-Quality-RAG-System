"""Calibra MIN_SIMILARITY_SCORE — o piso de relevância do baseline do estudo.

Por que isso existe: o operador R4 remove o limiar de relevância. Se o baseline
já tiver limiar 0,0, R4 não remove nada e vira mutante equivalente por
construção — apply.py recusa aplicá-lo justamente para não deixar esse erro
passar despercebido. O limiar precisa então de um valor, e um valor chutado
viraria confundidor silencioso na classe "fora do corpus".

Método: sem geração, só embeddings. Para cada caso da suíte mede-se a
similaridade do melhor chunk recuperado. Casos que devem ser respondidos são
positivos (precisam passar do piso); casos fora do corpus e fora de escopo são
negativos (precisam ficar abaixo). Varre-se o piso e escolhe-se o que separa
melhor, preferindo o ponto médio da maior folga quando a separação é perfeita.

Custo: uma chamada de embedding por caso — segundos, não minutos.

Uso:
    python -m tests.mutation.tools.calibrate_retrieval_gate
    python -m tests.mutation.tools.calibrate_retrieval_gate --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation import stats as study_stats
from tests.mutation.operators import apply as operators
from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import ensure_index, load_study_config, sut_configuration
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.gate")

POSITIVE_BEHAVIORS = {"answer", "cite"}
NEGATIVE_BEHAVIORS = {"abstain", "refuse"}


def measure_top_similarity() -> list[dict]:
    from app.rag import retrieval

    measurements: list[dict] = []
    for case in load_cases():
        if case.expected_behavior not in POSITIVE_BEHAVIORS | NEGATIVE_BEHAVIORS:
            continue  # casos ambíguos não dependem de recuperação
        if case.is_draft and case.expected_behavior in POSITIVE_BEHAVIORS:
            continue  # pergunta ainda não escrita
        chunks, _ = retrieval.retrieve(case.question)
        top = max((c["similarity_score"] for c in chunks), default=0.0)
        measurements.append(
            {
                "case_id": case.case_id,
                "class": case.case_class,
                "expected_behavior": case.expected_behavior,
                "label": 1 if case.expected_behavior in POSITIVE_BEHAVIORS else 0,
                "top_similarity": round(top, 4),
            }
        )
    return measurements


def choose_threshold(measurements: list[dict]) -> dict:
    positives = [m["top_similarity"] for m in measurements if m["label"] == 1]
    negatives = [m["top_similarity"] for m in measurements if m["label"] == 0]
    if not positives or not negatives:
        raise SystemExit(
            "Preciso de casos positivos (answer/cite) e negativos (abstain/refuse) medidos. "
            "Escreva ao menos um caso factual pronto antes de calibrar o piso."
        )

    grid = sorted({round(v, 3) for v in positives + negatives} | {0.0})
    rows = []
    for threshold in grid:
        tp = sum(1 for v in positives if v >= threshold)
        fn = len(positives) - tp
        fp = sum(1 for v in negatives if v >= threshold)
        tn = len(negatives) - fp
        precision, recall, f1 = study_stats.precision_recall_f1(tp, fp, fn)
        rows.append(
            {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": round(f1, 4),
             "precision": round(precision, 4), "recall": round(recall, 4)}
        )

    best_f1 = max(row["f1"] for row in rows)
    candidates = [row for row in rows if row["f1"] == best_f1]

    separable = min(positives) > max(negatives)
    if separable:
        # Separação perfeita: o meio da folga é o ponto mais robusto a ruído.
        chosen_value = round((min(positives) + max(negatives)) / 2, 3)
        chosen = {"threshold": chosen_value, "f1": 1.0, "rule": "ponto médio da folga"}
    else:
        # Sem separação, prefere-se o maior limiar entre os de F1 máximo: ele
        # rejeita mais ruído, que é o comportamento que R4 depois remove.
        chosen = dict(max(candidates, key=lambda row: row["threshold"]))
        chosen["rule"] = "maior limiar com F1 máximo"

    return {
        "chosen": chosen,
        "separable": separable,
        "positives": {"n": len(positives), "min": min(positives), "max": max(positives)},
        "negatives": {"n": len(negatives), "min": min(negatives), "max": max(negatives)},
        "sweep": rows,
    }


def write_to_study(value: float) -> None:
    text = paths.STUDY_CONFIG.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"(?m)^(\s*min_similarity_score:\s*)([0-9.]+)\s*$", rf"\g<1>{value}", text, count=1
    )
    if count != 1:
        raise SystemExit("Não encontrei `min_similarity_score:` em config/study.yaml — edite à mão.")
    paths.STUDY_CONFIG.write_text(new_text, encoding="utf-8")
    logger.info("config/study.yaml atualizado: baseline_config.min_similarity_score = %s", value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibra o piso de similaridade do baseline.")
    parser.add_argument("--write", action="store_true", help="grava o valor em config/study.yaml")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="so constroi/atualiza o indice do estudo e sai (nao exige casos prontos)",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    study = load_study_config()
    # Mede-se SEM piso: o objetivo é observar a distribuição bruta.
    overrides = dict(study.baseline_overrides)
    overrides["min_similarity_score"] = 0.0

    # work/ precisa refletir corpus/base antes de indexar; sem mutante
    # aplicado, revert() e justamente "reconstruir do zero a partir do base".
    operators.revert()
    with sut_configuration(overrides) as resolved:
        report = ensure_index(resolved)
        if args.index_only:
            logger.info(
                "Indice do estudo pronto: %s chunks (%s).",
                report.get("chunks"),
                "reindexado" if report.get("reindexed") else "reaproveitado",
            )
            return 0
        measurements = measure_top_similarity()

    result = choose_threshold(measurements)
    result["measurements"] = measurements

    destination = paths.CONFIG_DIR / "retrieval_gate.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    chosen = result["chosen"]
    logger.info(
        "Positivos (n=%d): %.3f a %.3f | Negativos (n=%d): %.3f a %.3f",
        result["positives"]["n"], result["positives"]["min"], result["positives"]["max"],
        result["negatives"]["n"], result["negatives"]["min"], result["negatives"]["max"],
    )
    logger.info("Piso escolhido: %.3f (%s)", chosen["threshold"], chosen["rule"])
    if not result["separable"]:
        logger.warning(
            "Positivos e negativos se sobrepõem: nenhum piso separa perfeitamente. "
            "R4 continua válido, mas a classe 'fora do corpus' vai depender mais de P2 "
            "(instrução de abstenção) do que do piso — vale registrar na discussão."
        )
    logger.info("Detalhe em %s", destination)

    if args.write:
        write_to_study(float(chosen["threshold"]))
    else:
        logger.info("Use --write para gravar em config/study.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
