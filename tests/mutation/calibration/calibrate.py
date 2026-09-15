"""Calibra τ* do oráculo O1 sobre os 300 pares rotulados (§6.3).

Procedimento fixado no protocolo: varrer τ ∈ [0,60; 0,95] passo 0,01; reportar
precisão, revocação, F1 e AUC; escolher τ* por F1 máximo, com IC bootstrap de 95%.

τ* não é uma pergunta de pesquisa — é um instrumento. Por isso ele é calibrado
uma vez, antes da campanha, e fica congelado: recalibrar depois de ver os
resultados transformaria o instrumento em grau de liberdade do analista.

Uso:
    python -m tests.mutation.calibration.calibrate
    python -m tests.mutation.calibration.calibrate --write    # grava τ* em config/study.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation import stats as study_stats
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import load_study_config, sut_configuration

logger = get_logger("mutation.calibration")

TAU_GRID = [round(0.60 + 0.01 * i, 2) for i in range(36)]  # 0.60 .. 0.95


def score_pairs(pairs: list[dict[str, Any]], model: str) -> list[float]:
    from tests.mutation.oracles.o1_cosine import similarity

    scores: list[float] = []
    for index, pair in enumerate(pairs, start=1):
        scores.append(similarity(pair["candidate"], pair["reference"], model))
        if index % 50 == 0:
            logger.info("  %d/%d pares embedados", index, len(pairs))
    return scores


def sweep(scores: list[float], labels: list[int]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for tau in TAU_GRID:
        tp = sum(1 for s, y in zip(scores, labels) if s >= tau and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= tau and y == 0)
        fn = sum(1 for s, y in zip(scores, labels) if s < tau and y == 1)
        tn = sum(1 for s, y in zip(scores, labels) if s < tau and y == 0)
        precision, recall, f1 = study_stats.precision_recall_f1(tp, fp, fn)
        rows.append(
            {
                "tau": tau,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round((tp + tn) / len(scores), 4) if scores else 0.0,
            }
        )
    return rows


def _bootstrap_tau(scores: list[float], labels: list[int], n_boot: int = 2000, seed: int = 20261003) -> tuple[float, float]:
    """IC bootstrap do próprio τ*: reamostra pares e recalcula o argmax de F1."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(scores))
    taus: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        sample_scores = [scores[i] for i in sample]
        sample_labels = [labels[i] for i in sample]
        best = max(sweep(sample_scores, sample_labels), key=lambda row: row["f1"])
        taus.append(best["tau"])
    low, high = np.percentile(taus, [2.5, 97.5])
    return float(low), float(high)


def write_tau_to_study(tau: float) -> None:
    """Grava τ* em config/study.yaml preservando os comentários do arquivo."""
    text = paths.STUDY_CONFIG.read_text(encoding="utf-8")
    new_text, count = re.subn(r"(?m)^(\s*tau:\s*)(null|[0-9.]+)\s*$", rf"\g<1>{tau}", text, count=1)
    if count != 1:
        raise SystemExit("Não encontrei a chave `tau:` em config/study.yaml — edite à mão.")
    paths.STUDY_CONFIG.write_text(new_text, encoding="utf-8")
    logger.info("config/study.yaml atualizado: oracles.O1.tau = %s", tau)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibra o limiar τ* do oráculo O1.")
    parser.add_argument("--write", action="store_true", help="grava τ* em config/study.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000, help="reamostragens para o IC de τ*")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="calibra mesmo com pares EQUIV ainda não conferidos visualmente",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    pairs = jsonl.read(paths.CALIBRATION_PAIRS)
    if not pairs:
        logger.error("calibration/pairs.jsonl vazio — rode build_pairs.py antes.")
        return 1

    unreviewed = [p for p in pairs if p["label"] == "EQUIV" and not p.get("reviewed")]
    if unreviewed and not args.allow_unreviewed:
        logger.error(
            "%d pares EQUIV ainda não conferidos. O §6.3 pede conferência visual de 100%% "
            "antes de calibrar (uma paráfrase que perdeu um fato é um rótulo errado, e um "
            "rótulo errado desloca τ*). Confira e marque \"reviewed\": true, ou use "
            "--allow-unreviewed se estiver só testando o pipeline.",
            len(unreviewed),
        )
        return 1

    study = load_study_config()
    model = study.oracles["O1"]["embedding_model"]

    logger.info("Embedando %d pares com %s...", len(pairs), model)
    with sut_configuration(study.baseline_overrides):
        scores = score_pairs(pairs, model)

    labels = [1 if pair["label"] == "EQUIV" else 0 for pair in pairs]
    rows = sweep(scores, labels)
    best = max(rows, key=lambda row: row["f1"])
    auc = study_stats.auc_roc(scores, labels)

    # IC do F1 no τ* escolhido: reamostra os acertos/erros da classificação.
    correct = [
        1.0 if ((score >= best["tau"]) == bool(label)) else 0.0 for score, label in zip(scores, labels)
    ]
    accuracy_ci = study_stats.bootstrap_ci(correct)
    tau_low, tau_high = _bootstrap_tau(scores, labels, args.bootstrap)

    equiv_scores = [s for s, y in zip(scores, labels) if y == 1]
    erro_scores = [s for s, y in zip(scores, labels) if y == 0]

    report = {
        "pairs": len(pairs),
        "equiv": len(equiv_scores),
        "erro": len(erro_scores),
        "unreviewed_equiv": len(unreviewed),
        "embedding_model": model,
        "tau_star": best["tau"],
        "tau_ci95": [round(tau_low, 3), round(tau_high, 3)],
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "accuracy": best["accuracy"],
        "accuracy_ci95": [round(accuracy_ci.low, 4), round(accuracy_ci.high, 4)],
        "auc": round(auc, 4),
        "equiv_mean": round(float(np.mean(equiv_scores)), 4) if equiv_scores else None,
        "erro_mean": round(float(np.mean(erro_scores)), 4) if erro_scores else None,
        "sweep": rows,
        "limitation": (
            "Erros da classe ERRO são gerados programaticamente e podem ser mais fáceis de "
            "detectar que erros naturais de um LLM. τ* é, portanto, um limite superior do "
            "desempenho do oráculo O1 — declarar no artigo (§6.3, §10)."
        ),
    }
    paths.CALIBRATION_TAU.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("τ* = %.2f  IC95 [%.2f; %.2f]", best["tau"], tau_low, tau_high)
    logger.info(
        "F1=%.3f  precisão=%.3f  revocação=%.3f  AUC=%.3f",
        best["f1"], best["precision"], best["recall"], auc,
    )
    logger.info("cosseno médio: EQUIV=%s  ERRO=%s", report["equiv_mean"], report["erro_mean"])
    logger.info("Relatório completo em %s", paths.CALIBRATION_TAU)

    if tau_high - tau_low > 0.15:
        logger.warning(
            "IC de τ* largo (%.2f). Sinal de que EQUIV e ERRO se sobrepõem muito: "
            "reveja as paráfrases ou aumente o número de pares.",
            tau_high - tau_low,
        )

    if args.write:
        write_tau_to_study(best["tau"])
    else:
        logger.info("Use --write para gravar τ* em config/study.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
