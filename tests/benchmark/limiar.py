"""Recalibração do limiar de similaridade (MIN_SIMILARITY_SCORE) para o E6 do plano-benchmark.md.

A regra é uma cópia da choose_threshold() do estudo de mutação
(tests/mutation/tools/calibrate_retrieval_gate.py), aplicada às similaridades
do primeiro trecho que a C1 já mede: separação perfeita, o ponto médio da folga;
sem separação, o maior F1 mantendo margem de 0,02 abaixo do menor positivo
(barrar uma pergunta de dentro custa mais do que deixar passar uma de fora).
Não chama o script do estudo: ele só lê a suíte de lá e sobrescreve a
configuração do estudo.

Calibrado e avaliado nos mesmos dados, o falso bloqueio dá zero por construção.
Por isso calibra só em `selecao` e avalia só em `confirmacao` (ou deixando um
documento de fora por vez). Só biblioteca padrão.
"""
from __future__ import annotations

from collections import defaultdict

MARGIN = 0.02


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def choose_threshold(positives: list[float], negatives: list[float]) -> dict:
    """Positivas: similaridade do primeiro trecho das perguntas de dentro (têm de passar); negativas: as de fora."""
    if not positives or not negatives:
        raise ValueError("precisa de perguntas de dentro e de fora para calibrar")
    grid = sorted({round(v, 3) for v in positives + negatives} | {0.0})
    rows = []
    for threshold in grid:
        tp = sum(v >= threshold for v in positives)
        fp = sum(v >= threshold for v in negatives)
        rows.append({"threshold": threshold, "f1": round(_f1(tp, fp, len(positives) - tp), 4)})
    if min(positives) > max(negatives):
        return {"threshold": round((min(positives) + max(negatives)) / 2, 3), "rule": "ponto médio da folga",
                "separable": True}
    safe = [row for row in rows if row["threshold"] <= min(positives) - MARGIN]
    if safe:
        best = max(row["f1"] for row in safe)
        chosen = max((row for row in safe if row["f1"] == best), key=lambda row: row["threshold"])
        return {**chosen, "rule": f"maior F1 com margem de {MARGIN:.2f} abaixo do menor positivo", "separable": False}
    best = max(row["f1"] for row in rows)
    chosen = max((row for row in rows if row["f1"] == best), key=lambda row: row["threshold"])
    return {**chosen, "rule": "maior limiar com F1 máximo (sem margem possível)", "separable": False}


def evaluate(threshold: float, positives: list[float], negatives: list[float]) -> dict:
    """Abstenção correta (de fora barrada) e falso bloqueio (de dentro barrada) com o limiar dado."""
    return {
        "threshold": threshold,
        "correct_abstention": sum(v < threshold for v in negatives) / len(negatives) if negatives else None,
        "false_block": sum(v < threshold for v in positives) / len(positives) if positives else None,
        "false_block_count": sum(v < threshold for v in positives),
    }


def leave_one_document_out(inside: list[tuple[str, float]], outside: list[float]) -> dict:
    """Validação cruzada: calibra sem um documento e avalia nas perguntas dele, um documento por vez.

    `inside`: (documento, similaridade) das perguntas de dentro. As de fora entram todas na calibração e na
    avaliação (não têm documento para deixar de fora).
    """
    by_doc: dict[str, list[float]] = defaultdict(list)
    for doc, value in inside:
        by_doc[doc].append(value)
    blocked, total, thresholds = 0, 0, []
    for doc, values in by_doc.items():
        rest = [v for d, vs in by_doc.items() if d != doc for v in vs]
        threshold = choose_threshold(rest, outside)["threshold"]
        thresholds.append(threshold)
        blocked += sum(v < threshold for v in values)
        total += len(values)
    return {"false_block": blocked / total if total else None, "false_block_count": blocked,
            "thresholds": {"min": min(thresholds), "max": max(thresholds)} if thresholds else None}
