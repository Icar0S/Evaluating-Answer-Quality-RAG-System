"""Estatística da validação do instrumento e do O6, em Python puro.

As contas são as mesmas de tests/mutation/stats.py (AUC por Mann-Whitney,
precisão/revocação/F1, Wilson, kappa de Cohen, rótulos de Landis & Koch), para
os números daqui serem lidos ao lado do tau.json e do analyze.py. Reescritas sem
numpy/scipy porque o venv desta camada não tem nenhum dos dois, e os testes
herméticos rodam no CI com o mínimo instalado. Um teste confere os resultados
contra valores calculados à mão.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Sequence

ALPHA = 0.05
# Mesma semente do IC de τ* em tests/mutation/calibration/calibrate.py.
BOOTSTRAP_SEED = 20261003


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_list(self, digits: int = 4) -> list[float | None]:
        return [None if math.isnan(v) else round(v, digits) for v in (self.point, self.low, self.high)]


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Postos 1..n com empate pela média (o `rankdata` padrão do scipy)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC pela estatística de Mann-Whitney; rótulo 1 = classe positiva."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return float("nan")
    ranks = _average_ranks(positives + negatives)
    rank_sum = sum(ranks[: len(positives)])
    return (rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))


def bootstrap_ci(
    n: int,
    statistic: Callable[[list[int]], float],
    n_boot: int = 2000,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """IC percentílico: reamostra ÍNDICES (com reposição) e recalcula `statistic`.

    Reamostrar índices, e não valores, é o que permite estatísticas sobre pares
    (nota, rótulo), como a AUC. Amostras em que a estatística é indefinida (NaN,
    ex.: uma classe só) saem da conta.
    """
    if n == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        value = statistic([rng.randrange(n) for _ in range(n)])
        if not math.isnan(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    estimates.sort()
    return _percentile(estimates, 100 * alpha / 2), _percentile(estimates, 100 * (1 - alpha / 2))


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentil com interpolação linear (o padrão do numpy)."""
    position = (len(sorted_values) - 1) * q / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def wilson_ci(successes: int, total: int, alpha: float = ALPHA) -> Interval:
    if total == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin))


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    first, second = list(a), list(b)
    if len(first) != len(second) or not first:
        return float("nan")
    categories = sorted(set(first) | set(second))
    total = len(first)
    observed = sum(1 for x, y in zip(first, second) if x == y) / total
    expected = sum((first.count(c) / total) * (second.count(c) / total) for c in categories)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret_kappa(value: float) -> str:
    """Landis & Koch (1977), os mesmos rótulos de tests/mutation/stats.py."""
    if math.isnan(value):
        return "indefinido"
    if value < 0.0:
        return "pior que o acaso"
    if value < 0.20:
        return "leve"
    if value < 0.40:
        return "razoável"
    if value < 0.60:
        return "moderada"
    if value < 0.80:
        return "substancial"
    return "quase perfeita"


def confusion(scores: Sequence[float], labels: Sequence[int], tau: float) -> dict[str, float]:
    """Classificação por limiar, com a convenção do tau.json: positivo = EQUIV, previsto positivo = nota >= τ."""
    tp = sum(1 for s, y in zip(scores, labels) if s >= tau and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= tau and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < tau and y == 1)
    tn = sum(1 for s, y in zip(scores, labels) if s < tau and y == 0)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(scores), 4) if scores else 0.0,
    }
