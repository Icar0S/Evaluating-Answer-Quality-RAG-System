"""Estatística do estudo — as regras do §7.7, fixadas antes da coleta.

    5 oráculos, veredito binário pareado -> Cochran's Q -> McNemar + Holm
    5 camadas, MS_coerente               -> Kruskal-Wallis -> Dunn + Holm
    Concordância multi-oráculo           -> Fleiss' kappa
    Concordância intra-avaliador         -> Cohen's kappa
    Toda proporção                       -> IC de Wilson 95%
    Toda diferença de média              -> IC bootstrap 95%
    alpha = 0,05; nenhum resultado sem IC; tamanho de efeito junto do p-valor.

Implementado aqui em vez de importado de statsmodels por uma razão prática: o
pacote de replicação fica com menos superfície de dependência, e cada teste pode
ser lido e conferido por um revisor em poucas linhas. Kruskal-Wallis e as
distribuições vêm do scipy, que já é dependência do ecossistema científico.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

ALPHA = 0.05


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}; {self.high:.3f}]"

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_name: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "test": self.name,
            "statistic": round(self.statistic, 4),
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_name": self.effect_name,
            "detail": self.detail,
        }


# ------------------------------------------------------------------ proporções


def wilson_ci(successes: int, total: int, alpha: float = ALPHA) -> Interval:
    """IC de Wilson para proporção.

    Escolhido em vez do intervalo normal (Wald) porque as taxas deste estudo são
    de amostra pequena (18 mutantes) e frequentemente próximas de 0 ou de 1 —
    exatamente onde o Wald produz limites fora de [0, 1].
    """
    if total == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return Interval(p, max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = ALPHA,
    seed: int = 20261213,
) -> Interval:
    """IC percentílico por bootstrap, com semente fixa para ser reproduzível."""
    values = np.asarray(list(data), dtype=float)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    estimates = np.apply_along_axis(statistic, 1, samples)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(values)), float(low), float(high))


# ------------------------------------------------- comparação de classificação


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC pela estatística de Mann-Whitney (equivalente e sem precisar varrer limiares)."""
    scores_array = np.asarray(list(scores), dtype=float)
    labels_array = np.asarray(list(labels), dtype=int)
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([positives, negatives]))
    rank_sum_positives = ranks[: positives.size].sum()
    return float(
        (rank_sum_positives - positives.size * (positives.size + 1) / 2)
        / (positives.size * negatives.size)
    )


# ---------------------------------------------------- vereditos binários pareados


def cochran_q(matrix: np.ndarray) -> TestResult:
    """Cochran's Q: k tratamentos pareados, resposta binária. matrix = itens x tratamentos."""
    data = np.asarray(matrix, dtype=int)
    n_items, k = data.shape
    if k < 2:
        raise ValueError("Cochran's Q exige ao menos 2 tratamentos")

    column_totals = data.sum(axis=0)
    row_totals = data.sum(axis=1)
    grand_total = data.sum()

    numerator = (k - 1) * (k * np.sum(column_totals**2) - grand_total**2)
    denominator = k * grand_total - np.sum(row_totals**2)
    if denominator == 0:
        # Todo item com a mesma resposta em todos os tratamentos: sem variação
        # entre oráculos, Q é indefinido. Reportar p=1 é mais informativo do que
        # quebrar — e é o resultado substantivo (nenhuma diferença detectável).
        return TestResult("Cochran's Q", 0.0, 1.0, detail=f"n={n_items}, k={k}, sem variação intra-item")

    q = numerator / denominator
    p_value = float(stats.chi2.sf(q, k - 1))
    return TestResult("Cochran's Q", float(q), p_value, detail=f"n={n_items}, k={k}, gl={k - 1}")


def mcnemar(a: Sequence[int], b: Sequence[int], exact: bool = True) -> TestResult:
    """McNemar pareado entre dois vetores binários. Exato (binomial) por padrão.

    Com 18 x |S| observações e discordâncias tipicamente poucas, a aproximação
    qui-quadrado é ruim; o teste binomial exato não tem esse problema.
    """
    first = np.asarray(list(a), dtype=int)
    second = np.asarray(list(b), dtype=int)
    b_count = int(np.sum((first == 1) & (second == 0)))
    c_count = int(np.sum((first == 0) & (second == 1)))
    discordant = b_count + c_count

    if discordant == 0:
        return TestResult("McNemar", 0.0, 1.0, 0.0, "odds ratio", "sem discordância")

    if exact:
        p_value = float(stats.binomtest(b_count, discordant, 0.5).pvalue)
        statistic = float(b_count)
    else:
        statistic = float((abs(b_count - c_count) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, 1))

    odds_ratio = b_count / c_count if c_count else float("inf")
    return TestResult(
        "McNemar",
        statistic,
        p_value,
        odds_ratio,
        "odds ratio (b/c)",
        f"b={b_count}, c={c_count}",
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Correção de Holm-Bonferroni. Devolve p ajustados na ordem de entrada."""
    values = list(p_values)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted


# -------------------------------------------------------- comparação de grupos


def kruskal_wallis(groups: Sequence[Sequence[float]]) -> TestResult:
    """Kruskal-Wallis entre k grupos independentes, com eta-quadrado como efeito."""
    clean = [np.asarray(list(g), dtype=float) for g in groups if len(g) > 0]
    if len(clean) < 2:
        return TestResult("Kruskal-Wallis", float("nan"), float("nan"), detail="menos de 2 grupos com dados")

    statistic, p_value = stats.kruskal(*clean)
    n_total = sum(len(g) for g in clean)
    k = len(clean)
    # eta² = (H - k + 1) / (n - k), Tomczak & Tomczak (2014).
    eta_squared = (statistic - k + 1) / (n_total - k) if n_total > k else float("nan")
    return TestResult(
        "Kruskal-Wallis",
        float(statistic),
        float(p_value),
        float(eta_squared),
        "eta quadrado",
        f"k={k}, n={n_total}",
    )


def dunn(groups: dict[str, Sequence[float]]) -> list[dict[str, object]]:
    """Dunn post-hoc com ajuste de Holm, sobre os postos da amostra combinada."""
    names = [name for name, values in groups.items() if len(values) > 0]
    data = [np.asarray(list(groups[name]), dtype=float) for name in names]
    sizes = [len(values) for values in data]
    combined = np.concatenate(data)
    n_total = combined.size
    ranks = stats.rankdata(combined)

    mean_ranks: dict[str, float] = {}
    cursor = 0
    for name, size in zip(names, sizes):
        mean_ranks[name] = float(ranks[cursor : cursor + size].mean())
        cursor += size

    # Correção para empates no denominador (Dunn, 1964).
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_correction = float(np.sum(tie_counts**3 - tie_counts))
    sigma_base = n_total * (n_total + 1) / 12 - tie_correction / (12 * (n_total - 1))

    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            size_i, size_j = sizes[i], sizes[j]
            standard_error = math.sqrt(sigma_base * (1 / size_i + 1 / size_j))
            z = (mean_ranks[names[i]] - mean_ranks[names[j]]) / standard_error if standard_error else 0.0
            p_value = float(2 * stats.norm.sf(abs(z)))
            comparisons.append(
                {
                    "group_a": names[i],
                    "group_b": names[j],
                    "z": round(float(z), 4),
                    "p_raw": p_value,
                    "mean_rank_a": round(mean_ranks[names[i]], 2),
                    "mean_rank_b": round(mean_ranks[names[j]], 2),
                }
            )
            raw_p.append(p_value)

    for comparison, adjusted in zip(comparisons, holm(raw_p)):
        comparison["p_holm"] = adjusted
        comparison["significant"] = adjusted < ALPHA
    return comparisons


# ------------------------------------------------------------- concordância


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa. counts = itens x categorias, cada linha somando o nº de avaliadores."""
    matrix = np.asarray(counts, dtype=float)
    n_items, _ = matrix.shape
    raters = matrix.sum(axis=1)
    if not np.allclose(raters, raters[0]):
        raise ValueError("Fleiss' kappa exige o mesmo número de avaliadores em todo item")
    n_raters = raters[0]
    if n_raters < 2:
        return float("nan")

    agreement = (np.sum(matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(agreement.mean())
    category_proportions = matrix.sum(axis=0) / (n_items * n_raters)
    p_expected = float(np.sum(category_proportions**2))
    if math.isclose(p_expected, 1.0):
        return 1.0
    return (p_bar - p_expected) / (1 - p_expected)


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa entre dois conjuntos de rótulos — usado para o kappa intra-avaliador."""
    first = list(a)
    second = list(b)
    if len(first) != len(second) or not first:
        return float("nan")
    categories = sorted(set(first) | set(second))
    index = {category: i for i, category in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)))
    for x, y in zip(first, second):
        matrix[index[x], index[y]] += 1

    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float(np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / total**2)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret_kappa(value: float) -> str:
    """Landis & Koch (1977) — a referência que a área usa, com os rótulos originais."""
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
