"""Placar: comparação pareada entre braços (plano-benchmark.md, §4 e §11).

Provisório: a §11 do plano, que fixa a regra de veredito, não chegou. Até lá, o
que vale está aqui e no docs/benchmark-status.md:

- Δ é a diferença das médias (braço − referência) nas perguntas medidas NOS
  DOIS braços. Pergunta medida só num lado não entra (seria perda diferencial)
  e é contada à parte.
- IC de 95% por bootstrap de documentos, não de perguntas: as perguntas de um
  mesmo documento erram juntas (§7).
- A margem de ruído de uma métrica é o percentil 95 de |Δ| nos pares A/A do E0
  (§8, E0).
- Veredito: |Δ| dentro da margem é "sem diferença relevante", seja qual for o
  IC. Acima da margem, "melhor" ou "pior" se o IC exclui o zero, e
  "inconclusivo" se não exclui.

Só biblioteca padrão.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261018


def _percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_delta(
    reference: dict[str, float],
    arm: dict[str, float],
    clusters: dict[str, str],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict | None:
    """Δ = média(arm) − média(reference) nas perguntas dos dois lados, com IC e P95 de |Δ| por bootstrap.

    `clusters`: pergunta → documento (pergunta sem documento é um cluster sozinho).
    """
    names = sorted(set(reference) & set(arm))
    if not names:
        return None
    by_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for name in names:
        by_cluster[clusters.get(name) or f"__{name}"].append((float(reference[name]), float(arm[name])))
    keys = list(by_cluster)
    point = sum(arm[n] - reference[n] for n in names) / len(names)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        pairs = [pair for _ in keys for pair in by_cluster[rng.choice(keys)]]
        deltas.append(sum(b - a for a, b in pairs) / len(pairs))
    deltas.sort()
    absolute = sorted(abs(d) for d in deltas)
    return {
        "delta": round(point, 4),
        "ci_low": round(_percentile(deltas, 0.025), 4),
        "ci_high": round(_percentile(deltas, 0.975), 4),
        "abs_p95": round(_percentile(absolute, 0.95), 4),
        "reference_mean": round(sum(reference[n] for n in names) / len(names), 4),
        "arm_mean": round(sum(arm[n] for n in names) / len(names), 4),
        "n": len(names),
        "clusters": len(keys),
        "one_side_only": len(set(reference) ^ set(arm)),
    }


def mcnemar_exact(reference: dict[str, bool], arm: dict[str, bool]) -> dict:
    """McNemar exato (binomial) nas perguntas dos dois lados: b = só a referência acerta, c = só o braço."""
    names = set(reference) & set(arm)
    b = sum(1 for n in names if reference[n] and not arm[n])
    c = sum(1 for n in names if arm[n] and not reference[n])
    discordant = b + c
    if discordant == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    tail = sum(math.comb(discordant, i) for i in range(min(b, c) + 1)) / 2**discordant
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni, na ordem de entrada (a família do E6 é a varredura inteira, §8)."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def verdict(delta: dict | None, margin: float | None, higher_is_better: bool = True) -> str:
    """A regra provisória do docstring do módulo."""
    if delta is None:
        return "não medido"
    if margin is None:
        return "sem margem do E0"
    if abs(delta["delta"]) <= margin:
        return "sem diferença relevante"
    if delta["ci_low"] > 0 or delta["ci_high"] < 0:
        better = (delta["delta"] > 0) == higher_is_better
        return "melhor" if better else "pior"
    return "inconclusivo"


def noise_margins(pairs: list[dict[str, dict | None]]) -> dict[str, float]:
    """Margem de cada métrica: o maior P95 de |Δ| entre os pares A/A dados ({métrica: delta})."""
    margins: dict[str, float] = {}
    for pair in pairs:
        for metric, delta in pair.items():
            if delta is not None:
                margins[metric] = max(margins.get(metric, 0.0), delta["abs_p95"])
    return margins
