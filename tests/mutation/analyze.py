"""Gera todas as tabelas e figuras do artigo a partir dos logs. Nada de estado volátil.

Entra: results/runs.jsonl, results/verdicts.jsonl, results/coherence.jsonl.
Sai:   results/tables/*.csv + *.tex, results/figures/*.png + *.pdf, results/cost.jsonl.

Operacionalização das definições do §7.1 — o ponto que mais merece atenção de um
revisor, porque o protocolo define duas coisas com o mesmo nome "MS":

    taxa_de_morte(operador, oráculo) = casos de S que reprovam / |S|
    morte(operador, oráculo)         = taxa_de_morte > 0   (booleano)
    MS_bruto(oráculo)                = mortes / mutantes avaliados
    MS_coerente(oráculo)             = mortes coerentes / mutantes avaliados

O protocolo pede "MS_coerente por operador com IC de Wilson" (§7.3, passo 3). Com
um mutante por operador, MS por operador seria 0 ou 1 e o IC não diria nada. A
taxa sobre casos é a proporção que tem denominador |S| e IC interpretável; é ela
que entra na Tabela 1 e no Kruskal-Wallis entre camadas. MS_bruto e MS_coerente
ficam no nível do oráculo, como o §7.1 define, e é sobre eles que o swing da RQ2
é calculado. As duas leituras são reportadas, nunca misturadas.

S é calculado por oráculo e a partir do BASELINE apenas. Um caso instável no
sistema não-mutado sai do denominador — é a síntese metodológica do artigo (§7.1).

Uso:
    python -m tests.mutation.analyze
    python -m tests.mutation.analyze --level L2 --primary-oracle O5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.mutation import stats as study_stats
from tests.mutation.operators.apply import load_catalog
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.loader import CLASS_QUOTAS, SUITE_SIZE, load_cases

logger = get_logger("mutation.analyze")

BASELINE_ID = "baseline"
ORACLE_ORDER = ("O1", "O2", "O3", "O4", "O5")
LAYER_ORDER = ("corpus", "chunking", "index", "retrieval", "prompt")

# Paleta validada (checks de CVD, banda de luminosidade, piso de croma e
# contraste rodados antes de entrar aqui). Figuras de uma série só usam uma cor:
# a identidade vem do eixo, não do matiz — o que também sobrevive à impressão em
# tons de cinza, que é como parte dos leitores do IST vai ler.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = ("o", "s", "^", "D", "v")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#fcfcfb"
GRID = "#e5e5e1"

# Desligado por --skip-figures: permite conferir as tabelas num ambiente sem
# matplotlib (CI, servidor sem display) sem que a analise inteira falhe.
FIGURES_ENABLED = True


# --------------------------------------------------------------------- dados


@dataclass
class OperatorRow:
    operator: str
    layer: str
    failure_point: str | None
    cuttable: bool
    oracle: str
    failing_cases: int
    evaluable_cases: int
    kill_rate: study_stats.Interval
    killed: bool
    coherent_kill: bool
    coherent_cases: int
    codes: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    runs: list[dict]
    verdicts: list[dict]
    coherence: dict[tuple[str, str], dict]
    level: str

    @property
    def mutant_ids(self) -> list[str]:
        return sorted({v["mutant_id"] for v in self.verdicts if v["mutant_id"] != BASELINE_ID})


def load_dataset(level: str) -> Dataset:
    runs = jsonl.read(paths.RUNS_JSONL)
    verdicts = [v for v in jsonl.read(paths.VERDICTS_JSONL) if v.get("level", "L2") == level]
    if not verdicts:
        raise SystemExit(
            f"Nenhum veredito no nível {level} em {paths.VERDICTS_JSONL.name}. "
            "Rode run_campaign.py e depois evaluate.py."
        )
    coherence = {
        (record["mutant_id"], record["case_id"]): record
        for record in jsonl.read(paths.COHERENCE_JSONL)
    }
    return Dataset(runs=runs, verdicts=verdicts, coherence=coherence, level=level)


def evaluable_sets(dataset: Dataset) -> dict[str, set[str]]:
    """S por oráculo, definido só pelo baseline (§7.1)."""
    sets: dict[str, set[str]] = defaultdict(set)
    for verdict in dataset.verdicts:
        if verdict["mutant_id"] != BASELINE_ID:
            continue
        if verdict.get("evaluable"):
            sets[verdict["oracle"]].add(verdict["case_id"])
    return sets


def baseline_instability(dataset: Dataset) -> dict[str, dict[str, Any]]:
    """Casos instáveis e casos estavelmente reprovados no baseline, por oráculo.

    Os dois saem de S, mas por razões diferentes e com leituras diferentes: o
    primeiro é ruído do modelo (vira flakiness), o segundo é um caso que o
    sistema não-mutado já não passa (vira nota sobre a suíte). Somá-los num
    número só esconderia qual dos dois problemas o estudo tem.
    """
    report: dict[str, dict[str, Any]] = {}
    for oracle in ORACLE_ORDER:
        rows = [v for v in dataset.verdicts if v["mutant_id"] == BASELINE_ID and v["oracle"] == oracle]
        if not rows:
            continue
        unstable = sorted(v["case_id"] for v in rows if not v.get("stable"))
        stable_fail = sorted(
            v["case_id"] for v in rows if v.get("stable") and v["verdict"] == "fail"
        )
        report[oracle] = {
            "cases_measured": len(rows),
            "unstable": unstable,
            "stable_fail": stable_fail,
            "flakiness": study_stats.wilson_ci(len(unstable), SUITE_SIZE),
        }
    return report


def operator_rows(dataset: Dataset, sets: dict[str, set[str]]) -> list[OperatorRow]:
    catalog = load_catalog()
    by_mutant_oracle: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for verdict in dataset.verdicts:
        if verdict["mutant_id"] == BASELINE_ID:
            continue
        by_mutant_oracle[(verdict["mutant_id"], verdict["oracle"])].append(verdict)

    rows: list[OperatorRow] = []
    for mutant_id in dataset.mutant_ids:
        operator = catalog.by_id.get(mutant_id)
        for oracle in ORACLE_ORDER:
            verdicts = by_mutant_oracle.get((mutant_id, oracle), [])
            if not verdicts:
                continue
            evaluable = sets.get(oracle, set())
            considered = [v for v in verdicts if v["case_id"] in evaluable]
            failing = [v for v in considered if v["verdict"] == "fail"]

            coherent_cases = 0
            codes: list[str] = []
            for verdict in failing:
                record = dataset.coherence.get((mutant_id, verdict["case_id"]))
                if record is None:
                    continue
                codes.append(record["coherence_code"])
                if record.get("coherent"):
                    coherent_cases += 1

            rows.append(
                OperatorRow(
                    operator=mutant_id,
                    layer=operator.layer if operator else "?",
                    failure_point=operator.failure_point if operator else None,
                    cuttable=bool(operator.cuttable) if operator else False,
                    oracle=oracle,
                    failing_cases=len(failing),
                    evaluable_cases=len(considered),
                    kill_rate=study_stats.wilson_ci(len(failing), len(considered)),
                    killed=bool(failing),
                    coherent_kill=coherent_cases > 0,
                    coherent_cases=coherent_cases,
                    codes=sorted(set(codes)),
                )
            )
    return rows


# ------------------------------------------------------------------- saída


def write_table(
    name: str,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
    caption: str,
    label: str | None = None,
) -> None:
    """Escreve a mesma tabela em CSV (para reanálise) e LaTeX (para o artigo)."""
    rows = [list(row) for row in rows]
    paths.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = paths.TABLES_DIR / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    def escape(value: Any) -> str:
        text = "" if value is None else str(value)
        for char, replacement in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            text = text.replace(char, replacement)
        return text

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{escape(caption)}}}",
        f"\\label{{tab:{label or name}}}",
        f"\\begin{{tabular}}{{{'l' * len(header)}}}",
        r"\toprule",
        " & ".join(escape(column) for column in header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(escape(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (paths.TABLES_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    logger.info("tabela %s (%d linhas)", name, len(rows))


def _style_axes(ax) -> None:
    """Grade e eixos recessivos: os dados ficam no primeiro plano, a moldura some."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_SECONDARY)


def save_figure(figure, name: str) -> None:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):  # PDF vetorial é o que a Elsevier prefere
        figure.savefig(
            paths.FIGURES_DIR / f"{name}.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor=SURFACE,
        )
    logger.info("figura %s", name)


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")  # sem display: os scripts rodam por linha de comando
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib não instalado. pip install -r tests/mutation/requirements.txt"
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
        }
    )
    return plt


# --------------------------------------------------------------------- RQ1


def rq1(dataset: Dataset, sets: dict[str, set[str]], rows: list[OperatorRow], primary: str) -> dict:
    instability = baseline_instability(dataset)

    write_table(
        "T0_conjunto_avaliavel",
        ["oráculo", "|S|", "casos medidos", "instáveis", "reprovam no baseline", "flakiness [IC95]"],
        [
            [
                oracle,
                len(sets.get(oracle, set())),
                data["cases_measured"],
                len(data["unstable"]),
                len(data["stable_fail"]),
                str(data["flakiness"]),
            ]
            for oracle, data in instability.items()
        ],
        "Conjunto avaliável e flakiness do baseline, por oráculo.",
    )

    primary_rows = [row for row in rows if row.oracle == primary]
    write_table(
        "T1_taxa_morte_por_operador",
        ["operador", "camada", "FP", "cortável", "casos que reprovam", "|S|",
         "taxa de morte [IC95]", "mortes coerentes", "códigos"],
        [
            [
                row.operator, row.layer, row.failure_point or "-", "sim" if row.cuttable else "não",
                row.failing_cases, row.evaluable_cases, str(row.kill_rate),
                row.coherent_cases, ";".join(row.codes) or "-",
            ]
            for row in sorted(primary_rows, key=lambda r: (LAYER_ORDER.index(r.layer) if r.layer in LAYER_ORDER else 9, r.operator))
        ],
        f"Taxa de morte por operador sob o oráculo {primary}, com IC de Wilson 95%.",
    )

    groups = {layer: [] for layer in LAYER_ORDER}
    for row in primary_rows:
        if row.layer in groups:
            groups[row.layer].append(row.kill_rate.point)

    kruskal = study_stats.kruskal_wallis([groups[layer] for layer in LAYER_ORDER])
    dunn_rows = study_stats.dunn({layer: values for layer, values in groups.items() if values})

    write_table(
        "T2_camadas",
        ["camada", "operadores", "taxa de morte média [IC95 bootstrap]", "mutantes mortos", "mortes coerentes"],
        [
            [
                layer,
                len(groups[layer]),
                str(study_stats.bootstrap_ci(groups[layer])) if groups[layer] else "-",
                sum(1 for row in primary_rows if row.layer == layer and row.killed),
                sum(1 for row in primary_rows if row.layer == layer and row.coherent_kill),
            ]
            for layer in LAYER_ORDER
        ],
        f"Adequação por camada sob {primary}. Kruskal-Wallis H={kruskal.statistic:.3f}, "
        f"p={kruskal.p_value:.4f}, eta²={kruskal.effect_size:.3f}.",
    )

    if dunn_rows:
        write_table(
            "T2b_dunn",
            ["camada A", "camada B", "z", "p bruto", "p Holm", "significativo"],
            [
                [r["group_a"], r["group_b"], r["z"], f"{r['p_raw']:.4f}",
                 f"{r['p_holm']:.4f}", "sim" if r["significant"] else "não"]
                for r in dunn_rows
            ],
            "Dunn post-hoc entre camadas, com ajuste de Holm (alpha = 0,05).",
        )

    survivors = [row for row in primary_rows if not row.killed]
    write_table(
        "T4_sobreviventes",
        ["operador", "camada", "FP", "defeito", "requisito de teste não coberto"],
        [[row.operator, row.layer, row.failure_point or "-", *_survivor_requirement(row.operator)]
         for row in survivors],
        f"Mutantes sobreviventes sob {primary} — a entrega prática do artigo (§7.3, passo 6).",
    )

    _figure_layers(primary_rows, primary)
    return {
        "primary_oracle": primary,
        "kruskal": kruskal.as_dict(),
        "dunn": dunn_rows,
        "survivors": [row.operator for row in survivors],
        "flakiness": {oracle: data["flakiness"].as_dict() for oracle, data in instability.items()},
    }


def _survivor_requirement(operator_id: str) -> tuple[str, str]:
    """Traduz um operador sobrevivente no requisito de teste que falta na suíte."""
    catalog = load_catalog()
    operator = catalog.by_id.get(operator_id)
    if operator is None:
        return ("?", "?")
    import yaml

    codebook = yaml.safe_load(paths.CODEBOOK.read_text(encoding="utf-8"))
    expected = [
        item["label"] for item in codebook["codes"] if operator_id in (item.get("expected_from") or [])
    ]
    requirement = (
        f"caso capaz de evidenciar: {'; '.join(expected)}"
        if expected
        else f"caso sensível a {operator.defect.lower()}"
    )
    return (operator.defect, requirement)


def _figure_layers(rows: list[OperatorRow], primary: str) -> None:
    if not FIGURES_ENABLED:
        return
    plt = _import_pyplot()
    ordered = sorted(
        rows, key=lambda r: (LAYER_ORDER.index(r.layer) if r.layer in LAYER_ORDER else 9, r.operator)
    )
    if not ordered:
        return

    figure, ax = plt.subplots(figsize=(7.2, 3.4))
    positions = list(range(len(ordered)))
    values = [row.kill_rate.point for row in ordered]
    lower = [row.kill_rate.point - row.kill_rate.low for row in ordered]
    upper = [row.kill_rate.high - row.kill_rate.point for row in ordered]

    # Uma série, uma cor: a camada é lida no eixo e nos separadores, não no matiz.
    ax.bar(positions, values, width=0.62, color=SERIES[0], zorder=2)
    ax.errorbar(
        positions, values, yerr=[lower, upper], fmt="none",
        ecolor=INK_SECONDARY, elinewidth=1.1, capsize=2.5, zorder=3,
    )

    boundary = 0
    for layer in LAYER_ORDER:
        count = sum(1 for row in ordered if row.layer == layer)
        if not count:
            continue
        middle = boundary + count / 2 - 0.5
        ax.text(middle, 1.07, layer, ha="center", va="bottom", fontsize=8.5, color=INK)
        boundary += count
        if boundary < len(ordered):
            ax.axvline(boundary - 0.5, color=GRID, linewidth=1.0, zorder=1)

    for position, row in zip(positions, ordered):
        if row.kill_rate.point > 0:
            # Acima do limite superior do IC, nao do topo da barra: no topo o
            # rotulo cai em cima da haste de erro e os dois ficam ilegiveis.
            ax.text(position, row.kill_rate.high + 0.025, f"{row.kill_rate.point:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
        else:
            ax.text(position, 0.02, "sobrevive", ha="center", va="bottom",
                    fontsize=7, color=INK_MUTED, rotation=90)

    ax.set_xticks(positions)
    ax.set_xticklabels([row.operator for row in ordered], fontsize=8)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("casos de S que reprovam")
    ax.set_title(f"Taxa de morte por operador, oráculo {primary} (IC de Wilson 95%)", loc="left", color=INK)
    _style_axes(ax)
    figure.tight_layout()
    save_figure(figure, "F1_taxa_morte_por_camada")
    plt.close(figure)


# --------------------------------------------------------------------- RQ2


def rq2(dataset: Dataset, sets: dict[str, set[str]], rows: list[OperatorRow]) -> dict:
    import numpy as np

    present = [oracle for oracle in ORACLE_ORDER if sets.get(oracle)]
    if len(present) < 2:
        logger.warning("Menos de 2 oráculos com S definido — RQ2 não é calculável.")
        return {}

    # Base pareada: pares (caso, mutante) avaliáveis em TODOS os oráculos. Sem a
    # interseção, Cochran's Q compararia oráculos sobre conjuntos diferentes de
    # itens, que é exatamente o viés que a RQ2 quer isolar.
    common_cases = set.intersection(*(sets[oracle] for oracle in present))
    verdict_lookup: dict[tuple[str, str, str], str] = {
        (v["mutant_id"], v["case_id"], v["oracle"]): v["verdict"] for v in dataset.verdicts
    }
    pairs = [
        (mutant_id, case_id)
        for mutant_id in dataset.mutant_ids
        for case_id in sorted(common_cases)
        if all((mutant_id, case_id, oracle) in verdict_lookup for oracle in present)
    ]
    if not pairs:
        logger.warning("Nenhum par (caso, mutante) avaliável em todos os oráculos.")
        return {}

    matrix = np.array(
        [[1 if verdict_lookup[(m, c, oracle)] == "fail" else 0 for oracle in present] for m, c in pairs]
    )

    cochran = study_stats.cochran_q(matrix)
    mcnemar_rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            result = study_stats.mcnemar(matrix[:, i], matrix[:, j])
            mcnemar_rows.append(
                {"a": present[i], "b": present[j], "statistic": result.statistic,
                 "p_raw": result.p_value, "effect": result.effect_size, "detail": result.detail}
            )
            raw_p.append(result.p_value)
    for row, adjusted in zip(mcnemar_rows, study_stats.holm(raw_p)):
        row["p_holm"] = adjusted
        row["significant"] = adjusted < study_stats.ALPHA

    # MS no nível do oráculo (§7.1): morte = ao menos um caso de S reprova.
    total_mutants = len(dataset.mutant_ids)
    ms: dict[str, dict[str, Any]] = {}
    for oracle in present:
        oracle_rows = [row for row in rows if row.oracle == oracle]
        killed = sum(1 for row in oracle_rows if row.killed)
        coherent = sum(1 for row in oracle_rows if row.coherent_kill)
        ms[oracle] = {
            "bruto": study_stats.wilson_ci(killed, total_mutants),
            "coerente": study_stats.wilson_ci(coherent, total_mutants),
            "killed": killed,
            "coherent": coherent,
        }

    swing = max(m["bruto"].point for m in ms.values()) - min(m["bruto"].point for m in ms.values())

    counts = np.array([[int((row == 0).sum()), int(row.sum())] for row in matrix])
    fleiss = study_stats.fleiss_kappa(counts)

    write_table(
        "T3_vies_de_oraculo",
        ["oráculo", "mutantes mortos", "MS_bruto [IC95]", "mortes coerentes", "MS_coerente [IC95]"],
        [[oracle, ms[oracle]["killed"], str(ms[oracle]["bruto"]),
          ms[oracle]["coherent"], str(ms[oracle]["coerente"])] for oracle in present],
        f"Escore de mutação por oráculo sobre artefatos idênticos. swing = {swing:.3f}. "
        f"Cochran's Q = {cochran.statistic:.3f}, p = {cochran.p_value:.4f}. "
        f"Fleiss' kappa = {fleiss:.3f} ({study_stats.interpret_kappa(fleiss)}).",
    )

    write_table(
        "T3b_mcnemar",
        ["oráculo A", "oráculo B", "b (A reprova, B não)", "p bruto", "p Holm", "significativo"],
        [[row["a"], row["b"], row["detail"], f"{row['p_raw']:.4f}",
          f"{row['p_holm']:.4f}", "sim" if row["significant"] else "não"] for row in mcnemar_rows],
        "McNemar par a par entre oráculos, com ajuste de Holm.",
    )

    kappa_matrix = []
    for i, oracle_a in enumerate(present):
        row = [oracle_a]
        for j, _oracle_b in enumerate(present):
            value = 1.0 if i == j else study_stats.cohen_kappa(matrix[:, i], matrix[:, j])
            row.append(f"{value:.3f}")
        kappa_matrix.append(row)
    write_table(
        "T3c_cohen_kappa",
        ["oráculo", *present],
        kappa_matrix,
        "Matriz de Cohen's kappa par a par entre oráculos.",
    )

    divergence = _divergence_by_class(pairs, matrix, present)
    _figure_oracles(present, ms, swing)
    _figure_divergence(divergence)

    return {
        "oracles": present,
        "pairs": len(pairs),
        "swing": swing,
        "cochran": cochran.as_dict(),
        "mcnemar": mcnemar_rows,
        "fleiss_kappa": fleiss,
        "ms": {oracle: {"bruto": ms[oracle]["bruto"].as_dict(),
                        "coerente": ms[oracle]["coerente"].as_dict()} for oracle in present},
        "divergence": divergence,
    }


def _divergence_by_class(pairs, matrix, present) -> list[dict[str, Any]]:
    """Onde os oráculos discordam, por classe de caso.

    A hipótese do §7.5 é que a divergência se concentre em abstenção e
    rastreabilidade, onde cosseno e juiz discordam por natureza. Esta tabela é o
    que confirma ou desmente isso.
    """
    case_class = {case.case_id: case.case_class for case in load_cases()}
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "divergent": 0})
    for (_mutant, case_id), row in zip(pairs, matrix):
        bucket = by_class[case_class.get(case_id, "?")]
        bucket["total"] += 1
        if 0 < int(row.sum()) < len(present):
            bucket["divergent"] += 1

    rows = []
    for class_name in list(CLASS_QUOTAS) + [c for c in by_class if c not in CLASS_QUOTAS]:
        bucket = by_class.get(class_name)
        if not bucket or not bucket["total"]:
            continue
        interval = study_stats.wilson_ci(bucket["divergent"], bucket["total"])
        rows.append({"class": class_name, "total": bucket["total"],
                     "divergent": bucket["divergent"], "rate": interval})

    write_table(
        "T3d_divergencia_por_classe",
        ["classe do caso", "pares", "pares com divergência", "taxa [IC95]"],
        [[r["class"], r["total"], r["divergent"], str(r["rate"])] for r in rows],
        "Divergência entre oráculos por classe de caso (§7.5, análise qualitativa).",
    )
    return [{**r, "rate": r["rate"].as_dict()} for r in rows]


def _figure_oracles(present, ms, swing) -> None:
    if not FIGURES_ENABLED:
        return
    plt = _import_pyplot()
    figure, ax = plt.subplots(figsize=(5.4, 3.2))
    positions = list(range(len(present)))
    values = [ms[oracle]["bruto"].point for oracle in present]
    lower = [ms[o]["bruto"].point - ms[o]["bruto"].low for o in present]
    upper = [ms[o]["bruto"].high - ms[o]["bruto"].point for o in present]

    ax.bar(positions, values, width=0.55, color=SERIES[0], zorder=2)
    ax.errorbar(positions, values, yerr=[lower, upper], fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1.1, capsize=3, zorder=3)
    for position, value, high in zip(positions, values, [ms[o]["bruto"].high for o in present]):
        ax.text(position, high + 0.03, f"{value:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK_SECONDARY)

    # swing como texto ancorado no canto, sem linha-guia: a guia teria de
    # atravessar as barras para ligar o maior ao menor, e o numero e a
    # metrica-titulo do artigo — precisa ser lido de imediato, nao rastreado.
    ax.text(
        -0.35, 1.22, f"swing = {swing:.2f}",
        fontsize=9, color=INK, va="top", ha="left", fontweight="bold",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("MS_bruto")
    ax.set_title("Escore de mutação por oráculo, mesmos artefatos", loc="left", color=INK)
    _style_axes(ax)
    figure.tight_layout()
    save_figure(figure, "F2_vies_de_oraculo")
    plt.close(figure)


def _figure_divergence(divergence: list[dict[str, Any]]) -> None:
    if not divergence or not FIGURES_ENABLED:
        return
    plt = _import_pyplot()
    figure, ax = plt.subplots(figsize=(6.0, 3.0))
    labels = [row["class"].replace("_", " ") for row in divergence]
    values = [row["rate"]["point"] for row in divergence]
    positions = list(range(len(labels)))

    ax.barh(positions, values, height=0.6, color=SERIES[0], zorder=2)
    for position, value in zip(positions, values):
        ax.text(value + 0.012, position, f"{value:.2f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values + [0.1]) * 1.25)
    ax.set_xlabel("pares (caso, mutante) com divergência entre oráculos")
    ax.set_title("Onde os oráculos discordam", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    figure.tight_layout()
    save_figure(figure, "F3_divergencia_por_classe")
    plt.close(figure)


# --------------------------------------------------------------------- RQ3


def rq3(dataset: Dataset, primary: str) -> dict:
    """Custo por nível de rigor e por oráculo, normalizado por mutante morto."""
    import yaml

    study = yaml.safe_load(paths.STUDY_CONFIG.read_text(encoding="utf-8"))
    levels = study["execution"]["levels"]
    all_verdicts = jsonl.read(paths.VERDICTS_JSONL)
    cost_rows: list[dict[str, Any]] = []

    for level, repetitions in levels.items():
        level_verdicts = [v for v in all_verdicts if v.get("level", "L2") == level]
        if not level_verdicts:
            continue

        # Geração: soma sobre as invocações que aquele nível de fato consome.
        generation = {"wall_s": 0.0, "tokens": 0, "gpu_s": 0.0, "wh": 0.0, "invocations": 0}
        for run in dataset.runs:
            if run["repetition"] > repetitions or run.get("error"):
                continue
            generation["wall_s"] += run.get("wall_ms", 0.0) / 1000.0
            generation["tokens"] += (run.get("tokens_in") or 0) + (run.get("tokens_out") or 0)
            generation["gpu_s"] += run.get("gpu_s") or 0.0
            generation["wh"] += run.get("wh") or 0.0
            generation["invocations"] += 1

        level_dataset = Dataset(dataset.runs, level_verdicts, dataset.coherence, level)
        level_sets = evaluable_sets(level_dataset)
        level_rows = operator_rows(level_dataset, level_sets)

        for oracle in ORACLE_ORDER:
            oracle_verdicts = [v for v in level_verdicts if v["oracle"] == oracle]
            if not oracle_verdicts:
                continue
            oracle_cost = {"wall_s": 0.0, "gpu_s": 0.0, "wh": 0.0}
            for verdict in oracle_verdicts:
                entry = verdict.get("cost") or {}
                oracle_cost["wall_s"] += (entry.get("wall_ms") or 0.0) / 1000.0
                oracle_cost["gpu_s"] += entry.get("gpu_s") or 0.0
                oracle_cost["wh"] += entry.get("wh") or 0.0

            oracle_rows_level = [row for row in level_rows if row.oracle == oracle]
            killed = sum(1 for row in oracle_rows_level if row.killed)
            coherent = sum(1 for row in oracle_rows_level if row.coherent_kill)

            total_wall = generation["wall_s"] + oracle_cost["wall_s"]
            total_gpu = generation["gpu_s"] + oracle_cost["gpu_s"]
            total_wh = generation["wh"] + oracle_cost["wh"]

            cost_rows.append(
                {
                    "level": level,
                    "n": repetitions,
                    "oracle": oracle,
                    "wall_s": round(total_wall, 2),
                    "tokens": generation["tokens"],
                    "gpu_s": round(total_gpu, 2),
                    "wh_est": round(total_wh, 4),
                    "mutants_killed": killed,
                    "coherent_kills": coherent,
                    "invocations": generation["invocations"],
                    "oracle_wall_s": round(oracle_cost["wall_s"], 2),
                    "generation_wall_s": round(generation["wall_s"], 2),
                    "wall_s_per_kill": round(total_wall / killed, 2) if killed else None,
                    "wh_per_kill": round(total_wh / killed, 4) if killed else None,
                    "wall_s_per_coherent_kill": round(total_wall / coherent, 2) if coherent else None,
                }
            )

    if not cost_rows:
        logger.warning("Sem dados de custo — RQ3 não calculada.")
        return {}

    jsonl.write_all(paths.COST_JSONL, cost_rows)
    write_table(
        "T5_custo_por_nivel",
        ["nível", "N", "oráculo", "parede (s)", "tokens", "GPU-s", "Wh",
         "mutantes mortos", "s/morte", "Wh/morte"],
        [
            [r["level"], r["n"], r["oracle"], r["wall_s"], r["tokens"], r["gpu_s"], r["wh_est"],
             r["mutants_killed"], r["wall_s_per_kill"] or "-", r["wh_per_kill"] or "-"]
            for r in cost_rows
        ],
        "Custo por nível de rigor e por oráculo, normalizado por mutante morto (RQ3).",
    )

    _figure_cost(cost_rows)
    return {"rows": cost_rows, "primary_oracle": primary}


def _figure_cost(cost_rows: list[dict[str, Any]]) -> None:
    """Ganho de deteccao x custo por morte, lado a lado — a comparacao da H3.

    Dois paineis com o mesmo eixo x (o nivel de rigor) em vez de um grafico com
    duas escalas no y. Eixo duplo aqui daria a impressao de que detecao e custo
    compartilham unidade, que e exatamente a leitura errada: a H3 afirma que um
    cresce mais rapido que o outro, e isso se le comparando as duas inclinacoes,
    nao sobrepondo-as.
    """
    if not FIGURES_ENABLED:
        return
    plt = _import_pyplot()
    by_oracle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cost_rows:
        by_oracle[row["oracle"]].append(row)

    level_order = ["L1", "L2", "L3"]
    present_levels = [lv for lv in level_order if any(r["level"] == lv for r in cost_rows)]
    if not present_levels:
        return
    x_of = {level: index for index, level in enumerate(present_levels)}

    figure, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    oracles = [o for o in ORACLE_ORDER if o in by_oracle]
    endpoints: list[tuple[float, float, str]] = []

    for index, oracle in enumerate(oracles):
        series = sorted(
            (r for r in by_oracle[oracle] if r["level"] in x_of), key=lambda r: x_of[r["level"]]
        )
        if not series:
            continue
        color = SERIES[index % len(SERIES)]
        marker = MARKERS[index % len(MARKERS)]
        x = [x_of[r["level"]] for r in series]

        ax_left.plot(x, [r["mutants_killed"] for r in series], color=color, linewidth=2.0,
                     marker=marker, markersize=5, label=oracle, zorder=2)

        per_kill = [(xi, r["wall_s_per_kill"]) for xi, r in zip(x, series) if r["wall_s_per_kill"]]
        if per_kill:
            xs, ys = zip(*per_kill)
            ax_right.plot(xs, ys, color=color, linewidth=2.0, marker=marker,
                          markersize=5, label=oracle, zorder=2)
            endpoints.append((ys[-1], xs[-1], oracle))

    # Rotulos diretos na ponta das linhas, afastados o suficiente para nao se
    # sobreporem: os oraculos baratos (O1, O2, O5) terminam praticamente no mesmo
    # ponto, e tres rotulos empilhados no mesmo pixel nao identificam nada.
    if endpoints:
        span = max(y for y, _, _ in endpoints) - min(y for y, _, _ in endpoints)
        gap = max(span, 1.0) * 0.06
        placed: list[float] = []
        for y, x, oracle in sorted(endpoints):
            target = y
            while any(abs(target - used) < gap for used in placed):
                target += gap
            placed.append(target)
            ax_right.annotate(
                oracle, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=INK_SECONDARY, va="center",
                annotation_clip=False,
            ) if abs(target - y) < 1e-9 else ax_right.annotate(
                oracle, xy=(x, y), xytext=(x + 0.06, target), textcoords="data",
                fontsize=8, color=INK_SECONDARY, va="center", annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8},
            )

    for ax, title, ylabel in (
        (ax_left, "Ganho: mutantes mortos", "mutantes mortos"),
        (ax_right, "Custo: segundos por morte", "s / mutante morto"),
    ):
        ax.set_xticks(list(x_of.values()))
        ax.set_xticklabels([f"{lv}\nN={_n_of(cost_rows, lv)}" for lv in present_levels], fontsize=8)
        # Folga extra a direita no painel de custo: e la que ficam os rotulos
        # diretos das linhas, e sem margem o ultimo deles sai cortado.
        right_margin = 0.85 if ax is ax_right else 0.25
        ax.set_xlim(-0.25, len(present_levels) - 1 + right_margin)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        _style_axes(ax)

    legend = ax_left.legend(frameon=False, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.suptitle("Retorno decrescente do rigor (RQ3)", x=0.01, ha="left", color=INK, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, "F4_custo_por_nivel")
    plt.close(figure)


def _n_of(cost_rows: list[dict[str, Any]], level: str) -> int:
    return next((row["n"] for row in cost_rows if row["level"] == level), 0)


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera tabelas e figuras do artigo.")
    parser.add_argument("--level", default="L2", choices=["L1", "L2", "L3"])
    parser.add_argument("--primary-oracle", default="O5", choices=list(ORACLE_ORDER))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--results-dir",
        help="analisa outro conjunto de resultados (ex.: results/synthetic, para validar a análise)",
    )
    args = parser.parse_args(argv)

    if args.results_dir:
        paths.use_results_dir(args.results_dir)
        logger.warning("Analisando %s — confira se estes são os resultados oficiais.", paths.RESULTS_DIR)
    paths.ensure_dirs()
    dataset = load_dataset(args.level)
    sets = evaluable_sets(dataset)
    rows = operator_rows(dataset, sets)

    if not dataset.coherence:
        logger.warning(
            "results/coherence.jsonl vazio: MS_coerente sai zerado. Rode "
            "`python -m tests.mutation.coherence --classify` antes da análise final (§7.4)."
        )

    if args.skip_figures:
        global FIGURES_ENABLED  # noqa: PLW0603 — modo texto, sem matplotlib
        FIGURES_ENABLED = False

    summary = {
        "level": args.level,
        "mutants": len(dataset.mutant_ids),
        "rq1": rq1(dataset, sets, rows, args.primary_oracle),
        "rq2": rq2(dataset, sets, rows),
        "rq3": rq3(dataset, args.primary_oracle),
    }
    (paths.RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    logger.info("Tabelas em %s", paths.TABLES_DIR)
    logger.info("Figuras em %s", paths.FIGURES_DIR)
    logger.info("Resumo em %s", paths.RESULTS_DIR / "summary.json")

    rq2_summary = summary.get("rq2") or {}
    if rq2_summary:
        logger.info(
            "swing = %.3f | Cochran's Q p = %.4f | Fleiss' kappa = %.3f",
            rq2_summary["swing"], rq2_summary["cochran"]["p_value"], rq2_summary["fleiss_kappa"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
