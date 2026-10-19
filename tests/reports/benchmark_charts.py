"""Gráficos do relatório do benchmark (PNG), no estilo dos de report_charts.py.

Segue a skill de dataviz do projeto, com a mesma paleta validada:
- cor de identidade (braço, série) em ordem fixa da paleta categórica, nunca pelo ranking;
- cor de veredito só com o status da paleta (bom, crítico, aviso) e sempre com ícone + rótulo ao lado;
- marcas finas, grade e eixos em linha fina sólida, texto sempre na cor de texto;
- um eixo só por gráfico: métricas de unidades diferentes não dividem um gráfico.

Toda figura tem, no relatório, a tabela com os números logo abaixo ou acima (o PDF não tem hover).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from report_charts import (  # noqa: E402
    BASELINE,
    CATEGORICAL,
    CSS_PX,
    DPI,
    FONT,
    GRID,
    INK,
    INK_SECONDARY,
    MUTED,
    SURFACE,
    _ink_for,
    _rounded_hbar,
    _style,
)

WIDTH = 6.7  # polegadas: a largura útil do A4 com as margens do PDF
BAND = "#f0efec"  # a faixa do ruído: o cinza neutro da paleta
# Status da paleta (fixos); o rótulo e o ícone vão sempre juntos.
VERDICT_STYLE = {
    "melhor": ("#0ca30c", "✓"),
    "pior": ("#d03b3b", "✗"),
    "inconclusivo": ("#fab219", "?"),
    "sem diferença relevante": (MUTED, "≈"),
}


def _figure(height: float):
    plt.rc("font", **FONT)
    return plt.figure(figsize=(WIDTH, height), dpi=DPI, facecolor=SURFACE)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _grid_x(ax, ticks) -> None:
    for tick in ticks:
        ax.axvline(tick, color=GRID, linewidth=0.75, zorder=1)


def _grid_y(ax, ticks) -> None:
    for tick in ticks:
        ax.axhline(tick, color=GRID, linewidth=0.75, zorder=1)


def _legend(fig, ax, handles, labels, ncol: int | None = None) -> None:
    left = ax.get_position().x0
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left, 1.0), ncol=ncol or min(len(labels), 4),
               frameon=False, fontsize=FONT["size"], labelcolor=INK, handlelength=1.4, handleheight=0.8,
               borderaxespad=0.3, columnspacing=1.6)


def _comma(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _wrap(label: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(label, width)) or label


def _label_inches(labels: list[str]) -> float:
    """A largura (polegadas) que os rótulos do eixo y pedem, a 8 pt na DejaVu Sans (~0,062 pol. por letra)."""
    longest = max((len(line) for label in labels for line in label.split("\n")), default=10)
    return min(0.25 + 0.062 * longest, 3.2)


def forest(rows: list[dict], path: Path, xlabel: str) -> Path:
    """A diferença de cada par, com o IC de 95% e a faixa do ruído (a margem) em volta do zero.

    rows: label, delta, lo, hi, margin (pode faltar), verdict, value_text — já na unidade do eixo.
    """
    labels = [_wrap(r["label"]) for r in rows]
    height = 0.75 + sum(0.36 if "\n" not in label else 0.46 for label in labels)
    fig = _figure(height)
    # A coluna da direita (Δ e veredito) pede a largura do texto mais longo dela.
    right_text = max(len(f"{r.get('value_text', '')}   ? {r.get('verdict', '')}") for r in rows)
    left, right = _label_inches(labels) / WIDTH, 1 - (0.2 + 0.058 * right_text) / WIDTH
    bottom, top = 0.45 / height, 1 - 0.12 / height
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    _style(ax)
    extent = [0.0] + [v for r in rows for v in (r.get("lo"), r.get("hi"), r.get("margin"))
                      if v is not None] + [-(r.get("margin") or 0) for r in rows]
    low, high = min(extent), max(extent)
    pad = (high - low) * 0.08 or 0.05
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ticks = [t for t in ax.get_xticks() if low - pad <= t <= high + pad]
    _grid_x(ax, ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_comma(t, 2 if (high - low) < 1 else 1) for t in ticks])
    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
    for i, row in enumerate(rows):
        margin = row.get("margin")
        if margin:
            ax.add_patch(Rectangle((-margin, i - 0.32), 2 * margin, 0.64, facecolor=BAND, edgecolor="none", zorder=1.5))
        if row.get("delta") is None:
            ax.text(0, i, "não medido", ha="center", va="center", fontsize=7, color=MUTED)
            continue
        color, icon = VERDICT_STYLE.get(row["verdict"], (BASELINE, "·"))
        ax.plot([row["lo"], row["hi"]], [i, i], color=INK_SECONDARY, linewidth=1.5, solid_capstyle="round", zorder=3)
        ax.scatter([row["delta"]], [i], s=46, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=4)
        ax.text(1.03, i, f"{row['value_text']}   {icon} {row['verdict']}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color=INK, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=7.5)
    return _save(fig, path)


def small_bars(panels: list[dict], path: Path) -> Path:
    """Um painel por métrica, uma barra por braço (uma série: a cor do slot 1), o valor na ponta.

    panels: title, bars [(rótulo, valor, texto)], xmax (opcional).
    """
    n_bars = max(len(p["bars"]) for p in panels)
    height = 0.75 + 0.3 * n_bars
    fig = _figure(height)
    label_w = _label_inches([b[0] for b in panels[0]["bars"]]) / WIDTH
    gap = 0.25 / WIDTH
    panel_w = (1 - label_w - 0.02 - gap * (len(panels) - 1)) / len(panels)
    bottom, top = 0.3 / height, 1 - 0.4 / height
    for j, panel in enumerate(panels):
        x0 = label_w + j * (panel_w + gap)
        ax = fig.add_axes((x0, bottom, panel_w * 0.82, top - bottom))
        _style(ax)
        xmax = panel.get("xmax") or max((v for _, v, _ in panel["bars"] if v is not None), default=1) or 1
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(panel["bars"]) - 0.5, -0.5)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        for i, (_, value, text) in enumerate(panel["bars"]):
            if value is None:
                ax.text(0, i, " n/d", va="center", fontsize=7, color=MUTED)
                continue
            _rounded_hbar(ax, i - 0.28, i + 0.28, value, CATEGORICAL[0])
            ax.text(value, i, f" {text}", va="center", ha="left", fontsize=7.5, color=INK, clip_on=False)
        ax.set_yticks(range(len(panel["bars"])))
        ax.set_yticklabels([b[0] for b in panel["bars"]] if j == 0 else [], color=INK_SECONDARY)
        ax.set_title(panel["title"], loc="left", fontsize=8, color=INK, pad=6)
    return _save(fig, path)


def grouped_bars(categories: list[str], series: list[tuple[str, list[float | None]]], path: Path,
                 xlabel: str) -> Path:
    """Barras horizontais agrupadas: uma categoria por linha, uma série por cor (no máximo 3), com legenda."""
    n = len(series)
    bar, gap_px = min(0.26, 0.7 / n), 2
    height = 0.85 + len(categories) * (0.16 + 0.17 * n)
    fig = _figure(height)
    left, right, bottom, top = _label_inches(categories) / WIDTH, 0.96, 0.42 / height, 1 - 0.38 / height
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    _style(ax)
    xmax = max((v for _, values in series for v in values if v is not None), default=1) * 1.08 or 1
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(categories), 0)
    gap = gap_px * CSS_PX * len(categories) / ax.get_window_extent().height
    ticks = [t for t in ax.get_xticks() if 0 <= t <= xmax]
    _grid_x(ax, ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_comma(t, 2) for t in ticks])
    block = n * bar + (n - 1) * gap
    for i in range(len(categories)):
        start = i + (1 - block) / 2
        for j, (_, values) in enumerate(series):
            y = start + j * (bar + gap)
            if values[i]:
                _rounded_hbar(ax, y, y + bar, values[i], CATEGORICAL[j])
            elif values[i] == 0:
                ax.text(0, y + bar / 2, " 0", va="center", fontsize=7, color=MUTED)  # zero, e não falta de dado
    ax.set_yticks([i + 0.5 for i in range(len(categories))])
    ax.set_yticklabels(categories, color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=7.5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=CATEGORICAL[j], edgecolor="none") for j in range(n)]
    _legend(fig, ax, handles, [name for name, _ in series])
    return _save(fig, path)


def lines(series: list[dict], path: Path, xlabel: str, ylabel: str, references: list[dict] | None = None,
          xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None,
          marks: list[dict] | None = None) -> Path:
    """Linhas de 2 px, uma cor por série (ordem fixa) e a legenda em cima. Sem rótulo no fim das linhas: elas
    terminam juntas (Lorenz, ROC), e rótulos empilhados desgrudam das linhas; a legenda dá a identidade.

    series: name, points [(x, y)]. references: name, points, label_at (opcional) — linhas de referência em cinza
    (igualdade, acaso, esperado), rotuladas no próprio traço. marks: x, y, text — pontos destacados.
    """
    height = 3.0
    fig = _figure(height)
    left, right, bottom, top = 0.75 / WIDTH, 0.97, 0.5 / height, 1 - 0.42 / height
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    _style(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.75)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    # As referências em cinza, em tons diferentes, e na legenda: rótulo em cima da curva colidia com as séries.
    ref_colors = [MUTED, BASELINE]
    for k, ref in enumerate(references or []):
        xs, ys = zip(*ref["points"])
        ax.plot(xs, ys, color=ref_colors[k % 2], linewidth=1.0, zorder=2)
    for j, s in enumerate(series):
        xs, ys = zip(*s["points"])
        ax.plot(xs, ys, color=CATEGORICAL[j], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    for mark in marks or []:
        ax.scatter([mark["x"]], [mark["y"]], s=46, color=INK, edgecolors=SURFACE, linewidths=1.6, zorder=6)
        ax.annotate(mark["text"], (mark["x"], mark["y"]), xytext=(-8, 10), textcoords="offset points", fontsize=7,
                    color=INK_SECONDARY, ha="right")
    _grid_y(ax, [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]])
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=7.5)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _comma(v, 1 if v % 1 else 0)
                                                                 if abs(v) < 100 else f"{v:,.0f}".replace(",", ".")))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _comma(v, 1)))
    handles = [Line2D([0], [0], color=CATEGORICAL[j], linewidth=2) for j in range(len(series))]
    handles += [Line2D([0], [0], color=ref_colors[k % 2], linewidth=1) for k in range(len(references or []))]
    _legend(fig, ax, handles, [s["name"] for s in series] + [r["name"] for r in references or []])
    return _save(fig, path)


def stacked_bars(rows: list[tuple[str, list[float]]], categories: list[str], path: Path, xlabel: str) -> Path:
    """Barras empilhadas horizontais (contagens), 2 px de superfície entre os segmentos, legenda em cima."""
    legend_rows = (len(categories) + 2) // 3
    height = 0.75 + 0.2 * legend_rows + 0.42 * len(rows)
    fig = _figure(height)
    left, right = _label_inches([r[0] for r in rows]) / WIDTH, 0.97
    bottom, top = 0.45 / height, 1 - (0.25 + 0.2 * legend_rows) / height
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    _style(ax)
    total = max(sum(values) for _, values in rows) or 1
    ax.set_xlim(0, total)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    gap = 2 * CSS_PX * total / ax.get_window_extent().width
    px_per_unit = ax.get_window_extent().width / total
    for i, (_, values) in enumerate(rows):
        x = 0.0
        for j, value in enumerate(values):
            if value <= 0:
                continue
            width = max(value - gap, 0)
            ax.add_patch(Rectangle((x, i - 0.3), width, 0.6, facecolor=CATEGORICAL[j], edgecolor="none", zorder=3))
            label = _comma(value, 0)
            if width * px_per_unit > 9 * CSS_PX * len(label):
                ax.text(x + width / 2, i, label, ha="center", va="center", fontsize=7,
                        color=_ink_for(to_rgb(CATEGORICAL[j])), zorder=4)
            x += value
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK_SECONDARY)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=7.5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=CATEGORICAL[j], edgecolor="none") for j in range(len(categories))]
    _legend(fig, ax, handles, categories, ncol=3)
    return _save(fig, path)


def scatter_panels(panels: list[dict], names: list[str], path: Path) -> Path:
    """Painéis custo × qualidade: um ponto por braço (a mesma cor em todos os painéis), rótulo ao lado do ponto.

    panels: title, xlabel, points {nome: (x, y)}, ylim (opcional).
    """
    height = 2.6
    fig = _figure(height)
    gap = 0.55 / WIDTH
    panel_w = (1 - 0.6 / WIDTH - 0.05 - gap * (len(panels) - 1)) / len(panels)
    bottom, top = 0.5 / height, 1 - 0.55 / height
    for j, panel in enumerate(panels):
        ax = fig.add_axes((0.6 / WIDTH + j * (panel_w + gap), bottom, panel_w, top - bottom))
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(BASELINE)
        xs = [p[0] for p in panel["points"].values()]
        span = (max(xs) - min(xs)) or max(xs) or 1
        ax.set_xlim(min(xs) - span * 0.6, max(xs) + span * 0.6)
        ax.set_ylim(*(panel.get("ylim") or (0, 1.05)))
        _grid_y(ax, [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]])
        for k, name in enumerate(names):
            if name not in panel["points"]:
                continue
            x, y = panel["points"][name]
            ax.scatter([x], [y], s=60, color=CATEGORICAL[k], edgecolors=SURFACE, linewidths=1.8, zorder=4)
            ax.annotate(name, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(panel["title"], loc="left", fontsize=8, color=INK, pad=6)
        ax.set_xlabel(panel["xlabel"], color=INK_SECONDARY, fontsize=7.5)
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _comma(v, 1)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _comma(v, 1)))
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CATEGORICAL[k], markeredgecolor=SURFACE,
                      markersize=7) for k in range(len(names))]
    fig.legend(handles, names, loc="upper right", ncol=len(names), frameon=False, fontsize=FONT["size"],
               labelcolor=INK, borderaxespad=0.3)
    return _save(fig, path)
