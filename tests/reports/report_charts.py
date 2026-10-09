"""Gráficos do relatório (PNG), usados no Markdown, no PDF e na planilha.

Segue a skill de dataviz do projeto (paleta de referência validada):
- cor por ENTIDADE e em ordem fixa: DeepEval no slot 1 (azul), RAGAS no slot
  2 (laranja), outro framework no slot seguinte; a cor nunca segue o ranking;
- heatmap em escala DIVERGENTE centrada no limiar: o trabalho do gráfico é
  dizer de que lado do limiar cada nota caiu (vermelho abaixo, azul acima,
  cinza neutro no limiar), não só "quanto";
- barras finas com ponta arredondada só na extremidade do dado, grade e eixos
  em linha fina sólida, texto sempre na cor de texto (nunca na cor da série);
- toda figura tem a tabela com os números logo abaixo no relatório.

Estático (PDF/planilha): não há camada de hover; a tabela é o equivalente
acessível.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

from report_data import DIMENSIONS, Run, dimension_stats, fmt  # noqa: E402

# Paleta de referência (skill dataviz, references/palette.md), modo claro.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DIVERGING_LOW = "#e34948"  # vermelho: abaixo do limiar
DIVERGING_MID = "#f0efec"  # cinza neutro: no limiar
DIVERGING_HIGH = "#2a78d6"  # azul: acima do limiar
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FRAMEWORK_SLOT = {"DeepEval": 0, "RAGAS": 1}

DPI = 200
CSS_PX = DPI / 96  # 1 px de tela na resolução da figura
FONT = {"family": "DejaVu Sans", "size": 8}


def run_colors(runs: list[Run]) -> list[str]:
    """Cor de cada rodada, na mesma ordem de `runs`.

    O framework tem slot fixo (a cor segue a entidade, não a posição). Um
    framework sem slot reservado, ou uma segunda rodada do mesmo framework,
    pega o primeiro slot livre fora dos reservados. Mais de 8 rodadas não cabe
    na paleta categórica — aí o certo é dividir o relatório, não inventar cor.
    """
    if len(runs) > len(CATEGORICAL):
        raise ValueError(f"no máximo {len(CATEGORICAL)} rodadas por relatório")
    reserved = set(FRAMEWORK_SLOT.values())
    used: set[int] = set()
    colors = []
    for run in runs:
        slot = FRAMEWORK_SLOT.get(run.framework)
        if slot is None or slot in used:
            free = [i for i in range(len(CATEGORICAL)) if i not in used and i not in reserved]
            slot = free[0] if free else next(i for i in range(len(CATEGORICAL)) if i not in used)
        used.add(slot)
        colors.append(CATEGORICAL[slot])
    return colors


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=FONT["size"])


def _rounded_hbar(ax, y_bottom: float, y_top: float, width: float, color: str, radius_px: float = 4) -> None:
    """Barra horizontal de 0 até `width` com a ponta do dado arredondada e a base reta.

    O raio é dado em px de tela e convertido para unidades de dado em cada eixo
    — sem isso o arredondamento sai elíptico, porque x e y têm escalas
    diferentes. Precisa dos limites e da posição dos eixos já definidos.
    """
    if width <= 0:
        return
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = min(radius_px * CSS_PX * (x1 - x0) / bbox.width, width)
    ry = min(radius_px * CSS_PX * abs(y1 - y0) / bbox.height, (y_top - y_bottom) / 2)
    verts = [
        (0, y_bottom),
        (width - rx, y_bottom),
        (width, y_bottom),
        (width, y_bottom + ry),
        (width, y_top - ry),
        (width, y_top),
        (width - rx, y_top),
        (0, y_top),
        (0, y_bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def means_chart(runs: list[Run], path: Path) -> Path:
    """Média por dimensão, uma barra por rodada, com o limiar marcado."""
    plt.rc("font", **FONT)
    colors = run_colors(runs)
    stats = {run.label: dimension_stats(run) for run in runs}
    n_runs = len(runs)

    group_height = 1.0
    bar = min(0.32, 0.72 / n_runs)
    fig_h = 0.55 + len(DIMENSIONS) * (0.22 + 0.2 * n_runs)
    fig = plt.figure(figsize=(6.7, fig_h), dpi=DPI, facecolor=SURFACE)
    left, right, bottom, top = 1.55 / 6.7, 0.97, 0.42 / fig_h, 1 - 0.42 / fig_h
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    _style(ax)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(DIMENSIONS) * group_height, 0)  # primeira dimensão no topo
    gap = 2 * CSS_PX * len(DIMENSIONS) * group_height / ax.get_window_extent().height  # 2 px entre barras vizinhas

    for tick in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axvline(tick, color=GRID, linewidth=0.75, zorder=1)
    thresholds = sorted({run.threshold for run in runs})
    for t in thresholds:
        ax.axvline(t, color=MUTED, linewidth=0.75, zorder=4)
        ax.text(t, 1.01, f"limiar {fmt(t)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color=INK_SECONDARY)

    block = n_runs * bar + (n_runs - 1) * gap
    for i, dim in enumerate(DIMENSIONS):
        start = i * group_height + (group_height - block) / 2
        for j, run in enumerate(runs):
            y_bottom = start + j * (bar + gap)
            mean = stats[run.label][dim.key].mean
            if mean is None:
                ax.text(0.01, y_bottom + bar / 2, "n/d", va="center", fontsize=7, color=MUTED)
                continue
            _rounded_hbar(ax, y_bottom, y_bottom + bar, mean, colors[j])

    ax.set_yticks([i * group_height + group_height / 2 for i in range(len(DIMENSIONS))])
    ax.set_yticklabels([d.label for d in DIMENSIONS], color=INK_SECONDARY)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels([fmt(x, 1) for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colors]
    fig.legend(handles, [r.label for r in runs], loc="upper left", bbox_to_anchor=(left, 1.0), ncol=min(n_runs, 3),
               frameon=False, fontsize=FONT["size"], labelcolor=INK, handlelength=1.0, handleheight=0.8,
               borderaxespad=0.3, columnspacing=1.6)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def framework_colors(frameworks: list[str]) -> dict[str, str]:
    """Cor de cada framework (para o histórico, onde a entidade é o framework, não a rodada)."""
    reserved = set(FRAMEWORK_SLOT.values())
    free = [i for i in range(len(CATEGORICAL)) if i not in reserved]
    colors = {}
    for name in frameworks:
        if name in colors:
            continue
        slot = FRAMEWORK_SLOT.get(name)
        colors[name] = CATEGORICAL[slot if slot is not None else free.pop(0)]
    return colors


def history_chart(runs: list[dict], path: Path, threshold: float = 0.7) -> Path | None:
    """Evolução da média de cada dimensão: um painel por dimensão, uma linha por framework.

    `runs` vem de report_history.unique_runs (uma entrada por rodada avaliada,
    com a data da avaliação). Eixo x é tempo de verdade, então rodadas no mesmo
    dia ficam juntas e um intervalo longo aparece longo. Sem pelo menos dois
    pontos não há evolução para mostrar: devolve None.
    """
    from datetime import datetime

    import matplotlib.dates as mdates

    points = [r for r in runs if r.get("run_generated_at") and any(v is not None for v in r["means"].values())]
    per_framework = {r["framework"]: sum(1 for p in points if p["framework"] == r["framework"]) for r in points}
    if not per_framework or max(per_framework.values()) < 2:
        return None  # nenhum framework com duas rodadas: não há linha, só pontos soltos
    plt.rc("font", **FONT)
    colors = framework_colors([r["framework"] for r in points])
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(6.7, 2.35), dpi=DPI, facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.17, wspace=0.12)
    for ax, dim in zip(axes, DIMENSIONS):
        _style(ax)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0,5", "1"])
        for y in (0.5, 1.0):
            ax.axhline(y, color=GRID, linewidth=0.75, zorder=1)
        ax.axhline(threshold, color=MUTED, linewidth=0.75, zorder=2)
        for framework, color in colors.items():
            series = [
                (datetime.fromisoformat(r["run_generated_at"]), r["means"].get(dim.key))
                for r in points
                if r["framework"] == framework and r["means"].get(dim.key) is not None
            ]
            if not series:
                continue
            xs, ys = zip(*series)
            ax.plot(xs, ys, color=color, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=5, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        ax.set_title(dim.short, fontsize=7.5, color=INK_SECONDARY, pad=4)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.margins(x=0.15)
    axes[0].text(0.0, threshold, f"limiar {fmt(threshold)}", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=6.5, color=INK_SECONDARY)
    handles = [plt.Line2D([0], [0], color=c, linewidth=1.5, marker="o", markersize=5, markerfacecolor=c,
                          markeredgecolor=SURFACE) for c in colors.values()]
    fig.legend(handles, list(colors), loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=len(colors), frameon=False,
               fontsize=FONT["size"], labelcolor=INK, handlelength=1.6, borderaxespad=0.3, columnspacing=1.6)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _diverging_cmap(threshold: float) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "limiar", [(0.0, DIVERGING_LOW), (threshold, DIVERGING_MID), (1.0, DIVERGING_HIGH)]
    )


def _luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink_for(fill: tuple[float, float, float]) -> str:
    """Branco ou tinta, o que tiver mais contraste (WCAG) com o fundo da célula."""
    lum = _luminance(fill)
    on_white = 1.05 / (lum + 0.05)
    on_ink = (lum + 0.05) / (_luminance(to_rgb(INK)) + 0.05)
    return "#ffffff" if on_white > on_ink else INK


def heatmap(run: Run, path: Path) -> Path:
    """Caso x dimensão, cor divergente em torno do limiar e a nota escrita na célula."""
    plt.rc("font", **FONT)
    cmap = _diverging_cmap(run.threshold)
    n_rows, n_cols = len(run.cases), len(DIMENSIONS)
    cell_w, cell_h = 0.78, 0.27
    label_w = 2.25
    fig_w = label_w + n_cols * cell_w + 0.2
    fig_h = 0.38 + n_rows * cell_h + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((label_w / fig_w, 0.55 / fig_h, n_cols * cell_w / fig_w, n_rows * cell_h / fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.axis("off")

    gap_x = 1 * CSS_PX / (cell_w * DPI)  # metade do vão de 2 px de cada lado
    gap_y = 1 * CSS_PX / (cell_h * DPI)
    for i, case in enumerate(run.cases):
        for j, dim in enumerate(DIMENSIONS):
            score = case.score(dim.key)
            x, y, w, h = j + gap_x, i + gap_y, 1 - 2 * gap_x, 1 - 2 * gap_y
            if not score.measured:
                ax.add_patch(Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=BASELINE, hatch="////",
                                       linewidth=0.5))
                ax.text(j + 0.5, i + 0.5, "n/d", ha="center", va="center", fontsize=7, color=INK_SECONDARY)
                continue
            fill = to_rgb(cmap(score.value))
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, fmt(score.value), ha="center", va="center", fontsize=7.5,
                    color=_ink_for(fill))
        label = case.name + ("  †" if case.abstained else "")
        ax.text(-0.08, i + 0.5, label, ha="right", va="center", fontsize=7.5, color=INK_SECONDARY)

    for j, dim in enumerate(DIMENSIONS):
        ax.text(j + 0.5, -0.25, dim.short, ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    # Escala (a legenda de cor): gradiente com 0, limiar e 1 marcados.
    cax = fig.add_axes((label_w / fig_w, 0.2 / fig_h, n_cols * cell_w / fig_w, 0.1 / fig_h))
    gradient = [[v / 200 for v in range(201)]]
    cax.imshow(gradient, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1), vmin=0, vmax=1)
    cax.set_yticks([])
    cax.set_xticks([0, run.threshold, 1])
    cax.set_xticklabels(["0", f"{fmt(run.threshold)} (limiar)", "1"], fontsize=7, color=INK_SECONDARY)
    cax.tick_params(length=0)
    for spine in cax.spines.values():
        spine.set_visible(False)

    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path
