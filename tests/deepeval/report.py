"""Gera o painel HTML autocontido de uma rodada do DeepEval.

Lê só o JSON produzido por run_and_export.py — nada aqui é específico de uma
rodada ou de um dataset em particular, então continua funcionando conforme
goldens/dataset.json crescer ou mudar. Sem dependência de rede além das
fontes do Google Fonts (funciona offline, só sem a tipografia customizada).
"""
from __future__ import annotations

import html
from typing import Any

from app.rag.generation import NOT_FOUND_MARKER

METRIC_ORDER = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Precision",
    "Contextual Recall",
    "Contextual Relevancy",
]
METRIC_ABBR = {
    "Faithfulness": "Faith",
    "Answer Relevancy": "AnsRel",
    "Contextual Precision": "CtxPrec",
    "Contextual Recall": "CtxRec",
    "Contextual Relevancy": "CtxRel",
}


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _fmt(score: float) -> str:
    return f"{score:.2f}"


def _metric_order_key(metric: dict[str, Any]) -> int:
    try:
        return METRIC_ORDER.index(metric["name"])
    except ValueError:
        return len(METRIC_ORDER)


def _is_refusal(actual_output: str) -> bool:
    return NOT_FOUND_MARKER.lower() in actual_output.lower()


def _status_and_display(md: dict[str, Any]) -> tuple[str, str]:
    """(classe-css, texto) pra uma métrica — cobre o caso de erro do juiz
    (ignore_errors=True em run_and_export.py: score/success ficam None em vez
    de derrubar a rodada inteira por uma chamada malformada do juiz local)."""
    if md.get("error"):
        return "error", "ERRO"
    if md.get("score") is None:
        return "error", "—"
    return ("pass" if md["success"] else "fail"), _fmt(md["score"])


def _render_summary_tiles(data: dict[str, Any]) -> str:
    threshold = data["threshold"]
    tick_pct = round(threshold * 100)
    tiles = []
    for m in METRIC_ORDER:
        if m not in data["metrics_avg"]:
            continue
        avg = data["metrics_avg"][m]
        status = "pass" if avg >= threshold else "fail"
        pct = round(avg * 100)
        tiles.append(f"""
      <div class="tile">
        <div class="tile-head">
          <span class="tile-name">{_esc(m)}</span>
          <span class="chip chip-{status}">{status.upper()}</span>
        </div>
        <div class="tile-score">{_fmt(avg)}</div>
        <div class="tile-bar" role="img" aria-label="{_esc(m)}: {_fmt(avg)} de 1.0, limiar {threshold}">
          <div class="tile-bar-fill tile-bar-{status}" style="width:{pct}%"></div>
          <div class="tile-bar-tick" style="left:{tick_pct}%"></div>
        </div>
        <div class="tile-foot"><span>0</span><span class="tile-threshold">limiar {threshold}</span><span>1</span></div>
      </div>""")
    return "\n".join(tiles)


def _render_grid(data: dict[str, Any]) -> str:
    rows = []
    for case in data["cases"]:
        name = case["name"] or ""
        cells = []
        for m in METRIC_ORDER:
            md = next((x for x in case["metrics"] if x["name"] == m), None)
            if md is None:
                cells.append("<td>—</td>")
                continue
            status, display = _status_and_display(md)
            cells.append(
                f'<td><a class="cell cell-{status}" href="#case-{_esc(name)}" '
                f'title="{_esc(m)}: {display}">{display}</a></td>'
            )
        overall = "pass" if case["success"] else "fail"
        refusal_tag = ' <span class="tag-refusal">recusa</span>' if _is_refusal(case.get("actual_output") or "") else ""
        rows.append(f"""
        <tr>
          <th scope="row"><a href="#case-{_esc(name)}">{_esc(name)}</a>{refusal_tag}</th>
          {''.join(cells)}
          <td><span class="chip chip-{overall}">{overall.upper()}</span></td>
        </tr>""")
    header = "".join(f'<th scope="col">{_esc(METRIC_ABBR.get(m, m))}</th>' for m in METRIC_ORDER)
    return header, "\n".join(rows)


def _render_cases(data: dict[str, Any]) -> str:
    cards = []
    for i, case in enumerate(data["cases"], start=1):
        name = case["name"] or f"caso-{i}"
        metrics_sorted = sorted(case["metrics"], key=_metric_order_key)
        chips = "".join(
            f'<span class="mchip mchip-{_status_and_display(md)[0]}">'
            f'{_esc(METRIC_ABBR.get(md["name"], md["name"]))} {_status_and_display(md)[1]}</span>'
            for md in metrics_sorted
        )
        metric_rows = "".join(f"""
          <div class="metric-row">
            <div class="metric-row-head">
              <span class="metric-row-name">{_esc(md["name"])}</span>
              <span class="metric-row-score chip chip-{_status_and_display(md)[0]}">{_status_and_display(md)[1]}</span>
            </div>
            <p class="metric-reason">{_esc(md.get("error") or md.get("reason") or "—")}</p>
          </div>""" for md in metrics_sorted)

        refusal_note = ""
        if _is_refusal(case.get("actual_output") or ""):
            refusal_note = (
                '<p class="callout">Recusa: a resposta usou a frase-âncora de "não encontrado" '
                "(NOT_FOUND_MARKER). Se Contextual Precision/Recall estiverem altos aqui, o contexto "
                "certo foi recuperado mesmo assim — vale conferir se Faithfulness/Answer Relevancy "
                "estão penalizando uma recusa como se fosse uma resposta infiel, o que é um artefato "
                "da métrica, não um julgamento real da recusa.</p>"
            )

        expected = case.get("expected_output")
        expected_block = (
            f"""
            <div>
              <span class="eyebrow">Resposta esperada</span>
              <p class="case-text">{_esc(expected)}</p>
            </div>"""
            if expected
            else ""
        )

        cards.append(f"""
      <details class="case" id="case-{_esc(name)}">
        <summary>
          <span class="case-num">{i:02d}</span>
          <span class="case-title">{_esc(name)}</span>
          <span class="case-chips">{chips}</span>
        </summary>
        <div class="case-body">
          <p class="case-question"><span class="eyebrow">Pergunta</span>{_esc(case.get("input") or "")}</p>
          {refusal_note}
          <div class="case-cols">
            {expected_block}
            <div>
              <span class="eyebrow">Resposta gerada ({_esc(data["generation_model"])})</span>
              <p class="case-text">{_esc(case.get("actual_output") or "(geração falhou — ver erro nas métricas abaixo)")}</p>
            </div>
          </div>
          <span class="eyebrow">Métricas ({_esc(data["judge_model"])} como juiz)</span>
          <div class="metric-rows">{metric_rows}</div>
        </div>
      </details>""")
    return "\n".join(cards)


_CSS = """
  :root {
    --bg: #f4f6f7; --surface: #ffffff; --surface-2: #eef1f3;
    --border: #dde2e6; --border-soft: #e7ebee;
    --text-primary: #14181c; --text-secondary: #4c555d; --text-muted: #7c848c;
    --amber: #a8701c; --amber-soft: #f5ecdb; --cyan: #16809a;
    --good: #0ca30c; --good-soft: #e3f5e3;
    --critical: #d03b3b; --critical-soft: #fbe7e6;
    --font-sans: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
    --radius: 10px; --radius-sm: 6px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0b0d10; --surface: #14171b; --surface-2: #1a1e23;
      --border: #262b31; --border-soft: #20242a;
      --text-primary: #edeff1; --text-secondary: #a9b0b8; --text-muted: #6c737b;
      --amber: #e3a346; --amber-soft: #2b2216; --cyan: #5cc9dd;
      --good: #0ca30c; --good-soft: #102015;
      --critical: #d03b3b; --critical-soft: #2a1614;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0b0d10; --surface: #14171b; --surface-2: #1a1e23;
    --border: #262b31; --border-soft: #20242a;
    --text-primary: #edeff1; --text-secondary: #a9b0b8; --text-muted: #6c737b;
    --amber: #e3a346; --amber-soft: #2b2216; --cyan: #5cc9dd;
    --good: #0ca30c; --good-soft: #102015;
    --critical: #d03b3b; --critical-soft: #2a1614;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text-primary); font-family: var(--font-sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 40px 24px 96px; }
  .eyebrow { display: block; font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
  header.masthead { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 20px; padding-bottom: 24px; margin-bottom: 32px; border-bottom: 1px solid var(--border); }
  header.masthead h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }
  header.masthead .sub { color: var(--text-secondary); font-size: 0.95rem; max-width: 60ch; }
  .runinfo { font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-secondary); text-align: right; line-height: 1.7; }
  .runinfo b { color: var(--text-primary); font-weight: 600; }
  .runinfo .arrow { color: var(--cyan); padding: 0 2px; }
  section { margin-bottom: 48px; }
  h2 { font-family: var(--font-mono); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); font-weight: 600; margin: 0 0 16px; padding-bottom: 10px; border-bottom: 1px solid var(--border-soft); }
  h2 span.count { font-weight: 400; color: var(--text-muted); text-transform: none; letter-spacing: 0; }
  .tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  @media (max-width: 900px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 480px) { .tiles { grid-template-columns: 1fr; } }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .tile-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 6px; }
  .tile-name { font-size: 0.72rem; color: var(--text-secondary); font-weight: 500; }
  .tile-score { font-family: var(--font-mono); font-size: 1.9rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-bottom: 10px; }
  .tile-bar { position: relative; height: 6px; background: var(--surface-2); border-radius: 999px; overflow: visible; }
  .tile-bar-fill { height: 100%; border-radius: 999px; }
  .tile-bar-pass { background: var(--good); }
  .tile-bar-fail { background: var(--critical); }
  .tile-bar-tick { position: absolute; top: -3px; width: 2px; height: 12px; background: var(--text-muted); opacity: 0.6; }
  .tile-foot { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.66rem; color: var(--text-muted); margin-top: 6px; }
  .tile-threshold { color: var(--text-secondary); }
  .chip { font-family: var(--font-mono); font-size: 0.66rem; font-weight: 600; letter-spacing: 0.04em; padding: 2px 7px; border-radius: 999px; white-space: nowrap; }
  .chip-pass { color: var(--good); background: var(--good-soft); }
  .chip-fail { color: var(--critical); background: var(--critical-soft); }
  .chip-error { color: var(--amber); background: var(--amber-soft); }
  .stats-strip { display: flex; gap: 28px; flex-wrap: wrap; margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--border-soft); }
  .stat .n { font-family: var(--font-mono); font-size: 1.4rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat .l { font-size: 0.78rem; color: var(--text-muted); }
  .grid-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); }
  table.grid { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
  table.grid th, table.grid td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border-soft); }
  table.grid thead th { font-family: var(--font-mono); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); background: var(--surface-2); white-space: nowrap; }
  table.grid tbody th { font-weight: 500; white-space: nowrap; text-transform: capitalize; }
  table.grid tbody th a { color: var(--text-primary); text-decoration: none; }
  table.grid tbody th a:hover { color: var(--cyan); text-decoration: underline; }
  table.grid tbody tr:last-child td, table.grid tbody tr:last-child th { border-bottom: none; }
  table.grid tbody tr:hover { background: var(--surface-2); }
  .cell { display: inline-block; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 0.82rem; text-decoration: none; padding: 2px 6px; border-radius: var(--radius-sm); }
  .cell-pass { color: var(--good); }
  .cell-fail { color: var(--critical); background: var(--critical-soft); }
  .cell-error { color: var(--amber); background: var(--amber-soft); }
  .tag-refusal { font-family: var(--font-mono); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--amber); border: 1px solid currentColor; border-radius: 999px; padding: 1px 6px; margin-left: 6px; }
  .case { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 10px; }
  .case + .case { margin-top: 10px; }
  .case summary { list-style: none; cursor: pointer; display: flex; align-items: center; gap: 14px; padding: 14px 16px; }
  .case summary::-webkit-details-marker { display: none; }
  .case summary::before { content: "\\25b8"; color: var(--text-muted); font-size: 0.75rem; transition: transform 0.15s ease; flex-shrink: 0; }
  .case[open] summary::before { transform: rotate(90deg); }
  .case-num { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); flex-shrink: 0; }
  .case-title { font-weight: 600; flex-shrink: 0; margin-right: auto; text-transform: capitalize; }
  .case-chips { display: flex; gap: 5px; flex-wrap: wrap; justify-content: flex-end; }
  .mchip { font-family: var(--font-mono); font-size: 0.64rem; font-variant-numeric: tabular-nums; padding: 2px 6px; border-radius: var(--radius-sm); white-space: nowrap; }
  .mchip-pass { color: var(--good); background: var(--good-soft); }
  .mchip-fail { color: var(--critical); background: var(--critical-soft); }
  .mchip-error { color: var(--amber); background: var(--amber-soft); }
  .case-body { padding: 4px 20px 22px; border-top: 1px solid var(--border-soft); }
  .case-question { margin: 16px 0 14px; font-size: 1rem; font-weight: 500; }
  .case-question .eyebrow { margin-bottom: 4px; }
  .callout { background: var(--surface-2); border-left: 3px solid var(--amber); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 10px 14px; font-size: 0.86rem; color: var(--text-secondary); margin: 0 0 18px; }
  .case-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 22px; }
  @media (max-width: 700px) { .case-cols { grid-template-columns: 1fr; } }
  .case-text { font-size: 0.88rem; color: var(--text-secondary); margin: 0; max-width: 65ch; }
  .metric-rows { display: flex; flex-direction: column; gap: 12px; margin-top: 10px; }
  .metric-row { background: var(--surface-2); border-radius: var(--radius-sm); padding: 10px 14px; }
  .metric-row-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .metric-row-name { font-size: 0.82rem; font-weight: 600; }
  .metric-row-score { font-variant-numeric: tabular-nums; }
  .metric-reason { margin: 0; font-size: 0.82rem; color: var(--text-secondary); max-width: 75ch; }
  footer { margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--border-soft); font-size: 0.8rem; color: var(--text-muted); }
  footer code { font-family: var(--font-mono); background: var(--surface-2); padding: 1px 5px; border-radius: 4px; font-size: 0.78rem; }
"""


def render_html(data: dict[str, Any], source_label: str = "") -> str:
    """Renderiza o painel HTML autocontido a partir do dict carregado de um results/*.json."""
    grid_header, grid_rows = _render_grid(data)
    overall_pass_n = sum(1 for c in data["cases"] if c["success"])
    faithfulness_avg = data["metrics_avg"].get("Faithfulness")
    generated_at = (data.get("generated_at") or "")[:19].replace("T", " ")

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Painel DeepEval</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div>
      <h1>Avaliação de qualidade do RAG — DeepEval</h1>
      <div class="sub">{data.get("n_cases", len(data["cases"]))} perguntas do golden dataset, pipeline real (retrieval + geração), 5 métricas RAG julgadas por um LLM local via Ollama.</div>
    </div>
    <div class="runinfo">
      gerador <b>{_esc(data["generation_model"])}</b><br>
      juiz <b>{_esc(data["judge_model"])}</b><br>
      limiar <b>{data["threshold"]}</b> <span class="arrow">·</span> {data.get("n_cases", len(data["cases"]))} casos<br>
      {_esc(generated_at)} UTC
    </div>
  </header>

  <section>
    <h2>Médias por métrica</h2>
    <div class="tiles">{_render_summary_tiles(data)}
    </div>
    <div class="stats-strip">
      <div class="stat"><div class="n">{overall_pass_n}/{data.get("n_cases", len(data["cases"]))}</div><div class="l">casos com as 5 métricas acima do limiar</div></div>
      {f'<div class="stat"><div class="n">{_fmt(faithfulness_avg)}</div><div class="l">Faithfulness média (o número em GET /stats)</div></div>' if faithfulness_avg is not None else ''}
    </div>
  </section>

  <section>
    <h2>Todos os casos <span class="count">— clique num score pra abrir o caso</span></h2>
    <div class="grid-scroll">
      <table class="grid">
        <thead><tr><th scope="col">Caso</th>{grid_header}<th scope="col">Geral</th></tr></thead>
        <tbody>{grid_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Detalhe por caso</h2>
    {_render_cases(data)}
  </section>

  <footer>
    Gerado a partir de <code>{_esc(source_label)}</code>. Reproduzir com <code>python tests/deepeval/run_and_export.py</code> — ver <code>tests/deepeval/README.md</code>.
  </footer>
</div>
</body>
</html>
"""
