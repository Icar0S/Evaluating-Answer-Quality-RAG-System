"""O conteúdo do relatório, uma vez só, como lista de blocos.

O Markdown e o PDF renderizam ESTA lista (report_markdown.py, report_pdf.py),
então os dois dizem exatamente a mesma coisa. Texto dos blocos aceita duas
marcações inline: **negrito** e `código`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from report_data import (
    DIMENSION_BY_KEY,
    DIMENSIONS,
    NATIVE_METRIC,
    PROJECT_ROOT,
    Comparison,
    Run,
    Score,
    dimension_stats,
    fmt,
    pct,
    origin_stats,
    origins,
    run_highlights,
    signed,
)
from report_history import history_table_rows

MAX_DIVERGENCES = 20


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Para:
    text: str
    small: bool = False


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    widths: list[float]  # proporções da largura útil
    align: list[str] = field(default_factory=list)  # "l" | "r" | "c" por coluna
    caption: str | None = None
    small: bool = False


@dataclass
class Figure:
    path: Path
    caption: str
    width: float = 1.0  # fração da largura útil


@dataclass
class Callout:
    title: str
    items: list[str]


@dataclass
class PageBreak:
    pass


Block = Heading | Para | Bullets | Table | Figure | Callout | PageBreak


def score_cell(score: Score) -> str:
    if not score.applicable:
        return "N/A"
    if not score.measured:
        return "n/d"
    return f"{fmt(score.value)} {'✓' if score.passed else '✗'}"


def clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _run_table(runs: list[Run]) -> Table:
    rows = []
    for run in runs:
        stats = dimension_stats(run)
        measured = sum(s.n_measured for s in stats.values())
        # Denominador = notas que se aplicam: N/A não é nota que faltou.
        applicable = sum(s.n_cases for s in stats.values())
        rows.append(
            [
                f"**{run.name}**",
                run.when,
                run.generation_model or "—",
                run.judge_model or "—",
                fmt(run.threshold),
                pct(len(run.cases), run.n_cases_planned),
                pct(measured, applicable),
                f"`{_rel(run.source)}`",
            ]
        )
    return Table(
        headers=["Rodada", "Data", "Modelo de geração", "Juiz", "Limiar", "Casos", "Notas medidas", "Arquivo"],
        rows=rows,
        widths=[0.115, 0.11, 0.11, 0.10, 0.07, 0.07, 0.10, 0.325],
        align=["l", "l", "l", "l", "r", "r", "r", "l"],
        small=True,
    )


def _extras(run: Run) -> list[str]:
    """Metadados próprios de cada framework, quando o JSON traz."""
    items = []
    e = run.extra
    if e.get("answers_source"):
        items.append(
            f"Respostas julgadas a partir de `{Path(e['answers_source']).name}` (geradas uma vez, as mesmas "
            "para todos os frameworks da execução)."
        )
    if e.get("ragas_version"):
        items.append(f"Versão do ragas: `{e['ragas_version']}`.")
    if e.get("instrument_sha"):
        items.append(
            f"Instrumento JevEval `{e['instrument_sha']}` (deepeval `{e.get('deepeval_version') or '?'}`): as "
            "perguntas feitas ao Jev estão em `question_sets` no JSON da rodada."
        )
    if e.get("jev_calls") is not None:
        cost = e.get("cost_usd")
        items.append(
            f"Chamadas ao Jev: {e['jev_calls']} (falhas: {e.get('jev_failures', 0)})"
            + (f"; custo: US$ {cost:.4f}." if cost else ".")
        )
    if run.embedding_model:
        items.append(f"Modelo de embeddings: `{run.embedding_model}`.")
    if e.get("judge_calls") is not None:
        items.append(f"Chamadas ao juiz: {e['judge_calls']} (falhas: {e.get('judge_failures', 0)}).")
    if e.get("wall_seconds") is not None:
        items.append(f"Tempo de parede: {fmt(e['wall_seconds'] / 60, 1)} min.")
    for metric, change in (e.get("prompt_overrides") or {}).items():
        items.append(f"Alteração no prompt de fábrica ({metric}): `{change}`.")
    return items


def _means_table(runs: list[Run]) -> Table:
    stats = [dimension_stats(r) for r in runs]
    headers = ["Dimensão"]
    for run in runs:
        headers += [f"{run.name}: média", f"{run.name}: ≥ limiar"]
    rows = []
    for dim in DIMENSIONS:
        row = [dim.label]
        for run, st in zip(runs, stats):
            s = st[dim.key]
            row += [fmt(s.mean), pct(s.n_passed, s.n_cases)]
        rows.append(row)
    n = len(runs)
    return Table(
        headers=headers,
        rows=rows,
        widths=[0.28] + [0.72 / (2 * n)] * (2 * n),
        align=["l"] + ["r"] * (2 * n),
    )


def _agreement_table(comp: Comparison) -> Table:
    rows = []
    for dim in DIMENSIONS:
        a = comp.per_dimension[dim.key]
        rows.append(
            [
                dim.label,
                str(a.n_both),
                pct(a.n_agree, a.n_both),
                fmt(a.mean_abs_diff),
            ]
        )
    return Table(
        headers=["Dimensão", "Casos com as duas notas", "Mesmo veredito", "Diferença média de nota"],
        rows=rows,
        widths=[0.34, 0.22, 0.22, 0.22],
        align=["l", "r", "r", "r"],
        caption=(
            "Veredito = nota ≥ limiar. Diferença média = média de |nota A − nota B| nos casos em que as duas "
            "rodadas mediram a dimensão."
        ),
    )


def _divergence_table(comp: Comparison) -> Table | None:
    if not comp.divergences:
        return None
    rows = [
        [
            d.case,
            DIMENSION_BY_KEY[d.dimension].label,
            fmt(d.value_a),
            fmt(d.value_b),
            signed(d.diff),
        ]
        for d in comp.divergences[:MAX_DIVERGENCES]
    ]
    extra = len(comp.divergences) - MAX_DIVERGENCES
    return Table(
        headers=["Caso", "Dimensão", comp.a.name, comp.b.name, "Diferença"],
        rows=rows,
        widths=[0.34, 0.26, 0.13, 0.13, 0.14],
        align=["l", "l", "r", "r", "r"],
        caption=(
            "Casos em que um framework aprova e o outro reprova, da maior diferença de nota para a menor"
            + (f" ({extra} a mais na planilha)." if extra > 0 else ".")
        ),
    )


def _dimension_table(run: Run) -> Table:
    stats = dimension_stats(run)
    rows = []
    for dim in DIMENSIONS:
        s = stats[dim.key]
        rows.append(
            [
                dim.label,
                f"`{NATIVE_METRIC.get(run.framework, {}).get(dim.key, dim.key)}`",
                fmt(s.mean),
                fmt(s.median),
                fmt(s.minimum),
                fmt(s.maximum),
                pct(s.n_measured, s.n_cases),
                pct(s.n_passed, s.n_cases),
            ]
        )
    return Table(
        headers=["Dimensão", "Métrica", "Média", "Mediana", "Mín.", "Máx.", "Medidas", "≥ limiar"],
        rows=rows,
        widths=[0.20, 0.29, 0.08, 0.09, 0.08, 0.08, 0.09, 0.09],
        align=["l", "l", "r", "r", "r", "r", "r", "r"],
        small=True,
    )


def _cases_table(run: Run) -> Table:
    rows = []
    for case in run.cases:
        name = case.name + (" †" if case.abstained else "")
        rows.append([name] + [score_cell(case.score(d.key)) for d in DIMENSIONS] + ["sim" if case.all_passed else "não"])
    return Table(
        headers=["Caso"] + [d.short for d in DIMENSIONS] + ["Todas ≥ limiar"],
        rows=rows,
        widths=[0.27] + [0.115] * len(DIMENSIONS) + [0.155],
        align=["l"] + ["r"] * len(DIMENSIONS) + ["c"],
        caption="✓ nota ≥ limiar · ✗ abaixo do limiar · n/d não medida · N/A não se aplica · † o assistente se absteve.",
        small=True,
    )


def _unmeasured(run: Run) -> list[str]:
    items = []
    for dim in DIMENSIONS:
        for case_name, error in dimension_stats(run)[dim.key].errors:
            items.append(f"**{case_name}** · {dim.label}: {clip(error, 200)}")
    return items


STATUS_LABEL = {
    "passou": "✓ passou",
    "concluída": "✓ concluída",
    "falhou": "✗ falhou",
    "pulada": "⊘ pulada",
    "erro": "⚠ erro",
}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{fmt(seconds / 60, 1)} min"


def _tests_cell(tests: dict | None) -> str:
    if not tests or not tests.get("total"):
        return "—"
    total = tests["total"]
    failed = tests.get("failed", 0) + tests.get("errors", 0)
    skipped = tests.get("skipped", 0)
    parts = [f"{total - failed - skipped} ✓"]
    if failed:
        parts.append(f"{failed} ✗")
    if skipped:
        parts.append(f"{skipped} ⊘")
    return " · ".join(parts) + f" (de {total})"


def _execution_blocks(execution: dict) -> list[Block]:
    started = execution.get("started_at")
    when = datetime.fromisoformat(started).astimezone().strftime("%d/%m/%Y %H:%M") if started else "—"
    text = (
        f"Execução de `{execution.get('tool', 'tests/run_all.py')}` iniciada em {when}, com duração total de "
        f"{_duration(execution.get('duration_seconds'))}."
    )
    shared = execution.get("shared_answers")
    if shared:
        text += (
            f" As suítes de qualidade julgaram as **mesmas respostas**: {shared.get('n', '?')} pergunta(s), "
            f"geradas uma vez só por `{shared.get('generation_model', '?')}`."
        )
    reused = execution.get("reused_results") or []
    if reused:
        text += (
            " Sem avaliação nova nesta execução para parte dos frameworks: o relatório reaproveita o resultado "
            "mais recente de " + ", ".join(f"`{r}`" for r in reused) + " (veja a data de cada rodada)."
        )
    if execution.get("note"):
        text += f" {execution['note']}"
    rows = [
        [
            step["name"],
            STATUS_LABEL.get(step["status"], step["status"]),
            _tests_cell(step.get("tests")),
            _duration(step.get("seconds")),
            step.get("detail") or "",
        ]
        for step in execution.get("steps", [])
    ]
    return [
        Para(text),
        Table(
            headers=["Suíte", "Resultado", "Testes", "Duração", "Observação"],
            rows=rows,
            widths=[0.22, 0.12, 0.19, 0.09, 0.38],
            align=["l", "l", "l", "r", "l"],
            caption="✓ passou · ✗ falhou · ⊘ pulada · ⚠ erro de execução. O log de cada suíte está em `logs/`, "
            "na pasta deste relatório.",
            small=True,
        ),
    ]


def _execution_warnings(execution: dict) -> list[Callout]:
    out = []
    steps = execution.get("steps", [])
    broken = [s for s in steps if s["status"] in ("falhou", "erro")]
    if broken:
        out.append(
            Callout(
                "Suítes com falha nesta execução",
                [f"**{s['name']}**: {s.get('detail') or s['status']} (log: `{s.get('log') or '—'}`)" for s in broken],
            )
        )
    controls = next((s for s in steps if s.get("key") == "ragas_controles"), None)
    if controls and controls["status"] == "falhou":
        out.append(
            Callout(
                "Controles do juiz RAGAS reprovaram",
                [
                    "Casos com resposta conhecida não passaram: nesta execução, as notas do RAGAS medem o juiz, "
                    "não o RAG. Ver `tests/ragas/README.md`, \"Qual juiz usar\"."
                ],
            )
        )
    controls = next((s for s in steps if s.get("key") == "jeveval_controles"), None)
    if controls and controls["status"] == "falhou":
        out.append(
            Callout(
                "Controles do instrumento JevEval reprovaram",
                [
                    "Casos com resposta conhecida não passaram: nesta execução, as notas do JevEval medem as "
                    "perguntas ou o Jev, não o RAG. Ver `tests/jeveval/README.md`, \"Validação do instrumento\"."
                ],
            )
        )
    return out


def _history_blocks(history: list[dict], figure: Path | None) -> list[Block]:
    first = history[0].get("run_generated_at")
    since = datetime.fromisoformat(first).astimezone().strftime("%d/%m/%Y") if first else "—"
    blocks: list[Block] = [
        Para(
            f"{len(history)} rodada(s) de avaliação registrada(s) desde {since}, cada uma contada uma vez só "
            "(pela data da avaliação, não do relatório). O índice de todos os relatórios, com links, fica em "
            "`tests/reports/out/HISTORICO.md`."
        )
    ]
    if figure:
        blocks.append(
            Figure(
                figure,
                "Evolução da média de cada dimensão, por framework e data da avaliação. A linha horizontal "
                "marca o limiar.",
            )
        )
    blocks.append(
        Table(
            headers=["Data", "Rodada"] + [d.short for d in DIMENSIONS] + ["Todas ≥ limiar", "Respostas comp."],
            rows=history_table_rows(history),
            widths=[0.135, 0.095] + [0.11] * len(DIMENSIONS) + [0.10, 0.12],
            align=["l", "l"] + ["r"] * len(DIMENSIONS) + ["r", "c"],
            caption="Média por dimensão em cada rodada, mais recente primeiro (até 15). \"Respostas comp.\" = a "
            "rodada julgou as respostas compartilhadas da execução, as mesmas dos outros frameworks.",
            small=True,
        )
    )
    return blocks


def _origin_table(runs: list[Run]) -> Table:
    in_index = {True: "sim", False: "não", None: "—"}
    rows = []
    for run in runs:
        for o in origin_stats(run):
            rows.append(
                [run.name, o.origin, in_index[o.in_index], str(o.n_cases), pct(o.abstained, o.n_cases)]
                + [fmt(o.means[d.key]) for d in DIMENSIONS]
                + [pct(o.all_passed, o.n_cases)]
            )
    return Table(
        headers=["Rodada", "Origem", "No índice", "Casos", "Abstenções"]
        + [d.short for d in DIMENSIONS]
        + ["Todas ≥ limiar"],
        rows=rows,
        widths=[0.17, 0.065, 0.07, 0.055, 0.085] + [0.09] * len(DIMENSIONS) + [0.105],
        align=["l", "l", "c", "r", "r"] + ["r"] * len(DIMENSIONS) + ["r"],
        caption="Médias por dimensão separadas pela origem da pergunta. \"No índice\" = o índice da rodada contém "
        "os artigos dessa origem.",
        small=True,
    )


def _retrieval_blocks(diagnostic: dict) -> list[Block]:
    """Tabela do diagnóstico de recuperação (tests/compare_indices.py): o documento certo veio entre os k trechos?"""
    k = diagnostic.get("top_k", 4)
    indices = diagnostic["indices"]
    langs = diagnostic.get("langs", ["pt"])
    lang_name = {"pt": "PT", "en": "EN"}
    # Quem gera o diagnóstico pode dizer de onde ele veio (a demonstração E2E o
    # tira dos trechos que o chat devolveu, não de uma busca à parte).
    how = diagnostic.get("description") or (
        "Só embeddings, sem LLM: mede a recuperação isolada do resto do pipeline. "
        "A mesma pergunta em inglês separa o efeito do idioma do efeito do índice."
    )
    blocks: list[Block] = [
        Para(
            f"Para cada golden com documento-fonte conhecido, a posição do documento certo entre os {k} trechos "
            f"recuperados (— = não veio). {how}"
        )
    ]
    totals = []
    for index in indices:
        for lang in langs:
            hits = sum(1 for row in diagnostic["rows"] if row["ranks"].get(index, {}).get(lang))
            totals.append(f"**{index}**, perguntas em {lang_name.get(lang, lang)}: {hits}/{len(diagnostic['rows'])}")
    blocks.append(Bullets(totals))
    headers = ["Golden", "Conversão"] + [f"{i} {lang_name.get(lang, lang)}" for i in indices for lang in langs]
    rows = [
        [row["name"], row.get("conversion") or "—"]
        + [str(row["ranks"].get(i, {}).get(lang) or "—") for i in indices for lang in langs]
        for row in diagnostic["rows"]
    ]
    n = len(indices) * len(langs)
    blocks.append(
        Table(
            headers=headers,
            rows=rows,
            widths=[0.38, 0.14] + [0.48 / n] * n,
            align=["l", "l"] + ["c"] * n,
            caption="Posição (1 = primeiro) do documento-fonte entre os trechos recuperados.",
            small=True,
        )
    )
    return blocks


def build_document(
    runs: list[Run],
    comparisons: list[Comparison],
    figures: dict[str, Path],
    generated_at: datetime,
    execution: dict | None = None,
    history: list[dict] | None = None,
) -> list[Block]:
    # Origem da pergunta -> arquivo de goldens (tests/generate_answers.py, KNOWN_ORIGINS).
    golden_files = [{"pdf": "dataset.json", "md": "dataset_md.json"}.get(o, o) for o in origins(runs)]
    frameworks = {r.framework for r in runs}
    if frameworks == {"JevEval"}:
        judge = "com o Jev, da TypeSafe AI, como juiz (perguntas fechadas, respondidas com probabilidades)"
    elif "JevEval" in frameworks:
        judge = "com um LLM local como juiz (RAGAS, DeepEval) ou com o Jev, da TypeSafe AI (JevEval)"
    else:
        judge = "com um LLM local como juiz"
    doc: list[Block] = [
        Heading(1, "Relatório de qualidade de resposta do RAG"),
        Para(
            f"Gerado em {generated_at.strftime('%d/%m/%Y %H:%M')} a partir de {len(runs)} rodada(s) de avaliação: "
            + ", ".join(f"**{r.label}**" for r in runs)
            + ". Cada rodada julgou as respostas do assistente para os mesmos goldens ("
            + " e ".join(f"`tests/deepeval/goldens/{f}`" for f in golden_files or ["dataset.json"])
            + f") em cinco dimensões, {judge}."
        ),
    ]
    number = 0

    def section(title: str) -> Heading:
        nonlocal number
        number += 1
        return Heading(2, f"{number}. {title}")

    if execution:
        doc.append(section("Execução dos testes"))
        doc += _execution_blocks(execution)

    doc.append(section("Resumo"))
    if execution:
        doc += _execution_warnings(execution)
    for run in runs:
        doc.append(Para(f"**{run.label}**"))
        doc.append(Bullets(run_highlights(run, dimension_stats(run))))
    for comp in comparisons:
        warnings = comp.warnings
        if warnings:
            doc.append(
                Callout(
                    f"Comparabilidade: {comp.a.label} × {comp.b.label}",
                    warnings
                    + [
                        "Leia as diferenças abaixo como diferença entre RODADAS (método + juiz + respostas), "
                        "não só entre métodos, enquanto estes avisos existirem."
                    ],
                )
            )

    doc.append(section("Rodadas avaliadas"))
    doc.append(_run_table(runs))
    for run in runs:
        extras = _extras(run)
        if extras:
            doc.append(Para(f"**{run.name}** — detalhes da rodada:", small=True))
            doc.append(Bullets(extras))

    if len(runs) > 1:
        same_system = len({r.variant for r in runs}) == 1
        doc.append(section("Comparação entre frameworks" if same_system else "Comparação entre as rodadas"))
        if not same_system:
            doc.append(
                Para(
                    "As rodadas usam versões do índice (ou idiomas de pergunta) diferentes: respostas e trechos "
                    "recuperados diferem por construção — é o que está sendo comparado. Para a comparação valer, "
                    "o resto precisa ser igual: perguntas, gerador, juiz e limiar (os avisos de comparabilidade "
                    "do resumo dizem se não foi)."
                )
            )
        doc.append(Figure(figures["means"], "Média de cada dimensão por rodada. A linha vertical marca o limiar."))
        doc.append(_means_table(runs))
        for comp in comparisons:
            doc.append(Heading(3, f"Concordância: {comp.a.name} × {comp.b.name}"))
            doc.append(
                Para(
                    f"{len(comp.common_cases)} casos em comum. Respostas idênticas nas duas rodadas: "
                    f"{comp.identical_answers}; trechos recuperados idênticos: {comp.identical_contexts}."
                )
            )
            doc.append(_agreement_table(comp))
            divergences = _divergence_table(comp)
            if divergences:
                doc.append(Heading(3, f"Divergências de veredito: {comp.a.name} × {comp.b.name}"))
                doc.append(divergences)

    doc.append(section("Resultados por framework" if len({r.variant for r in runs}) == 1 else "Resultados por rodada"))
    for i, run in enumerate(runs, start=1):
        doc.append(Heading(3, f"{number}.{i} {run.label}"))
        doc.append(
            Para(
                f"Geração: `{run.generation_model or '—'}` · juiz: `{run.judge_model or '—'}` · "
                f"limiar: {fmt(run.threshold)} · arquivo: `{_rel(run.source)}`."
            )
        )
        doc.append(Bullets(run_highlights(run, dimension_stats(run))))
        doc.append(_dimension_table(run))
        doc.append(
            Figure(
                figures[f"heatmap_{i}"],
                f"{run.label}: nota por caso e dimensão. Vermelho abaixo do limiar, azul acima, cinza no "
                "limiar; hachurado = não medida; N/A = não se aplica; † = o assistente se absteve.",
            )
        )
        doc.append(_cases_table(run))
        unmeasured = _unmeasured(run)
        if unmeasured:
            doc.append(Para("**Notas não medidas** (erro do juiz, timeout ou NaN):", small=True))
            doc.append(Bullets(unmeasured))

    if origins(runs) and len(origins(runs)) > 1:
        doc.append(section("Por origem da pergunta"))
        doc.append(
            Para(
                "Origem = o acervo de onde vieram a pergunta e a resposta de referência. O pdf-v1 e o md-v1 têm "
                "um acervo cada; as versões v2 têm os mesmos artigos dos dois, num formato só. Quando o índice "
                "não contém a origem da pergunta (\"No índice\" = não), o comportamento certo é se abster "
                "(\"não encontrei...\"), e as notas de fidelidade e de relevância julgam uma recusa. Quando "
                "contém, uma abstenção é resposta perdida."
            )
        )
        doc.append(_origin_table(runs))

    diagnostic = (execution or {}).get("retrieval_diagnostic")
    if diagnostic:
        doc.append(section("Recuperação do documento certo"))
        doc += _retrieval_blocks(diagnostic)

    if history:
        doc.append(section("Histórico"))
        doc += _history_blocks(history, figures.get("history"))

    doc.append(PageBreak())
    doc.append(section("Detalhe por caso"))
    doc.append(
        Para(
            "Pergunta, resposta de referência e, para cada rodada, a resposta avaliada e as notas. "
            "Respostas e justificativas do juiz foram encurtadas; o texto completo está na planilha e nos JSONs.",
            small=True,
        )
    )
    names: list[str] = []
    for run in runs:
        names += [c.name for c in run.cases if c.name not in names]
    for name in names:
        first = next(c for r in runs if (c := r.case(name)))
        doc.append(Heading(3, name))
        doc.append(Para(f"**Pergunta:** {first.question}"))
        doc.append(Para(f"**Referência:** {clip(first.expected, 600)}", small=True))
        answers = [run.case(name).answer for run in runs if run.case(name)]
        shared_answer = len(answers) > 1 and all(a.strip() == answers[0].strip() for a in answers)
        if shared_answer:
            doc.append(Para(f"Resposta (a mesma em todas as rodadas): {clip(answers[0], 700)}", small=True))
        for run in runs:
            case = run.case(name)
            if case is None:
                continue
            notas = " · ".join(f"{d.short} {score_cell(case.score(d.key))}" for d in DIMENSIONS)
            doc.append(Para(f"**{run.name}** — {notas}", small=True))
            if not shared_answer:
                doc.append(Para(f"Resposta: {clip(case.answer, 700)}", small=True))
            reasons = [
                f"{d.short}: {clip(case.score(d.key).reason, 260)}" for d in DIMENSIONS if case.score(d.key).reason
            ]
            if reasons:
                doc.append(Bullets(reasons))

    doc.append(section("Como ler este relatório"))
    with_jeveval = "JevEval" in frameworks
    columns = ["RAGAS", "DeepEval"] + (["JevEval"] if with_jeveval else [])
    doc.append(
        Table(
            headers=["Dimensão", "Pergunta que a métrica responde"] + columns,
            rows=[[d.label, d.question] + [f"`{NATIVE_METRIC[f][d.key]}`" for f in columns] for d in DIMENSIONS],
            widths=[0.17, 0.27] + [0.56 / len(columns)] * len(columns),
            align=["l"] * (2 + len(columns)),
            caption="Equivalência entre as métricas dos frameworks.",
            small=True,
        )
    )
    notes = [
        "Notas vão de 0 a 1. **Veredito** = nota ≥ limiar da rodada. O limiar é um ponto de partida, "
        "não um valor calibrado.",
        "A mesma dimensão não é a mesma conta nos frameworks: a fidelidade do RAGAS exige que cada "
        "afirmação seja *inferível* do contexto; a do DeepEval conta afirmações que não o *contradizem*. "
        "Divergência de veredito é o ponto de partida da análise, não um erro de um dos lados.",
        "A geração não é determinística: rodadas que geram as próprias respostas julgam textos diferentes "
        "para a mesma pergunta. O `tests/run_all.py` evita isso gerando as respostas uma vez só; a seção "
        "de comparabilidade diz o que aconteceu em cada caso.",
        "Quando o assistente se abstém (\"não encontrei...\"), fidelidade e relevância da resposta julgam "
        "uma recusa; as métricas de contexto continuam medindo a recuperação.",
        "\"n/d\" = a métrica não produziu nota (erro, timeout ou NaN). Fica fora das médias, mas nas "
        "contagens \"≥ limiar\" e \"todas ≥ limiar\" conta como não aprovada: nota que não existe não "
        "aprova ninguém.",
    ]
    if with_jeveval:
        notes += [
            "No JevEval, o Jev não escreve veredito: responde perguntas fechadas com probabilidades, e a nota é "
            "a média ponderada dessas respostas, uma conta fixa feita fora do modelo. A fidelidade dele segue o "
            "RAGAS (afirmação sobre o que o contexto não menciona não tem apoio). As perguntas estão em "
            "`tests/jeveval/_jev_shared.py`.",
            "\"N/A\" = nenhuma pergunta do JevEval se aplicou ao caso, como a fidelidade de uma recusa. Diferente "
            "de \"n/d\": não é falha de medida, e por isso fica fora das médias e das contagens, sem aprovar nem "
            "reprovar.",
        ]
    doc.append(Bullets(notes))
    return doc
