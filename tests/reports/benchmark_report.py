"""Relatório do benchmark de estratégias de RAG: um PDF didático (e o mesmo conteúdo em Markdown) a partir das
rodadas que o runner gravou em tests/benchmark/out/.

O conteúdo fica em tests/benchmark/relatorio/ (ver o README de lá). O manifesto, relatorio.yaml, dá a ordem dos
textos. Cada texto é um Markdown com um cabeçalho YAML que declara as rodadas, as figuras, as tabelas e as
análises que ele usa; no corpo, marcadores {{...}} recebem os números do placar e diretivas [[...]] põem no
lugar o cartão, as figuras e as tabelas. O texto é de quem escreve; os números vêm sempre das rodadas. Se uma
rodada nova mudar um veredito que o texto afirma (o campo `vereditos` do cabeçalho), o relatório avisa na
própria seção e no apêndice.

Uso, da raiz, com o venv do relatório (tests/reports/.venv):

    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\benchmark_report.py
    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\benchmark_report.py --abrir
    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\benchmark_report.py --formatos pdf --out C:\\temp\\rel

A saída vai para tests/benchmark/out/relatorio_<data>/ (não versionada): relatorio-benchmark.pdf, o .md e
figuras/.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_charts as charts  # noqa: E402
from benchmark_data import (  # noqa: E402
    MANIFESTS,
    OUT_ROOT,
    PROJECT_ROOT,
    ReportError,
    Run,
    changed_answers,
    comparison,
    e5_analysis,
    fmt_delta,
    fmt_margin,
    fmt_point,
    fmt_value,
    load_run,
    metric_name,
    num,
    outside_separation,
    pct,
    seconds,
    split_pair,
    unit,
)
from report_document import Block, Bullets, Callout, Figure, Heading, PageBreak, Para, Table  # noqa: E402
from tests.benchmark import placar  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "tests" / "benchmark" / "relatorio" / "relatorio.yaml"
MISSING = "⟨?⟩"
PLACEHOLDER = re.compile(r"\{\{\s*(.+?)\s*\}\}")
DIRECTIVE = re.compile(r"^\[\[\s*(\w+)(?:\s+(.+?))?\s*\]\]$")
LANGS = {"pt": "português", "en": "inglês"}
ROLES = {"baseline": "referência", "candidato": "candidato", "ruido": "A/A (mede o ruído)", "teto": "teto",
         "referencia": "ponto de comparação"}
ATTRIBUTION = [("fonte", "cita a fonte"), ("outro_documento", "cita outro documento recebido"),
               ("sem_citacao", "sem citação"), ("fabricada", "citação fabricada"), ("recusa", "recusa")]


@dataclass
class Text:
    path: Path
    meta: dict
    body: str


def read_text(path: Path) -> Text:
    """Markdown com cabeçalho YAML entre duas linhas `---`."""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        try:
            meta = yaml.safe_load(header) or {}
        except yaml.YAMLError as exc:
            raise ReportError(f"o cabeçalho de {path.name} não é YAML válido: {exc}") from exc
    else:
        meta, body = {}, raw
    return Text(path=path, meta=meta, body=body.strip("\n"))


def _pair_label(pair: str, labels: dict | None = None) -> str:
    if labels and pair in labels:
        return labels[pair]
    a, b = split_pair(pair)
    return f"{a} → {b}"


def _date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso


def _duration(total: float) -> str:
    if total < 60:
        return f"{total:.0f} s"
    if total < 3600:
        return f"{total / 60:.0f} min"
    hours, minutes = divmod(round(total / 60), 60)
    return f"{hours} h {minutes:02d}"


class Report:
    """O relatório inteiro: as seções na ordem do manifesto, o cache de rodadas e os avisos."""

    def __init__(self, base: Path, out_root: Path, figures: Path, manifests: Path = MANIFESTS) -> None:
        self.base, self.out_root, self.figures, self.manifests = base, out_root, figures, manifests
        self.sections: list[Section] = []
        self.warnings: list[str] = []
        self._runs: dict[str, Run] = {}

    def run(self, name: str, alias: str) -> Run:
        if name not in self._runs:
            self._runs[name] = load_run(alias, name, self.out_root)
        return self._runs[name]

    def runs_used(self) -> list[Run]:
        return list(self._runs.values())

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


class Section:
    """Um texto do relatório, com as rodadas e as análises que o cabeçalho dele declara."""

    def __init__(self, report: Report, text: Text) -> None:
        self.report, self.text, self.meta = report, text, text.meta
        self.id = self.meta.get("id")
        self.where = self.id or text.path.name
        self.runs = {alias: report.run(name, alias) for alias, name in (self.meta.get("rodadas") or {}).items()}
        self._analyses: dict[str, dict] = {}

    # ---------------------------------------------------------------- dados

    def run(self, alias: str) -> Run:
        if alias not in self.runs:
            raise ReportError(f"a rodada {alias!r} não está em `rodadas`")
        return self.runs[alias]

    def analysis(self, analysis_id: str) -> dict:
        if analysis_id not in self._analyses:
            spec = (self.meta.get("analises") or {}).get(analysis_id)
            if not spec:
                raise ReportError(f"a análise {analysis_id!r} não está em `analises`")
            kind = spec.get("tipo")
            if kind == "e5":
                result = e5_analysis(self.run(spec["rodada"]), spec["braco"], self.report.manifests)
            elif kind == "comparacao":
                result = comparison(self.runs, spec["de"], spec["para"], spec["metricas"])
            elif kind == "mudancas":
                result = changed_answers(self.run(spec["rodada"]), spec["de"], spec["para"])
            else:
                raise ReportError(f"análise {analysis_id!r} de tipo desconhecido: {kind!r}")
            self._analyses[analysis_id] = result
        return self._analyses[analysis_id]

    # ---------------------------------------------------------------- marcadores

    def resolve(self, text: str | None) -> str:
        if not text:
            return ""
        # O `>` do YAML deixa uma quebra de linha no fim, e o texto do corpo já chega numa linha só.
        return PLACEHOLDER.sub(lambda m: self._placeholder(m.group(1)), " ".join(str(text).split()))

    def _placeholder(self, expression: str) -> str:
        kind, *args = expression.split()
        try:
            handler = getattr(self, f"_p_{kind}", None)
            if handler is None:
                raise ReportError(f"marcador desconhecido {kind!r}")
            value = handler(*args)
            if value in (None, "—"):
                raise ReportError("sem valor na rodada")
            return value
        except (ReportError, KeyError, TypeError, ValueError, IndexError) as exc:
            self.report.warn(self.where, f"{{{{{expression}}}}} não foi preenchido ({exc})")
            return MISSING

    def _p_media(self, alias: str, arm: str, metric: str, spec: str | None = None) -> str:
        value = self.run(alias).mean(arm, metric)
        if value is None:
            raise ReportError(f"{arm} não tem {metric}")
        return fmt_value(metric, value, spec)

    def _p_contagem(self, alias: str, arm: str, metric: str) -> str:
        counted = self.run(alias).count(arm, metric)
        if counted is None:
            raise ReportError(f"{arm} não tem {metric} por pergunta (medidas.json)")
        return f"{counted[0]} de {counted[1]}"

    def _p_n(self, alias: str, arm_or_pair: str, metric: str) -> str:
        run = self.run(alias)
        if "→" in arm_or_pair:
            delta = run.pair(arm_or_pair, metric)
            return str(delta["n"]) if delta else None
        counted = run.count(arm_or_pair, metric)
        return str(counted[1]) if counted else None

    def _p_delta(self, alias: str, pair: str, metric: str, spec: str | None = None) -> str:
        return fmt_delta(metric, self.run(alias).pair(pair, metric), spec)

    def _p_dif(self, alias: str, pair: str, metric: str) -> str:
        return fmt_point(metric, self.run(alias).pair(pair, metric))

    def _p_veredito(self, alias: str, pair: str, metric: str | None = None) -> str:
        run = self.run(alias)
        if metric:
            return run.judged(pair, metric)[2]
        verdict = run.verdict(pair)
        if not verdict:
            raise ReportError(f"o par {pair} não tem veredito no placar")
        return verdict["verdict"]

    def _p_margem(self, alias: str, metric_or_pair: str) -> str:
        run = self.run(alias)
        if "→" in metric_or_pair:
            verdict = run.verdict(metric_or_pair) or {}
            return fmt_margin(verdict.get("metric", ""), verdict.get("margin"))
        return fmt_margin(metric_or_pair, run.margin(metric_or_pair)[0])

    def _p_ruido(self, alias: str, group: str, metric: str) -> str:
        """A margem de um grupo A/A do próprio cartão (E0: gerador_juiz, juiz, c1)."""
        return fmt_margin(metric, (self.run(alias).placar.get("margins") or {}).get(group, {}).get(metric))

    def _p_buraco(self, alias: str, arm: str) -> str:
        """A fração do buraco fechado (E1): (braço − baseline) / (teto − baseline)."""
        return pct((self.run(alias).placar.get("gap_closed") or {}).get(arm))

    def _p_auc(self, alias: str, arm: str, which: str = "top1_similarity_auc") -> str:
        auc = ((self.run(alias).placar.get("summary", {}).get(arm) or {}).get("c1") or {}).get(which)
        if not auc:
            raise ReportError(f"{arm} não tem {which}")
        return f"{num(auc['auc'])} [{num(auc.get('ci_low'))}; {num(auc.get('ci_high'))}]"

    def _p_limiar(self, alias: str, arm: str) -> str:
        return num((self.run(alias).placar.get("thresholds") or {})[arm]["threshold"], 3)

    def _p_categoria(self, alias: str, arm: str, category: str) -> str:
        """Quantas respostas de dentro caem numa categoria de atribuição (E3: fonte, fabricada...)."""
        counts = (self.run(alias).placar.get("summary", {}).get(arm) or {}).get("attribution") or {}
        return str(counts[category])

    def _p_comparacao(self, analysis_id: str, metric: str, what: str = "delta") -> str:
        entry = self.analysis(analysis_id)["metrics"][metric]
        if what == "delta":
            return fmt_delta(metric, entry["delta"])
        if what == "veredito":
            return entry["verdict"]
        return fmt_value(metric, entry[what])

    def _p_valor(self, analysis_id: str, path: str, spec: str | None = None) -> str:
        value = self.analysis(analysis_id)
        for key in path.split("/"):
            value = value[int(key)] if isinstance(value, list) else value[key]
        if isinstance(value, str):
            return value
        if spec == "pct":
            return pct(value)
        if spec == "int":
            return num(value, 0)
        return num(value, int(spec) if spec and spec.isdigit() else 2)

    # ---------------------------------------------------------------- blocos

    def blocks(self) -> list[Block]:
        out: list[Block] = []
        if self.id:
            out += [PageBreak(), Heading(2, f"{self.id} — {self.meta.get('titulo', '')}")]
            out += self._verdict_check()
        else:
            if self.meta.get("quebra", True):
                out.append(PageBreak())
            if self.meta.get("titulo"):
                out.append(Heading(2, self.meta["titulo"]))
        return out + self._body()

    def _verdict_check(self) -> list[Block]:
        problems = []
        for key, expected in (self.meta.get("vereditos") or {}).items():
            alias, pair, *metric = key.split()
            try:
                run = self.run(alias)
                actual = run.judged(pair, metric[0])[2] if metric else (run.verdict(pair) or {}).get("verdict")
            except ReportError as exc:
                actual = f"erro: {exc}"
            if actual != expected:
                message = f"{pair}{' · ' + metric_name(metric[0]) if metric else ''}: o texto foi escrito para " \
                          f"“{expected}”, e o placar agora diz “{actual}”"
                self.report.warn(self.where, message)
                problems.append(message)
        if not problems:
            return []
        return [Callout("O texto desta seção foi escrito para outro resultado. Revise a leitura.", problems)]

    def _body(self) -> list[Block]:
        blocks: list[Block] = []
        paragraph: list[str] = []
        bullets: list[str] = []
        quote: list[str] = []
        table: list[str] = []
        numbered = False

        def flush() -> None:
            nonlocal paragraph, bullets, quote, table
            if paragraph:
                blocks.append(Para(self.resolve(" ".join(paragraph))))
            if bullets:
                blocks.append(Bullets([self.resolve(b) for b in bullets], numbered=numbered))
            if quote:
                blocks.append(self._callout(quote))
            if table:
                blocks.append(self._pipe_table(table))
            paragraph, bullets, quote, table = [], [], [], []

        for raw in self.text.body.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if quote and not stripped.startswith(">"):
                flush()
            if table and not stripped.startswith("|"):
                flush()
            if not stripped:
                flush()
                continue
            if stripped.startswith(">"):
                if not quote:
                    flush()
                quote.append(stripped[1:].strip())
            elif stripped.startswith("|"):
                if not table:
                    flush()
                table.append(stripped)
            elif match := DIRECTIVE.match(stripped):
                flush()
                blocks += self._directive(match.group(1), match.group(2))
            elif heading := re.match(r"^(#{2,4})\s+(.*)$", stripped):
                flush()
                blocks.append(Heading(min(len(heading.group(1)), 3), self.resolve(heading.group(2))))
            elif (item := re.match(r"^([-*]|\d+\.)\s+", stripped)) and not line.startswith("  "):
                is_number = item.group(1)[0].isdigit()
                if paragraph or (bullets and is_number != numbered):
                    flush()
                numbered = is_number
                bullets.append(stripped[item.end():])
            elif bullets and line.startswith("  "):
                bullets[-1] += " " + stripped
            else:
                if bullets:
                    flush()
                paragraph.append(stripped)
        flush()
        return blocks

    def _callout(self, lines: list[str]) -> Callout:
        first, rest = lines[0], lines[1:]
        tone, title = "nota", "Nota"
        if match := re.match(r"^\[!(aviso|nota)\]\s*(.*)$", first):
            tone, title = match.group(1), match.group(2) or ("Atenção" if match.group(1) == "aviso" else "Nota")
        elif match := re.match(r"^\*\*(.+?)\*\*\s*(.*)$", first):
            title = match.group(1)
            rest = ([match.group(2)] if match.group(2) else []) + rest
        else:
            rest = lines
        items, current = [], []
        for line in rest:
            if not line:
                if current:
                    items.append(" ".join(current))
                current = []
            elif line.startswith("- "):
                if current:
                    items.append(" ".join(current))
                current = [line[2:]]
            else:
                current.append(line)
        if current:
            items.append(" ".join(current))
        return Callout(self.resolve(title), [self.resolve(i) for i in items], tone=tone)

    def _pipe_table(self, lines: list[str]) -> Table:
        rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
        align = ["l"] * len(rows[0])
        if len(rows) > 1 and all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
            align = ["c" if c.startswith(":") and c.endswith(":") else "r" if c.endswith(":") else "l"
                     for c in rows[1]]
            rows = [rows[0]] + rows[2:]
        n = len(rows[0])
        # A largura de cada coluna segue o texto dela (com teto, para a coluna longa não espremer as outras).
        lengths = [min(max(len(r[i]) if i < len(r) else 0 for r in rows), 60) + 4 for i in range(n)]
        return Table([self.resolve(h) for h in rows[0]], [[self.resolve(c) for c in r[:n]] for r in rows[1:]],
                     [length / sum(lengths) for length in lengths], align)

    # ---------------------------------------------------------------- diretivas

    def _directive(self, name: str, arg: str | None) -> list[Block]:
        try:
            if name == "cartao":
                return self._card(self.run(arg))
            if name == "vereditos":
                return [self._verdicts_table(self.run(arg))]
            if name == "guardas":
                return [self._guards_table(self.run(arg))]
            if name == "figura":
                return self._figure(arg)
            if name == "tabela":
                return [self._table(arg)]
            if name == "resumo":
                return [self._summary_table()]
            if name == "ajustes":
                return [self._adjustments_table()]
            raise ReportError(f"diretiva desconhecida [[{name}]]")
        except (ReportError, KeyError, ValueError, TypeError) as exc:
            self.report.warn(self.where, f"[[{name} {arg or ''}]] não foi montada ({exc})")
            return [Para(f"{MISSING} [[{name} {arg or ''}]] não foi montada: {exc}", small=True)]

    def _card(self, run: Run) -> list[Block]:
        card = run.card
        primary = card.get("metrica_primaria") or {}
        guards = [f"{g.get('nome')} ({_pair_label('→'.join(g['par']))}, {metric_name(g['metrica'])})"
                  for g in card.get("guardas") or []]
        rows = [["A pergunta", card.get("pergunta", "—")],
                ["A hipótese, escrita antes de rodar", " ".join(str(card.get("hipotese", "—")).split())],
                ["A métrica que decide", metric_name(f"{primary.get('camada')}.{primary.get('metrica')}")],
                ["As guardas (o que não pode piorar)", "; ".join(guards) or "nenhuma"],
                ["O que a turma deve perceber", " ".join(str(card.get("didatica", "—")).split())]]
        arms = []
        for arm_id, spec in (card.get("bracos") or {}).items():
            spec = spec or {}
            if "rejulgar" in spec:
                arms.append([f"`{arm_id}`", "—", "—", f"julga de novo as respostas de `{spec['rejulgar']}`",
                             ROLES.get(spec.get("papel", "ruido"), "—")])
                continue
            changes = [f"`{k}` = {v}" for k, v in (spec.get("set") or {}).items()]
            if spec.get("ordem"):
                changes.append(f"perguntas em outra ordem (semente {spec['ordem']})")
            arms.append([f"`{arm_id}`", f"`{spec.get('indice', '—')}`", LANGS.get(spec.get("lingua", "pt"), "—"),
                         "; ".join(changes) or "—", ROLES.get(spec.get("papel", "candidato"), "—")])
        split = run.params.get("split")
        try:
            where = run.path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:  # uma pasta de rodadas fora do repositório (--manifesto com `pasta_rodadas`)
            where = run.path.as_posix()
        note = (f"O cartão, como foi escrito, está em `{where}/cartao.yaml`."
                + (f" Esta rodada usou só as perguntas do recorte de **{split}**." if split else ""))
        return [Table(["O cartão", ""], rows, [0.3, 0.7], caption=None),
                Table(["Braço", "Índice", "Pergunta em", "O que muda em relação à referência", "Papel"], arms,
                      [0.15, 0.17, 0.13, 0.37, 0.18], small=True, caption=note)]

    def _verdict_rows(self, run: Run, pairs: list[str], metrics: list[str] | None, labels: dict | None) -> list[list]:
        rows = []
        for pair in pairs:
            for metric in metrics or [(run.verdict(pair) or {}).get("metric")]:
                if not metric:
                    continue
                delta, margin, verdict = run.judged(pair, metric)
                rows.append([_pair_label(pair, labels), metric_name(metric), fmt_delta(metric, delta),
                             fmt_margin(metric, margin), verdict, str(delta["n"]) if delta else "—"])
        return rows

    def _verdicts_table(self, run: Run, pairs: list[str] | None = None, metrics: list[str] | None = None,
                        labels: dict | None = None) -> Table:
        pairs = pairs or list(run.placar.get("verdicts") or {})
        return Table(["Par", "Métrica", "Δ [IC 95%]", "Margem", "Veredito", "n"],
                     self._verdict_rows(run, pairs, metrics, labels), [0.24, 0.2, 0.22, 0.1, 0.17, 0.07],
                     ["l", "l", "r", "r", "l", "r"], small=True)

    def _guards_table(self, run: Run) -> Table:
        rows = []
        for guard in run.placar.get("guards") or []:
            metric = guard["metrica"]
            tolerance = guard.get("tolerance")
            rows.append([guard.get("nome", "—"), _pair_label("→".join(guard["par"])), fmt_delta(metric, guard.get("delta")),
                         fmt_margin(metric, tolerance) if tolerance is not None else "—",
                         {True: "✓ passa", False: "✗ não passa"}.get(guard.get("passed"), "não medida")])
        return Table(["Guarda", "Par", "Δ [IC 95%]", "Tolerância", "Resultado"], rows, [0.3, 0.18, 0.24, 0.12, 0.16],
                     ["l", "l", "r", "r", "l"], small=True)

    def _summary_table(self) -> Table:
        rows = []
        for section in self.report.sections:
            if not section.id:
                continue
            question = section.meta.get("pergunta")
            if not question and section.runs:
                question = next(iter(section.runs.values())).card.get("pergunta")
            rows.append([f"**{section.id}** {section.meta.get('titulo', '')}", section.resolve(question or "—"),
                         section.resolve(section.meta.get("resumo") or "—"),
                         section.resolve(section.meta.get("recomendacao") or "—")])
        return Table(["Experimento", "A pergunta", "O que mostrou", "Para o assistente"], rows,
                     [0.15, 0.22, 0.38, 0.25], small=True)

    def _adjustments_table(self) -> Table:
        rows = []
        for section in self.report.sections:
            for item in section.meta.get("ajuste") or []:
                rows.append([f"`{item['variavel']}`", f"`{item['valor']}`", section.resolve(item.get("motivo", "")),
                             section.id or "—"])
        return Table(["Variável (.env)", "Valor", "Por quê", "De onde"], rows, [0.27, 0.2, 0.43, 0.1], small=True)

    # ---------------------------------------------------------------- tabelas

    def _table(self, table_id: str) -> Table:
        spec = (self.meta.get("tabelas") or {}).get(table_id)
        if not spec:
            raise ReportError(f"a tabela {table_id!r} não está em `tabelas`")
        kind = spec.get("tipo")
        builder = getattr(self, f"_t_{kind}", None)
        if builder is None:
            raise ReportError(f"tabela de tipo desconhecido: {kind!r}")
        table = builder(spec)
        if spec.get("legenda"):
            table.caption = self.resolve(spec["legenda"])
        return table

    def _t_medias(self, spec: dict) -> Table:
        run, arms, labels = self.run(spec["rodada"]), spec["bracos"], spec.get("rotulos") or {}
        rows = []
        for metric in spec["metricas"]:
            cells = [metric_name(metric)]
            for arm in arms:
                value = run.mean(arm, metric)
                counted = run.count(arm, metric) if metric in placar.BINARY_METRICS else None
                text = fmt_value(metric, value)
                cells.append(f"{text} ({counted[0]} de {counted[1]})" if counted and value is not None else text)
            rows.append(cells)
        n = len(arms)
        return Table(["Métrica"] + [labels.get(a, f"`{a}`") for a in arms], rows, [0.3] + [0.7 / n] * n,
                     ["l"] + ["r"] * n, small=True)

    def _t_pares(self, spec: dict) -> Table:
        run = self.run(spec["rodada"])
        return self._verdicts_table(run, spec.get("pares"), spec.get("metricas"), spec.get("rotulos"))

    def _t_comparacao(self, spec: dict) -> Table:
        result = self.analysis(spec["analise"])
        rows = [[metric_name(m), fmt_value(m, e["de"]), fmt_value(m, e["para"]), fmt_delta(m, e["delta"]),
                 e["verdict"]] for m, e in result["metrics"].items()]
        return Table(["Métrica", spec.get("rotulo_de", result["de"]), spec.get("rotulo_para", result["para"]),
                      "Δ [IC 95%]", "Veredito"], rows, [0.28, 0.16, 0.16, 0.22, 0.18], ["l", "r", "r", "r", "l"],
                     small=True)

    def _t_e5(self, spec: dict) -> Table:
        ids, labels = spec["analises"], spec.get("rotulos") or {}
        results = [self.analysis(i) for i in ids]
        first = results[0]

        def row(name: str, observed, expected=None) -> list[str]:
            return [name] + [observed(r) for r in results] + [expected(first) if expected else "—"]

        rows = [
            row("Massa fora da fonte (trechos que não são do documento da pergunta)",
                lambda r: pct(r["off_source_mass"])),
            row("Distância até o esperado (TVD)", lambda r: num(r["total_variation"])),
            row(f"Documentos que somam 80% dos trechos (de {first['n_documents']})",
                lambda r: str(r["n80"]["observed"]), lambda r: str(r["n80"]["expected"])),
            row("Perguntas com todas as fontes no top 20%", lambda r: pct(r["statement"]["observed"]["all"]),
                lambda r: pct(r["statement"]["expected"]["all"])),
            row("Gini da participação", lambda r: num(r["gini"]["observed"]), lambda r: num(r["gini"]["expected"])),
            row("Documentos ímã", lambda r: str(len(r["magnets"]))),
            row("Documentos mortos", lambda r: str(len(r["dead"]))),
        ]
        n = len(ids)
        return Table(["Medida"] + [labels.get(i, i) for i in ids] + ["Esperado pelo desenho"], rows,
                     [0.42] + [0.4 / n] * n + [0.18], ["l"] + ["r"] * (n + 1), small=True)

    def _t_imas(self, spec: dict) -> Table:
        result = self.analysis(spec["analise"])
        rows = [[m["document"], str(m["undue"]), num(m["chance"], 1)] for m in result["magnets"][: spec.get("limite", 8)]]
        if not rows:
            rows = [["nenhum documento ímã", "—", "—"]]
        return Table(["Documento", "Perguntas alheias que ele aparece", "Ao acaso, esperado"], rows,
                     [0.62, 0.2, 0.18], ["l", "r", "r"], small=True)

    def _t_guardas(self, spec: dict) -> Table:
        return self._guards_table(self.run(spec["rodada"]))

    # ---------------------------------------------------------------- figuras

    def _figure(self, figure_id: str) -> list[Block]:
        spec = (self.meta.get("figuras") or {}).get(figure_id)
        if not spec:
            raise ReportError(f"a figura {figure_id!r} não está em `figuras`")
        kind = spec.get("tipo")
        builder = getattr(self, f"_f_{kind}", None)
        if builder is None:
            raise ReportError(f"figura de tipo desconhecido: {kind!r}")
        self.report.figures.mkdir(parents=True, exist_ok=True)
        path = self.report.figures / f"{(self.id or self.text.path.stem).lower()}-{figure_id}.png"
        table = builder(spec, path)
        blocks: list[Block] = []
        if spec.get("como_ler"):
            blocks.append(Callout("Como ler este gráfico", [self.resolve(spec["como_ler"])], tone="nota"))
        blocks.append(Figure(path, self.resolve(spec.get("legenda") or ""), width=spec.get("largura", 1.0)))
        if table is not None and spec.get("tabela", True):
            blocks.append(table)
        return blocks

    def _f_diferencas(self, spec: dict, path: Path) -> Table:
        """O gráfico das diferenças. `rodada` + `pares` (+ `metricas`), ou `linhas` com rodada, par, métrica e
        rótulo cada uma, para juntar rodadas diferentes (E7: o nomic e o bge-m3) num gráfico só."""
        labels = spec.get("rotulos") or {}
        lines = []
        if spec.get("linhas"):
            for line in spec["linhas"]:
                run = self.run(line["rodada"])
                metric = line.get("metrica") or (run.verdict(line["par"]) or {}).get("metric")
                lines.append((run, line["par"], metric, line.get("rotulo") or _pair_label(line["par"])))
        else:
            run = self.run(spec["rodada"])
            pairs = spec.get("pares") or list(run.placar.get("verdicts") or {})
            for pair in pairs:
                for metric in spec.get("metricas") or [(run.verdict(pair) or {}).get("metric")]:
                    lines.append((run, pair, metric, _pair_label(pair, labels)))
        several = len({metric for _, _, metric, _ in lines}) > 1
        one_pair = len({(run.name, pair) for run, pair, _, _ in lines}) == 1
        rows, table, units = [], [], set()
        for run, pair, metric, label in lines:
            delta, margin, verdict = run.judged(pair, metric)
            units.add(unit(metric))
            scale = 1000 if unit(metric) == "ms" else 1
            # Um par só (E9): a linha é a métrica; vários: o par, e a métrica quando ela muda de linha para linha.
            name = metric_name(metric) if one_pair and several else label + (f" · {metric_name(metric)}" if several
                                                                              else "")
            if not placar.higher_is_better(metric):
                name += " (menos é melhor)"
            rows.append({"label": name, "verdict": verdict, "value_text": fmt_point(metric, delta),
                         "delta": delta["delta"] / scale if delta else None,
                         "lo": delta["ci_low"] / scale if delta else None,
                         "hi": delta["ci_high"] / scale if delta else None,
                         "margin": margin / scale if margin is not None else None})
            table.append([label, metric_name(metric), fmt_delta(metric, delta), fmt_margin(metric, margin), verdict,
                          str(delta["n"]) if delta else "—"])
        if len(units) > 1:
            raise ReportError("métricas de unidades diferentes no mesmo gráfico (um eixo só)")
        axis = {"ms": "diferença em segundos", "tokens": "diferença em tokens"}.get(units.pop(), "diferença")
        charts.forest(rows, path, spec.get("eixo", f"{axis} (segundo braço − primeiro)"))
        return Table(["Par", "Métrica", "Δ [IC 95%]", "Margem", "Veredito", "n"], table,
                     [0.24, 0.2, 0.22, 0.1, 0.17, 0.07], ["l", "l", "r", "r", "l", "r"], small=True)

    def _f_por_grupo(self, spec: dict, path: Path) -> Table:
        run = self.run(spec["rodada"])
        labels, groups = spec.get("rotulos") or {}, spec.get("grupos") or {}
        rows, table = [], []
        for pair in spec["pares"]:
            verdict = run.verdict(pair) or {}
            metric, margin = verdict.get("metric"), verdict.get("margin")
            for cls, delta in (run.placar.get("by_class", {}).get(pair) or {}).items():
                if delta is None:
                    continue
                judged = placar.verdict(delta, margin, placar.higher_is_better(metric))
                name = f"{_pair_label(pair, labels)} · {groups.get(cls, cls)} (n={delta['n']})"
                rows.append({"label": name, "verdict": judged, "value_text": fmt_point(metric, delta),
                             "delta": delta["delta"], "lo": delta["ci_low"], "hi": delta["ci_high"], "margin": margin})
                table.append([_pair_label(pair, labels), groups.get(cls, cls), fmt_delta(metric, delta),
                              fmt_margin(metric, margin), judged, str(delta["n"])])
        charts.forest(rows, path, spec.get("eixo", "diferença (segundo braço − primeiro)"))
        return Table(["Par", "Grupo", "Δ [IC 95%]", "Margem", "Veredito", "n"], table,
                     [0.24, 0.2, 0.22, 0.1, 0.17, 0.07], ["l", "l", "r", "r", "l", "r"], small=True)

    def _f_medias(self, spec: dict, path: Path) -> Table:
        run, labels = self.run(spec["rodada"]), spec.get("rotulos") or {}
        panels = []
        for metric in spec["metricas"]:
            bars = [(labels.get(a, a), run.mean(a, metric), fmt_value(metric, run.mean(a, metric)))
                    for a in spec["bracos"]]
            panels.append({"title": metric_name(metric), "bars": bars, "xmax": 1.0 if unit(metric) == "nota" else None})
        charts.small_bars(panels, path)
        return self._t_medias(spec)

    def _f_margens(self, spec: dict, path: Path) -> Table:
        run = self.run(spec["rodada"])
        margins = run.placar.get("margins") or {}
        groups = spec["grupos"]
        metrics = spec["metricas"]
        series = [(label, [margins.get(group, {}).get(m) for m in metrics]) for group, label in groups.items()]
        charts.grouped_bars([metric_name(m) for m in metrics], series, path,
                            spec.get("eixo", "o quanto a nota muda sozinha (P95 de |Δ|)"))
        rows = [[metric_name(m)] + [num(margins.get(g, {}).get(m), 3) if margins.get(g, {}).get(m) is not None
                                    else "—" for g in groups] for m in metrics]
        n = len(groups)
        return Table(["Métrica"] + list(groups.values()), rows, [0.4] + [0.6 / n] * n, ["l"] + ["r"] * n, small=True)

    def _f_curva(self, spec: dict, path: Path) -> Table:
        run, labels = self.run(spec["rodada"]), spec.get("rotulos") or {}
        curves = run.placar.get("curves") or {}
        series, rows = [], []
        ks = spec.get("ks", [1, 2, 4, 8, 10])
        for arm in spec["bracos"]:
            points = curves.get(arm) or []
            if not points:
                raise ReportError(f"o braço {arm} não tem curva no placar")
            series.append({"name": labels.get(arm, arm), "points": [(p["tokens"], p["evidence"]) for p in points]})
            by_k = {p["k"]: p for p in points}
            rows.append([labels.get(arm, arm)] + [f"{num(by_k[k]['evidence'])} ({num(by_k[k]['tokens'], 0)})"
                                                 if k in by_k else "—" for k in ks])
        mark_k = spec.get("marca_k", 4)
        marked = {p["k"]: p for p in curves.get(spec.get("marca_braco", spec["bracos"][0]), [])}
        marks = [{"x": marked[mark_k]["tokens"], "y": marked[mark_k]["evidence"],
                  "text": spec.get("marca_texto", f"k = {mark_k} (o padrão)")}] if mark_k in marked else []
        charts.lines(series, path, "tokens de contexto que o gerador recebe", "evidência nos trechos",
                     ylim=(0, 1.02), marks=marks)
        n = len(ks)
        return Table(["Braço"] + [f"k = {k}" for k in ks], rows, [0.25] + [0.75 / n] * n, ["l"] + ["r"] * n,
                     small=True, caption="Evidência nos k primeiros trechos e, entre parênteses, os tokens de contexto.")

    def _f_lorenz(self, spec: dict, path: Path) -> Table | None:
        labels = spec.get("rotulos") or {}
        results = {i: self.analysis(i) for i in spec["analises"]}
        series = [{"name": labels.get(i, i), "points": r["lorenz"]["observed"]} for i, r in results.items()]
        first = next(iter(results.values()))
        references = [{"name": "esperado pelo desenho das perguntas", "points": first["lorenz"]["expected"]},
                      {"name": "igualdade", "points": [(0.0, 0.0), (1.0, 1.0)]}]
        charts.lines(series, path, "fração dos documentos (do que menos aparece ao que mais aparece)",
                     "fração acumulada dos trechos", references=references, xlim=(0, 1), ylim=(0, 1))
        return None

    def _f_roc(self, spec: dict, path: Path) -> Table:
        run = self.run(spec["rodada"])
        series, rows = [], []
        for curve in spec["curvas"]:
            result = outside_separation(run, curve["braco"], curve["nota"])
            name = curve.get("rotulo", f"{curve['braco']} · {curve['nota']}")
            series.append({"name": f"{name} (AUC {num(result['auc'])})", "points": result["points"]})
            rows.append([name, num(result["auc"]), str(result["n_inside"]), str(result["n_outside"])])
        charts.lines(series, path, "perguntas de fora que passariam pelo corte", "perguntas de dentro que passam",
                     references=[{"name": "acaso", "points": [(0.0, 0.0), (1.0, 1.0)]}],
                     xlim=(0, 1), ylim=(0, 1.02))
        return Table(["Nota do primeiro trecho", "AUC", "De dentro", "De fora"], rows, [0.55, 0.15, 0.15, 0.15],
                     ["l", "r", "r", "r"], small=True)

    def _f_categorias(self, spec: dict, path: Path) -> Table:
        run, labels = self.run(spec["rodada"]), spec.get("rotulos") or {}
        rows = []
        for arm in spec["bracos"]:
            counts = (run.placar.get("summary", {}).get(arm) or {}).get("attribution")
            if not counts:
                raise ReportError(f"o braço {arm} não tem as categorias de atribuição (rodou com citação?)")
            rows.append((labels.get(arm, arm), [float(counts.get(key, 0)) for key, _ in ATTRIBUTION]))
        charts.stacked_bars(rows, [name for _, name in ATTRIBUTION], path, "respostas às perguntas de dentro")
        n = len(rows)
        return Table(["Categoria"] + [r[0] for r in rows],
                     [[name] + [num(r[1][j], 0) for r in rows] for j, (_, name) in enumerate(ATTRIBUTION)],
                     [0.4] + [0.6 / n] * n, ["l"] + ["r"] * n, small=True)

    def _f_custo_qualidade(self, spec: dict, path: Path) -> Table:
        run, labels = self.run(spec["rodada"]), spec.get("rotulos") or {}
        x_metric = spec.get("x", "cost.generation_ms")
        names = [labels.get(a, a) for a in spec["bracos"]]
        panels = []
        for metric in spec["y"]:
            points = {}
            for arm in spec["bracos"]:
                x, y = run.mean(arm, x_metric), run.mean(arm, metric)
                if x is not None and y is not None:
                    points[labels.get(arm, arm)] = (x / 1000 if unit(x_metric) == "ms" else x, y)
            panels.append({"title": metric_name(metric), "xlabel": f"{metric_name(x_metric)} (s)", "points": points})
        charts.scatter_panels(panels, names, path)
        rows = [[labels.get(a, a), fmt_value(x_metric, run.mean(a, x_metric))]
                + [fmt_value(m, run.mean(a, m)) for m in spec["y"]] for a in spec["bracos"]]
        n = len(spec["y"]) + 1
        return Table(["Braço", metric_name(x_metric)] + [metric_name(m) for m in spec["y"]], rows,
                     [0.25] + [0.75 / n] * n, ["l"] + ["r"] * n, small=True)


# ------------------------------------------------------------------ o documento


def _appendix(report: Report) -> list[Block]:
    rows = []
    for run in sorted(report.runs_used(), key=lambda r: r.name):
        split = run.params.get("split") or "todas"
        rows.append([f"`{run.name}`", run.card.get("id", "—"), _date(run.started_at()), _duration(run.duration_seconds()),
                     split, run.judge() or "—"])
    blocks: list[Block] = [PageBreak(), Heading(2, "Apêndice: de onde vêm os números"),
                           Para("Cada número deste relatório foi lido das rodadas abaixo, em `tests/benchmark/out/`. "
                                "Nenhum foi digitado à mão: os textos têm marcadores que o gerador preenche."),
                           Table(["Rodada", "Experimento", "Quando", "Duração", "Perguntas", "Juiz"], rows,
                                 [0.33, 0.11, 0.17, 0.11, 0.12, 0.16], small=True)]
    if report.warnings:
        blocks.append(Callout(f"Avisos da geração ({len(report.warnings)})", report.warnings))
    else:
        blocks.append(Para("Nenhum aviso: todos os marcadores foram preenchidos, e os vereditos das rodadas batem "
                           "com os que os textos afirmam.", small=True))
    return blocks


def build_document(manifest_path: Path, out_dir: Path, out_root: Path | None = None,
                   manifests: Path = MANIFESTS) -> tuple[list[Block], Report]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    base = manifest_path.parent
    root = out_root or (base / manifest["pasta_rodadas"] if manifest.get("pasta_rodadas") else OUT_ROOT)
    report = Report(base, root, out_dir / "figuras", manifests)
    report.sections = [Section(report, read_text(base / rel)) for rel in manifest.get("textos") or []]
    blocks: list[Block] = [Heading(1, manifest.get("titulo", "Relatório do benchmark"))]
    if manifest.get("subtitulo"):
        blocks.append(Para(" ".join(str(manifest["subtitulo"]).split())))
    blocks.append(Para(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `tests/reports/benchmark_report.py`, "
                       f"a partir de {len({n for s in report.sections for n in (s.meta.get('rodadas') or {}).values()})} "
                       "rodadas do benchmark.", small=True))
    if manifest.get("avisos"):
        blocks.append(Callout(manifest.get("titulo_avisos", "Leia antes"),
                              [" ".join(str(item).split()) for item in manifest["avisos"]]))
    for i, section in enumerate(report.sections):
        section_blocks = section.blocks()
        if i == 0 and section_blocks and isinstance(section_blocks[0], PageBreak):
            section_blocks = section_blocks[1:]  # a primeira seção começa na capa
        blocks += section_blocks
    return blocks + _appendix(report), report


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Relatório didático do benchmark, a partir das rodadas gravadas.")
    parser.add_argument("--manifesto", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None, help="pasta de saída (padrão: tests/benchmark/out/relatorio_<data>)")
    parser.add_argument("--formatos", nargs="+", choices=("pdf", "md"), default=["pdf", "md"])
    parser.add_argument("--abrir", action="store_true", help="abre o PDF no fim")
    args = parser.parse_args(argv)

    out = args.out or OUT_ROOT / f"relatorio_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        blocks, report = build_document(args.manifesto.resolve(), out)
    except (ReportError, FileNotFoundError, KeyError) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    title = yaml.safe_load(args.manifesto.read_text(encoding="utf-8")).get("titulo", "Relatório do benchmark")
    written = []
    if "md" in args.formatos:
        from report_markdown import render_markdown

        written.append(render_markdown(blocks, out / "relatorio-benchmark.md"))
    if "pdf" in args.formatos:
        from report_pdf import render_pdf

        written.append(render_pdf(blocks, out / "relatorio-benchmark.pdf", title=title,
                                  author="tests/reports/benchmark_report.py"))
    for warning in report.warnings:
        print(f"[AVISO] {warning}")
    for path in written:
        print(f"Gerado: {path}")
    if args.abrir and "pdf" in args.formatos and hasattr(os, "startfile"):
        os.startfile(out / "relatorio-benchmark.pdf")  # noqa: S606 - abrir o PDF gerado, a pedido
    return 0


if __name__ == "__main__":
    sys.exit(main())
