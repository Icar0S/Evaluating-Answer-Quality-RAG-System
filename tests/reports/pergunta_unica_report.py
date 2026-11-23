"""O relatório da pergunta única (plano-escopo-final.md, §10): Markdown, PDF e planilha de uma rodada.

Lê a pasta de uma rodada de tests/pergunta_unica/rodar.py (respostas.json, os
exports dos três juízes, os pares) e monta as seções do plano:

1. o protocolo: o que foi fixado e o que foi conferido;
2. as 5 respostas, lado a lado;
3. os três juízes: as notas por resposta e a decomposição da variação (gerador x juiz);
4. a anatomia de uma nota: para a resposta #1, os passos da fidelidade em cada juiz;
5. os pares e os vieses (posição, verbosidade, concordância);
6. os mutantes e o checklist, quando a rodada os tiver.

O conteúdo sai uma vez, como lista de blocos (report_document.py), e o Markdown e
o PDF renderizam a mesma lista. As contas são as de tests/pergunta_unica
(analise.py e pares.py): o relatório só as mostra.

Rodar da raiz (venv do relatório), ou pelo build_report.py:
    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\build_report.py --pergunta-unica <pasta da rodada>
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (HERE, ROOT / "tests" / "pergunta_unica", ROOT / "tests" / "checklist"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analise  # noqa: E402
import avaliar  # noqa: E402
import mutacao  # noqa: E402
import pares  # noqa: E402
import report_charts as charts  # noqa: E402
from report_document import Block, Bullets, Callout, Figure, Heading, Para, Table, clip  # noqa: E402
from report_markdown import render_markdown  # noqa: E402
from report_pdf import render_pdf  # noqa: E402

TITLE = "A pergunta única: cinco respostas, três juízes e os pares"
FW_LABEL = {"ragas": "RAGAS", "deepeval": "DeepEval", "jeveval": "JevEval"}
METRIC_LABEL = {
    "faithfulness": "Fidelidade",
    "answer_relevancy": "Relevância da resposta",
    "context_precision": "Precisão do contexto",
    "context_recall": "Recall do contexto",
    "context_relevance": "Relevância do contexto",
    "correctness": "Correção",
}
JUDGE_LABEL = {"deepeval-arena": "DeepEval (ArenaGEval)", "jeveval-par": "JevEval (Choice A/B/Empate)"}


def num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def short(name: str) -> str:
    """`md-mtbench-vieses#3` -> `#3`."""
    return "#" + name.split("#")[-1] if "#" in name else name


# ------------------------------------------------------------------- os dados


@dataclass
class SingleQuestionRun:
    dir: Path
    meta: dict[str, Any]
    answers: list[dict[str, Any]]
    exports: list[dict[str, Any]]
    variance: dict[str, Any]
    pairs: dict[str, Any] | None
    pair_exports: dict[str, dict[str, Any]] = field(default_factory=dict)
    control: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    mutants: dict[str, Any] | None = None

    @property
    def protocol(self) -> dict[str, Any]:
        return self.meta.get("protocolo") or {}

    def exports_of(self, framework: str) -> list[dict[str, Any]]:
        found = [e for e in self.exports if analise.framework_of(e) == framework]
        return sorted(found, key=lambda e: (e.get("judge_repetition") is None, e.get("judge_repetition")))


def load(run_dir: Path, control: Path | None = None) -> SingleQuestionRun:
    """A rodada inteira. A decomposição e os pares são recalculados dos exports (a fonte), não lidos do resumo."""
    run_dir = Path(run_dir)
    answers_file = run_dir / "respostas.json"
    if not answers_file.is_file():
        raise ValueError(f"{run_dir} não é uma rodada da pergunta única (falta respostas.json)")
    meta = json.loads(answers_file.read_text(encoding="utf-8"))
    exports = [analise.load_export(p) for p in analise.find_exports(run_dir)]
    pair_exports = {}
    for path in sorted((run_dir / "pares").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pair_exports[data.get("juiz") or path.stem] = data
    _, by_name = pares.answers_by_name(answers_file)
    names = sorted(by_name)
    implicit = {}
    for fw in analise.FRAMEWORKS:
        means = pares.pointwise_means([e for e in exports if analise.framework_of(e) == fw])
        if means:
            implicit[fw] = pares.implicit_judgments(means, names)
    judges = {name: data["julgamentos"] for name, data in pair_exports.items()}
    pairs_report = pares.pairs_report(judges, by_name, implicit) if (judges or implicit) else None
    execution_file = run_dir / "execucao.json"
    return SingleQuestionRun(
        dir=run_dir,
        meta=meta,
        answers=list(meta.get("answers", [])),
        exports=exports,
        variance=analise.variance_report(exports),
        pairs=pairs_report,
        pair_exports=pair_exports,
        control=json.loads(Path(control).read_text(encoding="utf-8")) if control else None,
        execution=json.loads(execution_file.read_text(encoding="utf-8")) if execution_file.is_file() else None,
        mutants=(json.loads((run_dir / "mutantes-analise.json").read_text(encoding="utf-8"))
                 if (run_dir / "mutantes-analise.json").is_file() else None),
    )


# ------------------------------------------------------------------ a figura


def scores_chart(run: SingleQuestionRun, path: Path, metrics: tuple[str, ...] = ("faithfulness", "answer_relevancy"),
                 threshold: float = 0.7) -> Path | None:
    """Por métrica (um painel cada): a nota média de cada resposta em cada juiz, com a faixa dos julgamentos.

    Cor pelo framework (o slot fixo do relatório consolidado); o ponto é a média dos K julgamentos e o traço vai
    do mínimo ao máximo: o comprimento do traço é a variação do juiz, a distância entre as respostas é a do
    gerador. A tabela da decomposição vem logo abaixo no relatório.
    """
    fws = [fw for fw in analise.FRAMEWORKS if fw in run.variance["frameworks"]]
    if not fws:
        return None
    names = [a["name"] for a in run.answers]
    colors = charts.framework_colors([FW_LABEL[fw] for fw in fws])
    plt.rcParams.update({"font.family": charts.FONT["family"], "font.size": charts.FONT["size"]})
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 2.6), dpi=charts.DPI, sharey=True)
    axes = [axes] if len(metrics) == 1 else list(axes)
    fig.patch.set_facecolor(charts.SURFACE)
    width = 0.22
    for ax, metric in zip(axes, metrics):
        charts._style(ax)
        ax.axhline(threshold, color=charts.BASELINE, linewidth=0.75, linestyle=(0, (3, 3)), zorder=1)
        for i, fw in enumerate(fws):
            summary = run.variance["frameworks"][fw]["metricas"].get(metric)
            if not summary:
                continue
            offset = (i - (len(fws) - 1) / 2) * width
            for x, name in enumerate(names):
                lo_hi = summary["faixa_por_resposta"].get(name)
                mean = summary["media_por_resposta"].get(name)
                if mean is None:
                    continue
                if lo_hi and lo_hi[0] != lo_hi[1]:
                    ax.plot([x + offset] * 2, lo_hi, color=colors[FW_LABEL[fw]], linewidth=2, solid_capstyle="round",
                            zorder=2)
                ax.scatter([x + offset], [mean], s=30, color=colors[FW_LABEL[fw]], edgecolor=charts.SURFACE,
                           linewidth=1.5, zorder=3, label=FW_LABEL[fw] if x == 0 else None)
        ax.set_title(METRIC_LABEL.get(metric, metric), color=charts.INK, fontsize=charts.FONT["size"] + 1, loc="left")
        ax.set_xticks(range(len(names)), [short(n) for n in names])
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.grid(True, color=charts.GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=charts.FONT["size"],
                   labelcolor=charts.INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=charts.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------- as seções


def _protocol_blocks(run: SingleQuestionRun) -> list[Block]:
    p, meta = run.protocol, run.meta
    trechos = p.get("trechos") or {}
    same = all(t.get("iguais") for t in trechos.values()) if trechos else None
    hashes = ", ".join(f"`{t.get('sha256')}` ({t.get('respostas')} respostas)" for t in trechos.values()) or "—"
    rows = [
        ["Pergunta isolada", "Cada resposta é uma chamada nova, só com a pergunta: o histórico do chat nunca vai ao "
                             "modelo", "`chat-limpo.spec.ts` (E2E)"],
        ["Servidor limpo", "O gerador sai da memória do Ollama antes de cada resposta (`keep_alive: 0`)",
         "sim" if p.get("estado_limpo") else "**não**"],
        ["Sementes", ", ".join(map(str, p.get("sementes") or [])) or "—", "a semente de cada resposta voltou no metadata"],
        ["Temperatura", num(p.get("temperatura"), 1), p.get("temperatura_fonte") or "—"],
        ["Os mesmos trechos", "O hash do id e do texto dos trechos, por resposta",
         ("iguais nas respostas: " if same else "**diferentes**: ") + hashes if same is not None else "—"],
        ["Gerador", f"`{meta.get('generation_model')}` ({meta.get('provider')}), índice `{meta.get('rag_index')}`, "
                    f"top-k {meta.get('top_k')}, prompt {meta.get('prompt_version')}", "—"],
        ["Interrompido", p.get("interrompido") or "não", "—"],
    ]
    blocks: list[Block] = [
        Heading(2, "1. O protocolo"),
        Para("Tudo o que pode mudar a resposta, menos o gerador, fica fixo; e o que é fixo, é conferido."),
        Table(["Item", "O que foi fixado", "Conferido"], rows, [0.18, 0.5, 0.32], caption="O protocolo da rodada"),
    ]
    if run.control:
        summary = run.control.get("controle_estado") or {}
        blocks += [
            Heading(3, "O controle quente × frio"),
            Para(f"Por semente, a resposta com o modelo recarregado (frio), de novo fria, e logo depois de outra "
                 f"pergunta, sem recarregar (quente). O frio se reproduziu em **{summary.get('frio_se_reproduz')}** "
                 f"sementes; o estado do servidor mudou a resposta em **{summary.get('estado_mudou_a_resposta')}**."),
        ]
    return blocks


def _answers_blocks(run: SingleQuestionRun) -> list[Block]:
    rows = [[short(a["name"]), str(a.get("seed") if a.get("seed") is not None else "—"),
             str(len(a.get("actual_output") or "")), str(a.get("completion_tokens") or "—"),
             num((a.get("generation_ms") or 0) / 1000, 1) + " s"] for a in run.answers]
    blocks: list[Block] = [
        Heading(2, "2. As cinco respostas"),
        Para(f"**Pergunta:** {run.answers[0]['input'] if run.answers else '—'}"),
        Para(f"**Referência:** {run.answers[0].get('expected_output') if run.answers else '—'}"),
        Table(["Resposta", "Semente", "Caracteres", "Tokens", "Geração"], rows, [0.2, 0.2, 0.2, 0.2, 0.2],
              align=["l", "r", "r", "r", "r"], caption="As respostas, uma por semente"),
    ]
    for a in run.answers:
        blocks += [Heading(3, f"{short(a['name'])} (semente {a.get('seed')})"),
                   Para(" ".join((a.get("actual_output") or "(sem resposta)").split()), small=True)]
    return blocks


def _judges_blocks(run: SingleQuestionRun, figure: Path | None) -> list[Block]:
    blocks: list[Block] = [
        Heading(2, "3. Os três juízes"),
        Para("Cada resposta foi julgada K vezes por juiz, com a ordem das respostas embaralhada com semente a cada "
             "julgamento. A variação entre as respostas é a do **gerador**; a variação entre os julgamentos da mesma "
             "resposta é a do **juiz**."),
    ]
    if figure:
        blocks.append(Figure(figure, "Nota média de cada resposta, por juiz (ponto), e a faixa dos julgamentos (traço). "
                                     "A linha tracejada é o limiar de 0,7."))
    rows = []
    for fw, data in run.variance["frameworks"].items():
        for metric, s in data["metricas"].items():
            extra = " ".join(x for x in (f"{s['nao_se_aplica']} N/A" if s["nao_se_aplica"] else "",
                                         f"{s['nao_medidas']} não medidas" if s["nao_medidas"] else "") if x)
            rows.append([FW_LABEL.get(fw, fw), METRIC_LABEL.get(metric, metric),
                         num(s["media"]) + (f" ({extra})" if extra else ""),
                         f"{num(s['minimo'])}–{num(s['maximo'])}" if s["minimo"] is not None else "—",
                         num(s["dp_gerador"]), num(s["dp_juiz"]), str(len(s["vereditos_instaveis"])), s["domina"] or "—"])
    blocks += [
        Table(["Juiz", "Métrica", "Média", "Faixa", "DP gerador", "DP juiz", "Vereditos instáveis", "Domina"], rows,
              [0.11, 0.2, 0.14, 0.12, 0.1, 0.09, 0.12, 0.12], align=["l", "l", "r", "c", "r", "r", "r", "l"],
              caption="A decomposição da variação", small=True),
        Callout("Como ler", [
            "**DP gerador:** o desvio-padrão das médias das 5 respostas. **DP juiz:** a raiz da média das variâncias "
            "de cada resposta entre os julgamentos. **Domina:** o maior dos dois.",
            "**Vereditos instáveis:** respostas que passaram no limiar num julgamento e reprovaram noutro.",
            "As métricas de contexto julgam os trechos, que são os mesmos nas 5 respostas: nelas, toda variação é do "
            "juiz por construção.",
            "Com 5 respostas, a leitura é descritiva: médias, faixas e desvios, sem p-valor.",
        ], tone="nota"),
    ]
    return blocks


def _steps(export: dict[str, Any], answer: str, metric: str) -> tuple[dict | None, dict | None]:
    for case in export.get("cases", []):
        if case["name"] == answer:
            for m in case.get("metrics", []):
                if analise.canonical_metric(m["name"]) == metric:
                    return m, m.get("steps")
    return None, None


def anatomy(run: SingleQuestionRun, answer: str | None = None, metric: str = "faithfulness") -> dict[str, Any]:
    """Os passos da nota de uma resposta em cada juiz (o primeiro julgamento): as linhas e a conta."""
    answer = answer or (run.answers[0]["name"] if run.answers else None)
    out: dict[str, Any] = {"resposta": answer, "metrica": metric}
    ragas = run.exports_of("ragas")
    if ragas:
        m, steps = _steps(ragas[0], answer, metric)
        nli = next((s["output"]["statements"] for s in steps or [] if s.get("schema") == "NLIStatementOutput"), None)
        if m and nli is not None:
            ok = sum(1 for s in nli if s.get("verdict") == 1)
            out["ragas"] = {"nota": m.get("score"), "linhas": [[s["statement"], str(s.get("verdict")), s.get("reason", "")]
                                                               for s in nli],
                            "conta": f"inferíveis ÷ total = {ok} ÷ {len(nli)} = {num(ok / len(nli)) if nli else '—'}"}
    deepeval = run.exports_of("deepeval")
    if deepeval:
        m, steps = _steps(deepeval[0], answer, metric)
        if m and steps and steps.get("claims") is not None:
            verdicts = steps.get("verdicts") or []
            kept = sum(1 for v in verdicts if str(v.get("verdict", "")).strip().lower() != "no")
            out["deepeval"] = {
                "nota": m.get("score"),
                "linhas": [[claim, str(v.get("verdict")), v.get("reason") or ""]
                           for claim, v in zip(steps["claims"], verdicts)],
                "verdades": len(steps.get("truths") or []),
                "conta": f"(yes + idk) ÷ total = {kept} ÷ {len(verdicts)} = "
                         f"{num(kept / len(verdicts)) if verdicts else '—'}",
                "justificativa": m.get("reason"),
            }
    jev = run.exports_of("jeveval")
    if jev:
        m, _ = _steps(jev[0], answer, metric)
        breakdown = (m or {}).get("breakdown") or []
        if m and breakdown:
            applicable = [b for b in breakdown if b.get("applicable")]
            total_weight = sum(b["weight"] for b in applicable)
            weighted = sum(b["weight"] * b["value"] for b in applicable)
            out["jeveval"] = {
                "nota": m.get("score"),
                "linhas": [[b["question"].split("?")[0] + "?", num(b["weight"], 0),
                            ", ".join(f"{opt.split(':')[0]} {num(p)}" for opt, p in b["probabilities"].items() if p),
                            num(b["value"])] for b in breakdown],
                "conta": " + ".join(f"{num(b['weight'], 0)} × {num(b['value'])}" for b in applicable)
                         + f" ÷ {num(total_weight, 0)} = {num(weighted / total_weight) if total_weight else '—'}",
                "confianca": m.get("confidence"),
            }
    return out


def _anatomy_blocks(run: SingleQuestionRun) -> list[Block]:
    a = anatomy(run)
    blocks: list[Block] = [
        Heading(2, "4. A anatomia de uma nota"),
        Para(f"A fidelidade da resposta **{short(a['resposta'] or '—')}**, no primeiro julgamento de cada juiz: o "
             f"mesmo nome, três contas diferentes."),
    ]
    if "ragas" in a:
        r = a["ragas"]
        blocks += [Heading(3, f"RAGAS: {num(r['nota'])}"),
                   Para("O juiz quebra a resposta em afirmações e diz se cada uma **pode ser inferida diretamente** "
                        "dos trechos (1) ou não (0)."),
                   Table(["Afirmação", "Inferível", "Motivo do juiz"], [[clip(x, 220), v, clip(y, 220)] for x, v, y in
                                                                        r["linhas"]],
                         [0.45, 0.1, 0.45], small=True),
                   Para(f"**Conta:** {r['conta']}.")]
    if "deepeval" in a:
        d = a["deepeval"]
        blocks += [Heading(3, f"DeepEval: {num(d['nota'])}"),
                   Para(f"O juiz extrai as afirmações da resposta e {d['verdades']} \"verdades\" dos trechos, e diz se "
                        f"cada afirmação é `yes`, `no` ou `idk` (os trechos não dizem). **Só o `no` perde ponto.**"),
                   Table(["Afirmação", "Veredito", "Motivo do juiz"], [[clip(x, 220), v, clip(y, 220)] for x, v, y in
                                                                       d["linhas"]],
                         [0.45, 0.1, 0.45], small=True),
                   Para(f"**Conta:** {d['conta']}.")]
    if "jeveval" in a:
        j = a["jeveval"]
        blocks += [Heading(3, f"JevEval: {num(j['nota'])}"),
                   Para("O Jev responde perguntas fechadas com uma probabilidade por opção. O valor de cada pergunta é a "
                        "soma das probabilidades vezes o crédito de cada opção; a nota é a média ponderada."),
                   Table(["Pergunta", "Peso", "Probabilidades", "Valor"], j["linhas"], [0.42, 0.08, 0.38, 0.12],
                         align=["l", "r", "l", "r"], small=True),
                   Para(f"**Conta:** {j['conta']}. Confiança mínima: {num(j['confianca'])}.")]
    if len(a) <= 2:
        blocks.append(Para("Os passos não estão nos exports desta rodada (rode os juízes com `--passos`)."))
    return blocks


def _pairs_blocks(run: SingleQuestionRun) -> list[Block]:
    if not run.pairs:
        return [Heading(2, "5. Os pares"), Para("A rodada não tem comparação em pares.")]
    report = run.pairs
    rows = []
    for judge, s in report["juizes"].items():
        pos = s["posicao"]
        rows.append([
            JUDGE_LABEL.get(judge, judge.replace(" (implícita)", " (implícita, das notas)")),
            ", ".join(f"{short(r['resposta'])} {num(r['taxa_de_vitoria'])}" for r in s["ranking"]),
            f"{pos['consistentes']}/{pos['pares_nas_duas_ordens']}" if pos["pares_nas_duas_ordens"] else "—",
            f"{num(pos['taxa_do_primeiro'] * 100 if pos['taxa_do_primeiro'] is not None else None, 0)}% de "
            f"{pos['decididos']}" if pos["decididos"] else "—",
            f"{num(s['verbosidade']['taxa_da_mais_longa'] * 100 if s['verbosidade']['taxa_da_mais_longa'] is not None else None, 0)}%"
            if s["verbosidade"]["taxa_da_mais_longa"] is not None else "—",
            num(s["verbosidade"]["spearman"]),
        ])
    between = [[pair.replace("deepeval-arena", "DeepEval").replace("jeveval-par", "Jev"),
                f"{a['iguais']}/{a['pares']}", num(a["kendall_tau"])] for pair, a in report["entre_juizes"].items()]
    point = [[JUDGE_LABEL.get(j, j), f"{a['iguais']}/{a['pares']}"] for j, a in report["pontual_x_pares"].items()]
    blocks: list[Block] = [
        Heading(2, "5. Os pares"),
        Para(f"**Critério** (escrito antes de rodar): {report['criterio']} Cada par foi julgado nas duas ordens. A "
             f"preferência implícita vem das notas da resposta ({', '.join(report['metricas_da_preferencia_implicita'])}"
             f"), com margem de {num(report['margem_implicita'])}."),
        Table(["Juiz", "Ranking (taxa de vitória)", "Mesmo vencedor nas duas ordens", "Vence quem vem primeiro",
               "Vence a mais longa", "Spearman tamanho × preferência"], rows, [0.2, 0.3, 0.13, 0.13, 0.12, 0.12],
              align=["l", "l", "c", "c", "c", "r"], caption="Ranking e vieses, por juiz", small=True),
        Table(["Juízes", "Pares com o mesmo resultado", "τ de Kendall dos rankings"], between, [0.6, 0.2, 0.2],
              align=["l", "c", "r"], caption="Concordância entre os juízes", small=True),
    ]
    if point:
        blocks.append(Table(["Juiz de pares", "Concorda com as notas do mesmo framework"], point, [0.6, 0.4],
                            align=["l", "c"], caption="Pontual × pares", small=True))
    blocks.append(Callout("Como ler", [
        "**Mesmo vencedor nas duas ordens:** o par (A, B) e o par (B, A) deram o mesmo vencedor. É a medida do viés de "
        "posição do artigo do MT-Bench; 100% é um juiz sem viés de posição.",
        "**Vence quem vem primeiro:** entre os julgamentos com vencedor, quantos o primeiro levou. Sem viés, perto de "
        "50%.",
        "**Vence a mais longa:** entre os pares de tamanhos diferentes, quantos a resposta mais longa levou.",
        "O ArenaGEval não aceita empate; o Jev e a preferência implícita aceitam.",
    ], tone="nota"))
    return blocks


def _mutants_blocks(run: SingleQuestionRun) -> list[Block]:
    blocks: list[Block] = [Heading(2, "6. Os mutantes")]
    if not run.mutants:
        return blocks + [Para("A rodada não tem mutantes (rodar.py --mutantes C1 C2 P1 R1).")]
    rows = []
    original = run.mutants["original"]
    rows.append(["original", "—", f"{original['abstencoes']}/{original['respostas']}",
                 f"{original['com_o_artigo_nos_trechos']}/{original['respostas']}", "—", "—", "—"])
    for op, m in run.mutants["mutantes"].items():
        cells = []
        for fw in analise.FRAMEWORKS:
            j = m["juizes"].get(fw)
            if not j:
                cells.append("—")
            elif j["morto"]:
                cells.append("**morto**: " + ", ".join(
                    f"{mutacao.METRIC_SHORT.get(k, k)} {num(j['metricas'][k]['antes'])}→{num(j['metricas'][k]['depois'])}"
                    for k in j["matou_por"]))
            else:
                cells.append("vivo")
        b = m["comportamento"]
        rows.append([f"{op}: {m['nome']}", m["esperado"] or "—", f"{b['abstencoes']}/{b['respostas']}",
                     f"{b['com_o_artigo_nos_trechos']}/{b['respostas']}", *cells])
    blocks += [
        Para("Cada mutante injeta um defeito na pergunta única e roda o mesmo protocolo: as 5 respostas, com as "
             "mesmas sementes, e os três juízes. Um juiz **mata** o mutante quando reprova ao menos uma resposta que "
             "aprovava no original."),
        Table(["Mutante", "O certo seria", "Abstenções", "O artigo nos trechos", "RAGAS", "DeepEval", "JevEval"], rows,
              [0.16, 0.2, 0.08, 0.09, 0.16, 0.16, 0.15], caption="Quem matou cada mutante", small=True),
    ]
    artifacts = []
    for op in run.mutants["mutantes"]:
        meta_file = run.dir / "mutantes" / op / "mutante.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            artifacts.append(f"**{op}** ({meta.get('defeito')}): " + ", ".join(f"`mutantes/{op}/{a}`"
                                                                            for a in meta.get("artefatos", [])))
    if artifacts:
        blocks += [Para("O que cada mutante gerou:"), Bullets(artifacts)]
    return blocks


def _checklist_blocks(run: SingleQuestionRun) -> list[Block]:
    results = avaliar.evaluate(avaliar.load_run(run.dir))
    rows = [[r["id"], r["etapa"], r["item"], r["tipo"], f"{avaliar.STATE_MARK[r['estado']]} {r['estado']}",
             clip(r["detalhe"], 260)] for r in results]
    counts = avaliar.summary(results)
    return [
        Heading(2, "7. O checklist"),
        Para("O checklist de testes de RAG (`tests/checklist/checklist.yaml`), no formato do CheckList (Ribeiro et al., "
             "2020): MFT é funcionalidade mínima, INV é invariância e DIR é expectativa de direção. Cada item é "
             "conferido por uma conta sobre esta rodada, por um teste automatizado, ou está fora do escopo com o "
             "motivo. Resumo: " + ", ".join(f"{n} {state}" for state, n in sorted(counts.items())) + "."),
        Table(["", "Etapa", "Item", "Tipo", "Estado", "Detalhe"], rows, [0.07, 0.11, 0.24, 0.06, 0.12, 0.4],
              align=["l", "l", "l", "c", "l", "l"], caption="O checklist da rodada", small=True),
    ]


def _files_blocks(run: SingleQuestionRun) -> list[Block]:
    items = [f"`{Path(e['_path']).relative_to(run.dir).as_posix()}` ({FW_LABEL.get(analise.framework_of(e))}, "
             f"julgamento {e.get('judge_repetition')})" for e in run.exports]
    items += [f"`pares/{Path(name).name}.json`" for name in run.pair_exports]
    return [Heading(2, "Arquivos da rodada"), Para(f"Pasta: `{run.dir.as_posix()}`"), Bullets(items or ["—"])]


def build_document(run: SingleQuestionRun, figure: Path | None, generated_at: datetime) -> list[Block]:
    reps = sorted({e.get("judge_repetition") for e in run.exports if e.get("judge_repetition") is not None})
    blocks: list[Block] = [
        Heading(1, TITLE),
        Para(f"Gerado em {generated_at:%d/%m/%Y %H:%M}. {len(run.answers)} respostas do "
             f"`{run.meta.get('generation_model')}`, {len(reps) or 1} julgamento(s) por resposta e por juiz "
             f"({', '.join(FW_LABEL[fw] for fw in run.variance['frameworks']) or 'nenhum juiz'})."),
    ]
    blocks += _protocol_blocks(run)
    blocks += _answers_blocks(run)
    blocks += _judges_blocks(run, figure)
    blocks += _anatomy_blocks(run)
    blocks += _pairs_blocks(run)
    blocks += _mutants_blocks(run)
    blocks += _checklist_blocks(run)
    blocks += _files_blocks(run)
    return blocks


# ------------------------------------------------------------------ planilha


def write_xlsx(run: SingleQuestionRun, path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Respostas"
    ws.append(["resposta", "semente", "caracteres", "tokens", "geracao_s", "texto"])
    for a in run.answers:
        ws.append([a["name"], a.get("seed"), len(a.get("actual_output") or ""), a.get("completion_tokens"),
                   round((a.get("generation_ms") or 0) / 1000, 1), a.get("actual_output")])
    notes = wb.create_sheet("Notas")
    notes.append(["juiz", "julgamento", "resposta", "metrica", "nota", "aprovada", "aplicavel", "erro"])
    for e in run.exports:
        for case in e.get("cases", []):
            for m in case.get("metrics", []):
                notes.append([analise.framework_of(e), e.get("judge_repetition"), case["name"],
                              analise.canonical_metric(m["name"]), m.get("score"), m.get("success"),
                              m.get("applicable", True), m.get("error")])
    decomposition = wb.create_sheet("Decomposicao")
    decomposition.append(["juiz", "metrica", "media", "minimo", "maximo", "dp_gerador", "dp_juiz", "amplitude_juiz_max",
                          "vereditos_instaveis", "domina", "nao_se_aplica", "nao_medidas"])
    for fw, data in run.variance["frameworks"].items():
        for metric, s in data["metricas"].items():
            decomposition.append([fw, metric, s["media"], s["minimo"], s["maximo"], s["dp_gerador"], s["dp_juiz"],
                                  s["amplitude_juiz_maxima"], len(s["vereditos_instaveis"]), s["domina"],
                                  s["nao_se_aplica"], s["nao_medidas"]])
    pairs_ws = wb.create_sheet("Pares")
    pairs_ws.append(["juiz", "par", "ordem", "primeiro", "segundo", "vencedor", "p_A", "p_B", "p_empate", "erro"])
    for name, data in run.pair_exports.items():
        for j in data.get("julgamentos", []):
            probs = j.get("probabilidades") or {}
            pairs_ws.append([name, " x ".join(j["par"]), j.get("ordem"), j["primeiro"], j["segundo"], j["vencedor"],
                             probs.get("A"), probs.get("B"), probs.get("Empate"), j.get("erro")])
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True)
    wb.save(path)
    return path


def build(run_dir: Path, out_dir: Path, formats: list[str], control: Path | None = None) -> dict[str, Path]:
    run = load(run_dir, control)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = scores_chart(run, out_dir / "figuras" / "notas_por_resposta.png")
    outputs: dict[str, Path] = {}
    if "md" in formats or "pdf" in formats:
        blocks = build_document(run, figure, datetime.now())
        if "md" in formats:
            outputs["md"] = render_markdown(blocks, out_dir / "relatorio.md")
        if "pdf" in formats:
            outputs["pdf"] = render_pdf(blocks, out_dir / "relatorio.pdf", title=TITLE,
                                        author="tests/reports/pergunta_unica_report.py")
    if "xlsx" in formats:
        outputs["xlsx"] = write_xlsx(run, out_dir / "relatorio.xlsx")
    return outputs
