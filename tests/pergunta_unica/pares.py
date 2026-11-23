"""A comparação em pares da pergunta única (plano-escopo-final.md, §6): o plano, o critério e as métricas.

Um juiz recebe duas respostas à mesma pergunta e diz qual é melhor. Com 5
respostas, são 10 pares; cada par é julgado nas duas ordens, (A, B) e (B, A):
20 julgamentos por juiz. Quem julga:

- o DeepEval, com o ArenaGEval e o qwen3:8b (tests/deepeval/pairwise.py): escolhe
  um vencedor, sem empate;
- o JevEval, com uma pergunta Choice "A", "B" ou "Empate" (tests/jeveval/pairwise.py):
  devolve a probabilidade de cada opção;
- a preferência IMPLÍCITA das notas pontuais (aqui): A vence B se a média das
  notas de A passa a de B por mais de MARGEM (D4); senão, empate. Vale para os
  três frameworks, e é o único jeito de o RAGAS entrar, porque ele não tem
  métrica de pares.

A preferência implícita usa só as métricas da RESPOSTA (fidelidade, relevância
da resposta e, no Jev, a correção): as de contexto (precisão, recall e relevância
do contexto) julgam os trechos, que são os mesmos nas 5 respostas por
construção (o protocolo confere o hash). Nelas, toda diferença entre as
respostas seria ruído do juiz.

O que se mede (sem p-valor: são 5 respostas):

- o ranking das 5 respostas pela taxa de vitória (vitória 1, empate 0,5);
- o viés de posição: a porcentagem de pares com o mesmo vencedor nas duas ordens
  e a taxa de vitória de quem aparece primeiro (a medida do artigo do MT-Bench);
- o viés de verbosidade: a correlação de Spearman entre a diferença de tamanho
  das duas respostas e a preferência;
- a concordância entre juízes: a porcentagem de pares com o mesmo resultado e o
  tau de Kendall entre os rankings;
- pontual x pares no mesmo framework: o par concorda com a preferência implícita
  das notas daquele framework?

Só biblioteca padrão: roda em qualquer venv e é importada pelos dois scripts de
pares e pelo relatório.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analise import canonical_metric

# D3: o critério, escrito antes de rodar. Em inglês, com o conteúdo em português, como os prompts do RAGAS e do
# DeepEval e as perguntas do Jev neste projeto (tests/jeveval/_jev_shared.py explica o porquê).
CRITERION_PT = (
    "Qual resposta responde melhor à pergunta, considerando a referência e sem afirmar nada que não esteja nos "
    "trechos?"
)
CRITERION = (
    "Which answer better answers the question in `input`, considering the reference answer in `expected_output`, "
    "without claiming anything that is not supported by the retrieved passages?"
)
# D4: a margem da preferência implícita.
MARGIN = 0.05
# As métricas da resposta, as únicas que entram na preferência implícita (ver o docstring).
ANSWER_METRICS = ("faithfulness", "answer_relevancy", "correctness")
TIE = "empate"


# ------------------------------------------------------------------- o plano


def pair_plan(names: list[str], both_orders: bool = True) -> list[dict[str, Any]]:
    """Os pares, em ordem canônica (a, b) com a < b, e as ordens a julgar: AB (a primeiro) e BA (b primeiro)."""
    plan = []
    for a, b in itertools.combinations(sorted(names), 2):
        plan.append({"par": [a, b], "ordem": "AB", "primeiro": a, "segundo": b})
        if both_orders:
            plan.append({"par": [a, b], "ordem": "BA", "primeiro": b, "segundo": a})
    return plan


def answers_by_name(answers_file: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """O arquivo do generate_answers.py e as respostas por nome (só as que têm texto)."""
    meta = json.loads(Path(answers_file).read_text(encoding="utf-8"))
    return meta, {a["name"]: a for a in meta.get("answers", []) if a.get("actual_output")}


def judgment(item: dict[str, Any], winner: str | None, **extra: Any) -> dict[str, Any]:
    """Um julgamento no formato comum dos três juízes. `winner`: o nome, TIE, ou None (não julgado)."""
    return {**item, "vencedor": winner, "probabilidades": None, "justificativa": None, "segundos": None,
            "erro": None, **extra}


# ----------------------------------------------------- preferência implícita


def pointwise_means(exports: list[dict[str, Any]], metrics: tuple[str, ...] = ANSWER_METRICS) -> dict[str, float]:
    """Por resposta: a média das notas medidas das métricas da resposta, em todos os julgamentos dados."""
    scores: dict[str, list[float]] = {}
    for export in exports:
        for case in export.get("cases", []):
            for m in case.get("metrics", []):
                # O DeepEval grava "Faithfulness", "Answer Relevancy": o mesmo nome dos outros dois, por extenso.
                if (canonical_metric(m["name"]) in metrics and m.get("score") is not None
                        and m.get("applicable", True)):
                    scores.setdefault(case["name"], []).append(float(m["score"]))
    return {name: statistics.fmean(s) for name, s in scores.items() if s}


def implicit_judgments(means: dict[str, float], names: list[str], margin: float = MARGIN) -> list[dict[str, Any]]:
    """O par decidido pelas notas pontuais: sem ordem (as notas não dependem de qual resposta vem primeiro)."""
    out = []
    for item in pair_plan(names, both_orders=False):
        a, b = item["par"]
        if a not in means or b not in means:
            out.append(judgment({**item, "ordem": None}, None, erro="sem nota pontual medida"))
            continue
        diff = means[a] - means[b]
        winner = a if diff > margin else b if diff < -margin else TIE
        out.append(judgment({**item, "ordem": None}, winner, diferenca=round(diff, 4)))
    return out


# -------------------------------------------------------------------- métricas


def _points(judgments: list[dict[str, Any]], names: list[str]) -> dict[str, float]:
    """Taxa de vitória: vitória 1, empate 0,5, sobre os julgamentos válidos de que a resposta participou."""
    won: dict[str, float] = {n: 0.0 for n in names}
    played: dict[str, int] = {n: 0 for n in names}
    for j in judgments:
        if j["vencedor"] is None:
            continue
        for n in j["par"]:
            played[n] += 1
            won[n] += 1.0 if j["vencedor"] == n else 0.5 if j["vencedor"] == TIE else 0.0
    return {n: round(won[n] / played[n], 4) for n in names if played[n]}


def ranking(judgments: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    rates = _points(judgments, names)
    return [{"resposta": n, "taxa_de_vitoria": r} for n, r in sorted(rates.items(), key=lambda kv: (-kv[1], kv[0]))]


def pair_outcome(judgments: list[dict[str, Any]]) -> dict[tuple[str, str], str | None]:
    """O resultado de cada par, juntando as ordens: quem fez mais pontos nas ordens julgadas (ou empate)."""
    points: dict[tuple[str, str], list[float]] = {}
    for j in judgments:
        if j["vencedor"] is None:
            continue
        key = tuple(j["par"])
        a, _ = key
        points.setdefault(key, []).append(1.0 if j["vencedor"] == a else 0.5 if j["vencedor"] == TIE else 0.0)
    out: dict[tuple[str, str], str | None] = {}
    for key, pts in points.items():
        mean = statistics.fmean(pts)
        out[key] = key[0] if mean > 0.5 else key[1] if mean < 0.5 else TIE
    return out


def position_bias(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Consistência nas duas ordens e a taxa de vitória de quem aparece primeiro (empates fora desta)."""
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for j in judgments:
        if j["vencedor"] is not None and j.get("ordem") in ("AB", "BA"):
            by_pair.setdefault(tuple(j["par"]), {})[j["ordem"]] = j["vencedor"]
    both = {k: v for k, v in by_pair.items() if "AB" in v and "BA" in v}
    consistent = [k for k, v in both.items() if v["AB"] == v["BA"]]
    decided = [j for j in judgments if j["vencedor"] not in (None, TIE) and j.get("ordem") in ("AB", "BA")]
    first_wins = sum(1 for j in decided if j["vencedor"] == j["primeiro"])
    ties = sum(1 for j in judgments if j["vencedor"] == TIE and j.get("ordem") in ("AB", "BA"))
    return {
        "pares_nas_duas_ordens": len(both),
        "consistentes": len(consistent),
        "consistencia": round(len(consistent) / len(both), 4) if both else None,
        "decididos": len(decided),
        "empates": ties,
        "vitorias_de_quem_vem_primeiro": first_wins,
        "taxa_do_primeiro": round(first_wins / len(decided), 4) if decided else None,
        "inconsistentes": [list(k) for k in both if k not in consistent],
    }


def _ranks(values: list[float]) -> list[float]:
    """Postos com empate pela média (para o Spearman e o Kendall)."""
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


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if statistics.pstdev(rx) == 0 or statistics.pstdev(ry) == 0:
        return None
    return round(statistics.correlation(rx, ry), 4)


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    """Tau-b de Kendall (com correção para empates)."""
    n = len(x)
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return round((concordant - discordant) / denom, 4) if denom else None


def verbosity_bias(judgments: list[dict[str, Any]], lengths: dict[str, int]) -> dict[str, Any]:
    """Spearman entre (tamanho de A - tamanho de B) e a preferência (+1 A, 0 empate, -1 B), na ordem canônica."""
    diffs, prefs = [], []
    longer_wins = decided = 0
    for j in judgments:
        if j["vencedor"] is None:
            continue
        a, b = j["par"]
        if a not in lengths or b not in lengths:
            continue
        diff = lengths[a] - lengths[b]
        pref = 1 if j["vencedor"] == a else -1 if j["vencedor"] == b else 0
        diffs.append(diff)
        prefs.append(pref)
        if pref != 0 and diff != 0:
            decided += 1
            longer_wins += (pref > 0) == (diff > 0)
    return {
        "spearman": spearman(diffs, prefs),
        "julgamentos": len(diffs),
        "vitorias_da_mais_longa": longer_wins,
        "decididos_com_tamanhos_diferentes": decided,
        "taxa_da_mais_longa": round(longer_wins / decided, 4) if decided else None,
    }


def agreement(outcomes_a: dict, outcomes_b: dict) -> dict[str, Any]:
    common = [k for k in outcomes_a if k in outcomes_b and outcomes_a[k] is not None and outcomes_b[k] is not None]
    same = [k for k in common if outcomes_a[k] == outcomes_b[k]]
    return {"pares": len(common), "iguais": len(same), "concordancia": round(len(same) / len(common), 4) if common
            else None}


def judge_summary(judgments: list[dict[str, Any]], names: list[str], lengths: dict[str, int]) -> dict[str, Any]:
    errors = [j for j in judgments if j["vencedor"] is None]
    return {
        "julgamentos": len(judgments),
        "nao_julgados": len(errors),
        "ranking": ranking(judgments, names),
        "posicao": position_bias(judgments),
        "verbosidade": verbosity_bias(judgments, lengths),
        "resultado_por_par": {f"{a} x {b}": w for (a, b), w in sorted(pair_outcome(judgments).items())},
    }


def pairs_report(judges: dict[str, list[dict[str, Any]]], answers: dict[str, dict[str, Any]],
                 implicit: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """O relatório dos pares: por juiz (os de pares e os implícitos), a concordância entre eles e pontual x pares.

    `judges`: juiz -> julgamentos (DeepEval, Jev). `implicit`: framework -> julgamentos implícitos.
    """
    names = sorted(answers)
    lengths = {n: len(a["actual_output"]) for n, a in answers.items()}
    everyone = {**judges, **{f"{fw} (implícita)": js for fw, js in (implicit or {}).items()}}
    summaries = {judge: judge_summary(js, names, lengths) for judge, js in everyone.items()}
    outcomes = {judge: pair_outcome(js) for judge, js in everyone.items()}
    rates = {judge: dict((r["resposta"], r["taxa_de_vitoria"]) for r in s["ranking"]) for judge, s in summaries.items()}
    between = {}
    for a, b in itertools.combinations(everyone, 2):
        shared = [n for n in names if n in rates[a] and n in rates[b]]
        between[f"{a} x {b}"] = {
            **agreement(outcomes[a], outcomes[b]),
            "kendall_tau": kendall_tau_b([rates[a][n] for n in shared], [rates[b][n] for n in shared]),
        }
    # Pontual x pares no mesmo framework: o par do DeepEval contra as notas do DeepEval, o do Jev contra as do Jev.
    same_framework = {}
    for judge in judges:
        fw = judge.split(" ")[0].split("-")[0]
        key = f"{fw} (implícita)"
        if key in outcomes:
            same_framework[judge] = agreement(outcomes[judge], outcomes[key])
    return {
        "criterio": CRITERION_PT,
        "criterio_enviado": CRITERION,
        "margem_implicita": MARGIN,
        "metricas_da_preferencia_implicita": list(ANSWER_METRICS),
        "respostas": {n: {"caracteres": lengths[n]} for n in names},
        "juizes": summaries,
        "entre_juizes": between,
        "pontual_x_pares": same_framework,
    }


def markdown(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.0f}%"

    def num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}".replace(".", ",")

    lines = ["| Juiz | Ranking (taxa de vitória) | Consistência nas duas ordens | Vitórias de quem vem primeiro | "
             "Spearman (tamanho x preferência) |", "| :--- | :--- | :---: | :---: | :---: |"]
    for judge, s in report["juizes"].items():
        rank = ", ".join(f"{r['resposta'].split('#')[-1]} ({num(r['taxa_de_vitoria'])})" for r in s["ranking"])
        pos = s["posicao"]
        consistency = (f"{pos['consistentes']}/{pos['pares_nas_duas_ordens']}" if pos["pares_nas_duas_ordens"]
                       else "—")
        first = f"{pct(pos['taxa_do_primeiro'])} de {pos['decididos']}" if pos["decididos"] else "—"
        lines.append(f"| {judge} | {rank} | {consistency} | {first} | {num(s['verbosidade']['spearman'])} |")
    lines += ["", "| Juízes | Pares com o mesmo resultado | τ de Kendall |", "| :--- | :---: | :---: |"]
    for pair, a in report["entre_juizes"].items():
        lines.append(f"| {pair} | {a['iguais']}/{a['pares']} | {num(a['kendall_tau'])} |")
    return "\n".join(lines)
