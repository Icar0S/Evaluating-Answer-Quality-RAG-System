"""Qual juiz percebe cada mutante da pergunta única (plano-escopo-final.md, §8).

Um juiz reage ao mutante numa métrica quando reprova ao menos uma das respostas que
aprovava no original. A comparação é resposta a resposta (os nomes são os mesmos:
`md-mtbench-vieses#1` … `#5`, as mesmas sementes), com a nota média dos
julgamentos de cada lado contra o limiar da métrica.

Reagir não é perceber: o juiz também muda de nota sem defeito nenhum. O controle
nulo diz quanto, de dois jeitos:

- **deixando um julgamento de fora:** com K julgamentos do ORIGINAL, o julgamento k
  faz o papel do mutante e os outros K-1 fazem o do original. É a mesma conta dos
  mutantes (um julgamento contra a média de vários), sem defeito. K comparações;
- **o N0** (mutantes.py): o mutante que não muda nada, pelo mesmo caminho dos outros.

A morte só conta se caíram MAIS respostas do que na pior comparação nula daquela
métrica. O estado do mutante para o juiz:

- `morto`: alguma métrica reprovou mais respostas do que reprova sem defeito nenhum;
- `ruído`: alguma reprovou, mas não mais do que no controle nulo;
- `sem controle`: alguma reprovou, e não há controle (um julgamento só, sem N0);
- `vivo`: nenhuma reprovou.

Além das notas, o comportamento do RAG no mutante: quantas respostas se
abstiveram, em quantas o artigo do MT-Bench ainda chegou nos trechos, quantas
saíram iguais às do original e quantos fatos certos (e quantos trocados, no C2)
cada uma cita (fatos.py, sem juiz). Um mutante que não muda a resposta não tem
defeito na resposta para o juiz perceber.

Só biblioteca padrão (importada pelo rodar.py e pelo relatório).
"""
from __future__ import annotations

import statistics
from typing import Any

from analise import FRAMEWORKS, canonical_metric, framework_of
from fatos import FACTS, SWAPPED_FACTS, facts_in

DOCUMENT_PREFIX = "NeurIPS-2023-judging-llm-as-a-judge"
NULL_OPERATOR = "N0"
METRIC_SHORT = {"faithfulness": "fidelidade", "answer_relevancy": "relevância", "context_precision": "precisão ctx",
                "context_recall": "recall ctx", "context_relevance": "relevância ctx", "correctness": "correção"}


def per_answer(exports: list[dict[str, Any]], framework: str) -> dict[str, dict[str, dict[str, Any]]]:
    """métrica -> resposta -> {media, aprovada, limiar}, com a média dos julgamentos medidos e aplicáveis."""
    scores: dict[str, dict[str, list[float]]] = {}
    thresholds: dict[str, float] = {}
    for export in exports:
        if framework_of(export) != framework:
            continue
        for case in export.get("cases", []):
            for m in case.get("metrics", []):
                metric = canonical_metric(m["name"])
                limit = m.get("threshold", export.get("threshold"))
                if limit is not None:
                    thresholds.setdefault(metric, float(limit))
                if m.get("score") is not None and m.get("applicable", True):
                    scores.setdefault(metric, {}).setdefault(case["name"], []).append(float(m["score"]))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for metric, by_answer in scores.items():
        limit = thresholds.get(metric, 0.7)
        out[metric] = {a: {"media": statistics.fmean(s), "aprovada": statistics.fmean(s) >= limit, "limiar": limit}
                       for a, s in by_answer.items()}
    return out


def behavior(answers_meta: dict[str, Any] | None, original_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    answers = (answers_meta or {}).get("answers", [])
    original = {a["name"]: a.get("actual_output") for a in (original_meta or {}).get("answers", [])}
    right = [len(facts_in(a.get("actual_output"))) for a in answers]
    swapped = [len(facts_in(a.get("actual_output"), SWAPPED_FACTS)) for a in answers]
    return {
        "respostas": len(answers),
        "abstencoes": sum(1 for a in answers if a.get("abstained")),
        "com_o_artigo_nos_trechos": sum(1 for a in answers
                                        if any(d.startswith(DOCUMENT_PREFIX) for d in a.get("source_documents") or [])),
        "caracteres_medios": round(statistics.fmean([len(a.get("actual_output") or "") for a in answers]), 1)
        if answers else None,
        "iguais_ao_original": sum(1 for a in answers if a["name"] in original
                                  and a.get("actual_output") == original[a["name"]]) if original else None,
        # De quantos: FACTS e SWAPPED_FACTS têm 4 fatos cada.
        "com_todos_os_fatos": sum(1 for n in right if n == len(FACTS)),
        "com_algum_fato_trocado": sum(1 for n in swapped if n),
        "fatos_certos_por_resposta": right,
        "fatos_trocados_por_resposta": swapped,
    }


def reactions(before_exports: list[dict[str, Any]], after_exports: list[dict[str, Any]],
              framework: str) -> dict[str, dict[str, Any]]:
    """Por métrica: a média antes e depois, e as respostas aprovadas antes e reprovadas depois."""
    before, after = per_answer(before_exports, framework), per_answer(after_exports, framework)
    metrics = {}
    for metric in sorted(set(before) | set(after)):
        b, a = before.get(metric, {}), after.get(metric, {})
        common = sorted(set(b) & set(a))
        fell = [n for n in common if b[n]["aprovada"] and not a[n]["aprovada"]]
        metrics[metric] = {
            "antes": round(statistics.fmean(v["media"] for v in b.values()), 4) if b else None,
            "depois": round(statistics.fmean(v["media"] for v in a.values()), 4) if a else None,
            "aprovadas_antes": sum(1 for n in common if b[n]["aprovada"]),
            "aprovadas_depois": sum(1 for n in common if a[n]["aprovada"]),
            "respostas_comparadas": len(common),
            "respostas_que_cairam": fell,
            "reagiu": bool(fell),
        }
    return metrics


def null_control(original: list[dict[str, Any]], null_exports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """framework -> {comparacoes, n0, reacoes}: o que cada juiz reprova sem defeito nenhum.

    `reacoes`: métrica -> em quantas comparações reprovou alguma resposta, e o máximo de respostas numa delas.
    """
    out: dict[str, dict[str, Any]] = {}
    for fw in FRAMEWORKS:
        mine = [e for e in original if framework_of(e) == fw]
        null = [e for e in null_exports if framework_of(e) == fw]
        pairs = [([e for e in mine if e is not held_out], [held_out]) for held_out in mine] if len(mine) >= 2 else []
        if null:
            pairs.append((mine, null))
        found: dict[str, dict[str, int]] = {}
        for before, after in pairs:
            for metric, m in reactions(before, after, fw).items():
                if m["reagiu"]:
                    row = found.setdefault(metric, {"comparacoes": 0, "maximo_de_respostas": 0})
                    row["comparacoes"] += 1
                    row["maximo_de_respostas"] = max(row["maximo_de_respostas"], len(m["respostas_que_cairam"]))
        out[fw] = {"comparacoes": len(pairs), "n0": bool(null), "reacoes": found}
    return out


def mutation_report(original: list[dict[str, Any]], original_answers: dict[str, Any] | None,
                    mutants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`mutants`: operador -> {"meta": mutante.json, "exports": [...], "answers": respostas.json do mutante}."""
    control = null_control(original, mutants.get(NULL_OPERATOR, {}).get("exports", []))
    report: dict[str, Any] = {"original": behavior(original_answers), "controle_nulo": control, "mutantes": {}}
    for op, mutant in mutants.items():
        judges = {}
        for fw in FRAMEWORKS:
            if not per_answer(mutant.get("exports", []), fw):
                continue
            metrics = reactions(original, mutant.get("exports", []), fw)
            reacted = [m for m, v in metrics.items() if v["reagiu"]]
            noise = control[fw]["reacoes"]
            confirmed = [m for m in reacted if len(metrics[m]["respostas_que_cairam"])
                         > noise.get(m, {}).get("maximo_de_respostas", 0)]
            if op == NULL_OPERATOR:
                state = "ruído" if reacted else "vivo"
                confirmed = []
            elif not reacted:
                state = "vivo"
            elif not control[fw]["comparacoes"]:
                state, confirmed = "sem controle", []
            else:
                state = "morto" if confirmed else "ruído"
            judges[fw] = {"estado": state, "morto": state == "morto", "reagiu_por": reacted,
                          "confirmado_por": confirmed, "metricas": metrics}
        report["mutantes"][op] = {
            "nome": (mutant.get("meta") or {}).get("nome"),
            "esperado": (mutant.get("meta") or {}).get("esperado"),
            "comportamento": behavior(mutant.get("answers"), original_answers),
            "juizes": judges,
        }
    return report


def judge_cell(judge: dict[str, Any] | None) -> str:
    """O que um juiz fez com um mutante, para as tabelas (terminal e relatório)."""
    if not judge:
        return "—"
    if judge["estado"] == "vivo":
        return "vivo"

    def num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}".replace(".", ",")

    why = judge["confirmado_por"] or judge["reagiu_por"]
    detail = ", ".join(f"{METRIC_SHORT.get(m, m)} {num(judge['metricas'][m]['antes'])}→"
                       f"{num(judge['metricas'][m]['depois'])}" for m in why)
    return f"**{judge['estado']}** ({detail})" if judge["estado"] == "morto" else f"{judge['estado']} ({detail})"


def effect_cell(b: dict[str, Any]) -> str:
    """O efeito do mutante na resposta, sem juiz."""
    parts = []
    if b.get("iguais_ao_original") is not None:
        parts.append(f"{b['iguais_ao_original']}/{b['respostas']} iguais ao original")
    parts.append(f"{b['com_todos_os_fatos']}/{b['respostas']} com os 4 fatos")
    if b["com_algum_fato_trocado"]:
        parts.append(f"{b['com_algum_fato_trocado']}/{b['respostas']} com fato trocado")
    if b["abstencoes"]:
        parts.append(f"{b['abstencoes']}/{b['respostas']} abstenções")
    return "; ".join(parts)


def control_lines(report: dict[str, Any]) -> list[str]:
    """Uma frase por juiz: o que ele reprova sem defeito nenhum."""
    lines = []
    for fw, c in report["controle_nulo"].items():
        if not c["comparacoes"]:
            lines.append(f"{fw}: sem controle nulo (um julgamento só e sem o N0)")
            continue
        noisy = ", ".join(f"{METRIC_SHORT.get(m, m)} em {row['comparacoes']} (até {row['maximo_de_respostas']} "
                          f"resposta(s))" for m, row in sorted(c["reacoes"].items()))
        lines.append(f"{fw}: {c['comparacoes']} comparações sem defeito{' (uma é o N0)' if c['n0'] else ''}; "
                     + (f"reprovou respostas em {noisy}" if noisy else "não reprovou nenhuma resposta"))
    return lines


def markdown(report: dict[str, Any]) -> str:
    lines = ["| Mutante | O certo seria | O efeito na resposta | O artigo nos trechos | RAGAS | DeepEval | JevEval |",
             "| :--- | :--- | :--- | :---: | :--- | :--- | :--- |"]
    original = report["original"]
    lines.append(f"| original | — | {effect_cell(original)} | "
                 f"{original['com_o_artigo_nos_trechos']}/{original['respostas']} | — | — | — |")
    for op, m in report["mutantes"].items():
        b = m["comportamento"]
        cells = [judge_cell(m["juizes"].get(fw)) for fw in FRAMEWORKS]
        lines.append(f"| {op}: {m['nome']} | {m['esperado']} | {effect_cell(b)} | "
                     f"{b['com_o_artigo_nos_trechos']}/{b['respostas']} | " + " | ".join(cells) + " |")
    return "\n".join([*lines, "", "Controle nulo: " + "; ".join(control_lines(report)) + "."])
