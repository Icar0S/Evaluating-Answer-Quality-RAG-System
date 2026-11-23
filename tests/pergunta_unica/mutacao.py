"""Qual juiz percebe cada mutante da pergunta única (plano-escopo-final.md, §8).

Um juiz MATA o mutante numa métrica quando reprova ao menos uma das respostas que
aprovava no original. A comparação é resposta a resposta (os nomes são os mesmos:
`md-mtbench-vieses#1` … `#5`, as mesmas sementes), com a nota média dos
julgamentos de cada lado contra o limiar da métrica. O mutante está morto para o
juiz se alguma métrica o matou.

Além das notas, o comportamento do RAG no mutante: quantas respostas se
abstiveram e em quantas o artigo do MT-Bench ainda chegou nos trechos.

Só biblioteca padrão (importada pelo rodar.py e pelo relatório).
"""
from __future__ import annotations

import statistics
from typing import Any

from analise import FRAMEWORKS, canonical_metric, framework_of

DOCUMENT_PREFIX = "NeurIPS-2023-judging-llm-as-a-judge"
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


def behavior(answers_meta: dict[str, Any] | None) -> dict[str, Any]:
    answers = (answers_meta or {}).get("answers", [])
    return {
        "respostas": len(answers),
        "abstencoes": sum(1 for a in answers if a.get("abstained")),
        "com_o_artigo_nos_trechos": sum(1 for a in answers
                                        if any(d.startswith(DOCUMENT_PREFIX) for d in a.get("source_documents") or [])),
        "caracteres_medios": round(statistics.fmean([len(a.get("actual_output") or "") for a in answers]), 1)
        if answers else None,
    }


def mutation_report(original: list[dict[str, Any]], original_answers: dict[str, Any] | None,
                    mutants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`mutants`: operador -> {"meta": mutante.json, "exports": [...], "answers": respostas.json do mutante}."""
    report: dict[str, Any] = {"original": behavior(original_answers), "mutantes": {}}
    for op, mutant in mutants.items():
        judges = {}
        for fw in FRAMEWORKS:
            before, after = per_answer(original, fw), per_answer(mutant.get("exports", []), fw)
            if not after:
                continue
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
                    "matou": bool(fell),
                }
            judges[fw] = {"morto": any(m["matou"] for m in metrics.values()),
                          "matou_por": [m for m, v in metrics.items() if v["matou"]], "metricas": metrics}
        report["mutantes"][op] = {
            "nome": (mutant.get("meta") or {}).get("nome"),
            "esperado": (mutant.get("meta") or {}).get("esperado"),
            "comportamento": behavior(mutant.get("answers")),
            "juizes": judges,
        }
    return report


def markdown(report: dict[str, Any]) -> str:
    def num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}".replace(".", ",")

    lines = ["| Mutante | O certo seria | Abstenções | O artigo nos trechos | RAGAS | DeepEval | JevEval |",
             "| :--- | :--- | :---: | :---: | :--- | :--- | :--- |"]
    original = report["original"]
    lines.append(f"| original | — | {original['abstencoes']}/{original['respostas']} | "
                 f"{original['com_o_artigo_nos_trechos']}/{original['respostas']} | — | — | — |")
    for op, m in report["mutantes"].items():
        cells = []
        for fw in FRAMEWORKS:
            j = m["juizes"].get(fw)
            if not j:
                cells.append("—")
            elif j["morto"]:
                why = ", ".join(f"{METRIC_SHORT.get(k, k)} {num(j['metricas'][k]['antes'])}→"
                                f"{num(j['metricas'][k]['depois'])}" for k in j["matou_por"])
                cells.append(f"**morto** ({why})")
            else:
                cells.append("vivo")
        b = m["comportamento"]
        lines.append(f"| {op}: {m['nome']} | {m['esperado']} | {b['abstencoes']}/{b['respostas']} | "
                     f"{b['com_o_artigo_nos_trechos']}/{b['respostas']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
