"""A análise dos julgamentos repetidos da pergunta única (plano-escopo-final.md, §5).

Lê os exports dos três juízes (RAGAS, DeepEval, JevEval), K julgamentos de cada,
sobre as mesmas N respostas, e separa as duas fontes de variação de cada métrica:

- a do GERADOR: entre as N respostas, a diferença entre as médias de cada uma;
- a do JUIZ: entre os K julgamentos da mesma resposta, quanto a nota muda.

Com 5 respostas, a leitura é descritiva: médias, faixas (mínimo e máximo) e
desvios, sem p-valor. As contas são as de sempre, ditas por extenso:

- `media`: a média de todas as notas medidas da métrica (N x K);
- `amplitude_juiz`: para cada resposta, máximo menos mínimo dos K julgamentos; a
  média e o máximo dessas amplitudes;
- `dp_juiz`: a raiz da média das variâncias de cada resposta entre julgamentos
  (o desvio "dentro");
- `dp_gerador`: o desvio-padrão das médias das N respostas (o desvio "entre");
- `vereditos_instaveis`: quantas respostas tiveram julgamentos dos dois lados do
  limiar (passou num, reprovou noutro).

N/A (o Jev, quando nenhuma pergunta se aplica) e nota não medida (erro, timeout)
ficam fora das contas e são contados à parte.

Só biblioteca padrão: roda com qualquer Python 3.10+ e é importada pelo
relatório (tests/reports).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

FRAMEWORKS = ("ragas", "deepeval", "jeveval")
# O DeepEval grava o nome de exibição da métrica; o RAGAS e o Jev, a chave. A mesma tabela de
# tests/reports/report_data.py (METRIC_TO_DIMENSION), copiada aqui porque este módulo é só biblioteca padrão.
DEEPEVAL_NAMES = {
    "Faithfulness": "faithfulness",
    "Answer Relevancy": "answer_relevancy",
    "Contextual Precision": "context_precision",
    "Contextual Recall": "context_recall",
    "Contextual Relevancy": "context_relevance",
}


def canonical_metric(name: str) -> str:
    """O nome da métrica nos três frameworks: `faithfulness`, `answer_relevancy`, `context_*`, `correctness`."""
    return DEEPEVAL_NAMES.get(name, name)


def load_export(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def framework_of(export: dict[str, Any]) -> str:
    """O DeepEval não grava `framework` no export; os outros dois gravam."""
    return export.get("framework") or "deepeval"


def is_gap(metric: dict[str, Any]) -> bool:
    """Nota que devia existir e não existe (erro, timeout). N/A do Jev não é buraco: é um resultado."""
    return metric.get("score") is None and metric.get("applicable", True)


def unmeasured(export: dict[str, Any]) -> list[tuple[str, str]]:
    """Os buracos de um export: (resposta, métrica), na ordem do arquivo."""
    return [(case["name"], canonical_metric(m["name"])) for case in export.get("cases", [])
            for m in case.get("metrics", []) if is_gap(m)]


def merge_retry(export: dict[str, Any], retry: dict[str, Any], attempt: int, source: str) -> list[tuple[str, str]]:
    """Preenche os buracos de `export` com a nova tentativa; devolve o que preencheu. Altera `export`.

    Só os buracos: uma nota já medida nunca é trocada, então a tentativa não escolhe entre valores. O registro
    preenchido guarda a tentativa e o erro anterior, e o export lista as tentativas (de qual arquivo veio o quê).
    """
    again = {case["name"]: {m["name"]: m for m in case.get("metrics", [])} for case in retry.get("cases", [])}
    filled: list[tuple[str, str]] = []
    for case in export.get("cases", []):
        for i, m in enumerate(case.get("metrics", [])):
            new = again.get(case["name"], {}).get(m["name"])
            if not is_gap(m) or new is None or is_gap(new):
                continue
            case["metrics"][i] = {**new, "tentativa": attempt, "erro_anterior": m.get("error")}
            filled.append((case["name"], canonical_metric(m["name"])))
        applicable = [m for m in case.get("metrics", []) if m.get("applicable", True)]
        case["success"] = all(bool(m.get("success")) for m in applicable)
    export.setdefault("tentativas", []).append(
        {"tentativa": attempt, "arquivo": source, "preenchidas": [list(f) for f in filled]})
    # Os resumos que o framework gravou são refeitos: só as chaves que o export já tinha.
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for case in export.get("cases", []):
        for m in case.get("metrics", []):
            if m.get("applicable", True):
                by_metric.setdefault(m["name"], []).append(m)
    scores = {name: [m["score"] for m in ms if m.get("score") is not None] for name, ms in by_metric.items()}
    fresh = {
        "metrics_avg": {n: round(statistics.fmean(s), 4) if s else None for n, s in scores.items()},
        "metrics_measured": {n: f"{len(scores[n])}/{len(ms)}" for n, ms in by_metric.items()},
        "metrics_passed": {n: f"{sum(1 for m in ms if m.get('success'))}/{len(ms)}" for n, ms in by_metric.items()},
    }
    for key, values in fresh.items():
        if isinstance(export.get(key), dict):
            export[key].update(values)
    return filled


def collect(exports: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """framework -> métrica -> resposta -> [julgamentos], cada um {repeticao, score, success, applicable, error}."""
    table: dict = {}
    for export in exports:
        fw = framework_of(export)
        rep = export.get("judge_repetition")
        for case in export.get("cases", []):
            for m in case.get("metrics", []):
                metric = canonical_metric(m["name"])
                table.setdefault(fw, {}).setdefault(metric, {}).setdefault(case["name"], []).append({
                    "repeticao": rep,
                    "score": m.get("score"),
                    "success": m.get("success"),
                    "applicable": m.get("applicable", True),
                    "error": m.get("error"),
                    "threshold": m.get("threshold", export.get("threshold")),
                })
    return table


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def summarize_metric(by_answer: dict[str, list[dict[str, Any]]], threshold: float | None) -> dict[str, Any]:
    scores_by_answer: dict[str, list[float]] = {}
    not_applicable = not_measured = 0
    unstable = []
    for answer, judgments in sorted(by_answer.items()):
        scores = []
        for j in judgments:
            if not j["applicable"]:
                not_applicable += 1
            elif j["score"] is None:
                not_measured += 1
            else:
                scores.append(float(j["score"]))
        if scores:
            scores_by_answer[answer] = scores
        limit = threshold if threshold is not None else next((j["threshold"] for j in judgments if j["threshold"]), None)
        if limit is not None and len(scores) >= 2 and min(scores) < limit <= max(scores):
            unstable.append(answer)
    all_scores = [s for scores in scores_by_answer.values() for s in scores]
    means = {a: statistics.fmean(s) for a, s in scores_by_answer.items()}
    ranges = {a: max(s) - min(s) for a, s in scores_by_answer.items() if len(s) >= 2}
    variances = [statistics.pvariance(s) for s in scores_by_answer.values() if len(s) >= 2]
    return {
        "media": _round(statistics.fmean(all_scores)) if all_scores else None,
        "minimo": _round(min(all_scores)) if all_scores else None,
        "maximo": _round(max(all_scores)) if all_scores else None,
        "notas": len(all_scores),
        "nao_se_aplica": not_applicable,
        "nao_medidas": not_measured,
        "media_por_resposta": {a: _round(m) for a, m in means.items()},
        "faixa_por_resposta": {a: [_round(min(s)), _round(max(s))] for a, s in scores_by_answer.items()},
        "amplitude_juiz_media": _round(statistics.fmean(ranges.values())) if ranges else None,
        "amplitude_juiz_maxima": _round(max(ranges.values())) if ranges else None,
        "dp_juiz": _round(math.sqrt(statistics.fmean(variances))) if variances else None,
        "dp_gerador": _round(statistics.pstdev(means.values())) if len(means) >= 2 else None,
        "vereditos_instaveis": unstable,
        "limiar": threshold,
    }


def variance_report(exports: list[dict[str, Any]]) -> dict[str, Any]:
    """O resumo por framework e métrica, mais quem domina a variação (gerador ou juiz) em cada uma."""
    table = collect(exports)
    thresholds: dict[str, float | None] = {}
    for export in exports:
        fw = framework_of(export)
        thresholds.setdefault(fw, export.get("threshold"))
    result: dict[str, Any] = {"frameworks": {}, "exports": [e.get("_path") for e in exports]}
    for fw in FRAMEWORKS:
        if fw not in table:
            continue
        reps = sorted({e.get("judge_repetition") for e in exports if framework_of(e) == fw},
                      key=lambda r: (r is None, r))
        metrics = {}
        for metric, by_answer in table[fw].items():
            # O Jev tem limiar próprio por métrica (a correção usa 0,5); os outros, um só.
            per_metric = next((j["threshold"] for js in by_answer.values() for j in js if j["threshold"] is not None),
                              thresholds.get(fw))
            summary = summarize_metric(by_answer, per_metric)
            if summary["dp_juiz"] is not None and summary["dp_gerador"] is not None:
                summary["domina"] = "juiz" if summary["dp_juiz"] >= summary["dp_gerador"] else "gerador"
            else:
                summary["domina"] = None
            metrics[metric] = summary
        result["frameworks"][fw] = {"julgamentos": reps, "metricas": metrics}
    return result


def find_exports(run_dir: Path) -> list[Path]:
    """Os exports de uma rodada da pergunta única: <run>/<framework>/*.json (sem os .html e os do relatório)."""
    paths = []
    for fw in FRAMEWORKS:
        paths += sorted((Path(run_dir) / fw).glob("*.json"))
    return paths


def markdown(report: dict[str, Any]) -> str:
    """A tabela da decomposição, para o terminal e o relatório."""
    lines = [
        "| Juiz | Métrica | Média | Faixa | DP entre respostas (gerador) | DP entre julgamentos (juiz) | "
        "Vereditos instáveis | Domina |",
        "| :--- | :--- | ---: | :---: | ---: | ---: | ---: | :--- |",
    ]

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}".replace(".", ",")

    for fw, data in report["frameworks"].items():
        for metric, s in data["metricas"].items():
            faixa = "—" if s["minimo"] is None else f"{fmt(s['minimo'])}–{fmt(s['maximo'])}"
            extra = []
            if s["nao_se_aplica"]:
                extra.append(f"{s['nao_se_aplica']} N/A")
            if s["nao_medidas"]:
                extra.append(f"{s['nao_medidas']} não medidas")
            media = fmt(s["media"]) + (f" ({', '.join(extra)})" if extra else "")
            lines.append(
                f"| {fw} | {metric} | {media} | {faixa} | {fmt(s['dp_gerador'])} | {fmt(s['dp_juiz'])} | "
                f"{len(s['vereditos_instaveis'])} | {s['domina'] or '—'} |"
            )
    return "\n".join(lines)
