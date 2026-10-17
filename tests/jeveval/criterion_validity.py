"""Validade de critério da correção do JevEval sobre os 247 pares rotulados à mão do estudo de mutação.

Pergunta: a nota de correção (_jev_shared.CORRECTNESS) separa paráfrase que
preserva os fatos (EQUIV) de resposta com erro injetado (ERRO)? É o que o O1
(cosseno de embeddings) não faz: AUC 0,48 sobre estes mesmos pares
(tests/mutation/calibration/tau.json) — paráfrase e erro dão o mesmo cosseno
médio (0,970 x 0,967).

Os pares são os da calibração do O1, com o mesmo filtro (sai quem foi excluído
na conferência visual): 247 pares, 111 EQUIV e 136 ERRO. A pergunta de cada par
vem da suíte (tests/mutation/suite/golden.jsonl) pelo `case_id`.

O limiar de decisão é o 0,5 fixado ANTES de olhar os pares
(JEVEVAL_CORRECTNESS_THRESHOLD). A varredura de limiares e o melhor F1 saem no
relatório para comparação com o τ* do O1, mas não escolhem nada: escolher o
limiar nos mesmos pares em que se mede o desempenho infla o resultado.

Mesma limitação declarada no tau.json: os erros são gerados por regra (troca de
número, de entidade, inversão de condição) e podem ser mais fáceis de detectar
que os erros naturais de um LLM. O desempenho daqui é um limite superior.

Saída: tests/jeveval/results/criterion_<timestamp>.json, regravado a cada 10
pares (uma interrupção não perde as chamadas já pagas).

Rodar da raiz do repo (~247 chamadas ao Jev):
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\criterion_validity.py
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\criterion_validity.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _jev_mutation as mutation  # noqa: E402
import _jev_shared as shared  # noqa: E402
from _jev_stats import auc_roc, bootstrap_ci, confusion  # noqa: E402

SWEEP = [round(0.05 * k, 2) for k in range(1, 20)]  # 0,05 .. 0,95


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(records: list[dict[str, Any]], threshold: float, tau_json: dict[str, Any] | None) -> dict[str, Any]:
    """Os números do relatório, só a partir dos registros (dá para refazer sem chamar o Jev)."""
    measured = [r for r in records if r["score"] is not None]
    scores = [r["score"] for r in measured]
    labels = [r["positive"] for r in measured]

    auc = auc_roc(scores, labels)
    auc_low, auc_high = bootstrap_ci(
        len(measured), lambda idx: auc_roc([scores[i] for i in idx], [labels[i] for i in idx])
    )
    sweep = [confusion(scores, labels, tau) for tau in SWEEP]
    best = max(sweep, key=lambda row: row["f1"]) if measured else None

    # Por regra de construção do par: quanto de cada tipo de erro a correção
    # pega (nota < limiar), e quanto das paráfrases ela aceita (nota >= limiar).
    by_rule: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in measured:
        grouped[r["rule"]].append(r)
    for rule, rows in sorted(grouped.items()):
        accepted = sum(1 for r in rows if r["score"] >= threshold)
        positive = rows[0]["positive"] == 1
        by_rule[rule] = {
            "label": "EQUIV" if positive else "ERRO",
            "pairs": len(rows),
            "accepted": accepted,
            # Para paráfrase, acerto = aceitar; para erro, acerto = rejeitar.
            "hit_rate": round((accepted if positive else len(rows) - accepted) / len(rows), 4),
            "mean_score": _mean([r["score"] for r in rows]),
            "causes": dict(Counter(r.get("cause") or "none" for r in rows)),
        }

    summary: dict[str, Any] = {
        "pairs": len(records),
        "measured": len(measured),
        "not_applicable": sum(1 for r in records if not r["applicable"]),
        "errors": sum(1 for r in records if r["error"]),
        "equiv": sum(1 for r in measured if r["positive"] == 1),
        "erro": sum(1 for r in measured if r["positive"] == 0),
        "threshold": threshold,
        "auc": None if auc != auc else round(auc, 4),
        "auc_ci95": [None if v != v else round(v, 4) for v in (auc_low, auc_high)],
        "at_threshold": confusion(scores, labels, threshold) if measured else None,
        "best_f1_in_sweep": best,
        "equiv_mean": _mean([s for s, y in zip(scores, labels) if y == 1]),
        "erro_mean": _mean([s for s, y in zip(scores, labels) if y == 0]),
        "mean_confidence": _mean([r["confidence"] for r in measured if r["confidence"] is not None]),
        "by_rule": by_rule,
        "sweep": sweep,
    }
    if tau_json:
        summary["o1_reference"] = {
            "source": "tests/mutation/calibration/tau.json",
            "pairs": tau_json.get("pairs"),
            "auc": tau_json.get("auc"),
            "tau_star": tau_json.get("tau_star"),
            "f1": tau_json.get("f1"),
            "accuracy": tau_json.get("accuracy"),
            "equiv_mean": tau_json.get("equiv_mean"),
            "erro_mean": tau_json.get("erro_mean"),
        }
    return summary


def main(argv: list[str] | None = None, model=None) -> int:
    """`model` existe para os testes herméticos injetarem o Jev falso; sem ele, usa o Jev de verdade."""
    parser = argparse.ArgumentParser(description="Validade de critério da correção do JevEval (pares rotulados).")
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros pares (fumaça)")
    parser.add_argument("--pairs", type=Path, default=mutation.PAIRS, help="pares rotulados (padrão: os do estudo)")
    parser.add_argument("--golden", type=Path, default=mutation.GOLDEN, help="suíte com as perguntas de cada caso")
    parser.add_argument("--out-dir", type=Path, default=shared.RESULTS_DIR, help="pasta de saída")
    args = parser.parse_args(argv)

    settings = shared.get_settings()
    if model is None:
        reason = shared.jev_status(settings)
        if reason:
            _log(f"[ERRO] Jev indisponível: {reason}")
            return 1
        model = shared.build_jev(settings)

    pairs = mutation.load_pairs(args.pairs, args.golden)[: args.limit]
    if not pairs:
        _log("[ERRO] Nenhum par rotulado encontrado.")
        return 1
    tau_json = json.loads(mutation.TAU_JSON.read_text(encoding="utf-8")) if mutation.TAU_JSON.is_file() else None
    threshold = settings.jeveval_correctness_threshold

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"criterion_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output: dict[str, Any] = {
        "framework": "jeveval",
        "analysis": "validade de critério da correção (pares rotulados do estudo de mutação)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": model.get_model_name(),
        "instrument_sha": shared.instrument_sha([shared.CORRECTNESS.key]),
        "question_set": shared.CORRECTNESS.question_set(),
        "pairs_source": str(args.pairs),
        "limitation": (
            "Erros dos pares gerados por regra (troca de número, de entidade, inversão de condição): podem ser "
            "mais fáceis de detectar que erros naturais de um LLM. O desempenho é um limite superior — a mesma "
            "ressalva do tau.json do O1."
        ),
    }

    _log(f"Validade de critério | juiz: {output['judge_model']} | {len(pairs)} pares | limiar {threshold}")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, pair in enumerate(pairs, start=1):
        outcome = shared.measure(
            shared.CORRECTNESS,
            model,
            shared.CaseInputs(input=pair.question, actual_output=pair.candidate, expected_output=pair.reference),
            threshold,
        )
        records.append(
            {
                "pair_id": pair.pair_id,
                "case_id": pair.case_id,
                "label": pair.label,
                "positive": pair.positive,
                "rule": pair.rule,
                "score": None if outcome.score is None else round(outcome.score, 4),
                "applicable": outcome.applicable,
                "error": outcome.error,
                "confidence": None if outcome.confidence is None else round(outcome.confidence, 4),
                "cause": outcome.cause,
                "seconds": outcome.seconds,
            }
        )
        shown = f"{outcome.score:.3f}" if outcome.measured else f"NÃO MEDIDA ({(outcome.error or 'N/A')[:100]})"
        _log(f"  [{i}/{len(pairs)}] {pair.pair_id:12s} {pair.label:5s} {shown}")
        if i % 10 == 0 or i == len(pairs):
            output.update(
                records=records,
                summary=summarize(records, threshold, tau_json),
                wall_seconds=round(time.perf_counter() - started, 1),
            )
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    s = output["summary"]
    _log(f"\nPares medidos: {s['measured']}/{s['pairs']} ({s['equiv']} EQUIV, {s['erro']} ERRO); erros: {s['errors']}")
    _log(f"AUC JevEval: {s['auc']}  IC95 {s['auc_ci95']}")
    if s.get("o1_reference"):
        _log(f"AUC O1 (cosseno, tau.json): {s['o1_reference']['auc']}")
    if s["at_threshold"]:
        t = s["at_threshold"]
        _log(
            f"No limiar {threshold}: acurácia {t['accuracy']}, precisão {t['precision']}, revocação {t['recall']}, "
            f"F1 {t['f1']} (tp={t['tp']} fp={t['fp']} fn={t['fn']} tn={t['tn']})"
        )
    _log("Por regra (acerto = aceitar paráfrase / rejeitar erro):")
    for rule, row in s["by_rule"].items():
        _log(f"  {rule:18s} {row['label']:5s} {row['pairs']:4d} pares  acerto {row['hit_rate']:.2f}  nota média {row['mean_score']}")
    _log(f"JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
