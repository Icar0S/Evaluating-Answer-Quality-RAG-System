"""O6: a correção do JevEval aplicada ao log da campanha de mutação — análise EXPLORATÓRIA.

Executar != julgar: o log (tests/mutation/results/runs.jsonl) guarda pergunta,
resposta e trechos de cada invocação, então um oráculo novo é julgado sobre ele
sem reexecutar o assistente — o mesmo caminho que o study.yaml deixa aberto para
o O3. As regras (resposta modal, conjunto S, morte) são as do evaluate.py; ver
_jev_mutation.py.

Exploratória porque o O6 nasceu depois da campanha. Os vereditos ficam em
tests/jeveval/results/o6/ e NÃO entram em tests/mutation/results/verdicts.jsonl.
Levar um número daqui para o artigo exige declará-lo como análise post hoc
(lista de desvios do protocolo, docs/paper/main.tex §6.7).

Saídas em tests/jeveval/results/o6/:

- judgments_<nível>.jsonl: um julgamento por (mutante, caso), acrescentado a cada
  chamada — a rodada é retomável e nenhuma chamada paga se perde;
- verdicts_<nível>.jsonl: os mesmos registros no formato do verdicts.jsonl da
  campanha (oracle "O6"), com `evaluable` e `kill` calculados;
- summary_<nível>.json: |S|, escore de mutação (IC de Wilson), taxa de morte por
  operador e a comparação com o O4 da campanha no mesmo nível: os dois escores
  lado a lado e a concordância caso a caso (kappa de Cohen).

Rodar da raiz do repo (570 chamadas ao Jev no L2: 18 mutantes e o baseline, x 30 casos):
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\o6_mutation.py
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\o6_mutation.py --level L1
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\o6_mutation.py --mutants baseline,R1 --limit 10
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\o6_mutation.py --summary-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _jev_mutation as mutation  # noqa: E402
import _jev_shared as shared  # noqa: E402

MAX_DISAGREEMENTS = 30


def _log(msg: str) -> None:
    print(msg, flush=True)


def judge_pair(pair: mutation.ModalPair, case: dict[str, Any], model, threshold: float, level: str) -> dict[str, Any]:
    outcome = shared.measure(
        shared.CORRECTNESS,
        model,
        shared.CaseInputs(input=case["question"], actual_output=pair.answer, expected_output=case["expected_output"]),
        threshold,
    )
    # Sem medida, o veredito é "pass" com evaluable_oracle=False — a convenção do
    # O4 (tests/mutation/oracles/o4_judge.py): reprovar por não ter conseguido
    # medir inflaria a taxa de mortes com falha de infraestrutura.
    verdict = "fail" if outcome.measured and outcome.score < threshold else "pass"
    if outcome.measured:
        confidence = "-" if outcome.confidence is None else f"{outcome.confidence:.2f}"
        detail = f"P(correta)={outcome.score:.2f}; causa={outcome.cause}; confiança={confidence}"
    else:
        detail = f"não medido: {outcome.error}"
    return {
        "level": level,
        "mutant_id": pair.mutant_id,
        "operator": pair.operator,
        "case_id": pair.case_id,
        "oracle": mutation.ORACLE,
        "verdict": verdict,
        "score": None if outcome.score is None else round(outcome.score, 4),
        "evaluable_oracle": outcome.measured,
        "modal": True,
        "stable": True,
        "runs": pair.runs,
        "detail": detail[:500],
        "components": {
            "cause": outcome.cause,
            "confidence": outcome.confidence,
            "probabilities": (outcome.breakdown[0].get("probabilities") if outcome.breakdown else None),
            "error": outcome.error,
        },
        "cost": {"wall_ms": round(outcome.seconds * 1000, 1), "cost_usd": outcome.cost_usd, "input_tokens": outcome.input_tokens},
        "instrument_sha": shared.instrument_sha([shared.CORRECTNESS.key]),
        "judge_model": model.get_model_name(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def build_summary(
    judgments: list[dict[str, Any]],
    o4_records: list[dict[str, Any]],
    level: str,
    threshold: float,
    requires_pass: bool,
) -> tuple[dict[tuple[str, str], dict], dict[str, Any]]:
    """Vereditos com S e mortes calculados, e o resumo comparado ao O4. Só lê registros: não chama o Jev."""
    o6 = {(r["mutant_id"], r["case_id"]): dict(r) for r in judgments}  # o último julgamento de cada par vale
    s_o6 = mutation.evaluable_set(o6, requires_pass)
    mutation.mark_kills(o6, s_o6)

    o4 = mutation.last_records(o4_records, "O4", level)
    s_o4 = {case for (mutant, case), v in o4.items() if mutant == mutation.BASELINE_ID and v.get("evaluable")}
    both = s_o6 & s_o4
    agreement = mutation.agreement(o6, o4, both)
    disagreements = [
        {
            "mutant_id": mutant,
            "case_id": case,
            "o6": {"verdict": o6[(mutant, case)]["verdict"], "score": o6[(mutant, case)]["score"],
                   "cause": o6[(mutant, case)]["components"].get("cause")},
            "o4": {"verdict": o4[(mutant, case)]["verdict"], "score": o4[(mutant, case)].get("score"),
                   "detail": (o4[(mutant, case)].get("detail") or "")[:240]},
        }
        for mutant, case in agreement.pop("disagreements")[:MAX_DISAGREEMENTS]
    ]

    baseline = [v for (mutant, _), v in o6.items() if mutant == mutation.BASELINE_ID]
    summary = {
        "framework": "jeveval",
        "analysis": (
            "EXPLORATÓRIA: oráculo O6 (correção do JevEval) aplicado ao log da campanha depois de ela fechar. "
            "Fora do protocolo; não entra em tests/mutation/results/verdicts.jsonl."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "threshold": threshold,
        "instrument_sha": shared.instrument_sha([shared.CORRECTNESS.key]),
        "judge_models": dict(Counter(v.get("judge_model") for v in o6.values())),
        "pairs_judged": len(o6),
        "not_measured": sum(1 for v in o6.values() if not v.get("evaluable_oracle", True)),
        "baseline": {
            "cases": len(baseline),
            "pass": sum(1 for v in baseline if v["verdict"] == "pass" and v.get("evaluable_oracle", True)),
            "fail": sum(1 for v in baseline if v["verdict"] == "fail"),
            "failing_cases": sorted(v["case_id"] for v in baseline if v["verdict"] == "fail"),
        },
        "causes_of_kills": dict(Counter(v["components"].get("cause") for v in o6.values() if v["kill"])),
        "o6": mutation.mutation_score(o6, s_o6),
        "o4": {
            "source": "tests/mutation/results/verdicts.jsonl (O4 da campanha, mesmo nível)",
            **mutation.mutation_score(o4, s_o4),
        } if o4 else None,
        "agreement_o6_o4": {**agreement, "cases_in_both_s": len(both), "disagreements_sample": disagreements},
    }
    return o6, summary


def main(argv: list[str] | None = None, model=None) -> int:
    """`model` existe para os testes herméticos injetarem o Jev falso; sem ele, usa o Jev de verdade."""
    parser = argparse.ArgumentParser(description="O6 (correção do JevEval) sobre o log da campanha de mutação.")
    parser.add_argument("--level", default="L2", choices=["L1", "L2", "L3"], help="nível de rigor (padrão do estudo: L2)")
    parser.add_argument("--mutants", help="subconjunto de mutant_id, separado por vírgula")
    parser.add_argument("--limit", type=int, default=None, help="julga no máximo N pares nesta chamada (fumaça)")
    parser.add_argument("--summary-only", action="store_true", help="refaz vereditos e resumo sem chamar o Jev")
    parser.add_argument("--runs", type=Path, default=mutation.RUNS_JSONL, help="log da campanha")
    parser.add_argument("--verdicts", type=Path, default=mutation.VERDICTS_JSONL, help="vereditos da campanha (O4)")
    parser.add_argument("--study", type=Path, default=mutation.STUDY_CONFIG, help="config/study.yaml do estudo")
    parser.add_argument("--golden", type=Path, default=mutation.GOLDEN, help="suíte do estudo")
    parser.add_argument("--out-dir", type=Path, default=shared.RESULTS_DIR / "o6", help="pasta de saída")
    args = parser.parse_args(argv)

    study = mutation.load_study(args.study)
    max_repetitions = int(study["execution"]["levels"][args.level])
    cap = study["baseline_config"].get("generation_num_predict")
    requires_pass = bool(study["execution"].get("evaluable_requires_baseline_pass", True))
    settings = shared.get_settings()
    threshold = settings.jeveval_correctness_threshold
    sha = shared.instrument_sha([shared.CORRECTNESS.key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = args.out_dir / f"judgments_{args.level}.jsonl"
    previous = mutation.read_jsonl(judgments_path)
    other_instrument = [r for r in previous if r.get("instrument_sha") != sha]
    if other_instrument:
        _log(
            f"[AVISO] {len(other_instrument)} julgamento(s) em {judgments_path.name} são de outro instrumento "
            f"(instrument_sha diferente de {sha}) e ficam fora — o resumo não mistura instrumentos."
        )
    previous = [r for r in previous if r.get("instrument_sha") == sha]

    if not args.summary_only:
        if model is None:
            reason = shared.jev_status(settings)
            if reason:
                _log(f"[ERRO] Jev indisponível: {reason}")
                return 1
            model = shared.build_jev(settings)
        runs = mutation.read_jsonl(args.runs)
        if not runs:
            _log(
                f"[ERRO] Log da campanha não encontrado em {args.runs}. Ele não é versionado: vem do pacote de "
                "replicação (tests/mutation/results/replication-*.zip) ou de uma campanha local."
            )
            return 1
        golden = mutation.load_golden(args.golden)
        pairs = mutation.modal_pairs(runs, max_repetitions, cap)
        if args.mutants:
            wanted = {m.strip() for m in args.mutants.split(",")}
            pairs = [p for p in pairs if p.mutant_id in wanted]
        done = {(r["mutant_id"], r["case_id"]) for r in previous if r.get("judge_model") == model.get_model_name()}
        todo = [p for p in pairs if (p.mutant_id, p.case_id) not in done and p.case_id in golden]
        if args.limit:
            todo = todo[: args.limit]
        _log(
            f"O6 (exploratório) | nível {args.level} (até {max_repetitions} execuções) | juiz {model.get_model_name()} | "
            f"{len(pairs)} pares no log, {len(done)} já julgados, {len(todo)} agora"
        )
        with judgments_path.open("a", encoding="utf-8") as f:
            for i, pair in enumerate(todo, start=1):
                record = judge_pair(pair, golden[pair.case_id], model, threshold, args.level)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                previous.append(record)
                shown = record["score"] if record["evaluable_oracle"] else "NÃO MEDIDO"
                _log(f"  [{i}/{len(todo)}] {pair.mutant_id}/{pair.case_id}: {record['verdict']} ({shown}) {record['components']['cause'] or ''}")

    if not previous:
        _log(f"[ERRO] Nenhum julgamento do O6 em {judgments_path} para resumir.")
        return 1
    o6, summary = build_summary(previous, mutation.read_jsonl(args.verdicts), args.level, threshold, requires_pass)

    verdicts_path = args.out_dir / f"verdicts_{args.level}.jsonl"
    with verdicts_path.open("w", encoding="utf-8") as f:
        for key in sorted(o6):
            f.write(json.dumps(o6[key], ensure_ascii=False) + "\n")
    summary_path = args.out_dir / f"summary_{args.level}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    o6s, o4s, agree = summary["o6"], summary["o4"], summary["agreement_o6_o4"]
    _log(f"\n|S| O6 = {o6s['s_size']} casos; baseline: {summary['baseline']['pass']} passam, {summary['baseline']['fail']} reprovam")
    _log(f"Escore de mutação O6: {o6s['killed']}/{o6s['mutants']} = {o6s['ms']}")
    if o4s:
        _log(f"Escore de mutação O4 (campanha): {o4s['killed']}/{o4s['mutants']} = {o4s['ms']} (|S| = {o4s['s_size']})")
        _log(
            f"Concordância O6 x O4 em {agree['pairs']} pares de S(O6) ∩ S(O4): {agree['agree']} iguais, "
            f"kappa {agree['kappa']} ({agree['kappa_label']})"
        )
    _log(f"Vereditos: {verdicts_path}\nResumo:    {summary_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
