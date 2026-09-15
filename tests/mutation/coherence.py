"""Classificação de coerência das mortes — procedimento de validade (§7.4).

Cada morte é coerente quando a assertiva violada, ou a justificativa do oráculo,
corresponde ao defeito injetado; incoerente quando o caso reprovou por outra
razão. Sem essa distinção, MS mede "o mutante reprovou alguma coisa", não "o
mutante foi detectado pelo defeito que ele injeta".

Adaptação para pesquisador único, como o protocolo determina:

1. codebook fechado A PRIORI (config/codebook.yaml), antes de olhar resultado;
2. classificação completa;
3. reclassificação cega de 30% da amostra após >= 7 dias, com os códigos
   anteriores ocultos -> kappa intra-avaliador;
4. limitação declarada: não há concordância inter-avaliadores.

A unidade de classificação é a morte (mutante, caso), não (mutante, caso,
oráculo): a causa é uma propriedade da resposta, não do instrumento que a
reprovou. Um mesmo código vale para todos os oráculos que mataram aquele par — e
é isso que permite comparar MS_coerente entre oráculos sobre a mesma base.

Uso:
    python -m tests.mutation.coherence --freeze-codebook   # semana 6, antes de ver resultado
    python -m tests.mutation.coherence --classify
    python -m tests.mutation.coherence --blind             # >= 7 dias depois
    python -m tests.mutation.coherence --kappa
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.mutation import stats as study_stats
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.coherence")

# Mapeia a causa que o juiz (O4) enxerga para uma sugestão de código do codebook.
# É sugestão, nunca atribuição automática: o juiz não vê o que foi recuperado e
# portanto não consegue distinguir RET-MISS de GEN-HALLUC com segurança.
JUDGE_CAUSE_HINT = {
    "missing_info": "RET-MISS",
    "wrong_value": "RET-STALE",
    "unsupported_claim": "GEN-HALLUC",
    "no_abstention": "GEN-NOABSTAIN",
    "no_citation": "FMT-BREAK",
    "off_topic": "FLOW-DESYNC",
}


def load_codebook() -> dict[str, Any]:
    raw = yaml.safe_load(paths.CODEBOOK.read_text(encoding="utf-8"))
    raw["by_code"] = {item["code"]: item for item in raw["codes"]}
    return raw


def freeze_codebook() -> int:
    codebook_text = paths.CODEBOOK.read_text(encoding="utf-8")
    raw = yaml.safe_load(codebook_text)
    if raw.get("frozen_at"):
        logger.info("Codebook já estava congelado em %s.", raw["frozen_at"])
        return 0
    if paths.VERDICTS_JSONL.exists() and jsonl.read(paths.VERDICTS_JSONL):
        logger.warning(
            "Já existem vereditos calculados. Congelar o codebook AGORA não cumpre o "
            "'a priori' do §7.4 — registre isso como ameaça à validade se seguir adiante."
        )
    today = date.today().isoformat()
    codebook_text = codebook_text.replace("frozen_at: null", f"frozen_at: {today}", 1)
    paths.CODEBOOK.write_text(codebook_text, encoding="utf-8")
    logger.info("Codebook congelado em %s.", today)
    return 0


def deaths_to_classify() -> list[dict[str, Any]]:
    """Mortes distintas (mutante, caso), com os oráculos que mataram cada uma."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in jsonl.read(paths.VERDICTS_JSONL):
        if not record.get("kill"):
            continue
        key = (record["mutant_id"], record["case_id"])
        entry = grouped.setdefault(
            key,
            {
                "mutant_id": record["mutant_id"],
                "operator": record.get("operator") or record["mutant_id"],
                "case_id": record["case_id"],
                "oracles": [],
                "details": [],
                "judge_cause": None,
            },
        )
        entry["oracles"].append(record["oracle"])
        if record.get("detail"):
            entry["details"].append(f"{record['oracle']}: {record['detail']}")
        if record["oracle"] == "O4":
            entry["judge_cause"] = (record.get("components") or {}).get("cause")
    return sorted(grouped.values(), key=lambda item: (item["mutant_id"], item["case_id"]))


def _modal_answer_for(mutant_id: str, case_id: str) -> str:
    from tests.mutation.oracles.base import modal_answer

    answers = [
        run["answer"]
        for run in jsonl.read(paths.RUNS_JSONL)
        if run["mutant_id"] == mutant_id and run["case_id"] == case_id and not run.get("error")
    ]
    return modal_answer(answers)


def _prompt_for_code(codebook: dict[str, Any], suggestion: str | None) -> str | None:
    codes = [item["code"] for item in codebook["codes"]]
    print("\nCódigos:")
    for index, item in enumerate(codebook["codes"], start=1):
        marker = " (sugestão do juiz)" if item["code"] == suggestion else ""
        print(f"  {index}. {item['code']:<15} {item['label']}{marker}")
    while True:
        raw = input("Código (número, sigla, 's' para pular, 'q' para sair): ").strip()
        if raw.lower() == "q":
            return None
        if raw.lower() == "s":
            return "SKIP"
        if raw.isdigit() and 1 <= int(raw) <= len(codes):
            return codes[int(raw) - 1]
        if raw.upper() in codes:
            return raw.upper()
        print("Entrada inválida.")


def classify(blind: bool = False, fraction: float = 0.30, seed: int = 20261124) -> int:
    codebook = load_codebook()
    if not codebook.get("frozen_at"):
        logger.error(
            "Codebook não congelado. Rode --freeze-codebook antes de olhar qualquer "
            "resultado — é o que sustenta o 'a priori' do §7.4."
        )
        return 1

    destination = paths.COHERENCE_BLIND_JSONL if blind else paths.COHERENCE_JSONL
    already = {(r["mutant_id"], r["case_id"]) for r in jsonl.read(destination)}

    if blind:
        first_pass = jsonl.read(paths.COHERENCE_JSONL)
        if not first_pass:
            logger.error("Não há primeira classificação para reclassificar às cegas.")
            return 1
        oldest = min(datetime.fromisoformat(r["classified_at"]) for r in first_pass)
        days = (datetime.now(timezone.utc) - oldest).days
        minimum = int(yaml.safe_load(paths.STUDY_CONFIG.read_text(encoding="utf-8"))["coherence"]["blind_min_days"])
        if days < minimum:
            logger.error(
                "Só se passaram %d dias desde a primeira classificação (mínimo %d). "
                "O intervalo é o que torna a reclassificação de fato cega.",
                days, minimum,
            )
            return 1
        rng = random.Random(seed)
        population = [(r["mutant_id"], r["case_id"]) for r in first_pass]
        sample_size = max(1, round(len(population) * fraction))
        sample = set(rng.sample(population, sample_size))
        pending = [d for d in deaths_to_classify() if (d["mutant_id"], d["case_id"]) in sample]
        rng.shuffle(pending)  # ordem diferente da primeira passada
        logger.info("Reclassificação cega de %d mortes (%.0f%% de %d).", len(pending), fraction * 100, len(population))
    else:
        pending = deaths_to_classify()

    pending = [d for d in pending if (d["mutant_id"], d["case_id"]) not in already]
    if not pending:
        logger.info("Nada a classificar — tudo já registrado em %s.", destination.name)
        return 0

    cases = {case.case_id: case for case in load_cases()}
    codebook_by_code = codebook["by_code"]

    for index, death in enumerate(pending, start=1):
        case = cases[death["case_id"]]
        print("\n" + "=" * 78)
        print(f"[{index}/{len(pending)}] mutante {death['mutant_id']} | caso {death['case_id']} ({case.case_class})")
        if not blind:
            # Na passada cega o operador fica visível (é parte do artefato), mas
            # nada do que foi decidido antes aparece.
            print(f"oráculos que mataram: {', '.join(sorted(set(death['oracles'])))}")
        print("-" * 78)
        print(f"PERGUNTA: {case.question}")
        print(f"\nREFERÊNCIA: {case.expected_output}")
        print(f"\nRESPOSTA DO MUTANTE: {_modal_answer_for(death['mutant_id'], death['case_id'])[:1500]}")
        if death["details"]:
            print("\nMOTIVOS DOS ORÁCULOS:")
            for detail in death["details"][:5]:
                print(f"  - {detail}")

        suggestion = JUDGE_CAUSE_HINT.get(death.get("judge_cause") or "")
        code = _prompt_for_code(codebook, suggestion)
        if code is None:
            logger.info("Interrompido pelo usuário; %d mortes já classificadas nesta sessão.", index - 1)
            break
        if code == "SKIP":
            continue

        expected = codebook_by_code[code].get("expected_from", []) or []
        coherent = death["operator"] in expected
        jsonl.append(
            destination,
            {
                "mutant_id": death["mutant_id"],
                "operator": death["operator"],
                "case_id": death["case_id"],
                "oracles": sorted(set(death["oracles"])),
                "coherence_code": code,
                "coherent": coherent,
                "classified_at": datetime.now(timezone.utc).isoformat(),
                "pass": "blind" if blind else "first",
            },
        )
        print(f"  -> {code} | {'COERENTE' if coherent else 'INCOERENTE'} para {death['operator']}")

    return 0


def kappa() -> int:
    first = {(r["mutant_id"], r["case_id"]): r["coherence_code"] for r in jsonl.read(paths.COHERENCE_JSONL)}
    blind = {(r["mutant_id"], r["case_id"]): r["coherence_code"] for r in jsonl.read(paths.COHERENCE_BLIND_JSONL)}
    shared = sorted(set(first) & set(blind))
    if not shared:
        logger.error("Sem itens em comum entre a primeira classificação e a cega.")
        return 1

    labels_first = [first[key] for key in shared]
    labels_blind = [blind[key] for key in shared]
    value = study_stats.cohen_kappa(labels_first, labels_blind)
    agreement = sum(1 for a, b in zip(labels_first, labels_blind) if a == b) / len(shared)

    logger.info("Kappa intra-avaliador (n=%d): %.3f (%s)", len(shared), value, study_stats.interpret_kappa(value))
    logger.info("Concordância bruta: %.1f%%", agreement * 100)
    logger.info(
        "Reportar no artigo junto da limitação declarada: não há concordância "
        "inter-avaliadores (§7.4, item 4)."
    )

    disagreements = [(key, first[key], blind[key]) for key in shared if first[key] != blind[key]]
    if disagreements:
        logger.info("Discordâncias (%d):", len(disagreements))
        for key, a, b in disagreements:
            logger.info("  %s/%s: %s -> %s", key[0], key[1], a, b)
    return 0


def summary() -> int:
    records = jsonl.read(paths.COHERENCE_JSONL)
    if not records:
        logger.info("Nenhuma morte classificada ainda.")
        return 0
    by_operator: dict[str, list[bool]] = defaultdict(list)
    by_code: dict[str, int] = defaultdict(int)
    for record in records:
        by_operator[record["operator"]].append(bool(record["coherent"]))
        by_code[record["coherence_code"]] += 1

    print(f"{'operador':<10} {'mortes':>7} {'coerentes':>10} {'taxa':>7}")
    for operator in sorted(by_operator):
        flags = by_operator[operator]
        coherent = sum(flags)
        print(f"{operator:<10} {len(flags):>7} {coherent:>10} {coherent / len(flags):>7.2f}")
    print("\nDistribuição de códigos:")
    for code, count in sorted(by_code.items(), key=lambda item: -item[1]):
        print(f"  {code:<15} {count}")
    if by_code.get("OTHER"):
        share = by_code["OTHER"] / sum(by_code.values())
        print(
            f"\nOTHER representa {share:.0%} das classificações. "
            "Proporção alta é resultado a reportar (catálogo de causas incompleto), não defeito a esconder."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classificação de coerência das mortes (§7.4).")
    parser.add_argument("--freeze-codebook", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--kappa", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--fraction", type=float, default=0.30)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if args.freeze_codebook:
        return freeze_codebook()
    if args.classify:
        return classify(blind=False)
    if args.blind:
        return classify(blind=True, fraction=args.fraction)
    if args.kappa:
        return kappa()
    if args.summary:
        return summary()
    parser.error("escolha uma ação")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
