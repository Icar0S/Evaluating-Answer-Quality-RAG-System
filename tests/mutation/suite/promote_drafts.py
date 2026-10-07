"""Promove rascunhos conferidos de drafts.jsonl para golden.jsonl.

A promoção é deliberadamente burra: copia pergunta, resposta e dica de evidência
para o slot correspondente e marca `status: ready`. Ela NÃO decide se o rascunho
presta — isso é a conferência humana, e é o passo que o protocolo chama de maior
tempo humano irredutível (§12).

O que ela faz de útil é recusar promover o que sabidamente não serve:

- rascunho com pendência automática aberta (`auto_check` diferente de `["ok"]`);
- rascunho ainda com o texto `PREENCHER`;
- caso de rastreabilidade sem citação entre colchetes na resposta;
- caso que mira C2/C4 sem citar valor que as variantes perturbam.

`--force` ignora a recusa para um caso específico, com a suposição de que você
olhou e sabe por que está promovendo assim mesmo.

Uso:
    python -m tests.mutation.suite.promote_drafts --dry-run
    python -m tests.mutation.suite.promote_drafts
    python -m tests.mutation.suite.promote_drafts --case c03 --force
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.draft_cases import DRAFTS_PATH, _altered_values, _numbers

logger = get_logger("mutation.promote")

CITATION = re.compile(r"\[[^\]]+,\s*p[aá]g\.?\s*\d+\]", re.IGNORECASE)


def blockers(draft: dict, golden_case: dict, altered: set[str]) -> list[str]:
    reasons: list[str] = []

    if draft.get("auto_check") != ["ok"]:
        reasons.append(f"pendência automática aberta: {draft.get('auto_check')}")
    if "PREENCHER" in draft.get("question", "") or "PREENCHER" in draft.get("expected_output", ""):
        reasons.append("ainda tem PREENCHER")
    if not draft.get("question", "").strip() or not draft.get("expected_output", "").strip():
        reasons.append("pergunta ou resposta vazia")

    if golden_case.get("expected_behavior") == "cite" and not CITATION.search(draft["expected_output"]):
        reasons.append("caso de rastreabilidade sem citação [documento, pág. N] na resposta")

    if {"C2", "C4"} & set(golden_case.get("targets", [])):
        body = re.sub(r"\[[^\]]*\]", " ", draft["expected_output"])

        def bare(value: str) -> str:
            return value.rstrip("%").rstrip(".,").replace(",", ".")

        if not ({bare(n) for n in _numbers(body)} & {bare(n) for n in altered}):
            reasons.append("mira C2/C4 mas não cita valor perturbado pelas variantes")

    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promove rascunhos para golden.jsonl.")
    parser.add_argument("--case", help="promove só estes case_id (separados por vírgula)")
    parser.add_argument("--force", action="store_true", help="promove mesmo com pendência")
    parser.add_argument("--dry-run", action="store_true", help="só mostra o que faria")
    args = parser.parse_args(argv)

    drafts = {row["case_id"]: row for row in jsonl.read(DRAFTS_PATH)}
    if not drafts:
        logger.error("%s vazio — rode draft_cases.py antes.", DRAFTS_PATH)
        return 1

    golden = jsonl.read(paths.GOLDEN)
    altered = _altered_values()
    wanted = {c.strip() for c in args.case.split(",")} if args.case else None

    promoted, refused = [], []
    for case in golden:
        case_id = case["case_id"]
        if wanted and case_id not in wanted:
            continue
        draft = drafts.get(case_id)
        if draft is None:
            continue

        reasons = blockers(draft, case, altered)
        if reasons and not args.force:
            refused.append((case_id, reasons))
            continue

        case["question"] = draft["question"]
        case["expected_output"] = draft["expected_output"]
        case["evidence_hint"] = draft.get("evidence_hint", case.get("evidence_hint", ""))
        case["status"] = "ready"
        promoted.append(case_id)

    for case_id, reasons in refused:
        logger.warning("%s NÃO promovido: %s", case_id, "; ".join(reasons))
    logger.info("%d promovidos: %s", len(promoted), ", ".join(promoted) or "-")

    if args.dry_run:
        logger.info("dry-run: golden.jsonl não foi alterado.")
        return 0

    jsonl.write_all(paths.GOLDEN, golden)
    logger.info("golden.jsonl atualizado. Próximo passo: escrever as assertivas e rodar validate_suite.")
    return 0 if not refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
