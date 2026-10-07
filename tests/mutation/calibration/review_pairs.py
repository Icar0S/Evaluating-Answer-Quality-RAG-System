"""Apoia a conferência visual dos 150 pares EQUIV exigida pelo §6.3.

O protocolo pede conferência de 100% pelo pesquisador, e com razão: uma paráfrase
que perdeu um fato é um rótulo errado, e rótulo errado desloca τ*, que por sua
vez decide o veredito de todo o oráculo O1 na campanha inteira.

Este script NÃO substitui essa leitura. Ele faz a parte mecânica — conferir se
todo número e toda entidade da referência sobreviveram na paráfrase — e separa os
pares em dois montes:

- **limpos**: passaram nas checagens mecânicas e precisam só de leitura de
  sentido (a paráfrase diz a mesma coisa?);
- **marcados**: têm evidência objetiva de que perderam, alteraram ou
  acrescentaram informação, e provavelmente devem ser rejeitados.

A checagem mecânica pega o erro mais comum (número sumiu ou mudou) e não pega o
mais sutil (mesma informação, sentido invertido). Por isso `--approve` exige que
você diga quais aprovou; não há atalho que marque tudo como conferido.

Uso:
    python -m tests.mutation.calibration.review_pairs --check
    python -m tests.mutation.calibration.review_pairs --list --only-flagged
    python -m tests.mutation.calibration.review_pairs --list --case c01
    python -m tests.mutation.calibration.review_pairs --approve c01-eq1,c01-eq2
    python -m tests.mutation.calibration.review_pairs --reject c07-eq3 --reason "perdeu o 26,0%"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger

logger = get_logger("mutation.revisao")

NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")
# Entidade: token que começa com maiúscula no meio da frase, ou identificador
# com dígito/underscore (nomes de modelo, ferramentas, configurações).
ENTITY = re.compile(r"\b(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ]{2,}|[A-Za-z]+[-_][A-Za-z0-9-]+|[A-Z]{2,})\b")

LENGTH_MIN, LENGTH_MAX = 0.45, 2.2


def normalized_numbers(text: str) -> set[str]:
    """Números comparáveis: sem separador decimal e sem pontuação de borda."""
    found = set()
    for raw in NUMBER.findall(text):
        cleaned = raw.rstrip(".,").replace(",", ".").rstrip("%")
        if cleaned:
            found.add(cleaned)
    return found


def entities(text: str) -> set[str]:
    stop = {"O", "A", "Os", "As", "Um", "Uma", "De", "Da", "Do", "Em", "Na", "No",
            "Para", "Com", "Por", "Que", "Quando", "Cada", "Este", "Essa", "Foram",
            "Foi", "Sao", "Sua", "Seu", "Demonstram", "Segundo"}
    return {e for e in ENTITY.findall(text) if e not in stop and len(e) > 2}


def check_pair(pair: dict) -> list[str]:
    """Evidência objetiva de que a paráfrase não preserva a referência."""
    reference, candidate = pair["reference"], pair["candidate"]
    problems: list[str] = []

    missing = normalized_numbers(reference) - normalized_numbers(candidate)
    if missing:
        problems.append(f"perdeu o(s) valor(es) {sorted(missing)}")

    added = normalized_numbers(candidate) - normalized_numbers(reference)
    if added:
        problems.append(f"acrescentou valor(es) inexistente(s) na referência: {sorted(added)}")

    lost_entities = entities(reference) - entities(candidate)
    if len(lost_entities) > 2:
        problems.append(f"perdeu entidades: {sorted(lost_entities)[:5]}")

    ratio = len(candidate) / max(1, len(reference))
    if not (LENGTH_MIN <= ratio <= LENGTH_MAX):
        problems.append(f"tamanho {ratio:.2f}x o da referência (esperado {LENGTH_MIN}-{LENGTH_MAX}x)")

    if candidate.strip() == reference.strip():
        problems.append("idêntica à referência — não é paráfrase")

    return problems


def load() -> list[dict]:
    pairs = jsonl.read(paths.CALIBRATION_PAIRS)
    if not pairs:
        raise SystemExit(f"{paths.CALIBRATION_PAIRS} vazio — rode build_pairs.py antes.")
    return pairs


def save(pairs: list[dict]) -> None:
    jsonl.write_all(paths.CALIBRATION_PAIRS, pairs)


def command_check(pairs: list[dict]) -> int:
    equiv = [p for p in pairs if p["label"] == "EQUIV"]
    flagged = 0
    for pair in equiv:
        problems = check_pair(pair)
        pair["auto_check"] = problems or ["ok"]
        flagged += bool(problems)
    save(pairs)

    logger.info("%d pares EQUIV conferidos mecanicamente; %d marcados.", len(equiv), flagged)
    logger.info("%d limpos aguardam leitura de sentido.", len(equiv) - flagged)
    reviewed = sum(1 for p in equiv if p.get("reviewed"))
    logger.info("%d/%d já marcados como conferidos.", reviewed, len(equiv))
    return 0


def command_list(pairs: list[dict], only_flagged: bool, case: str | None, limit: int) -> int:
    shown = 0
    for pair in pairs:
        if pair["label"] != "EQUIV":
            continue
        if case and pair["case_id"] != case:
            continue
        problems = pair.get("auto_check") or check_pair(pair) or ["ok"]
        if only_flagged and problems == ["ok"]:
            continue
        marker = "OK " if problems == ["ok"] else "!! "
        status = "conferido" if pair.get("reviewed") else "pendente"
        print(f"\n{marker}{pair['pair_id']} [{status}]")
        if problems != ["ok"]:
            print(f"   check: {'; '.join(problems)}")
        print(f"   REF: {pair['reference']}")
        print(f"   PAR: {pair['candidate']}")
        shown += 1
        if shown >= limit:
            print(f"\n(limite de {limit} atingido; use --limit para ver mais)")
            break
    if shown == 0:
        print("nenhum par corresponde ao filtro.")
    return 0


def command_mark(pairs: list[dict], ids: str, reviewed: bool, reason: str | None) -> int:
    wanted = {i.strip() for i in ids.split(",")}
    index = {pair["pair_id"]: pair for pair in pairs}
    missing = wanted - set(index)
    if missing:
        raise SystemExit(f"pair_id inexistente: {sorted(missing)}")

    for pair_id in wanted:
        pair = index[pair_id]
        pair["reviewed"] = reviewed
        if reason:
            pair["review_note"] = reason
        if not reviewed:
            # Par rejeitado sai do conjunto de calibração: rótulo duvidoso é pior
            # que amostra menor, porque desloca τ* sem deixar rastro.
            pair["excluded"] = True
    save(pairs)
    logger.info("%d pares marcados como %s.", len(wanted), "conferidos" if reviewed else "REJEITADOS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conferência dos pares EQUIV da calibração.")
    parser.add_argument("--check", action="store_true", help="roda as checagens mecânicas")
    parser.add_argument("--list", action="store_true", help="imprime os pares para leitura")
    parser.add_argument("--only-flagged", action="store_true")
    parser.add_argument("--case", help="filtra por case_id")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--approve", help="pair_id(s) conferidos e aprovados")
    parser.add_argument("--reject", help="pair_id(s) rejeitados (saem da calibração)")
    parser.add_argument("--reason", help="nota do porquê da rejeição")
    args = parser.parse_args(argv)

    pairs = load()
    if args.check:
        return command_check(pairs)
    if args.list:
        return command_list(pairs, args.only_flagged, args.case, args.limit)
    if args.approve:
        return command_mark(pairs, args.approve, True, args.reason)
    if args.reject:
        return command_mark(pairs, args.reject, False, args.reason)
    parser.error("escolha uma ação: --check, --list, --approve ou --reject")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
