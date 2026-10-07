"""Gera as assertivas determinísticas (oráculo O2) a partir das respostas de referência.

As assertivas de um caso factual são, quase sempre, "não se absteve" mais "citou
os valores que a resposta de referência cita". Escrever isso à mão para 19 casos
é trabalho mecânico e propenso a erro de digitação — um valor errado numa
assertiva reprova o baseline e derruba o caso do conjunto avaliável, que é a pior
forma de perder um caso, porque some silenciosamente do denominador.

O que este script NÃO decide: se a assertiva certa para o caso é a óbvia. Ele
gera a linha de base por classe e preserva o que já estiver escrito à mão
(`--keep-existing`, padrão), porque caso ambíguo, de recusa ou de abstenção tem
assertiva que nenhuma regra deriva da resposta.

Regras por classe:
  fato_direto, fato_distribuido  -> não abstém + cita cada valor numérico
  filtro_condicional             -> idem + enuncia a condição (se/quando/caso)
  rastreabilidade                -> idem + cita documento e página
  fora_do_corpus                 -> abstém, resposta curta      (escrita à mão)
  fora_de_escopo                 -> recusa                      (escrita à mão)
  ambiguo                        -> pede esclarecimento         (escrita à mão)

Uso:
    python -m tests.mutation.suite.build_assertions --dry-run
    python -m tests.mutation.suite.build_assertions
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.assertivas")

# Só valores com dígito viram assertiva literal. Termo conceitual ("métricas
# tradicionais") o modelo reformula legitimamente entre execuções, e exigi-lo
# transformaria variação de redação em reprovação.
VALUE = re.compile(r"\d+(?:[.,]\d+)?%?")
# Fronteira de palavra obrigatoria: `contains_any` com "se " casava dentro de
# "base em", e a assertiva de condicao do c13 passava pelo motivo errado —
# assertiva que aprova por acaso e tao ruim quanto a que reprova por acaso.
CONDITION_REGEX = (
    r"@B@b(se|quando|caso|casos|desde que|apenas|somente|porque|com base em|"
    r"a partir de|dependendo|condicao|criterio)@B@b"
).replace("@B@", chr(92))

DERIVED_CLASSES = {"fato_direto", "fato_distribuido", "filtro_condicional", "rastreabilidade"}


def values_of(answer: str) -> list[str]:
    """Valores citáveis da resposta, fora de colchetes (que carregam a página)."""
    body = re.sub(r"\[[^\]]*\]", " ", answer)
    seen: list[str] = []
    for value in VALUE.findall(body):
        cleaned = value.rstrip(".,")
        # Número de um dígito costuma ser ordinal ou item de lista, não fato.
        if len(cleaned.rstrip("%")) >= 2 and cleaned not in seen:
            seen.append(cleaned)
    return seen


def assertions_for(case) -> list[dict]:
    values = values_of(case.expected_output)
    built: list[dict] = [{"type": "must_not_abstain"}]

    if values:
        built.append({"type": "contains_all", "values": values})
    else:
        # Sem valor numérico, cai-se nos termos longos da resposta — menos
        # robusto, e por isso o script avisa para revisar à mão.
        terms = [w for w in re.findall(r"[A-Za-zÀ-ú][A-Za-zÀ-ú-]{7,}", case.expected_output)][:2]
        if terms:
            built.append({"type": "contains_any", "values": terms})

    if case.case_class == "filtro_condicional":
        built.append({"type": "regex", "pattern": CONDITION_REGEX})
    if case.case_class == "rastreabilidade":
        built.append({"type": "must_cite_document"})
        built.append({"type": "must_cite_page"})

    return built


def self_check(case, built: list[dict], defaults: dict) -> list[str]:
    """Roda o proprio oraculo O2 contra a resposta de referencia.

    Assertiva que reprova a referencia reprova o BASELINE, e um caso que reprova
    no baseline sai do conjunto avaliavel (§7.1) — some do denominador sem
    quebrar nada e sem avisar ninguem. E a falha mais cara que este arquivo pode
    conter, entao ela e verificada aqui, na geracao, e nao descoberta na
    campanha.
    """
    from tests.mutation.oracles.o2_assertions import evaluate

    verdict = evaluate(case.expected_output, built, defaults)
    if verdict.verdict == "fail":
        return [verdict.detail]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera assertivas a partir das respostas de referência.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="regrava mesmo os casos já escritos à mão")
    args = parser.parse_args(argv)

    raw = yaml.safe_load(paths.ASSERTIONS.read_text(encoding="utf-8"))
    existing = raw.get("cases", {}) or {}
    cases = {case.case_id: case for case in load_cases()}

    written, skipped, warned = [], [], []
    for case_id, case in cases.items():
        if case.case_class not in DERIVED_CLASSES:
            skipped.append(case_id)
            continue
        if case.is_draft:
            warned.append(f"{case_id}: ainda é rascunho — assertiva não derivada")
            continue

        current = existing.get(case_id) or []
        hand_written = any("PREENCHER" not in str(a.get("values", "")) for a in current) and not any(
            a.get("todo") for a in current
        )
        placeholder = any("PREENCHER" in str(a.get("values", "")) or a.get("todo") for a in current)
        if hand_written and not placeholder and not args.overwrite:
            skipped.append(case_id)
            continue

        built = assertions_for(case)
        failures = self_check(case, built, raw.get("defaults", {}))
        if failures:
            # Descarta a assertiva que a propria referencia nao cumpre, em vez de
            # gravar uma bomba-relogio. Sobra a checagem de abstencao, que e
            # fraca mas honesta — e o aviso diz o que escrever a mao.
            built = [a for a in built if not self_check(case, [a], raw.get("defaults", {}))]
            warned.append(
                f"{case_id}: assertiva descartada por reprovar a propria referencia "
                f"({failures[0][:90]}) — escreva a mao se o caso exigir mais que 'nao se absteve'"
            )
        if not values_of(case.expected_output):
            warned.append(f"{case_id}: resposta sem valor numérico — assertiva por termo, revise à mão")
        existing[case_id] = built
        written.append(case_id)

    raw["cases"] = existing
    for message in warned:
        logger.warning(message)
    logger.info("%d casos com assertiva gerada: %s", len(written), ", ".join(written) or "-")
    logger.info("%d preservados (escritos à mão ou fora do escopo da derivação)", len(skipped))

    if args.dry_run:
        logger.info("dry-run: assertions.yaml não foi alterado.")
        return 0

    header = paths.ASSERTIONS.read_text(encoding="utf-8").split("defaults:")[0]
    body = yaml.safe_dump(
        {"defaults": raw.get("defaults", {}), "cases": existing},
        allow_unicode=True, sort_keys=False, default_flow_style=None, width=100,
    )
    paths.ASSERTIONS.write_text(header + body, encoding="utf-8")
    logger.info("assertions.yaml atualizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
