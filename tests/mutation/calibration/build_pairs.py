"""Gera os 300 pares rotulados por construção que calibram τ* (§6.3).

Sem rotulação manual. O rótulo é conhecido porque quem constrói o par sabe o que
fez:

- **EQUIV (150)**: paráfrases das respostas de referência, geradas por um modelo
  de família distinta da do SUT, com instrução explícita de preservar todos os
  fatos. O protocolo prevê conferência visual de 100% (~2 h); por isso cada par
  nasce com `reviewed: false` e o campo existe para ser marcado — calibrate.py
  avisa quando calibra sobre pares não conferidos.

- **ERRO (150)**: perturbações programáticas da referência, uma por item — trocar
  um número, inverter uma condição, alterar um prazo, substituir uma entidade
  nomeada. O script registra exatamente o que mudou, então o rótulo não depende
  de julgamento.

Limitação que vai declarada no artigo: erro gerado programaticamente pode ser
mais fácil de detectar que erro natural de um LLM, então τ* é um **limite
superior** do desempenho do oráculo O1.

Uso:
    python -m tests.mutation.calibration.build_pairs
    python -m tests.mutation.calibration.build_pairs --per-case 5 --only erro
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import load_study_config
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.calibration")

PARAPHRASE_SYSTEM = """Você reescreve respostas técnicas em português preservando TODOS os fatos.

Regras:
1. Preserve cada número, unidade, condição, prazo e nome próprio EXATAMENTE como estão.
2. Mude apenas a forma: ordem das orações, sinônimos, voz verbal, conectivos.
3. Não acrescente informação nova e não remova nenhuma.
4. Responda apenas com a reescrita, sem comentários."""

# --- perturbações programáticas (classe ERRO) --------------------------------

CONDITION_FLIPS = [
    (r"\bdeve\b", "não deve"),
    (r"\bdevem\b", "não devem"),
    (r"\bmaior\b", "menor"),
    (r"\bmenor\b", "maior"),
    (r"\bantes\b", "depois"),
    (r"\bdepois\b", "antes"),
    (r"\bsempre\b", "nunca"),
    (r"\bcom\b", "sem"),
    (r"\bpode\b", "não pode"),
    (r"\bapenas\b", "inclusive"),
    (r"\bporque\b", "embora"),
    (r"\bexclusivamente\b", "parcialmente"),
    (r"\bsomente\b", "também"),
    (r"\btodas\b", "algumas"),
    (r"\btodos\b", "alguns"),
    (r"\bmantem\b", "perde"),
    (r"\bestavel\b", "instavel"),
    (r"\bsuperando\b", "ficando atrás de"),
    (r"\binfluencia\b", "não influencia"),
    (r"\bnenhum\b", "algum"),
    (r"\bem vez de\b", "além de"),
    (r"\bnao revelou\b", "revelou"),
    (r"\bassociados\b", "desvinculados"),
    (r"\bassociado\b", "desvinculado"),
    (r"\bselecionada\b", "descartada"),
    (r"\bselecionadas\b", "descartadas"),
    (r"\breexecutada\b", "executada uma única vez"),
    (r"\bacima\b", "abaixo"),
    (r"\babaixo\b", "acima"),
    (r"\bmantém\b", "perde"),
    (r"\bestável\b", "instável"),
    (r"\bé necessário\b", "é dispensável"),
    (r"\bnão revelou\b", "revelou"),
    (r"\binfluência\b", "ausência de influência"),
    (r"\bassociadas\b", "desvinculadas"),
    (r"\breexecutadas\b", "executadas uma única vez"),
]

# Marcadores de prosa tecnica em portugues que as treze regras originais nao
# cobriam. Sem eles, catorze dos trinta casos produziam menos de cinco
# perturbacoes e a classe ERRO ficava com 87 pares contra 150 da EQUIV — um
# desequilibrio que desloca o tau* escolhido por F1 maximo.
DEADLINE_RE = re.compile(r"\b(\d{1,4})\s*(dias?|horas?|minutos?|segundos?|semanas?|meses|anos?)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,6}(?:[.,]\d{1,3})?)(%?)(?![\w])")
ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)?)\b")

ENTITY_SUBSTITUTES = ["Delta", "Orion", "Vega", "Atlas", "Nimbus", "Sigma"]


def _perturb_number(text: str, rng: random.Random) -> tuple[str, str] | None:
    """Troca um numero por outro plausivel.

    Varias transformacoes em vez de uma so: resposta curta com um unico numero
    (`"16 classes"`) admitia exatamente uma perturbacao, e tres casos ficavam com
    um par ERRO cada. Todas continuam dentro da regra "trocar um numero" do §6.3;
    o que muda e que a mesma referencia rende erros distintos, o que e o que a
    classe precisa para chegar ao tamanho previsto.
    """
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    original = match.group(1)
    try:
        value = float(original.replace(",", "."))
    except ValueError:
        return None

    decimals = len(original.split(",")[-1]) if "," in original else (
        len(original.split(".")[-1]) if "." in original else 0
    )
    transforms = [
        ("inflado", lambda v: v * 1.5 + 1),
        ("reduzido", lambda v: v * 0.5),
        ("deslocado", lambda v: v + 10),
        ("rebaixado", lambda v: max(0.0, v - 10)),
        ("arredondado", lambda v: v * 2),
    ]
    label, transform = rng.choice(transforms)
    changed = transform(value) if value != 0 else 7

    replacement = f"{changed:.{decimals}f}" if decimals else str(int(round(changed)))
    if "," in original:
        replacement = replacement.replace(".", ",")
    if replacement == original:
        return None

    new_text = text[: match.start(1)] + replacement + text[match.end(1) :]
    return new_text, f"numero {original} -> {replacement} ({label})"


def _perturb_condition(text: str, rng: random.Random) -> tuple[str, str] | None:
    candidates = [(pattern, target) for pattern, target in CONDITION_FLIPS if re.search(pattern, text, re.IGNORECASE)]
    if not candidates:
        return None
    pattern, target = rng.choice(candidates)
    new_text = re.sub(pattern, target, text, count=1, flags=re.IGNORECASE)
    return new_text, f"condicao {pattern} -> {target}"


def _perturb_deadline(text: str, rng: random.Random) -> tuple[str, str] | None:
    matches = list(DEADLINE_RE.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    quantity = int(match.group(1))
    changed = quantity * 2 + 1
    new_text = text[: match.start(1)] + str(changed) + text[match.end(1) :]
    return new_text, f"prazo {quantity} -> {changed} {match.group(2)}"


def _perturb_entity(text: str, rng: random.Random) -> tuple[str, str] | None:
    matches = [m for m in ENTITY_RE.finditer(text) if m.start() > 0]
    if not matches:
        return None
    match = rng.choice(matches)
    original = match.group(1)
    replacement = rng.choice([s for s in ENTITY_SUBSTITUTES if s.lower() != original.lower()])
    new_text = text[: match.start(1)] + replacement + text[match.end(1) :]
    return new_text, f"entidade {original} -> {replacement}"


PERTURBATIONS: list[tuple[str, Callable[[str, random.Random], "tuple[str, str] | None"]]] = [
    ("swap_number", _perturb_number),
    ("invert_condition", _perturb_condition),
    ("change_deadline", _perturb_deadline),
    ("replace_entity", _perturb_entity),
]


def build_error_pairs(reference: str, case_id: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Uma perturbação por item, variando a regra; pula regras que não se aplicam."""
    pairs: list[dict[str, Any]] = []
    attempts = 0
    rule_index = 0
    seen: set[str] = set()

    while len(pairs) < count and attempts < count * len(PERTURBATIONS) * 2:
        attempts += 1
        rule_name, rule = PERTURBATIONS[rule_index % len(PERTURBATIONS)]
        rule_index += 1
        outcome = rule(reference, rng)
        if outcome is None:
            continue
        candidate, change = outcome
        if candidate == reference or candidate in seen:
            continue
        seen.add(candidate)
        pairs.append(
            {
                "pair_id": f"{case_id}-err{len(pairs) + 1}",
                "case_id": case_id,
                "label": "ERRO",
                "reference": reference,
                "candidate": candidate,
                "rule": rule_name,
                "change": change,
                "reviewed": True,  # rótulo conhecido por construção
            }
        )
    return pairs


def generate_paraphrases(
    reference: str, model: str, base_url: str, count: int, timeout: int
) -> list[str]:
    paraphrases: list[str] = []
    # Mais tentativas que o alvo: com temperatura 0,8 e seeds diferentes o modelo
    # ainda repete bastante, e pedir exatamente N rendia 80 parafrases unicas onde
    # se esperava 154 — a classe EQUIV ficava MENOR que a ERRO, invertendo o
    # desequilibrio que se queria corrigir.
    for index in range(count * 3):
        if len(paraphrases) >= count:
            break
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": f"Reescreva:\n\n{reference}"},
            ],
            "stream": False,
            # Temperatura > 0 aqui é proposital: queremos 5 paráfrases DIFERENTES
            # da mesma referência, não a mesma cinco vezes.
            "options": {"temperature": 0.8, "seed": index + 1, "num_ctx": 8192},
        }
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "").strip()
        # Copia da referencia nao e parafrase: entra no conjunto como par de
        # similaridade 1,0 e puxa o limiar para cima sem informar nada.
        if text and text not in paraphrases and text.strip() != reference.strip():
            paraphrases.append(text)
    return paraphrases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera calibration/pairs.jsonl (300 pares).")
    parser.add_argument("--per-case", type=int, default=5, help="pares por classe por caso (5 x 30 = 150)")
    parser.add_argument("--only", choices=["equiv", "erro"], help="gera apenas uma das classes")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--exclude-behaviors",
        default="abstain,refuse",
        help="comportamentos cuja referencia nao serve a calibracao (padrao: abstain,refuse)",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    study = load_study_config()
    models = study.models
    base_url = study.baseline_overrides.get("ollama_base_url") or "http://localhost:11434"
    timeout = int(study.baseline_overrides.get("generation_timeout_seconds", 300))

    excluded_behaviors = {b.strip() for b in args.exclude_behaviors.split(",") if b.strip()}
    todos = [case for case in load_cases() if not case.is_draft]
    cases = [case for case in todos if case.expected_behavior not in excluded_behaviors]
    if len(cases) < len(todos):
        # Caso de abstencao tem a MESMA frase fixa como referencia. Ele nao produz
        # ERRO (nao existe versao "errada plausivel" de uma abstencao) e suas
        # parafrases seriam N variantes de uma unica sentenca — pares quase
        # identicos entre si que inflam a classe EQUIV com similaridade alta e
        # empurram o tau* para cima. Calibrar O1 sobre eles mediria a capacidade
        # do cosseno de reconhecer uma frase fixa, nao de julgar uma resposta.
        logger.info(
            "%d casos fora da calibracao (comportamento em %s): a referencia deles e "
            "a frase fixa de abstencao/recusa.",
            len(todos) - len(cases), sorted(excluded_behaviors),
        )
    if not cases:
        logger.error(
            "Nenhum caso pronto em suite/golden.jsonl. A calibração usa as respostas de "
            "referência — escreva os casos antes (semana 2)."
        )
        return 1
    if len(cases) < 30:
        logger.warning(
            "Só %d casos prontos: serão gerados %d pares por classe em vez de 150.",
            len(cases),
            len(cases) * args.per_case,
        )

    rng = random.Random(args.seed)
    pairs: list[dict[str, Any]] = []

    for case in cases:
        reference = case.expected_output

        if args.only != "erro":
            try:
                paraphrases = generate_paraphrases(
                    reference, models["paraphraser"], base_url, args.per_case, timeout
                )
            except httpx.HTTPError as exc:
                logger.error("Falha ao gerar paráfrases para %s: %s", case.case_id, exc)
                return 1
            for index, paraphrase in enumerate(paraphrases, start=1):
                pairs.append(
                    {
                        "pair_id": f"{case.case_id}-eq{index}",
                        "case_id": case.case_id,
                        "label": "EQUIV",
                        "reference": reference,
                        "candidate": paraphrase,
                        "rule": "paraphrase",
                        "change": f"modelo {models['paraphraser']}",
                        # Conferência visual de 100% é exigência do §6.3.
                        "reviewed": False,
                    }
                )

        if args.only != "equiv":
            pairs.extend(build_error_pairs(reference, case.case_id, args.per_case, rng))

    jsonl.write_all(paths.CALIBRATION_PAIRS, pairs)
    equiv = sum(1 for p in pairs if p["label"] == "EQUIV")
    erro = sum(1 for p in pairs if p["label"] == "ERRO")
    logger.info("%s: %d pares (EQUIV=%d, ERRO=%d)", paths.CALIBRATION_PAIRS.name, len(pairs), equiv, erro)
    if equiv:
        logger.info(
            "Próximo passo (§6.3): conferir visualmente os %d pares EQUIV e marcar "
            '"reviewed": true nos que de fato preservam todos os fatos.',
            equiv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
