"""O2 — assertivas determinísticas sobre a resposta (§6.4).

Custo baixo, roda em cada execução, sem chamada de modelo. É o oráculo que um
time de verdade escreveria à mão, e é por isso que ele é a referência de
"suíte típica" contra a qual os outros quatro são comparados na RQ2.

Só olha a resposta. Ver o cabeçalho de suite/assertions.yaml para a justificativa.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from tests.mutation.runner import paths
from tests.mutation.oracles.base import Verdict, extract_numbers, normalize

# Frase-âncora de abstenção do SUT. Importada por valor (e não do módulo do
# backend) seria frágil: se o prompt mudar, o oráculo tem de mudar junto.
from tests.mutation.runner import sut  # noqa: F401  (garante backend no sys.path)
from app.rag.generation import NOT_FOUND_MARKER

REFUSAL_CUES = (
    "nao posso ajudar",
    "nao posso responder",
    "fora do escopo",
    "nao faz parte do escopo",
    "sou um assistente especializado",
    "minha especialidade",
    "nao e o meu dominio",
)

CLARIFICATION_CUES = (
    "esclarec",
    "poderia especificar",
    "pode especificar",
    "a qual",
    "a que ",
    "qual delas",
    "nao esta claro",
    "voce se refere",
    "se refere a",
    "reformular",
)

# [documento, pág. 4] / (pag 4) / p. 12 — todas as formas que o prompt de citação
# pode induzir, para não reprovar o SUT por uma vírgula.
PAGE_CITATION_RE = re.compile(r"(?:p[aá]g\.?|p\.|pagina|página)\s*\d+", re.IGNORECASE)


def abstained(answer: str) -> bool:
    return normalize(NOT_FOUND_MARKER) in normalize(answer)


@lru_cache(maxsize=1)
def _document_aliases() -> tuple[str, ...]:
    """Nomes pelos quais uma resposta pode citar um documento do corpus."""
    if not paths.CORPUS_MANIFEST.exists():
        return ()
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    aliases: set[str] = set()
    for document in manifest.get("documents", []):
        name = document["document"]
        stem = name[:-4] if name.endswith(".pdf") else name
        aliases.add(normalize(name))
        aliases.add(normalize(stem))
        aliases.add(normalize(stem.replace("_", " ")))
        # Prefixo distintivo: modelos frequentemente encurtam nomes longos.
        words = stem.replace("_", " ").split()
        if len(words) >= 4:
            aliases.add(normalize(" ".join(words[:4])))
    return tuple(sorted(a for a in aliases if len(a) >= 8))


def _check(assertion: dict[str, Any], answer: str, defaults: dict[str, Any]) -> tuple[bool, str]:
    kind = assertion["type"]
    ignore_case = assertion.get("ignore_case", defaults.get("ignore_case", True))
    ignore_accents = assertion.get("ignore_accents", defaults.get("ignore_accents", True))
    haystack = normalize(answer, ignore_case, ignore_accents)

    def norm(value: str) -> str:
        return normalize(value, ignore_case, ignore_accents)

    if kind == "contains_all":
        missing = [v for v in assertion["values"] if norm(v) not in haystack]
        return (not missing), ("" if not missing else f"faltou: {missing}")

    if kind == "contains_any":
        found = any(norm(v) in haystack for v in assertion["values"])
        return found, ("" if found else f"nenhum de {assertion['values']}")

    if kind == "not_contains":
        present = [v for v in assertion["values"] if norm(v) in haystack]
        return (not present), ("" if not present else f"nao deveria conter: {present}")

    if kind == "regex":
        ok = re.search(assertion["pattern"], answer, re.IGNORECASE | re.DOTALL) is not None
        return ok, ("" if ok else f"nao casou com /{assertion['pattern']}/")

    if kind == "not_regex":
        ok = re.search(assertion["pattern"], answer, re.IGNORECASE | re.DOTALL) is None
        return ok, ("" if ok else f"casou com /{assertion['pattern']}/ e nao deveria")

    if kind == "must_abstain":
        ok = abstained(answer)
        return ok, ("" if ok else "deveria usar a frase de abstencao")

    if kind == "must_not_abstain":
        ok = not abstained(answer)
        return ok, ("" if ok else "abstencao indevida")

    if kind == "must_refuse":
        ok = abstained(answer) or any(cue in haystack for cue in REFUSAL_CUES)
        return ok, ("" if ok else "deveria recusar e respondeu")

    if kind == "must_ask_clarification":
        ok = "?" in answer and any(cue in haystack for cue in CLARIFICATION_CUES)
        return ok, ("" if ok else "deveria pedir esclarecimento")

    if kind == "must_cite_document":
        aliases = _document_aliases()
        ok = any(alias in haystack for alias in aliases)
        return ok, ("" if ok else "nao citou nenhum documento do corpus")

    if kind == "must_cite_page":
        ok = PAGE_CITATION_RE.search(answer) is not None
        return ok, ("" if ok else "nao citou pagina")

    if kind == "numeric_equals":
        expected = float(assertion["value"])
        tolerance = float(assertion.get("tolerance", 0.0))
        ok = any(abs(number - expected) <= tolerance for number in extract_numbers(answer))
        return ok, ("" if ok else f"nenhum numero igual a {expected}")

    if kind == "max_words":
        limit = int(assertion["value"])
        count = len(answer.split())
        return count <= limit, ("" if count <= limit else f"{count} palavras (limite {limit})")

    raise ValueError(f"Tipo de assertiva desconhecido: {kind}")


def evaluate(
    answer: str,
    case_assertions: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> Verdict:
    defaults = defaults or {}
    if not case_assertions:
        return Verdict(
            oracle="O2",
            verdict="pass",
            evaluable=False,
            detail="caso sem assertivas declaradas",
        )

    failures: list[str] = []
    for index, assertion in enumerate(case_assertions):
        ok, reason = _check(assertion, answer, defaults)
        if not ok:
            failures.append(f"[{index}:{assertion['type']}] {reason}")

    passed = len(case_assertions) - len(failures)
    return Verdict(
        oracle="O2",
        verdict="fail" if failures else "pass",
        score=round(passed / len(case_assertions), 4),
        detail="; ".join(failures),
        components={"total": len(case_assertions), "passed": passed},
    )
