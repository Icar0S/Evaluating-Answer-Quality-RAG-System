"""Atribuição errada (plano-benchmark.md, §6, C2), medida nos braços com citação obrigatória (E3).

A regra de citação (backend/app/rag/generation.py, CITATION_RULE) pede
"[nome_do_arquivo.pdf, pág. N]". Cada resposta de uma pergunta de dentro cai numa
categoria:

| Categoria | Definição |
|---|---|
| recusa | a resposta é uma recusa (não precisa citar) |
| sem_citacao (a) | não cita nenhum trecho: não conformidade |
| fabricada (b) | cita arquivo (ou arquivo e página) que não está entre os trechos recebidos: **atribuição errada** |
| outro_documento (c) | cita só trechos recebidos de outro documento: erro de recuperação, já contado na C1 |
| fonte (d) | cita o documento do golden, num trecho que veio dele: correto |

Também é atribuição errada a resposta não abstida a uma pergunta de fora que cita
algum trecho. O denominador da taxa é o total de respostas, não só as com
citação. Documento é comparado pelo nome sem extensão (o .md e o .pdf do mesmo
artigo são o mesmo). Só biblioteca padrão.
"""
from __future__ import annotations

import re
from pathlib import Path

CATEGORIES = ("recusa", "sem_citacao", "fabricada", "outro_documento", "fonte")
_CITATION = re.compile(r"\[\s*([^\[\],]+?\.(?:pdf|md))\s*(?:,\s*p[áa]g(?:ina)?\.?\s*(\d+))?\s*\]", re.IGNORECASE)
_SOURCE = re.compile(r"^(.*?)(?: \(pág\. (\d+)\))?$")


def _stem(name: str) -> str:
    return Path(name.strip()).stem.lower()


def citations(answer: str | None) -> list[tuple[str, int | None]]:
    """(documento, página) de cada citação no formato da regra, na ordem do texto."""
    return [(_stem(name), int(page) if page else None) for name, page in _CITATION.findall(answer or "")]


def received(record: dict) -> set[tuple[str, int | None]]:
    """(documento, página) dos trechos que o gerador recebeu, de `sources` ("arquivo (pág. N)")."""
    pairs = set()
    for source in record.get("sources") or []:
        match = _SOURCE.match(source.strip())
        pairs.add((_stem(match.group(1)), int(match.group(2)) if match.group(2) else None))
    return pairs


def _matches(citation: tuple[str, int | None], got: set[tuple[str, int | None]]) -> bool:
    """A citação aponta para um trecho recebido: o documento veio e, se ela dá página, a página também."""
    document, page = citation
    pages = {p for d, p in got if d == document}
    return bool(pages) and (page is None or None in pages or page in pages)


def classify(record: dict, refused: bool) -> str:
    """A categoria (tabela do docstring) da resposta a uma pergunta de dentro."""
    if refused:
        return "recusa"
    cited = citations(record.get("actual_output"))
    if not cited:
        return "sem_citacao"
    got = received(record)
    if not all(_matches(c, got) for c in cited):
        return "fabricada"
    source = _stem(record["source_document"]) if record.get("source_document") else None
    return "fonte" if any(document == source for document, _ in cited) else "outro_documento"


def misattributed(record: dict, refused: bool, inside: bool) -> bool:
    """Atribuição errada: citação fabricada (de dentro) ou citação numa resposta não abstida (de fora)."""
    if inside:
        return classify(record, refused) == "fabricada"
    return not refused and bool(citations(record.get("actual_output")))
