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

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

CATEGORIES = ("recusa", "sem_citacao", "fabricada", "outro_documento", "fonte")
# O gemma3:4b cita de vários jeitos, e todos contam (visto nas respostas do E3, 19/10/2026): o formato da regra,
# "[arquivo.pdf, pág. N]"; o cabeçalho do trecho copiado, "[Trecho 2 - arquivo.pdf (pág. N)]" ou
# "[arquivo.pdf (pág. N)]", às vezes sem o parêntese final; o número do trecho, "[Trecho 2, pág. N]"; e o nome
# do artigo sem a extensão, ou o título (o cabeçalho contextual do E3 mostra o título), com a página:
# "[AI LitReview, pág. 3]". "[12]" sozinho é a referência do artigo copiada do trecho, e não citação.
_PAGE = r"(?:\s*,\s*p[áa]g(?:ina)?\.?\s*(\d+)|\s*\(\s*p[áa]g(?:ina)?\.?\s*(\d+)\s*\)?)"
_BY_FILE = re.compile(r"\[\s*(?:Trecho\s+\d+\s*[-–—]\s*)?([^\[\]\n]+?\.(?:pdf|md))" + _PAGE + r"?\s*\]", re.IGNORECASE)
_BY_NUMBER = re.compile(r"\[\s*Trecho\s+(\d+)\s*(?:,\s*p[áa]g(?:ina)?\.?\s*(\d+))?\s*\]", re.IGNORECASE)
# Nome sem extensão: só com a página, para não confundir com qualquer texto entre colchetes.
_BY_NAME = re.compile(r"\[\s*(?:Trecho\s+\d+\s*[-–—]\s*)?([^\[\]\n(),]{3,}?)" + _PAGE + r"\s*\]", re.IGNORECASE)
_SOURCE = re.compile(r"^(.*?)(?: \(pág\. (\d+)\))?$")
_TITLES = Path(__file__).resolve().parents[2] / "backend" / "app" / "rag" / "titles.json"


def _stem(name: str) -> str:
    return Path(name.strip()).stem.lower()


def _words(text: str) -> list[str]:
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))
    return re.findall(r"[0-9a-z]+", plain)


@lru_cache(maxsize=1)
def _titles() -> dict[str, str]:
    """Nome sem extensão (minúsculo) → título, da tabela do cabeçalho do E3 (backend/app/rag/titles.json)."""
    if not _TITLES.is_file():
        return {}
    return {stem.lower(): entry["titulo"] for stem, entry in json.loads(_TITLES.read_text(encoding="utf-8")).items()}


def _by_name(name: str, documents: list[str]) -> str | None:
    """O documento que o nome citado (sem extensão, ou o título) designa, entre `documents`: as palavras do nome
    são o começo das palavras do nome do arquivo ou do título ("ARES" designa "ARES An Automated...", "Eval" não
    designa "Evaluating...")."""
    wanted, titles = _words(name), _titles()
    if not wanted or len("".join(wanted)) < 3:
        return None
    for document in documents:
        for known in (_words(document), _words(titles.get(document, ""))):
            if known and known[: len(wanted)] == wanted:
                return document
    return None


def _source(entry: str) -> tuple[str, int | None]:
    match = _SOURCE.match(entry.strip())
    return _stem(match.group(1)), int(match.group(2)) if match.group(2) else None


def citations(answer: str | None, sources: list[str] | None = None) -> list[tuple[str, int | None]]:
    """(documento, página) de cada citação, na ordem do texto. "Trecho N" vira o N-ésimo trecho recebido
    (`sources`, na ordem do prompt); trecho que não existe vira um documento que não veio (fabricação). Nome sem
    extensão vira o documento recebido que ele designa, ou outro artigo do acervo (fabricação); nome que não
    designa nada não é citação."""
    text, found, taken = answer or "", [], []
    for match in _BY_FILE.finditer(text):
        page = match.group(2) or match.group(3)
        found.append((match.start(), (_stem(match.group(1)), int(page) if page else None)))
        taken.append(match.span())
    for match in _BY_NUMBER.finditer(text):
        n = int(match.group(1))
        document, page = _source(sources[n - 1]) if sources and 1 <= n <= len(sources) else (f"trecho {n}", None)
        found.append((match.start(), (document, int(match.group(2)) if match.group(2) else page)))
        taken.append(match.span())
    got = [_source(s)[0] for s in sources or []]
    for match in _BY_NAME.finditer(text):
        if any(start <= match.start() < end for start, end in taken) or re.match(r"\[\s*Trecho\s+\d+\s*[,\]]",
                                                                               match.group(0), re.IGNORECASE):
            continue
        document = _by_name(match.group(1), got) or _by_name(match.group(1), list(_titles()))
        if document:
            page = match.group(2) or match.group(3)
            found.append((match.start(), (document, int(page) if page else None)))
    return [citation for _, citation in sorted(found)]


def received(record: dict) -> set[tuple[str, int | None]]:
    """(documento, página) dos trechos que o gerador recebeu, de `sources` ("arquivo (pág. N)")."""
    return {_source(source) for source in record.get("sources") or []}


def _matches(citation: tuple[str, int | None], got: set[tuple[str, int | None]]) -> bool:
    """A citação aponta para um trecho recebido: o documento veio e, se ela dá página, a página também."""
    document, page = citation
    pages = {p for d, p in got if d == document}
    return bool(pages) and (page is None or None in pages or page in pages)


def classify(record: dict, refused: bool) -> str:
    """A categoria (tabela do docstring) da resposta a uma pergunta de dentro."""
    if refused:
        return "recusa"
    cited = citations(record.get("actual_output"), record.get("sources"))
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
    return not refused and bool(citations(record.get("actual_output"), record.get("sources")))
