"""C0 · Acervo: o texto que entra no índice presta? (plano-benchmark.md, §5 e §6)

Métricas por documento, sobre o MESMO texto que o chunker recebe (as páginas do
PyMuPDF, ou o Markdown já normalizado), gravadas no manifesto da versão
(app/rag/ingestion.py). Não barram nada: explicam de antemão o que a
recuperação vai mostrar. Sem LLM e sem embeddings: segundos para um acervo.

- glued_word_rate: fração das LETRAS que estão em tokens alfabéticos de 18+
  letras fora do dicionário (c0_long_words.txt). Medir em letras não depende de
  quantas palavras cada token grudado engoliu. É um limite inferior: grude
  curto ("ofthe") escapa. Sobrenome e identificador de código longos também
  contam: nos 39 PDFs originais, esse piso chega a 1,2%.
- references_share: fração dos tokens no bloco de referências (references_block).
- pages_total e pages_with_text: a ingestão descarta página sem texto, sem aviso.
- tokens e chunk_tokens (média, p10, p90): o PDF é cortado por página, e o
  chunk efetivo pode ficar bem abaixo do pedido.

A estrutura do Markdown (linhas de tabela, cabeçalhos) é contada pela ingestão,
que é dona da sintaxe (ingestion.markdown_structure).
"""
from __future__ import annotations

import re
import unicodedata
from collections import deque
from functools import lru_cache
from pathlib import Path

import tiktoken

# Muda quando a definição de alguma métrica muda: manifestos com versões diferentes não se comparam.
C0_VERSION = "c0-v1"
LONG_TOKEN_LETTERS = 18
LONG_WORDS_PATH = Path(__file__).with_name("c0_long_words.txt")

_encoding = tiktoken.get_encoding("cl100k_base")
_LETTERS = re.compile(r"[^\W\d_]+")

# Bloco de referências: do título "References"/"Referências" até um título de
# apêndice ou até as linhas deixarem de parecer referência. Nunca "tudo depois de
# References": em artigo do arXiv, os apêndices costumam vir depois.
_REF_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?(?:(?:\d{1,2}|[IVX]{1,4})\.?\s+)?"
    r"(?:references|referências|referencias|bibliography|bibliografia|works cited|literature cited)"
    r"(?:\*\*|__)?\s*:?$",
    re.IGNORECASE,
)
# Markdown degradado (md-v1): a conversão intercalou as colunas, e o título divide a linha com o texto da outra
# coluna ("REFERENCES TransactionsonReliability68,1(2018),45–66."). Só vale se nenhum título limpo valer. Em
# caixa alta, com qualquer continuação; em caixa normal, só seguido de maiúscula, número ou colchete
# ("References to prior work" é frase, não título).
_REF_HEADING_INTERLEAVED = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?"
    r"(?:(?:REFERENCES|REFERÊNCIAS)(?:\*\*|__)?\s+\S|(?:References|Referências)(?:\*\*|__)?\s+(?=[A-ZÀ-Ý\[\d]))"
)
_APPENDIX_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?(?:[A-Z](?:\.\d+)*\.?\s+)?"
    r"(?:appendix|appendices|apêndice|apêndices|anexo|anexos|supplementary|supplemental)\b",
    re.IGNORECASE,
)
# Título de apêndice sem a palavra, como sai do PDF: "A Experimental Setup", "B.2 Prompts", ou a letra numa
# linha e o título na seguinte. Sem ponto final, que é o que separa de uma linha de título de referência
# ("A Survey on LLMs.").
_LETTER_HEADING = re.compile(r"^[A-H](?:\.\d+)*\.?\s+[A-Z][\w-]*(?:\s+[\w-]+){0,7}$")
_LETTER_ALONE = re.compile(r"^[A-H](?:\.\d+)*\.?$")
_TITLE = re.compile(r"^[A-Z][\w-]*(?:\s+[\w-]+){0,9}$")
_MD_HEADING = re.compile(r"^#{1,6}\s")
_REF_NUMBERING = re.compile(r"^\s*(?:[-*]\s*)?(?:\[\d{1,3}\]|\d{1,3}\.\s|\(\d{1,3}\))")
_REF_MARKER = re.compile(
    r"(?<!\d)(?:19|20)\d{2}[a-z]?(?!\d)"
    r"|\bet\s?al\b|doi|arxiv|https?://|\bpp\.|\bvol\.|proceedings|conference|journal|transactions"
    r"|preprint|ieee|\bacm\b|springer|elsevier|workshop|symposium|university press",
    re.IGNORECASE,
)
# Linha só de autores, que é como a referência quebra em duas colunas: "Ziwei Ji, Nayeon Lee, Rita Frieske,"
# ou "J. Altenschmidt, S. Altman," (a lista de autores do relatório do GPT-4 ocupa dezenas de linhas sem ano),
# e a versão grudada do Markdown degradado ("Lachaux,TimothéeLacroix,BaptisteRozière,").
_NAME = r"[A-ZÀ-Ý][\w'’-]+"
_AUTHORS = re.compile(
    rf"(?:\b{_NAME}(?:\s+[A-Z]\.)*\s+{_NAME},\s*){{2,}}"
    rf"|(?:\b(?:[A-Z]\.\s?){{1,3}}{_NAME},\s*){{2,}}"
    rf"|\b{_NAME},\s(?:[A-Z]\.\s?){{1,3}}(?:,|and\b|&)"
    r"|(?:[A-ZÀ-Ý][a-zà-ÿ]+[A-ZÀ-Ý][a-zà-ÿ]+,){2,}"
)
# O título só conta se as primeiras linhas depois dele parecem referência (sumário e menção no texto não).
REF_CHECK_LINES = 20
REF_MIN_MARKERS = 3
# Fim por densidade: numa janela de 25 linhas não vazias, menos de 15% com marca de referência.
REF_WINDOW = 25
REF_MIN_DENSITY = 0.15
# Título de letra no PDF só fecha o bloco se as linhas seguintes não têm cara de referência. Aqui contam só
# as marcas fortes (ano, veículo, doi, autores): lista numerada também aparece em apêndice ("1. Extract...").
APPENDIX_MAX_STRONG_DENSITY = 0.3


@lru_cache(maxsize=1)
def long_words() -> frozenset[str]:
    lines = LONG_WORDS_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text)) if text else 0


def glued_word_rate(text: str) -> float | None:
    """Fração das letras em tokens de 18+ letras fora do dicionário; None se o texto não tem letras."""
    tokens = _LETTERS.findall(unicodedata.normalize("NFKC", text))
    letters = sum(len(token) for token in tokens)
    if not letters:
        return None
    dictionary = long_words()
    glued = sum(len(t) for t in tokens if len(t) >= LONG_TOKEN_LETTERS and t.lower() not in dictionary)
    return glued / letters


def _is_strong_marker(line: str) -> bool:
    return bool(_REF_MARKER.search(line) or _AUTHORS.search(line))


def _is_marker(line: str) -> bool:
    return bool(_REF_NUMBERING.match(line)) or _is_strong_marker(line)


def _strong_density(lines: list[str]) -> float:
    return sum(map(_is_strong_marker, lines)) / len(lines) if lines else 0.0


def _appendix_starts(lines: list[str], body: list[int], position: int, markdown: bool) -> bool:
    """A linha body[position] abre um apêndice (e fecha o bloco de referências)?"""
    line = lines[body[position]].strip()
    if _APPENDIX_HEADING.match(line) or (markdown and _MD_HEADING.match(line)):
        return True
    if markdown:
        return False
    following = [lines[j].strip() for j in body[position + 1 : position + 1 + REF_CHECK_LINES]]
    if _LETTER_HEADING.match(line):
        return _strong_density(following) < APPENDIX_MAX_STRONG_DENSITY
    if _LETTER_ALONE.match(line) and following and _TITLE.match(following[0]):
        return _strong_density(following[1:]) < APPENDIX_MAX_STRONG_DENSITY
    return False


def _block_end(lines: list[str], start: int, body: list[int], markdown: bool) -> int:
    window: deque[bool] = deque(maxlen=REF_WINDOW)
    last_marker = start
    for position, i in enumerate(body):
        if _appendix_starts(lines, body, position, markdown):
            return i
        marked = _is_marker(lines[i])
        if marked:
            last_marker = i
        window.append(marked)
        if len(window) == REF_WINDOW and sum(window) / REF_WINDOW < REF_MIN_DENSITY:
            return last_marker + 1
    return len(lines)


def references_block(lines: list[str], markdown: bool = False) -> tuple[int, int] | None:
    """[início, fim) do bloco de referências em `lines`, ou None se não há.

    Começa no último título de referências seguido de linhas com cara de
    referência (ano, "et al.", "Proceedings", doi, lista de autores, numeração
    [12]...). Termina no que vier primeiro: título de apêndice ("Appendix",
    "Supplementary"...); no Markdown, qualquer outro título; no PDF, um título
    de letra ("A Experimental Setup") seguido de linhas sem cara de referência;
    ou a densidade de linhas com marca abaixo de REF_MIN_DENSITY numa janela de
    REF_WINDOW linhas.
    """
    for pattern in (_REF_HEADING, _REF_HEADING_INTERLEAVED):
        heads = [i for i, line in enumerate(lines) if pattern.match(line.strip())]
        for start in reversed(heads):
            body = [i for i in range(start + 1, len(lines)) if lines[i].strip()]
            if sum(_is_marker(lines[i]) for i in body[:REF_CHECK_LINES]) >= REF_MIN_MARKERS:
                return start, _block_end(lines, start, body, markdown)
    return None


def _percentile(values: list[int], q: float) -> float:
    """Percentil por interpolação linear (o padrão do numpy), com os valores já ordenados."""
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values: list[int]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p10": round(_percentile(ordered, 0.10), 1),
        "p90": round(_percentile(ordered, 0.90), 1),
    }


def document_quality(
    *,
    fmt: str,
    text: str,
    chunk_texts: list[str],
    pages_total: int | None = None,
    pages_with_text: int | None = None,
    table_rows: int | None = None,
    headings: int = 0,
) -> dict:
    """As métricas do C0 de um documento (o dicionário vai para o manifesto, ao lado de file e sha256)."""
    lines = text.splitlines()
    block = references_block(lines, markdown=fmt == "md")
    tokens = count_tokens(text)
    reference_tokens = count_tokens("\n".join(lines[block[0] : block[1]])) if block else 0
    rate = glued_word_rate(text)
    return {
        "pages_total": pages_total,
        "pages_with_text": pages_with_text,
        "tokens": tokens,
        "chunk_tokens": distribution([count_tokens(chunk) for chunk in chunk_texts]),
        "glued_word_rate": round(rate, 4) if rate is not None else None,
        "table_rows": table_rows,
        "headings": headings,
        "references_found": block is not None,
        "references_share": round(reference_tokens / tokens, 4) if tokens else None,
    }


def summarize(documents: list[dict], chunk_token_counts: list[int]) -> dict:
    """O C0 da versão inteira, para o topo do manifesto."""
    tokens = sum(d.get("tokens") or 0 for d in documents)
    rates = [d["glued_word_rate"] for d in documents if d.get("glued_word_rate") is not None]
    pdfs = [d for d in documents if d.get("pages_total") is not None]
    return {
        "version": C0_VERSION,
        "documents": len(documents),
        "tokens": tokens,
        "chunk_tokens": distribution(chunk_token_counts),
        "pages_total": sum(d["pages_total"] for d in pdfs) if pdfs else None,
        "pages_without_text": sum(d["pages_total"] - d["pages_with_text"] for d in pdfs) if pdfs else None,
        "glued_word_rate_mean": round(sum(rates) / len(rates), 4) if rates else None,
        "documents_glued_over_5pct": sum(rate > 0.05 for rate in rates),
        "references_share": round(
            sum((d.get("references_share") or 0) * (d.get("tokens") or 0) for d in documents) / tokens, 4
        ) if tokens else None,
        "documents_without_references": sum(not d.get("references_found") for d in documents),
        "table_rows": sum(d["table_rows"] for d in documents if d.get("table_rows") is not None),
        "headings": sum(d.get("headings") or 0 for d in documents),
    }
