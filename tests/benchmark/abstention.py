"""A resposta é uma recusa? (plano-benchmark.md, §6, C2: abstenção correta e indevida)

Hoje a recusa é a frase NOT_FOUND_MARKER (backend/app/rag/generation.py) procurada sem diferenciar
maiúsculas, e uma recusa escrita de outro jeito ("Nao encontrei essa informacao...", sem acento) passa como
resposta. Aqui a comparação ignora acento, caixa, pontuação e espaço, e aceita uma lista pequena de
variantes, que o plano manda revisar à mão nos casos do E0. Só biblioteca padrão: o relatório também usa.
"""
from __future__ import annotations

import re
import unicodedata

# Variantes revisadas à mão: entra aqui só frase de recusa vista nas respostas de verdade. Revisão de 18/10/2026,
# nas 103 respostas do A1 do E0 (gemma3:4b): duas recusas passavam como resposta, "Com base nos documentos
# fornecidos, não encontrei essa informação." (a frase sem o fim) e "O trecho fornecido não contém informações
# sobre ...". As respostas parciais ("o documento não especifica o número exato", depois de responder o resto)
# continuam respostas.
VARIANTS: tuple[str, ...] = (
    "não encontrei essa informação",
    "o trecho fornecido não contém informações",
)
_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize(text: str) -> str:
    """NFKC, sem acento, minúsculas, e tudo que não é letra ou dígito vira um espaço."""
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
    return _NON_WORD.sub(" ", plain).strip()


def is_abstention(answer: str | None, marker: str, variants: tuple[str, ...] = VARIANTS) -> bool:
    if not answer or not marker:
        return False
    text = f" {normalize(answer)} "
    return any(f" {normalize(phrase)} " in text for phrase in (marker, *variants) if normalize(phrase))
