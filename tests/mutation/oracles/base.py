"""Contrato comum dos cinco oráculos e utilidades de normalização de texto.

Um oráculo recebe uma execução (ou a resposta modal de várias) e devolve um
`Verdict`. Três campos merecem atenção:

- `evaluable`: False quando o oráculo não consegue emitir juízo (juiz fora do ar,
  métrica que não pôde ser calculada). Diferente de `fail` — um oráculo que
  reprova por não ter conseguido medir inflaria a taxa de mortes.
- `score`: contínuo quando existe (cosseno, média RAGAS); None para oráculos
  puramente booleanos. É o que permite reanalisar com outro limiar sem reexecutar.
- `detail`: texto curto e legível com o motivo. Alimenta a classificação de
  coerência (§7.4), que sem isso viraria adivinhação.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Verdict:
    oracle: str
    verdict: str                      # "pass" | "fail"
    score: float | None = None
    evaluable: bool = True
    detail: str = ""
    components: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.evaluable and self.verdict == "fail"

    def to_record(self) -> dict[str, Any]:
        return {
            "oracle": self.oracle,
            "verdict": self.verdict,
            "score": self.score,
            "evaluable": self.evaluable,
            "detail": self.detail[:500],
            "components": self.components,
        }


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str, ignore_case: bool = True, ignore_accents: bool = True) -> str:
    """Normalização usada em toda comparação textual determinística.

    Caixa e acento são ruído do modelo local, não o defeito sob estudo: o
    assistente escrever "metrica" em vez de "métrica" não é uma falha que os
    operadores injetam, e deixar isso reprovar casos contaminaria toda a
    campanha com falsos positivos.
    """
    result = text
    if ignore_accents:
        result = strip_accents(result)
    if ignore_case:
        result = result.lower()
    # Separador decimal: as referências estão em português ("3,79%") e o modelo
    # copia a tabela em inglês ("3.79%"). Descoberto no baseline v2 (c03, c07):
    # a resposta certa reprovava por uma vírgula. Só entre dígitos, para não
    # tocar em pontuação de frase.
    result = re.sub(r"(?<=\d),(?=\d)", ".", result)
    return re.sub(r"\s+", " ", result).strip()


NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in NUMBER_RE.finditer(text):
        try:
            numbers.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue
    return numbers


def modal_answer(answers: list[str]) -> str:
    """Resposta modal de N execuções: a mais frequente após normalização.

    Empate resolve pela primeira ocorrência, o que mantém a escolha
    determinística dada a ordem das repetições (que é fixa: seed = índice da
    repetição). Devolve a forma ORIGINAL da resposta escolhida, não a
    normalizada — o juiz e o RAGAS precisam do texto como o usuário veria.
    """
    if not answers:
        return ""
    counts: dict[str, int] = {}
    first_original: dict[str, str] = {}
    for answer in answers:
        key = normalize(answer)
        counts[key] = counts.get(key, 0) + 1
        first_original.setdefault(key, answer)
    best_key = max(counts, key=lambda key: (counts[key], -list(first_original).index(key)))
    return first_original[best_key]
