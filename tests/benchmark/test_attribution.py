"""Atribuição errada (tests/benchmark/attribution.py): as categorias da §6 com respostas feitas à mão."""
from __future__ import annotations

from tests.benchmark import attribution

RECORD = {"source_document": "RAGAS.pdf", "sources": ["RAGAS.pdf (pág. 3)", "ARES.pdf (pág. 7)"]}


def answer(text: str) -> dict:
    return {**RECORD, "actual_output": text}


def test_citacoes_no_formato_da_regra():
    text = "Fidelidade [RAGAS.pdf, pág. 3] e o juiz [ARES.pdf, página 7]; sem página [tabela.md]."
    assert attribution.citations(text) == [("ragas", 3), ("ares", 7), ("tabela", None)]
    assert attribution.received(RECORD) == {("ragas", 3), ("ares", 7)}


def test_categorias_da_resposta_de_dentro():
    assert attribution.classify(answer("A métrica é a fidelidade [RAGAS.pdf, pág. 3]."), False) == "fonte"
    assert attribution.classify(answer("O ARES usa juízes [ARES.pdf, pág. 7]."), False) == "outro_documento"
    # Página que não veio e arquivo que não veio são fabricação.
    assert attribution.classify(answer("Diz isso [RAGAS.pdf, pág. 9]."), False) == "fabricada"
    assert attribution.classify(answer("Diz isso [MetaRAG.pdf, pág. 2]."), False) == "fabricada"
    assert attribution.classify(answer("A métrica é a fidelidade."), False) == "sem_citacao"
    assert attribution.classify(answer("Não encontrei essa informação."), True) == "recusa"


def test_atribuicao_errada_de_dentro_e_de_fora():
    assert attribution.misattributed(answer("Diz isso [MetaRAG.pdf, pág. 2]."), False, inside=True)
    assert not attribution.misattributed(answer("Fidelidade [RAGAS.pdf, pág. 3]."), False, inside=True)
    # Pergunta de fora: citar qualquer trecho numa resposta não abstida é atribuição errada.
    outside = {"source_document": None, "sources": ["RAGAS.pdf (pág. 3)"]}
    assert attribution.misattributed({**outside, "actual_output": "Sete pontos [RAGAS.pdf, pág. 3]."}, False, False)
    assert not attribution.misattributed({**outside, "actual_output": "Não encontrei."}, True, False)
