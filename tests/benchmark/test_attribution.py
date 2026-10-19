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


def test_os_formatos_que_o_gerador_usa_de_verdade():
    # Vistos nas respostas do E3: o cabeçalho copiado, o número do trecho e o exemplo da própria regra.
    text = ("Foram 16 [Trecho 1 - RAGAS.pdf (pág. 3)]. O juiz [ARES.pdf (pág. 7)] e de novo [Trecho 2, pág. 7]; "
            "o exemplo [relatorio_x.pdf, pág. 4], um trecho que não existe [Trecho 9] e uma referência [12].")

    cited = attribution.citations(text, RECORD["sources"])

    assert cited == [("ragas", 3), ("ares", 7), ("ares", 7), ("relatorio_x", 4), ("trecho 9", None)]
    assert attribution.classify(answer("Foram 16 [Trecho 1 - RAGAS.pdf (pág. 3)]."), False) == "fonte"
    assert attribution.classify(answer("O juiz [Trecho 2, pág. 7]."), False) == "outro_documento"
    # Copiar o exemplo da regra é citar um arquivo que não veio.
    assert attribution.classify(answer("Isso [relatorio_x.pdf, pág. 4]."), False) == "fabricada"


def test_nome_sem_extensao_e_titulo_viram_o_documento_recebido(monkeypatch):
    monkeypatch.setattr(attribution, "_titles", lambda: {"ragas": "RAGAS: Automated Evaluation", "ares": "ARES: A Framework",
                                                         "metarag": "MetaRAG: Metamorphic Testing"})
    record = answer("A fidelidade [RAGAS, pág. 3], o juiz [ARES: A Framework (pág. 7)] e o outro [MetaRAG, pág. 2].")

    cited = attribution.citations(record["actual_output"], record["sources"])

    # O MetaRAG existe no acervo e não veio: é fabricação; um nome que não designa nada não é citação.
    assert cited == [("ragas", 3), ("ares", 7), ("metarag", 2)]
    assert attribution.classify(record, False) == "fabricada"
    assert attribution.citations("Veja [Seção de resultados, pág. 2].", RECORD["sources"]) == []
    # O cabeçalho copiado sem o parêntese final ainda é citação.
    assert attribution.citations("Isso [Trecho 1 - RAGAS.pdf (pág. 3].", RECORD["sources"]) == [("ragas", 3)]
