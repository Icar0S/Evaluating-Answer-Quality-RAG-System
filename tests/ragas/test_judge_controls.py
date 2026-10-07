"""Controles do juiz: casos com resposta certa conhecida, sem passar pelo SUT.

Validam o INSTRUMENTO antes de confiar nas notas de test_rag_quality.py. Se um
destes reprova, o problema é o par juiz + métrica, não o RAG — e as notas da
outra suíte não devem ser lidas como medida de qualidade. Foi assim que o
gemma3:4b caiu como juiz: marca toda resposta como evasiva, e answer_relevancy
dá 0 até para resposta correta (ver README.md desta pasta).

Os textos usam o trecho do artigo do RAGAS que está no golden
`ragas-definicao` (tests/deepeval/goldens/dataset.json).
"""
from __future__ import annotations

import pytest

from _ragas_shared import CaseInputs, load_goldens, score_metric

GOLDEN = {g.name: g for g in load_goldens()}["ragas-definicao"]

# Só o que o trecho afirma, sem nada além. (A resposta de referência do golden
# não serve aqui: ela afirma mais do que o trecho abreviado sustenta, e a
# faithfulness do ragas, que exige afirmação *inferível* do contexto, dá 0,4 a
# ela — corretamente.)
RESPOSTA_APOIADA = (
    "O RAGAS é um framework para avaliação sem referência de pipelines RAG, com "
    "métricas que dispensam anotações humanas de verdade fundamental."
)
# A alucinação real do SUT no estudo de mutação (caso c16): fato inventado
# sobre um tópico que o contexto nem menciona.
ALUCINACAO = "A WCAG 2.2 define 14 critérios de sucesso no nível AA."
FORA_DO_ASSUNTO = "A capital da França é Paris, às margens do rio Sena."
TRECHO_DE_OUTRO_ASSUNTO = "A WCAG 2.2 organiza seus critérios de sucesso em três níveis de conformidade: A, AA e AAA."

# Abaixo disto conta como "o juiz percebeu o defeito".
NOTA_BAIXA = 0.3
# Distância mínima entre resposta relevante e fora do assunto em
# answer_relevancy. A nota é cosseno de embeddings e tem piso alto: com o
# nomic-embed-text, duas perguntas em português sem relação nenhuma ficam
# perto de 0,55 (medido). Por isso o controle é relativo, não absoluto.
MARGEM_RELEVANCIA = 0.15


def _medir(metrics, nome: str, resposta: str, contextos: list[str] | None = None) -> float:
    caso = CaseInputs(
        user_input=GOLDEN.input,
        response=resposta,
        retrieved_contexts=GOLDEN.context if contextos is None else contextos,
        reference=GOLDEN.expected_output,
    )
    resultado = score_metric(nome, metrics[nome], caso)
    if not resultado.measured:
        pytest.fail(f"{nome} não foi medida: {resultado.error}")
    return resultado.score


def test_faithfulness_alta_para_resposta_apoiada_no_contexto(metrics, ragas_settings) -> None:
    nota = _medir(metrics, "faithfulness", RESPOSTA_APOIADA)

    assert nota >= ragas_settings.ragas_threshold, f"faithfulness={nota:.2f} para resposta que só repete o trecho"


def test_faithfulness_baixa_para_alucinacao_sobre_topico_ausente(metrics) -> None:
    # O ponto cego do DeepEval registrado no estudo de mutação (§4.9): lá a
    # FaithfulnessMetric conta afirmações que *não contradizem* o contexto, e
    # inventar sobre um tópico ausente não contradiz nada — nota 1,0. A
    # definição do ragas exige que a afirmação seja *inferível*.
    nota = _medir(metrics, "faithfulness", ALUCINACAO)

    assert nota <= NOTA_BAIXA, f"faithfulness={nota:.2f} para afirmação sobre tópico que o contexto não menciona"


@pytest.mark.parametrize(
    "resposta_relevante",
    # As duas, e não só a curta: o gemma3:4b dá nota normal à curta mas marca a
    # resposta de referência do golden como evasiva ("noncommittal": 1), o que
    # zera a métrica. Com uma só, o controle aprovava esse juiz.
    [RESPOSTA_APOIADA, GOLDEN.expected_output],
    ids=["resposta-curta", "resposta-de-referencia"],
)
def test_answer_relevancy_separa_resposta_relevante_de_fora_do_assunto(
    metrics, ragas_settings, resposta_relevante: str
) -> None:
    relevante = _medir(metrics, "answer_relevancy", resposta_relevante)
    fora = _medir(metrics, "answer_relevancy", FORA_DO_ASSUNTO)

    assert relevante >= ragas_settings.ragas_threshold, f"answer_relevancy={relevante:.2f} para resposta direta à pergunta"
    assert relevante - fora >= MARGEM_RELEVANCIA, (
        f"relevante={relevante:.2f} vs fora do assunto={fora:.2f}: o juiz não separa os dois"
    )


def test_context_relevance_baixa_para_trecho_de_outro_assunto(metrics) -> None:
    nota = _medir(metrics, "context_relevance", RESPOSTA_APOIADA, contextos=[TRECHO_DE_OUTRO_ASSUNTO])

    assert nota <= NOTA_BAIXA, f"context_relevance={nota:.2f} para trecho sobre WCAG numa pergunta sobre RAGAS"
