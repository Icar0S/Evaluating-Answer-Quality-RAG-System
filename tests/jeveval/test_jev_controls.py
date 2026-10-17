"""Controles do instrumento: casos com resposta certa conhecida, julgados pelo Jev de verdade.

Validam o INSTRUMENTO (as perguntas de _jev_shared + o Jev) antes de qualquer
nota ser lida, como tests/ragas/test_judge_controls.py faz com o juiz local. Se
um controle reprova, o problema é o instrumento, não o RAG, e as notas do
run_and_export.py não medem qualidade.

Cada controle amarra um problema que o projeto já documentou:

1. alucinação sobre tópico ausente (docs/estudo-mutacao-status.md, §4.9): a
   FaithfulnessMetric com juiz LLM deu 1,0 — aqui a fidelidade tem de sair baixa;
2. recusa ("não encontrei..."): a FaithfulnessMetric dá 0,0 — aqui tem de sair N/A;
3. o O1 não separa paráfrase de erro (AUC 0,48 nos pares rotulados): a correção
   tem de aceitar a paráfrase e rejeitar o número trocado do MESMO par. A medida
   completa, nos 247 pares, é o criterion_validity.py.

Mais a estabilidade: o mesmo caso, julgado várias vezes, dá a mesma nota.

Chamam a API da TypeSafe (~25 chamadas, centavos). Sem TYPESAFE_API_KEY, se
auto-pulam (fixture `jev` do conftest.py). Rodar da raiz:
    tests\\jeveval\\.venv\\Scripts\\python.exe -m pytest tests/jeveval -m live
"""
from __future__ import annotations

import pytest

from _jev_mutation import load_golden, load_pairs
from _jev_shared import METRIC_SPECS, CaseInputs, Outcome, load_goldens, measure, read_not_found_marker, threshold_for

pytestmark = pytest.mark.live

# Os mesmos textos dos controles do RAGAS (tests/ragas/test_judge_controls.py),
# sobre o golden `ragas-definicao`: a comparação entre os instrumentos fica direta.
GOLDEN = load_goldens()["ragas-definicao"]
RESPOSTA_APOIADA = (
    "O RAGAS é um framework para avaliação sem referência de pipelines RAG, com "
    "métricas que dispensam anotações humanas de verdade fundamental."
)
# A alucinação real do SUT no estudo de mutação (caso c16): fato inventado sobre
# um tópico que o contexto nem menciona.
ALUCINACAO = "A WCAG 2.2 define 14 critérios de sucesso no nível AA."
FORA_DO_ASSUNTO = "A capital da França é Paris, às margens do rio Sena."
TRECHO_DE_OUTRO_ASSUNTO = "A WCAG 2.2 organiza seus critérios de sucesso em três níveis de conformidade: A, AA e AAA."
RECUSA = f"{read_not_found_marker()}."

# Um par rotulado do estudo de mutação: paráfrase (EQUIV) e número trocado (ERRO)
# da mesma referência. O cosseno do O1 dá praticamente a mesma nota aos dois.
PARES = {p.pair_id: p for p in load_pairs()}
PARAFRASE, NUMERO_TROCADO = PARES["c01-eq1"], PARES["c01-err1"]
# Caso fora do corpus da suíte de mutação: a referência é a abstenção.
FORA_DO_CORPUS = load_golden()["c16"]

# Abaixo disto conta como "o Jev percebeu o defeito" (o mesmo 0,3 do RAGAS).
NOTA_BAIXA = 0.3
REPETICOES = 3
# O Jev devolve probabilidades, não amostra texto: o mesmo estado deveria dar a
# mesma nota. A folga cobre arredondamento do lado do servidor, não variância.
TOLERANCIA_ESTABILIDADE = 0.02


def _medir(jev, settings, metrica: str, actual_output: str, **campos) -> Outcome:
    caso = CaseInputs(
        input=campos.get("input", GOLDEN["input"]),
        actual_output=actual_output,
        expected_output=campos.get("expected_output", GOLDEN["expected_output"]),
        retrieval_context=campos.get("retrieval_context", GOLDEN["context"]),
    )
    return measure(METRIC_SPECS[metrica], jev, caso, threshold_for(metrica, settings))


def _nota(outcome: Outcome) -> float:
    if outcome.error:
        pytest.fail(f"{outcome.name} não foi medida: {outcome.error}")
    if not outcome.applicable:
        pytest.fail(f"{outcome.name} saiu N/A num caso em que se aplica:\n{outcome.reason}")
    return outcome.score


def test_fidelidade_alta_para_resposta_apoiada_no_contexto(jev, settings) -> None:
    nota = _nota(_medir(jev, settings, "faithfulness", RESPOSTA_APOIADA))

    assert nota >= settings.jeveval_threshold, f"faithfulness={nota:.2f} para resposta que só repete o trecho"


def test_fidelidade_baixa_para_alucinacao_sobre_topico_ausente(jev, settings) -> None:
    # Problema 1: inventar sobre um tópico ausente não contradiz o contexto, e a
    # FaithfulnessMetric do deepeval com juiz LLM deu 1,0 a esta mesma frase.
    outcome = _medir(jev, settings, "faithfulness", ALUCINACAO)

    assert _nota(outcome) <= NOTA_BAIXA, f"faithfulness={outcome.score:.2f} para tópico ausente\n{outcome.reason}"


@pytest.mark.parametrize("metrica", ["faithfulness", "answer_relevancy"])
def test_recusa_sai_na_e_nao_zero(jev, settings, metrica: str) -> None:
    # Problema 2: recusa não tem afirmação para ser fiel ou relevante. Se a
    # abstenção foi certa ou errada, quem diz é a correção (teste abaixo).
    outcome = _medir(jev, settings, metrica, RECUSA)

    assert outcome.error is None, f"{metrica} não foi medida: {outcome.error}"
    assert not outcome.applicable, f"{metrica}={outcome.score} para uma recusa; o certo é N/A\n{outcome.reason}"


@pytest.mark.parametrize(
    "resposta_relevante",
    [RESPOSTA_APOIADA, GOLDEN["expected_output"]],
    ids=["resposta-curta", "resposta-de-referencia"],
)
def test_relevancia_separa_resposta_relevante_de_fora_do_assunto(jev, settings, resposta_relevante: str) -> None:
    relevante = _nota(_medir(jev, settings, "answer_relevancy", resposta_relevante))
    fora = _nota(_medir(jev, settings, "answer_relevancy", FORA_DO_ASSUNTO))

    # Sem a margem relativa do RAGAS: lá a nota é cosseno de embeddings, com piso
    # perto de 0,55; aqui é a resposta do Jev a perguntas fechadas, sem piso.
    assert relevante >= settings.jeveval_threshold, f"answer_relevancy={relevante:.2f} para resposta direta"
    assert fora <= NOTA_BAIXA, f"answer_relevancy={fora:.2f} para resposta fora do assunto"


def test_relevancia_do_contexto_baixa_para_trecho_de_outro_assunto(jev, settings) -> None:
    nota = _nota(_medir(jev, settings, "context_relevance", RESPOSTA_APOIADA, retrieval_context=[TRECHO_DE_OUTRO_ASSUNTO]))

    assert nota <= NOTA_BAIXA, f"context_relevance={nota:.2f} para trecho sobre WCAG numa pergunta sobre RAGAS"


def test_recall_do_contexto_separa_trecho_certo_de_trecho_errado(jev, settings) -> None:
    # Referência = só o que o trecho afirma (a do golden diz mais que o trecho
    # abreviado; ver o mesmo cuidado nos controles do RAGAS).
    certo = _nota(_medir(jev, settings, "context_recall", RESPOSTA_APOIADA, expected_output=RESPOSTA_APOIADA))
    errado = _nota(
        _medir(
            jev, settings, "context_recall", RESPOSTA_APOIADA,
            expected_output=RESPOSTA_APOIADA, retrieval_context=[TRECHO_DE_OUTRO_ASSUNTO],
        )
    )

    assert certo >= settings.jeveval_threshold, f"context_recall={certo:.2f} com o trecho que contém a referência"
    assert errado <= NOTA_BAIXA, f"context_recall={errado:.2f} com trecho de outro assunto"


def test_correcao_aceita_parafrase_e_rejeita_numero_trocado(jev, settings) -> None:
    # Problema 3: paráfrase e número trocado da MESMA referência.
    def correcao(par) -> Outcome:
        return _medir(jev, settings, "correctness", par.candidate, input=par.question, expected_output=par.reference)

    parafrase, trocado = correcao(PARAFRASE), correcao(NUMERO_TROCADO)

    assert _nota(parafrase) >= settings.jeveval_correctness_threshold, f"paráfrase reprovada\n{parafrase.reason}"
    assert _nota(trocado) < settings.jeveval_correctness_threshold, f"número trocado aprovado\n{trocado.reason}"


def test_correcao_reprova_recusa_quando_a_referencia_responde(jev, settings) -> None:
    outcome = _medir(jev, settings, "correctness", RECUSA)

    assert _nota(outcome) < settings.jeveval_correctness_threshold, f"recusa aprovada\n{outcome.reason}"


def test_correcao_fora_do_corpus_aprova_recusa_e_reprova_alucinacao(jev, settings) -> None:
    def correcao(resposta: str) -> Outcome:
        return _medir(
            jev, settings, "correctness", resposta,
            input=FORA_DO_CORPUS["question"], expected_output=FORA_DO_CORPUS["expected_output"],
        )

    recusa, alucinacao = correcao(RECUSA), correcao(ALUCINACAO)

    assert _nota(recusa) >= settings.jeveval_correctness_threshold, f"abstenção certa reprovada\n{recusa.reason}"
    assert _nota(alucinacao) < settings.jeveval_correctness_threshold, (
        f"resposta inventada aprovada quando a referência se abstém (causa={alucinacao.cause})\n{alucinacao.reason}"
    )


@pytest.mark.parametrize(
    ("metrica", "resposta", "campos"),
    [
        ("faithfulness", RESPOSTA_APOIADA, {}),
        ("correctness", PARAFRASE.candidate, {"input": PARAFRASE.question, "expected_output": PARAFRASE.reference}),
    ],
    ids=["fidelidade", "correcao"],
)
def test_mesma_nota_em_repeticoes(jev, settings, metrica: str, resposta: str, campos: dict) -> None:
    notas = [_nota(_medir(jev, settings, metrica, resposta, **campos)) for _ in range(REPETICOES)]

    assert max(notas) - min(notas) <= TOLERANCIA_ESTABILIDADE, f"{metrica}: notas {notas} em {REPETICOES} repetições"
