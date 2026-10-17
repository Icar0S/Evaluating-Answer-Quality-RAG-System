"""Testes herméticos do instrumento: perguntas, pesos, agregação e N/A, com o Jev falso.

Sem rede e sem chave: rodam em todo PR no CI. O Jev falso (_fake_jev.py)
substitui só a chamada à API; valor de cada pergunta, média ponderada, N/A e
reason são o código real do deepeval 4.3.1. Rodar da raiz:
    tests\\jeveval\\.venv\\Scripts\\python.exe -m pytest tests/jeveval -m "not live"
"""
from __future__ import annotations

import ast
from importlib import resources

import pytest
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.metrics.utils.system_one import parse_questions

import _jev_shared as shared
from _fake_jev import FakeJev, best, not_applicable_else, worst
from _jev_shared import (
    CORRECT,
    CORRECTNESS,
    CORRECTNESS_CAUSES,
    FAITHFULNESS,
    METRIC_SPECS,
    NOT_APPLICABLE,
    UNMENTIONED_IS_UNSUPPORTED,
    CaseInputs,
    measure,
)

CASO = CaseInputs(
    input="O que é o RAGAS?",
    actual_output="O RAGAS é um framework de avaliação de pipelines RAG.",
    expected_output="RAGAS é um framework para avaliação sem referência de pipelines RAG.",
    retrieval_context=["We introduce RAGAS, a framework for reference-free evaluation of RAG pipelines.", "t2"],
)
LIMIAR = 0.7

# Onde o deepeval guarda as perguntas que manda ao Jev no modo system_one.
DEEPEVAL_TEMPLATE = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "contextual_precision",
    "context_recall": "contextual_recall",
    "context_relevance": "contextual_relevancy",
}


def _deepeval_questions(key: str) -> list:
    path = resources.files("deepeval.metrics") / DEEPEVAL_TEMPLATE[key] / "templates" / "_experimental_system_one_questions.txt"
    return parse_questions(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------- as perguntas


@pytest.mark.parametrize("key", ["context_precision", "context_recall", "context_relevance"])
def test_perguntas_de_contexto_iguais_as_do_deepeval(key: str) -> None:
    # Cópia literal congelada no código. Se uma atualização do deepeval mudar as
    # perguntas do system_one, este teste falha: o instrumento não muda sozinho,
    # e quem atualiza decide se acompanha (e registra a mudança no README).
    ours = [q.model_dump() for q in METRIC_SPECS[key].questions]
    theirs = [q.model_dump() for q in _deepeval_questions(key)]

    assert ours == theirs


def test_fidelidade_tirou_a_pergunta_de_nao_contradicao() -> None:
    # A pergunta do deepeval que dá crédito a "não contradiz" é o ponto cego do
    # §4.9 (alucinação sobre tópico ausente não contradiz nada). Ela existe lá...
    original = [q.text for q in _deepeval_questions("faithfulness")]
    assert any("contradicts" in text for text in original)
    # ...e não existe aqui; a regra de inferibilidade entrou nas que ficaram.
    assert not any("contradicts the facts" in q.text for q in FAITHFULNESS.questions)
    assert all(q.text.endswith(UNMENTIONED_IS_UNSUPPORTED) for q in FAITHFULNESS.questions)


@pytest.mark.parametrize("key", ["faithfulness", "answer_relevancy"])
def test_perguntas_sobre_a_resposta_tem_opcao_na(key: str) -> None:
    for question in METRIC_SPECS[key].questions:
        assert isinstance(question, Choice)
        assert question.options[NOT_APPLICABLE] is None


@pytest.mark.parametrize("key", ["context_precision", "context_recall", "context_relevance", "correctness"])
def test_metricas_de_contexto_e_correcao_nunca_sao_na(key: str) -> None:
    # Contexto julga a recuperação, que existe mesmo quando o assistente se
    # abstém; a correção é justamente quem julga se a abstenção foi certa.
    for question in METRIC_SPECS[key].questions:
        assert not (isinstance(question, Choice) and question.not_applicable_options)


@pytest.mark.parametrize("key", ["faithfulness", "answer_relevancy"])
def test_adaptacoes_preservam_pesos_e_creditos_do_deepeval(key: str) -> None:
    # Noul -> Choice{Yes, No, N/A} e Score -> Choice{níveis, N/A}: mesmos pesos
    # e o mesmo valor por nível que o Score dá (índice / (níveis - 1)).
    original = _deepeval_questions(key)
    ours = METRIC_SPECS[key].questions
    kept = original if key == "answer_relevancy" else [q for q in original if "contradicts" not in q.text]

    assert [q.weight for q in ours] == [q.weight for q in kept]
    for theirs, mine in zip(kept, ours):
        if isinstance(theirs, Score):
            top = len(theirs.levels) - 1
            assert mine.applicable_options == pytest.approx({lvl: i / top for i, lvl in enumerate(theirs.levels)})
        else:
            assert isinstance(theirs, Noul)
            assert mine.applicable_options == {"No": 0.0, "Yes": 1.0}


def test_correcao_so_da_credito_a_resposta_certa_e_usa_as_causas_do_o4() -> None:
    (question,) = CORRECTNESS.questions
    assert question.applicable_options[CORRECT] == 1.0
    assert all(credit == 0.0 for option, credit in question.applicable_options.items() if option != CORRECT)

    # As causas são o vocabulário do O4, lido da fonte (o módulo importa httpx e o
    # backend, então não abre neste venv).
    source = (shared.PROJECT_ROOT / "tests" / "mutation" / "oracles" / "o4_judge.py").read_text(encoding="utf-8")
    cause_options = next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CAUSE_OPTIONS" for t in node.targets)
    )
    assert set(CORRECTNESS_CAUSES) == set(question.options)
    assert set(CORRECTNESS_CAUSES.values()) <= set(cause_options)


def test_impressao_digital_do_instrumento() -> None:
    assert shared.instrument_sha() == shared.instrument_sha()
    assert shared.instrument_sha(["faithfulness"]) != shared.instrument_sha(["answer_relevancy"])
    assert len(shared.instrument_sha()) == 12


# -------------------------------------------------------------- o estado


@pytest.mark.parametrize("key", list(METRIC_SPECS))
def test_estado_enviado_ao_jev_so_tem_os_campos_declarados(key: str) -> None:
    jev = FakeJev()
    measure(METRIC_SPECS[key], jev, CASO, LIMIAR)

    (state, questions), = jev.calls
    assert set(state["test_case"]) == {p.value for p in METRIC_SPECS[key].params}
    assert list(questions) == [f"q_{i}" for i in range(len(METRIC_SPECS[key].questions))]


def test_fidelidade_nao_ve_a_referencia() -> None:
    # A fidelidade é sobre o contexto recuperado. Se a referência entrasse no
    # estado, uma resposta certa mas sem apoio nos trechos poderia passar.
    jev = FakeJev()
    measure(FAITHFULNESS, jev, CASO, LIMIAR)

    assert "expected_output" not in jev.calls[0][0]["test_case"]


# ------------------------------------------------------------- a conta


def test_nota_e_a_media_ponderada_das_perguntas() -> None:
    jev = FakeJev({"q_0": {"Yes": 0.9, "No": 0.1}, "q_1": {"Fully grounded": 0.5, "Mostly grounded": 0.5}})

    outcome = measure(FAITHFULNESS, jev, CASO, LIMIAR)

    v0 = 0.9
    v1 = 0.5 * 1.0 + 0.5 * (2 / 3)
    assert outcome.score == pytest.approx((2 * v0 + 1 * v1) / 3)
    assert outcome.passed(LIMIAR) is True
    assert outcome.applicable and outcome.error is None
    assert [item["weight"] for item in outcome.breakdown] == [2.0, 1.0]


def test_nota_com_noul_e_score() -> None:
    jev = FakeJev({"q_0": 0.8, "q_1": 0.6, "q_2": {3: 0.5, 2: 0.5}})

    outcome = measure(METRIC_SPECS["context_precision"], jev, CASO, LIMIAR)

    assert outcome.score == pytest.approx((2 * 0.8 + 0.6 + 2.5 / 3) / 4)


@pytest.mark.parametrize(("rule", "expected"), [(best, 1.0), (worst, 0.0)])
@pytest.mark.parametrize("key", list(METRIC_SPECS))
def test_extremos_dao_zero_e_um(key: str, rule, expected: float) -> None:
    outcome = measure(METRIC_SPECS[key], FakeJev(rule), CASO, LIMIAR)

    assert outcome.score == pytest.approx(expected)


def test_correcao_e_a_probabilidade_da_resposta_certa_e_nomeia_a_causa() -> None:
    changed = "Changes or contradicts a value, entity or condition of `expected_output`"
    errada = measure(CORRECTNESS, FakeJev({"q_0": {CORRECT: 0.2, changed: 0.7}}), CASO, 0.5)
    certa = measure(CORRECTNESS, FakeJev({"q_0": (CORRECT, 0.9)}), CASO, 0.5)

    assert errada.score == pytest.approx(0.2)
    assert errada.cause == "wrong_value"
    assert errada.passed(0.5) is False
    assert certa.score == pytest.approx(0.9)
    assert certa.cause == "none"


# ------------------------------------------------------------------ o N/A


def test_jeveval_sozinho_da_um_quando_nada_se_aplica() -> None:
    # A armadilha que a camada existe para fechar: sem pergunta aplicável, o
    # JevEval devolve 1,0 e success=True (EMPTY_SCORE do deepeval).
    metric = shared.build_metric(FAITHFULNESS, FakeJev(not_applicable_else(worst)), LIMIAR)
    metric.measure(
        shared.LLMTestCase(input=CASO.input, actual_output="Não encontrei.", retrieval_context=CASO.retrieval_context),
        _show_indicator=False,
    )

    assert metric.score == 1.0
    assert metric.success is True


@pytest.mark.parametrize("key", ["faithfulness", "answer_relevancy"])
def test_recusa_vira_na_e_nao_nota(key: str) -> None:
    outcome = measure(METRIC_SPECS[key], FakeJev(not_applicable_else(worst)), CASO, LIMIAR)

    assert outcome.score is None
    assert outcome.applicable is False
    assert outcome.error is None
    assert outcome.passed(LIMIAR) is None
    assert outcome.reason.startswith("N/A")


def test_na_parcial_conta_so_as_perguntas_que_se_aplicam() -> None:
    jev = FakeJev({"q_0": NOT_APPLICABLE, "q_1": "Fabricated"})

    outcome = measure(FAITHFULNESS, jev, CASO, LIMIAR)

    assert outcome.applicable is True
    assert outcome.score == pytest.approx(0.0)
    assert [item["applicable"] for item in outcome.breakdown] == [False, True]


def test_na_vale_a_partir_de_metade_da_probabilidade() -> None:
    # NOT_APPLICABLE_THRESHOLD do deepeval: a pergunta sai com >= 50% em N/A.
    meio_a_meio = {"q_0": {NOT_APPLICABLE: 0.5, "Yes": 0.5}, "q_1": {NOT_APPLICABLE: 0.5, "Fully grounded": 0.5}}
    menos = {"q_0": {NOT_APPLICABLE: 0.49, "Yes": 0.51}, "q_1": {NOT_APPLICABLE: 0.49, "Fully grounded": 0.51}}

    assert measure(FAITHFULNESS, FakeJev(meio_a_meio), CASO, LIMIAR).applicable is False
    pouco = measure(FAITHFULNESS, FakeJev(menos), CASO, LIMIAR)
    assert pouco.applicable is True
    # O valor é normalizado pela massa das opções aplicáveis: 0,51 / 0,51 = 1.
    assert pouco.score == pytest.approx(1.0)


# ---------------------------------------------------------------- falhas


def test_falha_do_jev_vira_registro_e_nao_na() -> None:
    outcome = measure(FAITHFULNESS, FakeJev(fail=RuntimeError("rate limit")), CASO, LIMIAR)

    assert outcome.score is None
    assert outcome.applicable is True  # não medida != não se aplica
    assert outcome.error == "RuntimeError: rate limit"
    assert outcome.passed(LIMIAR) is None


def test_caso_sem_trechos_nao_vira_nota() -> None:
    sem_trechos = CaseInputs(input=CASO.input, actual_output=CASO.actual_output, retrieval_context=[])

    outcome = measure(FAITHFULNESS, FakeJev(), sem_trechos, LIMIAR)

    assert outcome.score is None
    assert "retrieval_context" in outcome.error


def test_marcador_de_abstencao_igual_ao_do_relatorio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(shared.PROJECT_ROOT / "tests" / "reports"))
    import report_data

    marker = shared.read_not_found_marker()
    assert marker == report_data.NOT_FOUND_MARKER
    assert "não encontrei" in marker.lower()
