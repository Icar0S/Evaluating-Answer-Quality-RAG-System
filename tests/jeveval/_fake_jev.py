"""Jev falso e determinístico para os testes herméticos: sem rede, sem chave, sem custo.

Implementa a mesma interface do Jev de verdade no deepeval
(`DeepEvalBaseSystemOneModel.decide`): recebe o estado e as perguntas já no
formato do System One e devolve `SystemOneAnswers` com probabilidades. Tudo o
que vem depois — valor de cada pergunta, média ponderada, N/A, reason — é o
código real do deepeval 4.3.1. É isso que torna o teste útil: o que se testa é
a camada e o JevEval, não um dublê deles.

As respostas vêm de uma regra `answer(state, key, question) -> spec`:

- Noul: float com P(verdadeiro);
- Choice: nome de uma opção (P = 1), (opção, p) com o resto dividido entre as
  outras, ou um dict opção -> p;
- Score: índice do nível (P = 1), (nível, p) ou um dict nível -> p.

`answer` também pode ser um dict pela chave da pergunta ("q_0", "q_1"...).
"""
from __future__ import annotations

from typing import Any, Callable

from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
)

from _jev_shared import METRIC_SPECS, NOT_APPLICABLE

Rule = Callable[[dict, str, Any], Any]
FAKE_MODEL_NAME = "fake-jev (teste)"


def _confidence(probabilities: list[float]) -> float:
    # A conta que a TypeSafe usa para Choice e Score (ver NoulAnswer.confidence
    # em deepeval/models/system_one/schema.py): 0 = uniforme, 1 = certeza.
    n = len(probabilities)
    return (n * max(probabilities) - 1) / (n - 1) if n > 1 else 1.0


def _spread(names: list, spec: Any) -> dict:
    """Distribuição completa sobre `names` a partir de uma escolha parcial."""
    if isinstance(spec, tuple):
        spec = {spec[0]: spec[1]}
    elif not isinstance(spec, dict):
        spec = {spec: 1.0}
    unknown = set(spec) - set(names)
    if unknown:
        raise ValueError(f"opções inexistentes na pergunta: {unknown}")
    out = {name: float(spec.get(name, 0.0)) for name in names}
    rest = [name for name in names if name not in spec]
    left = 1.0 - sum(spec.values())
    for name in rest:
        out[name] = left / len(rest)
    return out


class FakeJev(DeepEvalBaseSystemOneModel):
    def __init__(self, answer: Rule | dict[str, Any] | None = None, fail: Exception | None = None):
        self._rule: Rule = (lambda state, key, q: answer[key]) if isinstance(answer, dict) else (answer or best)
        self.fail = fail
        self.calls: list[tuple[dict, dict]] = []
        super().__init__(FAKE_MODEL_NAME)

    def load_model(self, *args, **kwargs):
        return None

    def decide(self, state: Any, questions: dict) -> tuple[SystemOneAnswers, None]:
        self.calls.append((state, questions))
        if self.fail is not None:
            raise self.fail
        out = SystemOneAnswers()
        for key, question in questions.items():
            spec = self._rule(state, key, question)
            if isinstance(question, NoulQuestion):
                out.nouls[key] = NoulAnswer(probability=float(spec))
            elif isinstance(question, ChoiceQuestion):
                probabilities = _spread(list(question.options), spec)
                out.choices[key] = ChoiceAnswer(
                    choice=max(probabilities, key=probabilities.get),
                    probabilities=probabilities,
                    confidence=_confidence(list(probabilities.values())),
                )
            elif isinstance(question, ScoreQuestion):
                probabilities = _spread(list(range(len(question.levels))), spec)
                out.scores[key] = ScoreAnswer(
                    score=sum(level * p for level, p in probabilities.items()),
                    probabilities=probabilities,
                    confidence=_confidence(list(probabilities.values())),
                )
        return out, None

    async def a_decide(self, state: Any, questions: dict) -> tuple[SystemOneAnswers, None]:
        return self.decide(state, questions)

    def get_model_name(self, *args, **kwargs) -> str:
        return FAKE_MODEL_NAME


# ------------------------------------------------- regras prontas

_QUESTIONS = {q.text: q for spec in METRIC_SPECS.values() for q in spec.questions}


def _instrument_question(question: Any):
    """A pergunta do instrumento (_jev_shared) que originou esta pergunta do System One."""
    found = _QUESTIONS.get(question.instructions)
    if found is None:
        raise KeyError(f"pergunta fora do instrumento: {question.instructions[:80]!r}")
    return found


def best(state: dict, key: str, question: Any) -> Any:
    """A melhor resposta possível a cada pergunta: nota 1 em toda métrica."""
    q = _instrument_question(question)
    if isinstance(question, NoulQuestion):
        return 1.0
    if isinstance(question, ScoreQuestion):
        return len(q.levels) - 1
    return max(q.applicable_options, key=q.applicable_options.get)


def worst(state: dict, key: str, question: Any) -> Any:
    """A pior resposta possível a cada pergunta: nota 0 em toda métrica."""
    q = _instrument_question(question)
    if isinstance(question, NoulQuestion):
        return 0.0
    if isinstance(question, ScoreQuestion):
        return 0
    return min(q.applicable_options, key=q.applicable_options.get)


def not_applicable_else(fallback: Rule) -> Rule:
    """N/A em toda pergunta que tem essa opção; `fallback` nas outras."""

    def rule(state: dict, key: str, question: Any) -> Any:
        if isinstance(question, ChoiceQuestion) and NOT_APPLICABLE in question.options:
            return NOT_APPLICABLE
        return fallback(state, key, question)

    return rule
