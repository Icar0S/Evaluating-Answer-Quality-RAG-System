"""Peças da camada JevEval, reaproveitadas pelos testes e pelos scripts.

Nome com prefixo `_` (não é coletado como teste) e diferente do `_shared.py` de
tests/deepeval e do `_ragas_shared.py` de tests/ragas: as suítes usam import por
diretório, e dois módulos com o mesmo nome no mesmo processo se sobrescreveriam
no sys.modules.

O JevEval (deepeval >= 4.3) separa o que o LLM-as-a-judge mistura:

- a DECISÃO fica com o Jev, o modelo "System One" da TypeSafe AI. Ele não gera
  texto: responde perguntas fechadas (Noul = proposição verdadeira ou falsa,
  Score = níveis ordenados, Choice = opções com crédito) com probabilidades;
- a NOTA sai de uma conta fixa, feita fora do modelo: a média ponderada do
  valor de cada pergunta (deepeval/metrics/jev_eval/utils.py).

O instrumento desta camada é, então, o conjunto de perguntas abaixo. Ele fica
congelado no código e vai registrado em cada export (`question_sets` e
`instrument_sha`), como o ragas registra as alterações de prompt.

De onde vêm as perguntas:

- precisão, recall e relevância do contexto: as MESMAS perguntas que o próprio
  deepeval manda ao Jev no modo `system_one` das métricas nativas
  (deepeval/metrics/<métrica>/templates/_experimental_system_one_questions.txt).
  Um teste hermético compara a cópia daqui com o arquivo do pacote instalado: se
  uma atualização do deepeval mudar as perguntas, o teste avisa, em vez de o
  instrumento mudar em silêncio;
- fidelidade e relevância da resposta: as perguntas do deepeval com duas
  adaptações motivadas por problemas que o projeto já documentou (ver as
  constantes NOT_APPLICABLE e UNMENTIONED_IS_UNSUPPORTED);
- correção (resposta x referência): própria, com a semântica do oráculo O4 do
  estudo de mutação, que foi calibrada no piloto. É ela que os 247 pares
  rotulados validam e que o oráculo O6 aplica ao log da campanha.

As perguntas estão em inglês de propósito, com o conteúdo em português: é a
mesma condição dos prompts do DeepEval e do RAGAS neste projeto. Mudar o idioma
só daqui misturaria idioma do instrumento com método na comparação entre os
três frameworks.

Diferente das outras duas camadas, o juiz NÃO é local: cada medição manda
pergunta, resposta, referência e trechos para https://api.typesafe.ai. O acervo
é de artigos públicos, mas é a primeira camada que tira conteúdo da máquina, e
só roda com TYPESAFE_API_KEY definida (ver README.md desta pasta).
"""
from __future__ import annotations

import os

# Antes de qualquer import do deepeval: a lib lê isto uma vez, ao criar o seu
# Settings interno. É o mesmo opt-out que o .env.example liga para tests/deepeval.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
GOLDENS_PATH = PROJECT_ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"

JevQuestion = Union[Noul, Score, Choice]


# ------------------------------------------------------------- configuração


class JevSettings(BaseSettings):
    """Configuração da camada, lida do ambiente ou do mesmo .env da raiz que o backend usa."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    typesafe_api_key: SecretStr | None = None
    # Versão FIXA, não o alias `jev-latest`: o alias passa a apontar para outro
    # modelo quando a TypeSafe publica uma versão nova, e o instrumento mudaria
    # entre duas rodadas sem nada mudar no repositório. 1.13.0 é a única versão
    # concreta no registro do deepeval 4.3.1 (deepeval/models/system_one/constants.py).
    jeveval_model: str = "jev-1.13.0"
    # O mesmo 0,7 do DeepEval e do RAGAS nas cinco dimensões: com limiares
    # iguais, a comparação de vereditos entre os três frameworks no relatório
    # consolidado é de método, não de régua. Ponto de partida, não valor medido.
    jeveval_threshold: float = 0.7
    # A correção é uma pergunta só, e a nota é a probabilidade de a resposta
    # estar certa. 0,5 = "mais provável certa do que errada", fixado ANTES de ver
    # os pares rotulados (a varredura de limiares sai no relatório, não escolhe).
    jeveval_correctness_threshold: float = 0.5
    # O padrão do typesafe-sdk é 10 s por operação HTTP; com 4 trechos de ~800
    # tokens no estado, folga para não transformar latência em erro.
    jeveval_timeout_seconds: float = 60.0


def get_settings() -> JevSettings:
    return JevSettings()


def _api_key(settings: JevSettings) -> str | None:
    key = settings.typesafe_api_key.get_secret_value().strip() if settings.typesafe_api_key else ""
    return key or None


class JevUnavailable(RuntimeError):
    """O Jev de verdade não pode ser usado (sem chave ou sem o SDK)."""


def jev_status(settings: JevSettings | None = None) -> str | None:
    """None se o Jev de verdade puder ser chamado; senão, o motivo (vira skip nos testes ao vivo)."""
    s = settings or get_settings()
    if not _api_key(s):
        return "TYPESAFE_API_KEY não definida (ambiente ou .env da raiz); ver tests/jeveval/README.md"
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        return "typesafe-sdk não instalado (tests\\jeveval\\.venv\\Scripts\\pip install -r tests\\jeveval\\requirements.txt)"
    return None


def build_jev(settings: JevSettings | None = None) -> DeepEvalBaseSystemOneModel:
    """O Jev de verdade, com a versão fixada e a chave passada explicitamente.

    Explícita porque o deepeval também procura a chave no próprio Settings, que
    lê arquivos .env do diretório de trabalho: passar daqui deixa um lugar só
    (JevSettings) decidindo qual chave e qual modelo valem.
    """
    s = settings or get_settings()
    reason = jev_status(s)
    if reason:
        raise JevUnavailable(reason)
    from deepeval.models.system_one import TypeSafeModel

    return TypeSafeModel(model=s.jeveval_model, api_key=_api_key(s), timeout=s.jeveval_timeout_seconds)


# --------------------------------------------------------------- instrumento

P = SingleTurnParams

# Opção "não se aplica" das perguntas sobre a resposta, com crédito None: o
# JevEval tira a pergunta da conta quando essa opção concentra >= 50% da
# probabilidade (NOT_APPLICABLE_THRESHOLD em deepeval/metrics/jev_eval/utils.py).
# Motivo: quando o assistente responde "não encontrei", a FaithfulnessMetric com
# juiz LLM dá 0,0 à recusa, e o certo é N/A — recusa não tem afirmação para ser
# fiel ou infiel. Se a abstenção foi certa ou errada é a pergunta da CORREÇÃO
# (a referência diz se a informação existe), não da fidelidade.
NOT_APPLICABLE = (
    "Not applicable: `actual_output` makes no factual claim "
    "(it declines, says the information was not found, or asks for clarification)"
)

# Fidelidade como afirmação INFERÍVEL do contexto, e não "não contradiz". No
# estudo de mutação (docs/estudo-mutacao-status.md, §4.9), "a WCAG 2.2 define 14
# critérios", com contexto que nem cita WCAG, levou faithfulness 1,0 do deepeval:
# inventar sobre tópico ausente não contradiz nada. Por isso a pergunta do
# deepeval "No claim in `actual_output` contradicts the facts in
# `retrieval_context`" SAIU daqui, e a regra abaixo entrou nas que ficaram.
UNMENTIONED_IS_UNSUPPORTED = (
    " A claim about something that `retrieval_context` does not mention is not supported,"
    " even if nothing in `retrieval_context` contradicts it."
)

# Opções da correção. Crédito 1 só para a resposta certa: a nota é a
# probabilidade que o Jev dá a ela. As demais nomeiam o defeito e, de graça,
# dão a causa provável (o argmax entre elas), como o O4 faz em texto.
CORRECT = (
    "Correct: it has every value, entity and condition of `expected_output`; "
    "extra detail or different wording is fine"
)
CORRECTNESS_OPTIONS: dict[str, float | None] = {
    CORRECT: 1.0,
    "Omits a value, entity or condition that `expected_output` has": 0.0,
    "Changes or contradicts a value, entity or condition of `expected_output`": 0.0,
    "Does not do what `expected_output` does: answers where it declines or asks for clarification, "
    "or declines where it answers": 0.0,
    "Drops the source citation that `expected_output` gives": 0.0,
    "Answers a different question than `input`": 0.0,
}
# Opção -> causa, no vocabulário do O4 (tests/mutation/oracles/o4_judge.py,
# CAUSE_OPTIONS), para o O6 poder ser cruzado com ele caso a caso.
CORRECTNESS_CAUSES = dict(
    zip(CORRECTNESS_OPTIONS, ("none", "missing_info", "wrong_value", "no_abstention", "no_citation", "off_topic"))
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    params: tuple[SingleTurnParams, ...]
    questions: tuple[JevQuestion, ...]
    source: str

    def question_set(self) -> list[dict[str, Any]]:
        return [q.model_dump() for q in self.questions]


FAITHFULNESS = MetricSpec(
    key="faithfulness",
    label="Fidelidade ao contexto",
    params=(P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT),
    questions=(
        Choice(
            "Is every factual claim in `actual_output` supported by the facts in `retrieval_context`?"
            + UNMENTIONED_IS_UNSUPPORTED,
            options={"No": 0.0, "Yes": 1.0, NOT_APPLICABLE: None},
            weight=2,
        ),
        Choice(
            "How much of `actual_output` is grounded in the facts in `retrieval_context`?"
            + UNMENTIONED_IS_UNSUPPORTED,
            options={
                "Fabricated": 0.0,
                "Mostly fabricated": 1 / 3,
                "Mostly grounded": 2 / 3,
                "Fully grounded": 1.0,
                NOT_APPLICABLE: None,
            },
        ),
    ),
    source=(
        "modo system_one da FaithfulnessMetric (deepeval 4.3.1), adaptado: sem a pergunta "
        "'nenhuma afirmação contradiz o contexto', com a regra de inferibilidade e a opção N/A"
    ),
)

ANSWER_RELEVANCY = MetricSpec(
    key="answer_relevancy",
    label="Relevância da resposta",
    params=(P.INPUT, P.ACTUAL_OUTPUT),
    questions=(
        Choice(
            "Is every statement in `actual_output` relevant to addressing `input`?",
            options={"No": 0.0, "Yes": 1.0, NOT_APPLICABLE: None},
            weight=2,
        ),
        Choice(
            "Does `actual_output` directly address what `input` asks for, rather than a different or broader topic?",
            options={"No": 0.0, "Yes": 1.0, NOT_APPLICABLE: None},
        ),
        Choice(
            "How much of `actual_output` is relevant to addressing `input`?",
            options={
                "None of it": 0.0,
                "Less than half": 1 / 3,
                "Most of it": 2 / 3,
                "All of it": 1.0,
                NOT_APPLICABLE: None,
            },
        ),
    ),
    source="modo system_one da AnswerRelevancyMetric (deepeval 4.3.1), com a opção N/A",
)

# As três de contexto são cópia literal das perguntas do deepeval (ver o teste
# test_perguntas_de_contexto_iguais_as_do_deepeval). Não pedem N/A: julgam a
# recuperação, que existe mesmo quando o assistente se abstém.
CONTEXT_PRECISION = MetricSpec(
    key="context_precision",
    label="Precisão do contexto",
    params=(P.INPUT, P.RETRIEVAL_CONTEXT, P.EXPECTED_OUTPUT),
    questions=(
        Noul(
            "In `retrieval_context`, every document useful for arriving at `expected_output` for `input` "
            "is listed before every document that is not useful.",
            weight=2,
        ),
        Noul("The first document in `retrieval_context` is useful for arriving at `expected_output` for `input`."),
        Score(
            "How well are the documents in `retrieval_context` ordered, with the ones useful for arriving at "
            "`expected_output` for `input` first?",
            levels=[
                "Useful documents are all at the end",
                "Useful documents are mixed with irrelevant ones",
                "Useful documents are mostly first",
                "Useful documents are all first",
            ],
        ),
    ),
    source="modo system_one da ContextualPrecisionMetric (deepeval 4.3.1), sem alteração",
)

CONTEXT_RECALL = MetricSpec(
    key="context_recall",
    label="Recall do contexto",
    params=(P.EXPECTED_OUTPUT, P.RETRIEVAL_CONTEXT),
    questions=(
        Noul("Every sentence in `expected_output` can be attributed to the facts in `retrieval_context`.", weight=2),
        Noul("`retrieval_context` contains the key facts needed to write `expected_output`."),
        Score(
            "How much of `expected_output` can be attributed to the facts in `retrieval_context`?",
            levels=["None of it", "Less than half", "Most of it", "All of it"],
        ),
    ),
    source="modo system_one da ContextualRecallMetric (deepeval 4.3.1), sem alteração",
)

CONTEXT_RELEVANCE = MetricSpec(
    key="context_relevance",
    label="Relevância do contexto",
    params=(P.INPUT, P.RETRIEVAL_CONTEXT),
    questions=(
        Noul("Every statement in `retrieval_context` is relevant to addressing `input`.", weight=2),
        Noul("Every node in `retrieval_context` contains at least one statement relevant to addressing `input`."),
        Score(
            "How much of `retrieval_context` is relevant to addressing `input`?",
            levels=["None of it", "Less than half", "Most of it", "All of it"],
        ),
    ),
    source="modo system_one da ContextualRelevancyMetric (deepeval 4.3.1), sem alteração",
)

CORRECTNESS = MetricSpec(
    key="correctness",
    label="Correção (resposta x referência)",
    params=(P.INPUT, P.EXPECTED_OUTPUT, P.ACTUAL_OUTPUT),
    questions=(
        Choice(
            "Compare `actual_output` with `expected_output`, the reference answer to `input`. "
            "`expected_output` is the minimum a correct answer must contain, not the maximum: "
            "you cannot see the source documents, so judge omissions and contradictions, "
            "not whether extra detail is true. Which describes `actual_output`?",
            options=CORRECTNESS_OPTIONS,
        ),
    ),
    source=(
        "própria, com a semântica do oráculo O4 do estudo de mutação (tests/mutation/oracles/o4_judge.py): "
        "a referência é o mínimo, não o máximo"
    ),
)

# As cinco dimensões do relatório consolidado (tests/reports/report_data.py,
# DIMENSIONS), na mesma ordem, e a correção, que só esta camada mede.
DIMENSION_SPECS = (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION, CONTEXT_RECALL, CONTEXT_RELEVANCE)
METRIC_SPECS: dict[str, MetricSpec] = {s.key: s for s in (*DIMENSION_SPECS, CORRECTNESS)}
METRIC_NAMES = tuple(METRIC_SPECS)


def question_sets(keys: tuple[str, ...] | list[str] = METRIC_NAMES) -> dict[str, list[dict[str, Any]]]:
    return {key: METRIC_SPECS[key].question_set() for key in keys}


def instrument_sha(keys: tuple[str, ...] | list[str] = METRIC_NAMES) -> str:
    """Impressão digital das perguntas: duas rodadas com o mesmo sha usaram o mesmo instrumento."""
    payload = json.dumps(
        {k: {"params": [p.value for p in METRIC_SPECS[k].params], "questions": METRIC_SPECS[k].question_set()}
         for k in keys},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def threshold_for(key: str, settings: JevSettings) -> float:
    return settings.jeveval_correctness_threshold if key == CORRECTNESS.key else settings.jeveval_threshold


# ------------------------------------------------------------------ medição


@dataclass(frozen=True)
class CaseInputs:
    input: str
    actual_output: str | None
    expected_output: str | None = None
    retrieval_context: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """Resultado de uma métrica num caso.

    Três estados, que não podem se confundir:
    - medida: `score` com a nota;
    - N/A: `applicable=False`, `score=None`, sem erro (nenhuma pergunta se aplicou);
    - não medida: `score=None` com `error` (falha do Jev ou do caso).
    """

    name: str
    score: float | None
    applicable: bool = True
    error: str | None = None
    reason: str | None = None
    confidence: float | None = None
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    cost_usd: float | None = None
    input_tokens: int | None = None

    @property
    def measured(self) -> bool:
        return self.score is not None

    def passed(self, threshold: float) -> bool | None:
        return None if self.score is None else self.score >= threshold

    @property
    def cause(self) -> str | None:
        """Causa mais provável, só para a correção: a opção de falha com mais probabilidade."""
        if self.name != CORRECTNESS.key or not self.breakdown:
            return None
        probabilities = self.breakdown[0].get("probabilities") or {}
        failures = {opt: p for opt, p in probabilities.items() if opt != CORRECT}
        if not failures or probabilities.get(CORRECT, 0.0) >= max(failures.values()):
            return "none"
        return CORRECTNESS_CAUSES.get(max(failures, key=failures.get), "other")


def build_metric(spec: MetricSpec, model: DeepEvalBaseSystemOneModel, threshold: float) -> JevEval:
    # async_mode=False: uma chamada decide() síncrona por medição. O modo
    # assíncrono passa por asyncio.wait_for dentro do deepeval, e a camada
    # DeepEval já perdeu uma rodada para um CancelledError vindo de lá
    # (tests/deepeval/run_and_export.py, _measure_raw). Aqui não há o que
    # paralelizar: é uma requisição por métrica.
    return JevEval(
        name=spec.key,
        evaluation_params=list(spec.params),
        questions=list(spec.questions),
        system_one_model=model,
        threshold=threshold,
        async_mode=False,
        include_reason=True,
        _include_jev_eval_suffix=False,
    )


def measure(spec: MetricSpec, model: DeepEvalBaseSystemOneModel, case: CaseInputs, threshold: float) -> Outcome:
    """Mede uma métrica num caso. Nunca propaga exceção do Jev: vira registro com o motivo.

    O ponto central é a nota quando NENHUMA pergunta se aplica: o JevEval devolve
    1,0 e success=True (EMPTY_SCORE em deepeval/metrics/jev_eval/utils.py — "não
    havia nada aplicável para reprovar"). É a mesma verdade vazia que deu 1,0 a
    uma alucinação no estudo de mutação, só que do outro lado. Aqui ela vira N/A
    explícito: fora das médias e das contagens, nunca nota.
    """
    metric = build_metric(spec, model, threshold)
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        expected_output=case.expected_output,
        retrieval_context=list(case.retrieval_context) or None,
    )
    t0 = time.perf_counter()
    try:
        metric.measure(test_case, _show_indicator=False)
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira registro, não exceção
        return Outcome(
            name=spec.key,
            score=None,
            error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - t0, 2),
        )

    breakdown = list(metric.score_breakdown or [])
    common = {
        "confidence": metric.confidence,
        "breakdown": breakdown,
        "seconds": round(time.perf_counter() - t0, 2),
        "cost_usd": metric.evaluation_cost,
        "input_tokens": getattr(metric, "input_tokens", None) or None,
    }
    if breakdown and not any(item.get("applicable") for item in breakdown):
        return Outcome(
            name=spec.key,
            score=None,
            applicable=False,
            reason=f"N/A: nenhuma pergunta se aplica a este caso.\n{metric.reason}",
            **common,
        )
    return Outcome(name=spec.key, score=float(metric.score), reason=metric.reason, **common)


# ------------------------------------------------------- respostas e goldens


@dataclass(frozen=True)
class Case:
    """Um caso a julgar: o golden e a resposta já gerada para ele."""

    name: str
    input: str
    expected_output: str
    context: list[str]
    actual_output: str | None
    retrieval_context: list[str]
    error: str | None = None
    origin: str | None = None
    seconds: float | None = None

    def inputs(self) -> CaseInputs:
        return CaseInputs(self.input, self.actual_output, self.expected_output, list(self.retrieval_context))


def load_goldens(path: Path = GOLDENS_PATH) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(g.get("name") or g["input"][:40]): g for g in raw}


def load_answers(path: Path) -> tuple[dict[str, Any], list[Case]]:
    """Respostas de tests/generate_answers.py, as mesmas que DeepEval e RAGAS julgam.

    O arquivo é autocontido (pergunta, referência e contexto de cada golden).
    Arquivos antigos, sem esses campos, caem nos goldens padrão pelo nome — o
    mesmo fallback dos run_and_export.py das outras duas camadas.
    """
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    goldens = load_goldens()
    cases = []
    for a in meta.get("answers", []):
        golden = a if "expected_output" in a else goldens.get(a["name"])
        if golden is None:
            continue
        error = a.get("error") or (None if a.get("actual_output") is not None else "sem resposta no arquivo")
        cases.append(
            Case(
                name=a["name"],
                input=golden.get("input") or a.get("input", ""),
                expected_output=golden.get("expected_output", ""),
                context=list(golden.get("context") or []),
                actual_output=a.get("actual_output"),
                retrieval_context=list(a.get("retrieval_context") or []),
                error=error,
                origin=a.get("origin"),
                seconds=a.get("seconds"),
            )
        )
    return meta, cases


def read_not_found_marker() -> str | None:
    """Frase de abstenção do assistente, lida da fonte (backend/app/rag/generation.py) sem importar o backend.

    Mesma leitura por `ast` de tests/reports/report_data.py; um teste hermético
    confere que as duas dão o mesmo texto.
    """
    source = PROJECT_ROOT / "backend" / "app" / "rag" / "generation.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "NOT_FOUND_MARKER" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


# -------------------------------------------------------------------- export


def metric_record(outcome: Outcome, threshold: float) -> dict[str, Any]:
    """Uma métrica de um caso no formato do export (o mesmo `metrics[]` do DeepEval e do RAGAS, mais o N/A)."""
    record = {
        "name": outcome.name,
        "score": None if outcome.score is None else round(outcome.score, 4),
        "success": outcome.passed(threshold),
        "applicable": outcome.applicable,
        "threshold": threshold,
        "reason": outcome.reason,
        "error": outcome.error,
        "confidence": None if outcome.confidence is None else round(outcome.confidence, 4),
        "seconds": outcome.seconds,
        "cost_usd": outcome.cost_usd,
        "input_tokens": outcome.input_tokens,
        "breakdown": outcome.breakdown,
    }
    if outcome.name == CORRECTNESS.key:
        record["cause"] = outcome.cause
    return record


def not_measured_record(name: str, threshold: float, error: str) -> dict[str, Any]:
    return metric_record(Outcome(name=name, score=None, error=error), threshold)


def case_success(metrics: list[dict[str, Any]]) -> bool:
    """Todas as notas aplicáveis medidas e >= limiar. N/A não reprova; nota que não existe, sim."""
    applicable = [m for m in metrics if m.get("applicable", True)]
    return all(m["success"] for m in applicable)


def metrics_summary(cases: list[dict[str, Any]], names: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Média (só notas medidas), medidas, aprovadas e N/A por métrica."""
    avg, measured, passed, not_applicable = {}, {}, {}, {}
    for name in names:
        records = [m for c in cases for m in c["metrics"] if m["name"] == name]
        applicable = [m for m in records if m.get("applicable", True)]
        scores = [m["score"] for m in applicable if m["score"] is not None]
        avg[name] = round(sum(scores) / len(scores), 4) if scores else None
        measured[name] = f"{len(scores)}/{len(applicable)}"
        passed[name] = f"{sum(1 for m in applicable if m['success'])}/{len(applicable)}"
        not_applicable[name] = len(records) - len(applicable)
    return {
        "metrics_avg": avg,
        "metrics_measured": measured,
        "metrics_passed": passed,
        "metrics_not_applicable": not_applicable,
    }
