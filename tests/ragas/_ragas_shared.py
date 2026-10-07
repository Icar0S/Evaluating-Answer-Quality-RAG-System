"""Peças da suíte RAGAS reaproveitadas por conftest.py, pelos testes e por run_and_export.py.

Nome com prefixo `_` (não é coletado como teste) e diferente do `_shared.py` de
tests/deepeval de propósito: as duas suítes usam import por diretório, e dois
módulos `_shared` no mesmo processo se sobrescreveriam no sys.modules.

O ponto central deste módulo é o juiz. O estudo de mutação (docs/
estudo-mutacao-status.md, §3.6 e §4.9) concluiu que o `ragas` "não parseia a
saída de nenhum juiz local" — com ragas 0.2.14 + ChatOllama. Relendo aquele
setup, o juiz rodava sem três coisas que a suíte DeepEval já tinha aprendido a
exigir (tests/deepeval/_shared.py):

1. `num_ctx` explícito. Sem ele o Ollama usa 4096 tokens e, quando o prompt
   passa disso, descarta o INÍCIO — justamente as instruções e o JSON schema
   de saída. Com 4 trechos de ~800 tokens, os prompts de faithfulness e
   context recall passam disso fácil.
2. Saída restrita ao schema (`format` do /api/chat). O Ollama decodifica sob a
   gramática do JSON schema; o modelo não consegue devolver JSON inválido.
3. `think: false`. Juiz com raciocínio (qwen3) gasta minutos pensando antes do
   veredito estruturado.

Por isso o juiz aqui não usa `llm_factory` (que fala com o Ollama pelo endpoint
compatível com OpenAI, onde não dá para passar `num_ctx`): é uma implementação
própria de `InstructorBaseRagasLLM` — a interface que as métricas de
`ragas.metrics.collections` aceitam — sobre o /api/chat nativo.
"""
from __future__ import annotations

import os

# Precisa vir ANTES de qualquer import de ragas: a lib envia eventos de uso
# para fora da máquina (ragas/_analytics.py, `track(...)` dentro de
# llm_factory, evaluate etc.) a menos que esta variável esteja ligada. O
# projeto já desliga a telemetria do deepeval pelo mesmo motivo
# (DEEPEVAL_TELEMETRY_OPT_OUT) — avaliação local não deve vazar nada.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import asyncio
import json
import math
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
)

from app import providers
from app.config import PROJECT_ROOT, get_settings
from app.rag import generation, retrieval

# Os mesmos goldens da suíte DeepEval, de propósito: avaliar o mesmo conjunto
# com os dois frameworks é o que permite comparar as métricas entre eles.
GOLDENS_PATH = PROJECT_ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Teto de tokens de SAÍDA do juiz. Com saída restrita ao schema o JSON termina
# sozinho; o teto só corta um modelo pequeno que entre em loop dentro de uma
# lista (visto com juízes de 4B repetindo o mesmo item).
JUDGE_NUM_PREDICT = 2048

T = TypeVar("T", bound=BaseModel)


class RagasSettings(BaseSettings):
    """Configuração da suíte, lida do mesmo .env da raiz que o backend usa."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # qwen3:8b, com o raciocínio desligado pelo juiz (think=false). O gemma3:4b
    # foi testado e reprovado: marca toda resposta como evasiva
    # ("noncommittal": 1), o que zera answer_relevancy até para resposta
    # correta. Para comparar com o DeepEval, use o mesmo juiz dos dois lados
    # (DEEPEVAL_JUDGE_MODEL).
    ragas_judge_model: str = "qwen3:8b"
    ragas_judge_num_ctx: int = 8192
    ragas_judge_timeout_seconds: int = 300
    # Ponto de partida, não dado medido — mesmo 0.7 do DeepEval. Calibrar com
    # os números reais de cada rodada.
    ragas_threshold: float = 0.7


def get_ragas_settings() -> RagasSettings:
    return RagasSettings()


# --------------------------------------------------------------------- juiz


class JudgeError(RuntimeError):
    """Falha do juiz que NÃO pode virar nota: prompt truncado, JSON inválido."""


class OllamaJudge(InstructorBaseRagasLLM):
    """Juiz LOCAL para as métricas do ragas, via /api/chat nativo do Ollama.

    Chamadas serializadas por um lock: as métricas disparam várias chamadas em
    paralelo (uma por trecho, por exemplo), e o Ollama local as enfileiraria de
    qualquer forma — com o lock, o timeout de cada chamada mede a chamada, não a
    fila. Conta chamadas e falhas para o relatório.
    """

    def __init__(self, model: str, base_url: str, num_ctx: int, timeout: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str, response_model: type[T]) -> T:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": response_model.model_json_schema(),
            "think": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": JUDGE_NUM_PREDICT},
        }
        with self._lock:
            self.calls += 1
            try:
                resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                prompt_tokens = data.get("prompt_eval_count") or 0
                # Acima disto o Ollama trunca o prompt pelo começo (limite efetivo
                # = num_ctx - num_predict, ver backend/app/config.py). Nota sobre
                # prompt truncado é pior que nota nenhuma: falha alto.
                if prompt_tokens >= self.num_ctx - JUDGE_NUM_PREDICT:
                    raise JudgeError(
                        f"prompt do juiz com {prompt_tokens} tokens provavelmente truncado "
                        f"(num_ctx={self.num_ctx}); aumente RAGAS_JUDGE_NUM_CTX"
                    )
                content = data.get("message", {}).get("content", "")
                try:
                    return response_model.model_validate_json(content)
                except ValueError as exc:
                    raise JudgeError(f"saída do juiz fora do schema: {exc}; início: {content[:200]!r}") from exc
            except Exception:
                self.failures += 1
                raise

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return await asyncio.to_thread(self.generate, prompt, response_model)


class OllamaEmbeddings(BaseRagasEmbedding):
    """Embeddings locais (o mesmo EMBEDDING_MODEL do app) para AnswerRelevancy."""

    def __init__(self, model: str, base_url: str, timeout: float = 60) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def build_judge() -> OllamaJudge:
    settings, rs = get_settings(), get_ragas_settings()
    return OllamaJudge(
        model=rs.ragas_judge_model,
        base_url=settings.ollama_base_url,
        num_ctx=rs.ragas_judge_num_ctx,
        timeout=rs.ragas_judge_timeout_seconds,
    )


def build_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


# ------------------------------------------------------------------ métricas


@dataclass(frozen=True)
class CaseInputs:
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


# Cada métrica do ragas pede um subconjunto diferente dos campos do caso.
# Equivalência com tests/deepeval (mesma ordem do README de lá):
#   faithfulness       <-> FaithfulnessMetric
#   answer_relevancy   <-> AnswerRelevancyMetric
#   context_precision  <-> ContextualPrecisionMetric (usa a referência)
#   context_recall     <-> ContextualRecallMetric
#   context_relevance  <-> ContextualRelevancyMetric
_METRIC_INPUTS: dict[str, Callable[[CaseInputs], dict[str, Any]]] = {
    "faithfulness": lambda c: {
        "user_input": c.user_input,
        "response": c.response,
        "retrieved_contexts": c.retrieved_contexts,
    },
    "answer_relevancy": lambda c: {"user_input": c.user_input, "response": c.response},
    "context_precision": lambda c: {
        "user_input": c.user_input,
        "reference": c.reference,
        "retrieved_contexts": c.retrieved_contexts,
    },
    "context_recall": lambda c: {
        "user_input": c.user_input,
        "retrieved_contexts": c.retrieved_contexts,
        "reference": c.reference,
    },
    "context_relevance": lambda c: {"user_input": c.user_input, "retrieved_contexts": c.retrieved_contexts},
}
METRIC_NAMES = tuple(_METRIC_INPUTS)


# Única alteração nos prompts de fábrica do ragas — registrada em cada export
# (run_and_export.py) para a análise saber que existe. AnswerRelevancy gera uma
# pergunta a partir da resposta e compara por embedding com a pergunta original.
# O prompt é todo em inglês, então o juiz gera a pergunta em inglês e a compara
# com a original em português: medido com qwen3:8b numa resposta correta, 0,68
# sem a instrução contra 0,83 com ela (resposta fora do assunto: 0,55 nos dois).
# O `prompt.adapt()` do ragas traduziria os exemplos usando o próprio juiz a
# cada execução — prompt diferente a cada rodada, ruim para teste.
ANSWER_RELEVANCY_LANGUAGE_HINT = "\nWrite the question in the same language as the response."
PROMPT_OVERRIDES = {"answer_relevancy": f"instruction += {ANSWER_RELEVANCY_LANGUAGE_HINT.strip()!r}"}


def build_metrics(llm: OllamaJudge, embeddings: OllamaEmbeddings) -> dict[str, Any]:
    # strictness=1, não o padrão 3: com o juiz em temperatura 0 as 3 perguntas
    # geradas saem idênticas (medido) — o triplo do custo pela mesma nota.
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
    answer_relevancy.prompt.instruction += ANSWER_RELEVANCY_LANGUAGE_HINT
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": answer_relevancy,
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "context_relevance": ContextRelevance(llm=llm),
    }


@dataclass
class MetricOutcome:
    name: str
    score: float | None
    error: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.score is not None


def score_metric(name: str, metric: Any, case: CaseInputs) -> MetricOutcome:
    """Mede uma métrica. Erro ou NaN viram `score=None` com o motivo — nunca nota.

    NaN não é detalhe: Faithfulness devolve NaN quando o juiz não extrai
    nenhuma afirmação da resposta (ragas/metrics/collections/faithfulness). No
    estudo de mutação, o equivalente silencioso disso no deepeval ("zero
    afirmações" lido como verdade vazia) deu nota 1,0 a uma alucinação.
    """
    try:
        result = asyncio.run(metric.ascore(**_METRIC_INPUTS[name](case)))
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira registro, não exceção
        return MetricOutcome(name=name, score=None, error=f"{type(exc).__name__}: {exc}")

    value = result.value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MetricOutcome(name=name, score=None, error="NaN: métrica indefinida para este caso")
    return MetricOutcome(name=name, score=float(value), reason=getattr(result, "reason", None))


# ------------------------------------------------------------------- goldens


@dataclass(frozen=True)
class Golden:
    name: str
    input: str
    expected_output: str
    context: list[str]


def load_goldens() -> list[Golden]:
    raw = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    return [
        Golden(
            name=g.get("name") or g["input"][:40],
            input=g["input"],
            expected_output=g["expected_output"],
            context=list(g.get("context") or []),
        )
        for g in raw
    ]


# ----------------------------------------------------------------------- SUT


def run_pipeline(question: str) -> tuple[str, list[str]]:
    """Chama o mesmo pipeline que POST /chat usa (backend/app/main.py), in-process."""
    settings = get_settings()
    chunks, _retrieval_latency_ms = retrieval.retrieve(question, settings.top_k)
    result = generation.generate_answer(question, chunks)
    return result["answer"], [c["text"] for c in chunks]


def stack_status() -> str | None:
    """None se Ollama, modelos e vector store estiverem prontos; senão, o motivo do skip."""
    settings, rs = get_settings(), get_ragas_settings()
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        resp.raise_for_status()
    except httpx.HTTPError:
        return f"Ollama inacessível em {settings.ollama_base_url}"

    tags = {m["name"] for m in resp.json().get("models", [])}

    def _has(model: str) -> bool:
        return model in tags or f"{model}:latest" in tags

    active = providers.get_active_provider()
    required = [settings.embedding_model, rs.ragas_judge_model]
    if active.kind == "ollama":
        required.append(active.generation_model)
    for model in required:
        if not _has(model):
            return f"Modelo '{model}' não está baixado no Ollama local (`ollama pull {model}`)"

    try:
        from app.rag import vector_store

        if vector_store.collection_count() == 0:
            return "Vector store vazio — rode `python scripts/ingest_documents.py` antes"
    except Exception as exc:  # noqa: BLE001 - índice inacessível = stack não pronta
        return f"Vector store inacessível: {exc}"

    if active.kind == "ollama" and active.generation_model == rs.ragas_judge_model:
        warnings.warn(
            f"RAGAS_JUDGE_MODEL ({rs.ragas_judge_model}) é o mesmo modelo que gera as respostas: "
            "o juiz estará avaliando a si mesmo (self-grading). Use outro modelo no .env.",
            stacklevel=2,
        )
    return None
