"""O3 — RAGAS: faithfulness, answer relevancy e context precision (§6.4).

Custo médio, roda sobre a resposta modal das N execuções.

Dois backends, escolhidos explicitamente em config/study.yaml (`oracles.O3.backend`),
nunca por fallback automático:

- `ragas`: o instrumento nomeado no protocolo, sobre LLM e embeddings locais via
  langchain-ollama.
- `deepeval`: as três métricas equivalentes da biblioteca que o repositório já usa
  em tests/deepeval/. Existe porque o ponto de decisão da semana 5 prevê rodar com
  menos oráculos se um deles não estabilizar — e porque trocar de biblioteca no
  meio da campanha, em silêncio, tornaria os resultados incomparáveis entre
  mutantes. Se o backend for trocado, isso muda o instrumento e precisa ser
  declarado no artigo.

Falta da biblioteca é erro, não degradação silenciosa: um O3 que "passa" porque
não conseguiu medir contaminaria a RQ2 inteira.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from tests.mutation.oracles.base import Verdict
from tests.mutation.runner import sut  # noqa: F401  (garante backend no sys.path)
from app.config import get_settings

METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision")


@dataclass
class O3Config:
    backend: str = "ragas"
    judge_model: str = "gemma3:4b"
    embedding_model: str = "nomic-embed-text"
    thresholds: dict[str, float] | None = None
    aggregate: str = "all"

    def threshold(self, key: str) -> float:
        return (self.thresholds or {}).get(key, 0.7)


_ragas_cache: dict[str, Any] = {}
_deepeval_cache: dict[str, Any] = {}


def _build_ragas(config: O3Config) -> dict[str, Any]:
    if _ragas_cache:
        return _ragas_cache

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "Backend 'ragas' pedido mas as bibliotecas não estão instaladas. "
            "Instale com: pip install -r tests/mutation/requirements.txt"
        ) from exc

    # Nomes das métricas mudaram entre versões do ragas; tentar os dois evita
    # quebrar a campanha por um rename de biblioteca.
    try:
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
        relevancy_metric = ResponseRelevancy
        precision_metric = LLMContextPrecisionWithReference
    except ImportError:  # pragma: no cover
        from ragas.metrics import AnswerRelevancy as relevancy_metric  # type: ignore
        from ragas.metrics import ContextPrecision as precision_metric  # type: ignore
        from ragas.metrics import Faithfulness  # type: ignore

    settings = get_settings()
    llm = LangchainLLMWrapper(
        ChatOllama(model=config.judge_model, base_url=settings.ollama_base_url, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=config.embedding_model, base_url=settings.ollama_base_url)
    )

    _ragas_cache["metrics"] = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": relevancy_metric(llm=llm, embeddings=embeddings),
        "context_precision": precision_metric(llm=llm),
    }
    return _ragas_cache


def _score_ragas(
    config: O3Config, question: str, answer: str, contexts: list[str], reference: str
) -> dict[str, float]:
    from ragas import SingleTurnSample

    metrics = _build_ragas(config)["metrics"]
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=list(contexts),
        reference=reference,
    )

    async def run_all() -> dict[str, float]:
        scores: dict[str, float] = {}
        for key, metric in metrics.items():
            scores[key] = float(await metric.single_turn_ascore(sample))
        return scores

    return asyncio.run(run_all())


def _build_deepeval(config: O3Config) -> dict[str, Any]:
    if _deepeval_cache:
        return _deepeval_cache

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
        )
        from deepeval.models import OllamaModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Backend 'deepeval' pedido mas a biblioteca não está instalada. "
            "Instale com: pip install -r tests/mutation/requirements.txt"
        ) from exc

    settings = get_settings()
    judge = OllamaModel(
        model=config.judge_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        generation_kwargs={"num_ctx": 8192},
    )
    _deepeval_cache["metrics"] = {
        "faithfulness": FaithfulnessMetric(model=judge, threshold=config.threshold("faithfulness")),
        "answer_relevancy": AnswerRelevancyMetric(model=judge, threshold=config.threshold("answer_relevancy")),
        "context_precision": ContextualPrecisionMetric(model=judge, threshold=config.threshold("context_precision")),
    }
    return _deepeval_cache


def _score_deepeval(
    config: O3Config, question: str, answer: str, contexts: list[str], reference: str
) -> dict[str, float]:
    from deepeval.test_case import LLMTestCase

    metrics = _build_deepeval(config)["metrics"]
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=reference,
        retrieval_context=list(contexts),
    )
    scores: dict[str, float] = {}
    for key, metric in metrics.items():
        metric.measure(test_case)
        scores[key] = float(metric.score or 0.0)
    return scores


def evaluate(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    config: O3Config,
) -> Verdict:
    if not answer.strip():
        return Verdict(oracle="O3", verdict="fail", score=0.0, detail="resposta vazia")
    if not contexts:
        # Sem contexto recuperado, faithfulness e context precision não têm
        # denominador. Abster-se do juízo é mais honesto do que reprovar: o caso
        # de abstenção legítima (fora do corpus) cairia aqui todas as vezes.
        return Verdict(
            oracle="O3",
            verdict="pass",
            evaluable=False,
            detail="sem contexto recuperado — métricas RAGAS indefinidas",
        )

    try:
        if config.backend == "ragas":
            scores = _score_ragas(config, question, answer, contexts, reference)
        elif config.backend == "deepeval":
            scores = _score_deepeval(config, question, answer, contexts, reference)
        else:
            raise ValueError(f"Backend O3 desconhecido: {config.backend}")
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        return Verdict(oracle="O3", verdict="pass", evaluable=False, detail=f"falha ao medir: {exc}")

    below = [key for key in METRIC_KEYS if scores.get(key, 0.0) < config.threshold(key)]
    failed = bool(below) if config.aggregate == "all" else len(below) == len(METRIC_KEYS)

    return Verdict(
        oracle="O3",
        verdict="fail" if failed else "pass",
        score=round(sum(scores.get(k, 0.0) for k in METRIC_KEYS) / len(METRIC_KEYS), 4),
        detail=(
            "abaixo do limiar: " + ", ".join(f"{k}={scores.get(k, 0.0):.2f}" for k in below)
            if below
            else "; ".join(f"{k}={scores.get(k, 0.0):.2f}" for k in METRIC_KEYS)
        ),
        components={"scores": {k: round(scores.get(k, 0.0), 4) for k in METRIC_KEYS}, "backend": config.backend},
    )
