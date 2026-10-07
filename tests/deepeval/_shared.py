"""Peças reaproveitadas por conftest.py, test_rag_quality.py e run_and_export.py.

Fica num módulo `_shared` (prefixo `_` de propósito, para não ser coletado
como suíte de teste) em vez de duplicado em cada arquivo.
"""
from __future__ import annotations

import os

# Precisa vir ANTES de qualquer import de deepeval — a lib lê essas variáveis
# uma vez, na inicialização do seu Settings interno. Setado aqui, não só no
# .env, porque não dá pra confiar que deepeval carrega o .env antes de
# instanciar Settings em todo modo de execução (pytest vs. script direto).
#
# DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE, não DEEPEVAL_DISABLE_TIMEOUTS:
# já tentamos desligar timeout por completo (histórico: run_and_export.py
# rodava em lote via evaluate(), cujo timeout de conjunto — um
# asyncio.wait_for em volta de TODOS os casos — estourava com um juiz local
# lento mesmo com cada chamada individual saudável). Sem timeout nenhum, uma
# ÚNICA chamada travada (visto na prática: qwen3:8b como juiz, que tem modo
# "thinking", entrou num loop de raciocínio e nunca terminou) trava a rodada
# inteira pra sempre, sem chance de recuperação. Um teto generoso mas finito
# (a chamada mais lenta já medida foi ~90s) resolve os dois problemas: dá
# espaço pra juiz lento terminar, e ainda corta uma chamada genuinamente
# travada — que _measure_one (run_and_export.py) transforma num registro de
# erro em vez de derrubar a rodada.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")

import httpx
from deepeval.constants import ProviderSlug as _PS
from deepeval.models import OllamaModel
from deepeval.models.retry_policy import create_retry_decorator as _create_retry_decorator
from deepeval.utils import check_if_multimodal as _check_if_multimodal
from deepeval.utils import convert_to_multi_modal_array as _convert_to_multi_modal_array

from app.config import get_settings
from app.rag import generation, retrieval

EVAL_JUDGE_MODEL = os.environ.get("DEEPEVAL_JUDGE_MODEL", "qwen3:8b")
THRESHOLD = 0.7

_retry_ollama = _create_retry_decorator(_PS.OLLAMA)


class _NoThinkOllamaModel(OllamaModel):
    """OllamaModel com think=False forçado nas duas chamadas de chat.

    deepeval.models.OllamaModel não expõe `think` (nem via generation_kwargs,
    que só alimenta `options`, e o Ollama trata `think` como parâmetro de
    nível superior do chat(), não dentro de `options`). Sem isso, um juiz
    thinking-capable (qwen3) gasta minutos "raciocinando" antes do veredito
    estruturado — medido na prática: 68-321s por chamada com thinking ligado
    contra ~13s desligado numa pergunta comparável. É praticamente uma cópia
    de OllamaModel.generate/a_generate (deepeval não dá um jeito mais fino de
    injetar só esse parâmetro) — se a assinatura desses métodos mudar numa
    atualização do deepeval, esta classe precisa acompanhar.
    """

    @_retry_ollama
    def generate(self, prompt, schema=None):
        chat_model = self.load_model()
        if _check_if_multimodal(prompt):
            prompt = _convert_to_multi_modal_array(prompt)
            messages = self.generate_messages(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]
        response = chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            think=False,
            options={**{"temperature": self.temperature}, **self.generation_kwargs},
        )
        return (
            schema.model_validate_json(response.message.content) if schema else response.message.content,
            0,
        )

    @_retry_ollama
    async def a_generate(self, prompt, schema=None):
        chat_model = self.load_model(async_mode=True)
        if _check_if_multimodal(prompt):
            prompt = _convert_to_multi_modal_array(prompt)
            messages = self.generate_messages(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]
        response = await chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            think=False,
            options={**{"temperature": self.temperature}, **self.generation_kwargs},
        )
        return (
            schema.model_validate_json(response.message.content) if schema else response.message.content,
            0,
        )


def build_judge_model() -> OllamaModel:
    """Juiz LOCAL, via Ollama nativo do deepeval — nunca OpenAI/nuvem.

    Deliberadamente um modelo diferente do GENERATION_MODEL (gemma3:4b): usar
    o mesmo modelo pra gerar e pra julgar a própria resposta é um viés
    conhecido ("self-grading"). Os dois cabem na mesma GPU de 8GB porque o
    Ollama troca os modelos em sequência (geração termina, descarrega, carrega
    o juiz) — não os mantém os dois na VRAM ao mesmo tempo.
    """
    settings = get_settings()
    return _NoThinkOllamaModel(
        model=EVAL_JUDGE_MODEL,
        base_url=settings.ollama_base_url,
        temperature=0,
        # num_ctx: o default do Ollama (2048-4096, depende da versão) e o
        # prompt de métricas como ContextualRelevancy (pergunta + os
        # retrieval_context inteiros + instruções + o JSON de vereditos de
        # saída) estoura isso fácil com TOP_K=4 chunks de ~800 tokens cada —
        # a resposta trunca no meio do JSON em vez de dar erro claro
        # (visto na prática: ValidationError "EOF while parsing an object").
        generation_kwargs={"num_ctx": 8192},
    )


def stack_status() -> str | None:
    """None se a stack real (Ollama + vector store) estiver pronta pra avaliação; senão, o motivo do skip."""
    settings = get_settings()
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        resp.raise_for_status()
    except httpx.HTTPError:
        return f"Ollama inacessível em {settings.ollama_base_url}"

    tags = {m["name"] for m in resp.json().get("models", [])}

    def _has(model: str) -> bool:
        return model in tags or any(t.startswith(f"{model}:") for t in tags)

    for required in (settings.generation_model, settings.embedding_model, EVAL_JUDGE_MODEL):
        if not _has(required):
            return f"Modelo '{required}' não está baixado no Ollama local (`ollama pull {required}`)"

    try:
        from app.rag import vector_store

        if vector_store.collection_count() == 0:
            return "Vector store vazio — rode `python scripts/ingest_documents.py` antes"
    except Exception as exc:  # vector store inacessível/corrompido — trate como não-pronto, não como erro de coleta
        return f"Vector store inacessível: {exc}"

    return None


def run_pipeline(question: str, top_k: int = 4) -> tuple[str, list[str]]:
    """Chama o mesmo pipeline que POST /chat usa (backend/app/main.py), in-process."""
    chunks, _retrieval_latency_ms = retrieval.retrieve(question, top_k)
    result = generation.generate_answer(question, chunks)
    return result["answer"], [c["text"] for c in chunks]
