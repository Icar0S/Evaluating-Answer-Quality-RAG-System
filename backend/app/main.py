"""API FastAPI do assistente RAG especialista em teste de sistemas LLM/RAG."""
from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_utils import log_interaction, logger, new_interaction_id
from app.rag import generation, retrieval, vector_store
from app.rag.ingestion import ingest_documents
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SourceChunk,
)

settings = get_settings()

app = FastAPI(
    title="RAG Test Specialist API",
    description="Assistente RAG local especializado em teste de sistemas LLM/RAG.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_reachable = True
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
    except httpx.HTTPError:
        ollama_reachable = False

    doc_count = await asyncio.to_thread(vector_store.collection_count)

    return HealthResponse(
        status="ok" if ollama_reachable else "degraded",
        ollama_reachable=ollama_reachable,
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        vector_store_documents=doc_count,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    try:
        result = await asyncio.to_thread(ingest_documents)
    except Exception as exc:
        logger.error(f"Falha na ingestão: {exc}")
        raise HTTPException(status_code=500, detail=f"Falha ao ingerir documentos: {exc}") from exc
    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    interaction_id = new_interaction_id()
    top_k = request.top_k or settings.top_k

    try:
        chunks, retrieval_latency_ms = await asyncio.to_thread(retrieval.retrieve, request.question, top_k)
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível conectar ao Ollama. Verifique se o serviço está em execução (ollama serve).",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao gerar embeddings: {exc}") from exc

    try:
        result = await asyncio.to_thread(generation.generate_answer, request.question, chunks)
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível conectar ao Ollama. Verifique se o serviço está em execução (ollama serve).",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Tempo limite excedido ao gerar a resposta.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao gerar resposta: {exc}") from exc

    total_latency_ms = retrieval_latency_ms + result["latency_ms"]

    metadata = ChatMetadata(
        model=settings.generation_model,
        embedding_model=settings.embedding_model,
        prompt_version=settings.prompt_version,
        latency_ms=round(total_latency_ms, 2),
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(result["latency_ms"], 2),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        top_k=top_k,
    )

    response = ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**c) for c in chunks],
        metadata=metadata,
        grounded=result["grounded"],
    )

    log_interaction(
        {
            "interaction_id": interaction_id,
            "question": request.question,
            "answer": result["answer"],
            "grounded": result["grounded"],
            "sources": chunks,
            "metadata": metadata.model_dump(),
        }
    )

    return response
