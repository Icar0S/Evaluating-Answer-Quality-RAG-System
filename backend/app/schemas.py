"""Schemas Pydantic para a API do RAG."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    document: str
    page: int | None = None
    text: str
    similarity_score: float


class ChatMetadata(BaseModel):
    model: str
    embedding_model: str
    prompt_version: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    top_k: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    metadata: ChatMetadata
    grounded: bool


class IngestResponse(BaseModel):
    status: Literal["ok", "no_documents"]
    documents_processed: int
    chunks_created: int
    collection: str
    duration_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    generation_model: str
    embedding_model: str
    vector_store_documents: int


class ErrorResponse(BaseModel):
    detail: str
