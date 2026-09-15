"""Ingestão de PDFs: extração de texto, chunking e indexação vetorial.

Chunking é feito por página (facilita citar a página como fonte) usando
contagem de tokens via tiktoken (cl100k_base como aproximação genérica,
independente do tokenizer real do modelo local).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import tiktoken

from app.config import get_settings
from app.logging_utils import logger
from app.rag import vector_store

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    page: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Retorna [(numero_da_pagina_1_indexed, texto)] para um PDF."""
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    boundary_offset_tokens: int = 0,
) -> list[str]:
    """Divide o texto em janelas de tokens com sobreposição.

    boundary_offset_tokens desloca todas as fronteiras: o primeiro chunk fica
    encurtado nesse tanto e os seguintes passam a cortar em posições diferentes,
    sem perder nenhum token. Serve ao operador de mutação K4, que verifica o que
    acontece quando um corte cai no meio de uma regra.
    """
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    if overlap_tokens >= chunk_size_tokens:
        overlap_tokens = max(chunk_size_tokens // 4, 0)

    offset = boundary_offset_tokens % chunk_size_tokens if chunk_size_tokens > 0 else 0

    chunks: list[str] = []
    start = 0
    first_size = chunk_size_tokens - offset if offset else chunk_size_tokens
    size = max(first_size, 1)
    while start < len(tokens):
        end = start + size
        chunks.append(_encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
        size = chunk_size_tokens
    return chunks


def build_chunk_records(
    pdf_path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int,
    boundary_offset_tokens: int = 0,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    doc_name = pdf_path.name
    for page_num, page_text in extract_pdf_pages(pdf_path):
        page_chunks = chunk_text(page_text, chunk_size_tokens, overlap_tokens, boundary_offset_tokens)
        for idx, chunk in enumerate(page_chunks):
            chunk_id = f"{pdf_path.stem}::p{page_num}::c{idx}"
            records.append(ChunkRecord(chunk_id=chunk_id, document=doc_name, page=page_num, text=chunk))
    return records


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Gera embeddings sempre via Ollama LOCAL, em lote quando /api/embed está disponível.

    Deliberadamente ignora o provider ativo: a API remota (smartdatatest, ver
    app/providers.py) não expõe rota de embeddings, e mesmo que expusesse,
    misturar espaços de embedding entre chamadas corromperia a busca por
    similaridade no vector store. Só a geração (app/rag/generation.py) muda com
    o provider ativo.
    """
    if not texts:
        return []
    settings = get_settings()
    base_url = settings.ollama_base_url
    # `model` explícito existe para quem precisa de um espaço de embedding fixo,
    # independente da configuração corrente — é o caso do oráculo O1 do estudo de
    # mutação, que não pode ser afetado pelo operador E1 (troca de embedding).
    embedding_model = model or settings.embedding_model
    url = f"{base_url}/api/embed"
    with httpx.Client(timeout=settings.generation_timeout_seconds) as client:
        try:
            resp = client.post(url, json={"model": embedding_model, "input": texts})
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except (httpx.HTTPError, ValueError, KeyError):
            logger.warning("Batch embedding falhou, caindo para embeddings um a um")

        # Fallback: endpoint legado, um texto por chamada.
        results: list[list[float]] = []
        legacy_url = f"{base_url}/api/embeddings"
        for text in texts:
            resp = client.post(legacy_url, json={"model": embedding_model, "prompt": text})
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results


def ingest_documents(source_dir: Path | None = None, reset: bool = True) -> dict:
    """Reprocessa todos os PDFs de source_dir (ou data/source_pdfs por padrão)."""
    settings = get_settings()
    start = time.perf_counter()
    pdf_dir = source_dir or settings.source_pdfs_path
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        return {
            "status": "no_documents",
            "documents_processed": 0,
            "chunks_created": 0,
            "collection": settings.vector_store_collection,
            "duration_ms": (time.perf_counter() - start) * 1000,
        }

    all_records: list[ChunkRecord] = []
    for pdf_path in pdf_files:
        try:
            records = build_chunk_records(
                pdf_path,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                settings.chunk_boundary_offset_tokens,
            )
            all_records.extend(records)
            logger.info(f"Ingerido {pdf_path.name}: {len(records)} chunks")
        except Exception as exc:
            logger.error(f"Falha ao processar {pdf_path.name}: {exc}")

    first_batch = True
    chunks_created = 0
    batch_size = 64
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i : i + batch_size]
        embeddings = embed_texts([r.text for r in batch])
        vector_store.add_chunks(
            ids=[r.chunk_id for r in batch],
            embeddings=embeddings,
            documents=[r.text for r in batch],
            metadatas=[{"document": r.document, "page": r.page} for r in batch],
            reset=reset and first_batch,
        )
        first_batch = False
        chunks_created += len(batch)

    return {
        "status": "ok",
        "documents_processed": len(pdf_files),
        "chunks_created": chunks_created,
        "collection": settings.vector_store_collection,
        "duration_ms": (time.perf_counter() - start) * 1000,
    }
