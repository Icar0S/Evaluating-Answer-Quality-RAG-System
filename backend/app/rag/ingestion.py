"""Ingestão de documentos (PDF e Markdown): extração de texto, chunking e indexação vetorial.

PDF: chunking dentro de cada página (facilita citar a página como fonte).
Markdown: o documento inteiro, sem página. Os dois usam o mesmo chunker por
tokens (tiktoken cl100k_base como aproximação genérica, independente do
tokenizer real do modelo local) e os mesmos parâmetros: comparar índices de
formatos diferentes (config.RAG_INDICES) só faz sentido com o mesmo corte.

Cada ingestão grava um manifesto da versão do índice (arquivos com hash,
parâmetros, modelo de embedding) em <vector_store>/manifests/<coleção>.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import tiktoken

from app.config import get_settings
from app.logging_utils import logger
from app.rag import vector_store

_encoding = tiktoken.get_encoding("cl100k_base")

SUPPORTED_SUFFIXES = (".pdf", ".md")

# Versão da conversão Markdown -> texto, gravada no manifesto. Só a SINTAXE de
# tabela vira texto: linhas separadoras somem e as células não vazias de cada
# linha são unidas por espaço. Nada de "consertar" o conteúdo (hifenização,
# carimbos do arXiv invertidos que a conversão PDF->MD deixou): isso seria
# tratar um formato melhor que o outro na comparação com o índice de PDFs. Nos
# 22 artigos de data/new_papers_marks, tira ~11% dos tokens (pipes e células
# vazias de layout) e devolve títulos que a conversão quebrou em células.
MD_NORMALIZATION = "v1: sintaxe de tabela vira texto; conteúdo intacto"
_MD_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    page: int | None
    text: str


def normalize_markdown(text: str) -> str:
    """Markdown -> texto para embedding (ver MD_NORMALIZATION)."""
    lines = []
    for line in text.splitlines():
        if _MD_TABLE_SEPARATOR.match(line):
            continue
        if line.lstrip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            line = " ".join(cell for cell in cells if cell)
            if not line:
                continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


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
    if pdf_path.suffix.lower() == ".md":
        text = normalize_markdown(pdf_path.read_text(encoding="utf-8", errors="replace"))
        for idx, chunk in enumerate(chunk_text(text, chunk_size_tokens, overlap_tokens, boundary_offset_tokens)):
            records.append(ChunkRecord(chunk_id=f"{pdf_path.stem}::c{idx}", document=doc_name, page=None, text=chunk))
        return records
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(source_dir: Path, documents: list[dict], chunks: int, duration_ms: float) -> Path:
    """Registra o que entrou nesta versão do índice: arquivos (com hash), parâmetros e modelo.

    É o que permite dizer, meses depois, contra QUAL acervo uma avaliação rodou.
    """
    settings = get_settings()
    manifest = {
        "index": settings.index_name,
        "collection": settings.vector_store_collection,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "chunk_boundary_offset_tokens": settings.chunk_boundary_offset_tokens,
        "md_normalization": MD_NORMALIZATION if any(d["format"] == "md" for d in documents) else None,
        "n_documents": len(documents),
        "n_chunks": chunks,
        "duration_ms": round(duration_ms),
        "documents": documents,
    }
    path = settings.vector_store_path / "manifests" / f"{settings.vector_store_collection}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ingest_documents(source_dir: Path | None = None, reset: bool = True) -> dict:
    """Reprocessa todos os PDFs e Markdown de source_dir (ou da pasta do índice ativo)."""
    settings = get_settings()
    start = time.perf_counter()
    doc_dir = source_dir or settings.source_pdfs_path
    doc_files = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES) if doc_dir.is_dir() else []

    if not doc_files:
        return {
            "status": "no_documents",
            "documents_processed": 0,
            "chunks_created": 0,
            "collection": settings.vector_store_collection,
            "duration_ms": (time.perf_counter() - start) * 1000,
        }

    all_records: list[ChunkRecord] = []
    manifest_docs: list[dict] = []
    for doc_path in doc_files:
        try:
            records = build_chunk_records(
                doc_path,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                settings.chunk_boundary_offset_tokens,
            )
            all_records.extend(records)
            manifest_docs.append(
                {
                    "file": doc_path.name,
                    "format": doc_path.suffix.lower().lstrip("."),
                    "bytes": doc_path.stat().st_size,
                    "sha256": _sha256(doc_path),
                    "chunks": len(records),
                }
            )
            logger.info(f"Ingerido {doc_path.name}: {len(records)} chunks")
        except Exception as exc:
            logger.error(f"Falha ao processar {doc_path.name}: {exc}")

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
            # Markdown não tem página: a chave fica de fora (o Chroma não aceita
            # metadado None) e a recuperação devolve page=None.
            metadatas=[
                {"document": r.document, "page": r.page} if r.page is not None else {"document": r.document}
                for r in batch
            ],
            reset=reset and first_batch,
        )
        first_batch = False
        chunks_created += len(batch)

    duration_ms = (time.perf_counter() - start) * 1000
    manifest = write_manifest(doc_dir, manifest_docs, chunks_created, duration_ms)
    return {
        "status": "ok",
        "documents_processed": len(manifest_docs),
        "chunks_created": chunks_created,
        "collection": settings.vector_store_collection,
        "index": settings.index_name,
        "manifest": str(manifest),
        "duration_ms": duration_ms,
    }
