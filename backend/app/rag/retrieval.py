"""Recuperação de contexto: embeda a pergunta e busca os top-k chunks mais similares."""
from __future__ import annotations

import time

from app.config import get_settings
from app.rag.ingestion import embed_texts
from app.rag import vector_store


def retrieve(question: str, top_k: int | None = None) -> tuple[list[dict], float]:
    """Retorna (lista de chunks com score, latência em ms)."""
    settings = get_settings()
    k = top_k or settings.top_k
    start = time.perf_counter()

    [query_embedding] = embed_texts([question])
    result = vector_store.query(query_embedding, top_k=k)

    ids = result["ids"][0] if result["ids"] else []
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    chunks = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # Espaço "cosine" no Chroma: distância = 1 - similaridade.
        similarity = max(0.0, 1.0 - distance)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document": meta.get("document", "unknown"),
                "page": meta.get("page"),
                "text": text,
                "similarity_score": round(similarity, 4),
            }
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return chunks, latency_ms
