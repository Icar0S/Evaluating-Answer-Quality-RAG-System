"""Recuperação de contexto: embeda a pergunta e busca os top-k chunks mais similares.

Três etapas configuráveis, nesta ordem:

1. busca vetorial (top-k, ou top-k × RETRIEVAL_FETCH_MULTIPLIER quando MMR está ligado);
2. corte por MIN_SIMILARITY_SCORE — abaixo do piso o chunk não entra no contexto,
   e um contexto vazio faz a geração cair na frase de abstenção em vez de
   responder apoiada em trecho irrelevante;
3. seleção final: ordem de similaridade pura, ou MMR (relevância × diversidade)
   quando RETRIEVAL_USE_MMR está ligado.

Os três pontos são alvo direto dos operadores de mutação da camada de
recuperação (R1–R4, ver tests/mutation/operators/catalog.yaml).
"""
from __future__ import annotations

import math
import time

from app.config import get_settings
from app.rag.ingestion import embed_texts
from app.rag import vector_store


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(
    query_embedding: list[float],
    candidates: list[dict],
    embeddings: list[list[float]],
    k: int,
    lambda_mult: float,
) -> list[dict]:
    """Maximal Marginal Relevance: maximiza λ·relevância − (1−λ)·redundância.

    Implementação gulosa clássica (Carbonell & Goldstein, 1998). A relevância vem
    do similarity_score já calculado na busca; a redundância é a similaridade
    máxima contra os chunks já escolhidos. Empates preservam a ordem original da
    busca, o que mantém a seleção determinística.
    """
    if not candidates:
        return []

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        best_idx = remaining[0]
        best_score = -math.inf
        for idx in remaining:
            relevance = candidates[idx]["similarity_score"]
            redundancy = 0.0
            if selected and embeddings[idx]:
                redundancy = max(
                    _cosine_similarity(embeddings[idx], embeddings[s]) for s in selected if embeddings[s]
                )
            score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def retrieve(question: str, top_k: int | None = None) -> tuple[list[dict], float]:
    """Retorna (lista de chunks com score, latência em ms)."""
    settings = get_settings()
    k = top_k or settings.top_k
    use_mmr = settings.retrieval_use_mmr
    fetch_k = max(k * settings.retrieval_fetch_multiplier, k) if use_mmr else k
    start = time.perf_counter()

    [query_embedding] = embed_texts([question])
    if use_mmr:
        result = vector_store.query(query_embedding, top_k=fetch_k, include_embeddings=True)
    else:
        result = vector_store.query(query_embedding, fetch_k)

    ids = result["ids"][0] if result["ids"] else []
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []
    raw_embeddings = result.get("embeddings") or [[]]
    candidate_embeddings = list(raw_embeddings[0]) if len(raw_embeddings) else []

    candidates = []
    kept_embeddings = []
    for i, (chunk_id, text, meta, distance) in enumerate(zip(ids, documents, metadatas, distances)):
        # Espaço "cosine" no Chroma: distância = 1 - similaridade.
        similarity = max(0.0, 1.0 - distance)
        if similarity < settings.min_similarity_score:
            continue
        candidates.append(
            {
                "chunk_id": chunk_id,
                "document": meta.get("document", "unknown"),
                "page": meta.get("page"),
                "text": text,
                "similarity_score": round(similarity, 4),
            }
        )
        kept_embeddings.append(list(candidate_embeddings[i]) if i < len(candidate_embeddings) else [])

    if use_mmr:
        chunks = _mmr_select(query_embedding, candidates, kept_embeddings, k, settings.retrieval_mmr_lambda)
    else:
        chunks = candidates[:k]

    latency_ms = (time.perf_counter() - start) * 1000
    return chunks, latency_ms
