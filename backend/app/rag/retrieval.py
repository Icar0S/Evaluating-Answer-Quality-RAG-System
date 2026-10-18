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

Com RETRIEVAL_QUERY_TRANSLATION, a busca embeda a pergunta traduzida para o
inglês (app/rag/translation.py); quem chama continua com a original.

Com RETRIEVAL_HYBRID, a etapa 1 junta a busca vetorial e a lexical (BM25,
app/rag/bm25.py) por RRF, cada uma com top-k × RETRIEVAL_FETCH_MULTIPLIER
candidatos; o corte e a seleção seguem iguais (E7 do plano-benchmark.md).

Com RETRIEVAL_RERANK, a etapa 3 é o LLM: ele dá nota a cada um dos
RETRIEVAL_RERANK_CANDIDATES candidatos (app/rag/rerank.py), e os k de maior nota
seguem (E8 do plano-benchmark.md).
"""
from __future__ import annotations

import math
import time

from app.config import get_settings
from app.rag.ingestion import embed_texts
from app.rag import bm25, rerank, translation, vector_store


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


def _fuse_with_bm25(
    query_text: str,
    query_embedding: list[float],
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    similarities: list[float],
    embeddings: list,
    fetch_k: int,
    rrf_k: int,
) -> tuple[list[str], list[str], list[dict], list[float], list, dict[str, str]]:
    """Busca híbrida (E7): a lista do vetor e a do BM25 fundidas por RRF, a soma de 1 / (rrf_k + posição).

    O trecho que só o BM25 achou ganha o cosseno com a pergunta, pelo vetor guardado: o limiar e o MMR usam o
    similarity_score. Empate no RRF fica com o maior cosseno. Devolve as listas na ordem fundida e de onde cada
    trecho veio (vetor, bm25 ou ambos).
    """
    lexical = [chunk_id for chunk_id, _ in bm25.index_for_collection().search(query_text, fetch_k)]
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)}
    lexical_rank = {chunk_id: rank for rank, chunk_id in enumerate(lexical, start=1)}
    pool = {
        chunk_id: (text, meta, similarity, embeddings[i] if i < len(embeddings) else [])
        for i, (chunk_id, text, meta, similarity) in enumerate(zip(ids, documents, metadatas, similarities))
    }
    extra = vector_store.get_by_ids([c for c in lexical if c not in pool], include_embeddings=True)
    for chunk_id, text, meta, vector in zip(extra["ids"], extra["documents"], extra["metadatas"], extra["embeddings"]):
        pool[chunk_id] = (text, meta, max(0.0, _cosine_similarity(query_embedding, vector)), vector)

    def fused(chunk_id: str) -> float:
        return sum(1 / (rrf_k + ranks[chunk_id]) for ranks in (vector_rank, lexical_rank) if chunk_id in ranks)

    order = sorted(pool, key=lambda c: (-fused(c), -pool[c][2], c))
    found_by = {c: "ambos" if c in vector_rank and c in lexical_rank else "vetor" if c in vector_rank else "bm25"
                for c in order}
    return (order, [pool[c][0] for c in order], [pool[c][1] for c in order], [pool[c][2] for c in order],
            [pool[c][3] for c in order], found_by)


def retrieve(question: str, top_k: int | None = None) -> tuple[list[dict], float]:
    """Retorna (lista de chunks com score, latência em ms)."""
    settings = get_settings()
    k = top_k or settings.top_k
    use_mmr = settings.retrieval_use_mmr
    use_hybrid = settings.retrieval_hybrid
    use_rerank = settings.retrieval_rerank
    if use_rerank:
        # Um número fixo de candidatos (20, os 4 × 5 do plano): a lista de 10 do diagnóstico reusa as notas.
        fetch_k = max(settings.retrieval_rerank_candidates, k)
    elif use_mmr or use_hybrid:
        fetch_k = max(k * settings.retrieval_fetch_multiplier, k)
    else:
        fetch_k = k
    if use_hybrid:
        # Fora do tempo medido: o índice BM25 é montado uma vez por coleção, e não a cada pergunta.
        bm25.index_for_collection()
    start = time.perf_counter()

    # Dentro do tempo medido: a tradução feita na hora é custo da busca.
    query_text = translation.query_for_retrieval(question)
    [query_embedding] = embed_texts([query_text])
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
    # Espaço "cosine" no Chroma: distância = 1 - similaridade.
    similarities = [max(0.0, 1.0 - distance) for distance in distances]
    found_by: dict[str, str] = {}
    if use_hybrid:
        ids, documents, metadatas, similarities, candidate_embeddings, found_by = _fuse_with_bm25(
            query_text, query_embedding, ids, documents, metadatas, similarities, candidate_embeddings, fetch_k,
            settings.retrieval_rrf_k,
        )

    candidates = []
    kept_embeddings = []
    for i, (chunk_id, text, meta, similarity) in enumerate(zip(ids, documents, metadatas, similarities)):
        if similarity < settings.min_similarity_score:
            continue
        candidate = {
            "chunk_id": chunk_id,
            "document": meta.get("document", "unknown"),
            "page": meta.get("page"),
            # Só o chunker por seção grava a seção (E2); nos outros índices, None.
            "section": meta.get("section"),
            "text": text,
            "similarity_score": round(similarity, 4),
        }
        if found_by:
            # Na busca híbrida, de qual lista o trecho veio (o E7 mostra os que só o BM25 trouxe).
            candidate["found_by"] = found_by[chunk_id]
        candidates.append(candidate)
        kept_embeddings.append(list(candidate_embeddings[i]) if i < len(candidate_embeddings) else [])

    if use_rerank:
        # O LLM lê a pergunta original e cada candidato, e a nota decide os k (E8); com ele, o MMR não roda.
        chunks = rerank.rerank(question, candidates)[:k]
    elif use_mmr:
        chunks = _mmr_select(query_embedding, candidates, kept_embeddings, k, settings.retrieval_mmr_lambda)
    else:
        chunks = candidates[:k]

    latency_ms = (time.perf_counter() - start) * 1000
    return chunks, latency_ms
