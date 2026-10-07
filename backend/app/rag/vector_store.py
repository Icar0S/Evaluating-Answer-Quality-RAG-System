"""Wrapper fino sobre ChromaDB para persistência local do índice vetorial."""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings = get_settings()
        settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.vector_store_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def reset_client() -> None:
    """Descarta o cliente em cache para que o proximo acesso releia o caminho da configuracao.

    Necessario quando VECTOR_STORE_DIR muda dentro do mesmo processo -- o caso do
    harness de mutacao, que aponta o estudo para um indice proprio em
    tests/mutation/work/ para nunca tocar em data/vector_store.
    """
    global _client
    _client = None


def get_collection(reset: bool = False):
    """Retorna (criando se necessário) a coleção usada pelo RAG.

    reset=True apaga e recria a coleção, usado por /ingest para reprocessar do zero.
    """
    settings = get_settings()
    client = get_client()
    if reset:
        try:
            client.delete_collection(settings.vector_store_collection)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=settings.vector_store_collection,
        metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def add_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    reset: bool = False,
) -> None:
    collection = get_collection(reset=reset)
    # Chroma tem limite prático de batch; envia em lotes de 100.
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def query(embedding: list[float], top_k: int, include_embeddings: bool = False) -> dict[str, Any]:
    """Busca os top_k chunks mais proximos.

    include_embeddings=True devolve tambem os vetores dos candidatos, necessarios
    para reordenacao por MMR em app/rag/retrieval.py (o Chroma nao devolve
    embeddings por padrao).
    """
    empty: dict[str, Any] = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    if include_embeddings:
        empty["embeddings"] = [[]]

    collection = get_collection()
    if collection.count() == 0:
        return empty

    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include = include + ["embeddings"]
    return collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=include,
    )
