"""Leitura da coleção inteira (vector_store.get_all), sem subir o Chroma."""
from __future__ import annotations

from app.rag import vector_store


class FakeCollection:
    """O pedaço da API do Chroma que o get_all usa: get(include, limit, offset)."""

    def __init__(self, n: int) -> None:
        self.ids = [f"doc.pdf::p{i}::c0" for i in range(n)]
        self.calls: list[tuple[int, int]] = []

    def get(self, include: list[str], limit: int, offset: int) -> dict:
        self.calls.append((limit, offset))
        ids = self.ids[offset : offset + limit]
        batch = {
            "ids": ids,
            "documents": [f"texto {i}" for i in ids],
            "metadatas": [{"document": "doc.pdf", "page": n} for n, _ in enumerate(ids, start=offset)],
        }
        if "embeddings" in include:
            batch["embeddings"] = [(0.1, 0.2) for _ in ids]  # o Chroma devolve arrays, não listas
        return batch


def test_get_all_le_a_colecao_inteira_em_lotes(monkeypatch):
    fake = FakeCollection(5)
    monkeypatch.setattr(vector_store, "get_collection", lambda reset=False: fake)

    result = vector_store.get_all(batch_size=2)

    assert result["ids"] == fake.ids
    assert [m["page"] for m in result["metadatas"]] == [0, 1, 2, 3, 4]
    assert "embeddings" not in result
    assert fake.calls == [(2, 0), (2, 2), (2, 4)]


def test_get_all_com_vetores_e_colecao_vazia(monkeypatch):
    fake = FakeCollection(3)
    monkeypatch.setattr(vector_store, "get_collection", lambda reset=False: fake)

    result = vector_store.get_all(include_embeddings=True, batch_size=3)

    assert result["embeddings"] == [[0.1, 0.2]] * 3
    # Lote cheio: confere se há mais, e a coleção acabou.
    assert fake.calls == [(3, 0), (3, 3)]
    monkeypatch.setattr(vector_store, "get_collection", lambda reset=False: FakeCollection(0))
    assert vector_store.get_all() == {"ids": [], "documents": [], "metadatas": []}
