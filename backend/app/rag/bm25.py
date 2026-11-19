"""BM25 em Python puro, para a busca híbrida (RETRIEVAL_HYBRID; E7 do plano-benchmark.md).

Sem dependência nova: os quatro requirements que importam o pipeline em processo
(backend, eval, ragas, mutation) continuam iguais. O índice é montado com o texto
de todos os chunks da coleção (vector_store.get_all) e fica em cache por coleção
e contagem de chunks: reingerir a versão muda a contagem e refaz o índice.

Tokens: letras e dígitos, em minúsculas e sem acento, sem stemming nem lista de
palavras vazias. A sigla e o nome próprio ("RAGAS", "Defects4J", "MetaRAG") são
o que a busca vetorial não acha e a lexical acha; é a hipótese do E7.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

from app.config import get_settings
from app.rag import vector_store

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"[0-9a-z]+")


def tokenize(text: str) -> list[str]:
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))
    return _TOKEN.findall(plain)


class BM25Index:
    """Okapi BM25 (k1 = 1,5; b = 0,75), com o idf que não fica negativo (o do Lucene) e índice invertido."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = K1, b: float = B) -> None:
        self.ids, self.k1, self.b = list(ids), k1, b
        counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(c.values()) for c in counts]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(counts):
            for term, frequency in c.items():
                self.postings[term].append((i, frequency))
        n = len(counts)
        self.idf = {term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Os k chunks de maior escore, com o escore; empate fica com a ordem da coleção."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            for i, frequency in self.postings.get(term, ()):
                saturation = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.average or 1.0))
                scores[i] += self.idf[term] * frequency * (self.k1 + 1) / (frequency + saturation)
        best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.ids[i], score) for i, score in best]


_CACHE: dict[tuple[str, int], BM25Index] = {}


def index_for_collection() -> BM25Index:
    """O índice BM25 da coleção em uso, montado na primeira busca e guardado por (coleção, contagem)."""
    key = (get_settings().vector_store_collection, vector_store.collection_count())
    if key not in _CACHE:
        everything = vector_store.get_all()
        _CACHE[key] = BM25Index(everything["ids"], everything["documents"])
    return _CACHE[key]
