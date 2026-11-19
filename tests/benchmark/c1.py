"""C1 · Recuperação: o trecho certo chega ao gerador? (plano-benchmark.md, §6)

Métricas sem juiz, calculadas sobre as listas ranqueadas da recuperação. O juiz
de contexto já mostrou que não ordena índices pela recuperação (README: recall
de contexto 1,00 com trechos sem os fatos da referência); a recuperação se
decide aqui. Só biblioteca padrão: roda no venv do backend
(tests/retrieval_diagnostic.py) e no do relatório.

- Documento: identidade pelo nome sem extensão ("artigo.pdf" e "artigo.md" são
  o mesmo artigo).
- Hit@k: o documento-fonte está entre os k primeiros trechos; rank: a posição
  do primeiro trecho dele (o `rank` de sempre do diagnóstico).
- MRR@10: pelo rank do documento entre os documentos distintos dos 10
  primeiros trechos.
- Evidência@k: a evidência do golden chegou ao gerador? Cada item do `context`
  é um fragmento, e dentro dele "[...]" e "..." também separam. O fragmento
  casa se pelo menos TAU dos seus NGRAM-gramas de caracteres estão na UNIÃO dos
  k trechos, com o texto normalizado (NFKC, minúsculas, só letras e dígitos):
  palavra grudada, hifenização, ligadura e página não importam. Cobertura@k é
  a fração dos fragmentos que casam; evidência@k é cobertura 1.
- Precisão@k: fração dos trechos recebidos que são do documento-fonte ou que,
  sozinhos, casam algum fragmento.
- Abstenção pelo limiar: a pergunta não tem nenhum trecho com similaridade
  acima de MIN_SIMILARITY_SCORE; a separação entre perguntas de dentro e de
  fora sai como AUC da similaridade do primeiro trecho.

Toda média sai com IC de 95% por bootstrap por documento: as perguntas de um
mesmo documento erram juntas, e o n efetivo fica entre o de documentos e o de
perguntas (plano, §7).
"""
from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from itertools import accumulate
from typing import Iterable

# Muda quando a definição de alguma métrica muda: diagnósticos de versões diferentes não se comparam.
C1_VERSION = "c1-v1"
NGRAM = 25
TAU = 0.60
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261017

_FRAGMENT_SPLIT = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
_SUFFIXES = (".pdf", ".md")


def doc_stem(name: str | None) -> str | None:
    """Nome do documento sem a extensão de acervo (só .pdf e .md: um ponto no meio do nome fica)."""
    if not name:
        return None
    lower = name.lower()
    for suffix in _SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize(text: str) -> str:
    """NFKC, minúsculas, só letras e dígitos: espaço, pontuação, hífen e hífen suave somem."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower() if ch.isalnum())


def split_fragments(context: Iterable[str] | None) -> list[str]:
    fragments = []
    for item in context or []:
        for piece in _FRAGMENT_SPLIT.split(item):
            if normalize(piece):
                fragments.append(piece.strip())
    return fragments


def grams(normalized: str, n: int = NGRAM) -> set[str]:
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def evidence(fragments: list[str], chunk_texts: list[str], tau: float = TAU) -> dict:
    """Casamento dos fragmentos do golden com os trechos (ver o docstring do módulo).

    Fragmento com menos de NGRAM caracteres normalizados não tem um grama: casa se
    algum trecho o contém inteiro.
    """
    chunk_norms = [normalize(text) for text in chunk_texts]
    chunk_grams = [grams(norm) for norm in chunk_norms]
    union = set().union(*chunk_grams) if chunk_grams else set()
    chunk_has_evidence = [False] * len(chunk_texts)
    union_fractions: list[float] = []
    single_chunk: list[bool] = []
    for fragment in fragments:
        norm = normalize(fragment)
        if len(norm) < NGRAM:
            per_chunk = [1.0 if norm in chunk else 0.0 for chunk in chunk_norms]
            in_union = max(per_chunk, default=0.0)
        else:
            fragment_grams = grams(norm)
            per_chunk = [len(fragment_grams & g) / len(fragment_grams) for g in chunk_grams]
            in_union = len(fragment_grams & union) / len(fragment_grams)
        for i, fraction in enumerate(per_chunk):
            chunk_has_evidence[i] = chunk_has_evidence[i] or fraction >= tau
        union_fractions.append(round(in_union, 3))
        single_chunk.append(max(per_chunk, default=0.0) >= tau)
    matched = sum(fraction >= tau for fraction in union_fractions)
    return {
        "fragments": len(fragments),
        "matched": matched,
        "coverage": matched / len(fragments) if fragments else None,
        "evidence": matched == len(fragments) if fragments else None,
        "union_fractions": union_fractions,
        "single_chunk": single_chunk,
        "chunk_has_evidence": chunk_has_evidence,
    }


def evidence_curve(fragments: list[str], chunk_texts: list[str], tau: float = TAU) -> list[bool] | None:
    """Evidência dos k primeiros trechos, para k = 1..len(chunk_texts), numa passada só.

    O item k − 1 é `evidence(fragments, chunk_texts[:k])["evidence"]`; os gramas de cada trecho saem uma vez.
    Com os tokens acumulados dos mesmos trechos, dá a curva evidência × tokens de contexto do E6.
    """
    if not fragments:
        return None
    norms = [normalize(fragment) for fragment in fragments]
    fragment_grams = [grams(norm) if len(norm) >= NGRAM else None for norm in norms]
    matched: list[set[str]] = [set() for _ in fragments]
    short_found = [False] * len(fragments)
    curve = []
    for text in chunk_texts:
        chunk_norm = normalize(text)
        chunk_grams = grams(chunk_norm)
        for i, fg in enumerate(fragment_grams):
            if fg is None:
                short_found[i] = short_found[i] or norms[i] in chunk_norm
            else:
                matched[i] |= fg & chunk_grams
        curve.append(all(short_found[i] if fg is None else len(matched[i]) / len(fg) >= tau
                         for i, fg in enumerate(fragment_grams)))
    return curve


def _tokens(chunks: list[dict]) -> list[int] | None:
    """Os tokens de cada trecho, se quem recuperou contou (tests/retrieval_diagnostic.py conta com o cl100k)."""
    return [c["tokens"] for c in chunks] if chunks and all("tokens" in c for c in chunks) else None


def _ranked(chunks: list[dict], relevant: list[bool] | None = None, with_text: bool = False) -> list[dict]:
    items = []
    for i, chunk in enumerate(chunks):
        item = {
            "chunk_id": chunk.get("chunk_id"),
            "document": chunk.get("document"),
            "page": chunk.get("page"),
            "similarity_score": chunk.get("similarity_score"),
        }
        if relevant is not None:
            item["relevant"] = relevant[i]
        if chunk.get("found_by"):
            # Busca híbrida (E7): o trecho veio do vetor, do BM25 ou dos dois.
            item["found_by"] = chunk["found_by"]
        if chunk.get("rerank_score") is not None:
            item["rerank_score"] = chunk["rerank_score"]
        if with_text:
            item["text"] = chunk.get("text")
        items.append(item)
    return items


def question_metrics(
    *,
    arm_chunks: list[dict],
    chunks_at_10: list[dict],
    source_document: str | None,
    fragments: list[str],
    top1_similarity: float | None,
    threshold: float = 0.0,
    with_text: bool = False,
) -> dict:
    """C1 de uma pergunta num braço: `arm_chunks` é o que o gerador recebe (top_k e MMR do braço)."""
    blocked = threshold > 0 and (top1_similarity is None or top1_similarity < threshold)
    arm_tokens = _tokens(arm_chunks)
    base = {"n_chunks": len(arm_chunks), "top1_similarity": top1_similarity, "blocked_by_threshold": blocked,
            # Os tokens de contexto que o gerador recebe: o custo do E6 (chunk × top_k com orçamento igual).
            "context_tokens": sum(arm_tokens) if arm_tokens is not None else (0 if not arm_chunks else None),
            # A nota do reranker no primeiro trecho (E8), quando houver.
            "top1_rerank": arm_chunks[0].get("rerank_score") if arm_chunks else None}
    if source_document is None:
        return {**base, "chunks": _ranked(arm_chunks, with_text=with_text)}

    stem = doc_stem(source_document)
    docs = [doc_stem(c["document"]) for c in arm_chunks]
    docs_at_10 = [doc_stem(c["document"]) for c in chunks_at_10]
    rank = next((i for i, doc in enumerate(docs, start=1) if doc == stem), None)
    distinct_at_10 = list(dict.fromkeys(docs_at_10))
    doc_rank_at_10 = distinct_at_10.index(stem) + 1 if stem in distinct_at_10 else None
    found = evidence(fragments, [c["text"] for c in arm_chunks])
    found_at_10 = evidence(fragments, [c["text"] for c in chunks_at_10])
    relevant = [doc == stem or hit for doc, hit in zip(docs, found["chunk_has_evidence"])]
    tokens_at_10 = _tokens(chunks_at_10)
    return {
        **base,
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_k": rank is not None,
        "hit_at_10": stem in docs_at_10,
        "mrr_at_10": 1 / doc_rank_at_10 if doc_rank_at_10 else 0.0,
        "precision_at_k": sum(relevant) / len(relevant) if relevant else None,
        "coverage_at_k": found["coverage"],
        "evidence_at_k": found["evidence"],
        "coverage_at_10": found_at_10["coverage"],
        "fragments": found["fragments"],
        "fragments_single_chunk": sum(found["single_chunk"]),
        "union_fractions": found["union_fractions"],
        "chunks": _ranked(arm_chunks, relevant, with_text=with_text),
        # Para o E5 ("documento morto": tem pergunta e nunca vem nos 10 primeiros trechos dela).
        "documents_at_10": [c.get("document") for c in chunks_at_10],
        # Para o E6: evidência e tokens acumulados dos k primeiros trechos da lista de 10, k = 1..10.
        "evidence_curve": evidence_curve(fragments, [c["text"] for c in chunks_at_10]),
        "tokens_curve": list(accumulate(tokens_at_10)) if tokens_at_10 else None,
    }


# ------------------------------------------------------------------ agregados


def _percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap(
    values: list[float], clusters: list[str], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict | None:
    """Média com IC de 95% (percentil) reamostrando os clusters (documentos), não as perguntas."""
    if not values:
        return None
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        by_cluster[cluster].append(float(value))
    keys = list(by_cluster)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [v for _ in keys for v in by_cluster[rng.choice(keys)]]
        estimates.append(sum(resample) / len(resample))
    estimates.sort()
    return {
        "mean": round(sum(values) / len(values), 4),
        "ci_low": round(_percentile(estimates, 0.025), 4),
        "ci_high": round(_percentile(estimates, 0.975), 4),
        "n": len(values),
        "clusters": len(keys),
    }


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """P(score de uma positiva > score de uma negativa), empate vale meio (Mann-Whitney)."""
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def auc_bootstrap(
    positives: list[float], negatives: list[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict | None:
    point = auc(positives, negatives)
    if point is None:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        auc([rng.choice(positives) for _ in positives], [rng.choice(negatives) for _ in negatives])
        for _ in range(samples)
    )
    return {
        "auc": round(point, 4),
        "ci_low": round(_percentile(estimates, 0.025), 4),
        "ci_high": round(_percentile(estimates, 0.975), 4),
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


IN_CORPUS_METRICS = (
    "hit_at_1", "hit_at_k", "hit_at_10", "mrr_at_10", "precision_at_k", "coverage_at_k", "evidence_at_k",
    "coverage_at_10",
)


def summarize(inside: list[tuple[str, dict]], outside: list[dict], threshold: float = 0.0) -> dict:
    """Agregados de um braço numa língua.

    `inside`: (documento-fonte, métricas) das perguntas cujo documento está no índice; `outside`: métricas das
    perguntas de fora (sem documento-fonte, ou com um documento que o índice não tem).
    """
    summary: dict = {"n_inside": len(inside), "n_outside": len(outside), "threshold": threshold}
    clusters = [doc_stem(source) for source, _ in inside]
    for metric in IN_CORPUS_METRICS:
        pairs = [(m[metric], c) for (_, m), c in zip(inside, clusters) if m.get(metric) is not None]
        summary[metric] = cluster_bootstrap([float(v) for v, _ in pairs], [c for _, c in pairs])
    inside_top1 = [m["top1_similarity"] for _, m in inside if m.get("top1_similarity") is not None]
    outside_top1 = [m["top1_similarity"] for m in outside if m.get("top1_similarity") is not None]
    summary["top1_similarity_auc"] = auc_bootstrap(inside_top1, outside_top1)
    # Com o reranker (E8): a nota do primeiro trecho separa as de fora melhor que o cosseno?
    inside_rerank = [m["top1_rerank"] for _, m in inside if m.get("top1_rerank") is not None]
    outside_rerank = [m["top1_rerank"] for m in outside if m.get("top1_rerank") is not None]
    if inside_rerank and outside_rerank:
        summary["top1_rerank_auc"] = auc_bootstrap(inside_rerank, outside_rerank)
    if threshold > 0:
        summary["false_block_rate"] = (
            sum(m["blocked_by_threshold"] for _, m in inside) / len(inside) if inside else None
        )
        summary["threshold_abstention_rate"] = (
            sum(m["blocked_by_threshold"] for m in outside) / len(outside) if outside else None
        )
    return summary
