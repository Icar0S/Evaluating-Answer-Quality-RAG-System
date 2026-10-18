"""E5 · Concentração das fontes, o 80/20 (plano-benchmark.md, §8, E5).

Não é um braço: é uma análise sobre o diagnóstico de recuperação de um braço
numa língua (tests/retrieval_diagnostic.py). Só entram perguntas cujo
documento-fonte está no índice; as de fora são da abstenção. Cada pergunta tem
massa 1, e n_q é o número de trechos que ela de fato recebeu (o limiar pode
devolver menos que k).

| Métrica | Definição |
|---|---|
| participação observada o_d | média, sobre as perguntas, de n_{q,d} / n_q |
| participação esperada e_d | fração das perguntas cuja fonte é d (o que uma recuperação perfeita daria) |
| massa fora da fonte | média de 1 − n_{q,fonte} / n_q: a medida principal de viés |
| TVD | ½ · Σ |o_d − e_d| |
| N80 | menor número de documentos cuja participação acumulada chega a 80%, sobre N do índice |
| teste do enunciado | fração das perguntas com TODAS as fontes no top 20% dos documentos, e com pelo menos uma; também para e_d |
| Gini | de o e de e sobre os N documentos do índice (com os zeros), e o Gini@1 do primeiro trecho |
| ímã | documento com pelo menos 3 aparições indevidas (pergunta de outro documento) acima do acaso: fração dos chunks dele × k × Q |
| morto | documento com pergunta que nunca aparece nos 10 primeiros trechos das perguntas dele |
| chunk ímã | a mesma conta por chunk_id, marcando os de bibliografia |

A curva "esperada" separa o que vem do desenho das perguntas (algumas fontes
têm mais perguntas) do que é comportamento do sistema.

Concentração × qualidade (com --respostas e --ragas, a C2 de um braço): por
documento, a participação e o excesso de ímã ao lado da taxa de reprovação das
respostas que o usam (reprovada = o RAGAS não aprova todas as dimensões, a
regra do relatório) e da nota média; e a nota das respostas apoiadas só no top
20% contra as demais, com IC por bootstrap dos documentos-fonte. Os trechos
são os que o gerador recebeu nessa rodada, gravados nas respostas. A
atribuição errada precisa de citação e fica para os braços com citação (E3).

Uso, da raiz (o --bibliografia lê o índice e precisa do venv do backend):

    backend\\.venv\\Scripts\\python.exe -m tests.benchmark.e5 diag.json --indice pdf-v1 --lingua pt --bibliografia
    backend\\.venv\\Scripts\\python.exe -m tests.benchmark.e5 diag.json --indice pdf-v1 \\
        --respostas tests\\benchmark\\out\\E0_<data>\\respostas_A1.json --ragas tests\\benchmark\\out\\E0_<data>\\ragas_A1.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tests.benchmark import c1

ROOT = Path(__file__).resolve().parents[2]
TOP_SHARE = 0.20
N80_SHARE = 0.80
MAGNET_EXCESS = 3
# Chunk de bibliografia: metade ou mais dos gramas dele estão no bloco de referências do documento.
BIBLIOGRAPHY_MIN = 0.5


def questions_from_diagnostic(diagnostic: dict, index: str, lang: str) -> list[dict]:
    """As perguntas de dentro do índice, no formato da análise: fonte, documentos e chunks recebidos."""
    questions = []
    for row in diagnostic["rows"]:
        metrics = row["c1"].get(index, {}).get(lang)
        if not metrics or not metrics.get("in_index"):
            continue
        questions.append({
            "name": row["name"],
            "source": c1.doc_stem(row["source_document"]),
            "documents": [c1.doc_stem(c["document"]) for c in metrics["chunks"]],
            "chunk_ids": [c["chunk_id"] for c in metrics["chunks"]],
            "documents_at_10": [c1.doc_stem(d) for d in metrics.get("documents_at_10", [])],
        })
    return questions


def observed_shares(questions: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    received = [q for q in questions if q["documents"]]
    for q in received:
        for doc, n in Counter(q["documents"]).items():
            totals[doc] += n / len(q["documents"])
    return {doc: total / len(received) for doc, total in totals.items()} if received else {}


def expected_shares(questions: list[dict]) -> dict[str, float]:
    counts = Counter(q["source"] for q in questions)
    return {doc: n / len(questions) for doc, n in counts.items()} if questions else {}


def off_source_mass(questions: list[dict]) -> float | None:
    received = [q for q in questions if q["documents"]]
    if not received:
        return None
    return sum(1 - q["documents"].count(q["source"]) / len(q["documents"]) for q in received) / len(received)


def total_variation(observed: dict[str, float], expected: dict[str, float]) -> float:
    return 0.5 * sum(abs(observed.get(d, 0.0) - expected.get(d, 0.0)) for d in set(observed) | set(expected))


def n80(shares: dict[str, float], share: float = N80_SHARE) -> int:
    cumulative, count = 0.0, 0
    for value in sorted(shares.values(), reverse=True):
        cumulative += value
        count += 1
        if cumulative >= share - 1e-9:
            break
    return count


def gini(values: list[float]) -> float | None:
    """Gini sobre todos os documentos do índice (os que não aparecem entram com zero)."""
    ordered = sorted(values)
    n, total = len(ordered), sum(ordered)
    if n == 0 or total == 0:
        return None
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def _padded(shares: dict[str, float], n_documents: int) -> list[float]:
    return list(shares.values()) + [0.0] * max(0, n_documents - len(shares))


def lorenz(shares: dict[str, float], n_documents: int) -> list[tuple[float, float]]:
    """Pontos (fração dos documentos, fração acumulada da participação), do menor para o maior."""
    ordered = sorted(_padded(shares, n_documents))
    total = sum(ordered) or 1.0
    points, cumulative = [(0.0, 0.0)], 0.0
    for i, value in enumerate(ordered, start=1):
        cumulative += value
        points.append((round(i / len(ordered), 4), round(cumulative / total, 4)))
    return points


def top_documents(shares: dict[str, float], n_documents: int, share: float = TOP_SHARE) -> set[str]:
    size = max(1, math.ceil(share * n_documents))
    return {doc for doc, _ in sorted(shares.items(), key=lambda item: (-item[1], item[0]))[:size]}


def statement_test(questions: list[dict], top: set[str], expected: bool = False) -> dict:
    """O enunciado literal ("80% das respostas se apoiam em 20% dos documentos"): todas as fontes no top, ou alguma."""
    sources = [[q["source"]] if expected else q["documents"] for q in questions]
    sources = [s for s in sources if s]
    if not sources:
        return {"all": None, "any": None}
    return {"all": sum(set(s) <= top for s in sources) / len(sources),
            "any": sum(bool(set(s) & top) for s in sources) / len(sources)}


def magnets(questions: list[dict], chunk_counts: dict[str, int], k: int) -> list[dict]:
    """Documentos que aparecem em perguntas de outros documentos acima do acaso (ver o docstring)."""
    total = sum(chunk_counts.values()) or 1
    undue = Counter(doc for q in questions for doc in set(q["documents"]) - {q["source"]})
    found = []
    for doc, n in undue.items():
        chance = chunk_counts.get(doc, 0) / total * k * len(questions)
        if n - chance >= MAGNET_EXCESS:
            found.append({"document": doc, "undue": n, "chance": round(chance, 2), "excess": round(n - chance, 2)})
    return sorted(found, key=lambda m: -m["excess"])


def dead_documents(questions: list[dict]) -> list[str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_source[q["source"]].append(q)
    return sorted(doc for doc, qs in by_source.items() if all(doc not in q["documents_at_10"] for q in qs))


def chunk_magnets(questions: list[dict], total_chunks: int, k: int, bibliography: dict[str, bool] | None = None) -> list[dict]:
    undue: Counter = Counter()
    for q in questions:
        for chunk_id, doc in set(zip(q["chunk_ids"], q["documents"])):
            if doc != q["source"]:
                undue[chunk_id] += 1
    chance = k * len(questions) / (total_chunks or 1)
    found = [{"chunk_id": chunk_id, "undue": n, "chance": round(chance, 3),
              "bibliography": (bibliography or {}).get(chunk_id)}
             for chunk_id, n in undue.items() if n - chance >= MAGNET_EXCESS]
    return sorted(found, key=lambda m: -m["undue"])


def analyze(questions: list[dict], n_documents: int, chunk_counts: dict[str, int], k: int,
            bibliography: dict[str, bool] | None = None) -> dict:
    observed = observed_shares(questions)
    expected = expected_shares(questions)
    first = Counter(q["documents"][0] for q in questions if q["documents"])
    first_shares = {doc: n / sum(first.values()) for doc, n in first.items()} if first else {}
    total_chunks = sum(chunk_counts.values())
    return {
        "n_questions": len(questions),
        "n_documents": n_documents,
        "k": k,
        "observed": dict(sorted(observed.items(), key=lambda item: -item[1])),
        "expected": dict(sorted(expected.items(), key=lambda item: -item[1])),
        "off_source_mass": off_source_mass(questions),
        "total_variation": total_variation(observed, expected),
        "n80": {"observed": n80(observed), "expected": n80(expected), "of": n_documents},
        "statement": {
            "observed": statement_test(questions, top_documents(observed, n_documents)),
            "expected": statement_test(questions, top_documents(expected, n_documents), expected=True),
        },
        "gini": {"observed": gini(_padded(observed, n_documents)), "expected": gini(_padded(expected, n_documents)),
                 "at_1": gini(_padded(first_shares, n_documents))},
        "magnets": magnets(questions, chunk_counts, k),
        "dead": dead_documents(questions),
        "chunk_magnets": chunk_magnets(questions, total_chunks, k, bibliography),
        "lorenz": {"observed": lorenz(observed, n_documents), "expected": lorenz(expected, n_documents)},
    }


def questions_from_answers(answers: list[dict], documents: set[str]) -> list[dict]:
    """As perguntas de dentro a partir das respostas da C2 (tests/generate_answers.py), com os trechos que o
    gerador recebeu nessa rodada: os que o juiz julgou."""
    questions = []
    for answer in answers:
        source = c1.doc_stem(answer.get("source_document"))
        if source and source in documents and not answer.get("error"):
            questions.append({"name": answer["name"], "source": source,
                              "documents": [c1.doc_stem(d) for d in answer.get("source_documents") or []],
                              "chunk_ids": list(answer.get("chunk_ids") or []), "documents_at_10": []})
    return questions


def _bootstrap_difference(top: dict[str, float], rest: dict[str, float], clusters: dict[str, str],
                          samples: int = 2000, seed: int = 20261018) -> dict | None:
    """média(top) − média(rest), com IC 95% reamostrando os documentos-fonte das perguntas (grupos independentes)."""
    if not top or not rest:
        return None
    by_cluster: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for name, value in top.items():
        by_cluster[clusters[name]][0].append(value)
    for name, value in rest.items():
        by_cluster[clusters[name]][1].append(value)
    keys = sorted(by_cluster)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        a: list[float] = []
        b: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            a += by_cluster[key][0]
            b += by_cluster[key][1]
        if a and b:
            deltas.append(statistics.fmean(a) - statistics.fmean(b))
    deltas.sort()
    return {"top_only": round(statistics.fmean(top.values()), 4), "rest": round(statistics.fmean(rest.values()), 4),
            "delta": round(statistics.fmean(top.values()) - statistics.fmean(rest.values()), 4),
            "ci_low": round(c1._percentile(deltas, 0.025), 4) if deltas else None,
            "ci_high": round(c1._percentile(deltas, 0.975), 4) if deltas else None}


def quality_cross(questions: list[dict], cases: list[dict], n_documents: int, chunk_counts: dict[str, int],
                  k: int) -> dict:
    """Concentração × qualidade (ver o docstring): `questions` de questions_from_answers, `cases` do RAGAS."""
    judged = {case["name"]: case for case in cases}
    questions = [q for q in questions if q["name"] in judged and q["documents"]]
    scores: dict[str, dict[str, float]] = {"approved": {}}
    for q in questions:
        case = judged[q["name"]]
        scores["approved"][q["name"]] = float(bool(case.get("success")))
        for metric in case.get("metrics", []):
            if metric.get("score") is not None:
                scores.setdefault(metric["name"], {})[q["name"]] = float(metric["score"])
    observed = observed_shares(questions)
    excess = {m["document"]: m["excess"] for m in magnets(questions, chunk_counts, k)}
    by_document = []
    for doc, share in sorted(observed.items(), key=lambda item: -item[1]):
        users = [q["name"] for q in questions if doc in q["documents"]]
        row = {"document": doc, "participation": round(share, 4), "magnet_excess": excess.get(doc, 0.0),
               "answers": len(users),
               "failure_rate": round(1 - statistics.fmean(scores["approved"][n] for n in users), 4)}
        for metric, values in scores.items():
            used = [values[n] for n in users if n in values]
            if metric != "approved" and used:
                row[metric] = round(statistics.fmean(used), 4)
        by_document.append(row)
    top = top_documents(observed, n_documents)
    top_only = {q["name"] for q in questions if set(q["documents"]) <= top}
    clusters = {q["name"]: q["source"] for q in questions}
    return {
        "n_judged": len(questions),
        "top_documents": sorted(top),
        "by_document": by_document,
        "top_only_vs_rest": {
            "n_top_only": len(top_only), "n_rest": len(questions) - len(top_only),
            "metrics": {metric: _bootstrap_difference({n: v for n, v in values.items() if n in top_only},
                                                      {n: v for n, v in values.items() if n not in top_only}, clusters)
                        for metric, values in scores.items()},
        },
        "misattribution": None,
    }


def _bibliography_flags(index: str, chunk_ids: set[str]) -> dict[str, bool]:
    """Quais desses chunks são de bibliografia: metade dos gramas no bloco de referências do documento."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.config import PROJECT_ROOT, RAG_INDICES, set_settings_overrides
    from app.rag import corpus_quality, ingestion, vector_store

    set_settings_overrides({"rag_index": index})
    try:
        found = vector_store.get_collection().get(ids=sorted(chunk_ids), include=["documents", "metadatas"])
    finally:
        set_settings_overrides({})
    folder = PROJECT_ROOT / RAG_INDICES[index]["source_dir"]
    block_grams: dict[str, set[str]] = {}
    flags = {}
    for chunk_id, text, meta in zip(found["ids"], found["documents"], found["metadatas"]):
        name = meta["document"]
        if name not in block_grams:
            doc = ingestion.read_document(folder / name)
            lines = doc.text.splitlines()
            block = corpus_quality.references_block(lines, markdown=doc.format == "md")
            block_grams[name] = c1.grams(c1.normalize("\n".join(lines[block[0]:block[1]]))) if block else set()
        grams = c1.grams(c1.normalize(text))
        flags[chunk_id] = bool(grams) and len(grams & block_grams[name]) / len(grams) >= BIBLIOGRAPHY_MIN
    return flags


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="E5: concentração das fontes sobre um diagnóstico de recuperação.")
    parser.add_argument("diagnostico", type=Path)
    parser.add_argument("--indice", required=True)
    parser.add_argument("--lingua", default="pt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bibliografia", action="store_true", help="marca os chunks ímã de bibliografia (lê o índice)")
    parser.add_argument("--respostas", type=Path, default=None, help="as respostas da C2 do braço (cruzamento)")
    parser.add_argument("--ragas", type=Path, default=None, help="o julgamento do RAGAS dessas respostas")
    args = parser.parse_args(argv)
    if bool(args.respostas) != bool(args.ragas):
        parser.error("--respostas e --ragas vão juntos")

    diagnostic = json.loads(args.diagnostico.read_text(encoding="utf-8"))
    arm = diagnostic["c1"]["arms"][args.indice]
    manifest = json.loads((ROOT / "data" / "vector_store" / "manifests" / f"{arm['collection']}.json")
                          .read_text(encoding="utf-8"))
    chunk_counts = {c1.doc_stem(d["file"]): d["chunks"] for d in manifest["documents"]}
    questions = questions_from_diagnostic(diagnostic, args.indice, args.lingua)
    k = arm["retrieval"]["top_k"]
    result = analyze(questions, manifest["n_documents"], chunk_counts, k)
    if args.bibliografia and result["chunk_magnets"]:
        flags = _bibliography_flags(args.indice, {m["chunk_id"] for m in result["chunk_magnets"]})
        for magnet in result["chunk_magnets"]:
            magnet["bibliography"] = flags.get(magnet["chunk_id"])
    result.update(index=args.indice, lang=args.lingua)
    if args.respostas:
        answers = json.loads(args.respostas.read_text(encoding="utf-8"))["answers"]
        cases = json.loads(args.ragas.read_text(encoding="utf-8"))["cases"]
        judged = questions_from_answers(answers, set(chunk_counts))
        result["quality"] = quality_cross(judged, cases, manifest["n_documents"], chunk_counts, k)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"{args.indice} ({args.lingua}): {result['n_questions']} perguntas, {result['n_documents']} documentos, k={k}")
    print(f"  massa fora da fonte {pct(result['off_source_mass'])} · TVD {result['total_variation']:.2f} · "
          f"N80 {result['n80']['observed']}/{result['n_documents']} (esperado {result['n80']['expected']})")
    s = result["statement"]
    print(f"  todas as fontes no top 20%: {pct(s['observed']['all'])} (esperado {pct(s['expected']['all'])}) · "
          f"alguma: {pct(s['observed']['any'])} (esperado {pct(s['expected']['any'])})")
    g = result["gini"]
    print(f"  Gini {g['observed']:.2f} (esperado {g['expected']:.2f}, primeiro trecho {g['at_1']:.2f})")
    for magnet in result["magnets"]:
        print(f"  ímã: {magnet['document'][:60]} — {magnet['undue']} perguntas alheias (acaso {magnet['chance']})")
    for doc in result["dead"]:
        print(f"  morto: {doc[:70]}")
    for magnet in result["chunk_magnets"][:8]:
        bib = {True: " [bibliografia]", False: "", None: ""}[magnet["bibliography"]]
        print(f"  chunk ímã: {magnet['chunk_id'][:70]} — {magnet['undue']} perguntas alheias{bib}")
    if "quality" in result:
        quality = result["quality"]
        print(f"\nConcentração × qualidade: {quality['n_judged']} respostas julgadas")
        for row in quality["by_document"][:10]:
            print(f"  {row['document'][:55]:55s} participação {row['participation']:.0%} · ímã +{row['magnet_excess']} · "
                  f"{row['answers']} respostas, reprovação {row['failure_rate']:.0%}")
        split = quality["top_only_vs_rest"]
        print(f"  só top 20% ({split['n_top_only']}) × as demais ({split['n_rest']}):")
        for metric, d in split["metrics"].items():
            if d:
                print(f"    {metric}: {d['top_only']:.3f} × {d['rest']:.3f}, Δ {d['delta']:+.3f} [{d['ci_low']}; {d['ci_high']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
