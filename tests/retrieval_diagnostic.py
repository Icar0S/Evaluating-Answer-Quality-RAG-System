"""Diagnóstico de recuperação (C1 do plano-benchmark.md): o trecho certo chega ao gerador?

Só embeddings, sem LLM — mede a recuperação isolada do resto do pipeline, em
segundos. Para cada golden que declara `source_document`, consulta cada versão
do índice com a pergunta em português e (se o golden tiver `input_en`) em
inglês. Golden com `source_document: null` é pergunta de fora do acervo: entra
só na abstenção pelo limiar e na AUC.

Por pergunta e braço, grava a lista ranqueada que o gerador recebe (o `top_k`
e o MMR do braço) e, à parte, a de k = 10, com as métricas de
tests/benchmark/c1.py: Hit@1, Hit@k, Hit@10, MRR@10, precisão@k, cobertura e
evidência@k (a evidência do `context` do golden casada por 25-gramas) e a
similaridade do primeiro trecho. O `rank` de sempre (posição do primeiro trecho
do documento certo) continua em `rows[].ranks`, que o relatório lê.

Pergunta cujo documento a versão não tem conta como de fora nessa versão.

Foi o que mostrou, no índice md-v1, 2 de 10 acertos com a pergunta em
português contra 10 de 10 em inglês: o nomic-embed-text puxa o único documento
em português do acervo para qualquer pergunta em português.

Roda no venv do backend (tests/compare_indices.py chama assim):

    backend\\.venv\\Scripts\\python.exe tests\\retrieval_diagnostic.py --out diag.json \\
        --indices pdf-v1 md-v1 --goldens tests/deepeval/goldens/dataset_md.json
    backend\\.venv\\Scripts\\python.exe tests\\retrieval_diagnostic.py --out diag.json --indices pdf-v1 \\
        --goldens tests/benchmark/goldens/v2.json --set retrieval_use_mmr=true --set min_similarity_score=0.43
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings, set_settings_overrides  # noqa: E402
from app.rag import ingestion, retrieval, vector_store  # noqa: E402
from tests.benchmark import arm, c1  # noqa: E402

AT_10 = 10


def load_goldens(paths: list[str]) -> list[dict]:
    """Goldens que dizem de onde vem a resposta: com `source_document` (de dentro) ou com ele nulo (de fora).

    Golden sem a chave (os 10 dos PDFs, em dataset.json) não entra: sem documento-fonte não há o que medir.
    """
    goldens = []
    for path in paths:
        goldens += [g for g in json.loads(Path(path).read_text(encoding="utf-8")) if "source_document" in g]
    return goldens


def index_documents(index: str) -> set[str]:
    """Os documentos da versão, pelo nome sem extensão: a lista registrada ou, sem lista, o que está na coleção."""
    registered = ingestion.registered_documents(index)
    names = list(registered) if registered is not None else [m.get("document") for m in vector_store.get_all()["metadatas"]]
    return {c1.doc_stem(name) for name in names if name}


def retrieve_lists(question: str, k: int, overrides: dict) -> tuple[list[dict], list[dict], float | None]:
    """(o que o gerador recebe, a lista de k = 10, a similaridade do primeiro trecho sem limiar).

    Cada lista sai da sua própria busca: a do Chroma é aproximada (HNSW), e os 4 primeiros de uma busca de 10
    podem não ser os de uma busca de 4, que é a que o gerador recebe.
    """
    settings = get_settings()
    arm_chunks, _ = retrieval.retrieve(question, k)
    chunks_at_10 = arm_chunks if k == AT_10 else retrieval.retrieve(question, AT_10)[0]
    if not settings.retrieval_use_mmr and settings.min_similarity_score <= 0:
        # Sem MMR e sem limiar, o primeiro trecho do braço já é o de maior similaridade.
        return arm_chunks, chunks_at_10, arm_chunks[0]["similarity_score"] if arm_chunks else None
    set_settings_overrides({**overrides, "min_similarity_score": 0.0, "retrieval_use_mmr": False})
    try:
        raw, _ = retrieval.retrieve(question, 1)
    finally:
        set_settings_overrides(overrides)
    return arm_chunks, chunks_at_10, raw[0]["similarity_score"] if raw else None


def run(indices: list[str], goldens: list[dict], top_k: int | None = None, overrides: dict | None = None,
        with_text: bool = False) -> dict:
    langs = ["pt"] + (["en"] if any(g.get("input_en") for g in goldens) else [])
    inside = [g for g in goldens if g.get("source_document")]
    # Pergunta ambígua não tem documento nem é de fora: o certo é pedir esclarecimento, coisa da C2.
    outside = [g for g in goldens if not g.get("source_document") and g.get("question_class") != "ambiguo"]
    rows = [
        {"name": g["name"], "conversion": g.get("conversion"), "source_document": g["source_document"],
         "ranks": {}, "c1": {}}
        for g in inside
    ]
    out_rows = [{"name": g["name"], "out_stratum": g.get("out_stratum"), "c1": {}} for g in outside]
    arms, summary = {}, {}
    try:
        for index in indices:
            arm_overrides = {**(overrides or {}), "rag_index": index}
            # A configuração é cacheada; trocar de braço no mesmo processo é trocar as sobrescritas.
            set_settings_overrides(arm_overrides)
            k = top_k or get_settings().top_k
            arms[index] = arm.provenance(overrides)
            arms[index]["retrieval"]["top_k"] = k
            documents = index_documents(index)
            summary[index] = {}
            for lang in langs:
                in_index: list[tuple[str, dict]] = []
                out_index: list[dict] = []
                for golden, row in zip(inside + outside, rows + out_rows):
                    question = golden["input"] if lang == "pt" else golden.get("input_en")
                    if not question:
                        continue
                    arm_chunks, chunks_at_10, top1 = retrieve_lists(question, k, arm_overrides)
                    source = golden.get("source_document")
                    metrics = c1.question_metrics(
                        arm_chunks=arm_chunks, chunks_at_10=chunks_at_10, source_document=source,
                        fragments=c1.split_fragments(golden.get("context")), top1_similarity=top1,
                        threshold=get_settings().min_similarity_score, with_text=with_text,
                    )
                    metrics["in_index"] = bool(source) and c1.doc_stem(source) in documents
                    row["c1"].setdefault(index, {})[lang] = metrics
                    if source:
                        row["ranks"].setdefault(index, {})[lang] = metrics["rank"]
                    if metrics["in_index"]:
                        in_index.append((source, metrics))
                    else:
                        out_index.append(metrics)
                summary[index][lang] = c1.summarize(in_index, out_index, get_settings().min_similarity_score)
            hits = {lang: sum(1 for row in rows if row["ranks"].get(index, {}).get(lang)) for lang in langs}
            print(f"{index}: " + " · ".join(f"{lang.upper()} {hits[lang]}/{len(rows)}" for lang in langs), flush=True)
            for lang in langs:
                s = summary[index][lang]
                evidence = s["evidence_at_k"]["mean"] if s.get("evidence_at_k") else None
                print(f"    {lang.upper()}: evidência@{k} {evidence} · {s['n_inside']} de dentro, "
                      f"{s['n_outside']} de fora", flush=True)
    finally:
        set_settings_overrides({})
    return {
        "top_k": top_k or arms[indices[0]]["retrieval"]["top_k"],
        "indices": indices,
        "langs": langs,
        "rows": rows,
        "out_rows": out_rows,
        "c1": {"version": c1.C1_VERSION, "ngram": c1.NGRAM, "tau": c1.TAU, "arms": arms, "summary": summary},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Posição do documento certo e evidência entre os trechos recuperados.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", required=True, help="versões do índice (RAG_INDICES)")
    parser.add_argument("--goldens", action="append", required=True, help="arquivo(s) de goldens com source_document")
    parser.add_argument("--top-k", type=int, default=None, help="padrão: TOP_K da configuração")
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescrita de configuração de consulta para todos os braços (ex.: retrieval_use_mmr=true)")
    parser.add_argument("--with-text", action="store_true", help="grava o texto dos trechos (o arquivo cresce muito)")
    args = parser.parse_args()

    goldens = load_goldens(args.goldens)
    if not goldens:
        print("[ERRO] Nenhum golden com source_document.", file=sys.stderr)
        return 1
    result = run(args.indices, goldens, args.top_k, arm.parse_overrides(args.set), args.with_text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
