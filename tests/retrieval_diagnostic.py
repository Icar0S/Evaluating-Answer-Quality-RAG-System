"""Diagnóstico de recuperação: o documento certo vem entre os k trechos recuperados?

Só embeddings, sem LLM — mede a recuperação isolada do resto do pipeline, em
segundos. Para cada golden que declara `source_document`, consulta cada versão
do índice com a pergunta em português e (se o golden tiver `input_en`) em
inglês, e registra a posição do documento certo (None = não veio).

Foi o que mostrou, no índice md-v1, 2 de 10 acertos com a pergunta em
português contra 10 de 10 em inglês: o nomic-embed-text puxa o único documento
em português do acervo para qualquer pergunta em português.

Roda no venv do backend (tests/compare_indices.py chama assim):

    backend\\.venv\\Scripts\\python.exe tests\\retrieval_diagnostic.py --out diag.json \\
        --indices pdf-v1 md-v1 --goldens tests/deepeval/goldens/dataset_md.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.rag import retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Posição do documento certo entre os trechos recuperados.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", required=True, help="versões do índice (RAG_INDICES)")
    parser.add_argument("--goldens", action="append", required=True, help="arquivo(s) de goldens com source_document")
    parser.add_argument("--top-k", type=int, default=None, help="padrão: TOP_K da configuração")
    args = parser.parse_args()

    goldens = []
    for path in args.goldens:
        goldens += [g for g in json.loads(Path(path).read_text(encoding="utf-8")) if g.get("source_document")]
    if not goldens:
        print("[ERRO] Nenhum golden com source_document.", file=sys.stderr)
        return 1
    langs = ["pt"] + (["en"] if any(g.get("input_en") for g in goldens) else [])

    rows = [
        {"name": g["name"], "conversion": g.get("conversion"), "source_document": g["source_document"], "ranks": {}}
        for g in goldens
    ]
    top_k = args.top_k
    for index in args.indices:
        # A configuração é cacheada; trocar de índice no mesmo processo é limpar o cache.
        os.environ["RAG_INDEX"] = index
        get_settings.cache_clear()
        k = top_k or get_settings().top_k
        hits = {lang: 0 for lang in langs}
        for golden, row in zip(goldens, rows):
            row["ranks"][index] = {}
            for lang in langs:
                question = golden["input"] if lang == "pt" else golden.get("input_en")
                if not question:
                    continue
                chunks, _ = retrieval.retrieve(question, k)
                # Pelo nome sem extensão: nas versões v2, o "X.md" de um golden está no pdf-v2 como "X.pdf".
                stem = Path(golden["source_document"]).stem
                rank = next((i for i, c in enumerate(chunks, start=1) if Path(c["document"]).stem == stem), None)
                row["ranks"][index][lang] = rank
                hits[lang] += rank is not None
        print(f"{index}: " + " · ".join(f"{lang.upper()} {hits[lang]}/{len(goldens)}" for lang in langs), flush=True)

    result = {"top_k": top_k or get_settings().top_k, "indices": args.indices, "langs": langs, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
