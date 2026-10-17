"""Gera UMA vez as respostas do assistente para os goldens, para todos os frameworks julgarem as mesmas.

Sem isto, cada suíte de qualidade (tests/ragas, tests/deepeval) chama o
pipeline por conta própria. Como a geração não é determinística — e a
recuperação muda quando o índice é reingerido —, os frameworks acabam julgando
respostas e trechos DIFERENTES para a mesma pergunta, e a comparação entre eles
mistura método com sistema (o relatório consolidado mostrou 0 de 10 respostas
iguais entre duas rodadas). Com o arquivo daqui, os dois run_and_export.py
recebem `--answers` e julgam exatamente o mesmo texto.

O arquivo é autocontido: cada resposta leva a pergunta, a referência e o
contexto do golden, e a origem dele (`--goldens tag=caminho`). É isso que deixa
avaliar conjuntos de goldens diferentes (o dos PDFs, o dos Markdown) e
comparar versões do índice (`--index`, ver backend/app/config.py RAG_INDICES).

Usa o mesmo pipeline do POST /chat (retrieval + generation, in-process), então
roda no venv do backend:

    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out respostas.json
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --index md-v1 \\
        --goldens pdf=tests/deepeval/goldens/dataset.json --goldens md=tests/deepeval/goldens/dataset_md.json
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --lang en   # usa input_en dos goldens
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# Mesmo motivo do run_and_export.py do DeepEval: com saída redirecionada para
# arquivo, o Windows usa cp1252 e um acento derruba o processo.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_GOLDENS = ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"
# Origem padrão de cada arquivo de goldens conhecido (o acervo de onde as
# respostas de referência vieram).
KNOWN_ORIGINS = {"dataset.json": "pdf", "dataset_md.json": "md"}


def parse_goldens_arg(value: str) -> tuple[str, Path]:
    """`tag=caminho` ou só `caminho` (a tag vem de KNOWN_ORIGINS ou do nome do arquivo)."""
    if "=" in value and not Path(value).exists():
        tag, path = value.split("=", 1)
        return tag, Path(path)
    path = Path(value)
    return KNOWN_ORIGINS.get(path.name, path.stem), path


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera as respostas do assistente para os goldens, uma vez só.")
    parser.add_argument("--out", type=Path, required=True, help="arquivo JSON de saída")
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros goldens de cada conjunto")
    parser.add_argument("--index", default=None, help="versão do índice (RAG_INDICES); padrão: a do .env")
    parser.add_argument(
        "--goldens", action="append", default=[],
        help="conjunto de goldens, `tag=caminho` (repetível); padrão: pdf=tests/deepeval/goldens/dataset.json",
    )
    parser.add_argument("--lang", choices=("pt", "en"), default="pt", help="en = usa o input_en dos goldens")
    args = parser.parse_args()

    if args.index:
        # Antes de importar o app: a configuração é lida (e cacheada) no import.
        os.environ["RAG_INDEX"] = args.index

    from app import providers
    from app.config import get_settings
    from app.rag import generation, retrieval, vector_store

    if vector_store.collection_count() == 0:
        print("[ERRO] Índice vazio — rode `python scripts/ingest_documents.py` antes.", file=sys.stderr)
        return 1

    goldens = []
    golden_sets = [parse_goldens_arg(g) for g in args.goldens] or [("pdf", DEFAULT_GOLDENS)]
    for origin, path in golden_sets:
        items = json.loads(path.read_text(encoding="utf-8"))[: args.limit]
        goldens += [(origin, g) for g in items]

    settings = get_settings()
    active = providers.get_active_provider()
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": active.name,
        "generation_model": active.generation_model,
        "rag_index": settings.index_name,
        "collection": settings.vector_store_collection,
        "embedding_model": settings.embedding_model,
        "lang": args.lang,
        "top_k": settings.top_k,
        "prompt_version": settings.prompt_version,
        "goldens": [f"{origin}={path.as_posix()}" for origin, path in golden_sets],
        "n_cases": len(goldens),
        "answers": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Gerando {len(goldens)} respostas com {active.generation_model} ({active.name}), "
        f"índice {settings.index_name}, perguntas em {args.lang}.",
        flush=True,
    )
    failures = 0
    for i, (origin, golden) in enumerate(goldens, start=1):
        question = golden.get("input_en") if args.lang == "en" and golden.get("input_en") else golden["input"]
        record = {
            "name": golden.get("name") or golden["input"][:40],
            "origin": origin,
            "input": question,
            "expected_output": golden.get("expected_output", ""),
            "context": golden.get("context") or [],
            "source_document": golden.get("source_document"),
        }
        t0 = time.perf_counter()
        # Uma tentativa extra, como no run_and_export.py do DeepEval: a maioria
        # dos timeouts do Ollama aqui é transitória.
        for attempt in (1, 2):
            try:
                chunks, _ = retrieval.retrieve(question, settings.top_k)
                result = generation.generate_answer(question, chunks)
                record.update(
                    actual_output=result["answer"],
                    retrieval_context=[c["text"] for c in chunks],
                    sources=[
                        c["document"] if c.get("page") is None else f"{c['document']} (pág. {c['page']})"
                        for c in chunks
                    ],
                    error=None,
                )
                break
            except Exception as exc:  # noqa: BLE001 - registra e segue; quem julga marca como não medido
                record.update(actual_output=None, retrieval_context=[], sources=[], error=f"{type(exc).__name__}: {exc}")
                print(f"  [{i}] tentativa {attempt}/2 falhou: {exc!r}", flush=True)
        record["seconds"] = round(time.perf_counter() - t0, 1)
        failures += record["error"] is not None
        output["answers"].append(record)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "ERRO" if record["error"] else f"{len(record['actual_output'])} caracteres"
        print(f"  [{i}/{len(goldens)}] {record['name']}: {status} ({record['seconds']}s)", flush=True)

    print(f"Respostas em {args.out} ({failures} falha(s)).", flush=True)
    return 0 if failures < len(goldens) else 1


if __name__ == "__main__":
    sys.exit(main())
