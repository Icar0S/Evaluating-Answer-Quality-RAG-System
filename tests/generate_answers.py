"""Gera UMA vez as respostas do assistente para os goldens, para todos os frameworks julgarem as mesmas.

Sem isto, cada suíte de qualidade (tests/ragas, tests/deepeval) chama o
pipeline por conta própria. Como a geração não é determinística — e a
recuperação muda quando o índice é reingerido —, os frameworks acabam julgando
respostas e trechos DIFERENTES para a mesma pergunta, e a comparação entre eles
mistura método com sistema (o relatório consolidado mostrou 0 de 10 respostas
iguais entre duas rodadas). Com o arquivo daqui, os dois run_and_export.py
recebem `--answers` e julgam exatamente o mesmo texto.

Usa o mesmo pipeline do POST /chat (retrieval + generation, in-process), então
roda no venv do backend:

    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out respostas.json
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out respostas.json --limit 2
"""
from __future__ import annotations

import argparse
import json
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

from app import providers  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag import generation, retrieval, vector_store  # noqa: E402

GOLDENS_PATH = ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera as respostas do assistente para os goldens, uma vez só.")
    parser.add_argument("--out", type=Path, required=True, help="arquivo JSON de saída")
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros goldens")
    args = parser.parse_args()

    if vector_store.collection_count() == 0:
        print("[ERRO] Vector store vazio — rode `python scripts/ingest_documents.py` antes.", file=sys.stderr)
        return 1

    settings = get_settings()
    active = providers.get_active_provider()
    goldens = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))[: args.limit]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": active.name,
        "generation_model": active.generation_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "prompt_version": settings.prompt_version,
        "n_cases": len(goldens),
        "answers": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Gerando {len(goldens)} respostas com {active.generation_model} ({active.name}).", flush=True)
    failures = 0
    for i, golden in enumerate(goldens, start=1):
        record = {"name": golden.get("name") or golden["input"][:40], "input": golden["input"]}
        t0 = time.perf_counter()
        # Uma tentativa extra, como no run_and_export.py do DeepEval: a maioria
        # dos timeouts do Ollama aqui é transitória.
        for attempt in (1, 2):
            try:
                chunks, _ = retrieval.retrieve(golden["input"], settings.top_k)
                result = generation.generate_answer(golden["input"], chunks)
                record.update(
                    actual_output=result["answer"],
                    retrieval_context=[c["text"] for c in chunks],
                    sources=[f"{c['document']} (pág. {c.get('page', '?')})" for c in chunks],
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
