"""CLI para (re)processar os PDFs de data/source_pdfs sem passar pela API.

Uso:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --source data/source_pdfs --no-reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.ingestion import ingest_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="add PDFs vectors do RAG.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Pasta com PDFs (padrão: data/source_pdfs)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Não apagar a coleção existente antes de ingerir",
    )
    args = parser.parse_args()

    source_dir = Path(args.source) if args.source else None
    result = ingest_documents(source_dir=source_dir, reset=not args.no_reset)

    print(f"Status: {result['status']}")
    print(f"Documentos processados: {result['documents_processed']}")
    print(f"Chunks criados: {result['chunks_created']}")
    print(f"Duração: {result['duration_ms']:.0f}ms")


if __name__ == "__main__":
    main()
