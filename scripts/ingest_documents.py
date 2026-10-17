"""CLI para (re)processar os documentos de um índice do RAG sem passar pela API.

Uso:
    python scripts/ingest_documents.py                      # índice padrão (pdf-v1, ou RAG_INDEX do .env)
    python scripts/ingest_documents.py --index md-v1        # os Markdown de data/new_papers_marks
    python scripts/ingest_documents.py --list               # versões registradas e o estado de cada uma
    python scripts/ingest_documents.py --source data/source_pdfs --no-reset

Cada versão do índice (backend/app/config.py, RAG_INDICES) tem pasta e coleção
próprias; ingerir uma não toca nas outras. Ao terminar, grava o manifesto da
versão (arquivos com hash, parâmetros, modelo de embedding).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def _list_indices() -> None:
    from app.config import RAG_INDICES, Settings
    from app.rag import vector_store

    for name, spec in RAG_INDICES.items():
        settings = Settings(rag_index=name)
        manifest = settings.vector_store_path / "manifests" / f"{spec['collection']}.json"
        try:
            count = vector_store.get_client().get_collection(spec["collection"]).count()
        except Exception:  # noqa: BLE001 - coleção ainda não existe
            count = 0
        built = json.loads(manifest.read_text(encoding="utf-8"))["created_at"][:16] if manifest.is_file() else "—"
        print(f"{name:8s} {count:5d} chunks  ingerido em {built:16s}  {spec['source_dir']}  ({spec['description']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere os documentos de um índice do RAG.")
    parser.add_argument("--index", default=None, help="versão do índice (ver --list); padrão: RAG_INDEX ou pdf-v1")
    parser.add_argument("--list", action="store_true", help="lista as versões registradas e o estado de cada uma")
    parser.add_argument("--source", type=str, default=None, help="pasta dos documentos (padrão: a da versão)")
    parser.add_argument("--no-reset", action="store_true", help="não apaga a coleção antes de ingerir")
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="só ingere se o índice estiver vazio (usado pelo docker compose a cada subida)",
    )
    args = parser.parse_args()

    if args.index:
        # Antes de importar o app: a configuração é lida (e cacheada) no import.
        os.environ["RAG_INDEX"] = args.index
    if args.list:
        _list_indices()
        return

    from app.config import get_settings
    from app.rag.ingestion import ingest_documents
    from app.rag.vector_store import collection_count

    settings = get_settings()
    print(f"Índice: {settings.index_name} (coleção {settings.vector_store_collection}, pasta {settings.source_pdfs_dir})")
    if args.if_empty:
        existing = collection_count()
        if existing:
            print(f"Índice já populado ({existing} chunks) — nada a fazer.")
            return

    source_dir = Path(args.source) if args.source else None
    result = ingest_documents(source_dir=source_dir, reset=not args.no_reset)

    print(f"Status: {result['status']}")
    print(f"Documentos processados: {result['documents_processed']}")
    print(f"Chunks criados: {result['chunks_created']}")
    print(f"Duração: {result['duration_ms']:.0f}ms")
    if result.get("manifest"):
        print(f"Manifesto: {result['manifest']}")


if __name__ == "__main__":
    main()
