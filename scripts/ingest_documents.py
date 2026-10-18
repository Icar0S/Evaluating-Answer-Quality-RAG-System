"""CLI para (re)processar os documentos de um índice do RAG sem passar pela API.

Uso:
    python scripts/ingest_documents.py                      # índice padrão (pdf-v1, ou RAG_INDEX do .env)
    python scripts/ingest_documents.py --index md-v1        # os Markdown originais de data/new_papers_marks
    python scripts/ingest_documents.py --list               # versões registradas e o estado de cada uma
    python scripts/ingest_documents.py --index md-v3 --register   # fixa a lista de documentos de uma versão nova
    python scripts/ingest_documents.py --source data/source_pdfs --no-reset

Cada versão do índice (backend/app/config.py, RAG_INDICES) tem coleção
própria; ingerir uma não toca nas outras. Versões podem ler a mesma pasta
(pdf-v1 e pdf-v2 leem data/source_pdfs): o que entra em cada uma é a lista de
documentos dela, em backend/app/rag/index_versions/<versão>.sha256. O
--register grava essa lista com o que está na pasta agora, uma vez só. Ao
terminar, a ingestão grava o manifesto da versão (arquivos com hash,
parâmetros, modelo de embedding, e o que a lista pedia e não estava na pasta).
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
    from app.rag.ingestion import registered_documents

    for name, spec in RAG_INDICES.items():
        settings = Settings(rag_index=name)
        manifest = settings.vector_store_path / "manifests" / f"{spec['collection']}.json"
        try:
            count = vector_store.get_client().get_collection(spec["collection"]).count()
        except Exception:  # noqa: BLE001 - coleção ainda não existe
            count = 0
        built = json.loads(manifest.read_text(encoding="utf-8"))["created_at"][:16] if manifest.is_file() else "—"
        registered = registered_documents(name)
        docs = f"{len(registered):3d} docs" if registered is not None else "sem lista"
        print(f"{name:8s} {docs}  {count:5d} chunks  ingerido em {built:16s}  {spec['source_dir']}  "
              f"({spec['description']})")


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
    parser.add_argument(
        "--register",
        action="store_true",
        help="grava a lista de documentos da versão (--index) com o que está na pasta agora, sem ingerir",
    )
    args = parser.parse_args()

    if args.index:
        # Antes de importar o app: a configuração é lida (e cacheada) no import.
        os.environ["RAG_INDEX"] = args.index
    if args.list:
        _list_indices()
        return

    from app.config import get_settings
    from app.rag.ingestion import ingest_documents, register_documents
    from app.rag.vector_store import collection_count

    settings = get_settings()
    if args.register:
        if not settings.rag_index:
            parser.error("--register precisa de --index (a versão registrada em backend/app/config.py)")
        try:
            path = register_documents(settings.rag_index, settings.source_pdfs_path)
        except (FileExistsError, FileNotFoundError) as exc:
            sys.exit(f"[ERRO] {exc}")
        print(f"Lista de documentos de {settings.rag_index}: {path}")
        return
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
    if result.get("registered_documents") is not None:
        print(f"Documentos na lista da versão: {result['registered_documents']} "
              f"({result['ignored_in_folder']} arquivo(s) da pasta fora da lista, de outras versões)")
    for name in result.get("missing_documents") or []:
        print(f"[AVISO] na lista da versão e fora da pasta: {name}")
    for name in result.get("changed_documents") or []:
        print(f"[AVISO] mudou desde o registro da versão (SHA-256 diferente): {name}")
    print(f"Chunks criados: {result['chunks_created']}")
    print(f"Duração: {result['duration_ms']:.0f}ms")
    if result.get("manifest"):
        print(f"Manifesto: {result['manifest']}")


if __name__ == "__main__":
    main()
