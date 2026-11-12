"""CLI para (re)processar os documentos de um índice do RAG sem passar pela API.

Uso:
    python scripts/ingest_documents.py                      # índice padrão (pdf-v1, ou RAG_INDEX do .env)
    python scripts/ingest_documents.py --index md-v1        # os Markdown originais de data/new_papers_marks
    python scripts/ingest_documents.py --list               # versões registradas e o estado de cada uma
    python scripts/ingest_documents.py --index md-v3 --register   # fixa a lista de documentos de uma versão nova
    python scripts/ingest_documents.py --index pdf-v1-bgem3 --register --like pdf-v1   # versão derivada
    python scripts/ingest_documents.py --index md-p4llm --register --like pdf-v1 --by-stem   # mesmos artigos, outro formato
    python scripts/ingest_documents.py --index pdf-v1 --c0  # qualidade do acervo no manifesto, sem reingerir
    python scripts/ingest_documents.py --source data/source_pdfs --no-reset

Cada versão do índice (backend/app/config.py, RAG_INDICES) tem coleção
própria; ingerir uma não toca nas outras. Versões podem ler a mesma pasta
(pdf-v1 e pdf-v2 leem data/source_pdfs): o que entra em cada uma é a lista de
documentos dela, em backend/app/rag/index_versions/<versão>.sha256. O
--register grava essa lista com o que está na pasta agora, uma vez só; com
--like <base>, copia a lista da versão-base (versão derivada: os mesmos
documentos e uma mudança só); com --like <base> --by-stem, pega da pasta os
mesmos artigos da base pelo nome sem a extensão (o E2 compara os 39 PDFs do
pdf-v1 com as conversões deles em Markdown). Ao terminar, a ingestão grava o manifesto da
versão (arquivos com hash, parâmetros, modelo de embedding, e o que a lista
pedia e não estava na pasta).

Os avisos de acervo vazio e de pasta diferente do índice existem por causa do
docker compose: o serviço ingest roda a cada subida, com --if-empty, e sem eles
um acervo que faltou (os documentos não são versionados) fazia a stack subir com
o índice vazio em silêncio, e os testes ao vivo se auto-pulavam sem dizer por quê.
Nenhum dos dois para a subida (o script sai com 0), e nenhum reindexa sozinho: no
Docker, o backend está com o banco aberto.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

RULE = "=" * 78


def reindex_hint(index: str | None) -> list[str]:
    """Como reindexar, no Docker (o serviço ingest define INGEST_COMPOSE_SERVICE) ou na máquina.

    `index` é a versão registrada (RAG_INDEX); sem ela, o comando da máquina vai sem --index.
    """
    service = os.environ.get("INGEST_COMPOSE_SERVICE")
    if service:
        return [
            "  Para reindexar (os backends param durante a ingestão e voltam no fim):",
            "      scripts\\docker_tests.bat reindexar        (Windows)",
            "      ./scripts/docker_tests.sh reindexar       (Linux/macOS)",
            "  ou, à mão:",
            "      docker compose stop backend backend-md",
            f"      docker compose run --rm {service} python scripts/ingest_documents.py",
            "      docker compose up -d",
        ]
    command = "python scripts/ingest_documents.py" + (f" --index {index}" if index else "")
    return [
        "  Para reindexar, com o backend parado:",
        f"      {command}   (no Windows: backend\\.venv\\Scripts\\python ...)",
    ]


def warn_no_documents(settings, result: dict) -> None:
    """O acervo não está na pasta: o RAG sobe com o índice vazio. Aviso grande, para não passar batido no log."""
    folder = settings.source_pdfs_dir
    lines = [
        RULE,
        f"[AVISO] Nenhum documento para indexar em {folder} (índice {settings.index_name}).",
        "  O acervo não é versionado: os PDFs e os Markdown ficam fora do Git, e cada máquina",
        "  precisa colocar os seus na pasta.",
    ]
    missing = result.get("missing_documents") or []
    if missing:
        lines.append(f"  A lista da versão pede {len(missing)} arquivo(s), e nenhum deles está na pasta.")
    lines += [
        "  - O RAG sobe assim mesmo, com o índice VAZIO: responde \"não encontrei\" a tudo.",
        "  - Os testes com geração real dele (E2E ao vivo, RAGAS, DeepEval) vão se auto-pular.",
        f"  Ponha os documentos em {folder} e reindexe.",
        *reindex_hint(settings.rag_index),
        RULE,
    ]
    print("\n".join(lines), flush=True)


def folder_divergence() -> dict | None:
    """O que a ingestão pegaria da pasta agora × o manifesto do índice; None se o índice não tem manifesto.

    Compara nome e SHA-256 dos arquivos que entram na versão (select_documents: numa versão registrada, só os da
    lista dela; os de outras versões na mesma pasta não contam).
    """
    from app.rag.ingestion import _sha256, manifest_path, select_documents

    path = manifest_path()
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    indexed = {d["file"]: d["sha256"] for d in manifest.get("documents", [])}
    _, files, _ = select_documents()
    current = {p.name: _sha256(p) for p in files}
    return {
        "created_at": manifest.get("created_at"),
        "added": sorted(set(current) - set(indexed)),
        "removed": sorted(set(indexed) - set(current)),
        "changed": sorted(name for name in set(current) & set(indexed) if current[name] != indexed[name]),
    }


def warn_if_folder_diverges(settings) -> None:
    """Índice já populado: avisa se a pasta mudou desde a ingestão. Não reindexa (o backend usa o banco)."""
    diff = folder_divergence()
    if diff is None:
        print(f"(O índice não tem manifesto em {settings.vector_store_dir}/manifests, porque foi criado antes deles: "
              "não dá para conferir se a pasta mudou. Uma reindexação grava o manifesto.)")
        return
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        return
    lines = [
        RULE,
        f"[AVISO] A pasta {settings.source_pdfs_dir} mudou desde a indexação de {settings.index_name} "
        f"({(diff['created_at'] or '?')[:10]}):",
        f"  {len(diff['added'])} novo(s), {len(diff['removed'])} fora da pasta, {len(diff['changed'])} alterado(s).",
    ]
    lines += [f"    novo:      {name}" for name in diff["added"]]
    lines += [f"    saiu:      {name}" for name in diff["removed"]]
    lines += [f"    alterado:  {name}" for name in diff["changed"]]
    lines += [
        "  O RAG continua respondendo com o índice de antes: a reindexação não é automática,",
        "  porque o backend está com o banco aberto.",
        *reindex_hint(settings.rag_index),
        RULE,
    ]
    print("\n".join(lines), flush=True)


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


def _measure_c0() -> None:
    from app.config import get_settings
    from app.rag.ingestion import measure_corpus_quality

    try:
        c0 = measure_corpus_quality()
    except FileNotFoundError as exc:
        sys.exit(f"[ERRO] {exc}")
    pct = lambda value: "—" if value is None else f"{value:.1%}"  # noqa: E731
    chunks = c0["chunk_tokens"] or {}
    print(f"C0 de {get_settings().index_name} ({c0['documents']} documentos, {c0['tokens']} tokens):")
    print(f"  tokens por chunk: média {chunks.get('mean')}, p10 {chunks.get('p10')}, p90 {chunks.get('p90')}")
    if c0["pages_total"] is not None:
        print(f"  páginas: {c0['pages_total']}, {c0['pages_without_text']} sem texto (descartadas na ingestão)")
    print(f"  palavras grudadas: média {pct(c0['glued_word_rate_mean'])} das letras, "
          f"{c0['documents_glued_over_5pct']} documento(s) acima de 5%")
    print(f"  bibliografia: {pct(c0['references_share'])} dos tokens, "
          f"{c0['documents_without_references']} documento(s) sem bloco de referências")
    for name in c0["not_measured"]:
        print(f"[AVISO] sem C0 (mudou desde a ingestão ou o corte não bate com o índice): {name}")


def _register(settings, like: str | None, by_stem: bool = False) -> None:
    from app.rag.ingestion import register_documents

    try:
        path = register_documents(settings.rag_index, settings.source_pdfs_path, like=like, by_stem=by_stem)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"[ERRO] {exc}")
    print(f"Lista de documentos de {settings.rag_index}: {path}")


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
    parser.add_argument(
        "--like",
        default=None,
        metavar="BASE",
        help="com --register: copia a lista da versão BASE em vez de pegar a pasta inteira (versão derivada)",
    )
    parser.add_argument(
        "--by-stem",
        action="store_true",
        help="com --like: os mesmos artigos da BASE pelo nome sem a extensão, em outro formato (E2)",
    )
    parser.add_argument(
        "--c0",
        action="store_true",
        help="calcula o C0 (qualidade do acervo) de uma versão já ingerida e grava no manifesto, sem reingerir",
    )
    args = parser.parse_args()
    if args.like and not args.register:
        parser.error("--like só vale com --register")
    if args.by_stem and not args.like:
        parser.error("--by-stem só vale com --like")

    if args.index:
        # Antes de importar o app: a configuração é lida (e cacheada) no import.
        os.environ["RAG_INDEX"] = args.index
    if args.list:
        _list_indices()
        return
    if args.c0:
        _measure_c0()
        return

    from app.config import get_settings
    from app.rag.ingestion import ingest_documents
    from app.rag.vector_store import collection_count

    settings = get_settings()
    if args.register:
        if not settings.rag_index:
            parser.error("--register precisa de --index (a versão registrada em backend/app/config.py)")
        _register(settings, args.like, args.by_stem)
        return
    print(f"Índice: {settings.index_name} (coleção {settings.vector_store_collection}, pasta {settings.source_pdfs_dir})")
    if args.if_empty:
        existing = collection_count()
        if existing:
            print(f"Índice já populado ({existing} chunks) — nada a fazer.")
            if not args.source:
                warn_if_folder_diverges(settings)
            return

    source_dir = Path(args.source) if args.source else None
    result = ingest_documents(source_dir=source_dir, reset=not args.no_reset)
    _print_result(result)
    if result["status"] == "no_documents":
        warn_no_documents(settings, result)


def _print_result(result: dict) -> None:
    print(f"Status: {result['status']}")
    print(f"Documentos processados: {result['documents_processed']}")
    if result.get("registered_documents") is not None:
        print(f"Documentos na lista da versão: {result['registered_documents']} "
              f"({result['ignored_in_folder']} arquivo(s) da pasta fora da lista, de outras versões)")
    # Sem nenhum documento, a lista inteira faltaria aqui (39 linhas no pdf-v1), repetindo o log da ingestão:
    # o aviso grande do fim (warn_no_documents) dá a contagem.
    if result["status"] != "no_documents":
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
