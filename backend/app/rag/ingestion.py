"""Ingestão de documentos (PDF e Markdown): extração de texto, chunking e indexação vetorial.

PDF: chunking dentro de cada página (facilita citar a página como fonte).
Markdown: o documento inteiro, sem página. Os dois usam o mesmo chunker por
tokens (tiktoken cl100k_base como aproximação genérica, independente do
tokenizer real do modelo local) e os mesmos parâmetros: comparar índices de
formatos diferentes (config.RAG_INDICES) só faz sentido com o mesmo corte.

Cada ingestão grava um manifesto da versão do índice (arquivos com hash,
parâmetros, modelo de embedding) em <vector_store>/manifests/<coleção>.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import tiktoken

from app.config import PROJECT_ROOT, RAG_INDICES, get_settings
from app.logging_utils import logger
from app.rag import corpus_quality, vector_store

_encoding = tiktoken.get_encoding("cl100k_base")

SUPPORTED_SUFFIXES = (".pdf", ".md")

# Lista de documentos de cada versão registrada (config.RAG_INDICES): uma linha "<sha256>  <arquivo>" por
# documento, "#" para comentário. Versões diferentes leem a mesma pasta; é a lista que diz o que entra em cada uma.
INDEX_VERSIONS_DIR = Path(__file__).resolve().parent / "index_versions"

# Versão da conversão Markdown -> texto, gravada no manifesto. Só a SINTAXE de
# tabela vira texto: linhas separadoras somem e as células não vazias de cada
# linha são unidas por espaço. Nada de "consertar" o conteúdo (hifenização,
# carimbos do arXiv invertidos que a conversão PDF->MD deixou): isso seria
# tratar um formato melhor que o outro na comparação com o índice de PDFs. Nos
# 22 artigos de data/new_papers_marks, tira ~11% dos tokens (pipes e células
# vazias de layout) e devolve títulos que a conversão quebrou em células.
MD_NORMALIZATION = "v1: sintaxe de tabela vira texto; conteúdo intacto"
_MD_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    page: int | None
    text: str


@dataclass
class DocumentText:
    """O texto de um documento como o chunker recebe, e o que a leitura viu (para o C0 do manifesto).

    PDF: uma entrada (página, texto) por página com texto. Markdown: uma só, (None, texto normalizado).
    """

    path: Path
    format: str
    segments: list[tuple[int | None, str]]
    raw_markdown: str | None = None
    pages_total: int | None = None

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.segments)


def normalize_markdown(text: str) -> str:
    """Markdown -> texto para embedding (ver MD_NORMALIZATION)."""
    lines = []
    for line in text.splitlines():
        if _MD_TABLE_SEPARATOR.match(line):
            continue
        if line.lstrip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            line = " ".join(cell for cell in cells if cell)
            if not line:
                continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def markdown_structure(raw: str) -> dict:
    """Linhas de tabela (sem as separadoras) e títulos do Markdown cru, com o critério do normalize_markdown."""
    lines = raw.splitlines()
    return {
        "table_rows": sum(1 for line in lines if line.lstrip().startswith("|") and not _MD_TABLE_SEPARATOR.match(line)),
        "headings": sum(1 for line in lines if re.match(r"#{1,6}(\s|$)", line)),
    }


def _read_pdf(pdf_path: Path) -> tuple[list[tuple[int, str]], int]:
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((i, text))
        return pages, doc.page_count


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Retorna [(numero_da_pagina_1_indexed, texto)] para um PDF."""
    return _read_pdf(pdf_path)[0]


def read_document(path: Path) -> DocumentText:
    """Lê um PDF ou Markdown do jeito que a ingestão lê (páginas sem texto ficam de fora)."""
    if path.suffix.lower() == ".md":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return DocumentText(path, "md", [(None, normalize_markdown(raw))], raw_markdown=raw)
    pages, total = _read_pdf(path)
    return DocumentText(path, "pdf", list(pages), pages_total=total)


def effective_overlap(chunk_size_tokens: int, overlap_tokens: int) -> int:
    """Sobreposição que o chunker usa de fato: uma sobreposição >= ao chunk vira chunk // 4."""
    if overlap_tokens >= chunk_size_tokens:
        return max(chunk_size_tokens // 4, 0)
    return overlap_tokens


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    boundary_offset_tokens: int = 0,
) -> list[str]:
    """Divide o texto em janelas de tokens com sobreposição.

    boundary_offset_tokens desloca todas as fronteiras: o primeiro chunk fica
    encurtado nesse tanto e os seguintes passam a cortar em posições diferentes,
    sem perder nenhum token. Serve ao operador de mutação K4, que verifica o que
    acontece quando um corte cai no meio de uma regra.
    """
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    overlap_tokens = effective_overlap(chunk_size_tokens, overlap_tokens)

    offset = boundary_offset_tokens % chunk_size_tokens if chunk_size_tokens > 0 else 0

    chunks: list[str] = []
    start = 0
    first_size = chunk_size_tokens - offset if offset else chunk_size_tokens
    size = max(first_size, 1)
    while start < len(tokens):
        end = start + size
        chunks.append(_encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
        size = chunk_size_tokens
    return chunks


def chunk_document(
    doc: DocumentText,
    chunk_size_tokens: int,
    overlap_tokens: int,
    boundary_offset_tokens: int = 0,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    doc_name, stem = doc.path.name, doc.path.stem
    if doc.format == "md":
        [(_, text)] = doc.segments
        for idx, chunk in enumerate(chunk_text(text, chunk_size_tokens, overlap_tokens, boundary_offset_tokens)):
            records.append(ChunkRecord(chunk_id=f"{stem}::c{idx}", document=doc_name, page=None, text=chunk))
        return records
    for page_num, page_text in doc.segments:
        page_chunks = chunk_text(page_text, chunk_size_tokens, overlap_tokens, boundary_offset_tokens)
        for idx, chunk in enumerate(page_chunks):
            chunk_id = f"{stem}::p{page_num}::c{idx}"
            records.append(ChunkRecord(chunk_id=chunk_id, document=doc_name, page=page_num, text=chunk))
    return records


def build_chunk_records(
    pdf_path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int,
    boundary_offset_tokens: int = 0,
) -> list[ChunkRecord]:
    return chunk_document(read_document(pdf_path), chunk_size_tokens, overlap_tokens, boundary_offset_tokens)


def document_quality(doc: DocumentText, records: list[ChunkRecord]) -> dict:
    """C0 do documento (app/rag/corpus_quality.py), sobre o texto que o chunker recebeu."""
    structure = markdown_structure(doc.raw_markdown) if doc.raw_markdown is not None else {}
    return corpus_quality.document_quality(
        fmt=doc.format,
        text=doc.text,
        chunk_texts=[r.text for r in records],
        pages_total=doc.pages_total,
        pages_with_text=len(doc.segments) if doc.format == "pdf" else None,
        table_rows=structure.get("table_rows"),
        headings=structure.get("headings", 0),
    )


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Gera embeddings sempre via Ollama LOCAL, em lote quando /api/embed está disponível.

    Deliberadamente ignora o provider ativo: a API remota (smartdatatest, ver
    app/providers.py) não expõe rota de embeddings, e mesmo que expusesse,
    misturar espaços de embedding entre chamadas corromperia a busca por
    similaridade no vector store. Só a geração (app/rag/generation.py) muda com
    o provider ativo.
    """
    if not texts:
        return []
    settings = get_settings()
    base_url = settings.ollama_base_url
    # `model` explícito existe para quem precisa de um espaço de embedding fixo,
    # independente da configuração corrente — é o caso do oráculo O1 do estudo de
    # mutação, que não pode ser afetado pelo operador E1 (troca de embedding).
    embedding_model = model or settings.embedding_model
    url = f"{base_url}/api/embed"
    with httpx.Client(timeout=settings.generation_timeout_seconds) as client:
        try:
            resp = client.post(url, json={"model": embedding_model, "input": texts})
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except (httpx.HTTPError, ValueError, KeyError):
            logger.warning("Batch embedding falhou, caindo para embeddings um a um")

        # Fallback: endpoint legado, um texto por chamada.
        results: list[list[float]] = []
        legacy_url = f"{base_url}/api/embeddings"
        for text in texts:
            resp = client.post(legacy_url, json={"model": embedding_model, "prompt": text})
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_documents(index: str) -> dict[str, str] | None:
    """{arquivo: sha256} registrados para a versão, ou None se ela não tem lista."""
    path = INDEX_VERSIONS_DIR / f"{index}.sha256"
    if not path.is_file():
        return None
    documents: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            documents[name.strip()] = digest
    return documents


def register_documents(index: str, source_dir: Path, like: str | None = None) -> Path:
    """Grava a lista de documentos da versão com o que está na pasta agora ou, com `like`, a lista de outra versão.

    `like` é para a versão derivada: os mesmos documentos da versão-base e uma mudança só (modelo de embedding,
    chunker, cabeçalho...). Sem ele, a lista pegaria a pasta inteira, que pode ter documentos de outras versões
    (data/source_pdfs tem os 61 do pdf-v2), e comparar a derivada com a base mediria também a troca de acervo.
    Os arquivos da base precisam estar na pasta com o mesmo conteúdo.

    Uma vez só: versão registrada não muda, e mudar o acervo é registrar uma versão nova.
    """
    path = INDEX_VERSIONS_DIR / f"{index}.sha256"
    if path.exists():
        raise FileExistsError(f"{path} já existe: versão registrada não muda (registre uma versão nova)")
    today = datetime.now(timezone.utc).date().isoformat()
    if like is not None:
        base = registered_documents(like)
        if base is None:
            raise FileNotFoundError(f"{like} não tem lista de documentos em {INDEX_VERSIONS_DIR}")
        problems = []
        for name, digest in base.items():
            if not (source_dir / name).is_file():
                problems.append(f"{name}: não está em {source_dir}")
            elif _sha256(source_dir / name) != digest:
                problems.append(f"{name}: mudou desde o registro de {like}")
        if problems:
            raise ValueError(f"a pasta não tem os documentos de {like}: " + "; ".join(problems))
        header = (
            f"# {index}: os {len(base)} documento(s) de {like}, em {source_dir.name}, registrados em {today} "
            f"(scripts/ingest_documents.py --register --like {like})"
        )
        entries = list(base.items())
    else:
        files = sorted(
            (p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES), key=lambda p: p.name.lower()
        )
        if not files:
            raise FileNotFoundError(f"nenhum documento em {source_dir}")
        header = (
            f"# {index}: {len(files)} documento(s) de {source_dir.name}, registrados em {today} "
            "(scripts/ingest_documents.py --register)"
        )
        entries = [(p.name, _sha256(p)) for p in files]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header] + [f"{digest}  {name}" for name, digest in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def select_documents(source_dir: Path | None = None) -> tuple[Path, list[Path], dict]:
    """Pasta e documentos que a ingestão do índice em uso pega.

    Na pasta da própria versão registrada, só os arquivos da lista dela, na ordem de sempre (a da pasta). Com
    outra pasta (`source_dir`, ou uma versão sem lista), tudo o que estiver nela, como antes das listas. O dict
    diz o que a lista pedia e não estava na pasta ("missing") e quantos arquivos da pasta ficaram de fora.
    """
    settings = get_settings()
    doc_dir = source_dir or settings.source_pdfs_path
    files = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES) if doc_dir.is_dir() else []
    spec = RAG_INDICES.get(settings.index_name)
    own_folder = spec is not None and (PROJECT_ROOT / spec["source_dir"]).resolve() == doc_dir.resolve()
    registered = registered_documents(settings.index_name) if source_dir is None and own_folder else None
    if registered is None:
        return doc_dir, files, {"registered": None, "missing": [], "ignored_in_folder": 0}
    selected = [p for p in files if p.name in registered]
    present = {p.name for p in files}
    return doc_dir, selected, {
        "registered": registered,
        "missing": sorted(name for name in registered if name not in present),
        "ignored_in_folder": len(files) - len(selected),
    }


def write_manifest(
    source_dir: Path, documents: list[dict], chunks: int, duration_ms: float, extra: dict | None = None
) -> Path:
    """Registra o que entrou nesta versão do índice: arquivos (com hash), parâmetros e modelo.

    É o que permite dizer, meses depois, contra QUAL acervo uma avaliação rodou.
    """
    settings = get_settings()
    manifest = {
        "index": settings.index_name,
        "collection": settings.vector_store_collection,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "embedding_model": settings.embedding_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        # O pedido acima e o que o chunker usou (sobreposição >= chunk vira chunk // 4, sem aviso).
        "chunk_overlap_tokens_effective": effective_overlap(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
        "chunk_boundary_offset_tokens": settings.chunk_boundary_offset_tokens,
        "md_normalization": MD_NORMALIZATION if any(d["format"] == "md" for d in documents) else None,
        "n_documents": len(documents),
        "n_chunks": chunks,
        "duration_ms": round(duration_ms),
        **(extra or {}),
        "documents": documents,
    }
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def manifest_path() -> Path:
    settings = get_settings()
    return settings.vector_store_path / "manifests" / f"{settings.vector_store_collection}.json"


def measure_corpus_quality() -> dict:
    """Calcula o C0 da versão em uso, já ingerida, e grava no manifesto dela, sem reingerir.

    Para as versões ingeridas antes do C0 (o pdf-v1 é de 13/10): lê os arquivos do manifesto e corta com os
    parâmetros dele. Documento que mudou (SHA-256) ou cujo corte de hoje não dá o mesmo número de chunks fica
    sem C0, listado em "not_measured": o texto dele não é o que está no índice.
    """
    path = manifest_path()
    if not path.is_file():
        raise FileNotFoundError(f"{get_settings().index_name} não tem manifesto em {path}: ingira a versão antes")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    folder = Path(manifest["source_dir"])
    if not folder.is_dir():
        folder = get_settings().source_pdfs_path
    params = (
        manifest["chunk_size_tokens"],
        manifest["chunk_overlap_tokens"],
        manifest.get("chunk_boundary_offset_tokens", 0),
    )
    counts: list[int] = []
    not_measured: list[str] = []
    for entry in manifest["documents"]:
        file = folder / entry["file"]
        if not file.is_file() or _sha256(file) != entry["sha256"]:
            not_measured.append(entry["file"])
            continue
        doc = read_document(file)
        records = chunk_document(doc, *params)
        if len(records) != entry["chunks"]:
            not_measured.append(entry["file"])
            continue
        entry.update(document_quality(doc, records))
        counts += [corpus_quality.count_tokens(r.text) for r in records]
    measured = [entry for entry in manifest["documents"] if entry["file"] not in not_measured]
    c0 = {
        **corpus_quality.summarize(measured, counts),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "not_measured": not_measured,
    }
    documents = manifest.pop("documents")
    manifest.pop("c0", None)
    path.write_text(json.dumps({**manifest, "c0": c0, "documents": documents}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return c0


def ingest_documents(source_dir: Path | None = None, reset: bool = True) -> dict:
    """Reprocessa os PDFs e Markdown do índice ativo (ver select_documents) ou de source_dir."""
    settings = get_settings()
    start = time.perf_counter()
    doc_dir, doc_files, selection = select_documents(source_dir)
    for name in selection["missing"]:
        logger.warning(f"{settings.index_name}: {name} está na lista da versão e não está em {doc_dir}")

    if not doc_files:
        return {
            "status": "no_documents",
            "documents_processed": 0,
            "chunks_created": 0,
            "collection": settings.vector_store_collection,
            "missing_documents": selection["missing"],
            "duration_ms": (time.perf_counter() - start) * 1000,
        }

    all_records: list[ChunkRecord] = []
    manifest_docs: list[dict] = []
    for doc_path in doc_files:
        try:
            doc = read_document(doc_path)
            records = chunk_document(
                doc,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                settings.chunk_boundary_offset_tokens,
            )
            all_records.extend(records)
            manifest_docs.append(
                {
                    "file": doc_path.name,
                    "format": doc_path.suffix.lower().lstrip("."),
                    "bytes": doc_path.stat().st_size,
                    "sha256": _sha256(doc_path),
                    "chunks": len(records),
                    **document_quality(doc, records),
                }
            )
            logger.info(f"Ingerido {doc_path.name}: {len(records)} chunks")
        except Exception as exc:
            logger.error(f"Falha ao processar {doc_path.name}: {exc}")

    first_batch = True
    chunks_created = 0
    batch_size = 64
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i : i + batch_size]
        embeddings = embed_texts([r.text for r in batch])
        vector_store.add_chunks(
            ids=[r.chunk_id for r in batch],
            embeddings=embeddings,
            documents=[r.text for r in batch],
            # Markdown não tem página: a chave fica de fora (o Chroma não aceita
            # metadado None) e a recuperação devolve page=None.
            metadatas=[
                {"document": r.document, "page": r.page} if r.page is not None else {"document": r.document}
                for r in batch
            ],
            reset=reset and first_batch,
        )
        first_batch = False
        chunks_created += len(batch)

    # Arquivo da lista que mudou depois do registro: a ingestão segue, mas o índice já não é a versão registrada.
    registered = selection["registered"]
    changed = [d["file"] for d in manifest_docs if registered and d["sha256"] != registered[d["file"]]]
    for name in changed:
        logger.warning(f"{settings.index_name}: {name} mudou desde que a versão foi registrada (SHA-256 diferente)")
    extra = {
        "registered_documents": len(registered) if registered is not None else None,
        "missing_documents": selection["missing"],
        "changed_documents": changed,
        "ignored_in_folder": selection["ignored_in_folder"],
    }

    duration_ms = (time.perf_counter() - start) * 1000
    c0 = corpus_quality.summarize(manifest_docs, [corpus_quality.count_tokens(r.text) for r in all_records])
    manifest = write_manifest(doc_dir, manifest_docs, chunks_created, duration_ms, {**extra, "c0": c0})
    return {
        "status": "ok",
        "documents_processed": len(manifest_docs),
        "chunks_created": chunks_created,
        "collection": settings.vector_store_collection,
        "index": settings.index_name,
        "manifest": str(manifest),
        "duration_ms": duration_ms,
        **extra,
    }
