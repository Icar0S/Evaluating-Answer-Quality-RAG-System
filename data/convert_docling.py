"""Converte os 39 PDFs do pdf-v1 em Markdown com o Docling, para o degrau 2 do E2 (plano-benchmark.md).

    data/source_pdfs  ->  data/md_dl/<mesmo nome>.md

A outra ferramenta do degrau 2 é o pymupdf4llm, cuja conversão dos mesmos 39
artigos já está em data/new_papers_marks (data/sync_corpora.py, 17/10/2026).
As duas saem sem OCR (os PDFs têm texto) e sem imagens; o Docling reconhece a
estrutura das tabelas (TableFormer), que é o que o E2 quer medir.

O manifesto data/conversions_docling.json registra, por arquivo, o SHA-256 da
fonte e do convertido, as páginas e o tempo, e as versões das ferramentas. Fica
fora da pasta do acervo, como o data/conversions.json. Retomável: arquivo já
convertido de uma fonte que não mudou não é convertido de novo.

Venv próprio (o Docling traz o torch e modelos), ver data/requirements-docling.txt:

    data\\.venv-docling\\Scripts\\python data\\convert_docling.py
    data\\.venv-docling\\Scripts\\python data\\convert_docling.py --limit 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_pdfs"
TARGET = ROOT / "data" / "md_dl"
MANIFEST = ROOT / "data" / "conversions_docling.json"
BASE_LIST = ROOT / "backend" / "app" / "rag" / "index_versions" / "pdf-v1.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_documents() -> list[str]:
    """Os PDFs da lista do pdf-v1, na ordem dela."""
    lines = BASE_LIST.read_text(encoding="utf-8").splitlines()
    return [line.split(None, 1)[1].strip() for line in lines if line.strip() and not line.startswith("#")]


def converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.generate_picture_images = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Converte os PDFs do pdf-v1 em Markdown com o Docling (E2).")
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros (fumaça)")
    args = parser.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {"files": {}}
    tools = {name: version(name) for name in ("docling", "docling-core", "docling-ibm-models", "docling-parse", "torch")}
    if manifest.get("tools") and manifest["tools"] != tools:
        print(f"[ERRO] {MANIFEST} foi feito com {manifest['tools']}, não com {tools}.", file=sys.stderr)
        return 1
    names = base_documents()[: args.limit]
    pending = [n for n in names if (manifest["files"].get(n) or {}).get("source_sha256") != sha256(SOURCE / n)
               or not (TARGET / f"{Path(n).stem}.md").is_file()]
    print(f"{len(pending)} de {len(names)} PDF(s) a converter com o Docling {tools['docling']}.", flush=True)
    engine = converter() if pending else None
    for i, name in enumerate(pending, start=1):
        source, target = SOURCE / name, TARGET / f"{Path(name).stem}.md"
        t0 = time.perf_counter()
        result = engine.convert(str(source))
        markdown = result.document.export_to_markdown(image_placeholder="")
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(target)
        seconds = round(time.perf_counter() - t0, 1)
        manifest["files"][name] = {
            "output": target.name, "source_sha256": sha256(source), "output_sha256": sha256(target),
            "pages": len(result.document.pages), "seconds": seconds, "characters": len(markdown),
        }
        manifest.update(tools=tools, options={"do_ocr": False, "do_table_structure": True, "images": False},
                        updated_at=datetime.now(timezone.utc).isoformat())
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  [{i}/{len(pending)}] {seconds:.0f}s, {manifest['files'][name]['pages']} pág.: {name[:70]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
