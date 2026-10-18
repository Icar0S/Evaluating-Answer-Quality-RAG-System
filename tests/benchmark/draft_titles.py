"""Rascunho da tabela de títulos dos artigos (backend/app/rag/titles.json), para o cabeçalho do E3.

O plano (§8, E3) pede os títulos reais, "tirados dos metadados do PDF ou da primeira página e revisados à
mão": hoje o gerador vê só o nome do arquivo. Este script faz o rascunho. O título dos metadados vale quando
parece título (tem 4 palavras ou mais e não é nome de arquivo nem de ferramenta); senão, vale o texto de maior
fonte da primeira página. Cada entrada sai com `revisado: false` e a origem, e a revisão humana troca o que
estiver errado e marca `revisado: true`. Entrada já revisada não é sobrescrita.

    backend\\.venv\\Scripts\\python.exe tests\\benchmark\\draft_titles.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "source_pdfs"
TARGET = ROOT / "backend" / "app" / "rag" / "titles.json"
BASE_LIST = ROOT / "backend" / "app" / "rag" / "index_versions" / "pdf-v1.sha256"
_NOT_A_TITLE = re.compile(r"(\.pdf|\.docx?|microsoft word|untitled|^arxiv|latex|overleaf|^\s*$)", re.I)


def metadata_title(doc: fitz.Document) -> str | None:
    title = " ".join((doc.metadata or {}).get("title", "").split())
    return title if len(title.split()) >= 4 and not _NOT_A_TITLE.search(title) else None


def largest_font_text(page: fitz.Page) -> str | None:
    """As linhas com a maior fonte da página (o título, na primeira página de um artigo)."""
    spans = [(round(span["size"], 1), span["text"].strip(), line["bbox"][1])
             for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])
             for span in line["spans"] if span["text"].strip()]
    if not spans:
        return None
    biggest = max(size for size, _, _ in spans)
    lines: dict[float, list[str]] = {}
    for size, text, top in spans:
        if size == biggest:
            lines.setdefault(round(top), []).append(text)
    title = " ".join(" ".join(parts) for _, parts in sorted(lines.items()))
    return " ".join(title.split()) or None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = [l.split(None, 1)[1].strip() for l in BASE_LIST.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    table = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.is_file() else {}
    for name in names:
        stem = Path(name).stem
        if (table.get(stem) or {}).get("revisado"):
            continue
        with fitz.open(SOURCE / name) as doc:
            from_metadata = metadata_title(doc)
            title, origin = (from_metadata, "metadados") if from_metadata else (largest_font_text(doc[0]), "primeira página")
        table[stem] = {"titulo": title or stem, "origem": origin if title else "nome do arquivo", "revisado": False}
        print(f"[{table[stem]['origem']:15s}] {stem[:45]:45s} → {table[stem]['titulo'][:90]}")
    TARGET.write_text(json.dumps(dict(sorted(table.items())), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(table)} títulos em {TARGET.relative_to(ROOT)}; revisados: {sum(t['revisado'] for t in table.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
