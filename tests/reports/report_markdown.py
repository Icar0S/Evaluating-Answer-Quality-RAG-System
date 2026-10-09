"""Renderiza os blocos de report_document.py em Markdown (GitHub-flavored).

As figuras são referenciadas por caminho relativo ao .md (pasta figuras/ ao
lado), então a pasta de saída pode ser movida ou anexada inteira.
"""
from __future__ import annotations

from pathlib import Path

from report_document import Block, Bullets, Callout, Figure, Heading, PageBreak, Para, Table

_ALIGN = {"l": ":---", "r": "---:", "c": ":---:"}


def _text(text: str) -> str:
    # Respostas do LLM podem trazer "<algo>": o GitHub trataria como tag HTML e
    # sumiria com o texto. O relatório não usa HTML próprio, então escapar é seguro.
    return str(text).replace("<", "&lt;")


def _cell(text: str) -> str:
    return _text(" ".join(str(text).split()).replace("|", "\\|"))


def _table(block: Table) -> list[str]:
    align = block.align or ["l"] * len(block.headers)
    lines = [
        "| " + " | ".join(_cell(h) for h in block.headers) + " |",
        "| " + " | ".join(_ALIGN.get(a, ":---") for a in align) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in block.rows]
    if block.caption:
        lines += ["", f"_{_text(block.caption)}_"]
    return lines


def render_markdown(blocks: list[Block], path: Path) -> Path:
    out: list[str] = []
    for block in blocks:
        if isinstance(block, Heading):
            out += [f"{'#' * block.level} {_text(block.text)}", ""]
        elif isinstance(block, Para):
            out += [_text(block.text), ""]
        elif isinstance(block, Bullets):
            out += [f"- {_text(item)}" for item in block.items] + [""]
        elif isinstance(block, Table):
            out += _table(block) + [""]
        elif isinstance(block, Figure):
            rel = Path(block.path).relative_to(path.parent).as_posix()
            out += [f"![{_text(block.caption)}]({rel})", "", f"_{_text(block.caption)}_", ""]
        elif isinstance(block, Callout):
            out += [f"> **⚠ {_text(block.title)}**", ">"] + [f"> - {_text(item)}" for item in block.items] + [""]
        elif isinstance(block, PageBreak):
            out += ["---", ""]
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    return path
