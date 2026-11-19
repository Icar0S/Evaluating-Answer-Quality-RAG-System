"""Títulos dos artigos e o cabeçalho contextual do chunk (CHUNK_HEADER; E3 do plano-benchmark.md).

A tabela backend/app/rag/titles.json tem um título por documento (pelo nome sem
extensão), rascunhado por tests/benchmark/draft_titles.py e revisado à mão
(`revisado`). Documento fora da tabela fica com o nome do arquivo sem extensão.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

TITLES_PATH = Path(__file__).resolve().parent / "titles.json"


@lru_cache(maxsize=1)
def load_titles() -> dict[str, str]:
    if not TITLES_PATH.is_file():
        return {}
    return {stem: entry["titulo"] for stem, entry in json.loads(TITLES_PATH.read_text(encoding="utf-8")).items()}


def titles_sha256() -> str | None:
    return hashlib.sha256(TITLES_PATH.read_bytes()).hexdigest() if TITLES_PATH.is_file() else None


def header(document: str, page: int | None) -> str:
    """"Artigo: <título> · pág. N" (sem a página no Markdown)."""
    title = load_titles().get(Path(document).stem, Path(document).stem)
    return f"Artigo: {title}" if page is None else f"Artigo: {title} · pág. {page}"
