"""Leitura e escrita de JSON Lines com escrita atômica por linha.

Toda a análise do artigo é refeita a partir destes arquivos, nunca de estado
volátil de execução (§4.1 do protocolo). Por isso a escrita é append imediato e
com flush: uma campanha interrompida na hora 6 preserva as 6 horas anteriores.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    return _locks.setdefault(str(path), threading.Lock())


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def write_all(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read(path: Path) -> list[dict[str, Any]]:
    return list(iter_read(path))


def iter_read(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} não é JSON válido: {exc}") from exc
