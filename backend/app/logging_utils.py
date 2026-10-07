"""Logging estruturado em JSON Lines de cada interação do RAG.

Cada linha de logs/interactions.jsonl é um registro completo e independente,
pensado para ser lido depois por pandas/DeepEval (ver tests/deepeval/) sem
parsing adicional.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

_write_lock = threading.Lock()

logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_stream_handler)


def _interactions_file() -> Path:
    settings = get_settings()
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    return settings.logs_path / "interactions.jsonl"


def new_interaction_id() -> str:
    return str(uuid.uuid4())


def log_interaction(record: dict[str, Any]) -> None:
    """Acrescenta um registro de interação ao arquivo JSONL, thread-safe."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    line = json.dumps(payload, ensure_ascii=False)
    with _write_lock:
        with _interactions_file().open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_interactions() -> list[dict[str, Any]]:
    """Lê todas as interações logadas, usado para agregações (ex.: /stats)."""
    path = _interactions_file()
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
