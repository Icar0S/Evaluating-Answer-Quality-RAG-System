"""Saída de console segura em Windows e um logger único para os scripts do estudo.

O console padrão do Windows usa cp1252: um print com "τ" ou "✂" derruba o script
com UnicodeEncodeError no meio de uma campanha de horas. Reconfigurar o stdout
para UTF-8 com errors="replace" custa duas linhas e elimina a classe inteira de
falha.
"""
from __future__ import annotations

import logging
import sys


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def get_logger(name: str = "mutation") -> logging.Logger:
    configure_console()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
