"""Carregamento da suíte de 30 casos e das assertivas determinísticas."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.mutation.runner import jsonl, paths

# §6.1 do protocolo — particionamento em classes de equivalência sobre o corpus.
CLASS_QUOTAS: dict[str, int] = {
    "fato_direto": 6,
    "fato_distribuido": 5,
    "filtro_condicional": 4,
    "fora_do_corpus": 5,
    "fora_de_escopo": 3,
    "ambiguo": 3,
    "rastreabilidade": 4,
}

SUITE_SIZE = sum(CLASS_QUOTAS.values())

# Marcador usado nos campos ainda não escritos pelo pesquisador.
PLACEHOLDER = "PREENCHER"


@dataclass
class Case:
    case_id: str
    case_class: str
    status: str
    question: str
    expected_output: str
    expected_behavior: str
    evidence_role: str
    evidence_hint: str
    targets: list[str]
    notes: str = ""

    @property
    def is_draft(self) -> bool:
        if self.status != "ready":
            return True
        return PLACEHOLDER in self.question or PLACEHOLDER in self.expected_output


def load_cases(path: Path | None = None) -> list[Case]:
    records = jsonl.read(path or paths.GOLDEN)
    return [
        Case(
            case_id=record["case_id"],
            case_class=record["class"],
            status=record.get("status", "draft"),
            question=record["question"],
            expected_output=record["expected_output"],
            expected_behavior=record.get("expected_behavior", "answer"),
            evidence_role=record.get("evidence_role", "any"),
            evidence_hint=record.get("evidence_hint", ""),
            targets=list(record.get("targets", [])),
            notes=record.get("notes", ""),
        )
        for record in records
    ]


def load_assertions(path: Path | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw = yaml.safe_load((path or paths.ASSERTIONS).read_text(encoding="utf-8"))
    return raw.get("cases", {}) or {}, raw.get("defaults", {}) or {}


def ready_cases(cases: list[Case]) -> list[Case]:
    return [case for case in cases if not case.is_draft]
