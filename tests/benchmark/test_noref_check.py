"""A conferência antes do H5b (tests/benchmark/noref_check.py): evidência no bloco cortado é evidência perdida."""
from __future__ import annotations

from pathlib import Path

from tests.benchmark import noref_check

BODY = "We evaluate retrieval augmented generation with a new metric called faithfulness of the answer."
REFS = "\n".join(f"[{i}] A. Author{i} et al. Title number {i}. In Proceedings of ICSE, 20{10 + i}." for i in range(1, 9))


def test_evidencia_no_bloco_de_referencias_e_perdida_e_fora_dele_nao(tmp_path: Path):
    (tmp_path / "artigo.md").write_text(f"# Intro\n{BODY}\n# References\n{REFS}\n", encoding="utf-8")
    goldens = [
        {"name": "ok", "source_document": "artigo.pdf", "context": [BODY]},
        {"name": "perdida", "source_document": "artigo.pdf", "context": ["A. Author3 et al. Title number 3"]},
        {"name": "fora", "source_document": None, "context": ["qualquer"]},
    ]

    checked, lost = noref_check.lost_evidence(goldens, tmp_path, {"artigo": "artigo.md"})

    assert checked == 2
    assert [item["golden"] for item in lost] == ["perdida"]
