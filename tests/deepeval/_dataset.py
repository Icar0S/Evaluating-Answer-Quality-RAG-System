"""Carrega o golden dataset público de goldens/dataset.json."""
from __future__ import annotations

from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden

DATASET_PATH = Path(__file__).parent / "goldens" / "dataset.json"


def load_goldens() -> list[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(DATASET_PATH))
    return dataset.goldens
