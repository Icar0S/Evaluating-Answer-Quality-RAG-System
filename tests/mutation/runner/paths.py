"""Caminhos canônicos do estudo de mutação, todos derivados de tests/mutation/.

Um único lugar para responder "onde fica X" evita que script e README discordem
sobre a localização de um artefato — e é o que permite rodar qualquer script
destes de qualquer diretório de trabalho.
"""
from __future__ import annotations

from pathlib import Path

MUTATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MUTATION_ROOT.parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

CONFIG_DIR = MUTATION_ROOT / "config"
STUDY_CONFIG = CONFIG_DIR / "study.yaml"
CODEBOOK = CONFIG_DIR / "codebook.yaml"

CORPUS_DIR = MUTATION_ROOT / "corpus"
CORPUS_SOURCES = CORPUS_DIR / "sources.yaml"
CORPUS_BASE = CORPUS_DIR / "base"
CORPUS_VARIANTS = CORPUS_DIR / "variants"
CORPUS_MANIFEST = CORPUS_DIR / "manifest.json"

OPERATORS_DIR = MUTATION_ROOT / "operators"
OPERATOR_CATALOG = OPERATORS_DIR / "catalog.yaml"

SUITE_DIR = MUTATION_ROOT / "suite"
GOLDEN = SUITE_DIR / "golden.jsonl"
ASSERTIONS = SUITE_DIR / "assertions.yaml"

CALIBRATION_DIR = MUTATION_ROOT / "calibration"
CALIBRATION_PAIRS = CALIBRATION_DIR / "pairs.jsonl"
CALIBRATION_TAU = CALIBRATION_DIR / "tau.json"

RESULTS_DIR = MUTATION_ROOT / "results"
RUNS_JSONL = RESULTS_DIR / "runs.jsonl"
VERDICTS_JSONL = RESULTS_DIR / "verdicts.jsonl"
COST_JSONL = RESULTS_DIR / "cost.jsonl"
COHERENCE_JSONL = RESULTS_DIR / "coherence.jsonl"
COHERENCE_BLIND_JSONL = RESULTS_DIR / "coherence_blind.jsonl"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# work/ é descartável: tudo aqui é reconstruível a partir de corpus/base/ e do
# catálogo de operadores, e por isso não é versionado.
WORK_DIR = MUTATION_ROOT / "work"
WORK_CORPUS = WORK_DIR / "corpus"
WORK_INDEX_SRC = WORK_DIR / "index_src"
WORK_VECTOR_STORE = WORK_DIR / "vector_store"
WORK_STATE = WORK_DIR / "state.json"


def use_results_dir(directory: "Path | str") -> None:
    """Redireciona todos os artefatos de resultado para outro diretório.

    Existe por um motivo só: permitir analisar um conjunto de resultados que nao
    e o oficial -- tipicamente os dados sinteticos de
    tools/make_synthetic_results.py, usados para validar a analise antes da
    campanha. Sem isso, testar a analise exigiria sobrescrever results/.
    """
    global RESULTS_DIR, RUNS_JSONL, VERDICTS_JSONL, COST_JSONL
    global COHERENCE_JSONL, COHERENCE_BLIND_JSONL, TABLES_DIR, FIGURES_DIR

    RESULTS_DIR = Path(directory)
    if not RESULTS_DIR.is_absolute():
        RESULTS_DIR = PROJECT_ROOT / RESULTS_DIR
    RUNS_JSONL = RESULTS_DIR / "runs.jsonl"
    VERDICTS_JSONL = RESULTS_DIR / "verdicts.jsonl"
    COST_JSONL = RESULTS_DIR / "cost.jsonl"
    COHERENCE_JSONL = RESULTS_DIR / "coherence.jsonl"
    COHERENCE_BLIND_JSONL = RESULTS_DIR / "coherence_blind.jsonl"
    TABLES_DIR = RESULTS_DIR / "tables"
    FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, WORK_DIR, CALIBRATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def add_backend_to_syspath() -> None:
    """Torna `import app.*` possível sem instalar o backend como pacote.

    Mesmo truque do pytest.ini (`pythonpath = backend`), repetido aqui porque os
    scripts do estudo rodam fora do pytest.
    """
    import sys

    backend = str(BACKEND_DIR)
    if backend not in sys.path:
        sys.path.insert(0, backend)
