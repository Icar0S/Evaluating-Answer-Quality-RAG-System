"""Ponte entre o harness de mutação e o SUT (backend/app).

Três responsabilidades, nesta ordem de importância:

1. **Reconfigurar o SUT sem editar o SUT.** Todo operador do catálogo é, no fim,
   um delta sobre `app.config.Settings` (mais, para a camada de corpus, uma
   mudança nos arquivos indexados). O harness aplica esse delta via
   `set_settings_overrides`, roda, e limpa. Nenhum arquivo de `backend/app/` é
   reescrito durante a campanha — o que elimina a classe de falha "o mutante
   continuou aplicado depois do revert", a mais cara de diagnosticar depois.

2. **Isolar o estudo dos dados do usuário.** O índice do estudo vive em
   `tests/mutation/work/vector_store`, nunca em `data/vector_store`. Rodar a
   campanha não invalida o assistente que a pessoa usa no dia a dia.

3. **Reindexar só quando precisa.** Reindexar 5-10 PDFs custa minutos; 18
   mutantes x 30 casos não podem pagar isso por invocação. A assinatura de índice
   resume tudo que afeta o índice (parâmetros de chunking, modelo de embedding e
   o conteúdo de work/index_src); se não mudou, o índice é reaproveitado.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import cost, paths
from .console import get_logger

paths.add_backend_to_syspath()

from app import providers  # noqa: E402  (precisa do sys.path ajustado acima)
from app.config import get_settings, set_settings_overrides  # noqa: E402
from app.rag import generation, retrieval, vector_store  # noqa: E402
from app.rag.ingestion import ingest_documents  # noqa: E402

logger = get_logger()

# Parâmetros cuja mudança obriga reindexação (os demais são de tempo de consulta).
INDEX_AFFECTING_KEYS = (
    "chunk_size_tokens",
    "chunk_overlap_tokens",
    "chunk_boundary_offset_tokens",
    "embedding_model",
    "vector_store_collection",
)


@dataclass
class StudyConfig:
    raw: dict[str, Any]

    @property
    def baseline_overrides(self) -> dict[str, Any]:
        return dict(self.raw["baseline_config"])

    @property
    def execution(self) -> dict[str, Any]:
        return dict(self.raw["execution"])

    @property
    def oracles(self) -> dict[str, Any]:
        return dict(self.raw["oracles"])

    @property
    def models(self) -> dict[str, Any]:
        return dict(self.raw.get("models", {}))


def load_study_config(path: Path | None = None) -> StudyConfig:
    config_path = path or paths.STUDY_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return StudyConfig(raw=raw)


def _resolve_paths(overrides: dict[str, Any]) -> dict[str, Any]:
    """Converte caminhos relativos do study.yaml em absolutos da raiz do projeto.

    `Settings` já resolve caminhos relativos contra PROJECT_ROOT, mas ser
    explícito aqui evita depender de onde o script foi chamado.
    """
    resolved = dict(overrides)
    for key in ("vector_store_dir", "source_pdfs_dir", "logs_dir"):
        value = resolved.get(key)
        if value and not os.path.isabs(str(value)):
            resolved[key] = str((paths.PROJECT_ROOT / str(value)).resolve())
    return resolved


@contextmanager
def sut_configuration(overrides: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Aplica overrides sobre o SUT durante o bloco e restaura ao sair.

    Restaurar também limpa o cliente do ChromaDB: ele guarda o caminho do índice
    em estado de módulo e, sem o reset, continuaria apontando para o índice do
    estudo depois do bloco.
    """
    resolved = _resolve_paths(overrides)
    set_settings_overrides(resolved)
    vector_store.reset_client()
    # A geração remota (Mac mini) não expõe temperatura, seed nem medição de GPU
    # local — o estudo é declaradamente sobre o provider local.
    providers.set_active_provider("local")
    try:
        yield resolved
    finally:
        set_settings_overrides({})
        vector_store.reset_client()


def _hash_directory(directory: Path) -> str:
    """Hash estável do conteúdo indexável de um diretório (nome + bytes)."""
    digest = hashlib.sha256()
    if directory.is_dir():
        for file_path in sorted(directory.glob("*.pdf")):
            digest.update(file_path.name.encode("utf-8"))
            digest.update(hashlib.sha256(file_path.read_bytes()).digest())
    return digest.hexdigest()


def index_signature(overrides: dict[str, Any], index_src: Path | None = None) -> str:
    """Resume tudo que determina o conteúdo do índice."""
    source = index_src or paths.WORK_INDEX_SRC
    parts: dict[str, Any] = {key: overrides.get(key) for key in INDEX_AFFECTING_KEYS}
    parts["corpus"] = _hash_directory(source)
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def read_state() -> dict[str, Any]:
    if paths.WORK_STATE.exists():
        return json.loads(paths.WORK_STATE.read_text(encoding="utf-8"))
    return {}


def write_state(state: dict[str, Any]) -> None:
    paths.WORK_STATE.parent.mkdir(parents=True, exist_ok=True)
    paths.WORK_STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_index(overrides: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Garante que o índice corresponde à configuração corrente. Idempotente.

    Deve ser chamado **dentro** de `sut_configuration`, porque a ingestão lê o
    modelo de embedding e os parâmetros de chunking da configuração ativa.
    """
    signature = index_signature(overrides)
    state = read_state()
    if not force and state.get("index_signature") == signature and vector_store.collection_count() > 0:
        return {"reindexed": False, "index_signature": signature, "chunks": vector_store.collection_count()}

    started = time.perf_counter()
    result = ingest_documents(source_dir=paths.WORK_INDEX_SRC, reset=True)
    duration_s = time.perf_counter() - started
    state["index_signature"] = signature
    state["indexed_at"] = datetime.now(timezone.utc).isoformat()
    state["chunks"] = result.get("chunks_created", 0)
    state["documents"] = result.get("documents_processed", 0)
    write_state(state)
    logger.info(
        "Reindexado: %s documentos, %s chunks, %.1fs (assinatura %s)",
        result.get("documents_processed"),
        result.get("chunks_created"),
        duration_s,
        signature,
    )
    return {
        "reindexed": True,
        "index_signature": signature,
        "chunks": result.get("chunks_created", 0),
        "duration_s": round(duration_s, 2),
    }


@dataclass
class Invocation:
    """Uma execução completa do pipeline para um caso — vira uma linha de runs.jsonl."""

    run_id: str
    mutant_id: str
    operator: str | None
    case_id: str
    repetition: int
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_documents: list[str]
    retrieval_context: list[str]
    similarity_scores: list[float]
    grounded: bool
    model: str
    embed_model: str
    temperature: float | None
    seed: int | None
    top_k: int
    tokens_in: int | None
    tokens_out: int | None
    wall_ms: float
    retrieval_ms: float
    generation_ms: float
    gpu_s: float | None
    wh: float | None
    error: str | None = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def invoke(
    *,
    mutant_id: str,
    operator: str | None,
    case_id: str,
    question: str,
    repetition: int,
) -> Invocation:
    """Roda um caso no SUT já configurado e devolve o registro da invocação.

    Erros de rede/modelo viram um registro com `error` preenchido em vez de
    exceção: uma campanha de 3.000 invocações não pode morrer na invocação 1.847,
    e um registro de erro é informação — sai do numerador e do denominador na
    análise, mas fica auditável no log.
    """
    settings = get_settings()
    run_id = f"{mutant_id}-{case_id}-r{repetition}"

    with cost.measure() as holder:
        try:
            chunks, retrieval_ms = retrieval.retrieve(question)
            result = generation.generate_answer(question, chunks)
            error = None
        except Exception as exc:  # noqa: BLE001 — ver docstring
            logger.warning("Falha em %s: %s", run_id, exc)
            chunks, retrieval_ms = [], 0.0
            result = {
                "answer": "",
                "latency_ms": 0.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "grounded": False,
            }
            error = f"{type(exc).__name__}: {exc}"

    sample = holder[0]
    return Invocation(
        run_id=run_id,
        mutant_id=mutant_id,
        operator=operator,
        case_id=case_id,
        repetition=repetition,
        question=question,
        answer=result["answer"],
        retrieved_chunk_ids=[c["chunk_id"] for c in chunks],
        retrieved_documents=sorted({c["document"] for c in chunks}),
        retrieval_context=[c["text"] for c in chunks],
        similarity_scores=[c["similarity_score"] for c in chunks],
        grounded=bool(result.get("grounded")),
        model=settings.generation_model,
        embed_model=settings.embedding_model,
        temperature=settings.generation_temperature,
        seed=settings.generation_seed,
        top_k=settings.top_k,
        tokens_in=result.get("prompt_tokens"),
        tokens_out=result.get("completion_tokens"),
        wall_ms=round(sample.wall_ms, 2),
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(result.get("latency_ms", 0.0), 2),
        gpu_s=sample.gpu_s,
        wh=sample.wh,
        error=error,
    )
