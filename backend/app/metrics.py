"""Coleta de métricas de recursos (CPU/RAM/GPU) e estado da última geração.

Estado em memória, escopo de processo único — adequado para uma API local de
usuário único. GPU é opcional: se não houver NVIDIA/pynvml disponível, os
campos correspondentes ficam com available=False em vez de falhar.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import psutil

from app.config import get_settings

# psutil.cpu_percent(interval=None) só é significativo a partir da 2a chamada
# (a 1a fica sem baseline). Chamamos uma vez aqui para "aquecer" o medidor.
psutil.cpu_percent(interval=None)

_gpu_lock = threading.Lock()
_gpu_available = False
_gpu_handle = None
_gpu_name = None

try:
    import pynvml

    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _name = pynvml.nvmlDeviceGetName(_gpu_handle)
    _gpu_name = _name.decode() if isinstance(_name, bytes) else _name
    _gpu_available = True
except Exception:
    _gpu_available = False
    _gpu_handle = None
    pynvml = None  # type: ignore[assignment]

_state_lock = threading.Lock()
_active_generations = 0
_last_generation: dict[str, Any] | None = None

# Buffers para as sparklines (últimas ~30 amostras, preenchidos pelo loop do /ws/metrics).
_cpu_history: list[float] = []
_gpu_history: list[float] = []
_HISTORY_MAX_LEN = 30


def mark_generation_start() -> None:
    global _active_generations
    with _state_lock:
        _active_generations += 1


def mark_generation_end() -> None:
    global _active_generations
    with _state_lock:
        _active_generations = max(0, _active_generations - 1)


def record_generation_stats(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prompt_eval_duration_ns: int | None,
    eval_duration_ns: int | None,
    total_duration_ns: int | None,
) -> None:
    """Registra as métricas da última geração, derivadas da resposta do Ollama."""
    tokens_per_second = None
    if completion_tokens and eval_duration_ns and eval_duration_ns > 0:
        tokens_per_second = round(completion_tokens / (eval_duration_ns / 1e9), 2)

    global _last_generation
    with _state_lock:
        _last_generation = {
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
            "total_latency_ms": round(total_duration_ns / 1e6, 2) if total_duration_ns else None,
            "prompt_eval_latency_ms": round(prompt_eval_duration_ns / 1e6, 2) if prompt_eval_duration_ns else None,
            "generation_latency_ms": round(eval_duration_ns / 1e6, 2) if eval_duration_ns else None,
        }


def _gpu_snapshot() -> dict[str, Any]:
    if not _gpu_available or _gpu_handle is None:
        return {"available": False}
    try:
        with _gpu_lock:
            util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        return {
            "available": True,
            "name": _gpu_name,
            "utilization_percent": util.gpu,
            "vram_used_gb": round(mem.used / (1024**3), 2),
            "vram_total_gb": round(mem.total / (1024**3), 2),
        }
    except Exception:
        return {"available": False}


def get_snapshot() -> dict[str, Any]:
    settings = get_settings()
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    gpu = _gpu_snapshot()

    with _state_lock:
        status = "processing" if _active_generations > 0 else "idle"
        last_generation = _last_generation

    _cpu_history.append(cpu_percent)
    if len(_cpu_history) > _HISTORY_MAX_LEN:
        _cpu_history.pop(0)
    gpu_util = gpu["utilization_percent"] if gpu.get("available") else 0.0
    _gpu_history.append(gpu_util)
    if len(_gpu_history) > _HISTORY_MAX_LEN:
        _gpu_history.pop(0)

    return {
        "generation_model": settings.generation_model,
        "status": status,
        "cpu_percent": cpu_percent,
        "ram": {
            "used_gb": round(vm.used / (1024**3), 2),
            "total_gb": round(vm.total / (1024**3), 2),
            "percent": vm.percent,
        },
        "gpu": gpu,
        "last_generation": last_generation,
        "cpu_history": list(_cpu_history),
        "gpu_history": list(_gpu_history),
    }
