"""Contabilidade de custo por invocação: tempo de parede, GPU-segundo e energia.

RQ3 pede "tempo de parede, tokens, GPU-segundo e energia". Tokens vêm da própria
resposta do Ollama; os outros três saem daqui.

Sobre GPU-segundo e Wh: em vez de estimar a partir do TDP nominal da placa,
amostramos utilização e potência instantânea via NVML durante a chamada e
integramos. Com amostragem a 10 Hz numa chamada de ~10 s são ~100 amostras, o
suficiente para uma integral trapezoidal estável. Sem NVIDIA/pynvml, os campos
voltam None em vez de zero — zero seria indistinguível de "GPU ociosa" e
contaminaria a média da RQ3.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

try:
    import pynvml

    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _name = pynvml.nvmlDeviceGetName(_HANDLE)
    GPU_NAME = _name.decode() if isinstance(_name, bytes) else _name
    NVML_AVAILABLE = True
except Exception:  # driver ausente, GPU não-NVIDIA, WSL sem passthrough...
    pynvml = None  # type: ignore[assignment]
    _HANDLE = None
    GPU_NAME = None
    NVML_AVAILABLE = False

SAMPLE_INTERVAL_S = 0.1


@dataclass
class CostSample:
    """Resultado de uma janela de medição."""

    wall_ms: float
    gpu_s: float | None = None
    wh: float | None = None
    gpu_util_mean: float | None = None
    power_w_mean: float | None = None
    samples: int = 0


@dataclass
class _Sampler:
    stop: threading.Event = field(default_factory=threading.Event)
    timestamps: list[float] = field(default_factory=list)
    utils: list[float] = field(default_factory=list)
    powers: list[float] = field(default_factory=list)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_HANDLE).gpu
            except Exception:
                util = 0.0
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(_HANDLE) / 1000.0
            except Exception:
                power_w = 0.0
            self.timestamps.append(time.perf_counter())
            self.utils.append(float(util))
            self.powers.append(float(power_w))
            self.stop.wait(SAMPLE_INTERVAL_S)


def _integrate(timestamps: list[float], values: list[float]) -> float:
    """Integral trapezoidal de values(t) em unidades de value·segundo."""
    total = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


@contextmanager
def measure() -> Iterator[list[CostSample]]:
    """Mede uma janela. O resultado é anexado à lista devolvida pelo `with`.

    A lista existe porque um context manager não pode devolver um valor calculado
    na saída; quem chama lê `holder[0]` depois do bloco.
    """
    holder: list[CostSample] = []
    start = time.perf_counter()

    sampler = None
    thread = None
    if NVML_AVAILABLE:
        sampler = _Sampler()
        thread = threading.Thread(target=sampler.run, name="gpu-sampler", daemon=True)
        thread.start()

    try:
        yield holder
    finally:
        wall_ms = (time.perf_counter() - start) * 1000.0
        if sampler is not None and thread is not None:
            sampler.stop.set()
            thread.join(timeout=2.0)

        if sampler is not None and len(sampler.timestamps) >= 2:
            # GPU-segundo = segundos ponderados pela utilização (util em 0..1).
            gpu_s = _integrate(sampler.timestamps, [u / 100.0 for u in sampler.utils])
            joules = _integrate(sampler.timestamps, sampler.powers)
            holder.append(
                CostSample(
                    wall_ms=wall_ms,
                    gpu_s=round(gpu_s, 4),
                    wh=round(joules / 3600.0, 6),
                    gpu_util_mean=round(sum(sampler.utils) / len(sampler.utils), 2),
                    power_w_mean=round(sum(sampler.powers) / len(sampler.powers), 2),
                    samples=len(sampler.timestamps),
                )
            )
        else:
            holder.append(CostSample(wall_ms=wall_ms))
