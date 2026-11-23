"""A máquina acordada durante uma rodada de horas: bloquear a suspensão, ver a tomada e perceber se dormiu.

Por que existe: na rodada de 23/11/2026 (out/20261123T110351), o notebook saiu da tomada, suspendeu por quase
duas horas no meio do DeepEval e voltou com a GPU em erro. A rodada perdeu notas do DeepEval e quase todas as do
Jev, e nada no registro dizia o motivo: ele só apareceu no log de eventos do Windows.

Três peças, todas do processo do rodar.py (nada muda na configuração de energia da máquina):

- `power()`: está na tomada? Na bateria, o rodar.py não começa (`--na-bateria` para insistir): a GPU cai de
  desempenho e a métrica mais lenta do DeepEval (~175 s na tomada) passa do teto de 300 s;
- `keep_awake()`: pede ao Windows para não suspender por inatividade enquanto a rodada dura. Não vence a tampa
  fechada, o botão de energia nem a bateria no fim;
- `SuspensionWatch`: uma batida a cada 5 s; um salto de mais de 60 s no relógio é a máquina que dormiu (ou o
  processo congelado). O salto vai ao registro do passo em que aconteceu.

Só biblioteca padrão. Fora do Windows, `power()` devolve "não sei" e `keep_awake()` não faz nada.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def power() -> dict[str, Any]:
    """`{"tomada": True | False | None, "bateria": 0-100 | None}`; None é "não sei"."""
    if os.name != "nt":
        return {"tomada": None, "bateria": None}
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("system_flag", ctypes.c_ubyte), ("life_time", ctypes.c_uint32), ("full_life_time", ctypes.c_uint32)]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {"tomada": None, "bateria": None}
    return {"tomada": {0: False, 1: True}.get(status.ac_line),
            "bateria": status.percent if status.percent <= 100 else None}


@contextmanager
def keep_awake() -> Iterator[bool]:
    """Segura a suspensão por inatividade enquanto o bloco dura; devolve se o sistema aceitou o pedido."""
    if os.name != "nt":
        yield False
        return
    import ctypes

    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes, set_state.restype = [ctypes.c_uint32], ctypes.c_uint32
    accepted = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    try:
        yield accepted
    finally:
        set_state(ES_CONTINUOUS)


class SuspensionWatch:
    """Percebe a suspensão pelo salto do relógio entre duas batidas. `take()` devolve os saltos e os esquece."""

    def __init__(self, interval: float = 5.0, threshold: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        self.interval, self.threshold, self.clock = interval, threshold, clock
        self._last = clock()
        self._found: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> None:
        now = self.clock()
        with self._lock:
            jump = now - self._last
            if jump > self.threshold:
                self._found.append({"de": datetime.fromtimestamp(self._last).isoformat(timespec="seconds"),
                                    "ate": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                                    "segundos": round(jump, 1)})
            self._last = now

    def take(self) -> list[dict[str, Any]]:
        self.tick()  # o passo pode terminar antes da próxima batida
        with self._lock:
            found, self._found = self._found, []
        return found

    def start(self) -> "SuspensionWatch":
        def beat() -> None:
            while not self._stop.wait(self.interval):
                self.tick()

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
