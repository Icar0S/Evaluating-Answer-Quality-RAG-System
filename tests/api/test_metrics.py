"""Métricas do monitor: o que existe no provider local e o que não existe no remoto.

`psutil`/`pynvml` só enxergam a máquina onde o backend roda. Com o provider
remoto ativo, CPU/RAM/GPU não são do modelo que está de fato respondendo, então
a API sinaliza isso em vez de devolver números da máquina errada.
"""
from __future__ import annotations

from conftest import REMOTE_MODEL


def test_local_expoe_as_metricas_do_host(client):
    body = client.get("/metrics").json()

    assert body["provider"] == "local"
    assert body["host_metrics_available"] is True
    assert body["ram"]["total_gb"] > 0


def test_remoto_marca_as_metricas_de_host_como_indisponiveis(client, remote_configured):
    client.post("/providers/active", json={"name": "remote"})

    body = client.get("/metrics").json()

    assert body["provider"] == "remote"
    assert body["host_metrics_available"] is False
    assert body["generation_model"] == REMOTE_MODEL
    # Zerado em vez de repetir os números da máquina local, que seriam enganosos.
    assert body["cpu_percent"] == 0.0
    assert body["ram"]["total_gb"] == 0.0
    assert body["gpu"]["available"] is False


def test_historico_das_sparklines_nao_recebe_zeros_do_provider_remoto(client, remote_configured):
    # Duas amostras com o local ativo alimentam o histórico...
    client.get("/metrics")
    client.get("/metrics")
    tamanho_com_local = len(client.get("/metrics").json()["cpu_history"])
    assert tamanho_com_local == 3

    # ...e o provider remoto não deve empurrar zeros para dentro dele, senão a
    # sparkline do local apareceria despencando ao trocar de aba no monitor.
    client.post("/providers/active", json={"name": "remote"})
    client.get("/metrics")
    client.get("/metrics")

    assert len(client.get("/metrics").json()["cpu_history"]) == tamanho_com_local


def test_status_volta_para_ocioso_quando_nao_ha_geracao(client):
    assert client.get("/metrics").json()["status"] == "idle"
