"""Providers Ollama: listagem, troca em runtime e reflexo no /health.

Cobre as validações feitas à mão ao construir a feature (curl em /providers,
/providers/active e /health alternando entre local e remoto).
"""
from __future__ import annotations

from conftest import (
    LOCAL_BASE_URL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_MODEL,
    REMOTE_BASE_URL,
    REMOTE_LABEL,
    REMOTE_MODEL,
)


def test_lista_so_o_local_quando_o_remoto_nao_esta_configurado(client):
    body = client.get("/providers").json()

    assert [p["name"] for p in body["providers"]] == ["local"]
    assert body["active"] == "local"


def test_lista_os_dois_providers_quando_o_remoto_esta_configurado(client, remote_configured):
    body = client.get("/providers").json()

    assert [p["name"] for p in body["providers"]] == ["local", "remote"]
    remote = body["providers"][1]
    assert remote["label"] == REMOTE_LABEL
    assert remote["base_url"] == REMOTE_BASE_URL
    assert remote["generation_model"] == REMOTE_MODEL
    # Embeddings são sempre locais (a API remota não expõe essa rota) — o campo
    # aqui existe só para exibição/log, e reflete o modelo local mesmo no remoto.
    assert remote["embedding_model"] == LOCAL_EMBEDDING_MODEL


def test_reachability_reflete_o_estado_real_de_cada_provider(client, remote_configured, ollama):
    # Cenário do dia a dia: Ollama local no ar, Mac mini fora.
    body = client.get("/providers").json()
    por_nome = {p["name"]: p for p in body["providers"]}

    assert por_nome["local"]["reachable"] is True
    assert por_nome["remote"]["reachable"] is False

    # Servidor volta ao ar -> a listagem acompanha, sem reiniciar o backend.
    ollama.set_online(REMOTE_BASE_URL)
    por_nome = {p["name"]: p for p in client.get("/providers").json()["providers"]}
    assert por_nome["remote"]["reachable"] is True


def test_troca_o_provider_ativo_em_runtime(client, remote_configured):
    resposta = client.post("/providers/active", json={"name": "remote"})

    assert resposta.status_code == 200
    assert resposta.json()["name"] == "remote"
    assert client.get("/providers").json()["active"] == "remote"

    client.post("/providers/active", json={"name": "local"})
    assert client.get("/providers").json()["active"] == "local"


def test_provider_desconhecido_devolve_400(client, remote_configured):
    resposta = client.post("/providers/active", json={"name": "inexistente"})

    assert resposta.status_code == 400
    assert "inexistente" in resposta.json()["detail"]


def test_nao_deixa_ativar_o_remoto_se_ele_nao_esta_configurado(client):
    resposta = client.post("/providers/active", json={"name": "remote"})

    assert resposta.status_code == 400
    assert client.get("/providers").json()["active"] == "local"


def test_health_descreve_o_provider_ativo(client, remote_configured):
    local = client.get("/health").json()
    assert local["status"] == "ok"
    assert local["ollama_reachable"] is True
    assert local["generation_model"] == LOCAL_MODEL
    assert local["embedding_model"] == LOCAL_EMBEDDING_MODEL

    client.post("/providers/active", json={"name": "remote"})

    remoto = client.get("/health").json()
    assert remoto["generation_model"] == REMOTE_MODEL
    # Mac mini fora do ar: a API se declara degradada em vez de fingir que está ok.
    assert remoto["status"] == "degraded"
    assert remoto["ollama_reachable"] is False


def test_health_checa_o_endereco_do_provider_ativo(client, remote_configured, ollama):
    client.post("/providers/active", json={"name": "remote"})
    ollama.requests.clear()

    client.get("/health")

    assert ollama.requests_to(REMOTE_BASE_URL), "o /health deveria consultar o Ollama remoto"
    assert not ollama.requests_to(LOCAL_BASE_URL), "o /health nao deveria consultar o local"
