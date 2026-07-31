"""/chat: roteamento por provider, metadados e log estruturado.

O ponto central aqui não é a qualidade da resposta (isso é escopo do RAGAS na
Fase 2), e sim *para onde* a chamada foi e o que ficou registrado — trocar de
provider precisa redirecionar de verdade a geração e os embeddings.
"""
from __future__ import annotations

import json

from conftest import (
    LOCAL_BASE_URL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_MODEL,
    REMOTE_BASE_URL,
    REMOTE_EMBEDDING_MODEL,
    REMOTE_MODEL,
)


def _corpo_da_chamada(request):
    return json.loads(request.content)


def test_usa_o_provider_local_por_padrao(client, ollama):
    resposta = client.post("/chat", json={"question": "O que e RAG?"})

    assert resposta.status_code == 200
    metadata = resposta.json()["metadata"]
    assert metadata["provider"] == "local"
    assert metadata["model"] == LOCAL_MODEL
    assert metadata["embedding_model"] == LOCAL_EMBEDDING_MODEL

    assert not ollama.requests_to(REMOTE_BASE_URL)
    chamadas = {r.url.path for r in ollama.requests_to(LOCAL_BASE_URL)}
    assert {"/api/embed", "/api/chat"} <= chamadas


def test_troca_de_provider_redireciona_geracao_e_embeddings(client, remote_configured, ollama):
    ollama.set_online(REMOTE_BASE_URL)
    client.post("/providers/active", json={"name": "remote"})
    ollama.requests.clear()

    resposta = client.post("/chat", json={"question": "O que e RAG?"})

    assert resposta.status_code == 200
    metadata = resposta.json()["metadata"]
    assert metadata["provider"] == "remote"
    assert metadata["model"] == REMOTE_MODEL
    assert metadata["embedding_model"] == REMOTE_EMBEDDING_MODEL

    # Nada pode ter vazado para o Ollama local.
    assert not ollama.requests_to(LOCAL_BASE_URL)

    geracao = [r for r in ollama.requests_to(REMOTE_BASE_URL) if r.url.path == "/api/chat"]
    embeddings = [r for r in ollama.requests_to(REMOTE_BASE_URL) if r.url.path == "/api/embed"]
    assert _corpo_da_chamada(geracao[0])["model"] == REMOTE_MODEL
    assert _corpo_da_chamada(embeddings[0])["model"] == REMOTE_EMBEDDING_MODEL


def test_registra_o_provider_no_log_de_interacoes(client, tmp_path):
    client.post("/chat", json={"question": "O que e RAG?"})

    linhas = (tmp_path / "logs" / "interactions.jsonl").read_text(encoding="utf-8").splitlines()
    registro = json.loads(linhas[-1])

    # Sem o provider no log, não dá para segmentar as métricas da Fase 2 por
    # modelo/máquina que gerou cada resposta.
    assert registro["metadata"]["provider"] == "local"
    assert registro["question"] == "O que e RAG?"
    assert registro["sources"]


def test_devolve_503_com_o_endereco_do_provider_quando_ele_esta_fora(client, remote_configured, ollama):
    ollama.set_online(REMOTE_BASE_URL)
    client.post("/providers/active", json={"name": "remote"})
    ollama.set_offline(REMOTE_BASE_URL)

    resposta = client.post("/chat", json={"question": "O que e RAG?"})

    assert resposta.status_code == 503
    # A mensagem precisa dizer *qual* provider caiu — foi o que motivou trocar o
    # texto genérico de "rode ollama serve" por algo endereçado.
    assert REMOTE_BASE_URL in resposta.json()["detail"]


def test_alimenta_o_monitor_com_as_estatisticas_da_geracao(client):
    assert client.get("/metrics").json()["last_generation"] is None

    client.post("/chat", json={"question": "O que e RAG?"})

    ultima = client.get("/metrics").json()["last_generation"]
    assert ultima["completion_tokens"] == 42
    assert ultima["tokens_per_second"] == 21.0  # 42 tokens / 2s de eval_duration
