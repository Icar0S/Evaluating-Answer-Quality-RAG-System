"""/chat: roteamento por provider, metadados e log estruturado.

O ponto central aqui não é a qualidade da resposta (isso é escopo de
tests/deepeval/, Fase 2), e sim *para onde* a chamada foi e o que ficou
registrado — trocar de
provider precisa redirecionar a geração de verdade, mas os embeddings ficam
sempre no Ollama local (a API remota não expõe essa rota, ver
llm-api-referencia.md e app/providers.py).
"""
from __future__ import annotations

import json

from app.config import get_settings
from conftest import (
    LOCAL_BASE_URL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_MODEL,
    REMOTE_API_KEY,
    REMOTE_BASE_URL,
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


def test_troca_de_provider_redireciona_so_a_geracao_embeddings_ficam_locais(client, remote_configured, ollama):
    ollama.set_online(REMOTE_BASE_URL)
    client.post("/providers/active", json={"name": "remote"})
    ollama.requests.clear()

    resposta = client.post("/chat", json={"question": "O que e RAG?"})

    assert resposta.status_code == 200
    metadata = resposta.json()["metadata"]
    assert metadata["provider"] == "remote"
    assert metadata["model"] == REMOTE_MODEL
    # Embeddings continuam com o modelo local mesmo com o provider remoto ativo —
    # a API remota não tem rota de embeddings, e misturar espaços de embedding
    # corromperia a busca por similaridade no vector store.
    assert metadata["embedding_model"] == LOCAL_EMBEDDING_MODEL

    # A geração foi para a API remota, via o protocolo dela (/v1/chat, não /api/chat)...
    geracao = [r for r in ollama.requests_to(REMOTE_BASE_URL) if r.url.path == "/v1/chat"]
    assert len(geracao) == 1
    assert _corpo_da_chamada(geracao[0])["model"] == REMOTE_MODEL
    assert geracao[0].headers.get("authorization") == f"Bearer {REMOTE_API_KEY}"
    # Sem corpus_id: o RAG usado é o nosso (retrieval local), não o deles.
    assert "corpus_id" not in _corpo_da_chamada(geracao[0])

    # ...e os embeddings continuaram 100% no Ollama local, nada foi ao host remoto.
    embeddings_remoto = [r for r in ollama.requests_to(REMOTE_BASE_URL) if "embed" in r.url.path]
    assert not embeddings_remoto
    embeddings_local = [r for r in ollama.requests_to(LOCAL_BASE_URL) if r.url.path == "/api/embed"]
    assert len(embeddings_local) == 1
    assert _corpo_da_chamada(embeddings_local[0])["model"] == LOCAL_EMBEDDING_MODEL


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


def test_devolve_429_com_retry_after_quando_a_fila_remota_esta_cheia(client, remote_configured, ollama, monkeypatch):
    import httpx

    ollama.set_online(REMOTE_BASE_URL)
    client.post("/providers/active", json={"name": "remote"})
    handle_original = ollama.handle

    def handle_fila_cheia(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat":
            return httpx.Response(429, headers={"Retry-After": "5"}, json={"detail": "fila cheia"})
        return handle_original(request)

    monkeypatch.setattr(ollama, "handle", handle_fila_cheia)

    resposta = client.post("/chat", json={"question": "O que e RAG?"})

    assert resposta.status_code == 429
    assert "5" in resposta.json()["detail"]


def test_alimenta_o_monitor_com_as_estatisticas_da_geracao(client):
    assert client.get("/metrics").json()["last_generation"] is None

    client.post("/chat", json={"question": "O que e RAG?"})

    ultima = client.get("/metrics").json()["last_generation"]
    assert ultima["completion_tokens"] == 42
    assert ultima["tokens_per_second"] == 21.0  # 42 tokens / 2s de eval_duration


def _corpo_da_geracao_local(ollama):
    geracao = [r for r in ollama.requests_to(LOCAL_BASE_URL) if r.url.path == "/api/chat"]
    assert len(geracao) == 1
    return _corpo_da_chamada(geracao[0])


def test_nao_envia_think_por_padrao(client, ollama):
    # Sem GENERATION_THINK o payload fica como sempre foi: o modelo decide se
    # raciocina. Mudar isso alteraria o baseline do estudo de mutacao.
    client.post("/chat", json={"question": "O que e RAG?"})

    assert "think" not in _corpo_da_geracao_local(ollama)


def test_generation_think_false_desliga_o_raciocinio(client, ollama, monkeypatch):
    monkeypatch.setenv("GENERATION_THINK", "false")
    get_settings.cache_clear()

    client.post("/chat", json={"question": "O que e RAG?"})

    assert _corpo_da_geracao_local(ollama)["think"] is False


# ---------------------------------------------- semente por requisição (D2, plano-escopo-final.md)


def test_sem_semente_nem_temperatura_nao_envia_options_e_o_metadata_diz_isso(client, ollama):
    # O chat da tela não manda semente: o payload fica como sempre foi, e o modelo usa o padrão dele.
    metadata = client.post("/chat", json={"question": "O que e RAG?"}).json()["metadata"]

    assert "options" not in _corpo_da_geracao_local(ollama)
    assert (metadata["temperature"], metadata["seed"]) == (None, None)


def test_a_semente_da_requisicao_vale_so_naquela_chamada_no_lugar_da_do_servidor(client, ollama, monkeypatch):
    monkeypatch.setenv("GENERATION_SEED", "42")
    monkeypatch.setenv("GENERATION_TEMPERATURE", "1")
    get_settings.cache_clear()

    com_semente = client.post("/chat", json={"question": "O que e RAG?", "seed": 3}).json()["metadata"]
    opcoes = _corpo_da_geracao_local(ollama)["options"]
    ollama.requests.clear()
    sem_semente = client.post("/chat", json={"question": "O que e RAG?"}).json()["metadata"]

    assert (opcoes["seed"], opcoes["temperature"]) == (3, 1.0)
    assert (com_semente["seed"], com_semente["temperature"]) == (3, 1.0)
    # A chamada seguinte, sem o campo, volta para a semente do servidor.
    assert _corpo_da_geracao_local(ollama)["options"]["seed"] == 42
    assert sem_semente["seed"] == 42


def test_semente_zero_e_semente(client, ollama):
    # 0 é uma semente válida, não "sem semente".
    metadata = client.post("/chat", json={"question": "O que e RAG?", "seed": 0}).json()["metadata"]

    assert _corpo_da_geracao_local(ollama)["options"] == {"seed": 0}
    assert metadata["seed"] == 0


def test_semente_fora_da_faixa_e_rejeitada_antes_de_chamar_o_modelo(client, ollama):
    for semente in (-1, 2**31, "abc"):
        assert client.post("/chat", json={"question": "O que e RAG?", "seed": semente}).status_code == 422
    assert not [r for r in ollama.requests if r.url.path == "/api/chat"]


def test_o_provider_remoto_ignora_a_semente_e_o_metadata_nao_finge_que_ela_valeu(client, remote_configured, ollama):
    ollama.set_online(REMOTE_BASE_URL)
    client.post("/providers/active", json={"name": "remote"})

    metadata = client.post("/chat", json={"question": "O que e RAG?", "seed": 3}).json()["metadata"]

    geracao = [r for r in ollama.requests_to(REMOTE_BASE_URL) if r.url.path == "/v1/chat"]
    assert "seed" not in _corpo_da_chamada(geracao[0]) and "options" not in _corpo_da_chamada(geracao[0])
    assert (metadata["provider"], metadata["seed"], metadata["temperature"]) == ("remote", None, None)
