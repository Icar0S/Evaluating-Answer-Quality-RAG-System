"""O protocolo da pergunta única em tests/generate_answers.py (plano-escopo-final.md, §4).

Uma pergunta, N respostas com sementes fixas, o gerador descarregado do Ollama
antes de cada uma e os mesmos trechos em todas; e o controle quente x frio.
Herméticos: um http.server local faz o papel da API do RAG e do Ollama do
backend ao mesmo tempo, e registra a ordem das chamadas.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemma3:4b"
GOLDENS = [
    {"name": "alvo", "input": "Quais vieses?", "expected_output": "Posição.", "context": ["posição"],
     "source_document": "mtbench.md"},
    {"name": "outra", "input": "Outra pergunta?", "expected_output": "Sim.", "context": ["sim"],
     "source_document": "outro.md"},
]
SOURCES = [
    {"chunk_id": "mtbench::c0", "document": "mtbench.md", "page": None, "section": None, "text": "vieses de posição",
     "similarity_score": 0.74},
]


@pytest.fixture
def ga():
    spec = importlib.util.spec_from_file_location("generate_answers_protocolo", ROOT / "tests" / "generate_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    """API do RAG (/health, /chat) e Ollama (/api/ps, /api/generate, /api/show) no mesmo endereço.

    `answer(question, seed)` monta o texto de cada resposta; `sources(n)` os trechos da n-ésima chamada ao /chat;
    `echo_seed=False` imita um backend antigo, que ignora o campo `seed`; `stuck=True`, um modelo que não sai
    da memória.
    """
    state = {
        "calls": [], "loaded": set(), "echo_seed": True, "stuck": False, "chats": 0,
        "answer": lambda question, seed: f"resposta de {question} com semente {seed}",
        "sources": lambda n: SOURCES,
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - nome da API do http.server
            state["calls"].append((self.path, None))
            if self.path == "/health":
                self._send(200, {"status": "ok", "generation_model": MODEL, "embedding_model": "nomic-embed-text",
                                 "vector_store_documents": 857, "rag_index": "md-v1"})
            elif self.path == "/api/ps":
                self._send(200, {"models": [{"name": m, "model": m} for m in sorted(state["loaded"])]})
            else:
                self._send(404, {"detail": "rota não existe"})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            state["calls"].append((self.path, body))
            if self.path == "/api/generate":
                if body.get("keep_alive") == 0 and not state["stuck"]:
                    state["loaded"].discard(body["model"])
                self._send(200, {"model": body["model"], "done": True, "done_reason": "unload"})
            elif self.path == "/api/show":
                self._send(200, {"parameters": 'stop                "<end_of_turn>"\ntemperature    1\ntop_k  64\n'
                                               'top_p  0.95'})
            elif self.path == "/chat":
                state["chats"] += 1
                state["loaded"].add(MODEL)
                seed = body.get("seed")
                self._send(200, {
                    "answer": state["answer"](body["question"], seed),
                    "sources": state["sources"](state["chats"]),
                    "metadata": {"model": MODEL, "embedding_model": "nomic-embed-text", "provider": "local",
                                 "prompt_version": "v1", "latency_ms": 10.0, "retrieval_latency_ms": 1.0,
                                 "generation_latency_ms": 9.0, "prompt_tokens": 900, "completion_tokens": 40,
                                 "top_k": 4, "rag_index": "md-v1", "temperature": None,
                                 "seed": seed if state["echo_seed"] else None},
                    "grounded": True,
                })
            else:
                self._send(404, {"detail": "rota não existe"})

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    # A ordem do que muda o estado: descarregar e perguntar (o /api/show e o /api/ps só leem).
    state["paths"] = lambda: [path for path, _ in state["calls"] if path not in ("/health", "/api/ps", "/api/show")]
    yield state
    httpd.shutdown()


@pytest.fixture
def goldens_file(tmp_path: Path) -> Path:
    path = tmp_path / "goldens_md.json"
    path.write_text(json.dumps(GOLDENS), encoding="utf-8")
    return path


def _run(ga, server, goldens_file, tmp_path, *extra: str) -> tuple[int, dict]:
    out = tmp_path / "r.json"
    args = ga.build_parser().parse_args(["--out", str(out), "--api", server["url"], "--goldens",
                                         f"md={goldens_file}", *extra])
    code = ga.api_main(args)
    return code, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else {})


# ------------------------------------------------------------------ o plano


def _ns(**kw) -> argparse.Namespace:
    base = {"only": [], "repeticoes": None, "sementes": None, "estado_limpo": False, "controle_estado": False,
            "aquecimento": None, "shuffle_seed": None}
    return argparse.Namespace(**{**base, **kw})


def test_sem_as_opcoes_do_protocolo_o_plano_e_o_de_sempre(ga):
    goldens = [("md", g) for g in GOLDENS]

    plan = ga.plan_asks(goldens, _ns())
    shuffled = ga.plan_asks(goldens, _ns(shuffle_seed=3))

    assert [(a.index, a.name, a.seed, a.unload, a.estado) for a in plan] == [
        (0, "alvo", None, False, None), (1, "outra", None, False, None)]
    assert [a.index for a in shuffled] == ga.processing_order(2, 3)
    assert not ga.protocol_active(_ns())


def test_uma_resposta_por_semente_com_o_nome_numerado(ga):
    plan = ga.plan_asks([("md", g) for g in GOLDENS], _ns(only=["alvo"], sementes="1-5", estado_limpo=True))

    assert [a.name for a in plan] == [f"alvo#{k}" for k in range(1, 6)]
    assert [a.seed for a in plan] == [1, 2, 3, 4, 5]
    assert all(a.unload and a.estado == "frio" and a.index == 0 for a in plan)
    assert [a.repeticao for a in plan] == [1, 2, 3, 4, 5]


def test_sementes(ga):
    assert ga.parse_seeds("1-5") == [1, 2, 3, 4, 5]
    assert ga.parse_seeds("1,3, 7") == [1, 3, 7]
    assert ga.parse_seeds("0") == [0]
    for ruim in ("", "5-1", "1,1", "a", "1,-2"):
        with pytest.raises(ValueError):
            ga.parse_seeds(ruim)


def test_erros_do_plano_saem_antes_de_perguntar(ga, server, goldens_file, tmp_path):
    for extra in (["--only", "nao-existe"], ["--sementes", "1-3", "--repeticoes", "5"],
                  ["--controle-estado", "--ollama", "x"], ["--repeticoes", "0"]):
        code, _ = _run(ga, server, goldens_file, tmp_path, *extra)
        assert code == 2, extra
    # Pela API, descarregar o gerador precisa saber onde está o Ollama dele.
    code, _ = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--estado-limpo")
    assert code == 2
    assert server["chats"] == 0


# ---------------------------------------------------------- as 5 respostas


def test_cinco_respostas_com_semente_estado_limpo_e_os_mesmos_trechos(ga, server, goldens_file, tmp_path):
    server["loaded"].add(MODEL)  # o gerador já estava carregado, quente de outra conversa

    code, data = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--sementes", "1-5", "--estado-limpo",
                      "--ollama", server["url"])

    assert code == 0
    answers = data["answers"]
    assert [a["name"] for a in answers] == [f"alvo#{k}" for k in range(1, 6)]
    assert [a["seed"] for a in answers] == [1, 2, 3, 4, 5] == [a["semente_pedida"] for a in answers]
    assert all(a["golden"] == "alvo" and a["estado"] == "frio" for a in answers)
    assert len({a["actual_output"] for a in answers}) == 5
    # Antes de cada resposta, o gerador sai da memória: descarregar, perguntar, descarregar, perguntar...
    assert server["paths"]() == ["/api/generate", "/chat"] * 5
    assert [path for path, _ in server["calls"]].count("/api/show") == 1
    unloads = [body for path, body in server["calls"] if path == "/api/generate"]
    assert all(body == {"model": MODEL, "keep_alive": 0} for body in unloads)
    # Cada /chat é uma chamada nova, só com a pergunta e a semente: nenhum histórico.
    chats = [body for path, body in server["calls"] if path == "/chat"]
    assert chats == [{"question": "Quais vieses?", "seed": k} for k in range(1, 6)]
    # Os trechos: o mesmo hash nas 5.
    assert len({a["sources_sha256"] for a in answers}) == 1
    protocol = data["protocolo"]
    assert protocol["trechos"] == {"alvo": {"sha256": answers[0]["sources_sha256"], "iguais": True, "respostas": 5}}
    assert (protocol["sementes"], protocol["repeticoes"], protocol["estado_limpo"]) == ([1, 2, 3, 4, 5], 5, True)
    assert protocol["interrompido"] is None
    # O backend não mandou temperatura: vale a padrão do modelo, lida do /api/show.
    assert (protocol["temperatura"], protocol["temperatura_fonte"]) == (1.0, "padrão do modelo (/api/show)")
    assert protocol["parametros_do_modelo"] == {"stop": "<end_of_turn>", "temperature": 1, "top_k": 64, "top_p": 0.95}


def test_trechos_diferentes_param_o_protocolo(ga, server, goldens_file, tmp_path):
    outro = [{**SOURCES[0], "chunk_id": "outro::c9", "text": "outra coisa"}]
    server["sources"] = lambda n: SOURCES if n < 3 else outro

    code, data = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--sementes", "1-5")

    assert code == ga.EXIT_PROTOCOL_BROKEN == 3
    assert len(data["answers"]) == 3  # parou na que divergiu
    assert data["protocolo"]["trechos"]["alvo"]["iguais"] is False
    assert "alvo#3 recebeu outros trechos" in data["protocolo"]["interrompido"]


def test_backend_que_ignora_a_semente_para_o_protocolo(ga, server, goldens_file, tmp_path):
    # Um backend sem o campo `seed` (anterior à fase 1) aceita o corpo e ignora a semente em silêncio.
    server["echo_seed"] = False

    code, data = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--sementes", "1-5")

    assert code == 3
    assert len(data["answers"]) == 1
    assert "não aplicou a semente" in data["protocolo"]["interrompido"]


def test_modelo_que_nao_sai_da_memoria_interrompe_sem_perguntar(ga, server, goldens_file, tmp_path, monkeypatch):
    server["loaded"].add(MODEL)
    server["stuck"] = True
    real_unload = ga.unload_model
    monkeypatch.setattr(ga, "unload_model", lambda url, model: real_unload(url, model, wait_seconds=0.5))

    code, data = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--sementes", "1-2", "--estado-limpo",
                      "--ollama", server["url"])

    assert code == 1
    assert server["chats"] == 0 and data["answers"] == []
    assert "continua carregado" in data["protocolo"]["interrompido"]


def test_modelo_ja_fora_da_memoria_nao_e_descarregado_de_novo(ga, server):
    assert ga.unload_model(server["url"], MODEL) == 0.0
    assert server["paths"]() == []
    server["loaded"].add(MODEL)
    assert ga.unload_model(server["url"], "gemma3:4b") >= 0.0
    assert server["paths"]() == ["/api/generate"] and not server["loaded"]


# ------------------------------------------------------ controle quente x frio


def test_controle_quente_frio_pergunta_na_ordem_certa_e_resume_por_semente(ga, server, goldens_file, tmp_path):
    # O estado muda a resposta da semente 2 (como o cache de prefixo faria), e não a da semente 1.
    after_other = {"value": False}

    def answer(question, seed):
        text = f"resposta de {question} com semente {seed}"
        if question == "Outra pergunta?":
            after_other["value"] = True
            return text
        warm = after_other["value"]
        after_other["value"] = False
        return text + (" (quente)" if warm and seed == 2 else "")

    server["answer"] = answer

    code, data = _run(ga, server, goldens_file, tmp_path, "--only", "alvo", "--controle-estado", "--sementes", "1-2",
                      "--ollama", server["url"])

    assert code == 0
    assert [a["name"] for a in data["answers"]] == [
        "alvo#s1-frio", "alvo#s1-frio-2", "outra#s1-aquecimento", "alvo#s1-quente",
        "alvo#s2-frio", "alvo#s2-frio-2", "outra#s2-aquecimento", "alvo#s2-quente"]
    # O quente é a única pergunta sem descarregar antes: vem logo depois do aquecimento.
    assert server["paths"]()[:8] == ["/chat", "/api/generate", "/chat", "/api/generate", "/chat", "/chat",
                                     "/api/generate", "/chat"]
    summary = data["controle_estado"]
    assert summary["frio_se_reproduz"] == "2/2"
    assert summary["estado_mudou_a_resposta"] == "1/2"
    by_seed = {row["seed"]: row for row in summary["por_semente"]}
    assert by_seed[1]["quente_igual_ao_frio"] is True and by_seed[2]["quente_igual_ao_frio"] is False
    assert 0.8 < by_seed[2]["semelhanca_quente_frio"] < 1.0
    # O aquecimento é outra pergunta, com outros trechos: fica fora da conferência.
    assert data["protocolo"]["trechos"] == {"alvo": {"sha256": data["answers"][0]["sources_sha256"], "iguais": True,
                                                     "respostas": 6}}


def test_controle_precisa_de_uma_pergunta_e_de_outra_para_aquecer(ga):
    goldens = [("md", g) for g in GOLDENS]
    with pytest.raises(ValueError, match="uma pergunta só"):
        ga.plan_asks(goldens, _ns(controle_estado=True))
    with pytest.raises(ValueError, match="aquecimento"):
        ga.plan_asks(goldens, _ns(controle_estado=True, only=["alvo"], aquecimento="alvo"))
    with pytest.raises(ValueError, match="aquecimento precisa"):
        ga.plan_asks([("md", GOLDENS[0])], _ns(controle_estado=True, only=["alvo"]))
    plan = ga.plan_asks(goldens, _ns(controle_estado=True, only=["alvo"]))
    assert [a.seed for a in plan] == [1, 1, 1, 1]  # sem --sementes, a semente 1


# ------------------------------------------------------------- in-process


def test_in_process_a_semente_vai_ao_gerador_e_volta_no_registro(monkeypatch):
    from tests.benchmark.test_generate_answers import CHUNKS, GOLDEN

    spec = importlib.util.spec_from_file_location("generate_answers_ip", ROOT / "tests" / "generate_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []
    monkeypatch.setattr(module.retrieval, "retrieve", lambda question, top_k=None: (CHUNKS, 1.0))
    monkeypatch.setattr(module.generation, "generate_answer",
                        lambda q, c, seed=None: received.append(seed) or {"answer": "ok", "latency_ms": 1.0,
                                                                           "seed": seed, "temperature": 1.0})

    record = module.answer_record("md", GOLDEN, GOLDEN["input"], 4, seed=7)

    assert received == [7]
    assert (record["seed"], record["temperature"]) == (7, 1.0)
