"""tests/generate_answers.py --api: as respostas pela API do RAG no ar (o modo do container de testes).

O RAG é testado como caixa-preta: o script não importa o backend nem abre o
Chroma, e o arquivo sai no mesmo formato do modo in-process, para os juízes e o
relatório lerem igual. Herméticos: a "API" é um http.server local.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = {"name": "g1", "input": "Quantas classes?", "expected_output": "16.", "context": ["16 classes"],
          "source_document": "artigo"}
CHAT_REPLY = {
    "answer": "São 16 classes.",
    "sources": [
        {"chunk_id": "artigo::p4::c0", "document": "artigo.pdf", "page": 4, "section": None, "text": "16 classes",
         "similarity_score": 0.7},
        {"chunk_id": "outro::c3", "document": "outro.md", "page": None, "section": None, "text": "nada",
         "similarity_score": 0.5},
    ],
    "metadata": {"model": "gemma3:4b", "embedding_model": "nomic-embed-text", "provider": "local",
                 "prompt_version": "v1", "latency_ms": 1322.21, "retrieval_latency_ms": 87.65,
                 "generation_latency_ms": 1234.56, "prompt_tokens": 900, "completion_tokens": 40, "top_k": 4,
                 "rag_index": "pdf-v1"},
    "grounded": True,
}
HEALTH = {"status": "ok", "ollama_reachable": True, "generation_model": "gemma3:4b",
          "embedding_model": "nomic-embed-text", "vector_store_documents": 894, "rag_index": "pdf-v1"}


@pytest.fixture
def fake_api():
    """GET /health e POST /chat. `replies` é a fila de respostas do /chat, (status, corpo); vazia = CHAT_REPLY."""
    state = {"health": dict(HEALTH), "replies": [], "questions": []}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - nome da API do http.server
            if self.path == "/health":
                self._send(200, state["health"])
            else:
                self._send(404, {"detail": "rota não existe"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            state["questions"].append(json.loads(self.rfile.read(length))["question"])
            status, body = state["replies"].pop(0) if state["replies"] else (200, CHAT_REPLY)
            self._send(status, body)

        def log_message(self, *args):  # sem barulho na saída do pytest
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()


@pytest.fixture
def ga():
    spec = importlib.util.spec_from_file_location("generate_answers_api", ROOT / "tests" / "generate_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def goldens_file(tmp_path: Path) -> Path:
    path = tmp_path / "goldens.json"
    path.write_text(json.dumps([GOLDEN, {**GOLDEN, "name": "g2", "input": "Outra?"}]), encoding="utf-8")
    return path


def _args(ga, *extra: str):
    return ga.build_parser().parse_args(list(extra))


def test_grava_no_mesmo_formato_do_modo_in_process(ga, fake_api, goldens_file, tmp_path):
    out = tmp_path / "respostas.json"

    code = ga.api_main(_args(ga, "--out", str(out), "--api", fake_api["url"] + "/", "--goldens",
                             f"pdf={goldens_file}"))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert fake_api["questions"] == ["Quantas classes?", "Outra?"]
    # O topo vem do metadata do /chat: é o que diz qual RAG respondeu.
    assert (data["provider"], data["generation_model"], data["embedding_model"]) == (
        "local", "gemma3:4b", "nomic-embed-text")
    assert (data["top_k"], data["prompt_version"], data["rag_index"]) == (4, "v1", "pdf-v1")
    assert data["api_url"] == fake_api["url"]
    assert data["collected_by"].startswith(f"POST {fake_api['url']}/chat")
    assert data["n_cases"] == 2
    record = data["answers"][0]
    assert (record["name"], record["origin"], record["expected_output"]) == ("g1", "pdf", "16.")
    assert record["actual_output"] == "São 16 classes."
    assert record["retrieval_context"] == ["16 classes", "nada"]
    assert record["sources"] == ["artigo.pdf (pág. 4)", "outro.md"]
    assert record["source_documents"] == ["artigo.pdf", "outro.md"]
    assert record["chunk_ids"] == ["artigo::p4::c0", "outro::c3"]
    assert (record["retrieval_ms"], record["generation_ms"]) == (87.7, 1234.6)
    assert record["error"] is None and record["seconds"] >= 0 and record["position"] == 1


def test_tenta_duas_vezes_e_o_erro_do_backend_vira_registro(ga, fake_api, goldens_file, tmp_path):
    timeout = (504, {"detail": "Tempo limite excedido ao gerar a resposta."})
    fake_api["replies"] = [timeout, timeout]
    out = tmp_path / "r.json"

    code = ga.api_main(_args(ga, "--out", str(out), "--api", fake_api["url"], "--goldens", f"pdf={goldens_file}",
                             "--limit", "1"))

    record = json.loads(out.read_text(encoding="utf-8"))["answers"][0]
    assert len(fake_api["questions"]) == 2
    assert record["actual_output"] is None and record["retrieval_context"] == []
    assert record["error"] == "ApiError: HTTP 504: Tempo limite excedido ao gerar a resposta."
    assert code == 1  # todas falharam


def test_segunda_tentativa_que_da_certo_vale(ga, fake_api, goldens_file, tmp_path):
    fake_api["replies"] = [(502, {"detail": "Erro ao gerar resposta"})]
    out = tmp_path / "r.json"

    assert ga.api_main(_args(ga, "--out", str(out), "--api", fake_api["url"], "--goldens", f"pdf={goldens_file}",
                             "--limit", "1")) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["answers"][0]["error"] is None


def test_api_fora_do_ar_ou_indice_vazio_para_antes_de_perguntar(ga, fake_api, goldens_file, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{sock.getsockname()[1]}"
    assert ga.api_main(_args(ga, "--out", str(tmp_path / "a.json"), "--api", closed)) == 1

    fake_api["health"]["vector_store_documents"] = 0
    assert ga.api_main(_args(ga, "--out", str(tmp_path / "b.json"), "--api", fake_api["url"], "--goldens",
                             f"pdf={goldens_file}")) == 1
    assert fake_api["questions"] == []


def test_recusa_as_opcoes_que_so_valem_in_process(ga, fake_api, tmp_path):
    for extra in (["--index", "md-v1"], ["--set", "generation_seed=42"], ["--local"]):
        assert ga.api_main(_args(ga, "--out", str(tmp_path / "x.json"), "--api", fake_api["url"], *extra)) == 2
    assert fake_api["questions"] == []


def test_como_script_nao_importa_o_backend_nem_o_chroma(fake_api, goldens_file, tmp_path):
    out = tmp_path / "r.json"
    argv = ["generate_answers.py", "--out", str(out), "--api", fake_api["url"], "--goldens", f"pdf={goldens_file}"]
    code = "\n".join([
        "import runpy, sys",
        f"sys.argv = {argv!r}",
        "try:",
        f"    runpy.run_path({str(ROOT / 'tests' / 'generate_answers.py')!r}, run_name='__main__')",
        "except SystemExit as exc:",
        "    print('saida', exc.code)",
        "print('app' in sys.modules, 'chromadb' in sys.modules)",
    ])

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                            timeout=120)

    assert result.stdout.strip().splitlines()[-2:] == ["saida 0", "False False"], result.stdout + result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["n_cases"] == 2
