"""O quadro da stack em Docker (scripts/stack_status.py): serviço `status` e preflight dos testes.

Herméticos: os "serviços" são um servidor HTTP local da biblioteca padrão, e
nenhum container roda aqui. Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import stack_status  # noqa: E402

HEALTH_PDF = {"status": "ok", "ollama_reachable": True, "generation_model": "gemma3:4b",
              "embedding_model": "nomic-embed-text", "vector_store_documents": 894, "rag_index": "pdf-v1"}
HEALTH_MD = {**HEALTH_PDF, "vector_store_documents": 857, "rag_index": "md-v1"}
TAGS = {"models": [{"name": "gemma3:4b"}, {"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}]}
CHAT_PAGE = '<html><script src="components/chat.js"></script></html>'


@pytest.fixture
def services(monkeypatch):
    """Backend, backend-md, Ollama e frontend falsos, num servidor só, cada um num prefixo."""
    routes = {
        "/pdf/health": HEALTH_PDF,
        "/md/health": HEALTH_MD,
        "/ollama/api/tags": TAGS,
        "/front/chat.html": CHAT_PAGE,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - nome da API do http.server
            body = routes.get(self.path)
            if body is None:
                self.send_error(404)
                return
            data = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # sem barulho na saída do pytest
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    for name, prefix in (("PDF_API", "pdf"), ("MD_API", "md"), ("OLLAMA", "ollama"), ("FRONTEND", "front")):
        monkeypatch.setenv(f"STATUS_{name}", f"{base}/{prefix}")
    for name in ("DEMO_GENERATION_MODEL", "DEMO_JUDGE_MODEL", "DEMO_EMBEDDING_MODEL", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield routes
    server.shutdown()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_tudo_no_ar_mostra_os_enderecos_do_host_os_trechos_e_nenhum_aviso(services, capsys):
    code = stack_status.main([])

    out = capsys.readouterr().out
    assert code == 0
    # Os endereços que a pessoa abre no navegador, não os nomes de serviço da rede do Compose.
    assert "http://localhost:5510/chat.html?api=http://localhost:8011" in out
    assert "http://localhost:11435" in out
    assert "pdf-v1" in out and "894 trechos" in out
    assert "md-v1" in out and "857 trechos" in out
    assert "[ ok  ] juiz        qwen3:8b" in out
    assert "(nenhum)" in out
    assert "scripts\\docker_tests.bat --rapido" in out


def test_indice_vazio_e_modelo_faltando_sao_avisos_e_a_stack_segue(services, capsys):
    services["/md/health"] = {**HEALTH_MD, "vector_store_documents": 0}
    services["/ollama/api/tags"] = {"models": [{"name": "gemma3:4b"}, {"name": "nomic-embed-text:latest"}]}

    code = stack_status.main([])

    out = capsys.readouterr().out
    assert code == 0
    assert "Índice md-v1 VAZIO: faltam os documentos em data/new_papers_marks" in out
    assert "reindexar" in out
    assert "[FALTA] juiz        qwen3:8b" in out
    assert "Modelo qwen3:8b (juiz) não está no Ollama" in out


def test_servico_fora_do_ar_sai_com_1_e_diz_onde_olhar(services, monkeypatch, capsys):
    monkeypatch.setenv("STATUS_PDF_API", f"http://127.0.0.1:{_closed_port()}")

    code = stack_status.main([])

    out = capsys.readouterr().out
    assert code == 1
    assert "FORA DO AR" in out
    assert "docker compose logs backend" in out


def test_mesmo_modelo_para_gerar_e_julgar_e_aviso_de_self_grading(services, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_JUDGE_MODEL", "gemma3:4b")

    stack_status.main([])

    assert "self-grading" in capsys.readouterr().out


def test_preflight_diz_se_a_chave_existe_e_nunca_mostra_o_valor(services, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-segredo-de-teste-123")

    stack_status.main(["--preflight", "run_all", "--rapido"])

    out = capsys.readouterr().out
    assert "configurada: sim" in out
    assert "sk-segredo" not in out
    assert "Vai rodar: tests/run_all.py --rapido" in out
    assert "894 trechos" in out


def test_preflight_dos_hermeticos_nao_consulta_a_stack(monkeypatch, capsys):
    # Nenhum serviço no ar (os nomes da rede do Compose nem resolvem aqui): o modo não precisa deles.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(stack_status, "collect", lambda *a, **k: pytest.fail("hermeticos não consulta a stack"))

    assert stack_status.main(["--preflight", "hermeticos"]) == 0
    out = capsys.readouterr().out
    assert "não é necessária" in out
    assert "configurada: não" in out


@pytest.mark.parametrize(
    ("mode", "args", "has", "has_not"),
    [
        ("run_all", ["--rapido"], ["~3 min"], ["RAGAS", "DeepEval"]),
        ("run_all", ["--limite", "1", "--pular", "deepeval"], ["1 golden(s)", "RAGAS (controles + 1 golden(s)): ~3 min"],
         ["DeepEval ("]),
        ("run_all", [], ["10 golden(s)", "RAGAS (controles + 10 golden(s)): ~16 min", "DeepEval (10 golden(s))",
                         "JevEval ao vivo: pulado"], []),
        ("run_all", ["--qualidade-pytest"], ["--qualidade-pytest: pulado no Docker"], []),
        ("demo", ["--goldens", "1"], ["2 perguntas de referência em cada", "4 respostas"], []),
        ("demo-rapido", [], ["~5 min"], []),
    ],
)
def test_estimativa_de_cada_modo_sai_da_tabela_do_readme(mode, args, has, has_not):
    what, lines = stack_status.plan(mode, args, typesafe=False)

    text = what + "\n" + "\n".join(lines)
    for piece in has:
        assert piece in text
    for piece in has_not:
        assert piece not in text
