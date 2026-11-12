"""Os avisos do scripts/ingest_documents.py: acervo vazio e pasta diferente do índice.

O docker compose roda o script a cada subida (serviços ingest e ingest-md, com
--if-empty). Sem estes avisos, um acervo que faltou subia o RAG com o índice
vazio em silêncio, e os testes ao vivo se auto-pulavam sem dizer por quê.
Herméticos, como o resto de tests/api: a pasta e o manifesto ficam em tmp, o
Chroma e o Ollama vêm das fixtures do conftest.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag import ingestion

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli(monkeypatch, tmp_path, capsys):
    """O script como módulo, com a pasta de documentos em tmp e uma coleção que não é versão registrada."""
    spec = importlib.util.spec_from_file_location("ingest_documents_cli", ROOT / "scripts" / "ingest_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("SOURCE_PDFS_DIR", str(docs))
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "colecao_de_teste")
    monkeypatch.delenv("INGEST_COMPOSE_SERVICE", raising=False)
    get_settings.cache_clear()

    def run(*args: str) -> str:
        """Roda o main() com estes argumentos e devolve o que ele imprimiu."""
        monkeypatch.setattr(sys, "argv", ["ingest_documents.py", *args])
        module.main()
        return capsys.readouterr().out

    module.docs = docs
    module.run = run
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(documents: list[tuple[str, str]]) -> None:
    path = ingestion.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created_at": "2026-10-13T17:20:00+00:00",
                                "documents": [{"file": f, "sha256": s} for f, s in documents]}), encoding="utf-8")


def test_sem_documentos_avisa_grande_diz_o_que_fazer_e_nao_falha(cli):
    out = cli.run()  # sem SystemExit: a subida do docker compose não para

    assert "Status: no_documents" in out
    assert "[AVISO] Nenhum documento para indexar em" in out
    assert str(cli.docs) in out
    assert "índice VAZIO" in out and "auto-pular" in out
    # Coleção que não é versão registrada: o comando vai sem --index.
    assert "python scripts/ingest_documents.py   (no Windows" in out


def test_no_docker_o_aviso_mostra_o_comando_de_reindexar_do_compose(cli, monkeypatch):
    monkeypatch.setenv("INGEST_COMPOSE_SERVICE", "ingest-md")

    out = cli.run()

    assert "docker compose run --rm ingest-md python scripts/ingest_documents.py" in out
    assert "scripts\\docker_tests.bat reindexar" in out
    assert "./scripts/docker_tests.sh reindexar" in out


def test_indice_populado_e_pasta_diferente_do_manifesto_avisa_sem_reindexar(cli, monkeypatch):
    (cli.docs / "a.md").write_text("igual", encoding="utf-8")
    (cli.docs / "b.md").write_text("mudou", encoding="utf-8")
    (cli.docs / "c.md").write_text("novo", encoding="utf-8")
    _manifest([("a.md", _sha(cli.docs / "a.md")), ("b.md", "0" * 64), ("d.md", "1" * 64)])
    monkeypatch.setattr(ingestion, "ingest_documents", lambda **_: pytest.fail("não pode reindexar sozinho"))

    out = cli.run("--if-empty")  # o conftest diz que o índice tem 42 trechos

    assert "Índice já populado (42 chunks)" in out
    assert "[AVISO] A pasta" in out and "2026-10-13" in out
    assert "1 novo(s), 1 fora da pasta, 1 alterado(s)" in out
    assert "novo:      c.md" in out and "saiu:      d.md" in out and "alterado:  b.md" in out
    assert "não é automática" in out


def test_indice_populado_e_pasta_igual_ao_manifesto_fica_em_silencio(cli):
    (cli.docs / "a.md").write_text("igual", encoding="utf-8")
    _manifest([("a.md", _sha(cli.docs / "a.md"))])

    out = cli.run("--if-empty")

    assert "Índice já populado" in out
    assert "[AVISO]" not in out


def test_indice_sem_manifesto_diz_que_nao_da_para_conferir(cli):
    out = cli.run("--if-empty")

    assert "não tem manifesto" in out
    assert "[AVISO]" not in out


def test_versao_registrada_so_compara_os_documentos_da_lista(cli, monkeypatch):
    # Na pasta de uma versão com lista, os arquivos de outras versões não contam como "novos".
    (cli.docs / "da_lista.md").write_text("x", encoding="utf-8")
    (cli.docs / "de_outra_versao.md").write_text("y", encoding="utf-8")
    _manifest([("da_lista.md", _sha(cli.docs / "da_lista.md"))])
    monkeypatch.setattr(ingestion, "select_documents",
                        lambda source_dir=None: (cli.docs, [cli.docs / "da_lista.md"], {}))

    assert cli.folder_divergence() == {"created_at": "2026-10-13T17:20:00+00:00", "added": [], "removed": [],
                                           "changed": []}
