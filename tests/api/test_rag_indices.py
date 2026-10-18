"""Versões do índice (RAG_INDEX) e a ingestão de Markdown.

Herméticos: o Ollama é o mock do conftest e o Chroma é trocado por uma lista.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import RAG_INDICES, Settings, get_settings
from app.rag import generation, ingestion, vector_store


def test_sem_rag_index_vale_a_configuracao_historica():
    settings = Settings(_env_file=None)

    assert settings.index_name == "pdf-v1"
    assert settings.vector_store_collection == "rag_test_specialist"


def test_rag_index_fixa_colecao_pasta_e_modelo_de_embedding():
    settings = Settings(_env_file=None, rag_index="md-v1")
    spec = RAG_INDICES["md-v1"]

    assert settings.vector_store_collection == spec["collection"]
    assert settings.source_pdfs_dir == spec["source_dir"]
    # Índice só pode ser consultado com o modelo que o construiu.
    assert settings.embedding_model == spec["embedding_model"]


def test_rag_index_desconhecido_da_erro():
    with pytest.raises(ValueError):
        Settings(_env_file=None, rag_index="v99")


def test_health_e_chat_dizem_qual_indice_respondeu(client, monkeypatch):
    # Com dois RAGs no ar (a demonstração em Docker sobe um por acervo), é o
    # que os testes E2E conferem antes de atribuir uma resposta a um deles.
    assert client.get("/health").json()["rag_index"] == "pdf-v1"

    monkeypatch.setenv("RAG_INDEX", "md-v1")
    get_settings.cache_clear()

    assert client.get("/health").json()["rag_index"] == "md-v1"
    assert client.post("/chat", json={"question": "O que e RAG?"}).json()["metadata"]["rag_index"] == "md-v1"


def test_health_diz_de_quais_acervos_o_indice_tem_os_artigos(client, monkeypatch):
    # É por aqui que o teste E2E das perguntas de referência sabe se o certo é
    # responder (acervo do índice) ou se abster (acervo alheio).
    assert client.get("/health").json()["rag_index_origins"] == ["pdf"]

    monkeypatch.setenv("RAG_INDEX", "md-v2")
    get_settings.cache_clear()

    assert client.get("/health").json()["rag_index_origins"] == ["pdf", "md"]


def test_cada_versao_registrada_tem_lista_e_o_v2_contem_o_v1_inteiro():
    # As listas são versionadas: é o que faz pdf-v1 e pdf-v2 continuarem
    # diferentes lendo a mesma pasta, aqui e no Docker.
    lists = {name: ingestion.registered_documents(name) for name in RAG_INDICES}
    assert all(docs for docs in lists.values()), [name for name, docs in lists.items() if not docs]
    for v1, v2 in (("pdf-v1", "pdf-v2"), ("md-v1", "md-v2")):
        assert lists[v1].items() <= lists[v2].items(), f"{v1} não está inteiro (mesmo arquivo, mesmo hash) em {v2}"
        assert set(RAG_INDICES[v1]["origins"]) < set(RAG_INDICES[v2]["origins"])


@pytest.fixture
def test_version(tmp_path: Path, monkeypatch) -> Path:
    """Versão "teste-v1" registrada numa pasta temporária, com as listas também em tmp."""
    folder = tmp_path / "acervo"
    folder.mkdir()
    monkeypatch.setitem(
        RAG_INDICES,
        "teste-v1",
        {"collection": "teste_v1", "source_dir": str(folder), "embedding_model": "nomic-embed-text",
         "origins": ("pdf",), "description": "teste"},
    )
    monkeypatch.setattr(ingestion, "INDEX_VERSIONS_DIR", tmp_path / "index_versions")
    monkeypatch.setenv("RAG_INDEX", "teste-v1")
    get_settings.cache_clear()
    return folder


def test_register_grava_a_lista_da_pasta_uma_vez_so(test_version: Path):
    (test_version / "b.md").write_text("bbb", encoding="utf-8")
    (test_version / "A.md").write_text("aaa", encoding="utf-8")
    (test_version / "notas.txt").write_text("fora", encoding="utf-8")

    path = ingestion.register_documents("teste-v1", test_version)

    assert set(ingestion.registered_documents("teste-v1")) == {"A.md", "b.md"}
    assert path.read_text(encoding="utf-8").startswith("# teste-v1: 2 documento(s)")
    with pytest.raises(FileExistsError):
        ingestion.register_documents("teste-v1", test_version)


def test_ingestao_da_versao_so_pega_a_lista_e_avisa_o_que_faltou_ou_mudou(test_version: Path, monkeypatch, ollama):
    for name in ("a.md", "b.md", "de_outra_versao.md"):
        (test_version / name).write_text(f"{name} " + "conteudo " * 900, encoding="utf-8")
    ingestion.register_documents("teste-v1", test_version)
    lista = ingestion.INDEX_VERSIONS_DIR / "teste-v1.sha256"
    # A lista tem a.md e b.md (com o hash antigo) e um arquivo que não está mais na pasta.
    linhas = [l for l in lista.read_text(encoding="utf-8").splitlines() if "de_outra_versao" not in l]
    lista.write_text("\n".join(linhas + ["0" * 64 + "  sumiu.md"]) + "\n", encoding="utf-8")
    (test_version / "b.md").write_text("b.md mudou " + "conteudo " * 900, encoding="utf-8")
    added: list[dict] = []
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: added.append(kwargs))

    result = ingestion.ingest_documents()

    documents = {m["document"] for call in added for m in call["metadatas"]}
    assert documents == {"a.md", "b.md"}
    assert result["missing_documents"] == ["sumiu.md"]
    assert result["changed_documents"] == ["b.md"]
    assert result["ignored_in_folder"] == 1
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert (manifest["registered_documents"], manifest["missing_documents"]) == (3, ["sumiu.md"])
    assert [d["file"] for d in manifest["documents"]] == ["a.md", "b.md"]


def test_pasta_explicita_ignora_a_lista_da_versao(test_version: Path, monkeypatch, ollama):
    # --source (e o estudo de mutação, que ingere a cópia dele) pega a pasta inteira, como antes das listas.
    (test_version / "a.md").write_text("conteudo " * 900, encoding="utf-8")
    ingestion.register_documents("teste-v1", test_version)
    (test_version / "novo.md").write_text("conteudo " * 900, encoding="utf-8")
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: None)

    result = ingestion.ingest_documents(source_dir=test_version)

    assert result["documents_processed"] == 2
    assert result["registered_documents"] is None


def test_sintaxe_de_tabela_vira_texto_e_o_resto_fica_intacto():
    markdown = "| A Software |  | Engineering |\n| --- | --- | --- |\n|  | Models: |  |\nTexto com |pipe| no meio."

    assert ingestion.normalize_markdown(markdown) == "A Software Engineering\nModels:\nTexto com |pipe| no meio."


def test_markdown_vira_chunks_sem_pagina(tmp_path: Path):
    doc = tmp_path / "artigo.md"
    doc.write_text("# Título\n\n" + "palavra " * 2000, encoding="utf-8")

    records = ingestion.build_chunk_records(doc, 800, 120)

    assert len(records) > 1
    assert {r.page for r in records} == {None}
    assert records[0].chunk_id == "artigo::c0"


def test_prompt_de_pdf_nao_muda_e_trecho_de_markdown_sai_sem_pagina():
    pdf = generation.build_user_message("Q?", [{"document": "a.pdf", "page": 3, "text": "x"}])
    md = generation.build_user_message("Q?", [{"document": "b.md", "page": None, "text": "y"}])

    # Trecho de PDF: o mesmo cabeçalho do prompt v1 (o estudo de mutação depende disso).
    assert "[Trecho 1 - a.pdf (pág. 3)]\nx" in pdf
    assert "[Trecho 1 - b.md]\ny" in md
    assert "None" not in md


def test_ingestao_de_markdown_grava_manifesto_e_nao_poe_page_no_metadado(tmp_path: Path, monkeypatch, ollama):
    docs = tmp_path / "acervo"
    docs.mkdir()
    (docs / "a.md").write_text("| Título | |\n| --- | --- |\n" + "conteudo " * 900, encoding="utf-8")
    (docs / "ignorado.txt").write_text("fora do índice", encoding="utf-8")
    added: list[dict] = []
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: added.append(kwargs))

    result = ingestion.ingest_documents(source_dir=docs)

    assert result["documents_processed"] == 1
    metadatas = [m for call in added for m in call["metadatas"]]
    assert metadatas
    # O Chroma não aceita metadado None: Markdown fica sem a chave "page".
    assert all("page" not in m for m in metadatas)
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["documents"][0]["file"] == "a.md"
    assert len(manifest["documents"][0]["sha256"]) == 64
    assert manifest["md_normalization"] == ingestion.MD_NORMALIZATION
    assert manifest["n_chunks"] == len(metadatas)
