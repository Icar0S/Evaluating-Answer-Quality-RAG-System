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
