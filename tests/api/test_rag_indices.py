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


def test_toda_versao_registrada_tem_os_campos_que_o_backend_le():
    # /health lê "origins" sem valor padrão (Settings.index_origins): versão sem o campo derruba a rota.
    required = {"collection", "source_dir", "embedding_model", "origins", "description"}
    collections = [spec["collection"] for spec in RAG_INDICES.values()]
    assert len(collections) == len(set(collections)), "duas versões na mesma coleção se sobrescrevem"
    defaults = {key: Settings.model_fields[key].default for key in ("chunk_size_tokens", "chunk_overlap_tokens")}
    for name, spec in RAG_INDICES.items():
        assert required <= spec.keys(), f"{name}: falta {required - spec.keys()}"
        assert spec["origins"] and set(spec["origins"]) <= {"pdf", "md"}, name
        size = spec.get("chunk_size_tokens", defaults["chunk_size_tokens"])
        overlap = spec.get("chunk_overlap_tokens", defaults["chunk_overlap_tokens"])
        assert 0 <= overlap < size, f"{name}: sobreposição {overlap} >= chunk {size} seria trocada sem aviso"


def test_versao_pode_fixar_os_parametros_de_chunking(monkeypatch):
    monkeypatch.setitem(
        RAG_INDICES,
        "teste-k400",
        {**RAG_INDICES["pdf-v1"], "collection": "teste_k400", "chunk_size_tokens": 400, "chunk_overlap_tokens": 0},
    )
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "1200")

    settings = Settings(_env_file=None, rag_index="teste-k400")
    assert (settings.chunk_size_tokens, settings.chunk_overlap_tokens) == (400, 0)
    # Versão sem as chaves segue a configuração, como antes.
    assert Settings(_env_file=None, rag_index="pdf-v1").chunk_size_tokens == 1200


def _chunk_text_antes_das_janelas(text: str, size: int, overlap: int, offset_tokens: int = 0) -> list[str]:
    """O chunk_text de antes do _windows (commit 4d0c395), para provar que o corte não mudou."""
    encoding = ingestion._encoding
    tokens = encoding.encode(text)
    if not tokens:
        return []
    overlap = ingestion.effective_overlap(size, overlap)
    offset = offset_tokens % size if size > 0 else 0
    chunks, start = [], 0
    step = max(size - offset if offset else size, 1)
    while start < len(tokens):
        end = start + step
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap
        step = size
    return chunks


@pytest.mark.parametrize("size, overlap, offset", [(800, 120, 0), (100, 0, 0), (100, 150, 0), (120, 30, 45), (50, 10, 7)])
def test_chunk_text_corta_igual_ao_de_antes(size, overlap, offset):
    text = " ".join(f"palavra{i} número {i * 7} — fim." for i in range(400))

    assert ingestion.chunk_text(text, size, overlap, offset) == _chunk_text_antes_das_janelas(text, size, overlap, offset)


def test_janela_atravessa_a_pagina_e_o_chunk_leva_a_pagina_onde_comeca(tmp_path: Path):
    pages = [(1, "um " * 150), (2, "dois " * 30), (4, "quatro " * 150)]
    doc = ingestion.DocumentText(tmp_path / "artigo.pdf", "pdf", pages, pages_total=4)

    by_page = ingestion.chunk_document(doc, 100, 0)
    across = ingestion.chunk_document(doc, 100, 0, across_pages=True)

    count = ingestion.corpus_quality.count_tokens
    # Dentro da página, a página 2 (32 tokens) vira um chunk pequeno. Atravessando, ela fica no meio de um
    # chunk que começa na página 1, e só o último chunk é curto (o recorte de tokens pode juntar 1 ou 2 na
    # emenda das páginas).
    assert any(r.page == 2 and count(r.text) < 40 for r in by_page)
    assert [r.page for r in across] == [1, 1, 4, 4, 4]
    assert all(count(r.text) >= 95 for r in across[:-1])
    assert all(word in across[1].text for word in ("um", "dois", "quatro"))
    assert len({r.chunk_id for r in across}) == len(across)


def test_versao_liga_o_corte_atravessando_a_pagina():
    assert Settings(_env_file=None, rag_index="pdf-v1-cont").chunk_across_pages is True
    assert Settings(_env_file=None, rag_index="pdf-cont-400").chunk_size_tokens == 400
    assert Settings(_env_file=None, rag_index="pdf-v1").chunk_across_pages is False


def test_register_like_copia_a_lista_da_base_e_nao_a_pasta(test_version: Path):
    for name in ("b.md", "a.md"):
        (test_version / name).write_text(name, encoding="utf-8")
    ingestion.register_documents("teste-v1", test_version)
    (test_version / "de_outra_versao.md").write_text("outro acervo", encoding="utf-8")

    path = ingestion.register_documents("teste-v1-x", test_version, like="teste-v1")

    base = ingestion.registered_documents("teste-v1")
    assert list(ingestion.registered_documents("teste-v1-x").items()) == list(base.items())
    assert "--like teste-v1" in path.read_text(encoding="utf-8").splitlines()[0]


def test_register_like_recusa_pasta_sem_os_documentos_da_base(test_version: Path):
    for name in ("a.md", "b.md"):
        (test_version / name).write_text(name, encoding="utf-8")
    ingestion.register_documents("teste-v1", test_version)
    (test_version / "a.md").unlink()
    (test_version / "b.md").write_text("b mudou", encoding="utf-8")

    with pytest.raises(ValueError, match=r"a\.md: não está.*b\.md: mudou"):
        ingestion.register_documents("teste-v1-x", test_version, like="teste-v1")
    assert ingestion.registered_documents("teste-v1-x") is None
    with pytest.raises(FileNotFoundError):
        ingestion.register_documents("teste-v1-y", test_version, like="sem-lista")


def test_manifesto_grava_a_sobreposicao_pedida_e_a_efetiva(tmp_path: Path, monkeypatch, ollama):
    docs = tmp_path / "acervo"
    docs.mkdir()
    (docs / "a.md").write_text("conteudo " * 900, encoding="utf-8")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "150")
    get_settings.cache_clear()
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: None)

    manifest = json.loads(Path(ingestion.ingest_documents(source_dir=docs)["manifest"]).read_text(encoding="utf-8"))

    assert (manifest["chunk_overlap_tokens"], manifest["chunk_overlap_tokens_effective"]) == (150, 25)


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


def test_chunker_por_secao_corta_nos_titulos_e_grava_a_secao(tmp_path: Path):
    text = "Resumo sem título.\n# 1 Introdução\n" + "intro " * 300 + "\n## 2.1 Método\nMétodo curto."
    doc = ingestion.DocumentText(tmp_path / "artigo.md", "md", [(None, text)])

    records = ingestion.chunk_document(doc, 200, 0, chunker="section")

    assert [r.section for r in records][0] is None
    assert {r.section for r in records[1:]} == {"1 Introdução", "2.1 Método"}
    assert records[-1].chunk_id == "artigo::s2::c0" and records[-1].text.startswith("## 2.1 Método")
    # Nenhum chunk mistura duas seções: o corte vem antes da janela.
    assert not any("intro" in r.text and "Método curto" in r.text for r in records)
    assert records[-1].metadata == {"document": "artigo.md", "section": "2.1 Método"}
    assert ingestion.chunk_document(doc, 200, 0)[0].chunk_id == "artigo::c0"


def test_register_by_stem_pega_os_mesmos_artigos_em_outro_formato(test_version: Path):
    for name in ("a.pdf", "b.pdf"):
        (test_version / name).write_text(name, encoding="utf-8")
    ingestion.register_documents("teste-v1", test_version)
    for name in ("a.pdf", "b.pdf"):
        (test_version / name).unlink()
    for name in ("b.md", "a.md", "c.md"):
        (test_version / name).write_text(f"# {name}", encoding="utf-8")

    path = ingestion.register_documents("teste-v1-x", test_version, like="teste-v1", by_stem=True)

    assert list(ingestion.registered_documents("teste-v1-x")) == ["a.md", "b.md"]
    assert "--by-stem" in path.read_text(encoding="utf-8").splitlines()[0]
    (test_version / "b.pdf").write_text("de novo", encoding="utf-8")
    with pytest.raises(ValueError, match="b: 2 arquivo"):
        ingestion.register_documents("teste-v1-y", test_version, like="teste-v1", by_stem=True)
