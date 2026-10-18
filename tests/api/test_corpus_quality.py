"""C0 · Acervo: métricas do texto que entra no índice (app/rag/corpus_quality.py) e o manifesto que as guarda."""
from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from app.config import get_settings
from app.rag import corpus_quality, ingestion, vector_store


def test_palavras_grudadas_contam_em_letras_e_o_dicionario_salva_palavra_longa():
    glued = "Theperformanceofthemodelwasevaluated"

    # "representativeness" tem 18 letras e está no dicionário: não é grude.
    assert corpus_quality.glued_word_rate("The representativeness of the sample was checked.") == 0.0
    assert corpus_quality.glued_word_rate(f"{glued} with care.") == pytest.approx(len(glued) / (len(glued) + 8))
    assert corpus_quality.glued_word_rate("123 — 456") is None


PDF_TAIL = """5 Conclusion
We showed that the method works on the three systems we studied.
References
[1] A. Smith and B. Jones. A study of things. In Proceedings of ICSE, 2020.
[2] C. Wu et al. Another study. arXiv preprint arXiv:2101.00001, 2021.
[3] D. Lee. Yet another study. Journal of Software Testing, 12(3), 2019.
A Experimental Setup
We ran everything on a laptop with a single GPU and measured each step.
The prompts are listed below and were kept fixed during the whole study."""


def test_bloco_de_referencias_do_pdf_para_no_titulo_de_apendice():
    lines = PDF_TAIL.splitlines()

    assert corpus_quality.references_block(lines) == (2, 6)


def test_titulo_de_referencias_sem_referencias_depois_nao_conta():
    lines = PDF_TAIL.splitlines() + ["References", "No other work is cited in this appendix, as noted above."]

    # O último "References" não tem cara de referência depois: vale o anterior, que acaba no apêndice.
    assert corpus_quality.references_block(lines) == (2, 6)


def test_markdown_fecha_o_bloco_no_proximo_titulo():
    lines = [
        "## 6 Conclusão", "Texto da conclusão.", "## References",
        "- [1] Smith, J. 2020. A study. In Proceedings of ICSE.",
        "- [2] Wu, C. et al. 2021. Another. arXiv:2101.00001.",
        "- [3] Lee, D. 2019. Yet another. Journal of Testing.",
        "## **A Appendix**", "Prompt usado no estudo.",
    ]

    assert corpus_quality.references_block(lines, markdown=True) == (2, 6)


def test_titulo_grudado_na_outra_coluna_vale_so_sem_titulo_limpo():
    # Markdown degradado (md-v1): o título divide a linha com o texto da outra coluna.
    lines = [
        "Corpo do artigo.",
        "REFERENCES TransactionsonReliability68,1(2018),45–66.",
        "[2] Smith,J.,Wu,C.2020.Proceedingsofthething.",
        "[3] Lee,D.2021.Journalofthings.",
        "[4] Kim,H.2019.arXivpreprint.",
    ]

    assert corpus_quality.references_block(lines, markdown=True) == (1, 5)
    # Em caixa normal seguido de minúscula, é frase do texto, não título.
    assert corpus_quality.references_block(["References to prior work are listed in Table 2 (2020, 2021)."]) is None


def test_estrutura_do_markdown_conta_tabela_sem_separadora_e_titulos():
    raw = "# Título\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n#hashtag não é título\n## Seção\n"

    assert ingestion.markdown_structure(raw) == {"table_rows": 3, "headings": 2}


def test_pdf_conta_paginas_e_paginas_sem_texto(tmp_path: Path):
    pdf = fitz.open()
    for text in ("Primeira página com texto.", None, "Terceira página com texto."):
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(tmp_path / "artigo.pdf")

    doc = ingestion.read_document(tmp_path / "artigo.pdf")
    quality = ingestion.document_quality(doc, ingestion.chunk_document(doc, 800, 120))

    assert (quality["pages_total"], quality["pages_with_text"]) == (3, 2)
    assert (quality["table_rows"], quality["headings"]) == (None, 0)
    assert [page for page, _ in doc.segments] == [1, 3]


def _ingest_markdown(tmp_path: Path, monkeypatch) -> Path:
    docs = tmp_path / "acervo"
    docs.mkdir()
    (docs / "a.md").write_text("# Título\n\n| A | B |\n| --- | --- |\n" + "conteudo " * 900, encoding="utf-8")
    (docs / "b.md").write_text("Theperformanceofthemodel " * 300, encoding="utf-8")
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: None)
    return Path(ingestion.ingest_documents(source_dir=docs)["manifest"])


def test_ingestao_grava_o_c0_por_documento_e_o_resumo(tmp_path: Path, monkeypatch, ollama):
    manifest = json.loads(_ingest_markdown(tmp_path, monkeypatch).read_text(encoding="utf-8"))

    a, b = manifest["documents"]
    assert (a["table_rows"], a["headings"], a["glued_word_rate"]) == (1, 1, 0.0)
    assert b["glued_word_rate"] > 0.9
    assert a["chunk_tokens"]["mean"] <= 800
    assert manifest["c0"]["version"] == corpus_quality.C0_VERSION
    assert manifest["c0"]["documents_glued_over_5pct"] == 1
    assert manifest["c0"]["tokens"] == a["tokens"] + b["tokens"]


def test_c0_de_versao_ja_ingerida_sem_reingerir(tmp_path: Path, monkeypatch, ollama):
    path = _ingest_markdown(tmp_path, monkeypatch)
    # Manifesto de antes do C0: só os campos da ingestão.
    old = json.loads(path.read_text(encoding="utf-8"))
    old.pop("c0")
    keep = {"file", "format", "bytes", "sha256", "chunks"}
    old["documents"] = [{k: v for k, v in d.items() if k in keep} for d in old["documents"]]
    path.write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "acervo" / "b.md").write_text("mudou depois da ingestão", encoding="utf-8")
    monkeypatch.setattr(vector_store, "add_chunks", lambda **kwargs: pytest.fail("não pode reingerir"))
    get_settings.cache_clear()

    c0 = ingestion.measure_corpus_quality()

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert c0["not_measured"] == ["b.md"]
    assert c0["documents"] == 1
    assert "glued_word_rate" in manifest["documents"][0] and "glued_word_rate" not in manifest["documents"][1]
    assert list(manifest)[-1] == "documents"
