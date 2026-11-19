"""Goldens v2: esquema, validador, evidência literal e sorteio do split (tests/benchmark/goldens.py)."""
from __future__ import annotations

import copy

from tests.benchmark import goldens

SOURCE_TEXT = (
    "Class and Method Selection. In total, 16 classes were selected, eight from Defects4J and eight from "
    "TestBench. The cyclomatic complexity of the classes was measured using SonarQube."
)
INSIDE = {
    "name": "v2-dentro", "input": "Quantas classes?", "input_en": "How many classes?",
    "expected_output": "16 classes.", "context": ["In total, 16 classes were selected, eight from Defects4J"],
    "source_document": "artigo", "evidence_page": 4, "question_class": "fato_direto", "content_type": "numero",
    "has_acronym": True, "names_article": False, "out_stratum": None, "split": "selecao", "set": "goldens",
    "status": "rascunho",
}
OUTSIDE = {
    **INSIDE, "name": "v2-fora", "context": [], "source_document": None, "question_class": "fora_do_corpus",
    "content_type": None, "out_stratum": "perto", "split": "confirmacao",
}
OUTSIDE.pop("evidence_page")


def errors_of(*items: dict, **kwargs) -> list[str]:
    return goldens.validate([copy.deepcopy(item) for item in items], **kwargs)


def test_golden_de_dentro_e_de_fora_validos():
    assert errors_of(INSIDE, OUTSIDE) == []


def test_nome_repetido_no_arquivo_ou_em_outro_conjunto():
    assert any("repetido" in e for e in errors_of(INSIDE, INSIDE))
    assert any("outro conjunto" in e for e in errors_of(INSIDE, other_names={"v2-dentro"}))


def test_nota_editorial_no_context_vai_para_observacao():
    noted = {**INSIDE, "context": ["texto (No Markdown, este trecho está com as palavras grudadas.)"]}

    assert any("nota editorial" in e for e in errors_of(noted))


def test_consistencia_entre_documento_classe_estrato_e_evidencia():
    cases = {
        "sem extensão": {**INSIDE, "source_document": "artigo.pdf"},
        "não tem source_document": {**OUTSIDE, "source_document": "artigo"},
        "out_stratum": {**OUTSIDE, "out_stratum": None},
        "precisa de evidência": {**INSIDE, "context": []},
        "content_type": {**INSIDE, "content_type": "grafico"},
        "true ou false": {**INSIDE, "has_acronym": "sim"},
        "faltam": {k: v for k, v in INSIDE.items() if k != "split"},
        "fora do esquema": {**INSIDE, "conversion": "limpo"},
    }
    for expected, item in cases.items():
        assert any(expected in e for e in errors_of(item)), expected
    ambiguous = {**OUTSIDE, "name": "v2-amb", "question_class": "ambiguo", "out_stratum": None}
    assert errors_of(ambiguous) == []


def test_perguntas_do_mesmo_documento_ficam_no_mesmo_split_e_na_versao():
    other = {**INSIDE, "name": "v2-dentro-2", "split": "confirmacao"}

    assert any("splits diferentes" in e for e in errors_of(INSIDE, other))
    assert any("não está na versão" in e for e in errors_of(INSIDE, index_documents={"outro"}))


def test_evidencia_tem_de_ser_texto_literal_da_fonte():
    paraphrase = {**INSIDE, "name": "v2-parafrase", "context": ["Sixteen classes were chosen from two benchmarks"]}
    almost = {**INSIDE, "name": "v2-quase",
              "context": ["In total, 16 classes were selected, eight from Defects4J and eight from TestSuite."]}
    wrong_page = {**INSIDE, "name": "v2-pagina", "evidence_page": 5}

    problems = goldens.check_evidence([INSIDE, paraphrase, almost, wrong_page], lambda stem: {4: SOURCE_TEXT})

    assert not any("v2-dentro:" in p for p in problems)
    assert any(p.startswith("ERRO v2-parafrase") for p in problems)
    assert any(p.startswith("AVISO v2-quase") for p in problems)
    assert any(p.startswith("AVISO v2-pagina") and "pág. 5" in p for p in problems)
    assert goldens.check_evidence([INSIDE], lambda stem: None)[0].startswith("ERRO")


def test_split_por_documento_estratificado_e_fixo_depois_de_sorteado():
    items = []
    for doc in ("a", "b", "c", "d"):
        for i in range(3):
            items.append({**INSIDE, "name": f"{doc}{i}", "source_document": doc, "split": None,
                          "content_type": "tabela" if doc in ("a", "b") else "texto"})
    items += [{**OUTSIDE, "name": f"fora{i}", "split": None} for i in range(4)]

    assert goldens.assign_splits(items, seed=7) == 16
    by_doc = {doc: {g["split"] for g in items if g["source_document"] == doc} for doc in "abcd"}
    assert all(len(splits) == 1 for splits in by_doc.values())
    # Um documento de cada estrato em cada split; as perguntas de fora, sorteadas uma a uma.
    assert {by_doc["a"].pop(), by_doc["b"].pop()} == {"selecao", "confirmacao"}
    assert {by_doc["c"].pop(), by_doc["d"].pop()} == {"selecao", "confirmacao"}
    assert sorted(g["split"] for g in items if g["source_document"] is None) == ["confirmacao"] * 2 + ["selecao"] * 2
    before = [g["split"] for g in items]
    items.append({**INSIDE, "name": "novo", "source_document": "e", "split": None})
    assert goldens.assign_splits(items, seed=99) == 1
    assert [g["split"] for g in items[:-1]] == before


def test_o_arquivo_v2_do_repositorio_e_valido():
    # O esquema e a consistência; a evidência contra a fonte precisa dos PDFs, que não vão para o Git.
    assert goldens.validate(goldens.load(), goldens._other_names()) == []
