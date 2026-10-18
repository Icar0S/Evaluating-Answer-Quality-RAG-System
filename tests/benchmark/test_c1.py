"""C1 · Recuperação: as métricas sem juiz (tests/benchmark/c1.py)."""
from __future__ import annotations

from tests.benchmark import c1

SOURCE = (
    "We introduce RAGAS (Retrieval Augmented Generation Assessment), a framework for reference-free "
    "evaluation of Retrieval Augmented Generation pipelines."
)


def chunk(document: str, text: str, score: float, page: int | None = 1) -> dict:
    return {"chunk_id": f"{document}::c", "document": document, "page": page, "text": text, "similarity_score": score}


def test_normalizacao_ignora_espaco_hifenizacao_ligadura_e_caixa():
    assert c1.normalize("The ﬁnal evalu-\nation, in RAGAS") == c1.normalize("thefinalevaluationinragas")
    assert c1.normalize("soft­hyphen") == "softhyphen"


def test_fragmentos_quebram_em_reticencias_e_colchetes():
    assert c1.split_fragments(["Primeira parte [...] segunda parte ... terceira", "Outro item", " [...] "]) == [
        "Primeira parte", "segunda parte", "terceira", "Outro item",
    ]


def test_fragmento_casa_com_palavras_grudadas_e_partido_entre_dois_trechos():
    half = len(SOURCE) // 2

    # Markdown degradado: o mesmo texto sem nenhum espaço.
    assert c1.evidence([SOURCE], [SOURCE.replace(" ", "")])["evidence"] is True
    split = c1.evidence([SOURCE], [SOURCE[:half], SOURCE[half:]])
    assert split["evidence"] is True
    assert split["single_chunk"] == [False]
    # Metade do fragmento só, num trecho: abaixo de TAU.
    alone = c1.evidence([SOURCE], [SOURCE[:half]])
    assert alone["coverage"] == 0.0 and alone["union_fractions"][0] < c1.TAU


def test_fragmento_curto_casa_inteiro_dentro_de_um_trecho():
    assert c1.evidence(["56,58%"], ["a taxa foi de 56,58% no total"])["evidence"] is True
    assert c1.evidence(["56,58%"], ["a taxa foi de 56,5 no total"])["evidence"] is False
    assert c1.evidence([], ["qualquer"])["coverage"] is None


def test_metricas_da_pergunta_rank_hit_mrr_e_precisao():
    chunks = [
        chunk("outro.pdf", "nada a ver com a pergunta", 0.70),
        chunk("survey.pdf", f"Como diz o artigo: {SOURCE}", 0.65),
        chunk("ragas.md", "outro trecho do próprio RAGAS", 0.60, page=None),
        chunk("outro.pdf", "nada", 0.50),
    ]

    # O golden cita ragas.pdf e o índice tem ragas.md: é o mesmo artigo.
    m = c1.question_metrics(arm_chunks=chunks, chunks_at_10=chunks, source_document="ragas.pdf",
                            fragments=[SOURCE], top1_similarity=0.70)

    assert (m["rank"], m["hit_at_1"], m["hit_at_k"], m["hit_at_10"]) == (3, False, True, True)
    # Documentos distintos nos 10 primeiros: outro, survey, ragas.
    assert m["mrr_at_10"] == 1 / 3
    # Relevantes: o trecho do survey (tem a evidência) e o do ragas (é a fonte).
    assert [c["relevant"] for c in m["chunks"]] == [False, True, True, False]
    assert m["precision_at_k"] == 0.5
    assert (m["evidence_at_k"], m["coverage_at_k"], m["fragments_single_chunk"]) == (True, 1.0, 1)


def test_pergunta_de_fora_so_tem_limiar_e_lista():
    blocked = c1.question_metrics(arm_chunks=[], chunks_at_10=[], source_document=None, fragments=[],
                                  top1_similarity=0.41, threshold=0.428)
    open_gate = c1.question_metrics(arm_chunks=[chunk("a.pdf", "x", 0.5)], chunks_at_10=[], source_document=None,
                                    fragments=[], top1_similarity=0.5, threshold=0.0)

    assert blocked["blocked_by_threshold"] is True and "rank" not in blocked
    assert open_gate["blocked_by_threshold"] is False
    assert open_gate["chunks"][0]["document"] == "a.pdf" and "text" not in open_gate["chunks"][0]


def test_nome_do_documento_so_perde_a_extensao_de_acervo():
    assert c1.doc_stem("RAG v1.5 notes.pdf") == "RAG v1.5 notes"
    assert c1.doc_stem("RAG v1.5 notes") == "RAG v1.5 notes"
    assert c1.doc_stem(None) is None


def test_bootstrap_reamostra_documentos_e_nao_perguntas():
    # Um documento com 3 perguntas certas e outro com 1 errada: uma reamostra de 2 documentos só pode dar
    # média 1 (a, a), 0 (b, b) ou 0,75 (a, b). Reamostrando perguntas, o IC seria bem mais estreito.
    s = c1.cluster_bootstrap([1, 1, 1, 0], ["a", "a", "a", "b"])

    assert (s["mean"], s["n"], s["clusters"]) == (0.75, 4, 2)
    assert (s["ci_low"], s["ci_high"]) == (0.0, 1.0)
    assert c1.cluster_bootstrap([], []) is None


def test_auc_da_similaridade_e_resumo_com_limiar():
    assert c1.auc([0.6, 0.7], [0.4, 0.5]) == 1.0
    assert c1.auc([0.5], [0.5]) == 0.5
    assert c1.auc([], [0.1]) is None

    inside = [
        ("a.pdf", c1.question_metrics(arm_chunks=[chunk("a.pdf", SOURCE, 0.6)], chunks_at_10=[chunk("a.pdf", SOURCE, 0.6)],
                                      source_document="a.pdf", fragments=[SOURCE], top1_similarity=0.6, threshold=0.5)),
        ("b.pdf", c1.question_metrics(arm_chunks=[], chunks_at_10=[], source_document="b.pdf", fragments=[SOURCE],
                                      top1_similarity=0.45, threshold=0.5)),
    ]
    outside = [c1.question_metrics(arm_chunks=[], chunks_at_10=[], source_document=None, fragments=[],
                                   top1_similarity=0.4, threshold=0.5)]

    s = c1.summarize(inside, outside, threshold=0.5)

    assert (s["n_inside"], s["n_outside"]) == (2, 1)
    assert s["hit_at_k"]["mean"] == 0.5 and s["evidence_at_k"]["mean"] == 0.5
    # A pergunta sem nenhum trecho não tem precisão: entra só a que recebeu trechos.
    assert s["precision_at_k"]["n"] == 1
    assert (s["false_block_rate"], s["threshold_abstention_rate"]) == (0.5, 1.0)
    assert s["top1_similarity_auc"]["auc"] == 1.0
