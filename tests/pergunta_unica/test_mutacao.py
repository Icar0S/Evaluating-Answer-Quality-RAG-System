"""Os mutantes da pergunta única: a preparação (mutantes.py) e quem os mata (mutacao.py), herméticos."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import mutacao
import mutantes

DOC = mutantes.DOCUMENT


def _export(framework: str | None, scores: dict[str, dict[str, float]], threshold: float = 0.7) -> dict:
    data = {"judge_repetition": 1, "threshold": threshold, "cases": [
        {"name": a, "metrics": [{"name": m, "score": s, "applicable": True, "threshold": threshold}
                                for m, s in metrics.items()]} for a, metrics in scores.items()]}
    if framework:
        data["framework"] = framework
    return data


def test_um_juiz_mata_quando_reprova_uma_resposta_que_aprovava():
    original = [_export("ragas", {"q#1": {"faithfulness": 0.9}, "q#2": {"faithfulness": 0.8}}),
                _export(None, {"q#1": {"Faithfulness": 1.0}, "q#2": {"Faithfulness": 1.0}})]
    mutant = [_export("ragas", {"q#1": {"faithfulness": 0.3}, "q#2": {"faithfulness": 0.8}}),
              _export(None, {"q#1": {"Faithfulness": 0.9}, "q#2": {"Faithfulness": 1.0}})]
    answers = {"answers": [{"name": "q#1", "abstained": True, "source_documents": ["outro.md"], "actual_output": "x"},
                           {"name": "q#2", "abstained": False, "source_documents": [DOC], "actual_output": "yy"}]}

    report = mutacao.mutation_report(original, None, {"P1": {"meta": {"nome": "Remover ancoragem",
                                                                       "esperado": "a fidelidade cai"},
                                                              "exports": mutant, "answers": answers}})

    p1 = report["mutantes"]["P1"]
    assert p1["juizes"]["ragas"]["morto"] is True
    assert p1["juizes"]["ragas"]["matou_por"] == ["faithfulness"]
    assert p1["juizes"]["ragas"]["metricas"]["faithfulness"]["respostas_que_cairam"] == ["q#1"]
    assert p1["juizes"]["ragas"]["metricas"]["faithfulness"]["depois"] == pytest.approx(0.55)
    assert p1["juizes"]["deepeval"]["morto"] is False  # 0,9 ainda passa no limiar
    assert p1["comportamento"] == {"respostas": 2, "abstencoes": 1, "com_o_artigo_nos_trechos": 1,
                                   "caracteres_medios": 1.5}
    table = mutacao.markdown(report)
    assert "| P1: Remover ancoragem |" in table and "**morto** (fidelidade 0,85→0,55)" in table and "vivo" in table


def test_c2_troca_os_fatos_com_e_sem_espacos():
    glued = "includingposition,verbosity,andself-enhancement biases,aswellaslimitedreasoningability; over 80% agreement"
    spaced = "includingpositionbias, verbositybias, self-enhancementbias, andlimitedreasoningability, exceeding80%,"

    text1, counts1 = mutantes.substitute(glued)
    text2, _ = mutantes.substitute(spaced)

    assert "recency, formatting, and self-consistency" in text1 and "60% agreement" in text1
    assert "multilingual" in text1 and "80" not in text1
    assert "recency bias, formatting bias, self-consistency bias" in text2 and "exceeding60%" in text2
    assert counts1[r"80\s?%"] == 1


def test_indice_do_mutante_e_uma_copia_sem_o_artigo(tmp_path: Path):
    data = {"ids": ["a::c0", "mt::c0", "mt::c1"], "documents": ["texto a", "over 80%", "nada"],
            "metadatas": [{"document": "a.md"}, {"document": DOC}, {"document": DOC}],
            "embeddings": [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.2, 0.2]]}

    c1 = mutantes.copy_index(data, tmp_path / "c1", keep=lambda meta: meta["document"] != DOC)
    c2 = mutantes.copy_index(data, tmp_path / "c2", keep=lambda meta: True, replace={1: ("over 60%", [0.0, 0.0, 1.0])})

    import chromadb

    stored = chromadb.PersistentClient(path=str(tmp_path / "c2")).get_collection(mutantes.COLLECTION).get(
        ids=["mt::c0"], include=["documents"])
    assert (c1, c2) == (1, 3)
    assert stored["documents"] == ["over 60%"]
    overrides = mutantes.index_overrides(tmp_path / "c1", "nomic-embed-text")
    assert overrides["vector_store_collection"] == mutantes.COLLECTION
    assert Path(overrides["vector_store_dir"]).is_absolute()


def test_p1_guarda_o_prompt_antes_e_depois(tmp_path: Path):
    meta = mutantes.prepare("P1", tmp_path / "P1")

    diff = (tmp_path / "P1" / "artefatos" / "prompt.diff").read_text(encoding="utf-8")
    assert meta["set"] == {"prompt_require_grounding": "false"} and meta["usa_indice_base"] is True
    assert "-1. Responda SOMENTE com base no CONTEXTO" in diff
    assert json.loads((tmp_path / "P1" / "mutante.json").read_text(encoding="utf-8"))["operador"] == "P1"


def test_operador_desconhecido(tmp_path: Path):
    with pytest.raises(ValueError, match="desconhecido"):
        mutantes.prepare("X9", tmp_path)
