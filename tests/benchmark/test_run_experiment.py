"""Runner do benchmark (tests/benchmark/run_experiment.py): o cartão e o placar, com saídas falsas."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.benchmark import run_experiment as rx

CARDS = sorted((rx.ROOT / "tests" / "benchmark" / "cartoes").glob("*.yaml"))
CARD = {
    "id": "EX", "titulo": "teste", "pergunta": "?", "hipotese": "h", "goldens": "x.json", "didatica": "d",
    "metrica_primaria": {"camada": "c2", "metrica": "faithfulness"},
    "bracos": {
        "A1": {"indice": "pdf-v1", "camadas": ["c1", "c2"], "papel": "baseline"},
        "A2": {"indice": "pdf-v1", "camadas": ["c2"], "ordem": 2},
        "A4": {"rejulgar": "A1", "ordem_juiz": 4},
        "A5": {"indice": "pdf-v1-aa", "camadas": ["c1"]},
    },
    "pares": [["A1", "A2"], ["A1", "A4"], ["A1", "A5"]],
    "margem": {"gerador_juiz": [["A1", "A2"]], "juiz": [["A1", "A4"]], "c1": [["A1", "A5"]]},
}


def test_os_cartoes_do_repositorio_sao_validos():
    assert CARDS
    for path in CARDS:
        rx.load_card(path)


def test_bracos_com_os_padroes_preenchidos():
    arms = rx.arms_of(CARD)

    assert (arms["A1"]["lang"], arms["A1"]["order"], arms["A1"]["set"]) == ("pt", None, {})
    assert arms["A2"]["layers"] == ["c2"] and arms["A2"]["papel"] == "candidato"
    assert (arms["A4"]["kind"], arms["A4"]["of"]) == ("rejulgar", "A1")


def test_validador_do_cartao_aponta_os_erros():
    broken = copy.deepcopy(CARD)
    broken["bracos"]["A4"] = {"rejulgar": "A9"}
    broken["bracos"]["A6"] = {"indice": "versao-inexistente", "camadas": ["c3"], "papel": "baseline"}
    broken["pares"] = [["A1", "Z"]]
    broken["teto"] = "Z"
    broken["guardas"] = [{"nome": "g", "par": ["A1"], "metrica": "evidence_at_k"}]

    errors = rx.validate_card(broken)

    assert any("rejulgar" in e for e in errors)
    assert any("versao-inexistente" in e for e in errors)
    assert any("camadas" in e for e in errors)
    assert any("baseline" in e for e in errors)
    assert any("par [" in e for e in errors)
    assert any("teto" in e for e in errors)
    assert sum(e.startswith("guarda g") for e in errors) == 3
    assert rx.validate_card({"id": "X"})[0].startswith("falta")


def _c1(index: str, hits: dict[str, bool]) -> dict:
    rows = [{"name": n, "source_document": f"doc{n[-1]}", "c1": {index: {"pt": {
        "in_index": True, "hit_at_k": hit, "evidence_at_k": hit, "precision_at_k": 0.5}}}} for n, hit in hits.items()]
    return {"c1": {"summary": {index: {"pt": {"n_inside": len(rows), "evidence_at_k": {"mean": 0.5}}}}},
            "rows": rows, "out_rows": []}


def _answers(abstained: dict[str, bool]) -> dict:
    answers = [{"name": n, "source_document": None if n.startswith("f") else f"doc{n[-1]}", "abstained": a,
                "retrieval_ms": 10.0, "generation_ms": 100.0, "prompt_tokens": 900, "completion_tokens": 40,
                "truncated": False} for n, a in abstained.items()]
    return {"n_cases": len(answers), "answers": answers}


def _ragas(scores: dict[str, float]) -> dict:
    cases = [{"name": n, "metrics": [{"name": "faithfulness", "score": s}]} for n, s in scores.items()]
    return {"cases": cases, "cases_done": len(cases), "n_cases": len(cases), "judge_model": "qwen3:8b"}


@pytest.fixture
def out(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(rx, "index_documents", lambda index: {"docA", "docB"})
    files = {
        "c1_A1.json": _c1("pdf-v1", {"qA": True, "qB": False}),
        "c1_A5.json": _c1("pdf-v1-aa", {"qA": True, "qB": False}),
        "respostas_A1.json": _answers({"qA": False, "qB": True, "f1": True}),
        "respostas_A2.json": _answers({"qA": False, "qB": False, "f1": True}),
        "ragas_A1.json": _ragas({"qA": 0.8, "qB": 0.6}),
        "ragas_A2.json": _ragas({"qA": 0.7, "qB": 0.6}),
        "ragas_A4.json": _ragas({"qA": 0.8, "qB": 0.5}),
    }
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_placar_mede_os_pares_e_a_margem(out: Path):
    result = rx.scoreboard(CARD, out, limit=None)

    assert result["pairs"]["A1→A2"]["c2.faithfulness"]["delta"] == pytest.approx(-0.05)
    # Rejulgar compara só o juiz: sem recusa nem custo (as respostas são as mesmas).
    assert set(result["pairs"]["A1→A4"]) == {"c2.faithfulness"}
    assert result["pairs"]["A1→A5"]["c1.hit_at_k"]["delta"] == 0.0
    assert result["means"]["A1"]["abstention.correct_abstention"] == 1.0
    assert result["means"]["A1"]["abstention.undue_abstention"] == 0.5
    assert result["margins"]["c1"]["c1.hit_at_k"] == 0.0
    assert result["margins"]["gerador_juiz"]["c2.faithfulness"] > 0
    assert json.loads((out / "margem.json").read_text(encoding="utf-8"))["experiment"] == "EX"


def test_veredito_usa_a_margem_do_e0(out: Path, monkeypatch):
    card = copy.deepcopy(CARD)
    del card["margem"]
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (
        {"margins": {"gerador_juiz": {"c2.faithfulness": 0.2}}}, "tests/benchmark/out/E0_x/margem.json"))

    result = rx.scoreboard(card, out, limit=None)

    assert result["verdicts"]["A1→A2"]["verdict"] == "sem diferença relevante"
    assert result["verdicts"]["A1→A2"]["margin"] == 0.2
    summary = rx.write_summary(card, out, result, "id: EX\n").read_text(encoding="utf-8")
    assert "sem diferença relevante" in summary and "```yaml" in summary


def _c1_lang(lang: str, evidence: dict[str, float]) -> dict:
    rows = [{"name": n, "source_document": f"doc{n[-1]}", "c1": {"pdf-v1": {lang: {
        "in_index": True, "evidence_at_k": e, "retrieval_ms": 10.0}}}} for n, e in evidence.items()]
    return {"c1": {"summary": {"pdf-v1": {lang: {"n_inside": len(rows)}}}}, "rows": rows, "out_rows": []}


def test_guardas_buraco_fechado_e_tempo_da_traducao(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"docA", "docB"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    cache = tmp_path / "traducoes.json"
    cache.write_text(json.dumps({"translations": {"pergunta A": "question A"}, "milliseconds": {"pergunta A": 500.0}}),
                     encoding="utf-8")
    card = {
        "id": "EY", "titulo": "t", "pergunta": "?", "hipotese": "h", "goldens": "x.json", "didatica": "d",
        "metrica_primaria": {"camada": "c1", "metrica": "evidence_at_k"},
        "bracos": {
            "pt": {"indice": "pdf-v1", "camadas": ["c1"], "papel": "baseline"},
            "trad": {"indice": "pdf-v1", "camadas": ["c1"], "set": {
                "retrieval_query_translation": True, "retrieval_translation_cache": str(cache)}},
            "en": {"indice": "pdf-v1", "lingua": "en", "camadas": ["c1"], "papel": "referencia"},
        },
        "pares": [["pt", "trad"], ["pt", "en"]],
        "teto": "en",
        "guardas": [
            {"nome": "evidência", "par": ["pt", "trad"], "metrica": "c1.evidence_at_k", "tolerancia_perguntas": 1},
            {"nome": "latência", "par": ["pt", "trad"], "metrica": "c1.retrieval_ms", "tolerancia": 100},
        ],
    }
    for arm_id, lang, evidence in (("pt", "pt", {"qA": 0.0, "qB": 0.0}), ("trad", "pt", {"qA": 1.0, "qB": 0.0}),
                                   ("en", "en", {"qA": 1.0, "qB": 1.0})):
        (tmp_path / f"c1_{arm_id}.json").write_text(json.dumps(_c1_lang(lang, evidence)), encoding="utf-8")

    result = rx.scoreboard(card, tmp_path, limit=None, questions={"qA": "pergunta A", "qB": "pergunta B"})

    # A busca com a tradução gravada não pagou a tradução: o placar soma os 500 ms gravados à pergunta A.
    assert result["means"]["trad"]["c1.retrieval_ms"] == 260.0
    assert result["gap_closed"] == {"trad": 0.5}
    evidence, latency = result["guards"]
    assert (evidence["passed"], evidence["tolerance"]) == (True, 0.5)
    assert (latency["passed"], latency["delta"]["delta"]) == (False, 250.0)
    summary = rx.write_summary(card, tmp_path, result, "id: EY\n").read_text(encoding="utf-8")
    assert "NÃO PASSA" in summary and "50%" in summary
