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

    goldens = [{"name": "qA", "input": "pergunta A"}, {"name": "qB", "input": "pergunta B"}]
    result = rx.scoreboard(card, tmp_path, limit=None, goldens=goldens)

    # A busca com a tradução gravada não pagou a tradução: o placar soma os 500 ms gravados à pergunta A.
    assert result["means"]["trad"]["c1.retrieval_ms"] == 260.0
    assert result["gap_closed"] == {"trad": 0.5}
    evidence, latency = result["guards"]
    assert (evidence["passed"], evidence["tolerance"]) == (True, 0.5)
    assert (latency["passed"], latency["delta"]["delta"]) == (False, 250.0)
    summary = rx.write_summary(card, tmp_path, result, "id: EY\n").read_text(encoding="utf-8")
    assert "NÃO PASSA" in summary and "50%" in summary


SWEEP = {
    "id": "EZ", "titulo": "t", "pergunta": "?", "hipotese": "h", "goldens": "x.json", "didatica": "d",
    "exploratorio": True, "split": "selecao",
    "metrica_primaria": {"camada": "c1", "metrica": "evidence_at_k"},
    "bracos": {
        "base": {"indice": "pdf-v1", "camadas": ["c1"], "papel": "baseline"},
        "k8": {"indice": "pdf-v1", "camadas": ["c1"], "set": {"top_k": 8}},
        "lim": {"indice": "pdf-v1", "camadas": ["c1"], "set": {"min_similarity_score": "recalibrado"},
                "metrica_primaria": {"camada": "c1", "metrica": "correct_abstention"}},
    },
    "pares": [["base", "k8"], ["base", "lim"]],
    "curva": ["base", "k8"],
}


def _c1_sweep(rows: dict[str, tuple], out_rows: dict[str, tuple]) -> dict:
    """rows: nome → (evidência, similaridade do 1º trecho, barrada, curva); out_rows: nome → (similaridade, barrada)."""
    inside = [{"name": n, "source_document": f"doc{n[1]}", "c1": {"pdf-v1": {"pt": {
        "in_index": True, "evidence_at_k": e, "top1_similarity": s, "blocked_by_threshold": b,
        "evidence_curve": curve, "tokens_curve": [100 * (i + 1) for i in range(len(curve))]}}}}
        for n, (e, s, b, curve) in rows.items()]
    outside = [{"name": n, "c1": {"pdf-v1": {"pt": {"top1_similarity": s, "blocked_by_threshold": b}}}}
               for n, (s, b) in out_rows.items()]
    return {"c1": {"summary": {"pdf-v1": {"pt": {"n_inside": len(inside), "top1_similarity_auc": {
        "auc": 0.9, "ci_low": 0.8, "ci_high": 1.0}}}}}, "rows": inside, "out_rows": outside}


def test_split_recorta_os_goldens(tmp_path: Path):
    goldens = tmp_path / "g.json"
    goldens.write_text(json.dumps([{"name": "a", "split": "selecao", "source_document": "d.pdf"},
                                   {"name": "b", "split": "confirmacao", "source_document": "d.pdf"},
                                   {"name": "c", "split": "selecao", "source_document": None}]), encoding="utf-8")
    experiment = rx.Experiment({**SWEEP, "goldens": str(goldens)}, Path("EZ.yaml"), tmp_path, limit=None)

    experiment.prepare_goldens()

    assert [g["name"] for g in json.loads(experiment.goldens_path.read_text(encoding="utf-8"))] == ["a", "c"]


def test_limiar_recalibrado_sai_da_c1_do_baseline(tmp_path: Path):
    (tmp_path / "c1_base.json").write_text(json.dumps(_c1_sweep(
        {"q1": (1, 0.70, False, [1]), "q2": (0, 0.66, False, [0]), "q3": (1, 0.75, False, [1])},
        {"f1": (0.50, False), "f2": (0.58, False)})), encoding="utf-8")
    experiment = rx.Experiment(SWEEP, Path("EZ.yaml"), tmp_path, limit=None)

    resolved = experiment.resolved_set(experiment.arms["lim"])

    # Separação perfeita (0,66 > 0,58): o ponto médio da folga.
    assert resolved == {"min_similarity_score": 0.62}
    saved = json.loads((tmp_path / "limiar_lim.json").read_text(encoding="utf-8"))
    assert (saved["calibrated_on"], saved["n_inside"], saved["n_outside"]) == ("base", 3, 2)
    assert "leave_one_document_out" in saved


def test_varredura_abstencao_pelo_limiar_mcnemar_holm_classe_e_curva(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"doc1", "doc2", "doc3"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    files = {
        "base": _c1_sweep({"q1": (0, 0.7, False, [0, 0]), "q2": (0, 0.6, False, [0, 1]), "q3": (1, 0.8, False, [1, 1])},
                          {"f1": (0.5, False), "f2": (0.55, False)}),
        "k8": _c1_sweep({"q1": (1, 0.7, False, [0, 1]), "q2": (1, 0.6, False, [1, 1]), "q3": (1, 0.8, False, [1, 1])},
                        {"f1": (0.5, False), "f2": (0.55, False)}),
        "lim": _c1_sweep({"q1": (0, 0.7, False, [0]), "q2": (0, 0.6, True, [0]), "q3": (1, 0.8, False, [1])},
                         {"f1": (0.5, True), "f2": (0.55, True)}),
    }
    for arm_id, data in files.items():
        (tmp_path / f"c1_{arm_id}.json").write_text(json.dumps(data), encoding="utf-8")
    goldens = [{"name": "q1", "question_class": "fato_direto"}, {"name": "q2", "question_class": "fato_distribuido"},
               {"name": "q3", "question_class": "fato_direto"}]

    result = rx.scoreboard(SWEEP, tmp_path, limit=None, goldens=goldens)

    assert result["means"]["lim"]["c1.correct_abstention"] == 1.0
    assert result["means"]["lim"]["c1.false_block"] == pytest.approx(1 / 3, abs=1e-4)
    # Cada par usa a métrica primária do braço de chegada; o Holm junta os dois.
    assert result["verdicts"]["base→lim"]["metric"] == "c1.correct_abstention"
    assert result["tests"]["base→k8"]["metric"] == "c1.evidence_at_k"
    assert (result["tests"]["base→k8"]["b"], result["tests"]["base→k8"]["c"]) == (0, 2)
    assert result["tests"]["base→k8"]["p_holm"] >= result["tests"]["base→k8"]["p_value"]
    assert result["by_class"]["base→k8"]["fato_distribuido"]["delta"] == 1.0
    assert result["curves"]["base"] == [{"k": 1, "n": 3, "tokens": 100.0, "evidence": pytest.approx(1 / 3, abs=1e-4)},
                                        {"k": 2, "n": 3, "tokens": 200.0, "evidence": pytest.approx(2 / 3, abs=1e-4)}]
    summary = rx.write_summary(SWEEP, tmp_path, result, "id: EZ\n").read_text(encoding="utf-8")
    assert "exploratório" in summary and "Curva evidência" in summary and "Holm" in summary


def test_validador_do_split_da_curva_e_do_limiar_recalibrado():
    broken = copy.deepcopy(SWEEP)
    broken["split"] = "treino"
    broken["curva"] = ["zzz"]
    del broken["bracos"]["base"]
    broken["pares"] = [["k8", "lim"]]

    errors = rx.validate_card(broken)

    assert any(e.startswith("split") for e in errors)
    assert any(e.startswith("curva") for e in errors)
    assert any("recalibrado" in e for e in errors)
    assert rx.validate_card(SWEEP) == []


def test_delta_por_grupo_usa_o_campo_do_cartao(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"doc1", "doc2"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    card = {**SWEEP, "classe_por": "content_type", "pares": [["base", "k8"]]}
    for arm_id, hits in (("base", (0, 1)), ("k8", (1, 1))):
        rows = {"q1": (hits[0], 0.7, False, [hits[0]]), "q2": (hits[1], 0.7, False, [hits[1]])}
        (tmp_path / f"c1_{arm_id}.json").write_text(json.dumps(_c1_sweep(rows, {})), encoding="utf-8")
    goldens = [{"name": "q1", "content_type": "tabela"}, {"name": "q2", "content_type": "texto"}]

    result = rx.scoreboard(card, tmp_path, limit=None, goldens=goldens)

    assert result["class_field"] == "content_type"
    assert {cls: d["delta"] for cls, d in result["by_class"]["base→k8"].items()} == {"tabela": 1.0, "texto": 0.0}


def test_guarda_de_um_subgrupo_olha_so_as_perguntas_dele(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"doc1", "doc2"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    card = {**SWEEP, "pares": [["base", "k8"]], "guardas": [
        {"nome": "sem sigla", "par": ["base", "k8"], "metrica": "c1.evidence_at_k", "tolerancia_perguntas": 1,
         "grupo": {"campo": "has_acronym", "valor": False}}]}
    # Com sigla, k8 ganha uma; sem sigla, perde as duas: a guarda do subgrupo vê só a perda, e 2 perguntas
    # passam da tolerância de 1.
    for arm_id, hits in (("base", (0, 1, 1)), ("k8", (1, 0, 0))):
        rows = {f"q{i}": (hit, 0.7, False, [hit]) for i, hit in enumerate(hits, start=1)}
        (tmp_path / f"c1_{arm_id}.json").write_text(json.dumps(_c1_sweep(rows, {})), encoding="utf-8")
    goldens = [{"name": "q1", "has_acronym": True}, {"name": "q2", "has_acronym": False},
               {"name": "q3", "has_acronym": False}]

    [guard] = rx.scoreboard(card, tmp_path, limit=None, goldens=goldens)["guards"]

    assert (guard["delta"]["n"], guard["delta"]["delta"], guard["tolerance"], guard["passed"]) == (2, -1.0, 0.5, False)
    assert rx.validate_card({**card, "guardas": [{**card["guardas"][0], "grupo": {"campo": "x"}}]})[0].endswith(
        "grupo precisa de campo e valor")


def test_atribuicao_so_com_a_citacao_obrigatoria(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"docA", "docB"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    card = {**{k: v for k, v in CARD.items() if k != "margem"}, "pares": [],
            "bracos": {"A1": {"indice": "pdf-v1", "camadas": ["c2"], "papel": "baseline"}}}
    answers = [
        {"name": "qA", "source_document": "docA.pdf", "sources": ["docA.pdf (pág. 1)"],
         "actual_output": "É isso [docA.pdf, pág. 1]."},
        {"name": "qB", "source_document": "docB.pdf", "sources": ["docB.pdf (pág. 2)"],
         "actual_output": "É aquilo [docZ.pdf, pág. 5]."},
        {"name": "f1", "source_document": None, "sources": ["docA.pdf (pág. 1)"],
         "actual_output": "Inventei [docA.pdf, pág. 1]."},
    ]
    for require_citation in (False, True):
        data = {"n_cases": 3, "answers": answers, "arm": {"prompt": {"require_citation": require_citation}}}
        (tmp_path / "respostas_A1.json").write_text(json.dumps(data), encoding="utf-8")

        result = rx.scoreboard(card, tmp_path, limit=None)

        means = result["means"]["A1"]
        if not require_citation:
            assert "attribution.misattribution" not in means
            continue
        # qB fabrica a citação; f1, de fora, cita sem se abster: 2 de 3 respostas.
        assert means["attribution.misattribution"] == pytest.approx(2 / 3, abs=1e-4)
        assert means["attribution.cites_source"] == 0.5
        assert result["summary"]["A1"]["attribution"]["fabricada"] == 1


def test_margem_do_proprio_cartao_vale_para_as_metricas_que_ele_mediu(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"docA", "docB"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (
        {"margins": {"gerador_juiz": {"c2.faithfulness": 0.2}}}, "tests/benchmark/out/E0_x/margem.json"))
    card = {**{k: v for k, v in CARD.items() if k != "margem"},
            "metrica_primaria": {"camada": "attribution", "metrica": "misattribution"},
            "bracos": {"base": {"indice": "pdf-v1", "camadas": ["c2"], "papel": "baseline"},
                       "aa": {"indice": "pdf-v1", "camadas": ["respostas"], "ordem": 2, "papel": "ruido"},
                       "hdr": {"indice": "pdf-v1", "camadas": ["c2"]}},
            "pares": [["base", "hdr"], ["base", "aa"]], "margem": {"gerador_juiz": [["base", "aa"]]}}
    assert rx.validate_card(card) == []
    cited = {"source_document": "docA.pdf", "sources": ["docA.pdf (pág. 1)"]}
    for arm_id, outputs in (("base", ("[docA.pdf, pág. 1]", "[docZ.pdf, pág. 1]")),
                            ("aa", ("[docA.pdf, pág. 1]", "[docZ.pdf, pág. 1]")),
                            ("hdr", ("[docA.pdf, pág. 1]", "[docA.pdf, pág. 1]"))):
        answers = [{**cited, "name": f"q{i}", "actual_output": f"É isso {o}."} for i, o in enumerate(outputs, 1)]
        data = {"n_cases": 2, "answers": answers, "arm": {"prompt": {"require_citation": True}}}
        (tmp_path / f"respostas_{arm_id}.json").write_text(json.dumps(data), encoding="utf-8")

    result = rx.scoreboard(card, tmp_path, limit=None)

    # O par de margem não tem veredito; o outro usa a margem do A/A do cartão (0, as respostas são iguais).
    assert set(result["verdicts"]) == {"base→hdr"}
    verdict = result["verdicts"]["base→hdr"]
    assert (verdict["metric"], verdict["margin"], verdict["margin_from"]) == ("attribution.misattribution", 0.0,
                                                                              "cartão")
    assert verdict["delta"]["delta"] == -0.5


def test_delta_por_grupo_booleano_e_campo_ausente(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rx, "index_documents", lambda index: {"doc1", "doc2", "doc3"})
    monkeypatch.setattr(rx, "latest_margins", lambda exclude=None: (None, None))
    card = {**SWEEP, "classe_por": "has_acronym", "pares": [["base", "k8"]]}
    for arm_id, hits in (("base", (0, 1, 0)), ("k8", (1, 1, 1))):
        rows = {f"q{i}": (hit, 0.7, False, [hit]) for i, hit in enumerate(hits, start=1)}
        (tmp_path / f"c1_{arm_id}.json").write_text(json.dumps(_c1_sweep(rows, {})), encoding="utf-8")
    goldens = [{"name": "q1", "has_acronym": True}, {"name": "q2", "has_acronym": False}, {"name": "q3"}]

    result = rx.scoreboard(card, tmp_path, limit=None, goldens=goldens)

    groups = {cls: d["delta"] for cls, d in result["by_class"]["base→k8"].items()}
    assert groups == {"True": 1.0, "False": 0.0, "sem classe": 1.0}


def test_retomar_com_caminho_relativo_vira_absoluto(tmp_path: Path, monkeypatch):
    goldens = tmp_path / "goldens.json"
    goldens.write_text("[]", encoding="utf-8")
    card_path = tmp_path / "EX.yaml"
    card_path.write_text(json.dumps({k: v for k, v in CARD.items() if k != "margem"}
                                    | {"pares": [], "goldens": str(goldens)}), encoding="utf-8")
    out = tmp_path / "out" / "EX_1"
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(rx, "scoreboard", lambda card, folder, *args, **kwargs: seen.setdefault("out", folder) and {})
    monkeypatch.setattr(rx, "write_summary", lambda card, folder, result, text: folder / "resumo.md")

    rx.main([str(card_path), "--retomar", "out/EX_1", "--so-placar"])

    assert seen["out"] == out.resolve() and seen["out"].is_absolute()
