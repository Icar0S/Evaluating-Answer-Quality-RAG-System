"""Testes herméticos da validação de critério e do O6: estatística, regras do estudo e os dois scripts.

Os scripts rodam inteiros com o Jev falso, sobre artefatos sintéticos numa pasta
temporária (o log da campanha não é versionado). O que se confere é a mesma
regra do tests/mutation/evaluate.py e do analyze.py, e a convenção do tau.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

import _jev_mutation as mutation
import criterion_validity
import o6_mutation
from _fake_jev import FakeJev
from _jev_shared import CORRECT
from _jev_stats import auc_roc, bootstrap_ci, cohen_kappa, confusion, interpret_kappa, wilson_ci

CHANGED = "Changes or contradicts a value, entity or condition of `expected_output`"


def _correct_if_has_16(state: dict, key: str, question) -> tuple[str, float]:
    """Jev falso de correção: certo quando a resposta traz o 16 da referência."""
    return (CORRECT, 0.9) if "16" in (state["test_case"].get("actual_output") or "") else (CHANGED, 0.9)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# ------------------------------------------------------------ estatística


def test_auc_casos_conhecidos() -> None:
    assert auc_roc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc_roc([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert auc_roc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    # Exemplo clássico: 3 dos 4 pares (positivo, negativo) ordenados certo.
    assert auc_roc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert math.isnan(auc_roc([0.1, 0.2], [1, 1]))


def test_bootstrap_reproduzivel_e_contem_o_ponto() -> None:
    values = [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.65]
    labels = [0, 0, 1, 1, 1, 0, 1, 0]

    def stat(idx: list[int]) -> float:
        return auc_roc([values[i] for i in idx], [labels[i] for i in idx])

    first, second = bootstrap_ci(len(values), stat, n_boot=300), bootstrap_ci(len(values), stat, n_boot=300)
    assert first == second
    assert first[0] <= auc_roc(values, labels) <= first[1]
    assert all(math.isnan(v) for v in bootstrap_ci(0, stat))


def test_wilson_e_kappa() -> None:
    assert wilson_ci(0, 10).low == pytest.approx(0.0, abs=1e-12)
    assert wilson_ci(10, 10).high == pytest.approx(1.0, abs=1e-12)
    half = wilson_ci(5, 10)
    assert (half.point, round(half.low, 4), round(half.high, 4)) == (0.5, 0.2366, 0.7634)
    assert cohen_kappa(["pass", "fail", "pass"], ["pass", "fail", "pass"]) == 1.0
    # 70% de concordância com 50% esperada ao acaso: kappa 0,4.
    a = ["s"] * 5 + ["n"] * 5
    b = ["s"] * 4 + ["n"] + ["s"] * 2 + ["n"] * 3
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert interpret_kappa(0.4) == "moderada"


def test_confusao_na_convencao_do_tau_json() -> None:
    # Positivo = EQUIV (1); previsto positivo = nota >= τ.
    row = confusion([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0], 0.5)

    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["accuracy"] == 0.5


# -------------------------------------------------------- pares rotulados


def test_pares_sao_os_da_calibracao_do_o1() -> None:
    pairs = mutation.load_pairs()
    tau = json.loads(mutation.TAU_JSON.read_text(encoding="utf-8"))

    assert len(pairs) == tau["pairs"] == 247
    assert sum(p.positive for p in pairs) == tau["equiv"] == 111
    assert sum(1 for p in pairs if p.label == "ERRO") == tau["erro"] == 136
    assert all(p.question for p in pairs)
    assert {p.rule for p in pairs if p.label == "ERRO"} == {"swap_number", "replace_entity", "invert_condition"}


def test_validade_de_criterio_ponta_a_ponta(tmp_path: Path) -> None:
    golden = _write_jsonl(tmp_path / "golden.jsonl", [{"case_id": "c01", "question": "Quantas classes?",
                                                       "expected_output": "16 classes."}])
    pairs = _write_jsonl(
        tmp_path / "pairs.jsonl",
        [
            {"pair_id": "p1", "case_id": "c01", "label": "EQUIV", "rule": "paraphrase",
             "reference": "Foram 16 classes.", "candidate": "No total, 16 classes."},
            {"pair_id": "p2", "case_id": "c01", "label": "ERRO", "rule": "swap_number",
             "reference": "Foram 16 classes.", "candidate": "Foram 8 classes."},
            {"pair_id": "p3", "case_id": "c01", "label": "ERRO", "rule": "swap_number", "excluded": True,
             "reference": "Foram 16 classes.", "candidate": "Foram 32 classes."},
        ],
    )
    out = tmp_path / "out"

    code = criterion_validity.main(
        ["--pairs", str(pairs), "--golden", str(golden), "--out-dir", str(out)], model=FakeJev(_correct_if_has_16)
    )

    assert code == 0
    (path,) = out.glob("criterion_*.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    assert (s["pairs"], s["equiv"], s["erro"]) == (2, 1, 1)  # o excluído fica fora
    assert s["auc"] == 1.0
    assert s["at_threshold"]["accuracy"] == 1.0
    assert s["by_rule"]["swap_number"]["causes"] == {"wrong_value": 1}
    assert s["o1_reference"]["auc"] == pytest.approx(0.4846)
    assert [r["pair_id"] for r in data["records"]] == ["p1", "p2"]


# ------------------------------------------------------------- log e regras


def _run(mutant: str, case: str, rep: int, answer: str, **extra) -> dict:
    return {"mutant_id": mutant, "case_id": case, "repetition": rep, "answer": answer,
            "operator": None if mutant == "baseline" else mutant, "error": None, "tokens_out": 100, **extra}


def test_resposta_modal_com_as_regras_do_evaluate() -> None:
    runs = [
        _run("baseline", "c01", 1, "Foram 16."),
        _run("baseline", "c01", 2, "foram 16."),  # mesma resposta normalizada
        _run("baseline", "c01", 3, "Foram 8."),
        _run("baseline", "c01", 4, "x", error="ReadTimeout"),  # erro: fora
        _run("baseline", "c01", 5, "", tokens_out=4096),  # cortada pelo teto: fora
        _run("baseline", "c01", 6, "Foram 8."),  # fora do nível (L2 = 5 execuções)
        _run("baseline", "c01", 7, "Foram 8."),
        _run("l3-baseline", "c01", 1, "Foram 8."),  # baseline do L3: não é mutante
    ]

    (pair,) = mutation.modal_pairs(runs, max_repetitions=5, cap=4096)

    assert (pair.answer, pair.runs) == ("Foram 16.", 3)


def test_conjunto_avaliavel_mortes_e_escore() -> None:
    def v(verdict: str, measured: bool = True) -> dict:
        return {"verdict": verdict, "evaluable_oracle": measured}

    verdicts = {
        ("baseline", "c1"): v("pass"),
        ("baseline", "c2"): v("fail"),  # reprova no baseline: fora de S
        ("baseline", "c3"): v("pass", measured=False),  # não medido: fora de S
        ("M1", "c1"): v("fail"),  # morte
        ("M1", "c2"): v("fail"),  # fora de S: não é morte
        ("M2", "c1"): v("pass"),
        ("M2", "c3"): v("fail"),
        ("M3", "c2"): v("fail"),  # nenhum caso em S: sobrevive, mas conta no denominador
    }

    s = mutation.evaluable_set(verdicts)
    mutation.mark_kills(verdicts, s)
    score = mutation.mutation_score(verdicts, s)

    assert s == {"c1"}
    assert [k for k, r in verdicts.items() if r["kill"]] == [("M1", "c1")]
    # Denominador do analyze.py: todos os mutantes, não só os que têm caso em S.
    assert (score["killed"], score["mutants"]) == (1, 3)
    assert score["operators"]["M1"]["killed"] and not score["operators"]["M2"]["killed"]
    assert score["operators"]["M3"]["evaluable_cases"] == 0
    assert mutation.evaluable_set(verdicts, requires_pass=False) == {"c1", "c2"}


def test_o4_sem_juiz_conta_como_nao_medido() -> None:
    # O verdicts.jsonl não grava evaluable_oracle; O4 sem juiz tem score nulo.
    records = [
        {"oracle": "O4", "level": "L2", "mutant_id": "baseline", "case_id": "c1", "verdict": "pass", "score": None},
        {"oracle": "O4", "level": "L2", "mutant_id": "baseline", "case_id": "c2", "verdict": "pass", "score": 1.0},
        {"oracle": "O4", "level": "L1", "mutant_id": "baseline", "case_id": "c3", "verdict": "pass", "score": 1.0},
    ]

    o4 = mutation.last_records(records, "O4", "L2")

    assert set(o4) == {("baseline", "c1"), ("baseline", "c2")}  # o L1 fica de fora
    assert o4[("baseline", "c1")]["evaluable_oracle"] is False
    assert mutation.evaluable_set(o4) == {"c2"}
    assert "evaluable_oracle" not in records[0]  # não altera os registros lidos


def test_o6_ponta_a_ponta_retomavel_e_comparado_ao_o4(tmp_path: Path) -> None:
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump({
            "baseline_config": {"generation_num_predict": 4096},
            "execution": {"levels": {"L1": 1, "L2": 5, "L3": 30}, "evaluable_requires_baseline_pass": True},
        }),
        encoding="utf-8",
    )
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"case_id": "c01", "question": "Quantas classes?", "expected_output": "16 classes."},
        {"case_id": "c02", "question": "Quantas sementes?", "expected_output": "16 sementes."},
    ])
    runs = _write_jsonl(tmp_path / "runs.jsonl", [
        _run("baseline", "c01", 1, "16 classes."),
        _run("baseline", "c02", 1, "16 sementes."),
        _run("N1", "c01", 1, "8 classes."),  # o número trocado: O6 mata
        _run("N1", "c02", 1, "16 sementes."),
        _run("P2", "c01", 1, "16 classes."),  # sobrevive
    ])
    o4 = _write_jsonl(tmp_path / "verdicts.jsonl", [
        {"level": "L2", "oracle": "O4", "mutant_id": m, "case_id": c, "verdict": verdict, "evaluable": True,
         "score": 1.0}
        for m, c, verdict in [("baseline", "c01", "pass"), ("baseline", "c02", "pass"), ("N1", "c01", "pass"),
                              ("N1", "c02", "pass"), ("P2", "c01", "pass")]
    ])
    out = tmp_path / "o6"
    argv = ["--study", str(study), "--golden", str(golden), "--runs", str(runs), "--verdicts", str(o4),
            "--out-dir", str(out)]

    jev = FakeJev(_correct_if_has_16)
    assert o6_mutation.main(argv, model=jev) == 0
    assert len(jev.calls) == 5

    summary = json.loads((out / "summary_L2.json").read_text(encoding="utf-8"))
    assert summary["o6"]["s_size"] == 2
    assert (summary["o6"]["killed"], summary["o6"]["mutants"]) == (1, 2)
    assert summary["o4"]["killed"] == 0
    assert summary["causes_of_kills"] == {"wrong_value": 1}
    agree = summary["agreement_o6_o4"]
    assert (agree["pairs"], agree["agree"]) == (5, 4)
    assert agree["disagreements_sample"][0]["mutant_id"] == "N1"
    kills = [r for r in map(json.loads, (out / "verdicts_L2.jsonl").read_text(encoding="utf-8").splitlines()) if r["kill"]]
    assert [(r["mutant_id"], r["case_id"]) for r in kills] == [("N1", "c01")]

    # Retomável: nada novo a julgar, nenhuma chamada.
    again = FakeJev(_correct_if_has_16)
    assert o6_mutation.main(argv, model=again) == 0
    assert again.calls == []
    # E o resumo se refaz sem o Jev.
    assert o6_mutation.main([*argv, "--summary-only"]) == 0
