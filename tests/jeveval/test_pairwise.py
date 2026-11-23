"""A comparação em pares do Jev (pairwise.py), hermética: o Jev falso decide cada par por uma regra."""
from __future__ import annotations

import json
from pathlib import Path

import pairwise
import pares
from _fake_jev import FakeJev


def _answers(tmp_path: Path, lengths: dict[str, int]) -> Path:
    answers = [{"name": name, "input": "Quais vieses?", "expected_output": "Posição.",
                "retrieval_context": ["trecho 1", "trecho 2"], "actual_output": "x" * size}
               for name, size in lengths.items()]
    path = tmp_path / "respostas.json"
    path.write_text(json.dumps({"generation_model": "gemma3:4b", "answers": answers}), encoding="utf-8")
    return path


def _run(tmp_path: Path, rule, *extra: str, lengths: dict[str, int] | None = None) -> tuple[dict, FakeJev]:
    jev = FakeJev(rule)
    out = tmp_path / "pares.json"
    code = pairwise.main(["--answers", str(_answers(tmp_path, lengths or {"q#1": 300, "q#2": 600, "q#3": 900})),
                          "--out", str(out), *extra], model=jev)
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8")), jev


def test_o_estado_leva_as_duas_respostas_na_ordem_pedida_e_os_trechos_uma_vez(tmp_path):
    data, jev = _run(tmp_path, lambda state, key, q: "Empate")

    assert data["n_julgamentos"] == 6  # 3 pares x 2 ordens
    state, questions = jev.calls[1]  # o segundo julgamento: q#1 x q#2 na ordem BA
    assert data["julgamentos"][1]["ordem"] == "BA"
    assert (state["answer_A"], state["answer_B"]) == ("x" * 600, "x" * 300)
    assert state["retrieval_context"] == ["trecho 1", "trecho 2"]
    assert set(state) == {"input", "expected_output", "retrieval_context", "answer_A", "answer_B"}
    (question,) = questions.values()
    assert set(question.options) == {"A", "B", "Empate"}
    assert question.instructions.startswith(pares.CRITERION)
    assert all(j["vencedor"] == pares.TIE for j in data["julgamentos"])
    assert data["julgamentos"][0]["probabilidades"] == {"A": 0.0, "B": 0.0, "Empate": 1.0}


def test_jev_que_sempre_escolhe_a_primeira_tem_vies_de_posicao_total(tmp_path):
    data, _ = _run(tmp_path, lambda state, key, q: ("A", 0.8))
    report = pares.pairs_report({"jeveval-par": data["julgamentos"]},
                                pares.answers_by_name(tmp_path / "respostas.json")[1])

    pos = report["juizes"]["jeveval-par"]["posicao"]
    assert pos["consistentes"] == 0 and pos["pares_nas_duas_ordens"] == 3
    assert pos["taxa_do_primeiro"] == 1.0
    assert data["julgamentos"][0]["probabilidades"]["A"] == 0.8


def test_jev_que_prefere_a_mais_longa_tem_vies_de_verbosidade(tmp_path):
    def longer(state, key, q):
        return "A" if len(state["answer_A"]) > len(state["answer_B"]) else "B"

    # Tamanhos fora da ordem dos nomes: na ordem canônica do par, a diferença de tamanho muda de sinal (com a
    # primeira sempre a mais curta, a preferência seria constante e o Spearman, indefinido).
    data, _ = _run(tmp_path, longer, lengths={"q#1": 900, "q#2": 300, "q#3": 600})
    report = pares.pairs_report({"jeveval-par": data["julgamentos"]},
                                pares.answers_by_name(tmp_path / "respostas.json")[1])

    summary = report["juizes"]["jeveval-par"]
    assert summary["posicao"]["consistencia"] == 1.0  # a mesma escolha nas duas ordens
    assert summary["verbosidade"]["taxa_da_mais_longa"] == 1.0
    assert summary["verbosidade"]["spearman"] > 0.8
    assert [r["resposta"] for r in summary["ranking"]] == ["q#1", "q#3", "q#2"]


def test_falha_do_jev_vira_julgamento_nao_julgado(tmp_path):
    jev = FakeJev(fail=RuntimeError("API fora"))
    out = tmp_path / "pares.json"

    code = pairwise.main(["--answers", str(_answers(tmp_path, {"q#1": 1, "q#2": 2})), "--out", str(out),
                          "--uma-ordem"], model=jev)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1  # nenhum julgado
    assert data["julgamentos"][0]["vencedor"] is None
    assert data["julgamentos"][0]["erro"] == "RuntimeError: API fora"


def test_completar_so_manda_ao_jev_o_que_falta(tmp_path):
    answers = _answers(tmp_path, {"q#1": 300, "q#2": 600})
    out = tmp_path / "pares.json"
    flaky = FakeJev(lambda state, key, q: "A")
    assert pairwise.main(["--answers", str(answers), "--out", str(out)], model=flaky) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    data["julgamentos"][1].update(vencedor=None, erro="APIConnectionError")  # a ordem BA caiu
    out.write_text(json.dumps(data), encoding="utf-8")
    again = FakeJev(lambda state, key, q: "B")

    assert pairwise.main(["--answers", str(answers), "--out", str(out), "--completar"], model=again) == 0

    after = json.loads(out.read_text(encoding="utf-8"))
    assert len(again.calls) == 1  # só o que faltava
    assert after["julgamentos"][0] == data["julgamentos"][0]  # o julgado fica como estava
    assert after["julgamentos"][1]["vencedor"] == "q#1" and after["n_julgamentos"] == 2
