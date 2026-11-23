"""As contas da comparação em pares (pares.py), herméticas."""
from __future__ import annotations

import pytest

import pares

NAMES = ["q#1", "q#2", "q#3"]
ANSWERS = {"q#1": {"actual_output": "a" * 300}, "q#2": {"actual_output": "b" * 900},
           "q#3": {"actual_output": "c" * 600}}


def _j(a: str, b: str, order: str, winner: str | None) -> dict:
    first, second = (a, b) if order == "AB" else (b, a)
    return pares.judgment({"par": [a, b], "ordem": order, "primeiro": first, "segundo": second}, winner)


def test_plano_dez_pares_vinte_julgamentos():
    names = [f"q#{k}" for k in range(1, 6)]

    plan = pares.pair_plan(names)
    one = pares.pair_plan(names, both_orders=False)

    assert (len(plan), len(one)) == (20, 10)
    assert plan[0] == {"par": ["q#1", "q#2"], "ordem": "AB", "primeiro": "q#1", "segundo": "q#2"}
    assert plan[1] == {"par": ["q#1", "q#2"], "ordem": "BA", "primeiro": "q#2", "segundo": "q#1"}


def test_preferencia_implicita_so_com_as_metricas_da_resposta_e_a_margem():
    def export(scores):
        return {"cases": [{"name": n, "metrics": [{"name": m, "score": s, "applicable": True}
                                                  for m, s in metrics.items()]} for n, metrics in scores.items()]}

    # As de contexto (iguais por construção) diferem só por ruído: não podem decidir o par.
    exports = [export({"q#1": {"faithfulness": 0.9, "answer_relevancy": 0.9, "context_recall": 0.0},
                       "q#2": {"faithfulness": 0.88, "answer_relevancy": 0.9, "context_recall": 1.0},
                       "q#3": {"faithfulness": 0.5, "answer_relevancy": 0.7, "context_recall": 1.0}})]

    means = pares.pointwise_means(exports)
    judgments = pares.implicit_judgments(means, NAMES)

    assert means["q#1"] == pytest.approx(0.9)
    by_pair = {tuple(j["par"]): j["vencedor"] for j in judgments}
    assert by_pair[("q#1", "q#2")] == pares.TIE  # 0,01 de diferença, dentro da margem de 0,05
    assert by_pair[("q#1", "q#3")] == "q#1"
    assert by_pair[("q#2", "q#3")] == "q#2"
    assert all(j["ordem"] is None for j in judgments)


def test_preferencia_implicita_do_deepeval_le_o_nome_de_exibicao():
    export = {"cases": [{"name": "q#1", "metrics": [{"name": "Faithfulness", "score": 0.5},
                                                    {"name": "Answer Relevancy", "score": 1.0},
                                                    {"name": "Contextual Recall", "score": 0.0}]}]}

    assert pares.pointwise_means([export]) == {"q#1": 0.75}


def test_posicao_ranking_e_resultado_por_par():
    judgments = [
        _j("q#1", "q#2", "AB", "q#1"), _j("q#1", "q#2", "BA", "q#2"),  # inconsistente: quem vem primeiro vence
        _j("q#1", "q#3", "AB", "q#1"), _j("q#1", "q#3", "BA", "q#1"),  # consistente
        _j("q#2", "q#3", "AB", pares.TIE), _j("q#2", "q#3", "BA", None),  # um empate e um não julgado
    ]

    pos = pares.position_bias(judgments)
    rank = pares.ranking(judgments, NAMES)
    outcome = pares.pair_outcome(judgments)

    assert (pos["pares_nas_duas_ordens"], pos["consistentes"]) == (2, 1)
    assert pos["inconsistentes"] == [["q#1", "q#2"]]
    assert (pos["decididos"], pos["vitorias_de_quem_vem_primeiro"], pos["empates"]) == (4, 3, 1)
    assert rank[0] == {"resposta": "q#1", "taxa_de_vitoria": 0.75}
    assert outcome[("q#1", "q#2")] == pares.TIE and outcome[("q#1", "q#3")] == "q#1"
    assert outcome[("q#2", "q#3")] == pares.TIE


def test_spearman_e_kendall():
    assert pares.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert pares.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert pares.spearman([1, 2, 3], [5, 5, 5]) is None  # preferência constante: indefinido
    assert pares.kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert pares.kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert pares.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8165, abs=1e-4)


def test_relatorio_compara_juizes_e_o_pontual_do_mesmo_framework():
    always_first = [_j(a, b, o, a if o == "AB" else b) for a, b in [("q#1", "q#2"), ("q#1", "q#3"), ("q#2", "q#3")]
                    for o in ("AB", "BA")]
    longer = [_j(a, b, o, max((a, b), key=lambda n: len(ANSWERS[n]["actual_output"])))
              for a, b in [("q#1", "q#2"), ("q#1", "q#3"), ("q#2", "q#3")] for o in ("AB", "BA")]
    implicit = pares.implicit_judgments({"q#1": 0.4, "q#2": 0.9, "q#3": 0.7}, NAMES)

    report = pares.pairs_report({"deepeval-arena": always_first, "jeveval-par": longer}, ANSWERS,
                                implicit={"deepeval": implicit, "jeveval": implicit})

    assert report["juizes"]["deepeval-arena"]["posicao"]["taxa_do_primeiro"] == 1.0
    assert report["juizes"]["jeveval-par"]["verbosidade"]["taxa_da_mais_longa"] == 1.0
    # O Jev "prefere a mais longa" e as notas pontuais dão q#2 > q#3 > q#1: a mesma ordem dos tamanhos.
    assert report["pontual_x_pares"]["jeveval-par"] == {"pares": 3, "iguais": 3, "concordancia": 1.0}
    assert report["pontual_x_pares"]["deepeval-arena"]["iguais"] == 0  # só empates por posição
    assert report["entre_juizes"]["jeveval-par x jeveval (implícita)"]["kendall_tau"] == 1.0
    table = pares.markdown(report)
    assert "| deepeval-arena |" in table and "0/3" in table


def test_cada_resposta_e_alice_em_metade_dos_pares_dela():
    names = [f"q#{k}" for k in range(1, 6)]

    alice = pares.alice_of_pairs(names)

    assert len(alice) == 10 and all(who in pair for pair, who in alice.items())
    assert {n: sum(1 for who in alice.values() if who == n) for n in names} == dict.fromkeys(names, 2)


def _named(a: str, b: str, order: str, winner: str, alice: str) -> dict:
    first, second = (a, b) if order == "AB" else (b, a)
    names = {"Alice": alice, "Bob": b if alice == a else a}
    return pares.judgment({"par": [a, b], "ordem": order, "primeiro": first, "segundo": second}, winner,
                          nomes_mascarados=names)


def test_vies_de_nome_e_vies_de_posicao_se_separam():
    pairs = [("q#1", "q#2"), ("q#1", "q#3"), ("q#2", "q#3")]
    alice = pares.alice_of_pairs(NAMES)
    likes_alice = [_named(a, b, o, alice[(a, b)], alice[(a, b)]) for a, b in pairs for o in ("AB", "BA")]
    likes_first = [_named(a, b, o, a if o == "AB" else b, alice[(a, b)]) for a, b in pairs for o in ("AB", "BA")]

    name_only = pares.judge_summary(likes_alice, NAMES, {})
    position_only = pares.judge_summary(likes_first, NAMES, {})

    # Quem gosta do nome "Alice" é consistente nas duas ordens: não vira viés de posição.
    assert name_only["nome"]["taxa_da_alice"] == 1.0
    assert name_only["posicao"]["consistencia"] == 1.0 and name_only["posicao"]["taxa_do_primeiro"] == 0.5
    # E quem gosta do primeiro não vira viés de nome.
    assert position_only["nome"]["taxa_da_alice"] == 0.5
    assert position_only["posicao"]["taxa_do_primeiro"] == 1.0
    assert position_only["nome"]["alice_veio_primeiro"] == 3 and position_only["nome"]["julgamentos"] == 6
    # O Jev não recebe nomes: sem viés de nome para medir.
    assert pares.judge_summary([_j("q#1", "q#2", "AB", "q#1")], NAMES, {})["nome"] is None


def test_completar_so_aproveita_julgamento_das_mesmas_respostas(tmp_path):
    import json

    sha = pares.answers_sha(ANSWERS)
    done, failed = _j("q#1", "q#2", "AB", "q#1"), _j("q#1", "q#2", "BA", None)
    path = tmp_path / "pares.json"
    path.write_text(json.dumps({"respostas_sha256": sha, "julgamentos": [done, failed]}), encoding="utf-8")

    assert list(pares.already_judged(path, sha)) == [("q#1", "q#2", "AB")]  # o não julgado volta ao juiz
    other = pares.answers_sha({**ANSWERS, "q#1": {"actual_output": "outro texto"}})
    assert pares.already_judged(path, other) == {}
    assert pares.already_judged(tmp_path / "nao-existe.json", sha) == {}


@pytest.mark.parametrize("raw, name", [("Alice", "Alice"), ("$Alice$", "Alice"), (' "Bob" ', "Bob"), ("**Bob**", "Bob"),
                                       ("nenhum dos dois", "nenhum dos dois"), (None, "")])
def test_o_nome_do_vencedor_sem_os_cifroes_que_o_juiz_poe(raw, name):
    # Visto ao vivo em 23/11/2026: o qwen3:8b devolveu "$Alice$", e o ArenaGEval quebrou com KeyError.
    assert pares.clean_name(raw) == name
