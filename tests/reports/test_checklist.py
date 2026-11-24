"""O checklist de testes de RAG (tests/checklist/checklist.yaml e avaliar.py), com rodadas sintéticas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "checklist"))

import avaliar  # noqa: E402

DOC = "NeurIPS-2023-judging-llm-as-a-judge-with-mt-bench-and-chatbot-arena-Paper-Datasets_and_Benchmarks.md"
GOOD = ("Os vieses de posição, de verbosidade e de autoaprimoramento, e a capacidade de raciocínio limitada; o GPT-4 "
        "passa de 80% de concordância.")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _run(tmp_path: Path, *, sources=(DOC,), texts=(GOOD, GOOD), english=True,
         judges=("ragas", "deepeval", "jeveval"), faithfulness=0.9) -> Path:
    d = tmp_path / "rodada"
    answers = [{"name": f"q#{k}", "actual_output": t, "source_documents": list(sources), "seed": k,
                "semente_pedida": k, "estado": "frio"} for k, t in enumerate(texts, start=1)]
    _write(d / "respostas.json", {"generation_model": "gemma3:4b", "rag_index": "md-v1", "answers": answers,
                                  "protocolo": {"estado_limpo": True,
                                                "trechos": {"q": {"sha256": "abc", "iguais": True, "respostas": 2}}}})
    if english:
        _write(d / "ingles" / "respostas.json", {"answers": [
            {"name": f"q#{k}", "semente_pedida": k, "actual_output": "Position, verbosity and self-enhancement biases, "
                                                                     "limited reasoning ability; over 80% agreement."}
            for k in (1, 2)]})
    for fw, model in (("ragas", "qwen3:8b"), ("deepeval", "qwen3:8b"), ("jeveval", "jev-1.13.0 (TypeSafe AI)")):
        for k in (1, 2) if fw in judges else ():
            data = {"judge_model": model, "judge_repetition": k, "threshold": 0.7, "cases": [
                {"name": f"q#{i}", "metrics": [{"name": "faithfulness" if fw != "deepeval" else "Faithfulness",
                                                "score": faithfulness, "applicable": True, "threshold": 0.7}]}
                for i in (1, 2)]}
            if fw != "deepeval":
                data["framework"] = fw
            _write(d / fw / f"x-r{k}.json", data)
    plan = [("AB", "q#1", "q#2"), ("BA", "q#2", "q#1")]
    _write(d / "pares" / "deepeval-arena.json", {"juiz": "deepeval-arena", "julgamentos": [
        # Um juiz com viés de posição: escolhe sempre a primeira resposta.
        {"par": ["q#1", "q#2"], "ordem": o, "primeiro": f, "segundo": s, "vencedor": f, "erro": None}
        for o, f, s in plan]})
    _write(d / "mutantes-analise.json", {"mutantes": {
        "N0": {"comportamento": {"respostas": 2, "abstencoes": 0, "com_o_artigo_nos_trechos": 2},
               "juizes": {"ragas": {"morto": False, "estado": "ruído"}}},
        "C1": {"comportamento": {"respostas": 2, "abstencoes": 0, "com_o_artigo_nos_trechos": 0},
               "juizes": {"jeveval": {"morto": True, "estado": "morto"}, "ragas": {"morto": False, "estado": "vivo"}}},
        "P1": {"comportamento": {"respostas": 2, "abstencoes": 0, "com_o_artigo_nos_trechos": 2},
               "juizes": {fw: {"morto": False, "estado": "ruído" if fw == "ragas" else "vivo"} for fw in judges}}}})
    return d


def _states(results) -> dict[str, str]:
    return {r["id"]: r["estado"] for r in results}


def test_o_checklist_e_valido_e_cada_item_diz_como_e_conferido():
    data = avaliar.load_checklist()

    ids = [item["id"] for item in data["itens"]]
    assert len(ids) == 16 == len(set(ids))
    assert {item["tipo"] for item in data["itens"]} == {"MFT", "INV", "DIR"}
    assert all(any(k in item for k in ("evidencia", "teste", "fora")) for item in data["itens"])
    # Critério mudado depois dos dados fica escrito no próprio checklist, com os itens que muda e o efeito.
    assert data["versao"] == 2
    assert all(set(r) == {"data", "itens", "mudanca", "efeito"} and set(r["itens"]) <= set(ids)
               for r in data["revisoes"])


def test_estados_de_uma_rodada_completa(tmp_path):
    states = _states(avaliar.evaluate(avaliar.load_run(_run(tmp_path))))

    assert states["REC-1"] == states["REC-2"] == states["GER-1"] == states["GER-2"] == "passou"
    assert states["GER-3"] == states["GER-5"] == states["PRO-2"] == states["PRO-3"] == "passou"
    assert states["PRO-1"] == states["JUI-1"] == "coberto"
    assert states["JUI-2"] == "falhou"  # o par muda de vencedor quando a ordem muda (0 de 1 consistente)
    # C1 sem nenhuma abstenção: falha com registro; P1 vivo nos três: o JUI-5 falha.
    assert states["GER-4"] == "falhou" and states["JUI-5"] == "falhou"
    assert states["JUI-4"] == "passou"  # 0,9 nos dois julgamentos
    assert states["PRI-1"] == "passou"


def test_juiz_que_faltou_e_sem_dado_e_nao_falha(tmp_path):
    by_id = {r["id"]: r for r in avaliar.evaluate(avaliar.load_run(_run(tmp_path, judges=("ragas", "deepeval"))))}

    # Os dois juízes que julgaram passam do limiar: sem o Jev, o item não passa, mas também não falha.
    assert by_id["GER-2"]["estado"] == "sem dado" and "sem dado de: jeveval" in by_id["GER-2"]["detalhe"]
    assert by_id["JUI-4"]["estado"] == "sem dado"
    # O P1 está vivo nos dois que julgaram, mas o terceiro poderia matá-lo.
    assert by_id["JUI-5"]["estado"] == "sem dado" and "jeveval" in by_id["JUI-5"]["detalhe"]
    # O par só tem o ArenaGEval, e ele já reprova a consistência: isso é falha, com ou sem o Jev.
    assert by_id["JUI-2"]["estado"] == "falhou"


def test_nota_ruim_medida_falha_mesmo_com_juiz_faltando(tmp_path):
    by_id = {r["id"]: r for r in avaliar.evaluate(avaliar.load_run(_run(tmp_path, judges=("ragas",), faithfulness=0.3)))}

    assert by_id["GER-2"]["estado"] == "falhou"


def test_ruido_e_n0_nao_contam_como_morte(tmp_path):
    by_id = {r["id"]: r for r in avaliar.evaluate(avaliar.load_run(_run(tmp_path)))}

    assert by_id["JUI-5"]["estado"] == "falhou"
    assert "C1: morto por jeveval" in by_id["JUI-5"]["detalhe"]
    assert "P1: nenhuma morte confirmada (ruído em ragas)" in by_id["JUI-5"]["detalhe"]
    assert "N0" not in by_id["JUI-5"]["detalhe"]


def test_sem_rodada_so_os_testes_tem_estado():
    states = _states(avaliar.evaluate(None))

    assert states["PRO-1"] == "coberto" and states["PRO-3"] == "coberto"
    assert states["REC-1"] == "sem dado"


def test_fatos_faltando_e_documento_fora_da_lista(tmp_path):
    run = avaliar.load_run(_run(tmp_path, sources=(DOC, "segredo-interno.pdf"),
                                texts=(GOOD, "Só o viés de posição."), english=False))
    by_id = {r["id"]: r for r in avaliar.evaluate(run)}

    assert by_id["GER-1"]["estado"] == "falhou" and "#2" in by_id["GER-1"]["detalhe"]
    assert by_id["GER-5"]["estado"] == "falhou"
    assert by_id["GER-3"]["estado"] == "sem dado"
    assert by_id["PRI-1"]["estado"] == "falhou" and "segredo-interno.pdf" in by_id["PRI-1"]["detalhe"]


@pytest.mark.parametrize("text, facts", [
    ("position, verbosity, and self-enhancement; over 80%", {"posição", "verbosidade", "autoaprimoramento", "80%"}),
    ("limited reasoning ability", {"raciocínio"}),
    ("limitações na capacidade de raciocínio", {"raciocínio"}),
    ("self-preference bias", set()),  # em aberto com o orientador: não conta como self-enhancement
    ("viés de posição e auto-elogio", {"posição", "autoaprimoramento"}),
    ("nada aqui", set()),
])
def test_oraculo_de_fatos(text, facts):
    assert avaliar.facts_in(text) == facts
