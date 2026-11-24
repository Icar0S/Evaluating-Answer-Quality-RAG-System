"""O relatório da pergunta única (pergunta_unica_report.py), com uma rodada sintética numa pasta temporária."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

import pergunta_unica_report as pur

NAMES = ["md-mtbench-vieses#1", "md-mtbench-vieses#2"]
TEXT = {NAMES[0]: "Os vieses de posição e de verbosidade; mais de 80%.", NAMES[1]: "O viés de posição. " * 10}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _metric(name: str, score: float, **extra) -> dict:
    return {"name": name, "score": score, "success": score >= 0.7, "applicable": True, "error": None,
            "threshold": 0.7, **extra}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20261123T110351"
    answers = [{"name": n, "input": "Quais vieses?", "expected_output": "Posição, verbosidade, 80%.",
                "actual_output": TEXT[n], "seed": k, "completion_tokens": 100 + k, "generation_ms": 7000.0,
                "retrieval_context": ["trecho"], "error": None} for k, n in enumerate(NAMES, start=1)]
    _write(d / "respostas.json", {
        "generation_model": "gemma3:4b", "provider": "local", "rag_index": "md-v1", "top_k": 4, "prompt_version": "v1",
        "n_cases": 2, "answers": answers,
        "protocolo": {"sementes": [1, 2], "estado_limpo": True, "temperatura": 1.0,
                      "temperatura_fonte": "configurada", "interrompido": None,
                      "trechos": {"md-mtbench-vieses": {"sha256": "9979d94203e48277", "iguais": True, "respostas": 2}}}})
    nli = {"schema": "NLIStatementOutput", "output": {"statements": [
        {"statement": "Há viés de posição.", "reason": "O trecho diz.", "verdict": 1},
        {"statement": "A deflação de kappa é de posição.", "reason": "Outro artigo.", "verdict": 0}]}}
    for k in (1, 2):
        _write(d / "ragas" / f"x-r{k}.json", {"framework": "ragas", "judge_repetition": k, "threshold": 0.7, "cases": [
            {"name": NAMES[0], "metrics": [_metric("faithfulness", 0.5, steps=[{"schema": "StatementGeneratorOutput",
                                                                            "output": {}}, nli]),
                                           _metric("answer_relevancy", 0.9)]},
            {"name": NAMES[1], "metrics": [_metric("faithfulness", 1.0 if k == 1 else 0.6),
                                           _metric("answer_relevancy", 0.8)]}]})
    _write(d / "deepeval" / "x-r1.json", {"judge_repetition": 1, "threshold": 0.7, "cases": [
        {"name": NAMES[0], "metrics": [_metric("Faithfulness", 2 / 3, reason="Uma afirmação contradiz.", steps={
            "truths": ["t1", "t2", "t3"], "claims": ["c1", "c2", "c3"],
            "verdicts": [{"verdict": "yes", "reason": None}, {"verdict": "idk", "reason": "não diz"},
                         {"verdict": "no", "reason": "contradiz"}]})]},
        {"name": NAMES[1], "metrics": [_metric("Faithfulness", 1.0)]}]})
    breakdown = [
        {"question": "Is every factual claim supported? A claim...", "type": "choice", "weight": 2.0, "value": 0.28,
         "applicable": True, "probabilities": {"No": 0.72, "Yes": 0.28, "Not applicable: x": 0.0}, "confidence": 0.58},
        {"question": "How much of `actual_output` is grounded? A claim...", "type": "choice", "weight": 1.0,
         "value": 0.51, "applicable": True, "probabilities": {"Fabricated": 0.11, "Mostly grounded": 0.89},
         "confidence": 0.22}]
    _write(d / "jeveval" / "x-r1.json", {"framework": "jeveval", "judge_repetition": 1, "threshold": 0.7, "cases": [
        {"name": NAMES[0], "metrics": [_metric("faithfulness", 0.3567, breakdown=breakdown, confidence=0.22),
                                       _metric("correctness", 0.9, threshold=0.5)]},
        {"name": NAMES[1], "metrics": [_metric("faithfulness", 0.8), _metric("correctness", 0.2, threshold=0.5)]}]})
    plan = [{"par": NAMES, "ordem": "AB", "primeiro": NAMES[0], "segundo": NAMES[1]},
            {"par": NAMES, "ordem": "BA", "primeiro": NAMES[1], "segundo": NAMES[0]}]
    _write(d / "pares" / "deepeval-arena.json", {"juiz": "deepeval-arena", "n_julgamentos": 2, "duas_ordens": True,
                                                 "julgamentos": [{**p, "vencedor": p["primeiro"], "erro": None}
                                                                 for p in plan]})
    _write(d / "pares" / "jeveval-par.json", {"juiz": "jeveval-par", "n_julgamentos": 2, "duas_ordens": True,
                                              "julgamentos": [{**p, "vencedor": NAMES[0], "erro": None,
                                                               "probabilidades": {"A": 0.7, "B": 0.2, "Empate": 0.1}}
                                                              for p in plan]})
    return d


def test_anatomia_mostra_a_conta_de_cada_juiz(run_dir):
    a = pur.anatomy(pur.load(run_dir))

    assert a["resposta"] == NAMES[0]
    assert a["ragas"]["conta"] == "inferíveis ÷ total = 1 ÷ 2 = 0,50"
    # No DeepEval, o idk conta como fiel: 2 de 3.
    assert a["deepeval"]["conta"] == "(yes + idk) ÷ total = 2 ÷ 3 = 0,67"
    assert a["deepeval"]["verdades"] == 3
    assert a["jeveval"]["conta"] == "(2 × 0,28 + 1 × 0,51) ÷ 3 = 0,36"
    assert [row[1] for row in a["ragas"]["linhas"]] == ["1", "0"]
    assert all(a[fw]["confere"] for fw in ("ragas", "deepeval", "jeveval"))


def test_a_conta_refeita_bate_com_a_nota_e_o_relatorio_aponta_quando_nao_bate(run_dir, tmp_path):
    # Todos os julgamentos de fidelidade com passos: 2 do RAGAS (só a #1 tem passos), 1 do DeepEval e 1 do Jev.
    assert pur.check_arithmetic(pur.load(run_dir)) == {"conferidas": 4, "batem": 4, "nao_batem": []}

    path = run_dir / "deepeval" / "x-r1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"][0]["metrics"][0]["score"] = 1.0  # a nota gravada deixa de ser a dos vereditos (2 de 3)
    _write(path, data)

    check = pur.check_arithmetic(pur.load(run_dir))
    assert (check["batem"], check["conferidas"]) == (3, 4)
    assert check["nao_batem"][0]["juiz"] == "deepeval" and check["nao_batem"][0]["resposta"] == NAMES[0]
    md = pur.build(run_dir, tmp_path / "rel", ["md"])["md"].read_text(encoding="utf-8")
    assert "**NÃO confere** com a nota (1,0000)" in md
    assert "**3 de 4** batem" in md and "Contas que não batem com a nota" in md


def test_o_relatorio_mostra_o_controle_nulo_o_efeito_e_os_passos_que_nao_fecharam(run_dir, tmp_path):
    import mutacao

    exports = {fw: pur.load(run_dir).exports_of(fw) for fw in ("ragas", "deepeval", "jeveval")}
    original = [e for group in exports.values() for e in group]
    answers = json.loads((run_dir / "respostas.json").read_text(encoding="utf-8"))
    mutant = {"meta": {"nome": "Remover ancoragem", "esperado": "cair"}, "answers": answers,
              "exports": exports["ragas"][:1]}
    _write(run_dir / "mutantes-analise.json", mutacao.mutation_report(original, answers, {"P1": mutant}))
    _write(run_dir / "execucao.json", {
        "maquina": {"energia_no_inicio": {"tomada": True, "bateria": 90}, "suspensao_por_inatividade_bloqueada": True},
        "passos": [{"passo": "jeveval (julgamento 1)", "estado": "incompleta", "detalhe": "3/30 notas",
                    "suspensoes": [{"de": "2026-11-23T11:31:00", "ate": "2026-11-23T13:22:00", "segundos": 6660.0}]}]})

    md = pur.build(run_dir, tmp_path / "rel", ["md"])["md"].read_text(encoding="utf-8")

    assert "Esta rodada tem passos que não fecharam" in md and "3/30 notas" in md
    assert "suspendeu no meio" in md and "111 min" in md
    assert "2/2 iguais ao original" in md
    assert "RAGAS: 2 comparações sem defeito" in md and "Controle nulo" in md


def test_relatorio_em_markdown_pdf_e_planilha(run_dir, tmp_path):
    out = tmp_path / "relatorio"

    outputs = pur.build(run_dir, out, ["md", "pdf", "xlsx"])

    md = outputs["md"].read_text(encoding="utf-8")
    for section in ("1. O protocolo", "2. As cinco respostas", "3. Os três juízes", "4. A anatomia de uma nota",
                    "5. Os pares", "6. Os mutantes", "7. O checklist"):
        assert section in md
    # Sem mutantes, a seção diz o que rodar; o checklist sai com os itens que esta rodada permite conferir.
    assert "rodar.py --mutantes" in md
    assert "| REC-2 |" in md and "| PRO-1 |" in md
    assert "9979d94203e48277" in md
    assert "notas_por_resposta.png" in md
    # O ArenaGEval falso escolhe sempre o primeiro: 0 de 1 par com o mesmo vencedor nas duas ordens.
    assert "| DeepEval (ArenaGEval) |" in md and "| 0/1 |" in md
    assert outputs["pdf"].stat().st_size > 10_000
    wb = load_workbook(outputs["xlsx"])
    assert wb.sheetnames == ["Respostas", "Notas", "Decomposicao", "Pares"]
    # Uma linha por nota: RAGAS 2 julgamentos x 2 respostas x 2 métricas, DeepEval 1 x 2 x 1, Jev 1 x 2 x 2.
    assert wb["Notas"].max_row == 1 + 8 + 2 + 4
    assert wb["Pares"].max_row == 1 + 2 + 2
