"""A análise dos julgamentos repetidos (analise.py) e o orquestrador da pergunta única (rodar.py), herméticos."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import analise
import rodar


def _export(framework: str | None, rep: int, scores: dict[str, dict[str, float | None]], threshold: float = 0.7,
            not_applicable: tuple[str, ...] = ()) -> dict:
    """Um export mínimo: `scores` é resposta -> métrica -> nota."""
    data = {"judge_repetition": rep, "threshold": threshold, "cases": [
        {"name": answer, "metrics": [
            {"name": metric, "score": None if (answer, metric) in not_applicable else score,
             "success": None if score is None else score >= threshold,
             "applicable": (answer, metric) not in not_applicable, "error": None if score is not None else "timeout",
             "threshold": threshold}
            for metric, score in metrics.items()]}
        for answer, metrics in scores.items()]}
    if framework:
        data["framework"] = framework
    return data


def test_separa_a_variacao_do_gerador_da_do_juiz():
    # Fidelidade: as respostas diferem muito (0,2 x 1,0) e o juiz quase não varia -> domina o gerador.
    # Relevância: as respostas são iguais na média e o juiz oscila 0,5 <-> 1,0 -> domina o juiz.
    exports = [
        _export("ragas", 1, {"a#1": {"faithfulness": 0.2, "answer_relevancy": 0.5},
                             "a#2": {"faithfulness": 1.0, "answer_relevancy": 1.0}}),
        _export("ragas", 2, {"a#1": {"faithfulness": 0.2, "answer_relevancy": 1.0},
                             "a#2": {"faithfulness": 0.9, "answer_relevancy": 0.5}}),
    ]

    report = analise.variance_report(exports)
    faith = report["frameworks"]["ragas"]["metricas"]["faithfulness"]
    rel = report["frameworks"]["ragas"]["metricas"]["answer_relevancy"]

    assert report["frameworks"]["ragas"]["julgamentos"] == [1, 2]
    assert faith["media"] == pytest.approx(0.575)
    assert faith["media_por_resposta"] == {"a#1": 0.2, "a#2": 0.95}
    assert faith["dp_gerador"] == pytest.approx(0.375)  # desvio das médias 0,2 e 0,95
    # A raiz da média das variâncias de cada resposta (0 e 0,05²), arredondada a 4 casas como no export.
    assert faith["dp_juiz"] == pytest.approx(math.sqrt((0 + 0.05**2) / 2), abs=1e-4)
    assert faith["domina"] == "gerador"
    assert faith["amplitude_juiz_maxima"] == pytest.approx(0.1)
    assert rel["dp_gerador"] == 0.0 and rel["domina"] == "juiz"
    # Passou num julgamento e reprovou noutro (limiar 0,7): as duas respostas.
    assert rel["vereditos_instaveis"] == ["a#1", "a#2"]
    assert faith["vereditos_instaveis"] == []


def test_na_e_nao_medida_ficam_fora_e_sao_contadas():
    exports = [
        _export("jeveval", 1, {"a#1": {"faithfulness": 0.9}, "a#2": {"faithfulness": 0.8}},
                not_applicable=(("a#2", "faithfulness"),)),
        _export("jeveval", 2, {"a#1": {"faithfulness": None}, "a#2": {"faithfulness": 0.7}}),
    ]

    s = analise.variance_report(exports)["frameworks"]["jeveval"]["metricas"]["faithfulness"]

    assert (s["notas"], s["nao_se_aplica"], s["nao_medidas"]) == (2, 1, 1)
    assert s["media"] == pytest.approx(0.8)
    assert s["dp_juiz"] is None  # nenhuma resposta com 2 notas medidas


def test_deepeval_sem_campo_framework_e_o_limiar_proprio_da_correcao():
    exports = [_export(None, 1, {"a#1": {"faithfulness": 1.0}}),
               _export("jeveval", 1, {"a#1": {"correctness": 0.6}}, threshold=0.5),
               _export("jeveval", 2, {"a#1": {"correctness": 0.4}}, threshold=0.5)]

    report = analise.variance_report(exports)

    assert list(report["frameworks"]) == ["deepeval", "jeveval"]
    assert report["frameworks"]["jeveval"]["metricas"]["correctness"]["vereditos_instaveis"] == ["a#1"]
    table = analise.markdown(report)
    assert "| deepeval | faithfulness | 1,00 |" in table
    assert "| jeveval | correctness | 0,50 | 0,40–0,60 |" in table


# ------------------------------------------------------------- o orquestrador


@pytest.fixture
def fake_run(tmp_path: Path, monkeypatch):
    """rodar.Run com os subprocessos trocados por funções que escrevem o que cada passo escreveria."""
    calls: list[str] = []
    unloaded: list[tuple[str, str]] = []
    monkeypatch.setattr(rodar.run_all, "typesafe_key_configured", lambda: False)
    monkeypatch.setattr(rodar.run_all, "unload_model", lambda url, model: unloaded.append((url, model)) or True)
    monkeypatch.setattr(rodar.run_all, "PY", {k: Path(__file__) for k in rodar.run_all.PY})  # "existem"

    def fake_call(self, key, cmd, log_name):
        calls.append(log_name)
        cmd = [str(c) for c in cmd]
        if log_name == "respostas":
            answers = [{"name": f"q#{k}", "error": None, "actual_output": f"r{k}"} for k in range(1, 3)]
            self.answers.write_text(json.dumps({"n_cases": 2, "generation_model": "gemma3:4b", "answers": answers,
                                                "protocolo": {"interrompido": None, "temperatura": 1.0,
                                                              "ollama_url": "http://ollama:11434",
                                                              "trechos": {"q": {"iguais": True}}}}), encoding="utf-8")
        else:
            judge, k = log_name.rsplit("-r", 1)
            assert cmd[cmd.index("--repeticao") + 1] == k == cmd[cmd.index("--ordem-semente") + 1]
            assert ("--passos" in cmd) == (judge != "jeveval")
            out = Path(cmd[cmd.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            metric = {"name": "faithfulness", "score": 0.5 + int(k) / 10, "success": True, "applicable": True,
                      "error": None}
            (out / f"2026-r{k}.json").write_text(json.dumps({
                "framework": judge, "judge_repetition": int(k), "threshold": 0.7, "n_cases": 2, "cases_done": 2,
                "cases": [{"name": f"q#{i}", "metrics": [metric]} for i in (1, 2)]}), encoding="utf-8")
        return 0, 1.0

    monkeypatch.setattr(rodar.Run, "call", fake_call)
    monkeypatch.setattr(rodar, "OUT_ROOT", tmp_path)
    return calls, unloaded


def test_orquestrador_gera_libera_o_gerador_julga_por_k_e_analisa(fake_run, tmp_path):
    calls, unloaded = fake_run

    code = rodar.main(["--julgamentos", "2"])

    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert code == 0
    # Por julgamento, os três juízes; o Jev, sem chave, sai pulado.
    assert calls == ["respostas", "ragas-r1", "deepeval-r1", "ragas-r2", "deepeval-r2"]
    assert unloaded == [("http://ollama:11434", "gemma3:4b")]
    steps = json.loads((run_dir / "execucao.json").read_text(encoding="utf-8"))["passos"]
    assert [s["estado"] for s in steps if s["passo"].startswith("jeveval")] == ["pulada", "pulada"]
    report = json.loads((run_dir / "analise.json").read_text(encoding="utf-8"))
    assert report["frameworks"]["ragas"]["julgamentos"] == [1, 2]
    assert report["frameworks"]["ragas"]["metricas"]["faithfulness"]["dp_juiz"] == pytest.approx(0.05)


def test_retomar_nao_refaz_o_que_ja_tem_arquivo_completo(fake_run, tmp_path):
    calls, _ = fake_run
    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval"]) == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    calls.clear()

    assert rodar.main(["--julgamentos", "2", "--pular", "deepeval", "--retomar", str(run_dir)]) == 0

    assert calls == ["ragas-r2"]  # as respostas e o ragas-r1 já existiam


def test_api_sem_ollama_para_antes_de_gerar(fake_run):
    calls, _ = fake_run

    assert rodar.main(["--api", "http://localhost:8010"]) == 1
    assert calls == []
