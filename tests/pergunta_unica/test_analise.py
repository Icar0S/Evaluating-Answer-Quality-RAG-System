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


@pytest.mark.parametrize("generator, judge, expected", [
    (0.0, 0.0, "nenhum"),  # visto na rodada de 23/11/2026: nada variou, e a tabela acusava o juiz
    (0.0, 0.02, "juiz"), (0.33, 0.02, "gerador"), (0.1, None, None), (None, 0.1, None)])
def test_sem_variacao_ninguem_domina(generator, judge, expected):
    assert analise.dominant(generator, judge) == expected


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
    # O DeepEval grava o nome de exibição ("Faithfulness"): a análise usa a chave dos outros dois.
    exports = [_export(None, 1, {"a#1": {"Faithfulness": 1.0}}),
               _export("jeveval", 1, {"a#1": {"correctness": 0.6}}, threshold=0.5),
               _export("jeveval", 2, {"a#1": {"correctness": 0.4}}, threshold=0.5)]

    report = analise.variance_report(exports)

    assert list(report["frameworks"]) == ["deepeval", "jeveval"]
    assert report["frameworks"]["jeveval"]["metricas"]["correctness"]["vereditos_instaveis"] == ["a#1"]
    table = analise.markdown(report)
    assert "| deepeval | faithfulness | 1,00 |" in table
    assert "| jeveval | correctness | 0,50 | 0,40–0,60 |" in table


def test_buraco_e_erro_ou_timeout_e_nao_o_na_do_jev():
    export = _export("jeveval", 1, {"a#1": {"faithfulness": None, "answer_relevancy": 0.9},
                                    "a#2": {"faithfulness": 0.8, "answer_relevancy": 0.9}},
                     not_applicable=(("a#2", "answer_relevancy"),))

    assert analise.unmeasured(export) == [("a#1", "faithfulness")]


def test_a_nova_tentativa_preenche_so_o_buraco_e_nunca_troca_nota():
    export = _export("ragas", 1, {"a#1": {"faithfulness": None, "answer_relevancy": 0.4}, "a#2": {"faithfulness": 0.8}})
    export["metrics_avg"] = {"faithfulness": 0.8, "answer_relevancy": 0.4}
    # A tentativa julga o caso inteiro de novo: a relevância dela (1,0) não pode substituir a que já existia (0,4).
    retry = _export("ragas", 1, {"a#1": {"faithfulness": 0.6, "answer_relevancy": 1.0}})

    filled = analise.merge_retry(export, retry, 2, "tentativas/x-r1.json")

    metrics = {m["name"]: m for m in export["cases"][0]["metrics"]}
    assert filled == [("a#1", "faithfulness")]
    assert metrics["faithfulness"]["score"] == 0.6 and metrics["faithfulness"]["tentativa"] == 2
    assert metrics["faithfulness"]["erro_anterior"] == "timeout"
    assert metrics["answer_relevancy"]["score"] == 0.4 and "tentativa" not in metrics["answer_relevancy"]
    assert export["metrics_avg"] == {"faithfulness": 0.7, "answer_relevancy": 0.4}
    assert export["tentativas"] == [{"tentativa": 2, "arquivo": "tentativas/x-r1.json",
                                     "preenchidas": [["a#1", "faithfulness"]]}]
    assert analise.unmeasured(export) == []


# ------------------------------------------------------------- o orquestrador


class FakeRun:
    """O que os testes combinam com os subprocessos falsos."""

    gaps: dict[str, set[str]] = {}
    only: dict[str, list[str]] = {}


@pytest.fixture
def fake_run(tmp_path: Path, monkeypatch):
    """rodar.Run com os subprocessos trocados por funções que escrevem o que cada passo escreveria."""
    calls: list[str] = []
    unloaded: list[tuple[str, str]] = []
    # Os buracos que cada subprocesso deixa: nome do log -> {(resposta, métrica)}. Um log por tentativa.
    FakeRun.gaps, FakeRun.only = {}, {}
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://juiz:11434")
    monkeypatch.setattr(rodar, "RETRY_PAUSE_SECONDS", 0)
    monkeypatch.setattr(rodar.energia, "power", lambda: {"tomada": True, "bateria": 100})
    monkeypatch.setattr(rodar.run_all, "typesafe_key_configured", lambda: False)
    monkeypatch.setattr(rodar.run_all, "unload_model", lambda url, model: unloaded.append((url, model)) or True)
    monkeypatch.setattr(rodar.run_all, "PY", {k: Path(__file__) for k in rodar.run_all.PY})  # "existem"

    def fake_call(self, key, cmd, log_name):
        calls.append(log_name)
        cmd = [str(c) for c in cmd]
        if log_name == "relatorio":
            out = Path(cmd[cmd.index("--pergunta-unica") + 1]) / "relatorio"
            out.mkdir(parents=True, exist_ok=True)
            (out / "relatorio.md").write_text("# relatório", encoding="utf-8")
        elif "respostas" in log_name:
            answers = [{"name": f"q#{k}", "error": None, "actual_output": f"r{k}"} for k in range(1, 3)]
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps({"n_cases": 2, "generation_model": "gemma3:4b", "answers": answers,
                                       "protocolo": {"interrompido": None, "temperatura": 1.0,
                                                     "ollama_url": "http://ollama:11434",
                                                     "trechos": {"q": {"iguais": True}}},
                                       "_cmd": cmd}), encoding="utf-8")
        elif log_name.endswith("preparar"):
            op, out = cmd[cmd.index("--op") + 1], Path(cmd[cmd.index("--dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "mutante.json").write_text(json.dumps({
                "operador": op, "nome": f"nome {op}", "esperado": "cair", "artefatos": ["artefatos/x"],
                "set": {"top_k": "1"} if op == "R1" else {"vector_store_dir": str(out / "indice")},
                "usa_indice_base": op == "R1"}), encoding="utf-8")
        elif log_name.startswith("pares-"):
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            assert "--completar" in cmd
            plan = rodar.pares.pair_plan(["q#1", "q#2"], both_orders="--uma-ordem" not in cmd)
            missing = FakeRun.gaps.get(log_name, set())
            out.write_text(json.dumps({"n_julgamentos": len(plan), "duas_ordens": "--uma-ordem" not in cmd,
                                       "julgamentos": [rodar.pares.judgment(item, None if item["ordem"] in missing
                                                                            else item["primeiro"])
                                                       for item in plan]}), encoding="utf-8")
        else:
            prefix, k = log_name.split("-tentativa")[0].rsplit("-r", 1)
            judge = prefix.split("-")[-1]
            assert cmd[cmd.index("--repeticao") + 1] == k == cmd[cmd.index("--ordem-semente") + 1]
            assert ("--passos" in cmd) == (judge != "jeveval")
            out = Path(cmd[cmd.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            # No mutante, a nota cai abaixo do limiar: os juízes o matam.
            score = 0.1 if "mutantes" in out.parts else 0.7 + int(k) / 10
            missing = FakeRun.gaps.get(log_name, set())
            names = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--only"] or ["q#1", "q#2"]
            FakeRun.only[log_name] = names

            def metric(name: str) -> dict:
                got = None if name in missing else score
                return {"name": "faithfulness", "score": got, "success": None if got is None else got >= 0.7,
                        "applicable": True, "error": None if got is not None else "timeout", "threshold": 0.7}

            (out / f"2026-r{k}.json").write_text(json.dumps({
                "framework": judge, "judge_repetition": int(k), "threshold": 0.7, "n_cases": len(names),
                "cases_done": len(names), "cases": [{"name": n, "metrics": [metric(n)]} for n in names]}),
                encoding="utf-8")
        return 0, 1.0

    monkeypatch.setattr(rodar.Run, "call", fake_call)
    monkeypatch.setattr(rodar, "OUT_ROOT", tmp_path)
    return calls, unloaded


def test_orquestrador_gera_libera_o_gerador_julga_por_k_e_analisa(fake_run, tmp_path):
    calls, unloaded = fake_run

    code = rodar.main(["--julgamentos", "2"])

    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert code == 0
    # Por julgamento, os três juízes; depois os pares. O Jev, sem chave, sai pulado nos dois.
    assert calls == ["respostas", "ragas-r1", "deepeval-r1", "ragas-r2", "deepeval-r2", "pares-deepeval", "relatorio"]
    # O gerador sai uma vez; o embedding do RAGAS, antes de cada vez que o DeepEval usa a GPU.
    assert unloaded == [("http://ollama:11434", "gemma3:4b")] + [("http://juiz:11434", rodar.EMBEDDING_MODEL)] * 3
    steps = json.loads((run_dir / "execucao.json").read_text(encoding="utf-8"))["passos"]
    assert [s["estado"] for s in steps if "jeveval" in s["passo"]] == ["pulada", "pulada", "pulada"]
    report = json.loads((run_dir / "analise.json").read_text(encoding="utf-8"))
    assert report["frameworks"]["ragas"]["julgamentos"] == [1, 2]
    assert report["frameworks"]["ragas"]["metricas"]["faithfulness"]["dp_juiz"] == pytest.approx(0.05)
    # Os pares: o ArenaGEval falso escolhe sempre o primeiro (viés de posição total); as notas pontuais são
    # iguais nas duas respostas, então a preferência implícita é empate.
    pairs = json.loads((run_dir / "pares-analise.json").read_text(encoding="utf-8"))
    assert pairs["juizes"]["deepeval-arena"]["posicao"]["taxa_do_primeiro"] == 1.0
    assert pairs["juizes"]["ragas (implícita)"]["resultado_por_par"] == {"q#1 x q#2": "empate"}


def test_pares_numa_ordem_so_e_sem_pares(fake_run, tmp_path):
    calls, _ = fake_run

    assert rodar.main(["--julgamentos", "1", "--pular", "ragas", "--uma-ordem"]) == 0
    assert calls[-2:] == ["pares-deepeval", "relatorio"]
    calls.clear()
    assert rodar.main(["--julgamentos", "1", "--pular", "ragas", "--sem-pares"]) == 0
    assert "pares-deepeval" not in calls


def test_retomar_nao_refaz_o_que_ja_tem_arquivo_completo(fake_run, tmp_path):
    calls, _ = fake_run
    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval"]) == 0
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    calls.clear()

    assert rodar.main(["--julgamentos", "2", "--pular", "deepeval", "--retomar", str(run_dir)]) == 0

    assert calls == ["ragas-r2", "relatorio"]  # as respostas e o ragas-r1 já existiam


def test_api_sem_ollama_para_antes_de_gerar(fake_run):
    calls, _ = fake_run

    assert rodar.main(["--api", "http://localhost:8010"]) == 1
    assert calls == []


def test_mutantes_preparam_geram_com_as_sobrescritas_julgam_e_viram_tabela(fake_run, tmp_path):
    calls, unloaded = fake_run

    assert rodar.main(["--julgamentos", "2", "--sem-pares", "--pular", "deepeval", "--mutantes", "C1", "R1"]) == 0

    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert calls == ["respostas", "ragas-r1", "ragas-r2",
                     "mutante-C1-preparar", "mutante-C1-respostas", "mutante-C1-ragas-r1",
                     "mutante-R1-preparar", "mutante-R1-respostas", "mutante-R1-ragas-r1", "relatorio"]
    # O gerador sai da memória antes de cada rodada de juízes (sem DeepEval, o embedding fica).
    assert unloaded == [("http://ollama:11434", "gemma3:4b")] * 3
    # O C1 busca no índice dele (sem --index); o R1 usa o md-v1 com top_k=1.
    c1 = json.loads((run_dir / "mutantes" / "C1" / "respostas.json").read_text(encoding="utf-8"))["_cmd"]
    r1 = json.loads((run_dir / "mutantes" / "R1" / "respostas.json").read_text(encoding="utf-8"))["_cmd"]
    assert "--index" not in c1 and any(c.startswith("vector_store_dir=") for c in c1)
    assert r1[r1.index("--index") + 1] == "md-v1" and "top_k=1" in r1
    report = json.loads((run_dir / "mutantes-analise.json").read_text(encoding="utf-8"))
    # Com 2 julgamentos do original, o controle nulo existe (deixando um de fora) e não reprova nada: a morte vale.
    assert report["controle_nulo"]["ragas"] == {"comparacoes": 2, "n0": False, "reacoes": {}}
    assert report["mutantes"]["C1"]["juizes"]["ragas"]["estado"] == "morto"
    assert "**morto**" in (run_dir / "mutantes-analise.md").read_text(encoding="utf-8")


def test_mutantes_nao_rodam_pela_api(fake_run):
    calls, _ = fake_run

    assert rodar.main(["--api", "http://localhost:8010", "--ollama", "http://o:11435", "--julgamentos", "1",
                       "--sem-pares", "--pular", "ragas", "deepeval", "--mutantes", "P1"]) == 0
    assert not any(c.startswith("mutante") for c in calls)


def test_ingles_e_relatorio_no_fim(fake_run, tmp_path):
    calls, _ = fake_run

    assert rodar.main(["--julgamentos", "1", "--sem-pares", "--pular", "ragas", "deepeval", "--ingles"]) == 0

    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert calls == ["respostas", "respostas-em-inglês", "relatorio"]
    english = json.loads((run_dir / "ingles" / "respostas.json").read_text(encoding="utf-8"))["_cmd"]
    assert english[english.index("--lang") + 1] == "en"
    assert (run_dir / "relatorio" / "relatorio.md").is_file()


# ------------------------------------------------------- buracos, retomada e a máquina


def _run_dir(tmp_path: Path) -> Path:
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return run_dir


def _steps(run_dir: Path) -> dict[str, dict]:
    return {s["passo"]: s for s in json.loads((run_dir / "execucao.json").read_text(encoding="utf-8"))["passos"]}


def test_buraco_e_medido_de_novo_so_no_caso_que_faltou(fake_run, tmp_path):
    calls, _ = fake_run
    FakeRun.gaps = {"ragas-r1": {"q#2"}}

    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares"]) == 0

    run_dir = _run_dir(tmp_path)
    assert calls == ["respostas", "ragas-r1", "ragas-r1-tentativa2", "relatorio"]
    assert FakeRun.only["ragas-r1-tentativa2"] == ["q#2"]
    (export,) = (run_dir / "ragas").glob("*.json")  # a tentativa fica em tentativas/, fora do que a análise lê
    data = json.loads(export.read_text(encoding="utf-8"))
    assert analise.unmeasured(data) == [] and len(data["tentativas"]) == 1
    assert (run_dir / "ragas" / "tentativas" / export.name).is_file()
    step = _steps(run_dir)["ragas (julgamento 1)"]
    assert step["estado"] == "concluída" and "1 numa nova tentativa" in step["detalhe"]


def test_buraco_que_sobra_deixa_o_passo_incompleto_e_a_rodada_nao_sai_com_zero(fake_run, tmp_path):
    FakeRun.gaps = {"ragas-r1": {"q#2"}, "ragas-r1-tentativa2": {"q#2"}}

    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares"]) == 1

    step = _steps(_run_dir(tmp_path))["ragas (julgamento 1)"]
    assert step["estado"] == "incompleta"
    assert step["sem_nota"] == [["q#2", "faithfulness"]] and "#2 faithfulness" in step["detalhe"]


def test_retomar_completa_os_buracos_em_vez_de_dar_por_pronto(fake_run, tmp_path):
    # O defeito da rodada de 23/11/2026: um export cheio de falhas contava como julgamento pronto.
    calls, _ = fake_run
    FakeRun.gaps = {"ragas-r1": {"q#1", "q#2"}}
    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares", "--tentativas", "1"]) == 1
    run_dir = _run_dir(tmp_path)
    calls.clear()

    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares", "--retomar", str(run_dir)]) == 0

    assert calls == ["ragas-r1-tentativa2", "relatorio"]
    assert _steps(run_dir)["ragas (julgamento 1)"]["estado"] == "concluída"


def test_retomar_descarta_o_export_interrompido_no_meio(fake_run, tmp_path):
    calls, _ = fake_run
    assert rodar.main(["--julgamentos", "1", "--pular", "ragas", "deepeval", "--sem-pares"]) == 0
    run_dir = _run_dir(tmp_path)
    (run_dir / "ragas").mkdir()
    (run_dir / "ragas" / "antigo-r1.json").write_text(json.dumps({
        "framework": "ragas", "judge_repetition": 1, "n_cases": 2, "cases_done": 1,
        "cases": [{"name": "q#1", "metrics": []}]}), encoding="utf-8")
    (run_dir / "ragas" / "antigo-r1.csv").write_text("", encoding="utf-8")
    calls.clear()

    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares", "--retomar", str(run_dir)]) == 0

    assert calls == ["ragas-r1", "relatorio"]
    assert [p.name for p in (run_dir / "ragas").glob("*.json")] == ["2026-r1.json"]
    assert sorted(p.name for p in (run_dir / "ragas" / "descartados").iterdir()) == ["antigo-r1.csv", "antigo-r1.json"]


def test_pares_com_buraco_tentam_de_novo_e_contam_so_os_julgados(fake_run, tmp_path):
    calls, _ = fake_run
    FakeRun.gaps = {"pares-deepeval": {"BA"}}
    assert rodar.main(["--julgamentos", "1", "--pular", "ragas"]) == 0
    assert calls[-3:] == ["pares-deepeval", "pares-deepeval-tentativa2", "relatorio"]

    FakeRun.gaps = {"pares-deepeval": {"BA"}, "pares-deepeval-tentativa2": {"BA"}}
    other = tmp_path / "outra"
    assert rodar.main(["--julgamentos", "1", "--pular", "ragas", "--retomar", str(other)]) == 1
    step = _steps(other)["pares (deepeval)"]
    assert step["estado"] == "incompleta" and step["detalhe"] == "1/2 julgados"


def test_na_bateria_a_rodada_nao_comeca(fake_run, tmp_path, monkeypatch, capsys):
    calls, _ = fake_run
    monkeypatch.setattr(rodar.energia, "power", lambda: {"tomada": False, "bateria": 62})

    assert rodar.main(["--julgamentos", "1"]) == rodar.EXIT_ON_BATTERY
    assert calls == [] and list(tmp_path.iterdir()) == []
    assert "bateria (62%)" in capsys.readouterr().err

    assert rodar.main(["--julgamentos", "1", "--pular", "ragas", "deepeval", "--sem-pares", "--na-bateria"]) == 0
    execution = json.loads((_run_dir(tmp_path) / "execucao.json").read_text(encoding="utf-8"))
    assert execution["maquina"]["energia_no_inicio"] == {"tomada": False, "bateria": 62}


def test_a_suspensao_fica_no_passo_em_que_aconteceu(fake_run, tmp_path, monkeypatch, capsys):
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(rodar.energia, "SuspensionWatch",
                        lambda: _REAL_WATCH(interval=3600, clock=lambda: now["t"]))
    real_call = rodar.Run.call

    def call(self, key, cmd, log_name):
        result = real_call(self, key, cmd, log_name)
        if log_name == "ragas-r1":
            now["t"] += 4860  # 81 min: a máquina dormiu durante o RAGAS
        return result

    monkeypatch.setattr(rodar.Run, "call", call)

    assert rodar.main(["--julgamentos", "1", "--pular", "deepeval", "--sem-pares"]) == 0

    steps = _steps(_run_dir(tmp_path))
    assert [g["segundos"] for g in steps["ragas (julgamento 1)"]["suspensoes"]] == [4860.0]
    assert "suspensoes" not in steps["respostas"] and "suspensoes" not in steps["relatório"]
    assert "A MÁQUINA SUSPENDEU" in capsys.readouterr().out


_REAL_WATCH = rodar.energia.SuspensionWatch


def test_o_relogio_que_anda_normal_nao_e_suspensao():
    now = {"t": 0.0}
    watch = _REAL_WATCH(clock=lambda: now["t"])
    for _ in range(100):
        now["t"] += 5
        watch.tick()
    assert watch.take() == []
    now["t"] += 61
    assert [g["segundos"] for g in watch.take()] == [61.0]
    assert watch.take() == []  # cada suspensão é contada uma vez
