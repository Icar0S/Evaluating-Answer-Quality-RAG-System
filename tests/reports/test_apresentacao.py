"""A apresentação gravada (tests/apresentacao.py): o placar lido dos logs, o estado do palco, o servidor e o vídeo.

Herméticos: nenhum container, ffmpeg ou juiz roda aqui. O último teste abre um navegador sem janela, com o backend
falso, para conferir o espelho do E2E (pula sem o Playwright instalado). As linhas dos logs são as que os
run_and_export.py imprimem de verdade (conferidas em 12/11/2026). Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

import report_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, para importar apresentacao
import apresentacao as ap  # noqa: E402
import run_all as ra  # noqa: E402

LIMIARES = {**{k: 0.7 for k in ap.ROTULO}, "correctness": 0.5}

RAGAS_LOG = """$ tests/ragas/.venv/bin/python tests/ragas/run_and_export.py --answers respostas.json

RAGAS 0.4.3 | geração: gemma3:4b (respostas de respostas.json) | juiz: qwen3:8b | 2 goldens x 5 métricas

[1/2] ragas-definicao
    faithfulness       0.333
    answer_relevancy   0.641
    context_precision  0.500
    context_recall     1.000
    context_relevance  1.000
[2/2] md-seven-failure-points-lista
    faithfulness       0.900
    answer_relevancy   0.811
"""

DEEPEVAL_LOG = """Avaliando 2 perguntas — respostas de respostas.json, juiz qwen3:8b
Progresso salvo a cada caso em 20261112T153821Z.json / .html — dá pra abrir no meio da rodada.

[1/2] ragas-definicao (0s decorridos)
      usando a resposta do arquivo...
      Faithfulness           0.00  FAIL  (65s)
      Answer Relevancy       1.00  PASS  (6s)
      Contextual Precision   0.50  FAIL  (13s)
      Contextual Recall      ERRO: TimeoutError()  (300s)
      Contextual Relevancy   0.35  FAIL  (114s)
    concluído em 208s

Rodada completa em 208s (3.5 min).
"""

JEVEVAL_LOG = """JevEval (deepeval 4.3.1) | juiz: jev-1.13.0 (TypeSafe AI) | respostas: respostas.json (gemma3:4b) | 1 casos x 6 métricas
[1/1] ragas-definicao
    faithfulness       N/A
    answer_relevancy   0.669  confiança 0.3
    context_precision  0.350  confiança 0.08
    context_recall     0.348  confiança 0.28
    context_relevance  0.147  confiança 0.76
    correctness        0.600  confiança 0.45
"""


def _placar(framework: str, texto: str) -> dict:
    placar = ap.Placar(framework, "pdf-v1", LIMIARES)
    placar.alimentar(texto)
    return placar.resumo()


def test_placar_le_o_log_do_ragas_e_faz_a_media_de_cada_metrica():
    resumo = _placar("RAGAS", RAGAS_LOG)

    m = resumo["metricas"]
    assert (resumo["casos_feitos"], resumo["casos_total"]) == (2, 2)
    assert m["faithfulness"]["media"] == pytest.approx((0.333 + 0.900) / 2, abs=1e-3)
    assert m["faithfulness"]["n"] == 2 and m["faithfulness"]["passa"] is False
    assert m["context_recall"]["media"] == 1.0 and m["context_recall"]["passa"] is True
    assert m["context_precision"]["n"] == 1  # a 2ª resposta ainda não tinha essa nota
    assert "correctness" not in m  # só o JevEval mede a correção


def test_placar_le_o_log_do_deepeval_com_o_erro_de_uma_metrica():
    resumo = _placar("DeepEval", DEEPEVAL_LOG)

    m = resumo["metricas"]
    assert (resumo["casos_feitos"], resumo["casos_total"]) == (1, 2)
    assert m["faithfulness"]["media"] == 0.0 and m["answer_relevancy"]["media"] == 1.0
    assert m["context_relevance"]["media"] == 0.35
    assert (m["context_recall"]["media"], m["context_recall"]["erros"]) == (None, 1)  # erro não vira nota 0


def test_placar_le_o_log_do_jeveval_com_na_e_o_limiar_proprio_da_correcao():
    resumo = _placar("JevEval", JEVEVAL_LOG)

    m = resumo["metricas"]
    assert (m["faithfulness"]["media"], m["faithfulness"]["na"]) == (None, 1)  # recusa = N/A, fora da média
    assert m["correctness"]["media"] == 0.6 and m["correctness"]["limiar"] == 0.5
    assert m["correctness"]["passa"] is True and m["answer_relevancy"]["passa"] is False


def test_placar_junta_pedacos_de_linha_e_le_o_log_aos_poucos(tmp_path):
    log = tmp_path / "ragas_pdf-v1.log"
    placar = ap.Placar("RAGAS", "pdf-v1", LIMIARES, log)
    meio = RAGAS_LOG.index("context_recall") + 5
    log.write_text(RAGAS_LOG[:meio], encoding="utf-8")
    placar.ler_log()
    assert placar.resumo()["metricas"]["context_recall"]["n"] == 0  # a linha ainda não terminou

    with log.open("a", encoding="utf-8") as f:
        f.write(RAGAS_LOG[meio:])
    placar.ler_log()
    assert placar.resumo()["metricas"]["context_recall"]["media"] == 1.0


def _palco(tmp_path: Path) -> ap.Palco:
    return ap.Palco(tmp_path, {"gerador": "gemma3:4b", "juiz": "qwen3:8b", "embeddings": "nomic-embed-text"},
                    [{"index": "pdf-v1", "label": "PDFs"}], ["ragas", "jeveval"], 1, tmp_path / "video" / "a.mp4")


def test_palco_acompanha_fases_passos_e_o_cartao_do_placar(tmp_path):
    palco = _palco(tmp_path)
    palco.fase(1)
    palco.inicio_passo("subida", "docker compose up -d --build", "docker compose up -d --build", None)
    palco.fim_passo(ra.Step("subida", "docker compose up -d --build", "concluída", "", None, 15.0))
    palco.concluir_fase(1)
    palco.fase(4)
    log = tmp_path / "ragas_pdf-v1.log"
    log.write_text(RAGAS_LOG, encoding="utf-8")
    cartao = ap.Placar("RAGAS", "pdf-v1", LIMIARES)
    palco.placares = [cartao]
    palco.inicio_passo("ragas_pdf-v1", "RAGAS: RAG pdf-v1", "run_and_export.py", log, cartao)

    estado = palco.snapshot()
    assert [f["status"] for f in estado["fases"]] == ["concluída", "concluída", "concluída", "em andamento",
                                                      "pendente"]
    assert estado["passo"]["key"] == "ragas_pdf-v1"
    assert estado["placar"][0]["status"] == "em andamento"
    assert estado["placar"][0]["caso_atual"] == "md-seven-failure-points-lista"

    palco.fim_passo(ra.Step("ragas_pdf-v1", "RAGAS: RAG pdf-v1", "concluída", "", None, 80.0), cartao)
    # Um passo pulado, que nem começou, entra direto com o resultado.
    palco.fim_passo(ra.Step("jeveval_pdf-v1", "JevEval: RAG pdf-v1", "falhou", "1 falhou", None, 3.0))
    palco.concluir_fase(4)

    estado = palco.snapshot()
    fase4 = estado["fases"][3]
    assert [(p["key"], p["status"]) for p in fase4["passos"]] == [("ragas_pdf-v1", "concluída"),
                                                                  ("jeveval_pdf-v1", "falhou")]
    assert fase4["status"] == "com falhas"
    assert estado["placar"][0]["status"] == "concluída" and estado["placar"][0]["caso_atual"] is None
    assert estado["video"].endswith("a.mp4")


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - servidor local do teste
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def test_servidor_do_palco_entrega_o_estado_o_log_e_so_arquivos_da_execucao(tmp_path):
    palco = _palco(tmp_path)
    (tmp_path / "figuras").mkdir()
    (tmp_path / "figuras" / "notas.png").write_bytes(b"\x89PNG")
    (tmp_path.parent / "segredo.txt").write_text("fora da pasta da execução", encoding="utf-8")
    log = tmp_path / "api.log"
    log.write_bytes(b"linha 1\nlinha 2\n")  # bytes: no Windows, o write_text gravaria \r\n
    palco.fase(2)
    palco.inicio_passo("api", "API (pytest)", "pytest tests/api -v", log)
    servidor = ap.criar_servidor(palco)
    base = f"http://127.0.0.1:{servidor.server_address[1]}"
    try:
        status, tipo, corpo = _get(f"{base}/estado")
        assert status == 200 and "json" in tipo
        assert json.loads(corpo)["passo"]["key"] == "api"
        assert palco.visto.is_set()

        primeiro = json.loads(_get(f"{base}/log?desde=0")[2])
        assert primeiro["texto"] == "linha 1\nlinha 2\n" and primeiro["passo"] == "api"
        with log.open("ab") as f:
            f.write(b"linha 3\n")
        segundo = json.loads(_get(f"{base}/log?desde={primeiro['proximo']}")[2])
        assert segundo["texto"] == "linha 3\n"

        assert _get(f"{base}/arquivo/figuras/notas.png")[0:2] == (200, "image/png")
        assert _get(f"{base}/")[2].startswith(b"<!doctype html>")
        assert _get(f"{base}/styles/shared.css")[0] == 200
        assert _get(f"{base}/assets/fonts/IBMPlexMono-Medium-latin.woff2")[0] == 200
        for fora in ("/arquivo/../segredo.txt", "/arquivo/%2e%2e/segredo.txt", "/assets/fonts/../../index.html"):
            with pytest.raises(urllib.error.HTTPError) as erro:
                _get(base + fora)
            assert erro.value.code == 404
    finally:
        servidor.shutdown()


def test_servidor_recebe_os_quadros_do_e2e_e_entrega_o_ultimo(tmp_path):
    palco = _palco(tmp_path)
    servidor = ap.criar_servidor(palco)
    base = f"http://127.0.0.1:{servidor.server_address[1]}"
    try:
        req = urllib.request.Request(f"{base}/espelho.jpg")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - servidor local do teste
            assert resp.status == 204  # nenhum quadro ainda
        for corpo, teste in ((b"\xff\xd8quadro-1", "chat › pergunta"), (b"\xff\xd8quadro-2", None)):
            post = urllib.request.Request(f"{base}/espelho", data=corpo, method="POST",
                                          headers={"Content-Type": "image/jpeg",
                                                   **({"X-Teste": urllib.parse.quote(teste)} if teste else {})})
            with urllib.request.urlopen(post, timeout=5) as resp:  # noqa: S310
                assert resp.status == 204
        with urllib.request.urlopen(f"{base}/espelho.jpg", timeout=5) as resp:  # noqa: S310
            assert (resp.read(), resp.headers["X-Quadro"], resp.headers["Content-Type"]) == (
                b"\xff\xd8quadro-2", "2", "image/jpeg")
        # O nome do teste fica do último quadro que o trouxe.
        assert palco.snapshot()["espelho"] == {"n": 2, "teste": "chat › pergunta"}
        with pytest.raises(urllib.error.HTTPError) as erro:
            urllib.request.urlopen(urllib.request.Request(f"{base}/outra-coisa", data=b"x", method="POST"),
                                   timeout=5)  # noqa: S310
        assert erro.value.code == 404
    finally:
        servidor.shutdown()


def test_mp4_sai_do_mkv_sem_recodificar_e_a_falha_guarda_o_mkv(tmp_path, monkeypatch):
    mkv, mp4 = tmp_path / "apresentacao.mkv", tmp_path / "apresentacao.mp4"
    assert ap.converter_para_mp4("ffmpeg", mkv, mp4) is None  # nada foi gravado

    comandos = []

    def remux(cmd, check=False):
        comandos.append(cmd)
        Path(cmd[-1]).write_bytes(b"mp4")
        return ap.subprocess.CompletedProcess(cmd, 0)

    mkv.write_bytes(b"mkv")
    monkeypatch.setattr(ap.subprocess, "run", remux)
    assert ap.converter_para_mp4("ffmpeg", mkv, mp4) == mp4
    assert comandos[0][comandos[0].index("-c") + 1] == "copy" and "+faststart" in comandos[0]
    assert not mkv.exists()

    mkv.write_bytes(b"mkv")
    monkeypatch.setattr(ap.subprocess, "run", lambda cmd, check=False: ap.subprocess.CompletedProcess(cmd, 1))
    assert ap.converter_para_mp4("ffmpeg", mkv, tmp_path / "outro.mp4") == mkv  # o MKV continua tocável


def test_a_copia_do_ultimo_relatorio_nao_leva_o_video_ainda_gravando(tmp_path):
    out = tmp_path / "20261119T150322"
    (out / "video").mkdir(parents=True)
    (out / "video" / "apresentacao.mkv").write_bytes(b"ainda gravando")
    (out / "e2e" / "video").mkdir(parents=True)  # só a pasta video/ de cima fica de fora
    (out / "relatorio.pdf").write_bytes(b"%PDF")

    latest = report_history.refresh_latest(tmp_path, out)

    assert (latest / "relatorio.pdf").is_file()
    assert not (latest / "video").exists()
    assert (latest / "e2e" / "video").is_dir()


def test_padroes_da_linha_de_comando(monkeypatch):
    monkeypatch.delenv("DEMO_GOLDENS", raising=False)
    args = ap.construir_parser().parse_args([])
    assert (args.goldens, args.avaliadores, args.sem_video, args.assistir) == (
        1, ["ragas", "deepeval", "jeveval"], False, False)

    monkeypatch.setenv("DEMO_GOLDENS", "2")
    assert ap.construir_parser().parse_args([]).goldens == 2
    assert ap.construir_parser().parse_args(["--avaliadores", "jeveval", "ragas"]).avaliadores == ["jeveval", "ragas"]


def test_sem_ffmpeg_para_antes_de_abrir_qualquer_coisa(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apresentacao.py"])
    monkeypatch.setattr(ap, "Apresentacao", lambda args: pytest.fail("não podia começar a rotina"))
    monkeypatch.setattr(ap.shutil, "which", lambda nome: None)

    assert ap.main() == 2


def test_o_navegador_do_e2e_sem_janela_manda_os_quadros_ao_palco(tmp_path):
    # O espelho de e2e/fixtures.ts de verdade: um teste da interface, com o backend falso (page.route), sem
    # janela, mandando a página ao servidor do palco. Sem DEMO_ESPELHO, o fixture não faz nada disso.
    npx = shutil.which("npx")
    if not npx or not (ap.de.E2E_DIR / "node_modules").is_dir():
        pytest.skip("Playwright não instalado em frontend/tests (npm ci)")
    palco = _palco(tmp_path)
    servidor = ap.criar_servidor(palco)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DEMO_")}
    env.update(DEMO_HEADLESS="1", DEMO_SLOWMO="0",
               DEMO_ESPELHO=f"http://127.0.0.1:{servidor.server_address[1]}/espelho")
    try:
        result = subprocess.run(
            [npx, "playwright", "test", "-c", "playwright.demo.config.ts", "--project", "interface",
             "--reporter=line", "-g", "CPU/RAM/GPU aparecem como indisponiveis"],
            cwd=ap.de.E2E_DIR, env=env, capture_output=True, text=True, encoding="utf-8", timeout=240,
        )
    finally:
        servidor.shutdown()

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    quadro, n = palco.quadro()
    assert n > 0 and quadro.startswith(b"\xff\xd8")  # JPEG
    assert "indisponiveis" in palco.snapshot()["espelho"]["teste"]
