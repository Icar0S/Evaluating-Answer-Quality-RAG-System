"""A apresentação gravada: a aplicação sobe, os testes locais rodam e os avaliadores julgam; no fim, um MP4.

O atalho é o scripts\\apresentacao.bat, que prepara o ambiente (GPU, modelos, portas, Playwright, ffmpeg). Da raiz:

    python tests\\apresentacao.py                                # tudo, 1 pergunta de referência por acervo, vídeo
    python tests\\apresentacao.py --goldens 2                    # 2 perguntas por acervo (vídeo mais longo)
    python tests\\apresentacao.py --avaliadores ragas jeveval    # sem o DeepEval, o mais lento
    python tests\\apresentacao.py --sem-video                    # ensaio, sem gravar
    python tests\\apresentacao.py --assistir                     # abre o palco no navegador, para acompanhar

As fases, na ordem em que o vídeo mostra:

 1. A aplicação sobe: docker compose up -d --build, os dois RAGs no ar, o quadro do status e a homepage.
 2. Os testes locais: API, gerador de relatório e JevEval com o Jev falso (pytest, um teste por linha).
 3. O E2E com o Playwright nos dois RAGs (playwright.demo.config.ts), com o fluxo do chat, as perguntas de
    referência e o "quem responde não é quem julga": o navegador do teste roda sem janela e manda cada quadro da
    página ao palco, que o mostra ao vivo.
 4. Os avaliadores: RAGAS, DeepEval e JevEval julgam as MESMAS respostas que o navegador coletou, num placar que
    se preenche ao vivo, lido do log de cada um.
 5. O relatório consolidado, com as figuras.

O vídeo é o palco (tests/palco/): uma página com a fase, o passo, o log ao vivo, o navegador do E2E e o placar.
Quem a abre e a grava, quadro a quadro, é um Chromium sem janela (frontend/tests/apresentacao/palco.mjs); o estado
vem deste processo, por HTTP local. Nada da tela de quem roda entra no vídeo, e o computador fica livre durante a
gravação. A primeira versão gravava a tela inteira e pegou as outras janelas abertas no meio da rotina
(19/11/2026). O vídeo sai em MKV durante a execução, que não se perde se o processo cair, e vira MP4 no fim:
tests/reports/out/<data>/video/apresentacao.mp4.

Só biblioteca padrão; a gravação precisa do ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_e2e as de  # noqa: E402
import run_all as ra  # noqa: E402

ROOT = ra.ROOT
PALCO_DIR = ROOT / "tests" / "palco"
PALCO_SCRIPT = ROOT / "frontend" / "tests" / "apresentacao" / "palco.mjs"
STACK_STATUS = ROOT / "scripts" / "stack_status.py"
# Os endereços do host (docker-compose.yml). 127.0.0.1, e não localhost, nas chamadas deste processo: no Windows,
# localhost tenta o ::1 antes e pode custar ~2 s por conexão (ver docs/benchmark-status.md).
OLLAMA_HOST_URL = "http://127.0.0.1:11435"
APP_URL = "http://localhost:5510/"

FASES = [
    (1, "A aplicação sobe", "docker compose, os dois RAGs e o frontend"),
    (2, "Os testes locais", "API, gerador de relatório e JevEval com o Jev falso"),
    (3, "O E2E com o Playwright", "o navegador nos dois RAGs, com as perguntas de referência"),
    (4, "Os avaliadores", "RAGAS, DeepEval e JevEval julgam as mesmas respostas"),
    (5, "O relatório", "o consolidado, com os dois RAGs lado a lado"),
]
AVALIADORES = ("ragas", "deepeval", "jeveval")
NOME_AVALIADOR = {"ragas": "RAGAS", "deepeval": "DeepEval", "jeveval": "JevEval"}

def _relativo(caminho: Path) -> str:
    """O caminho a partir da raiz do projeto, se estiver dentro dela (é o que a tela e o resumo mostram)."""
    try:
        return caminho.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(caminho)


# ------------------------------------------------------------------ o placar ao vivo

# Os nomes das métricas como cada framework imprime -> a chave do relatório (tests/reports/report_data.DIMENSIONS).
METRICA = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy", "answer relevancy": "answer_relevancy",
    "context_precision": "context_precision", "contextual precision": "context_precision",
    "context_recall": "context_recall", "contextual recall": "context_recall",
    "context_relevance": "context_relevance", "contextual relevancy": "context_relevance",
    "correctness": "correctness",
}
ROTULO = {
    "faithfulness": "Fidelidade",
    "answer_relevancy": "Rel. resposta",
    "context_precision": "Precisão ctx",
    "context_recall": "Recall ctx",
    "context_relevance": "Rel. contexto",
    "correctness": "Correção",
}
# "[1/2] nome" (RAGAS, JevEval) e "[1/2] nome (0s decorridos)" (DeepEval).
LINHA_CASO = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)(?:\s+\(\d+s decorridos\))?\s*$")
# "    faithfulness       0.333" (RAGAS), "    faithfulness       0.710  confiança 0.39" (JevEval),
# "      Faithfulness           0.00  FAIL  (65s)" e "      Faithfulness           ERRO: ..." (DeepEval).
LINHA_METRICA = re.compile(r"^\s{2,}([A-Za-z][A-Za-z_ ]*?)\s{2,}(N/A|ERRO\b.*|-?\d+(?:\.\d+)?)")


class Placar:
    """Um cartão do placar: um framework julgando as respostas de um RAG, lido do log da avaliação aos poucos.

    O log é o mesmo que o run_all.run_logged grava; nenhum framework precisa saber do palco.
    """

    def __init__(self, framework: str, rag: str, limiares: dict[str, float], log: Path | None = None) -> None:
        self.framework = framework
        self.rag = rag
        self.limiares = limiares
        self.log = log
        self.status = "pendente"
        self.total: int | None = None
        self.caso_atual: str | None = None
        self.casos: dict[str, dict[str, float | str]] = {}
        self._offset = 0
        self._resto = ""

    def alimentar(self, texto: str) -> None:
        """Processa linhas completas; um pedaço de linha fica para a próxima leitura."""
        texto = self._resto + texto
        *linhas, self._resto = texto.split("\n")
        for linha in linhas:
            linha = linha.rstrip("\r")
            caso = LINHA_CASO.match(linha)
            if caso:
                self.total = int(caso.group(2))
                self.caso_atual = caso.group(3)
                self.casos.setdefault(self.caso_atual, {})
                continue
            metrica = LINHA_METRICA.match(linha)
            if metrica and self.caso_atual is not None:
                chave = METRICA.get(metrica.group(1).strip().lower())
                if chave is None:
                    continue
                valor = metrica.group(2)
                if valor == "N/A":
                    self.casos[self.caso_atual][chave] = "N/A"
                elif valor.startswith("ERRO"):
                    self.casos[self.caso_atual][chave] = "erro"
                else:
                    self.casos[self.caso_atual][chave] = float(valor)

    def ler_log(self) -> None:
        if self.log is None or not self.log.is_file():
            return
        with self.log.open("rb") as f:
            f.seek(self._offset)
            dados = f.read()
        self._offset += len(dados)
        self.alimentar(dados.decode("utf-8", "replace"))

    def resumo(self) -> dict:
        metricas = {}
        chaves = [k for k in ROTULO if k != "correctness" or self.framework == "JevEval"]
        for chave in chaves:
            notas = [c[chave] for c in self.casos.values() if isinstance(c.get(chave), float)]
            na = sum(1 for c in self.casos.values() if c.get(chave) == "N/A")
            erros = sum(1 for c in self.casos.values() if c.get(chave) == "erro")
            media = sum(notas) / len(notas) if notas else None
            limiar = self.limiares.get(chave, 0.7)
            metricas[chave] = {
                "rotulo": ROTULO[chave],
                "media": None if media is None else round(media, 3),
                "n": len(notas),
                "na": na,
                "erros": erros,
                "limiar": limiar,
                "passa": None if media is None else media >= limiar,
            }
        feitos = sum(1 for c in self.casos.values() if c)
        return {
            "framework": self.framework,
            "rag": self.rag,
            "status": self.status,
            "casos_feitos": feitos,
            "casos_total": self.total,
            "caso_atual": self.caso_atual if self.status == "em andamento" else None,
            "metricas": metricas,
        }


# ------------------------------------------------------------------ o estado do palco


class Palco:
    """O que a página do palco mostra. Tudo passa por um lock: o servidor lê de outra thread."""

    def __init__(self, out_dir: Path, modelos: dict[str, str], rags: list[dict], avaliadores: list[str],
                 goldens: int | None, video: Path | None) -> None:
        self._lock = threading.Lock()
        self.out_dir = out_dir
        self.inicio = time.monotonic()
        self.visto = threading.Event()  # o palco já pediu o estado ao menos uma vez
        self.placares: list[Placar] = []
        self._log: Path | None = None
        self._quadro: bytes | None = None  # o último quadro do navegador do E2E (o espelho)
        self.estado: dict = {
            "titulo": "Arquitetura de testes do RAG",
            "inicio": datetime.now(timezone.utc).astimezone().isoformat(),
            "modelos": modelos,
            "rags": rags,
            "avaliadores": [NOME_AVALIADOR[a] for a in avaliadores],
            "goldens": goldens,
            "fases": [{"n": n, "nome": nome, "descricao": desc, "status": "pendente", "passos": []}
                      for n, nome, desc in FASES],
            "fase_atual": 0,
            "passo": None,
            "visual": "abertura",
            "dados": {},
            "app_url": APP_URL,
            "espelho": {"n": 0, "teste": None},
            "video": _relativo(video) if video else None,
            "fim": False,
            "fechar": False,
        }

    def _fase(self, n: int) -> dict:
        return self.estado["fases"][n - 1]

    def fase(self, n: int) -> None:
        with self._lock:
            for f in self.estado["fases"]:
                if f["n"] < n and f["status"] in ("pendente", "em andamento"):
                    f["status"] = "concluída"
            self._fase(n)["status"] = "em andamento"
            self.estado["fase_atual"] = n

    def concluir_fase(self, n: int) -> None:
        with self._lock:
            fase = self._fase(n)
            ruins = [p for p in fase["passos"] if p["status"] in ("falhou", "erro")]
            fase["status"] = "com falhas" if ruins else "concluída"

    def inicio_passo(self, key: str, nome: str, comando: str | None, log: Path | None,
                     placar: Placar | None = None) -> None:
        with self._lock:
            fase = self._fase(self.estado["fase_atual"])
            fase["passos"].append({"key": key, "nome": nome, "status": "em andamento", "detalhe": "",
                                   "segundos": None})
            self.estado["passo"] = {"key": key, "nome": nome, "comando": comando, "fase": fase["n"],
                                    "desde": round(time.monotonic() - self.inicio, 1)}
            self._log = log
            if placar is not None:
                placar.log = log
                placar.status = "em andamento"

    def fim_passo(self, step: ra.Step | None, placar: Placar | None = None) -> None:
        if step is None:
            return
        with self._lock:
            for fase in self.estado["fases"]:
                for p in fase["passos"]:
                    if p["key"] == step.key and p["status"] == "em andamento":
                        p.update(status=step.status, detalhe=step.detail, segundos=step.seconds)
                        break
                else:
                    continue
                break
            else:  # passo que nem começou (pulado antes de rodar): entra direto com o resultado
                fase = self._fase(self.estado["fase_atual"])
                fase["passos"].append({"key": step.key, "nome": step.name, "status": step.status,
                                       "detalhe": step.detail, "segundos": step.seconds})
            if placar is not None:
                placar.ler_log()
                placar.status = {"concluída": "concluída", "passou": "concluída", "pulada": "pulada"}.get(
                    step.status, "erro")

    def visual(self, modo: str, **dados) -> None:
        with self._lock:
            self.estado["visual"] = modo
            self.estado["dados"] = dados

    def atualizar(self, **campos) -> None:
        with self._lock:
            self.estado.update(campos)

    def receber_quadro(self, dados: bytes, teste: str | None) -> None:
        """Um quadro do navegador do E2E (e2e/fixtures.ts, DEMO_ESPELHO), com o teste que o produziu."""
        with self._lock:
            self._quadro = dados
            self.estado["espelho"]["n"] += 1
            if teste:
                self.estado["espelho"]["teste"] = teste

    def quadro(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._quadro, self.estado["espelho"]["n"]

    def snapshot(self) -> dict:
        with self._lock:
            for placar in self.placares:
                if placar.status == "em andamento":
                    placar.ler_log()
            estado = json.loads(json.dumps(self.estado, default=str))
            estado["placar"] = [p.resumo() for p in self.placares]
            estado["decorrido_s"] = round(time.monotonic() - self.inicio, 1)
            return estado

    def log_desde(self, offset: int, limite: int = 65536) -> dict:
        with self._lock:
            log = self._log
            key = (self.estado["passo"] or {}).get("key")
        if log is None or not log.is_file():
            return {"passo": key, "texto": "", "proximo": 0}
        with log.open("rb") as f:
            f.seek(0, os.SEEK_END)
            tamanho = f.tell()
            if offset > tamanho:
                offset = 0
            # Primeira leitura de um log grande (o E2E): só o fim, que é o que a tela mostra.
            if offset == 0 and tamanho > limite:
                offset = tamanho - limite
            f.seek(offset)
            dados = f.read(limite)
        return {"passo": key, "texto": dados.decode("utf-8", "replace"), "proximo": offset + len(dados)}


# ------------------------------------------------------------------ o servidor do palco


def _dentro(base: Path, relativo: str) -> Path | None:
    """O arquivo `relativo` dentro de `base`, ou None se ele escapar da pasta (../)."""
    alvo = (base / unquote(relativo)).resolve()
    base = base.resolve()
    return alvo if alvo.is_file() and (alvo == base or base in alvo.parents) else None


def criar_servidor(palco: Palco, porta: int = 0) -> ThreadingHTTPServer:
    estaticos = {
        "/": PALCO_DIR / "index.html",
        "/palco.css": PALCO_DIR / "palco.css",
        "/palco.js": PALCO_DIR / "palco.js",
        "/styles/shared.css": ROOT / "frontend" / "styles" / "shared.css",
    }

    class Handler(BaseHTTPRequestHandler):
        def _enviar(self, status: int, corpo: bytes, tipo: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            for nome, valor in (extra or {}).items():
                self.send_header(nome, valor)
            self.end_headers()
            self.wfile.write(corpo)

        def do_POST(self):  # noqa: N802 - nome da API do http.server
            # Só o espelho: um quadro JPEG do navegador do E2E por requisição (e2e/fixtures.ts).
            tamanho = int(self.headers.get("Content-Length") or 0)
            if urlparse(self.path).path != "/espelho" or not 0 < tamanho <= 8 * 1024 * 1024:
                self._enviar(404, b"nao encontrado", "text/plain; charset=utf-8")
                return
            dados = self.rfile.read(tamanho)
            palco.receber_quadro(dados, unquote(self.headers.get("X-Teste") or "") or None)
            self._enviar(204, b"", "text/plain; charset=utf-8")

        def _arquivo(self, caminho: Path | None) -> None:
            if caminho is None or not caminho.is_file():
                self._enviar(404, b"nao encontrado", "text/plain; charset=utf-8")
                return
            tipo = mimetypes.guess_type(caminho.name)[0] or "application/octet-stream"
            if tipo.startswith("text/") or tipo in ("application/javascript",):
                tipo += "; charset=utf-8"
            self._enviar(200, caminho.read_bytes(), tipo)

        def do_GET(self):  # noqa: N802 - nome da API do http.server
            url = urlparse(self.path)
            if url.path == "/estado":
                palco.visto.set()
                corpo = json.dumps(palco.snapshot(), ensure_ascii=False).encode("utf-8")
                self._enviar(200, corpo, "application/json; charset=utf-8")
            elif url.path == "/log":
                desde = int((parse_qs(url.query).get("desde") or ["0"])[0] or 0)
                corpo = json.dumps(palco.log_desde(desde), ensure_ascii=False).encode("utf-8")
                self._enviar(200, corpo, "application/json; charset=utf-8")
            elif url.path == "/espelho.jpg":
                quadro, n = palco.quadro()
                if quadro is None:
                    self._enviar(204, b"", "image/jpeg")
                else:
                    self._enviar(200, quadro, "image/jpeg", {"X-Quadro": str(n)})
            elif url.path in estaticos:
                self._arquivo(estaticos[url.path])
            elif url.path.startswith("/assets/fonts/"):
                self._arquivo(_dentro(ROOT / "frontend" / "assets" / "fonts", url.path.removeprefix("/assets/fonts/")))
            elif url.path.startswith("/arquivo/"):
                self._arquivo(_dentro(palco.out_dir, url.path.removeprefix("/arquivo/")))
            else:
                self._enviar(404, b"nao encontrado", "text/plain; charset=utf-8")

        def log_message(self, *args):  # o console é dos passos, não das requisições do palco
            pass

    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    return servidor


# ------------------------------------------------------------------ a gravação


def comando_mp4(ffmpeg: str, mkv: Path, mp4: Path) -> list[str]:
    """MKV -> MP4 sem recodificar: só troca o contêiner e põe o índice no começo (o vídeo abre já tocando)."""
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mkv), "-c", "copy",
            "-movflags", "+faststart", str(mp4)]


def converter_para_mp4(ffmpeg: str, mkv: Path, mp4: Path) -> Path | None:
    """O vídeo final: o MP4, ou o MKV se a conversão falhar (ele continua tocável). None se nada foi gravado.

    O palco grava em MKV (frontend/tests/apresentacao/palco.mjs) porque um MKV interrompido ainda abre; um MP4 só
    fica legível depois de fechado.
    """
    if not mkv.is_file() or mkv.stat().st_size == 0:
        return None
    if subprocess.run(comando_mp4(ffmpeg, mkv, mp4), check=False).returncode == 0 and mp4.is_file():
        mkv.unlink()
        return mp4
    return mkv


# ------------------------------------------------------------------ a rotina


def _tempo(segundos: float) -> str:
    return f"{int(segundos // 60)} min {int(segundos % 60):02d} s" if segundos >= 60 else f"{segundos:.0f} s"


def _limiares() -> dict[str, float]:
    """Os limiares das suítes (ambiente, .env ou o padrão de cada uma)."""
    def valor(nome: str, padrao: float) -> float:
        try:
            return float(os.environ.get(nome) or ra.dotenv_value(nome) or padrao)
        except ValueError:
            return padrao

    limiar = valor("RAGAS_THRESHOLD", 0.7)
    return {**{k: limiar for k in ROTULO}, "correctness": valor("JEVEVAL_CORRECTNESS_THRESHOLD", 0.5)}


class Apresentacao:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = datetime.now(timezone.utc).astimezone()
        self.t0 = time.monotonic()
        self.out_dir = ra.OUT_ROOT / self.started.strftime("%Y%m%dT%H%M%S")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.runner = ra.Runner(
            argparse.Namespace(pular=[], rapido=False, limite=None, respostas_separadas=False, qualidade_pytest=False),
            self.out_dir,
        )
        self.modelos = {
            "gerador": os.environ.get("DEMO_GENERATION_MODEL") or "gemma3:4b",
            "juiz": os.environ.get("DEMO_JUDGE_MODEL") or "qwen3:8b",
            "embeddings": os.environ.get("DEMO_EMBEDDING_MODEL") or "nomic-embed-text",
        }
        self.goldens = de.load_goldens(args.goldens)
        self.video = None if args.sem_video else self.out_dir / "video" / "apresentacao.mp4"
        self.palco = Palco(self.out_dir, self.modelos, [{"index": r["index"], "label": r["label"]} for r in de.RAGS],
                           args.avaliadores, args.goldens, self.video)
        self.answer_files: dict[str, Path] = {}
        self.answer_sets: dict[str, dict] = {}
        self.espelho_url: str | None = None  # onde o navegador do E2E entrega os quadros (o palco, em main)

    # utilidades ---------------------------------------------------------------
    def comando(self, key: str, nome: str, cmd: list, cwd: Path = ROOT, env: dict | None = None,
                timeout: float | None = None) -> ra.Step:
        log = self.out_dir / "logs" / f"{key}.log"
        self.palco.inicio_passo(key, nome, " ".join(str(c) for c in cmd), log)
        self.runner.announce(nome)
        code, secs = ra.run_logged(cmd, log, cwd=cwd, env=env, timeout=timeout, echo=True)
        status = "concluída" if code == 0 else "erro"
        detalhe = "" if code == 0 else ("estourou o tempo limite" if code is None else f"saiu com código {code}")
        step = self.runner.record(ra.Step(key, nome, status, detalhe, None, secs, self.runner._rel(log)))
        self.palco.fim_passo(step)
        return step

    def pytest(self, key: str, nome: str, venv: str, alvo: str, extra: list | None = None,
               timeout: float = 1800) -> ra.Step:
        # -vv, e não -v: o pytest.ini da raiz põe -q nos addopts, e os dois se anulam (um teste por linha some).
        extra = [*(extra or []), "-vv"]
        self.palco.inicio_passo(key, nome, f"pytest {alvo} {' '.join(extra)}", self.out_dir / "logs" / f"{key}.log")
        step = self.runner.pytest_step(key, nome, venv, alvo, timeout=timeout, extra=extra)
        self.palco.fim_passo(step)
        return step

    def pular(self, key: str, nome: str, motivo: str, placar: Placar | None = None) -> None:
        step = self.runner.skip(key, nome, motivo)
        self.palco.fim_passo(step, placar)

    # 1 ------------------------------------------------------------------------
    def aplicacao(self) -> bool:
        self.palco.fase(1)
        self.palco.visual("terminal")
        step = self.comando("subida", "docker compose up -d --build", ["docker", "compose", "up", "-d", "--build"],
                            timeout=3600)
        if step.status != "concluída":
            return False
        self.palco.inicio_passo("rags", "Os dois RAGs no ar", "GET /health nas portas 8010 e 8011", None)
        t0, prontos = time.monotonic(), {}
        while time.monotonic() - t0 < 900 and len(prontos) < len(de.RAGS):
            for rag in de.RAGS:
                saude = de._health(rag["url"].replace("localhost", "127.0.0.1"))
                if saude and saude.get("ollama_reachable"):
                    prontos[rag["index"]] = saude.get("vector_store_documents")
            if len(prontos) < len(de.RAGS):
                time.sleep(3)
        ok = len(prontos) == len(de.RAGS)
        detalhe = ", ".join(f"{i}: {n} trechos" for i, n in prontos.items()) or "nenhum respondeu em 15 min"
        self.palco.fim_passo(self.runner.record(ra.Step("rags", "Os dois RAGs no ar", "concluída" if ok else "erro",
                                                        detalhe, None, round(time.monotonic() - t0, 1))))
        if not ok:
            return False
        # Carrega o gerador antes do primeiro teste, como o demo_e2e.bat: a primeira resposta não paga a carga.
        self.palco.inicio_passo("aquecer", f"Carregando o gerador ({self.modelos['gerador']})", None, None)
        t0 = time.monotonic()
        try:
            corpo = json.dumps({"model": self.modelos["gerador"], "keep_alive": "30m"}).encode()
            urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_HOST_URL}/api/generate", data=corpo,
                                                          headers={"Content-Type": "application/json"}), timeout=300)
            status, detalhe = "concluída", "na memória do Ollama do container"
        except OSError as exc:
            status, detalhe = "erro", f"não carregou ({exc}); a primeira resposta vai demorar mais"
        self.palco.fim_passo(self.runner.record(ra.Step("aquecer", f"Carregando o gerador ({self.modelos['gerador']})",
                                                        status, detalhe, None, round(time.monotonic() - t0, 1))))
        # Uma pausa para a aplicação aparecer no vídeo: um passo do palco, não da execução (fica fora do relatório).
        self.palco.inicio_passo("app", "A aplicação no ar", APP_URL, None)
        self.palco.visual("app", quadro=self._quadro())
        time.sleep(self.args.pausa_app)
        self.palco.fim_passo(ra.Step("app", "A aplicação no ar", "concluída", APP_URL, None, self.args.pausa_app))
        self.palco.concluir_fase(1)
        return True

    def _quadro(self) -> dict | None:
        """O quadro do serviço status (scripts/stack_status.py), visto do host."""
        os.environ.update(STATUS_PDF_API="http://127.0.0.1:8010", STATUS_MD_API="http://127.0.0.1:8011",
                          STATUS_FRONTEND="http://127.0.0.1:5510", STATUS_OLLAMA=OLLAMA_HOST_URL)
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("stack_status", STACK_STATUS)
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)
            estado = modulo.collect()
        except Exception as exc:  # noqa: BLE001 - o quadro é enfeite; a rotina segue sem ele
            return {"erro": str(exc)}
        tags = estado["tags"] or set()
        return {
            "rags": [{"label": r["label"], "index": (r["health"] or {}).get("rag_index"),
                      "trechos": (r["health"] or {}).get("vector_store_documents"), "api": r["host_api"]}
                     for r in estado["rags"]],
            "modelos": [{"papel": papel, "modelo": m, "ok": modulo.has_model(tags, m)}
                        for papel, m in estado["models"].items()],
            "avisos": estado["down"] + estado["warnings"],
        }

    # 2 ------------------------------------------------------------------------
    def testes_locais(self) -> None:
        self.palco.fase(2)
        self.palco.visual("terminal")
        self.pytest("api", "API (pytest)", "backend", "tests/api", timeout=900)
        self.pytest("relatorio", "Gerador de relatório (pytest)", "reports", "tests/reports", timeout=900)
        self.pytest("jeveval_hermeticos", "JevEval: testes herméticos (Jev falso)", "jeveval", "tests/jeveval",
                    extra=["-m", "not live"], timeout=900)
        self.palco.concluir_fase(2)

    # 3 ------------------------------------------------------------------------
    def e2e(self) -> None:
        self.palco.fase(3)
        self.palco.visual("espelho")
        npx = shutil.which("npx")
        if not npx or not (de.E2E_DIR / "node_modules").is_dir():
            self.pular("e2e", "E2E ao vivo (Playwright)", "Playwright não instalado (cd frontend\\tests && npm ci)")
            self.palco.concluir_fase(3)
            return
        answers_dir = self.out_dir / "e2e"
        answers_dir.mkdir(parents=True, exist_ok=True)
        results_json = answers_dir / "resultados.json"
        log = self.out_dir / "logs" / "e2e.log"
        nome = "E2E ao vivo nos dois RAGs (Playwright)"
        self.palco.inicio_passo("e2e", nome, "npx playwright test -c playwright.demo.config.ts", log)
        self.runner.announce(nome)
        env = {
            "DEMO_ANSWERS_DIR": str(answers_dir),
            "DEMO_RESULTS_JSON": str(results_json),
            "DEMO_GOLDENS": str(self.args.goldens or ""),
            "RAG_JUDGE_MODEL": self.modelos["juiz"],
            "RAG_JUDGE_OLLAMA_URL": os.environ.get("RAG_JUDGE_OLLAMA_URL") or "http://localhost:11435",
            # Sem janela: cada quadro da página vai ao palco (e2e/fixtures.ts), e é ele que entra no vídeo. A câmera
            # lenta da demonstração (DEMO_SLOWMO, 400 ms) fica, para quem assiste acompanhar.
            "DEMO_HEADLESS": "1",
        }
        if self.espelho_url:
            env["DEMO_ESPELHO"] = self.espelho_url
        code, secs = ra.run_logged([npx, "playwright", "test", "-c", "playwright.demo.config.ts"], log,
                                   cwd=de.E2E_DIR, env=env, timeout=4 * 3600, echo=True)
        if results_json.is_file():
            projetos = de.playwright_counts(json.loads(results_json.read_text(encoding="utf-8")))
            total = {"total": 0, "failed": 0, "errors": 0, "skipped": 0}
            for projeto, contagem in projetos.items():
                segundos = round(contagem.pop("ms") / 1000, 1)
                for k in total:
                    total[k] += contagem[k]
                status, detalhe = ra.status_from(0 if code is not None else None, contagem)
                self.runner.record(ra.Step(f"e2e_{de._slug(projeto)}", f"E2E ao vivo: {projeto}", status, detalhe,
                                           contagem, segundos, self.runner._rel(log)))
            status, detalhe = ra.status_from(0 if code is not None else None, total)
            self.palco.fim_passo(ra.Step("e2e", nome, status, detalhe, total, secs, self.runner._rel(log)))
        else:
            self.palco.fim_passo(self.runner.record(ra.Step(
                "e2e", nome, "erro", f"o Playwright saiu com código {code} sem resultados (ver o log)", None, secs,
                self.runner._rel(log))))
        for rag in de.RAGS:
            index = rag["index"]
            nome_r = f"Respostas do RAG {index} (coletadas no E2E)"
            records = de.read_jsonl(answers_dir / f"respostas_{index}.jsonl")
            if not records:
                self.pular(f"respostas_{index}", nome_r,
                           "nenhuma resposta gravada (RAG fora do ar, índice vazio ou testes pulados)")
                continue
            data = de.build_answers(index, records, self.goldens)
            alvo = self.out_dir / f"respostas_{index}_pt.json"
            alvo.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            falhas = sum(1 for a in data["answers"] if a["error"])
            step = self.runner.record(ra.Step(f"respostas_{index}", nome_r, "concluída",
                                              f"{len(data['answers'])} pergunta(s), {falhas} sem resposta"))
            self.palco.fim_passo(step)
            self.answer_sets[index] = data
            self.answer_files[index] = alvo
        self.palco.concluir_fase(3)

    # 4 ------------------------------------------------------------------------
    def avaliadores(self) -> None:
        self.palco.fase(4)
        judge_url = os.environ.get("APRESENTACAO_OLLAMA_URL") or OLLAMA_HOST_URL
        limiares = _limiares()
        cartoes = {(fw, index): Placar(NOME_AVALIADOR[fw], index, limiares)
                   for fw in self.args.avaliadores for index in self.answer_files}
        with self.palco._lock:
            self.palco.placares = list(cartoes.values())
        self.palco.visual("placar")
        if not self.answer_files:
            for fw in self.args.avaliadores:
                self.pular(fw, f"{NOME_AVALIADOR[fw]}: avaliação", "sem respostas do E2E para julgar")
            self.palco.concluir_fase(4)
            return
        gerador = next((d["generation_model"] for d in self.answer_sets.values() if d.get("generation_model")),
                       self.modelos["gerador"])
        # O ambiente dos juízes: o Ollama do container, o juiz da demonstração, e o gerador só para o aviso de
        # self-grading. A mesma configuração do demo_e2e.py, mais o juiz do DeepEval.
        os.environ.update(OLLAMA_BASE_URL=judge_url, RAGAS_JUDGE_MODEL=self.modelos["juiz"],
                          DEEPEVAL_JUDGE_MODEL=self.modelos["juiz"], GENERATION_MODEL=gerador,
                          EMBEDDING_MODEL=self.modelos["embeddings"], ACTIVE_PROVIDER="local", RAG_INDEX="")
        # As respostas estão gravadas: o gerador sai da memória antes dos juízes (o OOM de 13/10/2026).
        if ra.unload_model(judge_url, gerador):
            print(f"\n    {gerador} descarregado do Ollama: daqui em diante só os juízes e os embeddings usam a "
                  "memória.", flush=True)
        scripts = {
            "ragas": (ROOT / "tests" / "ragas" / "run_and_export.py", ra.QUALITY_RESULTS["ragas"], []),
            "deepeval": (ROOT / "tests" / "deepeval" / "run_and_export.py", ra.QUALITY_RESULTS["deepeval"],
                         ["--no-open"]),
            "jeveval": (ROOT / "tests" / "jeveval" / "run_and_export.py", ra.QUALITY_RESULTS["jeveval"], []),
        }
        for fw in self.args.avaliadores:
            nome_fw = NOME_AVALIADOR[fw]
            if fw == "jeveval" and not ra.typesafe_key_configured():
                for index in self.answer_files:
                    self.pular(f"jeveval_{index}", f"JevEval: RAG {index}",
                               "sem TYPESAFE_API_KEY no ambiente ou no .env (ver tests/jeveval/README.md)",
                               cartoes[(fw, index)])
                continue
            if fw == "ragas":
                self.pytest("ragas_controles", "RAGAS: controles do juiz", "ragas", "tests/ragas/test_judge_controls.py")
            if fw == "jeveval":
                # O único avaliador fora da máquina: as perguntas, as respostas e os trechos vão à API da TypeSafe.
                self.pytest("jeveval_controles", "JevEval: controles do instrumento", "jeveval",
                            "tests/jeveval/test_jev_controls.py", extra=["-m", "live"])
            script, results, extra = scripts[fw]
            for index, path in self.answer_files.items():
                key, nome = f"{fw}_{index}", f"{nome_fw}: RAG {index}"
                cartao = cartoes[(fw, index)]
                self.palco.inicio_passo(key, nome, f"{script.relative_to(ROOT).as_posix()} --answers {path.name}",
                                        self.out_dir / "logs" / f"{key}.log", cartao)
                self.runner.answers_file = path
                self.runner.evaluation(key, nome, fw, script, results, extra)
                self.palco.fim_passo(self.runner.steps[-1], cartao)
        self.palco.concluir_fase(4)

    # 5 ------------------------------------------------------------------------
    def relatorio(self) -> Path | None:
        self.palco.fase(5)
        self.palco.visual("terminal")
        nome = "Relatório consolidado"
        log = self.out_dir / "logs" / "relatorio.log"
        diagnostic = de.retrieval_diagnostic(self.answer_sets, self.goldens) if self.answer_sets else None
        execution = {
            "tool": "scripts/apresentacao.bat",
            "started_at": self.started.isoformat(),
            "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "duration_seconds": round(time.monotonic() - self.t0, 1),
            "options": {"goldens_por_acervo": self.args.goldens, "avaliadores": self.args.avaliadores,
                        "juiz": self.modelos["juiz"], "video": self.palco.estado["video"]},
            "shared_answers": None,
            "reused_results": [],
            "note": ("As respostas avaliadas são as que o chat deu nos testes E2E ao vivo, com os dois RAGs no "
                     "Docker; os frameworks de qualidade julgaram as mesmas respostas. A execução foi gravada em "
                     "vídeo (scripts/apresentacao.bat)."),
            "retrieval_diagnostic": diagnostic,
            "steps": [vars(s) for s in self.runner.steps],
        }
        exec_path = self.out_dir / "execucao.json"
        exec_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if not self.runner.new_results:
            self.pular("relatorio_pdf", nome, "nenhuma avaliação concluída")
            self.palco.concluir_fase(5)
            return None
        if not ra.PY["reports"].exists():
            self.pular("relatorio_pdf", nome, f"venv do relatório não encontrado ({ra.SETUP_HINT['reports']})")
            self.palco.concluir_fase(5)
            return None
        cmd = [ra.PY["reports"], ROOT / "tests" / "reports" / "build_report.py", "--out", self.out_dir,
               "--execution", exec_path]
        for result in self.runner.new_results.values():
            cmd += ["--input", result]
        self.palco.inicio_passo("relatorio_pdf", nome, "tests/reports/build_report.py", log)
        code, secs = ra.run_logged(cmd, log, timeout=900, echo=True)
        pdf = self.out_dir / "relatorio.pdf"
        ok = code == 0 and pdf.is_file()
        step = self.runner.record(ra.Step("relatorio_pdf", nome, "concluída" if ok else "erro",
                                          pdf.name if ok else f"não foi gerado (código {code}; ver o log)", None,
                                          secs, self.runner._rel(log)))
        self.palco.fim_passo(step)
        if ok:
            figuras = sorted((self.out_dir / "figuras").glob("*.png"))
            # As notas e as médias primeiro; o histórico, que junta rodadas antigas, por último.
            figuras.sort(key=lambda p: (p.stem == "historico", p.stem))
            self.palco.visual("relatorio", figuras=[f"figuras/{p.name}" for p in figuras],
                              pdf=_relativo(pdf), segundos_por_figura=self.args.segundos_figura)
            time.sleep(self.args.segundos_figura * max(1, len(figuras)) + 2)
        self.palco.concluir_fase(5)
        return pdf if ok else None

    # tudo ---------------------------------------------------------------------
    def resumo(self, pdf: Path | None) -> dict:
        passos = self.runner.steps
        contagem = {s: sum(1 for p in passos if p.status in grupo) for s, grupo in
                    (("ok", ("passou", "concluída")), ("falhas", ("falhou", "erro")), ("pulados", ("pulada",)))}
        return {
            **contagem,
            "duracao": _tempo(time.monotonic() - self.t0),
            "perguntas": len(self.goldens),
            "respostas_julgadas": sum(len(d["answers"]) for d in self.answer_sets.values()),
            "avaliacoes": len(self.runner.new_results),
            "relatorio": _relativo(pdf) if pdf else None,
            "video": self.palco.estado["video"],
        }


# ------------------------------------------------------------------ main


def _abrir_palco(url: str, ffmpeg: str | None, mkv: Path | None, fps: int) -> subprocess.Popen | None:
    """O Chromium sem janela que mostra o palco e, com ffmpeg e destino, o grava (frontend/tests/apresentacao/palco.mjs)."""
    node = shutil.which("node")
    if not node or not PALCO_SCRIPT.is_file():
        return None
    cmd = [node, str(PALCO_SCRIPT), url]
    if ffmpeg and mkv is not None:
        mkv.parent.mkdir(parents=True, exist_ok=True)
        cmd += [ffmpeg, str(mkv), str(fps)]
    log = (mkv.parent / "palco.log").open("w", encoding="utf-8", errors="replace") if mkv else subprocess.DEVNULL
    return subprocess.Popen(cmd, cwd=PALCO_SCRIPT.parent, stdout=log, stderr=subprocess.STDOUT)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A rotina completa num palco, gravada em MP4.")
    parser.add_argument("--goldens", type=int, default=de._per_set(os.environ.get("DEMO_GOLDENS")) or 1,
                        help="perguntas de referência por acervo, em cada RAG (padrão: 1, ou DEMO_GOLDENS)")
    parser.add_argument("--avaliadores", nargs="+", choices=AVALIADORES, default=list(AVALIADORES),
                        help="quais frameworks julgam (padrão: os três; o JevEval só roda com TYPESAFE_API_KEY)")
    parser.add_argument("--sem-video", action="store_true", help="não grava (ensaio)")
    parser.add_argument("--assistir", action="store_true",
                        help="abre o palco no navegador padrão, para acompanhar ao vivo (não muda o vídeo)")
    parser.add_argument("--fps", type=int, default=15, help="quadros por segundo do vídeo (padrão: 15)")
    parser.add_argument("--pausa-app", type=float, default=15, help="segundos mostrando a aplicação no ar")
    parser.add_argument("--segundos-figura", type=float, default=7, help="segundos de cada figura do relatório")
    parser.add_argument("--pausa-final", type=float, default=12, help="segundos do quadro final antes de parar")
    parser.add_argument("--abrir", action="store_true", help="abre o vídeo ao terminar")
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = construir_parser().parse_args()
    args.avaliadores = [a for a in AVALIADORES if a in args.avaliadores]  # a ordem do vídeo, não a digitada
    gravando = not args.sem_video
    ffmpeg = shutil.which("ffmpeg")
    if gravando and not ffmpeg:
        print("[ERRO] ffmpeg não encontrado: é ele que monta o vídeo. No Windows: winget install Gyan.FFmpeg (e abra "
              "um terminal novo); no Linux: o pacote ffmpeg. Para ensaiar sem gravar: --sem-video.", file=sys.stderr)
        return 2

    ap = Apresentacao(args)
    servidor = criar_servidor(ap.palco)
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    ap.espelho_url = f"{url}espelho"
    mkv = ap.video.with_suffix(".mkv") if ap.video else None
    print(f"=== Apresentação gravada — {ap.started.strftime('%d/%m/%Y %H:%M')} ===")
    print(f"Saída: {ap.out_dir}")
    print(f"Palco: {url}  (só a página do palco é gravada: a tela do computador fica de fora)")
    palco = _abrir_palco(url, ffmpeg if gravando else None, mkv, args.fps)
    if palco is None:
        print("[ERRO] Sem o Node ou sem frontend/tests/apresentacao/palco.mjs: o palco não abre (README, "
              "\"Apresentação gravada\").", file=sys.stderr)
        servidor.shutdown()
        return 2
    if not ap.palco.visto.wait(timeout=90):
        print("[AVISO] O palco não respondeu em 90 s (ver video/palco.log); a rotina segue.", flush=True)
    if gravando:
        print(f"Gravando o palco em {mkv} (vira MP4 no fim).", flush=True)
    if args.assistir:
        import webbrowser

        webbrowser.open(url)
    pdf = None
    try:
        time.sleep(5)  # a abertura: o título e as fases
        if ap.aplicacao():
            ap.testes_locais()
            ap.e2e()
            ap.avaliadores()
            pdf = ap.relatorio()
        else:
            print("    [ERRO ] a aplicação não subiu; a rotina para aqui (ver os logs da subida).", flush=True)
    except KeyboardInterrupt:
        print("\nInterrompido: fechando o vídeo com o que foi gravado.", flush=True)
    finally:
        resumo = ap.resumo(pdf)
        ap.palco.atualizar(fim=True)
        ap.palco.visual("fim", resumo=resumo)
        time.sleep(args.pausa_final)
        ap.palco.atualizar(fechar=True)  # o palco.mjs para de gravar, fecha o MKV e sai
        try:
            palco.wait(timeout=180)
        except subprocess.TimeoutExpired:
            palco.terminate()
        servidor.shutdown()
        video = converter_para_mp4(ffmpeg, mkv, ap.video) if gravando else None

    print(f"\n=== Resumo ({(time.monotonic() - ap.t0) / 60:.1f} min) ===")
    for step in ap.runner.steps:
        print(f"  {ra.STATUS_TAG.get(step.status, step.status)} {step.name}")
    if pdf:
        print(f"\nRelatório: {pdf}")
    if video:
        print(f"Vídeo: {video}")
        if args.abrir and hasattr(os, "startfile"):
            os.startfile(video)  # noqa: S606 - abre no player padrão do Windows
    elif gravando:
        print("[ERRO ] o vídeo não foi gerado (ver video/palco.log)")
    broken = [s for s in ap.runner.steps if s.status in ("falhou", "erro")]
    return 1 if broken or pdf is None or (gravando and video is None) else 0


if __name__ == "__main__":
    sys.exit(main())
