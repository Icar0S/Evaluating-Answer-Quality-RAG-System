"""Gera UMA vez as respostas do assistente para os goldens, para todos os frameworks julgarem as mesmas.

Sem isto, cada suíte de qualidade (tests/ragas, tests/deepeval) chama o
pipeline por conta própria. Como a geração não é determinística — e a
recuperação muda quando o índice é reingerido —, os frameworks acabam julgando
respostas e trechos DIFERENTES para a mesma pergunta, e a comparação entre eles
mistura método com sistema (o relatório consolidado mostrou 0 de 10 respostas
iguais entre duas rodadas). Com o arquivo daqui, os dois run_and_export.py
recebem `--answers` e julgam exatamente o mesmo texto.

O arquivo é autocontido: cada resposta leva a pergunta, a referência e o
contexto do golden, e a origem dele (`--goldens tag=caminho`). É isso que deixa
avaliar conjuntos de goldens diferentes (o dos PDFs, o dos Markdown) e
comparar versões do índice (`--index`, ver backend/app/config.py RAG_INDICES).

Para o benchmark (plano-benchmark.md), cada resposta grava também o custo (C4:
latência da recuperação e da geração, tokens do prompt e da resposta), se o
prompt passou do limite do Ollama (truncagem), os documentos e trechos que
chegaram e se a resposta é uma recusa (C2). No topo vai o braço: a
configuração efetiva da rodada (tests/benchmark/arm.py), com as sobrescritas de
`--set` e a semente de `--shuffle-seed`, que embaralha a ordem das perguntas
(o cache de prefixo do Ollama faz a resposta depender da pergunta anterior).

Usa o mesmo pipeline do POST /chat (retrieval + generation, in-process), então
roda no venv do backend:

    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out respostas.json
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --index md-v1 \\
        --goldens pdf=tests/deepeval/goldens/dataset.json --goldens md=tests/deepeval/goldens/dataset_md.json
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --lang en   # usa input_en dos goldens
    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --index pdf-v1 --local \\
        --goldens v2=tests/benchmark/goldens/v2.json --shuffle-seed 1 \\
        --set generation_temperature=0 --set generation_seed=42 --set generation_think=false

Com `--api URL`, as respostas vêm do RAG que está no ar, pela API (POST
/chat), e não do pipeline in-process: o script não importa o backend nem abre o
Chroma, e roda com qualquer Python 3.10+. É o modo do container de testes
(tests/run_all.py --api), em que o índice fica num volume do backend e só a API
o enxerga. O formato do arquivo é o mesmo; a configuração (índice, modelos,
top_k) é a do backend no ar, por isso --index, --set e --local não valem aqui:

    python tests/generate_answers.py --out r.json --api http://localhost:8010

O protocolo da pergunta única (plano-escopo-final.md, §4): uma pergunta, N
respostas com sementes fixas, cada uma com o servidor limpo e todas com os
mesmos trechos. Vale nos dois modos:

    backend\\.venv\\Scripts\\python.exe tests\\generate_answers.py --out r.json --index md-v1 --local \\
        --goldens md=tests/deepeval/goldens/dataset_md.json --only md-mtbench-vieses \\
        --sementes 1-5 --estado-limpo --set generation_temperature=1
    python tests/generate_answers.py --out r.json --api http://localhost:8010 --ollama http://127.0.0.1:11435 \\
        --goldens md=tests/deepeval/goldens/dataset_md.json --only md-mtbench-vieses --sementes 1-5 --estado-limpo

- `--only NOME` escolhe o golden (repetível); `--sementes 1-5` (ou `1,3,7`) dá uma resposta por semente, com o
  nome `NOME#1` … `NOME#5`; `--repeticoes N` sem sementes repete com a semente do servidor (respostas que não
  se reproduzem).
- `--estado-limpo` descarrega o gerador do Ollama (`keep_alive: 0`) antes de cada resposta: nenhuma herda o
  cache de prefixo da anterior. É o "chat limpo" que importa aqui, porque o histórico da tela nunca chega ao
  modelo (o POST /chat recebe só a pergunta). Pela API, `--ollama` diz onde está o Ollama do backend (no
  Docker, a 11435 da máquina).
- Os trechos de cada resposta viram um hash (`sources_sha256`). Se uma resposta da mesma pergunta receber
  outros trechos, o script para: a diferença entre as respostas tem de vir só do gerador.
- `--controle-estado` é o controle quente × frio: por semente, a resposta com o modelo recarregado (frio),
  de novo frio (se reproduz?) e logo depois de outra pergunta (quente). O topo do arquivo diz, por semente,
  se o estado do servidor mudou a resposta.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

# Mesmo motivo do run_and_export.py do DeepEval: com saída redirecionada para
# arquivo, o Windows usa cp1252 e um acento derruba o processo.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_GOLDENS = ROOT / "tests" / "deepeval" / "goldens" / "dataset.json"
# Origem padrão de cada arquivo de goldens conhecido (o acervo de onde as
# respostas de referência vieram).
KNOWN_ORIGINS = {"dataset.json": "pdf", "dataset_md.json": "md", "v2.json": "v2"}
# Acima do GENERATION_TIMEOUT_SECONDS do backend (300 s no docker-compose.yml):
# quando a geração estoura, quem responde é o backend, com 504 e o motivo, e não
# um timeout do cliente sem explicação.
API_TIMEOUT_SECONDS = 360
# Quanto esperar o gerador sair da memória do Ollama depois do keep_alive 0.
UNLOAD_WAIT_SECONDS = 30.0
# Código de saída quando o protocolo é violado e para: uma resposta da mesma pergunta recebeu outros
# trechos, ou o backend não aplicou a semente pedida (um backend sem o campo `seed` o ignora em silêncio).
EXIT_PROTOCOL_BROKEN = 3


def parse_goldens_arg(value: str) -> tuple[str, Path]:
    """`tag=caminho` ou só `caminho` (a tag vem de KNOWN_ORIGINS ou do nome do arquivo)."""
    if "=" in value and not Path(value).exists():
        tag, path = value.split("=", 1)
        return tag, Path(path)
    path = Path(value)
    return KNOWN_ORIGINS.get(path.name, path.stem), path


def processing_order(n: int, seed: int | None) -> list[int]:
    order = list(range(n))
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera as respostas do assistente para os goldens, uma vez só.")
    parser.add_argument("--out", type=Path, required=True, help="arquivo JSON de saída")
    parser.add_argument("--limit", type=int, default=None, help="só os N primeiros goldens de cada conjunto")
    parser.add_argument("--index", default=None, help="versão do índice (RAG_INDICES); padrão: a do .env")
    parser.add_argument(
        "--goldens", action="append", default=[],
        help="conjunto de goldens, `tag=caminho` (repetível); padrão: pdf=tests/deepeval/goldens/dataset.json",
    )
    parser.add_argument("--lang", choices=("pt", "en"), default="pt", help="en = usa o input_en dos goldens")
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescrita de configuração do braço (campos de Settings, ex.: generation_seed=42)")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="embaralha a ordem das perguntas")
    parser.add_argument("--local", action="store_true",
                        help="força o provider local (o remoto ignora temperatura, semente, num_ctx e think)")
    parser.add_argument("--api", default=None, metavar="URL",
                        help="gera pela API do RAG no ar (POST URL/chat), sem importar o backend")
    protocol = parser.add_argument_group("protocolo da pergunta única (plano-escopo-final.md, §4)")
    protocol.add_argument("--only", action="append", default=[], metavar="NOME",
                          help="só este golden (repetível), pelo `name`")
    protocol.add_argument("--repeticoes", type=int, default=None, metavar="N",
                          help="respostas por golden (padrão: uma por semente, ou 1)")
    protocol.add_argument("--sementes", default=None, metavar="1-5",
                          help="uma resposta por semente (`1-5` ou `1,3,7`), cada uma com a semente da requisição")
    protocol.add_argument("--estado-limpo", action="store_true",
                          help="descarrega o gerador do Ollama antes de cada resposta (sem cache de prefixo)")
    protocol.add_argument("--controle-estado", action="store_true",
                          help="controle quente x frio: por semente, frio, frio de novo e logo depois de outra pergunta")
    protocol.add_argument("--aquecimento", default=None, metavar="NOME",
                          help="a outra pergunta do controle (padrão: o primeiro golden que não é o alvo)")
    protocol.add_argument("--ollama", default=None, metavar="URL",
                          help="o Ollama do gerador, para descarregar o modelo; obrigatório com --api (Docker: "
                               "http://127.0.0.1:11435); padrão in-process: o do provider local")
    return parser


def load_goldens(golden_sets: list[tuple[str, Path]], limit: int | None) -> list[tuple[str, dict]]:
    goldens = []
    for origin, path in golden_sets:
        items = json.loads(path.read_text(encoding="utf-8"))[:limit]
        goldens += [(origin, g) for g in items]
    return goldens


def golden_name(golden: dict) -> str:
    return golden.get("name") or golden["input"][:40]


def golden_record(origin: str, golden: dict, question: str) -> dict:
    """O começo de cada resposta: o que vem do golden (pergunta, referência, contexto, documento-fonte)."""
    return {
        "name": golden_name(golden),
        "origin": origin,
        "input": question,
        "expected_output": golden.get("expected_output", ""),
        "context": golden.get("context") or [],
        "source_document": golden.get("source_document"),
    }


def question_for(golden: dict, lang: str) -> str:
    return golden.get("input_en") if lang == "en" and golden.get("input_en") else golden["input"]


# ------------------------------------------------- modo --api (biblioteca padrão)


class ApiError(Exception):
    """A API respondeu com erro (o detalhe vem do backend) ou não respondeu."""


def api_call(url: str, payload: dict | None = None, timeout: float = API_TIMEOUT_SECONDS) -> dict:
    """GET (sem payload) ou POST JSON; devolve o JSON. Erro HTTP vira ApiError com o `detail` do FastAPI."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - a API do RAG sob teste
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except (json.JSONDecodeError, AttributeError):
            detail = body
        raise ApiError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ApiError(f"{type(exc).__name__}: {exc}") from exc


def _round1(value: float | None) -> float | None:
    """Uma casa, como o modo in-process grava as latências."""
    return round(value, 1) if value is not None else None


def api_answer_record(api: str, origin: str, golden: dict, question: str,
                      seed: int | None = None) -> tuple[dict, dict | None]:
    """Pergunta pela API, como o chat faria. Devolve a resposta no formato do modo in-process e o metadata do /chat.

    Mesma política do modo in-process: duas tentativas, e erro vira registro (quem julga marca como não medido).
    Cada chamada é um POST novo, só com a pergunta (e a semente, no protocolo): nenhum histórico vai junto.
    """
    record = golden_record(origin, golden, question)
    metadata = None
    payload: dict = {"question": question}
    if seed is not None:
        payload["seed"] = seed
    t0 = time.perf_counter()
    for attempt in (1, 2):
        try:
            reply = api_call(f"{api}/chat", payload)
            sources = reply.get("sources") or []
            metadata = reply.get("metadata") or {}
            record.update(
                actual_output=reply["answer"],
                retrieval_context=[s["text"] for s in sources],
                sources=[
                    s["document"] if s.get("page") is None else f"{s['document']} (pág. {s['page']})"
                    for s in sources
                ],
                source_documents=[s["document"] for s in sources],
                chunk_ids=[s.get("chunk_id") for s in sources],
                similarity_scores=[s.get("similarity_score") for s in sources],
                retrieval_ms=_round1(metadata.get("retrieval_latency_ms")),
                generation_ms=_round1(metadata.get("generation_latency_ms")),
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
                temperature=metadata.get("temperature"),
                seed=metadata.get("seed"),
                error=None,
            )
            break
        except (ApiError, KeyError, TypeError) as exc:
            record.update(actual_output=None, retrieval_context=[], sources=[], source_documents=[],
                          error=f"{type(exc).__name__}: {exc}")
            print(f"  [{record['name']}] tentativa {attempt}/2 falhou: {exc}", flush=True)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    return record, metadata


# ------------------------------------ protocolo da pergunta única (biblioteca padrão)


def parse_seeds(spec: str) -> list[int]:
    """`1-5` -> [1, 2, 3, 4, 5]; `1,3,7` -> [1, 3, 7]; `42` -> [42]. Vazia, negativa ou repetida é erro."""
    seeds: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if end < start:
                raise ValueError(f"faixa de sementes invertida: {part!r}")
            seeds += range(start, end + 1)
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"nenhuma semente em {spec!r}")
    if any(s < 0 for s in seeds):
        raise ValueError(f"semente negativa em {spec!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"semente repetida em {spec!r}: cada resposta tem de ter a sua")
    return seeds


def sources_sha256(chunk_ids: list, texts: list[str]) -> str:
    """Impressão digital dos trechos que chegaram ao gerador: o id e o texto de cada um, na ordem."""
    payload = json.dumps([[cid, text] for cid, text in zip(chunk_ids, texts)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_tag(model: str) -> str:
    """O nome como o /api/ps mostra: sem tag, o Ollama usa `:latest`."""
    return model if ":" in model else f"{model}:latest"


def ollama_loaded_models(ollama_url: str) -> list[str]:
    models = api_call(f"{ollama_url}/api/ps", timeout=30).get("models") or []
    return [m.get("name") or m.get("model") for m in models]


def unload_model(ollama_url: str, model: str, wait_seconds: float = UNLOAD_WAIT_SECONDS) -> float:
    """Tira o gerador da memória do Ollama (`keep_alive: 0`) e espera ele sumir do /api/ps.

    É o "servidor limpo" do protocolo: com o modelo recarregado, nenhuma resposta herda o cache de prefixo
    da anterior. Devolve os segundos que levou (0 se já estava fora). ApiError se o Ollama não respondeu ou
    se o modelo não saiu a tempo: sem estado limpo garantido, o protocolo não segue.
    """
    tag = _model_tag(model)
    t0 = time.perf_counter()
    if tag not in ollama_loaded_models(ollama_url):
        return 0.0
    api_call(f"{ollama_url}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    while tag in ollama_loaded_models(ollama_url):
        if time.perf_counter() - t0 > wait_seconds:
            raise ApiError(f"o modelo {model} continua carregado em {ollama_url} depois de {wait_seconds:.0f}s")
        time.sleep(0.5)
    return round(time.perf_counter() - t0, 2)


def _parse_parameter(value: str) -> int | float | str:
    value = value.strip().strip('"')
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def model_parameters(ollama_url: str, model: str) -> dict:
    """Os parâmetros padrão do modelo no Ollama (/api/show): {"temperature": 1, "top_k": 64, ...}."""
    text = api_call(f"{ollama_url}/api/show", {"model": model}, timeout=60).get("parameters") or ""
    params: dict = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition(" ")
        if not key:
            continue
        parsed = _parse_parameter(value)
        if key in params:  # "stop" aparece uma vez por sequência
            previous = params[key] if isinstance(params[key], list) else [params[key]]
            params[key] = [*previous, parsed]
        else:
            params[key] = parsed
    return params


class Ask(NamedTuple):
    """Uma pergunta do plano da rodada: qual golden, com que nome sai no arquivo, que semente e que estado.

    NamedTuple, e não dataclass: os testes e os outros scripts carregam este arquivo pelo caminho
    (importlib), sem registrá-lo em sys.modules, e o @dataclass precisa do módulo lá.
    """

    index: int
    name: str
    seed: int | None = None
    unload: bool = False
    estado: str | None = None
    repeticao: int | None = None
    # O papel no controle quente x frio ("frio", "frio-2", "aquecimento", "quente").
    papel: str | None = None

    @property
    def target(self) -> bool:
        """Entra na conferência de trechos: o aquecimento é outra pergunta, com outros trechos."""
        return self.papel != "aquecimento"


def protocol_active(args: argparse.Namespace) -> bool:
    return bool(args.only or args.repeticoes is not None or args.sementes or args.estado_limpo
                or args.controle_estado)


def _select(goldens: list[tuple[str, dict]], only: list[str]) -> list[int]:
    names = [golden_name(g) for _, g in goldens]
    if not only:
        return list(range(len(goldens)))
    missing = [n for n in only if n not in names]
    if missing:
        raise ValueError(f"golden(s) não encontrado(s): {', '.join(missing)}")
    return [names.index(n) for n in only]


def _control_asks(goldens: list[tuple[str, dict]], targets: list[int], seeds: list[int],
                  warmup: str | None) -> list[Ask]:
    if len(targets) != 1:
        raise ValueError("--controle-estado pede uma pergunta só (--only NOME)")
    names = [golden_name(g) for _, g in goldens]
    target = targets[0]
    if warmup is not None:
        if warmup not in names:
            raise ValueError(f"golden de aquecimento não encontrado: {warmup}")
        warm = names.index(warmup)
    else:
        warm = next((i for i in range(len(goldens)) if i != target), None)
    if warm is None or warm == target:
        raise ValueError("o aquecimento precisa de outro golden, diferente do alvo")
    base, wbase = names[target], names[warm]
    asks: list[Ask] = []
    for s in seeds:
        asks += [
            Ask(target, f"{base}#s{s}-frio", s, True, "frio", 1, "frio"),
            Ask(target, f"{base}#s{s}-frio-2", s, True, "frio", 2, "frio-2"),
            # Recarrega e faz a outra pergunta: o estado da pergunta seguinte é exatamente "depois de outra".
            Ask(warm, f"{wbase}#s{s}-aquecimento", s, True, "frio", None, "aquecimento"),
            Ask(target, f"{base}#s{s}-quente", s, False, "quente", 3, "quente"),
        ]
    return asks


def plan_asks(goldens: list[tuple[str, dict]], args: argparse.Namespace) -> list[Ask]:
    """O que perguntar, em que ordem. Sem as opções do protocolo, é uma pergunta por golden, como sempre foi."""
    targets = _select(goldens, args.only)
    seeds = parse_seeds(args.sementes) if args.sementes else None
    reps = args.repeticoes if args.repeticoes is not None else (len(seeds) if seeds else 1)
    if reps < 1:
        raise ValueError("--repeticoes precisa ser 1 ou mais")
    if seeds and len(seeds) != reps:
        raise ValueError(f"--repeticoes {reps} com {len(seeds)} semente(s): cada resposta tem de ter a sua")
    if args.controle_estado:
        return _control_asks(goldens, targets, seeds or [1], args.aquecimento)
    asks = []
    for i in processing_order(len(targets), args.shuffle_seed):
        index = targets[i]
        base = golden_name(goldens[index][1])
        for k in range(1, reps + 1):
            asks.append(Ask(
                index=index,
                name=f"{base}#{k}" if reps > 1 else base,
                seed=seeds[k - 1] if seeds else None,
                unload=args.estado_limpo,
                estado="frio" if args.estado_limpo else None,
                repeticao=k if reps > 1 else None,
            ))
    return asks


def control_summary(answers: list[dict]) -> dict:
    """Por semente: o frio se reproduz? o quente sai igual ao frio? (com a semelhança dos textos)."""
    by_seed: dict = {}
    for a in answers:
        if a.get("papel"):
            by_seed.setdefault(a.get("semente_pedida"), {})[a["papel"]] = a.get("actual_output")
    rows = []
    for seed, texts in by_seed.items():
        frio, frio2, quente = texts.get("frio"), texts.get("frio-2"), texts.get("quente")
        complete = None not in (frio, frio2, quente)
        rows.append({
            "seed": seed,
            "frio_se_reproduz": (frio == frio2) if complete else None,
            "quente_igual_ao_frio": (quente == frio) if complete else None,
            "semelhanca_quente_frio": (round(difflib.SequenceMatcher(None, frio, quente).ratio(), 3)
                                       if complete else None),
            "caracteres": {"frio": len(frio or ""), "frio-2": len(frio2 or ""), "quente": len(quente or "")},
        })
    measured = [r for r in rows if r["frio_se_reproduz"] is not None]
    return {
        "por_semente": rows,
        "frio_se_reproduz": f"{sum(r['frio_se_reproduz'] for r in measured)}/{len(measured)}",
        "estado_mudou_a_resposta": f"{sum(not r['quente_igual_ao_frio'] for r in measured)}/{len(measured)}",
    }


def protocol_header(args: argparse.Namespace, model: str | None, ollama_url: str | None) -> dict:
    seeds = parse_seeds(args.sementes) if args.sementes else ([1] if args.controle_estado else None)
    return {
        "plano": "plano-escopo-final.md, §4",
        "pergunta_isolada": "cada resposta é uma chamada nova, só com a pergunta: nenhum histórico vai ao modelo",
        "repeticoes": args.repeticoes or (len(seeds) if seeds and not args.controle_estado else 1),
        "sementes": seeds,
        "estado_limpo": args.estado_limpo or args.controle_estado,
        "controle_estado": args.controle_estado,
        "ollama_url": ollama_url,
        "modelo": model,
        "temperatura": None,
        "temperatura_fonte": None,
        "parametros_do_modelo": None,
        "trechos": {},
        "interrompido": None,
    }


def fill_temperature(protocol: dict, explicit: float | None, ollama_url: str | None, model: str | None) -> None:
    """A temperatura que valeu: a configurada, ou a padrão do modelo (lida do /api/show)."""
    if ollama_url and model:
        try:
            protocol["parametros_do_modelo"] = model_parameters(ollama_url, model)
        except ApiError as exc:
            print(f"  [aviso] parâmetros do modelo não lidos em {ollama_url}: {exc}", flush=True)
    if explicit is not None:
        protocol.update(temperatura=float(explicit), temperatura_fonte="configurada")
    elif protocol["parametros_do_modelo"] and "temperature" in protocol["parametros_do_modelo"]:
        protocol.update(temperatura=float(protocol["parametros_do_modelo"]["temperature"]),
                        temperatura_fonte="padrão do modelo (/api/show)")
    else:
        protocol.update(temperatura=None, temperatura_fonte="desconhecida (padrão do modelo, não lido)")


AskFn = Callable[[str, dict, str, "int | None"], dict]


def run_asks(asks: list[Ask], goldens: list[tuple[str, dict]], args: argparse.Namespace, output: dict,
             ask_fn: AskFn, unload_fn: Callable[[], float] | None) -> int:
    """Pergunta cada item do plano, grava o arquivo a cada resposta e confere os trechos. Devolve o código."""
    protocol = output.get("protocolo")
    first_hash: dict[int, tuple[str, str]] = {}
    failures = 0
    for position, ask in enumerate(asks, start=1):
        origin, golden = goldens[ask.index]
        unload_seconds = None
        if ask.unload and unload_fn is not None:
            try:
                unload_seconds = unload_fn()
            except ApiError as exc:
                protocol["interrompido"] = f"não deu para descarregar o gerador antes de {ask.name}: {exc}"
                args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[ERRO] {protocol['interrompido']}", file=sys.stderr, flush=True)
                return 1
        record = ask_fn(origin, golden, question_for(golden, args.lang), ask.seed)
        record["name"] = ask.name
        record["position"] = position
        if protocol is not None:
            texts, ids = record.get("retrieval_context") or [], record.get("chunk_ids") or []
            record.update(
                golden=golden_name(golden),
                repeticao=ask.repeticao,
                estado=ask.estado,
                semente_pedida=ask.seed,
                unload_seconds=unload_seconds,
                sources_sha256=sources_sha256(ids, texts) if record.get("error") is None else None,
            )
            if ask.papel:
                record["papel"] = ask.papel
        failures += record["error"] is not None
        output["answers"].append(record)
        stop = None
        if protocol is None:
            pass
        elif ask.seed is not None and record["error"] is None and record.get("seed") != ask.seed:
            stop = (f"{ask.name}: a semente pedida era {ask.seed} e a resposta diz {record.get('seed')!r}. O backend "
                    "não aplicou a semente (uma versão sem o campo `seed` no POST /chat o ignora); as respostas "
                    "não se reproduziriam")
            protocol["interrompido"] = stop
        elif ask.target and record.get("sources_sha256"):
            base = golden_name(golden)
            first = first_hash.setdefault(ask.index, (record["sources_sha256"], ask.name))
            entry = protocol["trechos"].setdefault(base, {"sha256": first[0], "iguais": True, "respostas": 0})
            entry["respostas"] += 1
            if record["sources_sha256"] != first[0]:
                entry["iguais"] = False
                stop = (f"{ask.name} recebeu outros trechos ({record['sources_sha256']}) que {first[1]} "
                        f"({first[0]}): a diferença entre as respostas não viria só do gerador")
                protocol["interrompido"] = stop
        if args.controle_estado and protocol is not None:
            output["controle_estado"] = control_summary(output["answers"])
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "ERRO" if record["error"] else f"{len(record['actual_output'])} caracteres"
        flags = " [TRUNCADO]" if record.get("truncated") else ""
        extra = f", semente {ask.seed}" if ask.seed is not None else ""
        extra += f", {ask.papel or ask.estado}" if (ask.papel or ask.estado) else ""
        print(f"  [{position}/{len(asks)}] {ask.name}: {status}{flags} ({record['seconds']}s{extra})", flush=True)
        if stop:
            print(f"[ERRO] {stop}. O protocolo parou.", file=sys.stderr, flush=True)
            return EXIT_PROTOCOL_BROKEN
    return 0 if failures < len(asks) else 1


def _plan_or_exit(goldens: list[tuple[str, dict]], args: argparse.Namespace) -> tuple[list[Ask] | None, int]:
    try:
        return plan_asks(goldens, args), 0
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return None, 2


def api_main(args: argparse.Namespace) -> int:
    """As respostas pela API do RAG no ar. Não importa o backend: o RAG é testado como caixa-preta."""
    in_process_only = [name for name, value in (("--index", args.index), ("--set", args.set), ("--local", args.local))
                       if value]
    if in_process_only:
        print(f"[ERRO] {', '.join(in_process_only)} não vale(m) com --api: índice, modelos e configuração são os "
              "do backend que está no ar.", file=sys.stderr)
        return 2
    if (args.estado_limpo or args.controle_estado) and not args.ollama:
        print("[ERRO] --estado-limpo e --controle-estado pela API precisam de --ollama URL, o Ollama do backend "
              "(no Docker, http://127.0.0.1:11435): é lá que o gerador é descarregado.", file=sys.stderr)
        return 2
    api = args.api.rstrip("/")
    ollama_url = args.ollama.rstrip("/") if args.ollama else None
    try:
        health = api_call(f"{api}/health", timeout=30)
    except ApiError as exc:
        print(f"[ERRO] A API do RAG não respondeu em {api} ({exc}).", file=sys.stderr)
        return 1
    if health.get("vector_store_documents") == 0:
        print(f"[ERRO] Índice {health.get('rag_index')} vazio em {api}: o backend não tem documentos indexados.",
              file=sys.stderr)
        return 1

    golden_sets = [parse_goldens_arg(g) for g in args.goldens] or [("pdf", DEFAULT_GOLDENS)]
    goldens = load_goldens(golden_sets, args.limit)
    asks, code = _plan_or_exit(goldens, args)
    if asks is None:
        return code
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collected_by": f"POST {api}/chat (tests/generate_answers.py --api): o RAG no ar, como caixa-preta",
        "api_url": api,
        # Do /health por enquanto; a primeira resposta troca pelo metadata do /chat, que diz o que respondeu.
        "provider": None,
        "generation_model": health.get("generation_model"),
        "rag_index": health.get("rag_index"),
        "embedding_model": health.get("embedding_model"),
        "lang": args.lang,
        "top_k": None,
        "prompt_version": None,
        "goldens": [f"{origin}={path.as_posix()}" for origin, path in golden_sets],
        "goldens_sha256": {f"{origin}={path.as_posix()}": _sha256(path) for origin, path in golden_sets},
        "shuffle_seed": args.shuffle_seed,
        "n_cases": len(asks),
        "answers": [],
    }
    model = health.get("generation_model")
    if protocol_active(args):
        output["protocolo"] = protocol_header(args, model, ollama_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Gerando {len(asks)} respostas pela API em {api} (índice {health.get('rag_index')}, "
        f"gerador {model}), perguntas em {args.lang}"
        + (f", ordem embaralhada (semente {args.shuffle_seed})." if args.shuffle_seed is not None else "."),
        flush=True,
    )
    first_metadata: dict = {}

    def ask_fn(origin: str, golden: dict, question: str, seed: int | None) -> dict:
        record, metadata = api_answer_record(api, origin, golden, question, seed)
        if metadata:
            output.update(
                provider=metadata.get("provider"),
                generation_model=metadata.get("model") or output["generation_model"],
                embedding_model=metadata.get("embedding_model") or output["embedding_model"],
                top_k=metadata.get("top_k"),
                prompt_version=metadata.get("prompt_version"),
                rag_index=metadata.get("rag_index") or output["rag_index"],
            )
            if not first_metadata:
                first_metadata.update(metadata)
                if "protocolo" in output:
                    fill_temperature(output["protocolo"], metadata.get("temperature"), ollama_url,
                                     metadata.get("model") or model)
        return record

    unload_fn = (lambda: unload_model(ollama_url, output["generation_model"] or model)) if ollama_url else None
    code = run_asks(asks, goldens, args, output, ask_fn, unload_fn)
    if "protocolo" in output and output["protocolo"]["temperatura_fonte"] is None:
        # Nenhuma resposta deu certo: a temperatura sai do modelo, se o Ollama for conhecido.
        fill_temperature(output["protocolo"], None, ollama_url, output["generation_model"] or model)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(1 for a in output["answers"] if a["error"] is not None)
    print(f"Respostas em {args.out} ({failures} falha(s)).", flush=True)
    return code


def _wants_api(argv: list[str]) -> bool:
    return any(arg == "--api" or arg.startswith("--api=") for arg in argv)


# Rodando como script com --api, sai daqui, antes de importar o backend: é o que
# deixa o modo API rodar sem o venv do backend e sem abrir o Chroma. Importado
# como módulo (os testes), segue para o modo in-process.
if __name__ == "__main__" and _wants_api(sys.argv[1:]):
    sys.exit(api_main(build_parser().parse_args()))

from app import providers  # noqa: E402
from app.config import get_settings, set_settings_overrides  # noqa: E402
from app.rag import generation, retrieval, vector_store  # noqa: E402
from tests.benchmark import abstention, arm  # noqa: E402


def answer_record(origin: str, golden: dict, question: str, top_k: int, seed: int | None = None) -> dict:
    """Recupera, gera e mede uma resposta. Erro vira registro (quem julga marca como não medido)."""
    record = golden_record(origin, golden, question)
    t0 = time.perf_counter()
    # Uma tentativa extra, como no run_and_export.py do DeepEval: a maioria
    # dos timeouts do Ollama aqui é transitória.
    for attempt in (1, 2):
        try:
            chunks, retrieval_ms = retrieval.retrieve(question, top_k)
            result = generation.generate_answer(question, chunks, seed=seed)
            limit = generation.prompt_token_limit()
            prompt_tokens = result.get("prompt_tokens")
            record.update(
                actual_output=result["answer"],
                retrieval_context=[c["text"] for c in chunks],
                sources=[
                    c["document"] if c.get("page") is None else f"{c['document']} (pág. {c['page']})"
                    for c in chunks
                ],
                source_documents=[c["document"] for c in chunks],
                chunk_ids=[c.get("chunk_id") for c in chunks],
                similarity_scores=[c.get("similarity_score") for c in chunks],
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(result.get("latency_ms") or 0.0, 1),
                prompt_tokens=prompt_tokens,
                completion_tokens=result.get("completion_tokens"),
                temperature=result.get("temperature"),
                seed=result.get("seed"),
                # A regra do estudo de mutação (tests/mutation/runner/sut.py): no limite, o Ollama corta o
                # início do prompt, que é o system prompt, e só avisa no log dele.
                truncated=limit is not None and prompt_tokens is not None and prompt_tokens >= limit - 2,
                abstained=abstention.is_abstention(result["answer"], generation.NOT_FOUND_MARKER),
                error=None,
            )
            break
        except Exception as exc:  # noqa: BLE001 - registra e segue; quem julga marca como não medido
            record.update(actual_output=None, retrieval_context=[], sources=[], source_documents=[],
                          error=f"{type(exc).__name__}: {exc}")
            print(f"  [{record['name']}] tentativa {attempt}/2 falhou: {exc!r}", flush=True)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    return record


def main() -> int:
    args = build_parser().parse_args()
    if args.api:
        return api_main(args)

    overrides = arm.parse_overrides(args.set)
    if args.index:
        overrides["rag_index"] = args.index
    # A configuração é cacheada: as sobrescritas valem a partir daqui, para tudo que ler get_settings().
    set_settings_overrides(overrides)
    if args.local:
        providers.set_active_provider("local")

    if vector_store.collection_count() == 0:
        print("[ERRO] Índice vazio — rode `python scripts/ingest_documents.py` antes.", file=sys.stderr)
        return 1

    golden_sets = [parse_goldens_arg(g) for g in args.goldens] or [("pdf", DEFAULT_GOLDENS)]
    goldens = load_goldens(golden_sets, args.limit)
    asks, code = _plan_or_exit(goldens, args)
    if asks is None:
        return code

    settings = get_settings()
    active = providers.get_active_provider()
    if protocol_active(args) and active.kind == "smartdatatest" and (args.sementes or args.estado_limpo
                                                                    or args.controle_estado):
        print("[ERRO] O protocolo da pergunta única precisa do provider local (--local): o remoto ignora a semente "
              "e não tem como ser descarregado.", file=sys.stderr)
        return 2
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": active.name,
        "generation_model": active.generation_model,
        "rag_index": settings.index_name,
        "collection": settings.vector_store_collection,
        "embedding_model": settings.embedding_model,
        "lang": args.lang,
        "top_k": settings.top_k,
        "prompt_version": settings.prompt_version,
        "goldens": [f"{origin}={path.as_posix()}" for origin, path in golden_sets],
        "goldens_sha256": {f"{origin}={path.as_posix()}": _sha256(path) for origin, path in golden_sets},
        "shuffle_seed": args.shuffle_seed,
        "arm": arm.provenance({k: v for k, v in overrides.items() if k != "rag_index"}),
        "n_cases": len(asks),
        "answers": [],
    }
    ollama_url = (args.ollama or active.base_url).rstrip("/") if active.kind != "smartdatatest" else None
    if protocol_active(args):
        output["protocolo"] = protocol_header(args, active.generation_model, ollama_url)
        fill_temperature(output["protocolo"], settings.generation_temperature, ollama_url, active.generation_model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Gerando {len(asks)} respostas com {active.generation_model} ({active.name}), "
        f"índice {settings.index_name}, perguntas em {args.lang}"
        + (f", ordem embaralhada (semente {args.shuffle_seed})." if args.shuffle_seed is not None else "."),
        flush=True,
    )

    def ask_fn(origin: str, golden: dict, question: str, seed: int | None) -> dict:
        return answer_record(origin, golden, question, settings.top_k, seed)

    unload_fn = (lambda: unload_model(ollama_url, active.generation_model)) if ollama_url else None
    code = run_asks(asks, goldens, args, output, ask_fn, unload_fn)

    failures = sum(1 for a in output["answers"] if a["error"] is not None)
    truncated = sum(1 for a in output["answers"] if a.get("truncated"))
    print(f"Respostas em {args.out} ({failures} falha(s), {truncated} truncada(s)).", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
