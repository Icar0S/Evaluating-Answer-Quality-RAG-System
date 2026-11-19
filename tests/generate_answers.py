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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

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
    return parser


def load_goldens(golden_sets: list[tuple[str, Path]], limit: int | None) -> list[tuple[str, dict]]:
    goldens = []
    for origin, path in golden_sets:
        items = json.loads(path.read_text(encoding="utf-8"))[:limit]
        goldens += [(origin, g) for g in items]
    return goldens


def golden_record(origin: str, golden: dict, question: str) -> dict:
    """O começo de cada resposta: o que vem do golden (pergunta, referência, contexto, documento-fonte)."""
    return {
        "name": golden.get("name") or golden["input"][:40],
        "origin": origin,
        "input": question,
        "expected_output": golden.get("expected_output", ""),
        "context": golden.get("context") or [],
        "source_document": golden.get("source_document"),
    }


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


def api_answer_record(api: str, origin: str, golden: dict, question: str) -> tuple[dict, dict | None]:
    """Pergunta pela API, como o chat faria. Devolve a resposta no formato do modo in-process e o metadata do /chat.

    Mesma política do modo in-process: duas tentativas, e erro vira registro (quem julga marca como não medido).
    """
    record = golden_record(origin, golden, question)
    metadata = None
    t0 = time.perf_counter()
    for attempt in (1, 2):
        try:
            reply = api_call(f"{api}/chat", {"question": question})
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
                error=None,
            )
            break
        except (ApiError, KeyError, TypeError) as exc:
            record.update(actual_output=None, retrieval_context=[], sources=[], source_documents=[],
                          error=f"{type(exc).__name__}: {exc}")
            print(f"  [{record['name']}] tentativa {attempt}/2 falhou: {exc}", flush=True)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    return record, metadata


def api_main(args: argparse.Namespace) -> int:
    """As respostas pela API do RAG no ar. Não importa o backend: o RAG é testado como caixa-preta."""
    in_process_only = [name for name, value in (("--index", args.index), ("--set", args.set), ("--local", args.local))
                       if value]
    if in_process_only:
        print(f"[ERRO] {', '.join(in_process_only)} não vale(m) com --api: índice, modelos e configuração são os "
              "do backend que está no ar.", file=sys.stderr)
        return 2
    api = args.api.rstrip("/")
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
        "n_cases": len(goldens),
        "answers": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Gerando {len(goldens)} respostas pela API em {api} (índice {health.get('rag_index')}, "
        f"gerador {health.get('generation_model')}), perguntas em {args.lang}"
        + (f", ordem embaralhada (semente {args.shuffle_seed})." if args.shuffle_seed is not None else "."),
        flush=True,
    )
    failures = 0
    for position, i in enumerate(processing_order(len(goldens), args.shuffle_seed), start=1):
        origin, golden = goldens[i]
        question = golden.get("input_en") if args.lang == "en" and golden.get("input_en") else golden["input"]
        record, metadata = api_answer_record(api, origin, golden, question)
        if metadata:
            output.update(
                provider=metadata.get("provider"),
                generation_model=metadata.get("model") or output["generation_model"],
                embedding_model=metadata.get("embedding_model") or output["embedding_model"],
                top_k=metadata.get("top_k"),
                prompt_version=metadata.get("prompt_version"),
                rag_index=metadata.get("rag_index") or output["rag_index"],
            )
        record["position"] = position
        failures += record["error"] is not None
        output["answers"].append(record)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "ERRO" if record["error"] else f"{len(record['actual_output'])} caracteres"
        print(f"  [{position}/{len(goldens)}] {record['name']}: {status} ({record['seconds']}s)", flush=True)

    print(f"Respostas em {args.out} ({failures} falha(s)).", flush=True)
    return 0 if failures < len(goldens) else 1


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


def answer_record(origin: str, golden: dict, question: str, top_k: int) -> dict:
    """Recupera, gera e mede uma resposta. Erro vira registro (quem julga marca como não medido)."""
    record = golden_record(origin, golden, question)
    t0 = time.perf_counter()
    # Uma tentativa extra, como no run_and_export.py do DeepEval: a maioria
    # dos timeouts do Ollama aqui é transitória.
    for attempt in (1, 2):
        try:
            chunks, retrieval_ms = retrieval.retrieve(question, top_k)
            result = generation.generate_answer(question, chunks)
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

    settings = get_settings()
    active = providers.get_active_provider()
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
        "n_cases": len(goldens),
        "answers": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Gerando {len(goldens)} respostas com {active.generation_model} ({active.name}), "
        f"índice {settings.index_name}, perguntas em {args.lang}"
        + (f", ordem embaralhada (semente {args.shuffle_seed})." if args.shuffle_seed is not None else "."),
        flush=True,
    )
    failures = 0
    for position, i in enumerate(processing_order(len(goldens), args.shuffle_seed), start=1):
        origin, golden = goldens[i]
        question = golden.get("input_en") if args.lang == "en" and golden.get("input_en") else golden["input"]
        record = answer_record(origin, golden, question, settings.top_k)
        record["position"] = position
        failures += record["error"] is not None
        output["answers"].append(record)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "ERRO" if record["error"] else f"{len(record['actual_output'])} caracteres"
        flags = " [TRUNCADO]" if record.get("truncated") else ""
        print(f"  [{position}/{len(goldens)}] {record['name']}: {status}{flags} ({record['seconds']}s)", flush=True)

    truncated = sum(1 for a in output["answers"] if a.get("truncated"))
    print(f"Respostas em {args.out} ({failures} falha(s), {truncated} truncada(s)).", flush=True)
    return 0 if failures < len(goldens) else 1


if __name__ == "__main__":
    sys.exit(main())
