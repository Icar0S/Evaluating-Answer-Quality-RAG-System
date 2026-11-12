"""O quadro da stack em Docker: endereços, trechos de cada índice, modelos e avisos.

Dois usos, o mesmo script (só biblioteca padrão):

    python scripts/stack_status.py                      # serviço `status` do docker-compose.yml
    python scripts/stack_status.py --preflight [modo] [opções]   # antes dos testes (docker/tests-entrypoint.sh)

Confere tudo pelos nomes de serviço da rede do Compose (backend, backend-md,
frontend, ollama), mas mostra os endereços do HOST, que são os que a pessoa abre
no navegador. No preflight, diz também se a chave da TypeSafe está configurada
(só sim/não: o valor nunca é impresso), o que vai rodar e quanto deve levar, pela
tabela do README (medida numa RTX 4060 de 8 GB).

Sai com 1 só se um serviço não responder. Índice vazio e modelo faltando são
avisos: a stack sobe assim mesmo, e os testes que dependem deles se auto-pulam.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

# Os endereços do host (portas do docker-compose.yml).
HOST_HOME = "http://localhost:5510"
HOST_PDF_API = "http://localhost:8010"
HOST_MD_API = "http://localhost:8011"
HOST_OLLAMA = "http://localhost:11435"


def targets() -> dict[str, str]:
    """Onde conferir cada serviço: dentro da rede do Compose. As variáveis servem para rodar de outro lugar."""
    return {
        "pdf_api": os.environ.get("STATUS_PDF_API", "http://backend:8010"),
        "md_api": os.environ.get("STATUS_MD_API", "http://backend-md:8010"),
        "frontend": os.environ.get("STATUS_FRONTEND", "http://frontend"),
        "ollama": os.environ.get("STATUS_OLLAMA", "http://ollama:11434"),
    }


def rags() -> tuple[dict, dict]:
    """Os dois RAGs. Cada backend lê a pasta que o ingest dele monta (docker-compose.yml), seja qual for a versão."""
    where = targets()
    return (
        {"label": "PDFs", "service": "backend", "folder": "data/source_pdfs", "api": where["pdf_api"],
         "host_api": HOST_PDF_API, "chat": f"{HOST_HOME}/chat.html"},
        {"label": "Markdown", "service": "backend-md", "folder": "data/new_papers_marks", "api": where["md_api"],
         "host_api": HOST_MD_API, "chat": f"{HOST_HOME}/chat.html?api={HOST_MD_API}"},
    )

NEXT_STEP = (
    "Testes (num container, contra este RAG):",
    "  Windows:      scripts\\docker_tests.bat --rapido",
    "  Linux/macOS:  ./scripts/docker_tests.sh --rapido",
)
REINDEX = "scripts\\docker_tests.bat reindexar  (Linux/macOS: ./scripts/docker_tests.sh reindexar)"

# README, "Quanto tempo leva" e "Demonstração E2E ao vivo": medido numa RTX 4060 de 8 GB.
GOLDENS_PER_SET = 10


def http_text(url: str, timeout: float = 5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - só serviços da stack
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - qualquer falha = fora do ar
        return None


def http_json(url: str, timeout: float = 5) -> dict | None:
    body = http_text(url, timeout)
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError:
        return None


def models_from_env() -> dict[str, str]:
    """Os papéis e os modelos da stack (docker-compose.yml, DEMO_*)."""
    return {
        "gerador": os.environ.get("DEMO_GENERATION_MODEL") or "gemma3:4b",
        "juiz": os.environ.get("DEMO_JUDGE_MODEL") or "qwen3:8b",
        "embeddings": os.environ.get("DEMO_EMBEDDING_MODEL") or "nomic-embed-text",
    }


def has_model(tags: set[str], model: str) -> bool:
    return model in tags or f"{model}:latest" in tags


def collect(timeout: float = 5) -> dict:
    """Consulta cada serviço uma vez e devolve o estado, com os avisos já escritos."""
    where = targets()
    warnings: list[str] = []
    down: list[str] = []
    states = []
    for rag in rags():
        health = http_json(f"{rag['api']}/health", timeout=max(timeout, 15))
        states.append({**rag, "health": health})
        if health is None:
            down.append(f"O RAG dos {rag['label']} não respondeu em {rag['api']} (docker compose logs {rag['service']}).")
            continue
        index = health.get("rag_index") or "?"
        if not health.get("ollama_reachable"):
            warnings.append(f"O RAG dos {rag['label']} ({index}) não alcança o Ollama (docker compose logs ollama).")
        if health.get("vector_store_documents") == 0:
            warnings.append(
                f"Índice {index} VAZIO: faltam os documentos em {rag['folder']} (o acervo não é versionado). "
                f"O RAG dos {rag['label']} responde \"não encontrei\" a tudo, e os testes com geração real dele "
                f"vão se auto-pular. Ponha os arquivos lá e reindexe: {REINDEX}."
            )
    tags_body = http_json(f"{where['ollama']}/api/tags", timeout=timeout)
    tags = {m.get("name", "") for m in (tags_body or {}).get("models", [])} if tags_body is not None else None
    models = models_from_env()
    if tags is None:
        down.append(f"O Ollama não respondeu em {where['ollama']} (docker compose logs ollama).")
    else:
        for role, model in models.items():
            if not has_model(tags, model):
                warnings.append(f"Modelo {model} ({role}) não está no Ollama: veja docker compose logs models.")
    if models["gerador"] == models["juiz"]:
        warnings.append(f"Gerador e juiz são o mesmo modelo ({models['gerador']}): o juiz avalia a si mesmo "
                        "(self-grading). Troque DEMO_JUDGE_MODEL.")
    page = http_text(f"{where['frontend']}/chat.html", timeout=timeout)
    frontend_ok = bool(page) and "components/chat.js" in page
    if not frontend_ok:
        down.append(f"O frontend não serve o chat deste projeto em {where['frontend']} "
                    "(docker compose logs frontend).")
    return {"rags": states, "tags": tags, "models": models, "frontend_ok": frontend_ok, "warnings": warnings,
            "down": down, "ollama": where["ollama"]}


def _rag_line(rag: dict) -> str:
    health = rag["health"]
    if health is None:
        return f"  RAG dos {rag['label']:<9} {rag['host_api']}   FORA DO AR"
    chunks = health.get("vector_store_documents")
    return f"  RAG dos {rag['label']:<9} {rag['host_api']}   {health.get('rag_index') or '?':<14} {chunks} trechos"


def _model_lines(state: dict) -> list[str]:
    tags = state["tags"]
    lines = []
    for role, model in state["models"].items():
        mark = "?" if tags is None else ("ok" if has_model(tags, model) else "FALTA")
        lines.append(f"  [{mark:^5}] {role:<11} {model}")
    return lines


def _warning_lines(state: dict) -> list[str]:
    items = state["down"] + state["warnings"]
    return [f"  - {w}" for w in items] if items else ["  (nenhum)"]


def board(state: dict) -> str:
    """O quadro do serviço `status`, com os endereços do host."""
    rule = "=" * 78
    pdf, md = state["rags"]
    lines = [
        rule,
        " RAG no ar. Abra no navegador:",
        f"   Página inicial         {HOST_HOME}",
        f"   Chat do RAG dos PDFs   {pdf['chat']}",
        f"   Chat do RAG dos MD     {md['chat']}",
        "",
        " APIs e índices:",
        _rag_line(pdf),
        _rag_line(md),
        f"  Ollama            {HOST_OLLAMA}",
        "",
        " Modelos no Ollama:",
        *_model_lines(state),
        "",
        " Avisos:",
        *_warning_lines(state),
        "",
        *(f" {line}" for line in NEXT_STEP),
        " Parar: docker compose down   (índices e modelos ficam nos volumes)",
        rule,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- preflight


def typesafe_configured() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


def _run_all_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--pular", nargs="+", default=[])
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--qualidade-pytest", action="store_true")
    known, _ = parser.parse_known_args(args)
    return known


def plan(mode: str, args: list[str], typesafe: bool) -> tuple[str, list[str]]:
    """O que o modo vai rodar e a estimativa de cada parte (README, medida numa RTX 4060 de 8 GB)."""
    if mode == "hermeticos":
        return "API, gerador de relatório e JevEval com o Jev falso (sem a stack)", ["~1 min"]
    if mode == "demo-rapido":
        return "os testes ao vivo nos dois RAGs, sem perguntas de referência nem relatório", ["~5 min"]
    if mode == "demo":
        n = GOLDENS_PER_SET
        if "--goldens" in args:
            try:
                n = max(1, int(args[args.index("--goldens") + 1]))
            except (IndexError, ValueError):
                pass
        answers = 2 * n * 2  # os goldens dos PDFs e dos Markdown, em cada um dos dois RAGs
        return (
            f"E2E ao vivo nos dois RAGs ({2 * n} perguntas de referência em cada), RAGAS e o relatório comparativo",
            [
                "parte ao vivo: ~10 min com todas as perguntas (no ensaio de 13/10/2026 com 2 por acervo, a "
                "demonstração inteira levou 16,6 min)",
                f"julgamento: ~70 s por resposta com GPU, {answers} respostas = ~{round(70 * answers / 60)} min",
            ],
        )
    if mode in ("relatorio-e2e", "shell"):
        return {"relatorio-e2e": "o relatório HTML do Playwright em http://localhost:9323",
                "shell": "bash no container"}[mode], []
    # run_all, com as respostas geradas pela API do RAG no ar.
    opts = _run_all_args(args)
    skip = set(opts.pular)
    lines = ["API, gerador de relatório, JevEval hermético e E2E: ~3 min"]
    quality = [s for s in ("ragas", "deepeval") if s not in skip and not opts.rapido]
    jev_live = "jeveval" not in skip and not opts.rapido
    if quality or jev_live:
        n = opts.limite or GOLDENS_PER_SET
        lines.append(f"respostas pela API ({n} golden(s)): ~{max(1, round(0.2 * n))} min")
        if "ragas" in quality:
            lines.append(f"RAGAS (controles + {n} golden(s)): ~{round(2 + 1.4 * n)} min")
        if "deepeval" in quality:
            lines.append(f"DeepEval ({n} golden(s)): de ~{2 * n} min a horas (uma métrica já passou de 15 min "
                         "num golden); para encurtar, --pular deepeval")
        if jev_live:
            lines.append("JevEval: ainda não medido (~85 chamadas à API da TypeSafe com 10 goldens)" if typesafe
                         else "JevEval ao vivo: pulado (sem TYPESAFE_API_KEY)")
    if opts.qualidade_pytest:
        lines.append("--qualidade-pytest: pulado no Docker (chama o pipeline in-process e precisa do índice local)")
    what = "tests/run_all.py " + " ".join(args) if args else "tests/run_all.py (todas as suítes)"
    return what.strip(), lines


def preflight(mode: str, args: list[str], state: dict | None) -> str:
    rule = "-" * 78
    typesafe = typesafe_configured()
    what, estimate = plan(mode, args, typesafe)
    lines = [rule, " Preflight dos testes"]
    if state is None:
        lines.append("  stack: não é necessária neste modo")
    else:
        for rag in state["rags"]:
            health = rag["health"]
            where = rag["api"].split("//", 1)[-1]
            if health is None:
                lines.append(f"  RAG dos {rag['label']:<9} {where:<18} FORA DO AR (os testes dele vão se auto-pular)")
            else:
                lines.append(f"  RAG dos {rag['label']:<9} {where:<18} {health.get('rag_index') or '?':<14} "
                             f"{health.get('vector_store_documents')} trechos")
        lines.append(f"  Ollama           {state['ollama'].split('//', 1)[-1]}")
        lines += _model_lines(state)
        if state["down"] or state["warnings"]:
            lines.append("  Avisos:")
            lines += [f"    - {w}" for w in state["down"] + state["warnings"]]
    lines += [
        f"  Chave da TypeSafe (JevEval ao vivo): {'configurada: sim' if typesafe else 'configurada: não'}",
        f"  Vai rodar: {what}",
    ]
    if estimate:
        lines.append("  Estimativa (RTX 4060 de 8 GB; em CPU, bem mais):")
        lines += [f"    - {line}" for line in estimate]
    lines.append(rule)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Quadro da stack em Docker (e preflight dos testes).")
    parser.add_argument("--preflight", nargs=argparse.REMAINDER, default=None, metavar="MODO",
                        help="modo dos testes e as opções dele (docker/tests-entrypoint.sh)")
    args = parser.parse_args(argv)
    if args.preflight is not None:
        mode, rest = (args.preflight[0], args.preflight[1:]) if args.preflight else ("run_all", [])
        if mode.startswith("-"):
            mode, rest = "run_all", args.preflight
        state = None if mode in ("hermeticos", "shell", "relatorio-e2e") else collect()
        print(preflight(mode, rest, state), flush=True)
        return 0
    state = collect()
    print(board(state), flush=True)
    return 1 if state["down"] else 0


if __name__ == "__main__":
    sys.exit(main())
