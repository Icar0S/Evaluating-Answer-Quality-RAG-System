"""Roda a avaliação completa e salva o resultado em results/<timestamp>.{json,html}.

Script separado do pytest de propósito: `assert_test` (test_rag_quality.py)
serve pra CI/terminal (pass/fail por caso), mas não devolve os scores de um
jeito fácil de serializar. Aqui isso alimenta GET /stats
(deepeval_faithfulness_avg, ver backend/app/main.py::get_stats), o painel
HTML (report.py) e a seção de resultados do artigo, mesmo papel que
logs/interactions.jsonl já tem pras métricas de uso.

Rodar (venv isolado, da raiz do repo):
    tests\\deepeval\\.venv\\Scripts\\python.exe tests\\deepeval\\run_and_export.py

Imprime progresso caso a caso (pergunta X/N, depois cada métrica com score
assim que sai do juiz) e salva o JSON/HTML parcial a cada caso concluído —
não é preciso esperar o fim pra ver algo, e uma interrupção no meio não perde
o que já rodou. Ao terminar de verdade, abre o painel no navegador (ver
view_report.py pra reabrir sem rodar tudo de novo).

Roda os casos em sequência, um de cada vez, chamando `metric.measure()`
direto em vez de `deepeval.evaluate()` em lote — decisão tomada depois de
bater em quatro problemas do `evaluate()` nesta máquina (Windows): (1) o
console "rich" do resumo final imprime emoji que a codepage cp1252 não
decodifica e derruba o processo depois da avaliação real já ter rodado; (2)
o cache local usa portalocker pra lock compartilhado, que sem o extra
`pywin32` quebra do mesmo jeito; (3) concorrência default de até 20 casos ao
mesmo tempo contra um Ollama servindo UM modelo numa GPU só vira fila gigante
(é como um outro processo nesta máquina, sem `max_concurrent`, deixou uma
rodada presa por mais de 1h); (4) sem progresso visível, não dá pra saber se
está rodando ou travado. Medindo caso a caso, sequencial por construção, com
print a cada passo, os quatro problemas somem de uma vez.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

# Teto DURO por chamada de métrica — 300s (5min), pedido explícito depois de
# uma rodada real travar mais de 1h numa única métrica (juiz com "thinking"
# preso raciocinando; ver _shared.py::_NoThinkOllamaModel, que resolve a
# causa). Isto aqui é a rede de segurança, não a correção — thread daemon com
# deadline de verdade, que não depende de nenhuma configuração interna do
# deepeval (já tentamos DEEPEVAL_DISABLE_TIMEOUTS e depois
# DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE; nenhum dos dois cortou uma
# chamada que já passou por múltiplos passos internos do juiz, só bloqueava
# uma etapa de cada vez). daemon=True: se travar mesmo assim, a thread
# abandonada não impede o processo de terminar no fim da rodada.
METRIC_TIMEOUT_SECONDS = 300

# stdout/stderr do Windows, quando redirecionados pra arquivo (> log.txt),
# herdam a codepage do sistema (cp1252 nesta máquina) em vez de UTF-8. Sem
# isso, um caractere não-ASCII (acento, aspas curvas do juiz) pode derrubar
# o processo com UnicodeEncodeError no meio da avaliação real.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

import report  # noqa: E402
from _dataset import load_goldens  # noqa: E402
from _shared import EVAL_JUDGE_MODEL, THRESHOLD, build_judge_model, run_pipeline, stack_status  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _measure_raw(metric, test_case: LLMTestCase) -> dict:
    """A chamada de verdade — roda dentro de uma thread daemon, com deadline
    aplicado por fora (ver _measure_one). Nunca propaga exceção: um JSON
    malformado ou vazio do juiz (visto na prática com o modelo local) vira
    um registro de erro em vez de derrubar as outras ~49 chamadas da rodada.

    Captura BaseException, não só Exception: numa rodada real apareceu um
    CancelledError vindo de dentro de metric.measure() (asyncio interno do
    deepeval) — CancelledError herda de BaseException, não de Exception, e
    escapava de um `except Exception` normal, derrubando a rodada inteira na
    primeira métrica. Ctrl+C continua funcionando: KeyboardInterrupt e
    SystemExit são deliberadamente repassados, não engolidos.
    """
    t0 = time.monotonic()
    try:
        metric.measure(test_case, _show_indicator=False)
        elapsed = time.monotonic() - t0
        _log(f"      {metric.__name__:<22} {metric.score:.2f}  {'PASS' if metric.success else 'FAIL'}  ({elapsed:.0f}s)")
        return {"name": metric.__name__, "score": metric.score, "success": metric.success, "reason": metric.reason, "error": None}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 — deliberado, ver docstring da função
        elapsed = time.monotonic() - t0
        _log(f"      {metric.__name__:<22} ERRO: {exc!r}  ({elapsed:.0f}s)")
        return {"name": metric.__name__, "score": None, "success": None, "reason": None, "error": repr(exc)}


def _measure_one(metric, test_case: LLMTestCase) -> dict:
    """Roda _measure_raw numa thread daemon com um teto de parede de
    METRIC_TIMEOUT_SECONDS — ver comentário da constante pra motivação."""
    box: dict = {}

    def worker() -> None:
        box["result"] = _measure_raw(metric, test_case)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=METRIC_TIMEOUT_SECONDS)
    if t.is_alive():
        _log(f"      {metric.__name__:<22} ERRO: sem resposta em {METRIC_TIMEOUT_SECONDS}s, abandonando e seguindo")
        return {
            "name": metric.__name__,
            "score": None,
            "success": None,
            "reason": None,
            "error": f"timeout: sem resposta em {METRIC_TIMEOUT_SECONDS}s",
        }
    return box["result"]


def _metrics_avg(cases: list[dict]) -> dict[str, float]:
    scores_by_metric: dict[str, list[float]] = {}
    for c in cases:
        for md in c["metrics"]:
            if md["score"] is not None:
                scores_by_metric.setdefault(md["name"], []).append(md["score"])
    return {name: round(sum(s) / len(s), 4) for name, s in scores_by_metric.items() if s}


def _build_output(cases: list[dict], n: int, i: int, settings) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_model": settings.generation_model,
        "judge_model": EVAL_JUDGE_MODEL,
        "threshold": THRESHOLD,
        "n_cases": n,
        "cases_done": i,
        "metrics_avg": _metrics_avg(cases),
        "cases": cases,
    }


def _write_outputs(output: dict, json_path: Path) -> Path:
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    html_doc = report.render_html(output, source_label=f"tests/deepeval/results/{json_path.name}")
    json_path.with_suffix(".html").write_text(html_doc, encoding="utf-8")
    latest_path = RESULTS_DIR / "latest.html"
    latest_path.write_text(html_doc, encoding="utf-8")
    return latest_path


def main() -> None:
    reason = stack_status()
    if reason:
        print(f"Stack real não está pronta, abortando: {reason}", file=sys.stderr)
        raise SystemExit(1)

    judge_model = build_judge_model()
    goldens = load_goldens()

    from app.config import get_settings

    settings = get_settings()

    metrics = [
        FaithfulnessMetric(model=judge_model, threshold=THRESHOLD),
        AnswerRelevancyMetric(model=judge_model, threshold=THRESHOLD),
        ContextualPrecisionMetric(model=judge_model, threshold=THRESHOLD),
        ContextualRecallMetric(model=judge_model, threshold=THRESHOLD),
        ContextualRelevancyMetric(model=judge_model, threshold=THRESHOLD),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"{timestamp}.json"

    n = len(goldens)
    _log(f"Avaliando {n} perguntas — gerador {settings.generation_model}, juiz {EVAL_JUDGE_MODEL}")
    _log(f"Progresso salvo a cada caso em {json_path.name} / .html — dá pra abrir no meio da rodada.\n")

    cases: list[dict] = []
    run_start = time.monotonic()
    for i, golden in enumerate(goldens, start=1):
        case_start = time.monotonic()
        elapsed_total = case_start - run_start
        _log(f"[{i}/{n}] {golden.name} ({elapsed_total:.0f}s decorridos)")
        _log("      recuperando + gerando...")

        # A geração (não só o julgamento) também pode travar/dar timeout no
        # Ollama — visto na prática num caso real (httpx.ReadTimeout depois
        # de ~40min de uso contínuo, derrubando o script inteiro sem essa
        # proteção). Uma tentativa extra: a maioria dos timeouts aqui é
        # transitório (servidor momentaneamente sob pressão), não um travamento
        # persistente como o dos juízes. Se as duas falharem, marca o caso
        # inteiro como erro e segue pro próximo em vez de derrubar a rodada.
        answer = retrieval_context = None
        gen_error: str | None = None
        for attempt in (1, 2):
            try:
                answer, retrieval_context = run_pipeline(golden.input)
                gen_error = None
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 — mesma razão de _measure_raw
                gen_error = repr(exc)
                _log(f"      ERRO na geração (tentativa {attempt}/2): {exc!r}")

        if gen_error is not None:
            _log("      desistindo deste caso após 2 tentativas — seguindo pro próximo\n")
            case_metrics = [
                {"name": m.__name__, "score": None, "success": None, "reason": None, "error": f"geração falhou: {gen_error}"}
                for m in metrics
            ]
        else:
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=retrieval_context,
                name=golden.name,
            )
            case_metrics = [_measure_one(metric, test_case) for metric in metrics]

        overall = gen_error is None and all(m["success"] for m in case_metrics if m["error"] is None) and not any(
            m["error"] is not None for m in case_metrics
        )
        cases.append(
            {
                "name": golden.name,
                "input": golden.input,
                "actual_output": answer,
                "expected_output": golden.expected_output,
                "retrieval_context": retrieval_context,
                "success": overall,
                "metrics": case_metrics,
            }
        )

        output = _build_output(cases, n, i, settings)
        _write_outputs(output, json_path)
        _log(f"    concluído em {time.monotonic() - case_start:.0f}s\n")

    total_elapsed = time.monotonic() - run_start
    _log(f"Rodada completa em {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min).")
    _log(f"Resultado em {json_path}")
    _log(json.dumps(output["metrics_avg"], ensure_ascii=False, indent=2))

    latest_path = _write_outputs(output, json_path)
    _log(f"Painel salvo em {json_path.with_suffix('.html')} (e {latest_path})")
    webbrowser.open(latest_path.resolve().as_uri())


if __name__ == "__main__":
    main()
