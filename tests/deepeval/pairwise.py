"""A comparação em pares do DeepEval para a pergunta única (plano-escopo-final.md, §6): ArenaGEval, ordem fixa.

O ArenaGEval (deepeval 4.2.0) recebe N concorrentes e escolhe UM vencedor, sem
empate. Por padrão, ele troca os nomes dos concorrentes por nomes falsos e
embaralha a ordem com o `random` do módulo, sem semente
(deepeval/metrics/arena_g_eval/utils.py): bom para esconder o viés de posição,
ruim para medi-lo. Aqui o `random` daquele módulo é trocado por um de ordem
fixa: o primeiro concorrente é sempre "Alice", o segundo "Bob", e a ordem do
prompt é a pedida. Ao começar, o script confere isso formatando um par de teste
(sem chamar o juiz) e para se a ordem não for a pedida.

Os trechos vão UMA vez, no fim do `input`, e não como `retrieval_context`: o
ArenaGEval repete os campos de cada concorrente, e os 4 trechos (~10 mil
caracteres) iriam duas vezes ao juiz, perto do num_ctx de 8192 do qwen3:8b. As
duas respostas recebem os mesmos trechos por construção (o protocolo confere o
hash), então nada se perde.

Rodar da raiz do repo (venv do DeepEval):
    tests\\deepeval\\.venv\\Scripts\\python.exe tests\\deepeval\\pairwise.py --answers respostas.json --out pares.json
    tests\\deepeval\\.venv\\Scripts\\python.exe tests\\deepeval\\pairwise.py --verificar-ordem
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "backend"), str(ROOT / "tests" / "pergunta_unica")]

import deepeval.metrics.arena_g_eval.utils as arena_utils  # noqa: E402
from deepeval.metrics import ArenaGEval  # noqa: E402
from deepeval.test_case import ArenaTestCase, Contestant, LLMTestCase, SingleTurnParams  # noqa: E402

import pares  # noqa: E402
from _shared import EVAL_JUDGE_MODEL, build_judge_model, stack_status  # noqa: E402

# O mesmo teto por métrica do run_and_export.py (lá está o porquê).
PAIR_TIMEOUT_SECONDS = 300

# D3: os passos de avaliação, escritos antes de rodar. Fixos, em vez de gerados pelo juiz a partir do critério:
# gerados, mudariam a cada rodada (o juiz também os escreve).
EVALUATION_STEPS = [
    "Check which answer contains more of the facts in `expected_output`, the minimum a correct answer must contain.",
    "Penalize every claim in an answer that is not supported by the retrieved passages at the end of `input`.",
    "Prefer the answer that satisfies both checks better.",
]
PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]


class _FixedOrder:
    """No lugar do módulo `random` de arena_g_eval/utils.py: o shuffle deixa tudo onde está."""

    @staticmethod
    def shuffle(seq) -> None:
        return None


arena_utils.random = _FixedOrder()


def check_fixed_order() -> dict[str, str]:
    """Formata um par de teste como o ArenaGEval formataria e confere a ordem e os nomes. Não chama o juiz."""
    case = ArenaTestCase(contestants=[
        Contestant(name=name, test_case=LLMTestCase(input="q", actual_output=f"saida-{name}", expected_output="r"))
        for name in ("primeiro", "segundo")
    ])
    for _ in range(5):
        formatted, dummy = arena_utils.format_arena_test_case(PARAMS, case)
        text = str(formatted)
        if dummy != {"Alice": "primeiro", "Bob": "segundo"} or text.index("saida-primeiro") > text.index("saida-segundo"):
            raise RuntimeError(f"a ordem do ArenaGEval não ficou fixa (nomes {dummy}); o viés de posição não seria "
                               "medido. O deepeval mudou o arena_g_eval/utils.py?")
    return dummy


def pair_input(question: str, passages: list[str]) -> str:
    blocks = "\n\n".join(f"[Passage {i}]\n{text}" for i, text in enumerate(passages, start=1))
    return f"{question}\n\nRETRIEVED PASSAGES (the same for both answers):\n{blocks}"


def arena_case(item: dict, answers: dict[str, dict]) -> ArenaTestCase:
    first, second = answers[item["primeiro"]], answers[item["segundo"]]
    text = pair_input(first["input"], first.get("retrieval_context") or [])
    return ArenaTestCase(contestants=[
        Contestant(name=name, test_case=LLMTestCase(input=text, actual_output=answers[name]["actual_output"],
                                                    expected_output=first.get("expected_output") or ""))
        for name in (item["primeiro"], item["segundo"])
    ])


def judge_pair(metric: ArenaGEval, item: dict, answers: dict[str, dict]) -> dict:
    """Um julgamento, com teto de parede; erro e timeout viram registro (não julgado), nunca exceção."""
    box: dict = {}

    def worker() -> None:
        try:
            box["winner"] = metric.measure(arena_case(item, answers), _show_indicator=False)
            box["reason"] = metric.reason
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - mesma razão do run_and_export.py
            box["error"] = repr(exc)

    t0 = time.monotonic()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=PAIR_TIMEOUT_SECONDS)
    seconds = round(time.monotonic() - t0, 1)
    masked = {"Alice": item["primeiro"], "Bob": item["segundo"]}
    if thread.is_alive():
        return pares.judgment(item, None, erro=f"timeout: sem resposta em {PAIR_TIMEOUT_SECONDS}s", segundos=seconds)
    if "error" in box:
        return pares.judgment(item, None, erro=box["error"], segundos=seconds)
    winner = box["winner"]
    if winner not in item["par"]:
        return pares.judgment(item, None, erro=f"vencedor fora do par: {winner!r}", segundos=seconds)
    return pares.judgment(item, winner, justificativa=box.get("reason"), segundos=seconds, nomes_mascarados=masked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comparação em pares com o ArenaGEval (ordem fixa).")
    parser.add_argument("--answers", type=Path, help="respostas do tests/generate_answers.py")
    parser.add_argument("--out", type=Path, help="arquivo JSON de saída")
    parser.add_argument("--uma-ordem", action="store_true", help="só a ordem AB (a rodada da apresentação)")
    parser.add_argument("--verificar-ordem", action="store_true", help="só confere a ordem fixa e sai")
    args = parser.parse_args(argv)

    dummy = check_fixed_order()
    print(f"Ordem fixa conferida: {dummy}", flush=True)
    if args.verificar_ordem:
        return 0
    if not args.answers or not args.out:
        parser.error("--answers e --out são obrigatórios (ou use --verificar-ordem)")
    reason = stack_status(need_index=False)
    if reason:
        print(f"[ERRO] Stack não está pronta: {reason}", file=sys.stderr)
        return 1

    meta, answers = pares.answers_by_name(args.answers)
    plan = pares.pair_plan(list(answers), both_orders=not args.uma_ordem)
    metric = ArenaGEval(name="preferencia", criteria=pares.CRITERION, evaluation_steps=EVALUATION_STEPS,
                        evaluation_params=PARAMS, model=build_judge_model(), async_mode=False)
    from importlib.metadata import version

    output = {
        "framework": "deepeval",
        "juiz": "deepeval-arena",
        "metodo": "ArenaGEval (sem empate), ordem fixa: o primeiro concorrente é Alice, o segundo Bob",
        "deepeval_version": version("deepeval"),
        "judge_model": EVAL_JUDGE_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answers_source": str(args.answers),
        "generation_model": meta.get("generation_model"),
        "criterio": pares.CRITERION_PT,
        "criterio_enviado": pares.CRITERION,
        "passos_de_avaliacao": EVALUATION_STEPS,
        "duas_ordens": not args.uma_ordem,
        "n_julgamentos": len(plan),
        "julgamentos": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"ArenaGEval | juiz {EVAL_JUDGE_MODEL} | {len(answers)} respostas, {len(plan)} julgamentos", flush=True)
    for i, item in enumerate(plan, start=1):
        result = judge_pair(metric, item, answers)
        output["julgamentos"].append(result)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        shown = result["vencedor"] or f"NÃO JULGADO ({(result['erro'] or '')[:100]})"
        print(f"  [{i}/{len(plan)}] {item['primeiro']} x {item['segundo']} ({item['ordem']}): {shown} "
              f"({result['segundos']}s)", flush=True)
    failed = sum(1 for j in output["julgamentos"] if j["vencedor"] is None)
    print(f"Pares em {args.out} ({failed} não julgado(s)).", flush=True)
    return 0 if failed < len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
