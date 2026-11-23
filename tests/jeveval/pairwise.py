"""A comparação em pares do JevEval para a pergunta única (plano-escopo-final.md, §6): uma Choice A, B ou Empate.

O Jev não gera texto: responde uma pergunta fechada com a probabilidade de cada
opção. O par é UMA pergunta Choice sobre um estado com a pergunta, a referência,
os trechos e as duas respostas (`answer_A` primeiro, `answer_B` depois). As
opções são "A", "B" e "Empate", cada uma com a descrição do que quer dizer.
Diferente do ArenaGEval do DeepEval, aqui o empate existe e vem com
probabilidade: P(A), P(B) e P(Empate) vão para o export.

A ordem é a do plano (pares.pair_plan): na ordem BA, a segunda resposta do par
vai em `answer_A`. É assim que se mede o viés de posição do Jev.

Como as outras chamadas desta camada, cada par manda pergunta, referência,
trechos e respostas para https://api.typesafe.ai (só o acervo público; ver
README.md). Com o estado de ~5.400 tokens estimados, cabe com folga no limite de
32.000 do Jev (docs/escopo-final-status.md, §2).

Rodar da raiz do repo (venv do JevEval):
    tests\\jeveval\\.venv\\Scripts\\python.exe tests\\jeveval\\pairwise.py --answers respostas.json --out pares.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "tests" / "pergunta_unica"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _jev_shared as shared  # noqa: E402
import pares  # noqa: E402
from deepeval.models.system_one.schema import ChoiceQuestion  # noqa: E402

QUESTION_KEY = "preferencia"
QUESTION = (
    pares.CRITERION
    + " Here the retrieved passages are `retrieval_context`, and the two answers are `answer_A` and `answer_B`."
)
# Os valores vão ao Jev como a descrição de cada opção (os `criteria` da Choice no SDK).
OPTIONS = {
    "A": "`answer_A` is the better answer",
    "B": "`answer_B` is the better answer",
    "Empate": "both answers are equally good, or equally bad",
}


def pair_state(item: dict, answers: dict[str, dict]) -> dict:
    first = answers[item["primeiro"]]
    return {
        "input": first["input"],
        "expected_output": first.get("expected_output") or "",
        "retrieval_context": list(first.get("retrieval_context") or []),
        "answer_A": answers[item["primeiro"]]["actual_output"],
        "answer_B": answers[item["segundo"]]["actual_output"],
    }


def judge_pair(model, item: dict, answers: dict[str, dict]) -> tuple[dict, float]:
    """Um julgamento. Erro vira registro (não julgado), nunca exceção. Devolve o julgamento e o custo."""
    t0 = time.perf_counter()
    try:
        result, cost = model.decide(pair_state(item, answers),
                                    {QUESTION_KEY: ChoiceQuestion(instructions=QUESTION, options=OPTIONS)})
    except Exception as exc:  # noqa: BLE001 - qualquer falha do Jev vira registro
        return pares.judgment(item, None, erro=f"{type(exc).__name__}: {exc}",
                              segundos=round(time.perf_counter() - t0, 2)), 0.0
    seconds = round(time.perf_counter() - t0, 2)
    answer = result.choices.get(QUESTION_KEY)
    if answer is None or answer.choice not in OPTIONS:
        return pares.judgment(item, None, erro=f"escolha fora das opções: {getattr(answer, 'choice', None)!r}",
                              segundos=seconds), cost or 0.0
    winner = {"A": item["primeiro"], "B": item["segundo"], "Empate": pares.TIE}[answer.choice]
    probabilities = {option: round(float(answer.probabilities.get(option, 0.0)), 4) for option in OPTIONS}
    return pares.judgment(
        item, winner, probabilidades=probabilities, confianca=round(float(answer.confidence), 4), segundos=seconds,
        justificativa=f"Jev: P(A)={probabilities['A']:.2f}, P(B)={probabilities['B']:.2f}, "
                      f"P(Empate)={probabilities['Empate']:.2f}",
    ), cost or 0.0


def main(argv: list[str] | None = None, model=None) -> int:
    """`model` existe para os testes herméticos injetarem o Jev falso; sem ele, usa o Jev de verdade."""
    parser = argparse.ArgumentParser(description="Comparação em pares com o Jev (Choice A, B ou Empate).")
    parser.add_argument("--answers", type=Path, required=True, help="respostas do tests/generate_answers.py")
    parser.add_argument("--out", type=Path, required=True, help="arquivo JSON de saída")
    parser.add_argument("--uma-ordem", action="store_true", help="só a ordem AB (a rodada da apresentação)")
    args = parser.parse_args(argv)

    settings = shared.get_settings()
    if model is None:
        reason = shared.jev_status(settings)
        if reason:
            print(f"[ERRO] Jev indisponível: {reason}", file=sys.stderr)
            return 1
        model = shared.build_jev(settings)
    meta, answers = pares.answers_by_name(args.answers)
    plan = pares.pair_plan(list(answers), both_orders=not args.uma_ordem)
    output = {
        "framework": "jeveval",
        "juiz": "jeveval-par",
        "metodo": "uma Choice A / B / Empate sobre o estado do par; a ordem é a do plano",
        "judge_model": model.get_model_name(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answers_source": str(args.answers),
        "generation_model": meta.get("generation_model"),
        "criterio": pares.CRITERION_PT,
        "pergunta_enviada": QUESTION,
        "opcoes": OPTIONS,
        "duas_ordens": not args.uma_ordem,
        "n_julgamentos": len(plan),
        "custo_usd": 0.0,
        "julgamentos": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pares com o Jev ({output['judge_model']}) | {len(answers)} respostas, {len(plan)} julgamentos", flush=True)
    for i, item in enumerate(plan, start=1):
        result, cost = judge_pair(model, item, answers)
        output["julgamentos"].append(result)
        output["custo_usd"] = round(output["custo_usd"] + cost, 6)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        shown = (f"{result['vencedor']} ({result['justificativa']})" if result["vencedor"]
                 else f"NÃO JULGADO ({(result['erro'] or '')[:100]})")
        print(f"  [{i}/{len(plan)}] {item['primeiro']} x {item['segundo']} ({item['ordem']}): {shown}", flush=True)
    failed = sum(1 for j in output["julgamentos"] if j["vencedor"] is None)
    print(f"Pares em {args.out} ({failed} não julgado(s), US$ {output['custo_usd']:.4f}).", flush=True)
    return 0 if failed < len(plan) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
