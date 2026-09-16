"""Diagnostica o que o oráculo O1 consegue e não consegue ver.

A calibração de τ* devolveu AUC 0,485 — pior que o acaso. Um número só, porém,
diz "o oráculo é inútil", o que é impreciso e não serve ao artigo. Este script
abre o agregado em duas direções:

1. **por tipo de erro** — o AUC global esconde regimes opostos. O cosseno separa
   entidade trocada com folga e se INVERTE para número e condição: nesses dois o
   erro pontua mais alto que a paráfrase correta na maioria das comparações;

2. **por modelo de embedding** — se o efeito aparece em modelos de famílias
   diferentes, o achado é sobre o método (cosseno contra referência), não sobre
   uma escolha de modelo deste estudo. É a diferença entre uma nota de rodapé e
   um resultado.

A razão do avesso é estrutural: o erro injetado é a referência com UM token
alterado, textualmente quase idêntica; a paráfrase é uma reescrita honesta, com
outro verbo e outra ordem. O cosseno mede proximidade de superfície, então
prefere a corrupção mínima à reformulação legítima.

Os controles sintéticos (--controls) tornam isso legível numa tabela só, sem
depender do conjunto de calibração.

Uso:
    python -m tests.mutation.calibration.diagnose_o1
    python -m tests.mutation.calibration.diagnose_o1 --models nomic-embed-text,all-minilm
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import load_study_config, sut_configuration

logger = get_logger("mutation.diagnostico")

DIAGNOSTIC_PATH = paths.CALIBRATION_DIR / "oracle_diagnostic.json"

# Pares sintéticos que isolam cada dimensão. O par "negado" é o mais revelador:
# inverte o sentido da frase mudando uma palavra.
CONTROLS = [
    ("identico", "A taxa de elegibilidade foi de 56,58%.", "A taxa de elegibilidade foi de 56,58%."),
    ("parafrase correta", "A taxa de elegibilidade foi de 56,58%.", "Foi de 56,58% a taxa de elegibilidade."),
    ("frase negada", "O teste revelou diferencas significativas.", "O teste nao revelou diferencas significativas."),
    ("numero trocado", "A taxa de elegibilidade foi de 56,58%.", "A taxa de elegibilidade foi de 85,00%."),
    ("entidade trocada", "O PIT foi adotado para o teste de mutacao.", "O Orion foi adotado para o teste de mutacao."),
    ("assunto alheio", "A taxa de elegibilidade foi de 56,58%.", "O bolo de cenoura leva tres ovos e farinha."),
    ("outro idioma", "A taxa de elegibilidade foi de 56,58%.", "The quick brown fox jumps over the lazy dog."),
]


def auc_against(positives: list[float], negatives: list[float]) -> float:
    """Probabilidade de um par EQUIV pontuar acima de um par ERRO.

    0,5 é o acaso. Abaixo de 0,5 significa que o ERRO pontua mais alto — o
    oráculo não só erra, ele ordena ao contrário.
    """
    if not positives or not negatives:
        return float("nan")
    wins = sum(1 for a in positives for b in negatives if a > b)
    ties = sum(1 for a in positives for b in negatives if a == b)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def measure_model(model: str, pairs: list[dict]) -> dict:
    from tests.mutation.oracles.o1_cosine import similarity

    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        key = "EQUIV" if pair["label"] == "EQUIV" else pair["rule"].split(":")[0]
        grouped[key].append(similarity(pair["candidate"], pair["reference"], model))

    equiv = grouped.get("EQUIV", [])
    by_rule = {}
    for rule, scores in sorted(grouped.items()):
        if rule == "EQUIV":
            continue
        by_rule[rule] = {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "auc_vs_equiv": round(auc_against(equiv, scores), 3),
        }

    todos_erros = [s for rule, scores in grouped.items() if rule != "EQUIV" for s in scores]
    return {
        "equiv": {"n": len(equiv), "mean": round(statistics.mean(equiv), 4) if equiv else None},
        "by_rule": by_rule,
        "auc_global": round(auc_against(equiv, todos_erros), 3),
        "controls": {name: round(similarity(a, b, model), 4) for name, a, b in CONTROLS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico do oráculo O1.")
    parser.add_argument("--models", default="nomic-embed-text,all-minilm")
    args = parser.parse_args(argv)

    pairs = [p for p in jsonl.read(paths.CALIBRATION_PAIRS) if not p.get("excluded")]
    if not pairs:
        logger.error("calibration/pairs.jsonl vazio — rode build_pairs.py antes.")
        return 1

    study = load_study_config()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    report: dict[str, dict] = {}

    with sut_configuration(study.baseline_overrides):
        for model in models:
            logger.info("Medindo %s sobre %d pares...", model, len(pairs))
            try:
                report[model] = measure_model(model, pairs)
            except Exception as exc:  # noqa: BLE001 — modelo ausente não derruba o resto
                logger.warning("%s indisponível: %s", model, exc)

    DIAGNOSTIC_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for model, data in report.items():
        print(f"\n=== {model} ===  (EQUIV n={data['equiv']['n']}, média {data['equiv']['mean']})")
        print(f"{'tipo de erro':<22}{'n':>4}{'média':>9}{'AUC vs EQUIV':>14}")
        for rule, stats in sorted(data["by_rule"].items(), key=lambda kv: kv[1]["auc_vs_equiv"]):
            marca = "  <- ordena ao contrário" if stats["auc_vs_equiv"] < 0.5 else ""
            print(f"{rule:<22}{stats['n']:>4}{stats['mean']:>9.4f}{stats['auc_vs_equiv']:>14.3f}{marca}")
        print(f"{'AUC global':<22}{'':>4}{'':>9}{data['auc_global']:>14.3f}")
        print("  controles:")
        for name, score in data["controls"].items():
            print(f"    {name:<22}{score:.4f}")

    print(f"\nrelatório em {DIAGNOSTIC_PATH}")
    invertidos = {
        model: [r for r, s in data["by_rule"].items() if s["auc_vs_equiv"] < 0.5]
        for model, data in report.items()
    }
    if all(invertidos.values()) and len(invertidos) > 1:
        comuns = set.intersection(*(set(v) for v in invertidos.values()))
        if comuns:
            logger.warning(
                "Inversão replica em TODOS os modelos testados para: %s. O achado é sobre "
                "o método (cosseno contra referência), não sobre a escolha do modelo.",
                ", ".join(sorted(comuns)),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
