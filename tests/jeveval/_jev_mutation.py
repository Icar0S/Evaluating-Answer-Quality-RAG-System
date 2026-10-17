"""Leitura dos artefatos do estudo de mutação e as regras do O6.

O O6 é o oráculo de correção desta camada (_jev_shared.CORRECTNESS) aplicado ao
log da campanha (tests/mutation/results/runs.jsonl), como análise
EXPLORATÓRIA: a campanha fechou com O1, O2, O4 e O5, e um oráculo criado depois
de ver os resultados não pode entrar no resultado confirmatório sem ser
declarado como desvio. Por isso os vereditos do O6 ficam em
tests/jeveval/results/o6/, nunca em tests/mutation/results/verdicts.jsonl.

As regras são as do tests/mutation/evaluate.py, replicadas aqui porque aquele
módulo importa os cinco oráculos (ragas, httpx, o backend) no carregamento e
não abre neste venv:

- execuções com erro, ou cortadas pelo teto de geração, não entram;
- o O6 julga a resposta MODAL das N execuções do nível (como O3 e O4), com o
  mesmo `modal_answer` do estudo, importado de lá (só biblioteca padrão);
- conjunto avaliável S = casos cujo baseline foi medido e PASSOU (§7.1, com o
  refinamento `evaluable_requires_baseline_pass` do study.yaml);
- morte = mutante != baseline, veredito "fail", medido, caso em S;
- mutante morto = ao menos um caso de S reprovado (tests/mutation/analyze.py).

Um teste hermético confere cada regra contra um log sintético.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from _jev_shared import PROJECT_ROOT
from _jev_stats import cohen_kappa, interpret_kappa, wilson_ci

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.mutation.oracles.base import modal_answer  # noqa: E402

MUTATION_ROOT = PROJECT_ROOT / "tests" / "mutation"
STUDY_CONFIG = MUTATION_ROOT / "config" / "study.yaml"
GOLDEN = MUTATION_ROOT / "suite" / "golden.jsonl"
PAIRS = MUTATION_ROOT / "calibration" / "pairs.jsonl"
TAU_JSON = MUTATION_ROOT / "calibration" / "tau.json"
RUNS_JSONL = MUTATION_ROOT / "results" / "runs.jsonl"
VERDICTS_JSONL = MUTATION_ROOT / "results" / "verdicts.jsonl"
BASELINE_ID = "baseline"
# Blocos do log que não são mutantes: `l3-baseline` é o baseline do nível L3 (30
# execuções sobre 10 casos, tests/mutation/run_campaign.py::run_l3). Julgado como
# mutante, ele entraria no denominador do escore e o sistema NÃO mutado poderia
# "morrer". Os vereditos da campanha no L2 também não o contam.
NOT_MUTANTS = frozenset({"l3-baseline"})
ORACLE = "O6"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_study(path: Path = STUDY_CONFIG) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_golden(path: Path = GOLDEN) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in read_jsonl(path)}


# ------------------------------------------------------------ pares rotulados


@dataclass(frozen=True)
class LabelledPair:
    pair_id: str
    case_id: str
    question: str
    reference: str
    candidate: str
    label: str  # "EQUIV" (paráfrase que preserva os fatos) ou "ERRO" (erro injetado)
    rule: str

    @property
    def positive(self) -> int:
        return 1 if self.label == "EQUIV" else 0


def load_pairs(pairs_path: Path = PAIRS, golden_path: Path = GOLDEN) -> list[LabelledPair]:
    """Os pares que a calibração do O1 usou: os rotulados, menos os excluídos na conferência visual.

    Mesmo filtro de tests/mutation/calibration/calibrate.py (sai quem tem
    `excluded`), para a AUC daqui ser lida ao lado da do tau.json sobre o MESMO
    conjunto: 247 pares, 111 EQUIV e 136 ERRO.
    """
    golden = load_golden(golden_path)
    pairs = []
    for raw in read_jsonl(pairs_path):
        if raw.get("excluded"):
            continue
        pairs.append(
            LabelledPair(
                pair_id=raw["pair_id"],
                case_id=raw["case_id"],
                question=golden[raw["case_id"]]["question"],
                reference=raw["reference"],
                candidate=raw["candidate"],
                label=raw["label"],
                rule=raw.get("rule", "?"),
            )
        )
    return pairs


# ----------------------------------------------------------- log da campanha


def hit_generation_cap(run: dict[str, Any], cap: int | None) -> bool:
    """Execução cortada pelo teto de geração (cópia de tests/mutation/evaluate.py, _hit_generation_cap)."""
    if cap is None or run.get("error"):
        return False
    tokens_out = run.get("tokens_out")
    return not (run.get("answer") or "").strip() and tokens_out is not None and tokens_out >= int(cap)


@dataclass(frozen=True)
class ModalPair:
    mutant_id: str
    case_id: str
    operator: str | None
    answer: str
    runs: int


def modal_pairs(runs: Iterable[dict[str, Any]], max_repetitions: int, cap: int | None) -> list[ModalPair]:
    """Uma resposta modal por (mutante, caso), com as execuções válidas até o nível pedido."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["mutant_id"] in NOT_MUTANTS:
            continue
        if run["repetition"] > max_repetitions or run.get("error") or hit_generation_cap(run, cap):
            continue
        grouped[(run["mutant_id"], run["case_id"])].append(run)
    out = []
    for (mutant_id, case_id), pair_runs in sorted(grouped.items()):
        pair_runs.sort(key=lambda r: r["repetition"])
        answers = [r["answer"] for r in pair_runs]
        out.append(
            ModalPair(
                mutant_id=mutant_id,
                case_id=case_id,
                operator=next((r["operator"] for r in pair_runs if r.get("operator")), None),
                answer=modal_answer(answers),
                runs=len(pair_runs),
            )
        )
    return out


# --------------------------------------------------------- S, mortes e escore


def last_records(records: Iterable[dict[str, Any]], oracle: str, level: str) -> dict[tuple[str, str], dict]:
    """O último registro de cada (mutante, caso): o verdicts.jsonl só acrescenta, e --force regrava por cima.

    O verdicts.jsonl da campanha não grava se o oráculo conseguiu medir
    (`evaluable_oracle` fica só na memória do evaluate.py). O sinal que sobra é
    a nota: o O4 sem juiz devolve score nulo (o4_judge.py). Daí o padrão abaixo.
    """
    out: dict[tuple[str, str], dict] = {}
    for record in records:
        if record.get("oracle") == oracle and record.get("level", "L2") == level:
            row = dict(record)
            row.setdefault("evaluable_oracle", row.get("score") is not None)
            out[(row["mutant_id"], row["case_id"])] = row
    return out


def evaluable_set(verdicts: dict[tuple[str, str], dict], requires_pass: bool = True) -> set[str]:
    """S: casos com baseline medido (e, por padrão, aprovado)."""
    cases = set()
    for (mutant_id, case_id), v in verdicts.items():
        if mutant_id != BASELINE_ID or not v.get("evaluable_oracle", True):
            continue
        if not requires_pass or v["verdict"] == "pass":
            cases.add(case_id)
    return cases


def mark_kills(verdicts: dict[tuple[str, str], dict], s: set[str]) -> None:
    """Preenche `evaluable` e `kill` em cada registro, com a regra do evaluate.py."""
    for (mutant_id, case_id), v in verdicts.items():
        v["evaluable"] = case_id in s
        v["kill"] = bool(
            mutant_id != BASELINE_ID and v["verdict"] == "fail" and v.get("evaluable_oracle", True) and case_id in s
        )


def mutation_score(verdicts: dict[tuple[str, str], dict], s: set[str]) -> dict[str, Any]:
    """Escore de mutação do oráculo: mutantes com ao menos um caso de S reprovado / todos os mutantes.

    O denominador é o de tests/mutation/analyze.py: todo mutante julgado, mesmo
    o que não tem caso em S (esse conta como sobrevivente).
    """
    by_mutant: dict[str, list[dict]] = {}
    for (mutant_id, case_id), v in verdicts.items():
        if mutant_id != BASELINE_ID:
            by_mutant.setdefault(mutant_id, [])
            if case_id in s:
                by_mutant[mutant_id].append(v)
    operators = {}
    for mutant_id, rows in sorted(by_mutant.items()):
        failing = [r for r in rows if r["verdict"] == "fail" and r.get("evaluable_oracle", True)]
        operators[mutant_id] = {
            "evaluable_cases": len(rows),
            "failing_cases": len(failing),
            "kill_rate": round(len(failing) / len(rows), 4) if rows else None,
            "killed": bool(failing),
        }
    killed = sum(1 for o in operators.values() if o["killed"])
    interval = wilson_ci(killed, len(operators))
    return {
        "s_size": len(s),
        "mutants": len(operators),
        "killed": killed,
        "ms": interval.as_list(),
        "operators": operators,
    }


def agreement(a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict], s: set[str]) -> dict[str, Any]:
    """Concordância de veredito entre dois oráculos nos pares (mutante, caso) que os dois mediram em S."""
    keys = sorted(
        k for k in set(a) & set(b)
        if k[1] in s and a[k].get("evaluable_oracle", True) and b[k].get("evaluable_oracle", True)
    )
    va = [a[k]["verdict"] for k in keys]
    vb = [b[k]["verdict"] for k in keys]
    kappa = cohen_kappa(va, vb)
    return {
        "pairs": len(keys),
        "agree": sum(1 for x, y in zip(va, vb) if x == y),
        "kappa": None if kappa != kappa else round(kappa, 4),  # NaN -> None
        "kappa_label": interpret_kappa(kappa),
        "confusion": dict(Counter(f"{x}/{y}" for x, y in zip(va, vb))),
        "disagreements": [k for k, x, y in zip(keys, va, vb) if x != y],
    }
