"""Leitura dos resultados das suítes de qualidade e as contas do relatório.

Cada framework grava seu JSON num formato próprio (tests/ragas/run_and_export.py
e tests/deepeval/run_and_export.py). Aqui os dois viram o mesmo modelo — Run,
Case, Score — indexado pelas cinco DIMENSÕES que as duas suítes medem. É isso
que permite colocar RAGAS e DeepEval lado a lado.

Nada aqui importa ragas, deepeval ou o backend: o gerador roda em qualquer
máquina que tenha os JSONs.
"""
from __future__ import annotations

import ast
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Onde cada suíte grava seus resultados (run_and_export.py de cada uma).
RESULTS_DIRS = {
    "DeepEval": PROJECT_ROOT / "tests" / "deepeval" / "results",
    "RAGAS": PROJECT_ROOT / "tests" / "ragas" / "results",
}
_RESULT_FILE = re.compile(r"^\d{8}T\d{6}Z\.json$")


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    short: str
    question: str


# A ordem é a das duas suítes (tests/ragas/README.md, tests/deepeval/README.md).
DIMENSIONS = [
    Dimension("faithfulness", "Fidelidade ao contexto", "Fidelidade", "Cada afirmação da resposta se apoia nos trechos recuperados?"),
    Dimension("answer_relevancy", "Relevância da resposta", "Rel. resposta", "A resposta responde à pergunta feita?"),
    Dimension("context_precision", "Precisão do contexto", "Precisão ctx", "Os trechos úteis vêm primeiro no ranking?"),
    Dimension("context_recall", "Recall do contexto", "Recall ctx", "A recuperação trouxe o que a resposta de referência precisa?"),
    Dimension("context_relevance", "Relevância do contexto", "Rel. contexto", "Os trechos recuperados têm a ver com a pergunta?"),
]
DIMENSION_BY_KEY = {d.key: d for d in DIMENSIONS}

# Nome da métrica no JSON de cada framework -> dimensão.
_METRIC_TO_DIMENSION = {
    "RAGAS": {d.key: d.key for d in DIMENSIONS},
    "DeepEval": {
        "Faithfulness": "faithfulness",
        "Answer Relevancy": "answer_relevancy",
        "Contextual Precision": "context_precision",
        "Contextual Recall": "context_recall",
        "Contextual Relevancy": "context_relevance",
    },
}
# Nome da métrica no código de cada framework, para o leitor achar a definição.
NATIVE_METRIC = {
    "RAGAS": {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
        "context_precision": "ContextPrecisionWithReference",
        "context_recall": "ContextRecall",
        "context_relevance": "ContextRelevance",
    },
    "DeepEval": {
        "faithfulness": "FaithfulnessMetric",
        "answer_relevancy": "AnswerRelevancyMetric",
        "context_precision": "ContextualPrecisionMetric",
        "context_recall": "ContextualRecallMetric",
        "context_relevance": "ContextualRelevancyMetric",
    },
}


def _read_not_found_marker() -> str | None:
    """Frase de abstenção do assistente, lida da fonte (sem importar o backend).

    O valor mora em backend/app/rag/generation.py::NOT_FOUND_MARKER; copiar a
    string para cá deixaria as duas divergirem na primeira mudança de prompt.
    """
    source = PROJECT_ROOT / "backend" / "app" / "rag" / "generation.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "NOT_FOUND_MARKER" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


NOT_FOUND_MARKER = _read_not_found_marker()


# ------------------------------------------------------------------ modelo


@dataclass
class Score:
    value: float | None
    passed: bool | None
    reason: str | None = None
    error: str | None = None

    @property
    def measured(self) -> bool:
        return self.value is not None


@dataclass
class Case:
    name: str
    question: str
    expected: str
    answer: str
    contexts: list[str]
    scores: dict[str, Score]

    @property
    def abstained(self) -> bool:
        """O SUT respondeu com a frase de abstenção ("não encontrei...")."""
        return bool(NOT_FOUND_MARKER) and NOT_FOUND_MARKER.lower() in self.answer.lower()

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.scores.values()) and len(self.scores) == len(DIMENSIONS)

    def score(self, dimension: str) -> Score:
        return self.scores.get(dimension) or Score(value=None, passed=None, error="métrica ausente no resultado")


@dataclass
class Run:
    framework: str
    source: Path
    generated_at: datetime | None
    generation_model: str | None
    judge_model: str | None
    embedding_model: str | None
    threshold: float
    n_cases_planned: int
    cases: list[Case]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def when(self) -> str:
        return self.generated_at.astimezone().strftime("%d/%m/%Y %H:%M") if self.generated_at else "data desconhecida"

    @property
    def label(self) -> str:
        return f"{self.framework} ({self.when})"

    @property
    def complete(self) -> bool:
        return len(self.cases) >= self.n_cases_planned

    def case(self, name: str) -> Case | None:
        return next((c for c in self.cases if c.name == name), None)


# ------------------------------------------------------------------ leitura


def detect_framework(data: dict[str, Any]) -> str:
    if data.get("framework") == "ragas":
        return "RAGAS"
    names = {m.get("name") for c in data.get("cases", []) for m in c.get("metrics", [])}
    if names & set(_METRIC_TO_DIMENSION["DeepEval"]):
        return "DeepEval"
    raise ValueError("formato de resultado não reconhecido (nem RAGAS nem DeepEval)")


def _parse_when(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def load_run(path: Path) -> Run:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    framework = detect_framework(data)
    threshold = float(data.get("threshold", 0.7))
    to_dimension = _METRIC_TO_DIMENSION[framework]

    cases = []
    for raw in data.get("cases", []):
        scores: dict[str, Score] = {}
        for metric in raw.get("metrics", []):
            dimension = to_dimension.get(metric.get("name"))
            if dimension is None:
                continue
            value = metric.get("score")
            value = None if value is None else float(value)
            passed = metric.get("success")
            if passed is None and value is not None:
                passed = value >= threshold
            scores[dimension] = Score(value=value, passed=passed, reason=metric.get("reason"), error=metric.get("error"))
        cases.append(
            Case(
                name=raw.get("name") or raw.get("input", "")[:40],
                question=raw.get("input", ""),
                expected=raw.get("expected_output") or "",
                answer=raw.get("actual_output") or "",
                contexts=list(raw.get("retrieval_context") or []),
                scores=scores,
            )
        )

    known = {"framework", "generated_at", "generation_model", "judge_model", "embedding_model", "threshold", "n_cases", "cases"}
    extra = {k: v for k, v in data.items() if k not in known and not k.startswith("metrics_")}
    return Run(
        framework=framework,
        source=Path(path),
        generated_at=_parse_when(data.get("generated_at")),
        generation_model=data.get("generation_model"),
        judge_model=data.get("judge_model"),
        embedding_model=data.get("embedding_model"),
        threshold=threshold,
        n_cases_planned=int(data.get("n_cases") or len(cases)),
        cases=cases,
        extra=extra,
    )


def latest_result(directory: Path) -> Path | None:
    """O JSON mais recente de uma pasta results/ (nomes <AAAAMMDD>T<HHMMSS>Z.json)."""
    files = sorted(p for p in Path(directory).glob("*.json") if _RESULT_FILE.match(p.name))
    return files[-1] if files else None


def discover_latest() -> list[Path]:
    """O resultado mais recente de cada framework que tiver algum."""
    return [path for directory in RESULTS_DIRS.values() if (path := latest_result(directory))]


# ------------------------------------------------------------------- contas


@dataclass
class DimensionStats:
    dimension: str
    n_cases: int
    n_measured: int
    n_passed: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    errors: list[tuple[str, str]]

    @property
    def pass_rate(self) -> float | None:
        return self.n_passed / self.n_cases if self.n_cases else None


def dimension_stats(run: Run) -> dict[str, DimensionStats]:
    stats = {}
    for dim in DIMENSIONS:
        scores = [c.score(dim.key) for c in run.cases]
        values = [s.value for s in scores if s.measured]
        stats[dim.key] = DimensionStats(
            dimension=dim.key,
            n_cases=len(run.cases),
            n_measured=len(values),
            n_passed=sum(1 for s in scores if s.passed),
            mean=statistics.fmean(values) if values else None,
            median=statistics.median(values) if values else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            errors=[(c.name, c.score(dim.key).error or "não medida") for c in run.cases if not c.score(dim.key).measured],
        )
    return stats


@dataclass
class DimensionAgreement:
    dimension: str
    n_both: int
    n_agree: int
    mean_abs_diff: float | None

    @property
    def agreement(self) -> float | None:
        return self.n_agree / self.n_both if self.n_both else None


@dataclass
class Divergence:
    case: str
    dimension: str
    value_a: float
    value_b: float

    @property
    def diff(self) -> float:
        return self.value_b - self.value_a


@dataclass
class Comparison:
    """Duas rodadas lado a lado. `a` e `b` na ordem em que foram passadas."""

    a: Run
    b: Run
    common_cases: list[str]
    only_a: list[str]
    only_b: list[str]
    identical_answers: int
    identical_contexts: int
    per_dimension: dict[str, DimensionAgreement]
    divergences: list[Divergence]

    @property
    def warnings(self) -> list[str]:
        """O que impede ler a diferença entre as rodadas como diferença entre os métodos."""
        out = []
        if self.only_a or self.only_b:
            out.append(
                f"Os conjuntos de casos diferem ({len(self.common_cases)} em comum; "
                f"{len(self.only_a)} só em {self.a.label}, {len(self.only_b)} só em {self.b.label})."
            )
        if self.a.generation_model != self.b.generation_model:
            out.append(
                f"Modelos de geração diferentes ({self.a.generation_model} x {self.b.generation_model}): "
                "as respostas avaliadas não vêm do mesmo sistema."
            )
        if self.a.judge_model != self.b.judge_model:
            out.append(
                f"Juízes diferentes ({self.a.judge_model} x {self.b.judge_model}): parte da diferença "
                "pode ser do juiz, não da métrica."
            )
        if self.a.threshold != self.b.threshold:
            out.append(f"Limiares diferentes ({self.a.threshold} x {self.b.threshold}).")
        if self.common_cases and self.identical_answers < len(self.common_cases):
            out.append(
                f"Só {self.identical_answers} de {len(self.common_cases)} respostas avaliadas são idênticas "
                "nas duas rodadas: a geração não é determinística, então cada framework julgou respostas "
                "diferentes para a mesma pergunta."
            )
        if self.common_cases and self.identical_contexts < len(self.common_cases):
            out.append(
                f"Os trechos recuperados são os mesmos em só {self.identical_contexts} de "
                f"{len(self.common_cases)} casos: a recuperação mudou entre as rodadas (índice "
                "reingerido, corpus ou configuração diferentes), e as métricas de contexto refletem isso."
            )
        return out


def compare(a: Run, b: Run) -> Comparison:
    names_a = [c.name for c in a.cases]
    names_b = {c.name for c in b.cases}
    common = [n for n in names_a if n in names_b]

    per_dimension = {}
    divergences = []
    for dim in DIMENSIONS:
        n_both = n_agree = 0
        diffs = []
        for name in common:
            sa, sb = a.case(name).score(dim.key), b.case(name).score(dim.key)
            if not (sa.measured and sb.measured):
                continue
            n_both += 1
            diffs.append(abs(sb.value - sa.value))
            if bool(sa.passed) == bool(sb.passed):
                n_agree += 1
            else:
                divergences.append(Divergence(name, dim.key, sa.value, sb.value))
        per_dimension[dim.key] = DimensionAgreement(
            dimension=dim.key,
            n_both=n_both,
            n_agree=n_agree,
            mean_abs_diff=statistics.fmean(diffs) if diffs else None,
        )

    divergences.sort(key=lambda d: abs(d.diff), reverse=True)
    return Comparison(
        a=a,
        b=b,
        common_cases=common,
        only_a=[n for n in names_a if n not in names_b],
        only_b=[c.name for c in b.cases if c.name not in set(names_a)],
        identical_answers=sum(1 for n in common if a.case(n).answer.strip() == b.case(n).answer.strip()),
        identical_contexts=sum(1 for n in common if a.case(n).contexts == b.case(n).contexts),
        per_dimension=per_dimension,
        divergences=divergences,
    )


def run_highlights(run: Run, stats: dict[str, DimensionStats]) -> list[str]:
    """Frases do resumo de uma rodada, derivadas só dos números."""
    out = []
    n = len(run.cases)
    passed_all = sum(1 for c in run.cases if c.all_passed)
    out.append(f"{passed_all} de {n} casos passam em todas as cinco dimensões (limiar {fmt(run.threshold)}).")

    ranked = [s for s in stats.values() if s.mean is not None]
    if ranked:
        best = max(ranked, key=lambda s: s.mean)
        worst = min(ranked, key=lambda s: s.mean)
        out.append(
            f"Melhor dimensão: {DIMENSION_BY_KEY[best.dimension].label} (média {fmt(best.mean)}, "
            f"{best.n_passed}/{n} acima do limiar). Pior: {DIMENSION_BY_KEY[worst.dimension].label} "
            f"(média {fmt(worst.mean)}, {worst.n_passed}/{n})."
        )

    unmeasured = sum(len(s.errors) for s in stats.values())
    if unmeasured:
        out.append(f"{unmeasured} nota(s) não foram medidas (erro do juiz, timeout ou NaN).")
    abstained = [c.name for c in run.cases if c.abstained]
    if abstained:
        out.append(
            f"O assistente se absteve em {len(abstained)} caso(s) ({', '.join(abstained)}): nesses, "
            "notas de fidelidade e relevância da resposta julgam uma recusa, não uma resposta."
        )
    if not run.complete:
        out.append(f"Rodada incompleta: {len(run.cases)} de {run.n_cases_planned} casos.")
    return out


def fmt(value: float | None, digits: int = 2) -> str:
    """Número no formato do texto do relatório (vírgula decimal); '—' quando não há."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value: float | None, digits: int = 2) -> str:
    """Diferença com sinal explícito e o menos tipográfico (−), não o hífen."""
    if value is None:
        return "—"
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return sign + fmt(abs(value), digits)


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "—"
