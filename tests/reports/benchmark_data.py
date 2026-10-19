"""Os dados do relatório do benchmark: rodadas, placar, medidas por pergunta, análises e formatação.

Só lê o que o runner do benchmark (tests/benchmark/run_experiment.py) gravou em tests/benchmark/out/<rodada>/:
`cartao.yaml`, `placar.json`, `medidas.json`, `execucao.json`, `parametros.json` e os `c1_<braço>.json`. Usa os
módulos de biblioteca padrão do benchmark (placar, e5, c1) para as contas que cruzam rodadas. Não importa o
backend, não chama o Ollama e não refaz nenhum julgamento.

Números saem no padrão brasileiro (vírgula decimal, ponto de milhar, sinal de menos tipográfico).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark import c1, e5, placar  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "tests" / "benchmark" / "out"
MANIFESTS = PROJECT_ROOT / "data" / "vector_store" / "manifests"
MINUS = "−"

# O nome de cada métrica no relatório: o curto vai nos eixos e tabelas; o longo, na primeira menção.
METRIC_NAMES = {
    "c1.evidence_at_k": "evidência nos trechos",
    "c1.hit_at_k": "documento certo nos trechos",
    "c1.hit_at_1": "documento certo em 1º",
    "c1.hit_at_10": "documento certo nos 10 primeiros",
    "c1.mrr_at_10": "MRR@10",
    "c1.precision_at_k": "precisão dos trechos",
    "c1.coverage_at_k": "cobertura da evidência",
    "c1.coverage_at_10": "cobertura da evidência nos 10",
    "c1.retrieval_ms": "latência da busca",
    "c1.context_tokens": "tokens de contexto",
    "c1.correct_abstention": "de fora barradas pelo limiar",
    "c1.false_block": "de dentro barradas pelo limiar",
    "c2.faithfulness": "fidelidade",
    "c2.faithfulness_answered": "fidelidade (só respostas)",
    "c2.answer_relevancy": "relevância da resposta",
    "c2.answer_relevancy_answered": "relevância (só respostas)",
    "c2.context_precision": "precisão do contexto",
    "c2.context_recall": "revocação do contexto",
    "c2.context_relevance": "relevância do contexto",
    "abstention.correct_abstention": "abstenção correta",
    "abstention.undue_abstention": "recusa indevida",
    "abstention.with_evidence": "recusa com a evidência no prompt",
    "abstention.without_evidence": "recusa sem a evidência no prompt",
    "attribution.misattribution": "atribuição errada",
    "attribution.no_citation": "resposta sem citação",
    "attribution.cites_source": "cita a fonte certa",
    "cost.generation_ms": "latência da geração",
    "cost.retrieval_ms": "latência da busca",
    "cost.prompt_tokens": "tokens do prompt",
    "cost.completion_tokens": "tokens da resposta",
}
VERDICT_LABELS = ("melhor", "pior", "inconclusivo", "sem diferença relevante", "não medido", "sem margem do E0")


class ReportError(Exception):
    """Erro no manifesto ou num texto do relatório (rodada que não existe, figura sem tipo...)."""


# ------------------------------------------------------------------ formatação


def metric_name(metric: str) -> str:
    return METRIC_NAMES.get(metric, metric.split(".", 1)[-1].replace("_", " "))


def unit(metric: str) -> str:
    """'ms', 'tokens' ou 'nota' (taxa ou nota de 0 a 1)."""
    name = metric.split(".", 1)[-1]
    if name.endswith("_ms"):
        return "ms"
    if name.endswith("_tokens"):
        return "tokens"
    return "nota"


def num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{abs(value):,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return f"{MINUS}{text}" if value < 0 and text.strip("0,.") else text


def signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = num(value, digits)
    return text if text.startswith(MINUS) else f"+{text}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{num(100 * value, 0)}%"


def seconds(ms: float | None, digits: int = 1) -> str:
    return "—" if ms is None else f"{num(ms / 1000, digits)} s"


def fmt_value(metric: str, value: float | None, spec: str | None = None) -> str:
    """O valor no formato da métrica, ou no pedido: pct, int, s, ms ou o número de casas."""
    if value is None:
        return "—"
    if spec == "pct":
        return pct(value)
    if spec == "int":
        return num(value, 0)
    if spec == "s":
        return seconds(value)
    if spec == "ms":
        return f"{num(value, 0)} ms"
    if spec and spec.isdigit():
        return num(value, int(spec))
    if unit(metric) == "ms":
        return seconds(value) if abs(value) >= 1000 else f"{num(value, 0)} ms"
    if unit(metric) == "tokens":
        return num(value, 0)
    return num(value, 2)


def fmt_delta(metric: str, delta: dict | None, spec: str | None = None) -> str:
    """"+0,37 [+0,20; +0,53]" na unidade da métrica (latência em ms abaixo de 1 s, em segundos acima)."""
    if not delta:
        return "—"
    values = (delta["delta"], delta["ci_low"], delta["ci_high"])
    if unit(metric) == "ms":
        if max(abs(v) for v in values) < 1000:
            point, low, high = (signed(v, 0) for v in values)
            return f"{point} ms [{low}; {high}]"
        point, low, high = (signed(v / 1000, 1) for v in values)
        return f"{point} s [{low}; {high}]"
    digits = 0 if unit(metric) == "tokens" else int(spec) if spec and spec.isdigit() else 2
    point, low, high = (signed(v, digits) for v in values)
    return f"{point} [{low}; {high}]"


def fmt_point(metric: str, delta: dict | None) -> str:
    if not delta:
        return "—"
    if unit(metric) == "ms":
        if abs(delta["delta"]) < 1000:
            return f"{signed(delta['delta'], 0)} ms"
        return f"{signed(delta['delta'] / 1000, 1)} s"
    return signed(delta["delta"], 0 if unit(metric) == "tokens" else 2)


def fmt_margin(metric: str, margin: float | None) -> str:
    if margin is None:
        return "—"
    if unit(metric) == "ms":
        return f"{num(margin, 0)} ms" if margin < 1000 else seconds(margin, 2)
    return num(margin, 0) if unit(metric) == "tokens" else num(margin, 3)


# ------------------------------------------------------------------ rodadas


@dataclass
class Run:
    """Uma rodada do runner, pelo apelido que o texto usa."""

    alias: str
    path: Path
    card: dict
    card_text: str
    placar: dict
    medidas: dict | None
    execucao: dict | None
    params: dict = field(default_factory=dict)
    _e0_margins: dict | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def arms(self) -> list[str]:
        return list(self.placar.get("means", {}))

    def mean(self, arm: str, metric: str) -> float | None:
        return self.placar.get("means", {}).get(arm, {}).get(metric)

    def values(self, arm: str, metric: str) -> dict[str, float]:
        if not self.medidas:
            return {}
        return self.medidas.get(arm, {}).get("metrics", {}).get(metric, {})

    def clusters(self, *arms: str) -> dict[str, str]:
        merged: dict[str, str] = {}
        for arm in arms:
            merged.update((self.medidas or {}).get(arm, {}).get("clusters", {}))
        return merged

    def count(self, arm: str, metric: str) -> tuple[int, int] | None:
        """(acertos, perguntas) de uma métrica de 0 ou 1: 19 de 30."""
        values = self.values(arm, metric)
        if not values:
            return None
        return int(round(sum(values.values()))), len(values)

    def pair(self, pair: str, metric: str) -> dict | None:
        """O Δ do par no placar ou, se o par não está no cartão, calculado das medidas por pergunta."""
        delta = self.placar.get("pairs", {}).get(pair, {}).get(metric)
        if delta is not None:
            return delta
        a, b = split_pair(pair)
        if self.values(a, metric) and self.values(b, metric):
            return placar.paired_delta(self.values(a, metric), self.values(b, metric), self.clusters(a, b))
        return None

    def verdict(self, pair: str) -> dict | None:
        return (self.placar.get("verdicts") or {}).get(pair)

    def margin(self, metric: str) -> tuple[float | None, str]:
        """A margem de ruído da métrica: a do A/A do próprio cartão, se ele mediu; senão, a do E0 que o placar usou."""
        for values in (self.placar.get("margins") or {}).values():
            if metric in values:
                return values[metric], "cartão"
        e0 = self.e0_margins()
        for values in e0.values():
            if metric in values:
                return values[metric], "E0"
        return None, "—"

    def e0_margins(self) -> dict:
        if self._e0_margins is None:
            source = self.placar.get("margin_source")
            path = PROJECT_ROOT / Path(str(source).replace("\\", "/")) if source else None
            self._e0_margins = (json.loads(path.read_text(encoding="utf-8")).get("margins", {})
                                if path and path.is_file() else {})
        return self._e0_margins

    def judged(self, pair: str, metric: str) -> tuple[dict | None, float | None, str]:
        """(Δ, margem, veredito) de qualquer métrica do par, pela regra do placar."""
        delta = self.pair(pair, metric)
        given = self.verdict(pair)
        if given and given.get("metric") == metric:
            return delta, given.get("margin"), given["verdict"]
        margin, _ = self.margin(metric)
        return delta, margin, placar.verdict(delta, margin, placar.higher_is_better(metric))

    def started_at(self) -> str | None:
        runs = (self.execucao or {}).get("runs") or []
        first = next((r for r in runs if not r.get("so_placar")), runs[0] if runs else None)
        return first.get("started_at") if first else None

    def duration_seconds(self) -> float:
        runs = (self.execucao or {}).get("runs") or []
        return sum(r.get("duration_seconds") or 0 for r in runs if not r.get("so_placar"))

    def judge(self) -> str | None:
        for summary in (self.placar.get("summary") or {}).values():
            judge = (summary or {}).get("judge") or {}
            if judge.get("judge_model"):
                return judge["judge_model"]
        return None

    def c1_diagnostic(self, arm: str) -> dict:
        path = self.path / f"c1_{arm}.json"
        if not path.is_file():
            raise ReportError(f"{self.name}: não há {path.name} (o braço {arm} rodou a C1?)")
        return json.loads(path.read_text(encoding="utf-8"))

    def arm_spec(self, arm: str) -> dict:
        return dict((self.card.get("bracos") or {}).get(arm) or {})


def split_pair(pair: str) -> tuple[str, str]:
    for sep in ("→", "->"):
        if sep in pair:
            a, b = pair.split(sep, 1)
            return a.strip(), b.strip()
    raise ReportError(f"par {pair!r} sem seta (escreva A→B)")


def load_run(alias: str, name: str, out_root: Path = OUT_ROOT) -> Run:
    path = out_root / name
    placar_path = path / "placar.json"
    if not placar_path.is_file():
        raise ReportError(f"a rodada {name} não tem placar.json em {path}")
    card_text = (path / "cartao.yaml").read_text(encoding="utf-8") if (path / "cartao.yaml").is_file() else ""

    def read(file: str) -> dict | None:
        p = path / file
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    return Run(alias=alias, path=path, card=yaml.safe_load(card_text) or {}, card_text=card_text,
               placar=json.loads(placar_path.read_text(encoding="utf-8")), medidas=read("medidas.json"),
               execucao=read("execucao.json"), params=read("parametros.json") or {})


# ------------------------------------------------------------------ análises


def e5_analysis(run: Run, arm: str, manifests: Path = MANIFESTS) -> dict:
    """A concentração das fontes (E5) na C1 de um braço, como o `tests.benchmark.e5` calcula."""
    spec = run.arm_spec(arm)
    diagnostic = run.c1_diagnostic(arm)
    index, lang = spec.get("indice"), spec.get("lingua", "pt")
    arm_info = diagnostic["c1"]["arms"][index]
    manifest_path = manifests / f"{arm_info['collection']}.json"
    if not manifest_path.is_file():
        raise ReportError(f"o E5 precisa do manifesto da coleção, {manifest_path}, que não está nesta máquina")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_counts = {c1.doc_stem(d["file"]): d["chunks"] for d in manifest["documents"]}
    questions = e5.questions_from_diagnostic(diagnostic, index, lang)
    result = e5.analyze(questions, manifest["n_documents"], chunk_counts, arm_info["retrieval"]["top_k"])
    result.update(index=index, lang=lang, arm=arm, run=run.name)
    return result


def comparison(runs: dict[str, Run], de: str, para: str, metrics: list[str]) -> dict:
    """Δ pareado entre braços de rodadas diferentes ("e1 bgem3-pt" → "e3 base"), das medidas por pergunta.

    A margem é a da rodada de destino. Só vale quando as duas rodadas usam os mesmos goldens e o mesmo índice
    (quem escreve o texto garante; o relatório mostra de onde veio cada lado).
    """
    (ra, arm_a), (rb, arm_b) = (_run_arm(runs, side) for side in (de, para))
    clusters = {**ra.clusters(arm_a), **rb.clusters(arm_b)}
    result = {}
    for metric in metrics:
        a, b = ra.values(arm_a, metric), rb.values(arm_b, metric)
        delta = placar.paired_delta(a, b, clusters) if a and b else None
        margin, _ = rb.margin(metric)
        result[metric] = {"delta": delta, "margin": margin,
                          "verdict": placar.verdict(delta, margin, placar.higher_is_better(metric)),
                          "de": ra.mean(arm_a, metric), "para": rb.mean(arm_b, metric)}
    return {"de": de, "para": para, "metrics": result}


def _run_arm(runs: dict[str, Run], side: str) -> tuple[Run, str]:
    parts = side.split()
    if len(parts) != 2 or parts[0] not in runs:
        raise ReportError(f"{side!r}: escreva 'apelido braço', com um apelido de `rodadas`")
    return runs[parts[0]], parts[1]


def changed_answers(run: Run, arm_a: str, arm_b: str) -> dict:
    """Quantas respostas mudam de um braço para o outro (o ruído do gerador: o mesmo braço em outra ordem)."""

    def answers(arm: str) -> dict[str, str]:
        path = run.path / f"respostas_{arm}.json"
        if not path.is_file():
            raise ReportError(f"{run.name}: não há {path.name}")
        return {a["name"]: a.get("actual_output") for a in json.loads(path.read_text(encoding="utf-8"))["answers"]}

    a, b = answers(arm_a), answers(arm_b)
    common = sorted(set(a) & set(b))
    return {"changed": sum(a[n] != b[n] for n in common), "total": len(common)}


def roc_points(positives: list[float], negatives: list[float]) -> list[tuple[float, float]]:
    """(falso positivo, verdadeiro positivo) para cada corte, do mais alto ao mais baixo."""
    cuts = sorted(set(positives) | set(negatives), reverse=True)
    points = [(0.0, 0.0)]
    for cut in cuts:
        tpr = sum(p >= cut for p in positives) / len(positives)
        fpr = sum(n >= cut for n in negatives) / len(negatives)
        points.append((fpr, tpr))
    return points


def outside_separation(run: Run, arm: str, score: str) -> dict:
    """A nota do primeiro trecho (cosseno ou reranker) nas de dentro e nas de fora: curva ROC e AUC."""
    spec = run.arm_spec(arm)
    index, lang = spec.get("indice"), spec.get("lingua", "pt")
    diagnostic = run.c1_diagnostic(arm)

    def scores(rows: list[dict], inside: bool) -> list[float]:
        found = []
        for row in rows:
            metrics = row["c1"].get(index, {}).get(lang) or {}
            if inside and not metrics.get("in_index"):
                continue
            if metrics.get(score) is not None:
                found.append(float(metrics[score]))
        return found

    positives, negatives = scores(diagnostic["rows"], True), scores(diagnostic.get("out_rows", []), False)
    return {"points": roc_points(positives, negatives), "auc": c1.auc(positives, negatives),
            "n_inside": len(positives), "n_outside": len(negatives)}
