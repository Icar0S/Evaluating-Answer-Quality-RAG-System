"""Runner do benchmark: roda um experimento descrito num cartão (tests/benchmark/cartoes/<id>.yaml).

Provisório: a §9 do plano-benchmark.md, que especifica o runner e o YAML, não
chegou. O que está aqui segue as §4, §5 e §8 e o tests/compare_indices.py, que o
plano dá como modelo; o formato do cartão está em docs/benchmark-status.md.

O funil (§5), nesta ordem:

 1. o cartão, como foi escrito, no terminal e em <saída>/cartao.yaml;
 2. índices: cada versão dos braços é ingerida se estiver vazia;
 3. C1 (só embeddings) nos braços com a camada c1: tests/retrieval_diagnostic.py;
 4. C2 nos braços com a camada c2: tests/generate_answers.py com os fixos do
    §8.2 e as sobrescritas do braço, e o RAGAS julgando só as perguntas de
    dentro do índice, com o juiz e a janela do cartão. O processo do juiz não
    herda o índice nem o embedding do braço (§4, princípio 8). Braço
    `rejulgar` julga de novo as respostas de outro braço, em outra ordem;
 5. o placar: as métricas de cada braço, os pares com Δ e IC por bootstrap de
    documentos (tests/benchmark/placar.py), a margem de ruído (E0) e o
    veredito provisório da métrica primária.

Cada passo grava o seu arquivo em <saída> e é pulado se o arquivo já está
completo: uma rodada de horas se retoma com --retomar <saída>.

Uso, da raiz (venv do backend, que tem o PyYAML):

    backend\\.venv\\Scripts\\python.exe tests\\benchmark\\run_experiment.py tests\\benchmark\\cartoes\\E0.yaml
    backend\\.venv\\Scripts\\python.exe tests\\benchmark\\run_experiment.py tests\\benchmark\\cartoes\\E0.yaml --limite 3
    backend\\.venv\\Scripts\\python.exe tests\\benchmark\\run_experiment.py tests\\benchmark\\cartoes\\E0.yaml \\
        --retomar tests\\benchmark\\out\\E0_20261018T120000
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "backend"))
import run_all as ra  # noqa: E402

from app.config import RAG_INDICES  # noqa: E402
from tests.benchmark import c1, placar  # noqa: E402

OUT_ROOT = ROOT / "tests" / "benchmark" / "out"
RAGAS_RESULTS = ROOT / "tests" / "ragas" / "results"
INDEX_VERSIONS = ROOT / "backend" / "app" / "rag" / "index_versions"
# O Ollama desta máquina só escuta em IPv4; "localhost" custa ~2 s por conexão (docs/benchmark-status.md).
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
REQUIRED = ("id", "titulo", "pergunta", "hipotese", "goldens", "bracos", "metrica_primaria", "didatica")
C1_METRICS = ("hit_at_1", "hit_at_k", "hit_at_10", "mrr_at_10", "precision_at_k", "coverage_at_k", "evidence_at_k",
              "coverage_at_10", "retrieval_ms")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance")
# Métricas em que menos é melhor (o resto: mais é melhor).
LOWER_IS_BETTER = {"undue_abstention", "generation_ms", "retrieval_ms", "prompt_tokens", "completion_tokens"}
# Os grupos do placar: uma métrica do cartão (guarda) se escreve grupo.metrica, como c1.evidence_at_k.
METRIC_GROUPS = ("c1", "c2", "abstention", "cost")


# ------------------------------------------------------------------ o cartão


def load_card(path: Path) -> dict:
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_card(card)
    if errors:
        raise SystemExit("cartão inválido:\n  " + "\n  ".join(errors))
    return card


def arms_of(card: dict) -> dict[str, dict]:
    """Os braços com os padrões preenchidos: rodada (índice, língua, camadas, ordem) ou rejulgar."""
    arms = {}
    for arm_id, spec in card["bracos"].items():
        spec = dict(spec or {})
        if "rejulgar" in spec:
            arms[arm_id] = {"id": arm_id, "kind": "rejulgar", "of": spec["rejulgar"],
                            "judge_order": spec.get("ordem_juiz"), "papel": spec.get("papel", "ruido")}
            continue
        arms[arm_id] = {
            "id": arm_id, "kind": "rodada", "index": spec["indice"], "lang": spec.get("lingua", "pt"),
            "layers": list(spec.get("camadas", ["c1", "c2"])), "order": spec.get("ordem"),
            "set": {str(k): v for k, v in (spec.get("set") or {}).items()}, "papel": spec.get("papel", "candidato"),
        }
    return arms


def validate_card(card: dict) -> list[str]:
    errors = [f"falta '{key}'" for key in REQUIRED if key not in (card or {})]
    if errors:
        return errors
    try:
        arms = arms_of(card)
    except KeyError as exc:
        return [f"braço sem {exc}"]
    for arm in arms.values():
        if arm["kind"] == "rejulgar" and arms.get(arm["of"], {}).get("kind") != "rodada":
            errors.append(f"{arm['id']}: rejulgar aponta para {arm['of']!r}, que não é uma rodada")
        if arm["kind"] == "rodada" and not set(arm["layers"]) <= {"c1", "c2"}:
            errors.append(f"{arm['id']}: camadas {arm['layers']} (só c1 e c2)")
        if arm["kind"] == "rodada" and not (INDEX_VERSIONS / f"{arm['index']}.sha256").is_file():
            errors.append(f"{arm['id']}: a versão {arm['index']!r} não tem lista de documentos registrada")
    if sum(arm["papel"] == "baseline" for arm in arms.values()) > 1:
        errors.append("mais de um braço com papel baseline")
    for pair in card.get("pares", []) + [p for group in (card.get("margem") or {}).values() for p in group]:
        if len(pair) != 2 or not set(pair) <= set(arms):
            errors.append(f"par {pair}: tem de ser dois braços do cartão")
    primary = card["metrica_primaria"]
    if primary.get("camada") not in ("c1", "c2") or not primary.get("metrica"):
        errors.append("metrica_primaria precisa de camada (c1 ou c2) e metrica")
    if card.get("teto") and card["teto"] not in arms:
        errors.append(f"teto {card['teto']!r}: não é um braço do cartão")
    for guard in card.get("guardas") or []:
        name = guard.get("nome", guard)
        if len(guard.get("par") or []) != 2 or not set(guard["par"]) <= set(arms):
            errors.append(f"guarda {name}: o par tem de ser dois braços do cartão")
        if str(guard.get("metrica", "")).split(".", 1)[0] not in METRIC_GROUPS or "." not in str(guard.get("metrica")):
            errors.append(f"guarda {name}: metrica no formato grupo.metrica ({', '.join(METRIC_GROUPS)})")
        if ("tolerancia" in guard) == ("tolerancia_perguntas" in guard):
            errors.append(f"guarda {name}: precisa de tolerancia ou de tolerancia_perguntas (uma das duas)")
    return errors


def print_card(card: dict) -> None:
    arms = arms_of(card)
    print(f"=== {card['id']} — {card['titulo']} ===")
    print(f"Pergunta: {card['pergunta']}")
    print(f"Hipótese: {' '.join(str(card['hipotese']).split())}")
    print("Braços:")
    for arm in arms.values():
        if arm["kind"] == "rejulgar":
            print(f"  {arm['id']:10s} rejulga as respostas de {arm['of']} (ordem do juiz: semente {arm['judge_order']})")
        else:
            order = "fixa" if arm["order"] is None else f"semente {arm['order']}"
            extra = f", {arm['set']}" if arm["set"] else ""
            print(f"  {arm['id']:10s} {arm['index']}, perguntas em {arm['lang']}, {'+'.join(arm['layers'])}, "
                  f"ordem {order}{extra} [{arm['papel']}]")
    primary = card["metrica_primaria"]
    print(f"Métrica primária: {primary['metrica']} ({primary['camada']})")
    for guard in card.get("guardas") or []:
        print(f"Guarda: {guard}")
    print(f"Didática: {' '.join(str(card['didatica']).split())}\n")


# ------------------------------------------------------------------ utilidades


def index_documents(index: str) -> set[str]:
    lines = (INDEX_VERSIONS / f"{index}.sha256").read_text(encoding="utf-8").splitlines()
    return {c1.doc_stem(line.split(None, 1)[1].strip()) for line in lines if line.strip() and not line.startswith("#")}


def child_env(card: dict, extra: dict | None = None) -> dict:
    # RAG_INDEX vazio: nenhum processo filho herda uma versão do shell ou do .env por engano.
    return {"OLLAMA_BASE_URL": card.get("ollama_base_url", DEFAULT_OLLAMA), "RAG_INDEX": "", **(extra or {})}


def set_args(values: dict) -> list[str]:
    return [item for key, value in values.items() for item in ("--set", f"{key}={value}")]


def warm_up(card: dict, model: str | None, embedding: bool = False) -> None:
    """Carrega o modelo no Ollama antes do passo medido.

    Sem isso, a primeira pergunta de cada braço paga o carregamento: na fumaça do E0, 6 s contra ~1,5 s das
    outras, porque o juiz (qwen3:8b) tira o gerador da memória entre um braço e outro.
    """
    if not model:
        return
    path, body = (("/api/embed", {"model": model, "input": "aquecimento"}) if embedding
                  else ("/api/generate", {"model": model, "prompt": "", "stream": False}))
    request = urllib.request.Request(card.get("ollama_base_url", DEFAULT_OLLAMA) + path,
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response.read()
    except OSError as exc:
        print(f"    [aviso] o aquecimento de {model} falhou: {exc}", flush=True)


def complete_answers(path: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("answers", [])) == data.get("n_cases")


def complete_judgement(path: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases_done") == data.get("n_cases")


class Experiment:
    def __init__(self, card: dict, card_path: Path, out: Path, limit: int | None) -> None:
        self.card, self.card_path, self.out, self.limit = card, card_path, out, limit
        self.arms = arms_of(card)
        self.steps: list[ra.Step] = []
        self.goldens_path = ROOT / card["goldens"]

    def record(self, key: str, name: str, status: str, detail: str, seconds: float | None = None,
               log: Path | None = None) -> None:
        step = ra.Step(key, name, status, detail, None, seconds, str(log.relative_to(ROOT)) if log else None)
        self.steps.append(step)
        took = f" ({seconds:.0f}s)" if seconds is not None else ""
        print(f"    {ra.STATUS_TAG.get(status, status)} {name}: {detail}{took}", flush=True)

    # 0. goldens (com --limite, um recorte pequeno dos de dentro e dos de fora)
    def prepare_goldens(self) -> None:
        if not self.limit:
            return
        goldens = json.loads(self.goldens_path.read_text(encoding="utf-8"))
        inside = [g for g in goldens if g.get("source_document")][: self.limit]
        outside = [g for g in goldens if g.get("question_class") in ("fora_do_corpus", "fora_de_escopo")][: self.limit]
        self.goldens_path = self.out / "goldens_limite.json"
        self.goldens_path.write_text(json.dumps(inside + outside, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. índices
    def ensure_indices(self) -> None:
        for index in sorted({a["index"] for a in self.arms.values() if a["kind"] == "rodada"}):
            log = self.out / "logs" / f"indice_{index}.log"
            code, secs = ra.run_logged(
                [ra.PY["backend"], ROOT / "scripts" / "ingest_documents.py", "--index", index, "--if-empty"],
                log, env=child_env(self.card), timeout=3600)
            status = "concluída" if code == 0 else "erro"
            self.record(f"indice_{index}", f"Índice {index}", status, "pronto" if code == 0 else "ver o log", secs, log)
            if code != 0:
                raise SystemExit(f"índice {index} não ficou pronto")

    # 2. C1
    def c1(self, arm: dict) -> None:
        target = self.out / f"c1_{arm['id']}.json"
        if target.is_file():
            self.record(f"c1_{arm['id']}", f"C1 {arm['id']}", "concluída", "já estava pronto (retomada)")
            return
        log = self.out / "logs" / f"c1_{arm['id']}.log"
        # Os fixos do cartão também vão para a C1: os de recuperação valem aqui, e todos ficam na proveniência.
        cmd = [ra.PY["backend"], ROOT / "tests" / "retrieval_diagnostic.py", "--out", target, "--indices", arm["index"],
               "--goldens", self.goldens_path, "--langs", arm["lang"],
               *set_args({**(self.card.get("fixos") or {}), **arm["set"]})]
        warm_up(self.card, RAG_INDICES[arm["index"]]["embedding_model"], embedding=True)
        code, secs = ra.run_logged(cmd, log, env=child_env(self.card), timeout=3 * 3600)
        if code != 0 or not target.is_file():
            self.record(f"c1_{arm['id']}", f"C1 {arm['id']}", "erro", "ver o log", secs, log)
            return
        summary = json.loads(target.read_text(encoding="utf-8"))["c1"]["summary"][arm["index"]][arm["lang"]]
        evidence = (summary.get("evidence_at_k") or {}).get("mean")
        self.record(f"c1_{arm['id']}", f"C1 {arm['id']}", "concluída",
                    f"evidência@k {evidence} em {summary['n_inside']} de dentro", secs, log)

    # 3. C2: respostas
    def answers(self, arm: dict) -> Path | None:
        target = self.out / f"respostas_{arm['id']}.json"
        if complete_answers(target):
            self.record(f"respostas_{arm['id']}", f"Respostas {arm['id']}", "concluída", "já estavam prontas (retomada)")
            return target
        log = self.out / "logs" / f"respostas_{arm['id']}.log"
        settings = {**(self.card.get("fixos") or {}), **arm["set"]}
        cmd = [ra.PY["backend"], ROOT / "tests" / "generate_answers.py", "--out", target, "--index", arm["index"],
               "--lang", arm["lang"], "--goldens", f"v2={self.goldens_path}", "--local", *set_args(settings)]
        if arm["order"] is not None:
            cmd += ["--shuffle-seed", arm["order"]]
        warm_up(self.card, RAG_INDICES[arm["index"]]["embedding_model"], embedding=True)
        warm_up(self.card, settings.get("generation_model"))
        code, secs = ra.run_logged(cmd, log, env=child_env(self.card), timeout=6 * 3600)
        if code != 0 or not complete_answers(target):
            self.record(f"respostas_{arm['id']}", f"Respostas {arm['id']}", "erro", "ver o log", secs, log)
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
        truncated = sum(1 for a in data["answers"] if a.get("truncated"))
        self.record(f"respostas_{arm['id']}", f"Respostas {arm['id']}", "concluída",
                    f"{len(data['answers'])} respostas, {truncated} truncada(s)", secs, log)
        return target

    def inside_answers(self, arm: dict, answers: Path, order_seed: int | None = None, name: str | None = None) -> Path:
        """Só as perguntas de dentro do índice vão ao juiz (§6, C2); com semente, em outra ordem."""
        data = json.loads(answers.read_text(encoding="utf-8"))
        documents = index_documents(arm["index"])
        inside = [a for a in data["answers"] if a.get("source_document") and c1.doc_stem(a["source_document"]) in documents]
        if order_seed is not None:
            random.Random(order_seed).shuffle(inside)
        target = self.out / f"respostas_{name or arm['id']}_dentro.json"
        target.write_text(json.dumps({**data, "answers": inside, "n_cases": len(inside)}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target

    # 4. C2: juiz
    def judge(self, arm_id: str, answers: Path) -> None:
        target = self.out / f"ragas_{arm_id}.json"
        if complete_judgement(target):
            self.record(f"ragas_{arm_id}", f"RAGAS {arm_id}", "concluída", "já estava pronto (retomada)")
            return
        judge = self.card.get("juiz") or {}
        env = child_env(self.card, {
            "RAGAS_JUDGE_MODEL": str(judge.get("modelo", "qwen3:8b")),
            "RAGAS_JUDGE_NUM_CTX": str(judge.get("num_ctx", 8192)),
        })
        log = self.out / "logs" / f"ragas_{arm_id}.log"
        before = ra.results_snapshot(RAGAS_RESULTS)
        code, secs = ra.run_logged([ra.PY["ragas"], ROOT / "tests" / "ragas" / "run_and_export.py", "--answers", answers],
                                   log, env=env, timeout=12 * 3600)
        new = sorted(ra.results_snapshot(RAGAS_RESULTS) - before)
        if code != 0 or not new:
            self.record(f"ragas_{arm_id}", f"RAGAS {arm_id}", "erro", f"saiu com código {code} (ver o log)", secs, log)
            return
        shutil.copy(RAGAS_RESULTS / new[-1], target)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.record(f"ragas_{arm_id}", f"RAGAS {arm_id}", "concluída",
                    f"{data['cases_done']}/{data['n_cases']} casos, {data.get('judge_failures', 0)} falha(s) do juiz",
                    secs, log)

    def run(self) -> None:
        self.prepare_goldens()
        self.ensure_indices()
        for arm in self.arms.values():
            if arm["kind"] == "rodada" and "c1" in arm["layers"]:
                self.c1(arm)
        for arm in self.arms.values():
            if arm["kind"] == "rodada" and "c2" in arm["layers"]:
                answers = self.answers(arm)
                if answers:
                    self.judge(arm["id"], self.inside_answers(arm, answers))
        for arm in self.arms.values():
            if arm["kind"] == "rejulgar":
                source = self.arms[arm["of"]]
                answers = self.out / f"respostas_{source['id']}.json"
                if not complete_answers(answers):
                    self.record(f"ragas_{arm['id']}", f"RAGAS {arm['id']}", "pulada", f"{source['id']} sem respostas")
                    continue
                self.judge(arm["id"], self.inside_answers(source, answers, arm["judge_order"], name=arm["id"]))


# ------------------------------------------------------------------ o placar


def translation_milliseconds(arm: dict, questions: dict[str, str]) -> dict[str, float]:
    """No braço com tradução gravada, o tempo de cada tradução por pergunta: a busca com cache não pagou."""
    cache = arm.get("set", {}).get("retrieval_translation_cache")
    if not cache or not str(arm.get("set", {}).get("retrieval_query_translation", "")).lower() in ("true", "1"):
        return {}
    path = ROOT / cache
    timings = json.loads(path.read_text(encoding="utf-8")).get("milliseconds", {}) if path.is_file() else {}
    return {name: timings[text] for name, text in questions.items() if text in timings}


def arm_measures(out: Path, arm: dict, arms: dict[str, dict], questions: dict[str, str] | None = None) -> dict:
    """Por pergunta, tudo o que o placar compara: C1 (na língua do braço), RAGAS, recusa e custo.

    `questions`: nome → pergunta em português, para somar o tempo da tradução gravada à latência da busca.
    """
    source = arms[arm["of"]] if arm["kind"] == "rejulgar" else arm
    measures: dict = {"clusters": {}, "c1": {}, "c2": {}, "abstention": {}, "cost": {}, "summary": {}}
    translation_ms = translation_milliseconds(arm, questions or {}) if arm.get("lang") == "pt" else {}
    diag_path = out / f"c1_{arm['id']}.json"
    if arm["kind"] == "rodada" and diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        summary = diag["c1"]["summary"][arm["index"]][arm["lang"]]
        measures["summary"]["c1"] = summary
        for row in diag["rows"]:
            metrics = row["c1"].get(arm["index"], {}).get(arm["lang"])
            if metrics and metrics.get("in_index"):
                measures["clusters"][row["name"]] = c1.doc_stem(row["source_document"])
                for metric in C1_METRICS:
                    if metrics.get(metric) is not None:
                        value = float(metrics[metric])
                        if metric == "retrieval_ms":
                            value += translation_ms.get(row["name"], 0.0)
                        measures["c1"].setdefault(metric, {})[row["name"]] = value
    answers_path = out / f"respostas_{source['id']}.json"
    if answers_path.is_file() and arm["kind"] == "rodada" and "c2" in arm["layers"]:
        documents = index_documents(source["index"])
        answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
        inside = {a["name"]: a for a in answers if a.get("source_document")
                  and c1.doc_stem(a["source_document"]) in documents and not a.get("error")}
        outside = {a["name"]: a for a in answers if not a.get("source_document") and not a.get("error")}
        for name, a in inside.items():
            measures["clusters"][name] = c1.doc_stem(a["source_document"])
        measures["abstention"] = {
            "correct_abstention": {n: float(a["abstained"]) for n, a in outside.items() if "abstained" in a},
            "undue_abstention": {n: float(a["abstained"]) for n, a in inside.items() if "abstained" in a},
        }
        for metric in ("retrieval_ms", "generation_ms", "prompt_tokens", "completion_tokens"):
            values = {n: float(a[metric]) for n, a in {**inside, **outside}.items() if a.get(metric) is not None}
            if metric == "retrieval_ms":
                values = {n: v + translation_ms.get(n, 0.0) for n, v in values.items()}
            measures["cost"][metric] = values
        measures["summary"]["truncated"] = sum(1 for a in answers if a.get("truncated"))
        measures["summary"]["errors"] = sum(1 for a in answers if a.get("error"))
    ragas_path = out / f"ragas_{arm['id']}.json"
    if ragas_path.is_file():
        ragas = json.loads(ragas_path.read_text(encoding="utf-8"))
        for case in ragas["cases"]:
            for m in case["metrics"]:
                if m.get("score") is not None:
                    measures["c2"].setdefault(m["name"], {})[case["name"]] = float(m["score"])
        measures["summary"]["judge"] = {k: ragas.get(k) for k in ("judge_model", "judge_num_ctx", "wall_seconds",
                                                                   "judge_calls", "judge_failures", "metrics_avg")}
    if arm["kind"] == "rejulgar":
        # Rejulgar não refaz recusa nem custo: as respostas são as do braço de origem.
        measures["abstention"], measures["cost"] = {}, {}
        if answers_path.is_file():
            for a in json.loads(answers_path.read_text(encoding="utf-8"))["answers"]:
                if a.get("source_document"):
                    measures["clusters"][a["name"]] = c1.doc_stem(a["source_document"])
    return measures


def _all_metrics(measures: dict) -> dict[str, dict[str, float]]:
    flat = {}
    for group in METRIC_GROUPS:
        for metric, values in measures[group].items():
            flat[f"{group}.{metric}"] = values
    return flat


def compare(a: dict, b: dict) -> dict[str, dict | None]:
    """Δ (b − a) de toda métrica medida nos dois braços."""
    clusters = {**a["clusters"], **b["clusters"]}
    flat_a, flat_b = _all_metrics(a), _all_metrics(b)
    return {metric: placar.paired_delta(flat_a[metric], flat_b[metric], clusters)
            for metric in sorted(set(flat_a) & set(flat_b))}


def latest_margins(exclude: Path | None = None) -> tuple[dict | None, str | None]:
    """A margem do E0 mais recente em tests/benchmark/out (a de uma rodada completa, não de fumaça)."""
    for path in sorted(OUT_ROOT.glob("E0_*/margem.json"), reverse=True):
        if exclude and path.parent == exclude:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("limite"):
            return data, str(path.relative_to(ROOT))
    return None, None


def guard_results(card: dict, measures: dict[str, dict]) -> list[dict]:
    """Cada guarda do cartão: o Δ do par na métrica e se ele fica dentro da tolerância.

    `tolerancia_perguntas: n` vira n / (perguntas medidas); `tolerancia` é absoluta (ms, por exemplo). Métrica em
    que menos é melhor (latência) passa se o Δ não sobe mais que a tolerância; as outras, se não caem mais.
    """
    results = []
    for guard in card.get("guardas") or []:
        x, y = guard["par"]
        delta = compare(measures[x], measures[y]).get(guard["metrica"])
        if delta is None:
            results.append({**guard, "delta": None, "passed": None})
            continue
        tolerance = guard["tolerancia_perguntas"] / delta["n"] if "tolerancia_perguntas" in guard else guard["tolerancia"]
        lower_is_better = guard["metrica"].split(".", 1)[1] in LOWER_IS_BETTER
        passed = delta["delta"] <= tolerance if lower_is_better else delta["delta"] >= -tolerance
        results.append({**guard, "delta": delta, "tolerance": round(tolerance, 4), "passed": passed})
    return results


def gap_closed(card: dict, arms: dict[str, dict], means: dict[str, dict]) -> dict[str, float | None]:
    """Fração do buraco fechado (E1): (braço − baseline) / (teto − baseline), na métrica primária."""
    ceiling = card.get("teto")
    baseline = next((a for a, spec in arms.items() if spec["papel"] == "baseline"), None)
    if not ceiling or not baseline:
        return {}
    primary = card["metrica_primaria"]
    metric = f"{primary['camada']}.{primary['metrica']}"
    base, top = means.get(baseline, {}).get(metric), means.get(ceiling, {}).get(metric)
    result = {}
    for arm_id, spec in arms.items():
        value = means.get(arm_id, {}).get(metric)
        if spec["papel"] == "candidato":
            gap = None if None in (base, top, value) or top == base else round((value - base) / (top - base), 3)
            result[arm_id] = gap
    return result


def scoreboard(card: dict, out: Path, limit: int | None, questions: dict[str, str] | None = None) -> dict:
    arms = arms_of(card)
    measures = {arm_id: arm_measures(out, arm, arms, questions) for arm_id, arm in arms.items()}
    baseline = next((a for a, spec in arms.items() if spec["papel"] == "baseline"), None)
    pairs = card.get("pares") or ([[baseline, a] for a in arms if a != baseline] if baseline else [])
    comparisons = {f"{x}→{y}": compare(measures[x], measures[y]) for x, y in pairs}
    means = {a: {k: round(statistics.fmean(v.values()), 4) for k, v in _all_metrics(m).items() if v}
             for a, m in measures.items()}
    result: dict = {"limite": limit, "pairs": comparisons, "summary": {a: m["summary"] for a, m in measures.items()},
                    "means": means, "guards": guard_results(card, measures),
                    "gap_closed": gap_closed(card, arms, means)}
    if card.get("margem"):
        margins = {group: placar.noise_margins([compare(measures[x], measures[y]) for x, y in group_pairs])
                   for group, group_pairs in card["margem"].items()}
        result["margins"] = margins
        (out / "margem.json").write_text(json.dumps({"experiment": card["id"], "limite": limit, "margins": margins,
                                                      "generated_at": datetime.now(timezone.utc).isoformat()},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        margin_data, margin_source = latest_margins(exclude=out)
        primary = card["metrica_primaria"]
        metric = f"{primary['camada']}.{primary['metrica']}"
        group = "c1" if primary["camada"] == "c1" else "gerador_juiz"
        margin = ((margin_data or {}).get("margins", {}).get(group) or {}).get(metric)
        result["margin_source"] = margin_source
        result["verdicts"] = {
            pair: {"metric": metric, "margin": margin,
                   "verdict": placar.verdict(deltas.get(metric), margin,
                                             higher_is_better=primary["metrica"] not in LOWER_IS_BETTER),
                   "delta": deltas.get(metric)}
            for pair, deltas in comparisons.items()
        }
    (out / "placar.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fmt(delta: dict | None) -> str:
    if not delta:
        return "—"
    return f"{delta['delta']:+.3f} [{delta['ci_low']:+.3f}; {delta['ci_high']:+.3f}] (n={delta['n']}, p95|Δ| {delta['abs_p95']:.3f})"


def write_summary(card: dict, out: Path, result: dict, card_text: str) -> Path:
    lines = [f"# {card['id']} — {card['titulo']}", "", "## O cartão, como foi escrito", "", "```yaml",
             card_text.rstrip(), "```", "", "## Médias por braço", ""]
    arms = list(arms_of(card))
    metrics = sorted({m for a in arms for m in result["means"].get(a, {})})
    lines += ["| Métrica | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for metric in metrics:
        lines.append(f"| {metric} | " + " | ".join(
            f"{result['means'][a][metric]:.3f}" if metric in result["means"].get(a, {}) else "—" for a in arms) + " |")
    lines += ["", "## Pares (Δ = segundo − primeiro, IC 95% por bootstrap de documentos)", ""]
    for pair, deltas in result["pairs"].items():
        lines += [f"### {pair}", "", "| Métrica | Δ |", "|---|---|"]
        lines += [f"| {metric} | {_fmt(delta)} |" for metric, delta in deltas.items()]
        lines.append("")
    if result.get("gap_closed"):
        lines += [f"## Fração do buraco fechado até o teto ({card['teto']})", "",
                  "(braço − baseline) / (teto − baseline), nas médias da métrica primária.", ""]
        lines += [f"- **{arm}:** " + ("—" if gap is None else f"{gap:.0%}") for arm, gap in result["gap_closed"].items()]
        lines.append("")
    if result.get("guards"):
        lines += ["## Guardas", ""]
        for guard in result["guards"]:
            state = {True: "passa", False: "NÃO PASSA", None: "não medida"}[guard["passed"]]
            lines.append(f"- **{guard['nome']}** ({' → '.join(guard['par'])}, {guard['metrica']}): {state} — "
                         f"Δ {_fmt(guard['delta'])}, tolerância {guard.get('tolerance')}")
        lines.append("")
    if result.get("margins"):
        lines += ["## Margem de ruído (P95 de |Δ| nos pares A/A)", ""]
        for group, margins in result["margins"].items():
            lines += [f"**{group}:** " + ", ".join(f"{m} {v:.3f}" for m, v in sorted(margins.items())), ""]
    if result.get("verdicts"):
        lines += ["## Veredito da métrica primária (regra provisória, até a §11)", "",
                  f"Margem tirada de: {result.get('margin_source') or 'nenhum E0 completo'}", ""]
        lines += [f"- **{pair}:** {v['verdict']} — {v['metric']} {_fmt(v['delta'])}, margem {v['margin']}"
                  for pair, v in result["verdicts"].items()]
    if result.get("limite"):
        lines += ["", f"> Rodada de fumaça (--limite {result['limite']}): números sem valor de veredito."]
    path = out / "resumo.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Roda um experimento do benchmark a partir do cartão.")
    parser.add_argument("cartao", type=Path)
    parser.add_argument("--limite", type=int, default=None, help="fumaça: só N perguntas de dentro e N de fora")
    parser.add_argument("--retomar", type=Path, default=None, help="pasta de uma rodada interrompida")
    parser.add_argument("--so-placar", action="store_true", help="refaz só o placar, com o que a pasta já tem")
    args = parser.parse_args(argv)

    card_text = args.cartao.read_text(encoding="utf-8")
    card = load_card(args.cartao)
    print_card(card)
    started = datetime.now(timezone.utc).astimezone()
    out = args.retomar or OUT_ROOT / f"{card['id']}_{started.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    # Retomar com outro recorte ou outro cartão misturaria saídas de rodadas diferentes, sem aviso.
    params = {"limite": args.limite, "cartao": card_text}
    params_path = out / "parametros.json"
    if params_path.is_file() and json.loads(params_path.read_text(encoding="utf-8")) != params:
        print(f"[ERRO] {out} é de uma rodada com outro --limite ou outro cartão.", file=sys.stderr)
        return 1
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cartao.yaml").write_text(card_text, encoding="utf-8")
    print(f"Saída: {out}\n")
    t0 = time.monotonic()
    experiment = Experiment(card, args.cartao, out, args.limite)
    if not args.so_placar:
        if ra.http_get(f"{card.get('ollama_base_url', DEFAULT_OLLAMA)}/api/tags") is None:
            print("[ERRO] Ollama fora do ar.", file=sys.stderr)
            return 1
        experiment.run()
    goldens = json.loads((ROOT / card["goldens"]).read_text(encoding="utf-8"))
    result = scoreboard(card, out, args.limite, questions={g["name"]: g["input"] for g in goldens})
    summary = write_summary(card, out, result, card_text)
    # Uma entrada por chamada: a rodada original, cada retomada e cada --so-placar ficam no histórico.
    execution_path = out / "execucao.json"
    runs = json.loads(execution_path.read_text(encoding="utf-8")).get("runs", []) if execution_path.is_file() else []
    runs.append({"started_at": started.isoformat(), "duration_seconds": round(time.monotonic() - t0, 1),
                 "so_placar": args.so_placar, "steps": [vars(s) for s in experiment.steps]})
    execution = {"tool": "tests/benchmark/run_experiment.py", "card": str(args.cartao), "limite": args.limite,
                 "runs": runs}
    execution_path.write_text(json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nResumo: {summary}")
    broken = [s for s in experiment.steps if s.status in ("falhou", "erro")]
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
