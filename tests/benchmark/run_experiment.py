"""Runner do benchmark: roda um experimento descrito num cartão (tests/benchmark/cartoes/<id>.yaml).

Provisório: a §9 do plano-benchmark.md, que especifica o runner e o YAML, não
chegou. O que está aqui segue as §4, §5 e §8 e o tests/compare_indices.py, que o
plano dá como modelo; o formato do cartão está em docs/benchmark-status.md.

O funil (§5), nesta ordem:

 1. o cartão, como foi escrito, no terminal e em <saída>/cartao.yaml; os
    goldens do `split` do cartão (ou de --split) e, com --limite, um recorte;
 2. índices: cada versão dos braços é ingerida se estiver vazia;
 3. C1 (só embeddings) nos braços com a camada c1: tests/retrieval_diagnostic.py,
    o baseline primeiro. `min_similarity_score: recalibrado` no `set` de um
    braço vira o limiar calibrado na C1 do baseline (E6, limiar_<braço>.json);
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
from app.rag.generation import NOT_FOUND_MARKER  # noqa: E402
from tests.benchmark import abstention, attribution, c1, limiar, placar  # noqa: E402

OUT_ROOT = ROOT / "tests" / "benchmark" / "out"
RAGAS_RESULTS = ROOT / "tests" / "ragas" / "results"
INDEX_VERSIONS = ROOT / "backend" / "app" / "rag" / "index_versions"
# O Ollama desta máquina só escuta em IPv4; "localhost" custa ~2 s por conexão (docs/benchmark-status.md).
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
REQUIRED = ("id", "titulo", "pergunta", "hipotese", "goldens", "bracos", "metrica_primaria", "didatica")
C1_METRICS = ("hit_at_1", "hit_at_k", "hit_at_10", "mrr_at_10", "precision_at_k", "coverage_at_k", "evidence_at_k",
              "coverage_at_10", "retrieval_ms", "context_tokens")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance")
# Métricas em que menos é melhor (o resto: mais é melhor).
LOWER_IS_BETTER = {"undue_abstention", "generation_ms", "retrieval_ms", "prompt_tokens", "completion_tokens",
                   "context_tokens", "false_block", "misattribution", "no_citation"}
# Os grupos do placar: uma métrica do cartão (guarda) se escreve grupo.metrica, como c1.evidence_at_k.
METRIC_GROUPS = ("c1", "c2", "abstention", "attribution", "cost")
# Métricas de 0 ou 1 por pergunta: nelas o placar aplica o McNemar exato (provisório até a §11).
BINARY_METRICS = {"c1.hit_at_1", "c1.hit_at_k", "c1.hit_at_10", "c1.evidence_at_k", "c1.correct_abstention",
                  "c1.false_block", "abstention.correct_abstention", "abstention.undue_abstention",
                  "attribution.misattribution", "attribution.no_citation", "attribution.cites_source"}
SPLITS = ("selecao", "confirmacao")
# c1: a recuperação; c2: as respostas e o juiz; respostas: só as respostas (recusa, atribuição e custo).
LAYERS = ("c1", "c2", "respostas")
# Valor de `set` que o runner calcula: o limiar recalibrado na C1 do baseline da própria rodada (E6).
RECALIBRATED = "recalibrado"


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
            # A métrica do veredito deste braço, quando não é a do cartão (E6: cada tipo de braço tem a sua).
            "primary": spec.get("metrica_primaria"),
        }
    return arms


def primary_metric(card: dict, arm: dict) -> str:
    """A métrica do veredito do par que termina no braço: a dele, se tiver, ou a do cartão (grupo.metrica)."""
    primary = arm.get("primary") or card["metrica_primaria"]
    return f"{primary['camada']}.{primary['metrica']}"


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
        if arm["kind"] == "rodada" and not set(arm["layers"]) <= set(LAYERS):
            errors.append(f"{arm['id']}: camadas {arm['layers']} (só {', '.join(LAYERS)})")
        if arm["kind"] == "rodada" and not (INDEX_VERSIONS / f"{arm['index']}.sha256").is_file():
            errors.append(f"{arm['id']}: a versão {arm['index']!r} não tem lista de documentos registrada")
    if sum(arm["papel"] == "baseline" for arm in arms.values()) > 1:
        errors.append("mais de um braço com papel baseline")
    for pair in card.get("pares", []) + [p for group in (card.get("margem") or {}).values() for p in group]:
        if len(pair) != 2 or not set(pair) <= set(arms):
            errors.append(f"par {pair}: tem de ser dois braços do cartão")
    for owner, primary in [("o cartão", card["metrica_primaria"])] + [
            (arm["id"], arm["primary"]) for arm in arms.values() if arm.get("primary")]:
        if primary.get("camada") not in METRIC_GROUPS or not primary.get("metrica"):
            errors.append(f"metrica_primaria de {owner} precisa de camada ({', '.join(METRIC_GROUPS)}) e metrica")
    if card.get("split") and card["split"] not in SPLITS:
        errors.append(f"split {card['split']!r}: {' ou '.join(SPLITS)}")
    for arm_id in card.get("curva") or []:
        if arm_id not in arms:
            errors.append(f"curva: {arm_id!r} não é um braço do cartão")
    baseline = next((a for a in arms.values() if a["papel"] == "baseline"), None)
    for arm in arms.values():
        if arm["kind"] == "rodada" and arm["set"].get("min_similarity_score") == RECALIBRATED and not (
                baseline and "c1" in baseline["layers"] and baseline["index"] == arm["index"]):
            errors.append(f"{arm['id']}: limiar {RECALIBRATED} precisa de um baseline com c1 no mesmo índice")
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
        if guard.get("grupo") is not None and not (isinstance(guard["grupo"], dict) and "campo" in guard["grupo"]
                                                   and "valor" in guard["grupo"]):
            errors.append(f"guarda {name}: grupo precisa de campo e valor")
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
    def __init__(self, card: dict, card_path: Path, out: Path, limit: int | None, split: str | None = None) -> None:
        self.card, self.card_path, self.out, self.limit = card, card_path, out, limit
        self.split = split or card.get("split")
        self.arms = arms_of(card)
        self.steps: list[ra.Step] = []
        self.goldens_path = ROOT / card["goldens"]

    def record(self, key: str, name: str, status: str, detail: str, seconds: float | None = None,
               log: Path | None = None) -> None:
        step = ra.Step(key, name, status, detail, None, seconds, str(log.relative_to(ROOT)) if log else None)
        self.steps.append(step)
        took = f" ({seconds:.0f}s)" if seconds is not None else ""
        print(f"    {ra.STATUS_TAG.get(status, status)} {name}: {detail}{took}", flush=True)

    # 0. goldens: só o split do cartão (E6: a varredura em selecao, o veredito em confirmacao) e, com --limite,
    #    um recorte pequeno dos de dentro e dos de fora
    def prepare_goldens(self) -> None:
        if not self.limit and not self.split:
            return
        goldens = json.loads(self.goldens_path.read_text(encoding="utf-8"))
        if self.split:
            goldens = [g for g in goldens if g.get("split") == self.split]
        if self.limit:
            inside = [g for g in goldens if g.get("source_document")][: self.limit]
            outside = [g for g in goldens if g.get("question_class") in ("fora_do_corpus", "fora_de_escopo")]
            goldens = inside + outside[: self.limit]
        self.goldens_path = self.out / "goldens_recorte.json"
        self.goldens_path.write_text(json.dumps(goldens, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolved_set(self, arm: dict) -> dict:
        """O `set` do braço com os valores calculados: `min_similarity_score: recalibrado` vira o limiar."""
        values = dict(arm["set"])
        if values.get("min_similarity_score") == RECALIBRATED:
            values["min_similarity_score"] = self.recalibrated_threshold(arm)["threshold"]
        return values

    def recalibrated_threshold(self, arm: dict) -> dict:
        """O limiar do E6: a regra do estudo de mutação (tests/benchmark/limiar.py) na C1 do baseline da rodada.

        Positivas: a similaridade do primeiro trecho das perguntas de dentro; negativas, a das de fora. Calibrado
        e avaliado nos mesmos dados, o falso bloqueio dá zero por construção: o arquivo grava também a validação
        deixando um documento de fora, e o veredito do limiar sai da confirmacao.
        """
        target = self.out / f"limiar_{arm['id']}.json"
        if target.is_file():
            return json.loads(target.read_text(encoding="utf-8"))
        baseline = next(a for a in self.arms.values() if a["papel"] == "baseline")
        diag = json.loads((self.out / f"c1_{baseline['id']}.json").read_text(encoding="utf-8"))
        inside, outside = [], []
        for row in diag["rows"]:
            metrics = row["c1"].get(baseline["index"], {}).get(baseline["lang"])
            if metrics and metrics.get("top1_similarity") is not None:
                if metrics.get("in_index"):
                    inside.append((c1.doc_stem(row["source_document"]), metrics["top1_similarity"]))
                else:
                    outside.append(metrics["top1_similarity"])
        for row in diag.get("out_rows", []):
            metrics = row["c1"].get(baseline["index"], {}).get(baseline["lang"])
            if metrics and metrics.get("top1_similarity") is not None:
                outside.append(metrics["top1_similarity"])
        positives = [value for _, value in inside]
        choice = limiar.choose_threshold(positives, outside)
        result = {
            **choice, "calibrated_on": baseline["id"], "split": self.split, "n_inside": len(inside),
            "n_outside": len(outside), "in_sample": limiar.evaluate(choice["threshold"], positives, outside),
            "leave_one_document_out": limiar.leave_one_document_out(inside, outside),
        }
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    limiar recalibrado para {arm['id']}: {choice['threshold']} ({choice['rule']})", flush=True)
        return result

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
               *set_args({**(self.card.get("fixos") or {}), **self.resolved_set(arm)})]
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
        settings = {**(self.card.get("fixos") or {}), **self.resolved_set(arm)}
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
        # O baseline primeiro: o limiar recalibrado de outro braço sai da C1 dele.
        for arm in sorted(self.arms.values(), key=lambda a: a["papel"] != "baseline"):
            if arm["kind"] == "rodada" and "c1" in arm["layers"]:
                self.c1(arm)
        for arm in self.arms.values():
            if arm["kind"] == "rodada" and {"c2", "respostas"} & set(arm["layers"]):
                answers = self.answers(arm)
                # "respostas" para nas respostas: recusa, atribuição e custo não precisam do juiz.
                if answers and "c2" in arm["layers"]:
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


def refused(answer: dict) -> bool:
    """A recusa com o detector de hoje (tests/benchmark/abstention.py), não o `abstained` gravado na rodada:
    uma variante revisada depois vale também para as rodadas antigas."""
    if answer.get("actual_output") is None:
        return bool(answer.get("abstained"))
    return abstention.is_abstention(answer["actual_output"], NOT_FOUND_MARKER)


def arm_measures(out: Path, arm: dict, arms: dict[str, dict], questions: dict[str, str] | None = None,
                 classes: dict[str, str | None] | None = None) -> dict:
    """Por pergunta, tudo o que o placar compara: C1 (na língua do braço), RAGAS, recusa e custo.

    `questions`: nome → pergunta em português, para somar o tempo da tradução gravada à latência da busca.
    `classes`: nome → question_class. Pergunta ambígua não entra na abstenção correta: o certo é pedir
    esclarecimento, não se abster.
    """
    classes = classes or {}
    source = arms[arm["of"]] if arm["kind"] == "rejulgar" else arm
    measures: dict = {"clusters": {}, "c1": {}, "c2": {}, "abstention": {}, "attribution": {}, "cost": {},
                      "summary": {}, "curves": {}}
    translation_ms = translation_milliseconds(arm, questions or {}) if arm.get("lang") == "pt" else {}
    diag_path = out / f"c1_{arm['id']}.json"
    if arm["kind"] == "rodada" and diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        summary = diag["c1"]["summary"][arm["index"]][arm["lang"]]
        measures["summary"]["c1"] = summary
        # Abstenção pelo limiar, por pergunta: a de fora barrada acerta; a de dentro barrada é falso bloqueio.
        blocked_outside: dict[str, bool] = {}
        for row in diag["rows"]:
            metrics = row["c1"].get(arm["index"], {}).get(arm["lang"])
            if not metrics:
                continue
            if not metrics.get("in_index"):
                # O documento não está nesta versão: a pergunta conta como de fora nela.
                blocked_outside[row["name"]] = bool(metrics.get("blocked_by_threshold"))
                continue
            measures["clusters"][row["name"]] = c1.doc_stem(row["source_document"])
            for metric in C1_METRICS:
                if metrics.get(metric) is not None:
                    value = float(metrics[metric])
                    if metric == "retrieval_ms":
                        value += translation_ms.get(row["name"], 0.0)
                    measures["c1"].setdefault(metric, {})[row["name"]] = value
            measures["c1"].setdefault("false_block", {})[row["name"]] = float(bool(metrics.get("blocked_by_threshold")))
            if metrics.get("evidence_curve") and metrics.get("tokens_curve"):
                measures["curves"].setdefault("evidence", {})[row["name"]] = [float(v) for v in metrics["evidence_curve"]]
                measures["curves"].setdefault("tokens", {})[row["name"]] = metrics["tokens_curve"]
        for row in diag.get("out_rows", []):
            metrics = row["c1"].get(arm["index"], {}).get(arm["lang"])
            if metrics:
                blocked_outside[row["name"]] = bool(metrics.get("blocked_by_threshold"))
        if blocked_outside:
            measures["c1"]["correct_abstention"] = {n: float(b) for n, b in blocked_outside.items()}
    answers_path = out / f"respostas_{source['id']}.json"
    if answers_path.is_file() and arm["kind"] == "rodada" and {"c2", "respostas"} & set(arm["layers"]):
        documents = index_documents(source["index"])
        answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
        inside = {a["name"]: a for a in answers if a.get("source_document")
                  and c1.doc_stem(a["source_document"]) in documents and not a.get("error")}
        outside = {a["name"]: a for a in answers if not a.get("source_document") and not a.get("error")}
        for name, a in inside.items():
            measures["clusters"][name] = c1.doc_stem(a["source_document"])
        measures["abstention"] = {
            "correct_abstention": {n: float(refused(a)) for n, a in outside.items() if classes.get(n) != "ambiguo"},
            "undue_abstention": {n: float(refused(a)) for n, a in inside.items()},
        }
        for metric in ("retrieval_ms", "generation_ms", "prompt_tokens", "completion_tokens"):
            values = {n: float(a[metric]) for n, a in {**inside, **outside}.items() if a.get(metric) is not None}
            if metric == "retrieval_ms":
                values = {n: v + translation_ms.get(n, 0.0) for n, v in values.items()}
            measures["cost"][metric] = values
        measures["summary"]["truncated"] = sum(1 for a in answers if a.get("truncated"))
        measures["summary"]["errors"] = sum(1 for a in answers if a.get("error"))
        provenance = json.loads(answers_path.read_text(encoding="utf-8")).get("arm") or {}
        if (provenance.get("prompt") or {}).get("require_citation"):
            # Só nos braços com citação obrigatória (E3): sem a regra, toda resposta seria "sem citação".
            categories = {n: attribution.classify(a, refused(a)) for n, a in inside.items()}
            answered = [n for n, c in categories.items() if c != "recusa"]
            measures["attribution"] = {
                "misattribution": {n: float(attribution.misattributed(a, refused(a), inside=n in inside))
                                   for n, a in {**inside, **outside}.items()},
                "no_citation": {n: float(categories[n] == "sem_citacao") for n in answered},
                "cites_source": {n: float(categories[n] == "fonte") for n in answered},
            }
            measures["summary"]["attribution"] = {c: sum(v == c for v in categories.values())
                                                  for c in attribution.CATEGORIES}
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


def _restricted(measures: dict, names: set[str]) -> dict:
    """As medidas do braço só nestas perguntas (a guarda de um subgrupo)."""
    return {**measures, **{group: {metric: {n: v for n, v in values.items() if n in names}
                                   for metric, values in measures[group].items()} for group in METRIC_GROUPS}}


def guard_results(card: dict, measures: dict[str, dict], goldens: list[dict] | None = None) -> list[dict]:
    """Cada guarda do cartão: o Δ do par na métrica e se ele fica dentro da tolerância.

    `tolerancia_perguntas: n` vira n / (perguntas medidas); `tolerancia` é absoluta (ms, por exemplo). Métrica em
    que menos é melhor (latência) passa se o Δ não sobe mais que a tolerância; as outras, se não caem mais.
    `grupo: {campo, valor}` restringe a guarda às perguntas com esse valor no golden (E7: as sem sigla).
    """
    results = []
    for guard in card.get("guardas") or []:
        x, y = guard["par"]
        a, b = measures[x], measures[y]
        if guard.get("grupo"):
            field, value = guard["grupo"]["campo"], guard["grupo"]["valor"]
            names = {g["name"] for g in goldens or [] if g.get(field) == value}
            a, b = _restricted(a, names), _restricted(b, names)
        delta = compare(a, b).get(guard["metrica"])
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


def curve_means(measures: dict) -> list[dict]:
    """A curva evidência × tokens de contexto do braço: em cada k, as médias nas perguntas com k trechos."""
    evidence, tokens = measures["curves"].get("evidence", {}), measures["curves"].get("tokens", {})
    points = []
    for k in range(1, max((len(v) for v in evidence.values()), default=0) + 1):
        names = [n for n in evidence if len(evidence[n]) >= k and len(tokens.get(n, [])) >= k]
        if names:
            points.append({"k": k, "n": len(names), "tokens": round(statistics.fmean(tokens[n][k - 1] for n in names), 1),
                           "evidence": round(statistics.fmean(evidence[n][k - 1] for n in names), 4)})
    return points


def significance(card: dict, arms: dict[str, dict], measures: dict[str, dict], pairs: list) -> dict[str, dict]:
    """McNemar exato na métrica primária de cada par, se ela é de 0 ou 1, com Holm na família dos pares.

    Provisório até a §11. A família são os pares do cartão: no E6, a varredura inteira (§8).
    """
    tests = {}
    for x, y in pairs:
        metric = primary_metric(card, arms[y])
        a, b = _all_metrics(measures[x]).get(metric), _all_metrics(measures[y]).get(metric)
        if metric in BINARY_METRICS and a and b:
            test = placar.mcnemar_exact({n: v >= 0.5 for n, v in a.items()}, {n: v >= 0.5 for n, v in b.items()})
            tests[f"{x}→{y}"] = {"metric": metric, **test}
    for test, adjusted in zip(tests.values(), placar.holm([t["p_value"] for t in tests.values()])):
        test["p_value"], test["p_holm"] = round(test["p_value"], 4), round(adjusted, 4)
    return tests


def by_class(card: dict, arms: dict[str, dict], measures: dict[str, dict], pairs: list,
             classes: dict[str, str | None]) -> dict[str, dict]:
    """Δ da métrica primária de cada par em cada classe de pergunta (E6: o MMR em fato_distribuido)."""
    result = {}
    for x, y in pairs if classes else []:
        metric = primary_metric(card, arms[y])
        a, b = _all_metrics(measures[x]).get(metric) or {}, _all_metrics(measures[y]).get(metric) or {}
        clusters = {**measures[x]["clusters"], **measures[y]["clusters"]}
        groups: dict[str, list[str]] = {}
        for name in sorted(set(a) & set(b)):
            groups.setdefault(classes.get(name) or "sem classe", []).append(name)
        result[f"{x}→{y}"] = {
            cls: placar.paired_delta({n: a[n] for n in names}, {n: b[n] for n in names}, clusters)
            for cls, names in sorted(groups.items())
        }
    return result


def scoreboard(card: dict, out: Path, limit: int | None, goldens: list[dict] | None = None,
               split: str | None = None) -> dict:
    """`goldens`: os do cartão, para a tradução gravada (pela pergunta) e as classes de pergunta."""
    arms = arms_of(card)
    questions = {g["name"]: g["input"] for g in goldens or [] if g.get("input")}
    classes = {g["name"]: g.get("question_class") for g in goldens or []}
    # O Δ por grupo sai pela classe da pergunta ou por outro campo do golden (E2: `classe_por: content_type`).
    groups = {g["name"]: g.get(card.get("classe_por") or "question_class") for g in goldens or []}
    measures = {arm_id: arm_measures(out, arm, arms, questions, classes) for arm_id, arm in arms.items()}
    baseline = next((a for a, spec in arms.items() if spec["papel"] == "baseline"), None)
    pairs = card.get("pares") or ([[baseline, a] for a in arms if a != baseline] if baseline else [])
    comparisons = {f"{x}→{y}": compare(measures[x], measures[y]) for x, y in pairs}
    means = {a: {k: round(statistics.fmean(v.values()), 4) for k, v in _all_metrics(m).items() if v}
             for a, m in measures.items()}
    thresholds = {a: json.loads(p.read_text(encoding="utf-8")) for a in arms if (p := out / f"limiar_{a}.json").is_file()}
    result: dict = {"limite": limit, "split": split or card.get("split"), "exploratory": bool(card.get("exploratorio")),
                    "pairs": comparisons, "summary": {a: m["summary"] for a, m in measures.items()},
                    "means": means, "guards": guard_results(card, measures, goldens),
                    "gap_closed": gap_closed(card, arms, means),
                    "tests": significance(card, arms, measures, pairs),
                    "by_class": by_class(card, arms, measures, pairs, groups),
                    "class_field": card.get("classe_por") or "question_class",
                    "curves": {a: curve_means(m) for a, m in measures.items() if m["curves"]},
                    "thresholds": thresholds}
    own: dict = {}
    if card.get("margem"):
        own = {group: placar.noise_margins([compare(measures[x], measures[y]) for x, y in group_pairs])
               for group, group_pairs in card["margem"].items()}
        result["margins"] = own
        (out / "margem.json").write_text(json.dumps({"experiment": card["id"], "limite": limit, "margins": own,
                                                      "generated_at": datetime.now(timezone.utc).isoformat()},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    # Veredito nos pares que não são de margem. A margem de uma métrica vem do A/A do próprio cartão, se ele
    # mediu (E3: a citação muda o prompt, e a margem refaz-se); senão, do E0 completo mais recente.
    margin_pairs = {tuple(p) for group in (card.get("margem") or {}).values() for p in group}
    verdict_pairs = [(x, y) for x, y in pairs if (x, y) not in margin_pairs]
    if verdict_pairs:
        margin_data, margin_source = latest_margins(exclude=out)
        e0 = (margin_data or {}).get("margins", {})
        result["margin_source"] = margin_source
        result["verdicts"] = {}
        for x, y in verdict_pairs:
            metric = primary_metric(card, arms[y])
            # A C1 tem a margem dela (A1 × A5); o resto, a do gerador e do juiz (A1 × A2 × A3).
            group = "c1" if metric.startswith("c1.") else "gerador_juiz"
            from_card = metric in (own.get(group) or {})
            margin = (own if from_card else e0).get(group, {}).get(metric)
            delta = comparisons[f"{x}→{y}"].get(metric)
            result["verdicts"][f"{x}→{y}"] = {
                "metric": metric, "margin": margin, "delta": delta,
                "margin_from": "cartão" if from_card else margin_source,
                "verdict": placar.verdict(delta, margin, higher_is_better=metric.split(".", 1)[1] not in LOWER_IS_BETTER),
            }
    (out / "placar.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fmt(delta: dict | None) -> str:
    if not delta:
        return "—"
    return f"{delta['delta']:+.3f} [{delta['ci_low']:+.3f}; {delta['ci_high']:+.3f}] (n={delta['n']}, p95|Δ| {delta['abs_p95']:.3f})"


def _summary_extras(card: dict, result: dict, arms: list[str]) -> list[str]:
    """As seções do E6: limiar recalibrado, AUC do primeiro trecho, curva evidência × tokens, Δ por classe."""
    lines = []
    if result.get("thresholds"):
        lines += ["## Limiar recalibrado", ""]
        for arm, t in result["thresholds"].items():
            cv = t.get("leave_one_document_out") or {}
            lines.append(f"- **{arm}:** {t['threshold']} ({t['rule']}), na C1 de {t['calibrated_on']} "
                         f"({t['n_inside']} de dentro, {t['n_outside']} de fora). Falso bloqueio deixando um documento "
                         f"de fora: {cv.get('false_block_count')} pergunta(s); abstenção correta na amostra: "
                         f"{t['in_sample']['correct_abstention']}")
        lines.append("")
    c1_summary = {a: (result["summary"].get(a) or {}).get("c1") or {} for a in arms}
    if any(s.get("top1_similarity_auc") for s in c1_summary.values()):
        lines += ["## Separação entre perguntas de dentro e de fora (AUC do primeiro trecho)", ""]
        for a, s in c1_summary.items():
            auc, rerank_auc = s.get("top1_similarity_auc"), s.get("top1_rerank_auc")
            if auc:
                extra = (f"; nota do reranker {rerank_auc['auc']} [{rerank_auc.get('ci_low')}; "
                         f"{rerank_auc.get('ci_high')}]") if rerank_auc else ""
                lines.append(f"- **{a}:** cosseno {auc['auc']} [{auc.get('ci_low')}; {auc.get('ci_high')}]{extra}")
        lines.append("")
    shown = [a for a in card.get("curva") or [] if result.get("curves", {}).get(a)]
    if shown:
        lines += ["## Curva evidência × tokens de contexto (lista de 10, k primeiros trechos)", "",
                  "| k | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown)]
        for k in range(1, max(len(result["curves"][a]) for a in shown) + 1):
            cells = []
            for a in shown:
                point = next((p for p in result["curves"][a] if p["k"] == k), None)
                cells.append(f"{point['evidence']:.3f} · {point['tokens']:.0f} tok" if point else "—")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    if any(result.get("by_class", {}).values()):
        lines += [f"## Δ da métrica primária por `{result.get('class_field', 'question_class')}`", ""]
        for pair, classes in result["by_class"].items():
            parts = [f"{cls} {d['delta']:+.3f} (n={d['n']})" for cls, d in classes.items() if d]
            lines.append(f"- **{pair}:** " + ", ".join(parts))
        lines.append("")
    return lines


def write_summary(card: dict, out: Path, result: dict, card_text: str) -> Path:
    lines = [f"# {card['id']} — {card['titulo']}", ""]
    if result.get("split"):
        lines += [f"Goldens do split `{result['split']}`.", ""]
    if result.get("exploratory"):
        lines += ["> Experimento exploratório: a varredura escolhe; o Δ e o veredito do escolhido saem da "
                  "confirmação, rodada uma vez.", ""]
    lines += ["## O cartão, como foi escrito", "", "```yaml", card_text.rstrip(), "```", "", "## Médias por braço", ""]
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
    lines += _summary_extras(card, result, arms)
    if result.get("gap_closed"):
        lines += [f"## Fração do buraco fechado até o teto ({card['teto']})", "",
                  "(braço − baseline) / (teto − baseline), nas médias da métrica primária.", ""]
        lines += [f"- **{arm}:** " + ("—" if gap is None else f"{gap:.0%}") for arm, gap in result["gap_closed"].items()]
        lines.append("")
    if result.get("guards"):
        lines += ["## Guardas", ""]
        for guard in result["guards"]:
            state = {True: "passa", False: "NÃO PASSA", None: "não medida"}[guard["passed"]]
            group = f", só {guard['grupo']['campo']} = {guard['grupo']['valor']}" if guard.get("grupo") else ""
            lines.append(f"- **{guard['nome']}** ({' → '.join(guard['par'])}, {guard['metrica']}{group}): {state} — "
                         f"Δ {_fmt(guard['delta'])}, tolerância {guard.get('tolerance')}")
        lines.append("")
    if result.get("margins"):
        lines += ["## Margem de ruído (P95 de |Δ| nos pares A/A)", ""]
        for group, margins in result["margins"].items():
            lines += [f"**{group}:** " + ", ".join(f"{m} {v:.3f}" for m, v in sorted(margins.items())), ""]
    if result.get("verdicts"):
        title = "Veredito da métrica primária (regra provisória, até a §11)"
        lines += [f"## {title}" + (" — exploratório" if result.get("exploratory") else ""), "",
                  f"Margem tirada de: {result.get('margin_source') or 'nenhum E0 completo'}", ""]
        for pair, v in result["verdicts"].items():
            test = (result.get("tests") or {}).get(pair)
            extra = f"; McNemar p {test['p_value']}, Holm {test['p_holm']}" if test else ""
            own = " (do A/A do cartão)" if v.get("margin_from") == "cartão" else ""
            lines.append(f"- **{pair}:** {v['verdict']} — {v['metric']} {_fmt(v['delta'])}, margem {v['margin']}{own}"
                         f"{extra}")
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
    parser.add_argument("--split", choices=SPLITS, default=None, help="só os goldens deste split (padrão: o do cartão)")
    args = parser.parse_args(argv)

    card_text = args.cartao.read_text(encoding="utf-8")
    card = load_card(args.cartao)
    print_card(card)
    split = args.split or card.get("split")
    started = datetime.now(timezone.utc).astimezone()
    out = args.retomar or OUT_ROOT / f"{card['id']}_{started.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    # Retomar com outro recorte ou outro cartão misturaria saídas de rodadas diferentes, sem aviso.
    params = {"limite": args.limite, "cartao": card_text, **({"split": split} if split else {})}
    params_path = out / "parametros.json"
    if params_path.is_file() and json.loads(params_path.read_text(encoding="utf-8")) != params:
        print(f"[ERRO] {out} é de uma rodada com outro --limite, outro split ou outro cartão.", file=sys.stderr)
        return 1
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cartao.yaml").write_text(card_text, encoding="utf-8")
    print(f"Saída: {out}\n")
    t0 = time.monotonic()
    experiment = Experiment(card, args.cartao, out, args.limite, split)
    if not args.so_placar:
        if ra.http_get(f"{card.get('ollama_base_url', DEFAULT_OLLAMA)}/api/tags") is None:
            print("[ERRO] Ollama fora do ar.", file=sys.stderr)
            return 1
        experiment.run()
    goldens = json.loads((ROOT / card["goldens"]).read_text(encoding="utf-8"))
    result = scoreboard(card, out, args.limite, goldens, split)
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
