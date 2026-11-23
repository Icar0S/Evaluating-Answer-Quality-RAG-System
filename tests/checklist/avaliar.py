"""Avalia o checklist de testes de RAG (checklist.yaml) sobre uma rodada da pergunta única.

Cada item do checklist diz como é conferido: por uma conta sobre a pasta da
rodada (`evidencia`), por um teste automatizado que o guarda (`teste`) ou como
fora do escopo (`fora`). Este módulo faz as contas e devolve, por item, o estado
e o detalhe:

- passou / falhou: a conta da evidência, com o número que decidiu;
- coberto: o teste que guarda o item existe (quem diz se passa é o CI);
- fora: fora do escopo, com o motivo;
- sem dado: a rodada não tem o que a conta precisa (por exemplo, sem os mutantes).

Rodar (qualquer Python com PyYAML; o venv do relatório tem):
    tests\\reports\\.venv\\Scripts\\python.exe tests\\checklist\\avaliar.py --rodada tests\\pergunta_unica\\out\\<pasta>
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "tests" / "pergunta_unica") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "pergunta_unica"))

import analise  # noqa: E402
import mutacao  # noqa: E402
import pares  # noqa: E402

CHECKLIST = HERE / "checklist.yaml"
STATE_MARK = {"passou": "✓", "falhou": "✗", "coberto": "✓ (teste)", "fora": "—", "sem dado": "?"}
# Os fatos da referência de md-mtbench-vieses, por palavra-chave, em português e em inglês. É um oráculo
# determinístico: não depende de juiz. "auto-elogio" e "autopromoção" entram porque o gemma traduz assim.
FACTS = {
    "posição": (r"posi[çc][ãa]o", r"position"),
    "verbosidade": (r"verbos",),
    "autoaprimoramento": (r"self-?enhancement", r"auto-?aprimoramento", r"auto-?elogio", r"auto-?promo[çc][ãa]o",
                          r"auto-?aperfei[çc]oamento", r"auto-?favorecimento"),
    "80%": (r"80\s?%",),
}
ANSWER_METRICS = ("faithfulness", "answer_relevancy", "correctness")


@dataclass
class RunData:
    dir: Path
    answers: dict[str, Any]
    exports: list[dict[str, Any]]
    english: dict[str, Any] | None
    pairs: dict[str, Any] | None
    mutants: dict[str, Any] | None

    @property
    def answer_list(self) -> list[dict[str, Any]]:
        return self.answers.get("answers", [])


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    answers = _json(run_dir / "respostas.json")
    if answers is None:
        raise ValueError(f"{run_dir} não é uma rodada da pergunta única (falta respostas.json)")
    exports = [analise.load_export(p) for p in analise.find_exports(run_dir)]
    pair_judges = {}
    for path in sorted((run_dir / "pares").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pair_judges[data.get("juiz") or path.stem] = data["julgamentos"]
    _, by_name = pares.answers_by_name(run_dir / "respostas.json")
    pairs_report = pares.pairs_report(pair_judges, by_name) if pair_judges else None
    mutants_report = _json(run_dir / "mutantes-analise.json")
    return RunData(run_dir, answers, exports, _json(run_dir / "ingles" / "respostas.json"), pairs_report, mutants_report)


def facts_in(text: str | None) -> set[str]:
    low = (text or "").lower()
    return {fact for fact, patterns in FACTS.items() if any(re.search(p, low) for p in patterns)}


Result = tuple[str, str]


# ------------------------------------------------------------------ as contas


def artigo_no_topk(run: RunData) -> Result:
    hits = [a for a in run.answer_list
            if any(d.startswith(mutacao.DOCUMENT_PREFIX) for d in a.get("source_documents") or [])]
    n = len(run.answer_list)
    return ("passou" if n and len(hits) == n else "falhou", f"o artigo nos trechos de {len(hits)} de {n} respostas")


def trechos_iguais(run: RunData) -> Result:
    trechos = (run.answers.get("protocolo") or {}).get("trechos") or {}
    if not trechos:
        return "sem dado", "a rodada não gravou o hash dos trechos (rodou sem o protocolo?)"
    same = all(t.get("iguais") for t in trechos.values())
    detail = "; ".join(f"{name}: {t.get('sha256')} em {t.get('respostas')} respostas" for name, t in trechos.items())
    return ("passou" if same else "falhou", detail)


def fatos_da_referencia(run: RunData) -> Result:
    found = {a["name"]: facts_in(a.get("actual_output")) for a in run.answer_list}
    complete = [n for n, f in found.items() if f == set(FACTS)]
    missing = {pares_short(n): sorted(set(FACTS) - f) for n, f in found.items() if f != set(FACTS)}
    detail = f"{len(complete)} de {len(found)} respostas com os 4 fatos"
    if missing:
        detail += "; faltam: " + "; ".join(f"{n} ({', '.join(m)})" for n, m in missing.items())
    return ("passou" if found and len(complete) == len(found) else "falhou", detail)


def pares_short(name: str) -> str:
    return "#" + name.split("#")[-1] if "#" in name else name


def _metric_means(run: RunData, metric: str) -> dict[str, tuple[float, float]]:
    """framework -> (média das notas medidas, limiar)."""
    out = {}
    for fw in analise.FRAMEWORKS:
        scores, limit = [], None
        for e in run.exports:
            if analise.framework_of(e) != fw:
                continue
            for case in e.get("cases", []):
                for m in case.get("metrics", []):
                    if analise.canonical_metric(m["name"]) == metric:
                        limit = m.get("threshold", e.get("threshold"))
                        if m.get("score") is not None and m.get("applicable", True):
                            scores.append(float(m["score"]))
        if scores:
            out[fw] = (statistics.fmean(scores), float(limit if limit is not None else 0.7))
    return out


def fidelidade(run: RunData) -> Result:
    means = _metric_means(run, "faithfulness")
    if not means:
        return "sem dado", "nenhum juiz julgou a fidelidade nesta rodada"
    detail = "; ".join(f"{fw} {m:.2f}".replace(".", ",") for fw, (m, _) in means.items())
    ok = all(m >= limit for m, limit in means.values()) and len(means) == 3
    return ("passou" if ok else "falhou", detail + ("" if len(means) == 3 else f" (só {len(means)} de 3 juízes)"))


def mesmos_fatos_em_ingles(run: RunData) -> Result:
    if not run.english:
        return "sem dado", "a rodada não tem as respostas em inglês (rodar.py --ingles)"
    pt = {a.get("semente_pedida") or a.get("seed"): facts_in(a.get("actual_output")) for a in run.answer_list}
    en = {a.get("semente_pedida") or a.get("seed"): facts_in(a.get("actual_output"))
          for a in run.english.get("answers", [])}
    common = sorted(set(pt) & set(en), key=lambda s: (s is None, s))
    same = [s for s in common if pt[s] == en[s]]
    detail = f"os mesmos fatos em {len(same)} de {len(common)} sementes"
    diffs = [f"semente {s}: pt {sorted(pt[s])} x en {sorted(en[s])}" for s in common if pt[s] != en[s]]
    return ("passou" if common and len(same) == len(common) else "falhou", detail + ("; " + "; ".join(diffs) if diffs
                                                                                   else ""))


def abstencao_sem_o_documento(run: RunData) -> Result:
    c1 = ((run.mutants or {}).get("mutantes") or {}).get("C1")
    if not c1:
        return "sem dado", "a rodada não tem o mutante C1 (rodar.py --mutantes C1)"
    b = c1["comportamento"]
    detail = (f"{b['abstencoes']} de {b['respostas']} respostas se abstiveram sem o artigo; o artigo chegou nos "
              f"trechos de {b['com_o_artigo_nos_trechos']}")
    return ("passou" if b["abstencoes"] > 0 else "falhou", detail)


def respostas_concordam(run: RunData) -> Result:
    sets = [frozenset(facts_in(a.get("actual_output"))) for a in run.answer_list]
    distinct = sorted({tuple(sorted(s)) for s in sets})
    return ("passou" if len(distinct) == 1 else "falhou",
            f"{len(distinct)} conjunto(s) de fatos nas {len(sets)} respostas: "
            + "; ".join(", ".join(d) or "nenhum" for d in distinct))


def estado_limpo(run: RunData) -> Result:
    protocol = run.answers.get("protocolo") or {}
    if not protocol:
        return "sem dado", "a rodada não foi gerada com o protocolo"
    cold = [a for a in run.answer_list if a.get("estado") == "frio"]
    seeded = [a for a in run.answer_list if a.get("semente_pedida") is None or a.get("seed") == a["semente_pedida"]]
    n = len(run.answer_list)
    ok = protocol.get("estado_limpo") and len(cold) == n and len(seeded) == n
    return ("passou" if ok else "falhou", f"{len(cold)} de {n} respostas frias; a semente valeu em {len(seeded)} de {n}")


def gerador_nao_julga(run: RunData) -> Result:
    generator = run.answers.get("generation_model")
    judges = sorted({e.get("judge_model") for e in run.exports if e.get("judge_model")})
    if not judges:
        return "sem dado", "nenhum export de juiz nesta rodada"
    same = [j for j in judges if generator and str(j).split(" ")[0] == generator]
    return ("falhou" if same else "passou", f"gera {generator}; julgam {', '.join(judges)}")


def _pair_judges(run: RunData) -> dict[str, Any]:
    return {j: s for j, s in ((run.pairs or {}).get("juizes") or {}).items() if "(implícita)" not in j}


def consistencia_de_posicao(run: RunData) -> Result:
    judges = _pair_judges(run)
    if not judges:
        return "sem dado", "a rodada não tem comparação em pares"
    rows = {j: s["posicao"] for j, s in judges.items() if s["posicao"]["pares_nas_duas_ordens"]}
    if not rows:
        return "sem dado", "os pares rodaram numa ordem só (--uma-ordem)"
    detail = "; ".join(f"{j}: {p['consistentes']}/{p['pares_nas_duas_ordens']}" for j, p in rows.items())
    ok = all(p["consistencia"] >= 0.9 for p in rows.values())
    return ("passou" if ok else "falhou", detail)


def verbosidade(run: RunData) -> Result:
    judges = _pair_judges(run)
    if not judges:
        return "sem dado", "a rodada não tem comparação em pares"
    rows = {j: s["verbosidade"] for j, s in judges.items() if s["verbosidade"]["decididos_com_tamanhos_diferentes"] >= 4}
    if not rows:
        return "sem dado", "menos de 4 pares decididos com tamanhos diferentes"
    detail = "; ".join(f"{j}: a mais longa venceu {v['vitorias_da_mais_longa']}/{v['decididos_com_tamanhos_diferentes']}"
                       for j, v in rows.items())
    return ("passou" if all(v["taxa_da_mais_longa"] <= 0.7 for v in rows.values()) else "falhou", detail)


def vereditos_estaveis(run: RunData) -> Result:
    report = analise.variance_report(run.exports)
    reps = {fw: data["julgamentos"] for fw, data in report["frameworks"].items()}
    if not any(len(r) >= 2 for r in reps.values()):
        return "sem dado", "um julgamento só por juiz (rodar.py --julgamentos 3)"
    unstable = {f"{fw}/{metric}": s["vereditos_instaveis"] for fw, data in report["frameworks"].items()
                for metric, s in data["metricas"].items() if metric in ANSWER_METRICS and s["vereditos_instaveis"]}
    detail = ("nenhum veredito mudou entre julgamentos" if not unstable else
              "; ".join(f"{k}: {', '.join(pares_short(n) for n in v)}" for k, v in unstable.items()))
    return ("passou" if not unstable else "falhou", detail)


def mutantes_mortos(run: RunData) -> Result:
    mutants = (run.mutants or {}).get("mutantes") or {}
    if not mutants:
        return "sem dado", "a rodada não tem mutantes (rodar.py --mutantes C1 C2 P1 R1)"
    rows = {op: [fw for fw, j in m["juizes"].items() if j["morto"]] for op, m in mutants.items()}
    detail = "; ".join(f"{op}: {', '.join(k) if k else 'vivo nos três'}" for op, k in rows.items())
    return ("passou" if all(rows.values()) else "falhou", detail)


def _registered(index: str | None) -> set[str] | None:
    path = ROOT / "backend" / "app" / "rag" / "index_versions" / f"{index}.sha256"
    if not index or not path.is_file():
        return None
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            names.add(line.split(None, 1)[1].strip())
    return names


def so_acervo_publico(run: RunData) -> Result:
    jev_cases = {c["name"] for e in run.exports if analise.framework_of(e) == "jeveval" for c in e.get("cases", [])}
    if not jev_cases:
        return "sem dado", "o Jev não julgou nada nesta rodada (sem TYPESAFE_API_KEY?): nada saiu da máquina"
    index = run.answers.get("rag_index")
    registered = _registered(index)
    if registered is None:
        return "falhou", f"o índice {index!r} não tem lista registrada: não dá para provar que o acervo é o público"
    sent = {d for a in run.answer_list if a["name"] in jev_cases for d in a.get("source_documents") or []}
    outside = sorted(sent - registered)
    return ("passou" if not outside else "falhou",
            f"{len(sent)} documento(s) nos trechos enviados, todos da lista do {index}" if not outside
            else f"fora da lista do {index}: {', '.join(outside)}")


EVIDENCE: dict[str, Callable[[RunData], Result]] = {f.__name__: f for f in (
    artigo_no_topk, trechos_iguais, fatos_da_referencia, fidelidade, mesmos_fatos_em_ingles, abstencao_sem_o_documento,
    respostas_concordam, estado_limpo, gerador_nao_julga, consistencia_de_posicao, verbosidade, vereditos_estaveis,
    mutantes_mortos, so_acervo_publico)}


def load_checklist(path: Path = CHECKLIST) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in data["itens"]:
        if not any(k in item for k in ("evidencia", "teste", "fora")):
            raise ValueError(f"o item {item['id']} não diz como é conferido (evidencia, teste ou fora)")
        if "evidencia" in item and item["evidencia"] not in EVIDENCE:
            raise ValueError(f"o item {item['id']} cita uma evidência que não existe: {item['evidencia']}")
    return data


def _test_exists(spec: str) -> tuple[bool, str]:
    """`arquivo::nome` (ou só o arquivo; vários separados por vírgula): o teste existe?"""
    missing = []
    # O comentário entre parênteses ("ao vivo, fora do CI") sai antes: a vírgula dele não separa testes.
    for part in (p.strip() for p in re.sub(r"\s*\([^)]*\)", "", spec).split(",")):
        path, _, name = part.partition("::")
        file = ROOT / path.strip()
        if not file.is_file() or (name and name not in file.read_text(encoding="utf-8")):
            missing.append(part)
    return (not missing, f"falta: {', '.join(missing)}" if missing else spec)


def evaluate(run: RunData | None, checklist: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """O estado de cada item. Sem rodada, só os itens de teste e os fora do escopo têm estado."""
    checklist = checklist or load_checklist()
    results = []
    for item in checklist["itens"]:
        state, detail = "sem dado", ""
        if "fora" in item:
            state, detail = "fora", item["fora"]
        elif "evidencia" in item and run is not None:
            state, detail = EVIDENCE[item["evidencia"]](run)
        if "teste" in item and state in ("sem dado", "passou"):
            exists, test_detail = _test_exists(item["teste"])
            if state == "sem dado":
                state, detail = ("coberto" if exists else "falhou"), test_detail
            else:
                detail += f"; e o teste {test_detail}"
        results.append({**{k: item.get(k) for k in ("id", "etapa", "item", "tipo", "verifica")}, "estado": state,
                        "detalhe": detail})
    return results


def summary(results: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r["estado"]] = out.get(r["estado"], 0) + 1
    return out


def markdown(results: list[dict[str, Any]]) -> str:
    lines = ["| | Etapa | Item | Tipo | Estado | Detalhe |", "| :--- | :--- | :--- | :---: | :---: | :--- |"]
    for r in results:
        lines.append(f"| {r['id']} | {r['etapa']} | {r['item']} | {r['tipo']} | {STATE_MARK[r['estado']]} "
                     f"{r['estado']} | {r['detalhe']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="O checklist de testes de RAG sobre uma rodada da pergunta única.")
    parser.add_argument("--rodada", type=Path, required=True, help="a pasta de uma rodada do rodar.py")
    parser.add_argument("--saida", type=Path, default=None, help="grava o resultado em JSON (padrão: <rodada>/checklist.json)")
    args = parser.parse_args(argv)
    try:
        results = evaluate(load_run(args.rodada))
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    out = args.saida or args.rodada / "checklist.json"
    out.write_text(json.dumps({"resumo": summary(results), "itens": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(markdown(results))
    print(f"\n{summary(results)}  ->  {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
