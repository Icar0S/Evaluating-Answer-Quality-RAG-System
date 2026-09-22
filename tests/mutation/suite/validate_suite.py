"""Valida a suíte de 30 casos contra o protocolo. Portão da semana 2.

O que se checa aqui não é estilo — é a condição que torna a RQ1 interpretável:
"as classes existem para garantir que cada camada do pipeline tenha ao menos um
caso capaz de matá-la. Sem isso, a RQ1 mede a suíte e não o pipeline" (§6.1).
Um operador sem nenhum caso capaz de matá-lo produz mutante sobrevivente por
construção, e o sobrevivente entraria no resultado como se fosse uma lacuna da
suíte quando é, na verdade, uma lacuna do desenho.

Uso:
    python -m tests.mutation.suite.validate_suite               # exige suíte pronta
    python -m tests.mutation.suite.validate_suite --allow-draft # durante a escrita
    python -m tests.mutation.suite.validate_suite --check-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.operators.apply import load_catalog
from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.suite.loader import CLASS_QUOTAS, PLACEHOLDER, SUITE_SIZE, load_assertions, load_cases

logger = get_logger("mutation.suite")

BEHAVIORS = {"answer", "abstain", "refuse", "clarify", "cite"}


def validate(allow_draft: bool = False) -> list[str]:
    problems: list[str] = []
    cases = load_cases()
    assertions, _defaults = load_assertions()
    catalog = load_catalog()

    # --- tamanho, unicidade e cotas por classe ---
    if len(cases) != SUITE_SIZE:
        problems.append(f"A suíte tem {len(cases)} casos; o protocolo fixa {SUITE_SIZE}.")

    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            problems.append(f"case_id duplicado: {case.case_id}")
        seen.add(case.case_id)

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.case_class] = counts.get(case.case_class, 0) + 1
    for class_name, quota in CLASS_QUOTAS.items():
        actual = counts.get(class_name, 0)
        if actual != quota:
            problems.append(f"Classe '{class_name}': {actual} casos, esperado {quota}.")
    for class_name in counts:
        if class_name not in CLASS_QUOTAS:
            problems.append(f"Classe desconhecida: '{class_name}'.")

    # --- integridade de cada caso ---
    for case in cases:
        if case.expected_behavior not in BEHAVIORS:
            problems.append(f"{case.case_id}: expected_behavior '{case.expected_behavior}' inválido.")
        if case.case_id not in assertions:
            problems.append(f"{case.case_id}: sem assertivas em assertions.yaml.")
        elif not assertions[case.case_id]:
            problems.append(f"{case.case_id}: lista de assertivas vazia.")
        unknown = [t for t in case.targets if t not in catalog.by_id]
        if unknown:
            problems.append(f"{case.case_id}: alvos inexistentes no catálogo: {unknown}")
        if not case.targets:
            problems.append(f"{case.case_id}: nenhum operador alvo declarado.")

        if not allow_draft and case.is_draft:
            problems.append(f"{case.case_id}: ainda é rascunho (status={case.status}).")

    if not allow_draft:
        for case_id, case_assertions in assertions.items():
            for index, assertion in enumerate(case_assertions):
                if assertion.get("todo"):
                    problems.append(f"{case_id}: assertiva #{index} marcada como todo.")
                values = assertion.get("values") or []
                if any(PLACEHOLDER in str(v) for v in values) or PLACEHOLDER in str(assertion.get("pattern", "")):
                    problems.append(f"{case_id}: assertiva #{index} ainda tem '{PLACEHOLDER}'.")

    # --- cobertura: todo operador precisa de ao menos um caso capaz de matá-lo ---
    covered: set[str] = set()
    for case in cases:
        covered.update(case.targets)
    uncovered = [op.id for op in catalog.operators if op.id not in covered]
    if uncovered:
        problems.append(
            "Operadores sem nenhum caso alvo (sobreviveriam por construção): " + ", ".join(uncovered)
        )

    # --- amarrações específicas de corpus ---
    roles_needed = {"secondary": "C3", "recent": "E2", "primary": "C1/C2/C4"}
    for role, operators in roles_needed.items():
        if not any(case.evidence_role == role for case in cases):
            problems.append(f"Nenhum caso com evidence_role='{role}' — {operators} ficariam sem alvo.")

    problems.extend(_validate_variant_link(cases, allow_draft))
    return problems


def _validate_variant_link(cases, allow_draft: bool) -> list[str]:
    """C2 e C4 só são observáveis se algum caso citar um valor que as variantes alteram."""
    variants_file = paths.CORPUS_VARIANTS / "variants.json"
    if not variants_file.exists():
        return ["corpus/variants/variants.json não existe — rode build_corpus.py --make-variants."]
    if allow_draft:
        return []

    report = json.loads(variants_file.read_text(encoding="utf-8"))
    altered: set[str] = set()
    for variant in report.get("variants", []):
        for substitution in variant.get("substitutions", []):
            altered.add(substitution["original"].strip())

    if not altered:
        return ["As variantes não alteraram nenhum valor — C2 e C4 seriam mutantes equivalentes."]

    hits = [case.case_id for case in cases if any(value in case.expected_output for value in altered)]
    if not hits:
        return [
            "Nenhuma resposta de referência cita um valor alterado pelas variantes prev/conflict. "
            "C2 e C4 sobreviveriam por construção — escolha para os casos c01/c15 valores que as "
            "variantes perturbam (ver corpus/variants/variants.json)."
        ]
    return []


def check_baseline() -> list[str]:
    """Confere que o baseline realmente se comporta como os casos declaram.

    Instrumento de validação da suíte, não resultado: se um caso 'fora do corpus'
    já é respondido pelo sistema não-mutado, ou a pergunta está no corpus ou o
    limiar de recuperação está frouxo — nos dois casos a suíte precisa de ajuste
    antes da campanha.
    """
    runs = [r for r in jsonl.read(paths.RUNS_JSONL) if r.get("mutant_id") == "baseline"]
    if not runs:
        return ["results/runs.jsonl não tem execuções de baseline — rode run_campaign.py --baseline antes."]

    from tests.mutation.oracles.o2_assertions import abstained

    by_case: dict[str, list[dict]] = {}
    for run in runs:
        by_case.setdefault(run["case_id"], []).append(run)

    problems: list[str] = []
    for case in load_cases():
        case_runs = by_case.get(case.case_id, [])
        if not case_runs:
            problems.append(f"{case.case_id}: sem execução de baseline.")
            continue
        abstentions = sum(1 for run in case_runs if abstained(run["answer"]))
        total = len(case_runs)
        if case.expected_behavior in ("abstain", "refuse") and abstentions < total:
            problems.append(
                f"{case.case_id} ({case.expected_behavior}): o baseline respondeu em "
                f"{total - abstentions}/{total} execuções em vez de se abster."
            )
        if case.expected_behavior in ("answer", "cite") and abstentions > 0:
            message = (
                f"{case.case_id} ({case.expected_behavior}): o baseline se absteve em "
                f"{abstentions}/{total} execuções — a evidência não está sendo recuperada."
            )
            if case.accepted_gap:
                logger.warning("%s [lacuna aceita: %s]", message, case.accepted_gap)
            else:
                problems.append(message)
    return problems


def check_retrieval() -> list[str]:
    """Cada pergunta factual recupera o documento onde sua evidência mora?

    Descoberto no piloto: "Quantas classes foram selecionadas no total?" (c01)
    recuperava quatro documentos e nenhum era o `primary`. Num corpus onde todo
    artigo fala de "classes", pergunta genérica recupera o vizinho errado, o
    caso reprova no baseline por falta de evidência e sai de S — e c01 é a
    âncora de C2/C4.

    Custa uma chamada de embedding por caso, sem geração. Roda contra o índice
    do estudo sob a configuração do baseline, que é a que a campanha usa.
    """
    import json

    from tests.mutation.runner.sut import ensure_index, load_study_config, sut_configuration

    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    roles = manifest["roles"]

    def expected_documents(case) -> set[str]:
        if case.evidence_role in ("primary", "secondary"):
            return {roles[case.evidence_role]}
        if case.evidence_role == "recent":
            value = roles["recent"]
            return set(value if isinstance(value, list) else [value])
        # `any`: a dica de evidência nomeia o documento
        return {d["document"] for d in manifest["documents"] if d["document"][:-4] in case.evidence_hint}

    # Descoberto na semana 6, no baseline: conferir o documento não basta. Em 10
    # dos 19 casos factuais o documento certo vinha no top-4 e a PÁGINA da
    # evidência não — e o modelo respondia "o texto não menciona" com o fato a
    # duas páginas de distância. Quando a dica de evidência nomeia páginas
    # ("doc pág. N"), a checagem passa a exigir que ao menos uma delas venha.
    import re

    hint_re = re.compile(r"([a-z0-9_]+)\s+p[áa]g\.\s*(\d+)", re.I)

    problems: list[str] = []
    assertions, _defaults = load_assertions()
    study = load_study_config()
    with sut_configuration(study.baseline_overrides) as resolved:
        ensure_index(resolved)
        from app.rag import retrieval

        for case in load_cases():
            if case.expected_behavior not in ("answer", "cite") or case.is_draft:
                continue
            wanted = expected_documents(case)
            if not wanted:
                continue
            chunks, _ = retrieval.retrieve(case.question)
            got = {c["document"] for c in chunks}
            if not (wanted & got):
                problems.append(
                    f"{case.case_id}: a pergunta não recupera {sorted(wanted)} — veio "
                    f"{sorted(got) or 'nada'}. Reescreva a pergunta com termos específicos do documento."
                )
                continue
            pages = [(doc, int(page)) for doc, page in hint_re.findall(case.evidence_hint)]
            if not pages:
                continue
            ids = [c["chunk_id"] for c in chunks]
            hit = [f"{doc} p{page}" for doc, page in pages
                   if any(cid.startswith(f"{doc}::p{page}::") for cid in ids)]
            # Página certa não basta: uma página tem dois chunks e o valor mora em
            # um deles (baseline v2, c02). Quando o caso tem assertivas de valor,
            # exige que algum chunk recuperado contenha cada valor.
            values = [v for a in assertions.get(case.case_id, []) if a.get("type") == "contains_all"
                      for v in a.get("values", [])]
            if hit and values:
                from tests.mutation.oracles.base import normalize
                context = normalize(" ".join(c["text"] for c in chunks))
                missing = [v for v in values if normalize(v) not in context]
                if missing and not case.accepted_gap:
                    problems.append(
                        f"{case.case_id}: a página da evidência vem, mas o contexto recuperado não contém "
                        f"{missing} — o valor está no outro chunk da página. Reescreva com termos do trecho do valor."
                    )
                    continue
            if not hit and case.accepted_gap:
                logger.warning("%s: lacuna aceita (%s) — nenhuma página da evidência no top-%d; o caso sai de S.",
                               case.case_id, case.accepted_gap, len(ids))
            elif not hit:
                problems.append(
                    f"{case.case_id}: o documento vem, mas nenhuma página da evidência "
                    f"({', '.join(f'{d} p{p}' for d, p in pages)}) entra no top-{len(ids)} — "
                    f"veio {[i.split('::', 1)[1] for i in ids if i.startswith(pages[0][0])] or 'outras páginas'}. "
                    f"Reescreva a pergunta com termos do trecho da evidência."
                )
            elif len(hit) < len(pages):
                logger.warning("%s: só %s das páginas da evidência entram no top-k (%s).",
                               case.case_id, len(hit), ", ".join(hit))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida a suíte de casos do estudo de mutação.")
    parser.add_argument("--allow-draft", action="store_true", help="tolera casos ainda em rascunho")
    parser.add_argument("--check-baseline", action="store_true", help="confere o comportamento no baseline")
    parser.add_argument(
        "--check-retrieval",
        action="store_true",
        help="cada pergunta factual recupera o documento da sua evidência? (só embeddings)",
    )
    args = parser.parse_args(argv)

    problems = validate(allow_draft=args.allow_draft)
    if args.check_retrieval:
        problems.extend(check_retrieval())
    if args.check_baseline:
        problems.extend(check_baseline())

    cases = load_cases()
    drafts = [case.case_id for case in cases if case.is_draft]
    logger.info("%d casos carregados; %d ainda em rascunho.", len(cases), len(drafts))
    if drafts:
        logger.info("Rascunhos: %s", ", ".join(drafts))

    if problems:
        logger.error("VALIDAÇÃO FALHOU (%d problema(s)):", len(problems))
        for problem in problems:
            logger.error("  - %s", problem)
        return 1

    logger.info("VALIDAÇÃO OK — a suíte atende ao protocolo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
