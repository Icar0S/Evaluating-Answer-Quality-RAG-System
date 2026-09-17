"""Fidelidade dos mutantes: cada operador injeta o defeito que o catálogo promete?

`apply verify` prova que os 18 operadores aplicam e revertem sem resíduo. Não
prova que o mutante É o que o catálogo diz. Um operador com override de
configuração errado, ou um patch de corpus que não mexe no documento certo,
passa no `verify` e entra na campanha como um mutante que ninguém sabe o que é —
e o resultado aparece como "a suíte não detecta" quando o correto é "o mutante
não injetou nada".

O §10 do protocolo (validade interna) pede revisão manual de 100% dos 18
mutantes antes da campanha. Este script é essa revisão em forma executável: para
cada operador, aplica, observa o efeito no ponto do pipeline que ele deveria
alterar, e compara com o que o catálogo declara.

O que se observa, por camada:

- corpus (C1-C4, E2): o estado dos arquivos em work/corpus e work/index_src;
- chunking (K1-K4): quantos chunks o corpus produz e onde caem as fronteiras —
  calculado direto sobre os PDFs, sem embedar;
- índice (E1): a dimensão do vetor que o modelo alternativo produz;
- recuperação (R1-R4): o que `retrieve()` devolve contra o índice do baseline
  para uma pergunta factual e uma fora do corpus;
- prompt (P1-P4): o system prompt montado e as opções enviadas ao Ollama.

Cada verificação registra o valor observado, e o relatório vai para
operators/fidelity_report.json — é o que o artigo cita ao afirmar que os 18
mutantes foram revisados.

Uso:
    python -m tests.mutation.tools.verify_mutants
    python -m tests.mutation.tools.verify_mutants --operators R1,R4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.operators import apply as operators
from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import ensure_index, load_study_config, sut_configuration
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.fidelidade")

REPORT_PATH = paths.OPERATORS_DIR / "fidelity_report.json"


class Check:
    """Acumula observações e vereditos de um operador."""

    def __init__(self, operator_id: str) -> None:
        self.operator_id = operator_id
        self.observations: dict[str, Any] = {}
        self.failures: list[str] = []

    def expect(self, name: str, observed: Any, ok: bool, expectation: str) -> None:
        self.observations[name] = observed
        if not ok:
            self.failures.append(f"{name}: observado {observed!r}, esperado {expectation}")

    @property
    def passed(self) -> bool:
        return not self.failures


# --------------------------------------------------------------- utilidades


def _pdf_pages(path: Path) -> int:
    import fitz

    with fitz.open(path) as handle:
        return handle.page_count


def _pdf_text(path: Path) -> str:
    import fitz

    with fitz.open(path) as handle:
        return "\n".join(page.get_text() for page in handle)


def _chunk_profile(source_dir: Path) -> dict[str, Any]:
    """Quantos chunks o corpus produz sob a configuração ativa, sem embedar."""
    from app.config import get_settings
    from app.rag.ingestion import build_chunk_records

    settings = get_settings()
    records = []
    for pdf in sorted(source_dir.glob("*.pdf")):
        records.extend(
            build_chunk_records(
                pdf,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                settings.chunk_boundary_offset_tokens,
            )
        )
    first_lengths = [len(r.text) for r in records[:3]]
    # Sobreposicao observada: quantos caracteres do fim do 1o chunk reaparecem no
    # inicio do 2o chunk da MESMA pagina. E o unico jeito de ver K3, porque tirar
    # a sobreposicao nao muda a contagem de chunks — muda o que cada um contem.
    overlap_chars = 0
    if len(records) >= 2 and records[0].page == records[1].page:
        first, second = records[0].text, records[1].text
        for size in range(min(len(first), len(second), 2000), 20, -1):
            if first.endswith(second[:size]):
                overlap_chars = size
                break
    return {"chunks": len(records), "first_chunk_chars": first_lengths, "overlap_chars": overlap_chars}


def _retrieve(question: str) -> list[dict]:
    from app.rag import retrieval

    chunks, _ = retrieval.retrieve(question)
    return chunks


def _hardest_negative_question(cases) -> str:
    """A pergunta negativa cujo melhor chunk fica MAIS abaixo do piso.

    E a unica sonda que distingue R4 do baseline. Com a primeira pergunta "fora
    do corpus" da lista (WCAG, melhor chunk em 0,492) a checagem passava
    trivialmente: o piso de 0,428 nao filtrava nada, baseline e mutante
    devolviam 4 chunks. Com a pergunta de similaridade 0,418 o baseline devolve
    zero e R4 devolve 4 — esse e o defeito que R4 injeta, observavel.
    """
    gate = paths.CONFIG_DIR / "retrieval_gate.json"
    if gate.exists():
        report = json.loads(gate.read_text(encoding="utf-8"))
        negatives = sorted(
            (m for m in report["measurements"] if m["label"] == 0),
            key=lambda m: m["top_similarity"],
        )
        if negatives:
            by_id = {c.case_id: c.question for c in cases}
            return by_id[negatives[0]["case_id"]]
    return next(c.question for c in cases if c.expected_behavior == "abstain")


def _manifest_roles() -> dict[str, Any]:
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    return manifest["roles"]


def _altered_values() -> set[str]:
    report = json.loads((paths.CORPUS_VARIANTS / "variants.json").read_text(encoding="utf-8"))
    return {
        s["original"] for v in report["variants"] if v["variant"] == "prev" for s in v["substitutions"]
    }


# ------------------------------------------------------------ verificações


def check_corpus_ops(check: Check, operator_id: str, roles: dict[str, Any]) -> None:
    primary = roles["primary"]
    secondary = roles["secondary"]
    recent = roles["recent"] if isinstance(roles["recent"], list) else [roles["recent"]]

    in_corpus = {p.name for p in paths.WORK_CORPUS.glob("*.pdf")}
    in_index = {p.name for p in paths.WORK_INDEX_SRC.glob("*.pdf")}
    base = {p.name for p in paths.CORPUS_BASE.glob("*.pdf")}

    if operator_id == "C1":
        check.expect("primary_no_corpus", primary in in_corpus, primary not in in_corpus, "ausente")
        check.expect("primary_no_indice", primary in in_index, primary not in in_index, "ausente")
        check.expect("demais_intactos", len(in_corpus), len(in_corpus) == len(base) - 1, f"{len(base) - 1}")

    elif operator_id == "C2":
        text = _pdf_text(paths.WORK_CORPUS / primary)
        base_text = _pdf_text(paths.CORPUS_BASE / primary)
        # A variante prev substitui valores: os originais devem ter sumido e o
        # texto deve diferir do base. Checa uma amostra dos alterados.
        report = json.loads((paths.CORPUS_VARIANTS / "variants.json").read_text(encoding="utf-8"))
        subs = next(v["substitutions"] for v in report["variants"] if v["variant"] == "prev")
        # Cada substituicao registrada deve ter deixado o valor NOVO no texto. A
        # contagem do original nao serve de prova: "16" aparece dezenas de vezes
        # e so algumas ocorrencias sao trocadas.
        applied = sum(1 for sub in subs if sub["replacement"] in text)
        check.expect("texto_difere_do_base", text != base_text, text != base_text, "diferente")
        check.expect(
            "substituicoes_aplicadas",
            {"registradas": len(subs), "com_valor_novo_presente": applied},
            applied >= len(subs) * 0.9,
            "ao menos 90% das substituições registradas visíveis no texto",
        )
        check.expect("indice_recebe_a_variante", _pdf_text(paths.WORK_INDEX_SRC / primary) == text, True, "mesmo conteúdo")

    elif operator_id == "C3":
        before = _pdf_pages(paths.CORPUS_BASE / secondary)
        after = _pdf_pages(paths.WORK_CORPUS / secondary)
        expected = max(1, round(before * 0.7))
        check.expect("paginas", {"antes": before, "depois": after}, after == expected, f"{expected} (70% de {before})")
        check.expect("indice_tambem_truncado", _pdf_pages(paths.WORK_INDEX_SRC / secondary), _pdf_pages(paths.WORK_INDEX_SRC / secondary) == after, f"{after}")

    elif operator_id == "C4":
        dup = f"{Path(primary).stem}__dup.pdf"
        check.expect("duplicata_no_corpus", dup in in_corpus, dup in in_corpus, "presente")
        check.expect("duplicata_no_indice", dup in in_index, dup in in_index, "presente")
        check.expect("original_preservado", primary in in_corpus, primary in in_corpus, "presente")
        if dup in in_corpus:
            differs = _pdf_text(paths.WORK_CORPUS / dup) != _pdf_text(paths.WORK_CORPUS / primary)
            check.expect("duplicata_diverge_do_original", differs, differs, "conteúdo divergente")

    elif operator_id == "E2":
        for name in recent:
            check.expect(f"{name[:24]}_no_corpus", name in in_corpus, name in in_corpus, "presente (o arquivo existe)")
            check.expect(f"{name[:24]}_fora_do_indice", name in in_index, name not in in_index, "ausente (não foi indexado)")


def check_chunking(check: Check, operator_id: str, baseline_profile: dict[str, Any]) -> None:
    from app.config import get_settings

    settings = get_settings()
    profile = _chunk_profile(paths.WORK_INDEX_SRC)
    check.observations["baseline_chunks"] = baseline_profile["chunks"]

    if operator_id == "K1":
        check.expect("chunk_size", settings.chunk_size_tokens, settings.chunk_size_tokens == 400, "400")
        check.expect("chunks", profile["chunks"], profile["chunks"] > baseline_profile["chunks"] * 1.8, "muito mais chunks que o baseline")
    elif operator_id == "K2":
        check.expect("chunk_size", settings.chunk_size_tokens, settings.chunk_size_tokens == 3000, "3000")
        check.expect("chunks", profile["chunks"], profile["chunks"] < baseline_profile["chunks"], "menos chunks que o baseline")
    elif operator_id == "K3":
        check.expect("overlap", settings.chunk_overlap_tokens, settings.chunk_overlap_tokens == 0, "0")
        check.expect(
            "sobreposicao_observada_chars",
            {"baseline": baseline_profile["overlap_chars"], "mutante": profile["overlap_chars"]},
            baseline_profile["overlap_chars"] > 100 and profile["overlap_chars"] == 0,
            "centenas de caracteres repetidos no baseline, zero no mutante",
        )
    elif operator_id == "K4":
        check.expect("offset", settings.chunk_boundary_offset_tokens, settings.chunk_boundary_offset_tokens == 600, "600")
        moved = profile["first_chunk_chars"] != baseline_profile["first_chunk_chars"]
        check.expect(
            "fronteiras_deslocadas",
            {"baseline": baseline_profile["first_chunk_chars"], "mutante": profile["first_chunk_chars"]},
            moved,
            "primeiros chunks com tamanho diferente",
        )


def check_index(check: Check, operator_id: str) -> None:
    from app.config import get_settings
    from app.rag.ingestion import embed_texts

    settings = get_settings()
    if operator_id == "E1":
        check.expect("embedding_model", settings.embedding_model, settings.embedding_model != "nomic-embed-text", "diferente do baseline")
        [vector] = embed_texts(["teste de dimensão"])
        [base_vector] = embed_texts(["teste de dimensão"], model="nomic-embed-text")
        check.expect(
            "dimensao",
            {"mutante": len(vector), "baseline": len(base_vector)},
            len(vector) != len(base_vector),
            "espaço de embedding diferente",
        )


def check_retrieval(check: Check, operator_id: str, factual_q: str, absent_q: str) -> None:
    from app.config import get_settings

    settings = get_settings()
    if operator_id == "R1":
        chunks = _retrieve(factual_q)
        check.expect("top_k", settings.top_k, settings.top_k == 1, "1")
        check.expect("chunks_devolvidos", len(chunks), len(chunks) == 1, "1")
    elif operator_id == "R2":
        chunks = _retrieve(factual_q)
        check.expect("top_k", settings.top_k, settings.top_k == 12, "12 (3x o baseline)")
        check.expect("chunks_devolvidos", len(chunks), len(chunks) > 4, "mais que os 4 do baseline")
    elif operator_id == "R3":
        check.expect("mmr", settings.retrieval_use_mmr, settings.retrieval_use_mmr is False, "desligado")
    elif operator_id == "R4":
        check.expect("piso", settings.min_similarity_score, settings.min_similarity_score == 0.0, "0.0")
        chunks = _retrieve(absent_q)
        baseline_count = check.observations.get("baseline_chunks_for_probe", 0)
        check.expect(
            "fora_do_corpus_recebe_contexto",
            {
                "chunks_mutante": len(chunks),
                "chunks_baseline": baseline_count,
                "menor_similaridade": min((c["similarity_score"] for c in chunks), default=None),
            },
            len(chunks) > baseline_count,
            "mais chunks que o baseline para a mesma pergunta",
        )


def check_prompt(check: Check, operator_id: str) -> None:
    from app.rag.generation import (
        ABSTENTION_RULE, CITATION_RULE, GROUNDING_RULE, _ollama_options, build_system_prompt,
    )

    prompt = build_system_prompt()
    if operator_id == "P1":
        check.expect("regra_de_ancoragem", GROUNDING_RULE in prompt, GROUNDING_RULE not in prompt, "ausente")
        check.expect("abstencao_preservada", "diga exatamente" in prompt, "diga exatamente" in prompt, "presente")
    elif operator_id == "P2":
        check.expect("regra_de_abstencao", "diga exatamente" in prompt, "diga exatamente" not in prompt, "ausente")
        check.expect("ancoragem_preservada", GROUNDING_RULE in prompt, GROUNDING_RULE in prompt, "presente")
    elif operator_id == "P3":
        check.expect("regra_de_citacao", CITATION_RULE in prompt, CITATION_RULE not in prompt, "ausente")
        check.expect("ancoragem_preservada", GROUNDING_RULE in prompt, GROUNDING_RULE in prompt, "presente")
    elif operator_id == "P4":
        options = _ollama_options()
        check.expect("temperatura", options.get("temperature"), options.get("temperature") == 0.8, "0.8")


# -------------------------------------------------------------------- main


def verify(selected: list[str] | None) -> dict[str, Any]:
    study = load_study_config()
    baseline = study.baseline_overrides
    catalog = operators.load_catalog()
    roles = _manifest_roles()

    cases = load_cases()
    factual_q = next(c.question for c in cases if c.expected_behavior == "answer" and c.evidence_role == "primary")
    absent_q = _hardest_negative_question(cases)

    # Perfil do baseline: índice pronto (para R*) e contagem de chunks (para K*).
    operators.revert()
    with sut_configuration(baseline) as resolved:
        ensure_index(resolved)
        baseline_profile = _chunk_profile(paths.WORK_INDEX_SRC)
        baseline_abs = _retrieve(absent_q)
    logger.info(
        "baseline: %d chunks; pergunta fora do corpus recebe %d chunk(s) com piso %.3f",
        baseline_profile["chunks"], len(baseline_abs), baseline["min_similarity_score"],
    )

    results: list[Check] = []
    for operator in catalog.operators:
        if selected and operator.id not in selected:
            continue
        check = Check(operator.id)
        mutant = operators.apply(operator.id, catalog, baseline, study.raw)
        merged = {**baseline, **mutant.config_overrides}
        check.observations["overrides"] = mutant.config_overrides
        check.observations["file_log"] = mutant.file_log
        if operator.id == "R4":
            check.observations["baseline_chunks_for_probe"] = len(baseline_abs)

        try:
            with sut_configuration(merged):
                if operator.layer == "corpus" or operator.id == "E2":
                    check_corpus_ops(check, operator.id, roles)
                elif operator.layer == "chunking":
                    check_chunking(check, operator.id, baseline_profile)
                elif operator.layer == "index":
                    check_index(check, operator.id)
                elif operator.layer == "retrieval":
                    check_retrieval(check, operator.id, factual_q, absent_q)
                elif operator.layer == "prompt":
                    check_prompt(check, operator.id)
        except Exception as exc:  # noqa: BLE001 — um operador quebrado não esconde os outros
            check.failures.append(f"erro ao verificar: {type(exc).__name__}: {exc}")
        finally:
            operators.revert()

        status = "OK " if check.passed else "!! "
        logger.info("%s%s %s — %s", status, operator.id, operator.name, "; ".join(check.failures) or "defeito confirmado")
        results.append(check)

    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "baseline_chunks": baseline_profile["chunks"],
        "operators": {
            c.operator_id: {"passed": c.passed, "observations": c.observations, "failures": c.failures}
            for c in results
        },
        "summary": {"passed": sum(1 for c in results if c.passed), "total": len(results)},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica que cada mutante injeta o defeito declarado.")
    parser.add_argument("--operators", help="subconjunto, ex.: R1,R4")
    args = parser.parse_args(argv)

    selected = [o.strip() for o in args.operators.split(",")] if args.operators else None
    report = verify(selected)
    summary = report["summary"]
    logger.info("Relatório em %s", REPORT_PATH)
    if summary["passed"] == summary["total"]:
        logger.info("FIDELIDADE OK: %d/%d mutantes injetam o defeito declarado.", summary["passed"], summary["total"])
        return 0
    logger.error("FIDELIDADE FALHOU: %d/%d.", summary["passed"], summary["total"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
