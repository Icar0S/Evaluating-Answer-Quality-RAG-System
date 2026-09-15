"""Propõe rascunhos dos casos de teste a partir do corpus indexado.

**Isto não escreve a suíte por você.** As propostas saem em suite/drafts.jsonl,
nunca em golden.jsonl, e cada uma precisa ser conferida contra o documento antes
de virar caso. Uma resposta de referência errada não é um caso ruim: é um
oráculo errado, que reprova o baseline e contamina S, a campanha inteira e as
três RQs. O protocolo chama a escrita dos 30 casos de "item de caminho crítico
com maior tempo humano irredutível" (§12) — o que este script corta é a parte
mecânica (achar o trecho, formatar), não o julgamento.

O que ele faz de útil além de rascunhar:

- respeita o papel de evidência do slot (`primary`, `secondary`, `recent`), que é
  o que amarra os casos aos operadores C1-C4 e E2;
- para `fato_distribuido`, junta dois trechos distantes, que é o que dá sentido a
  R1 e K3;
- para `secondary` (alvo de C3, truncagem de -30%), busca no terço final do
  documento — evidência no começo faria C3 sobreviver por construção;
- para os slots que miram C2/C4, prefere trechos com valores numéricos que as
  variantes perturbam (corpus/variants/variants.json).

Uso:
    python -m tests.mutation.suite.draft_cases --all
    python -m tests.mutation.suite.draft_cases --case c03
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import jsonl, paths
from tests.mutation.runner.console import get_logger
from tests.mutation.runner.sut import ensure_index, load_study_config, sut_configuration
from tests.mutation.suite.loader import load_cases

logger = get_logger("mutation.drafts")

DRAFTS_PATH = paths.SUITE_DIR / "drafts.jsonl"

PROMPT_BY_CLASS = {
    "fato_direto": (
        "Escreva UMA pergunta cuja resposta seja um valor único e explícito do TRECHO, "
        "e a resposta de referência correspondente. Prefira um valor numérico."
    ),
    "fato_distribuido": (
        "Escreva UMA pergunta cuja resposta exija combinar informação dos DOIS trechos "
        "abaixo — uma pergunta respondível por apenas um deles não serve."
    ),
    "filtro_condicional": (
        "Escreva UMA pergunta cuja resposta dependa de uma condição declarada no TRECHO. "
        "A resposta de referência deve enunciar a condição, não só o resultado."
    ),
    "rastreabilidade": (
        "Escreva UMA pergunta factual sobre o TRECHO. A resposta de referência deve "
        "terminar citando a fonte no formato [documento, pág. N] com os valores dados."
    ),
}

SYSTEM_PROMPT = """Você ajuda a montar uma suíte de testes para um assistente de perguntas e respostas sobre documentos técnicos.

Regras:
1. A pergunta precisa ser respondível SOMENTE com o trecho fornecido.
2. A resposta de referência precisa ser curta (1 a 3 frases), factual e literal em relação ao trecho.
3. Nunca invente valores: todo número, nome ou condição citado precisa aparecer no trecho.
4. Escreva em português do Brasil.

Responda SOMENTE com o objeto JSON pedido."""

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "expected_output": {"type": "string"},
        "key_values": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "expected_output", "key_values"],
}


def _altered_values() -> set[str]:
    variants_file = paths.CORPUS_VARIANTS / "variants.json"
    if not variants_file.exists():
        return set()
    report = json.loads(variants_file.read_text(encoding="utf-8"))
    return {
        substitution["original"].strip()
        for variant in report.get("variants", [])
        for substitution in variant.get("substitutions", [])
    }


def _documents_for_role(role: str, manifest: dict) -> list[str]:
    roles = manifest.get("roles", {})
    if role == "any":
        return [doc["document"] for doc in manifest["documents"]]
    value = roles.get(role)
    if value is None:
        return [doc["document"] for doc in manifest["documents"]]
    return list(value) if isinstance(value, list) else [value]


def _chunks_of(document: str) -> list[dict]:
    """Chunks indexados de um documento, na ordem de página."""
    from app.rag import vector_store

    collection = vector_store.get_collection()
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"chunk_id": chunk_id, "text": text, "page": (metadata or {}).get("page")}
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
        if (metadata or {}).get("document") == document
    ]
    return sorted(chunks, key=lambda chunk: (chunk["page"] or 0, chunk["chunk_id"]))


def _pick_chunks(case, manifest: dict, rng: random.Random, prefer_altered: set[str]) -> list[dict]:
    documents = _documents_for_role(case.evidence_role, manifest)
    rng.shuffle(documents)

    for document in documents:
        chunks = _chunks_of(document)
        if not chunks:
            continue

        if case.evidence_role == "secondary":
            # C3 remove os últimos 30%: a evidência precisa morar lá.
            chunks = chunks[int(len(chunks) * 0.7) :] or chunks

        if prefer_altered:
            with_values = [c for c in chunks if any(value in c["text"] for value in prefer_altered)]
            chunks = with_values or chunks

        if case.case_class == "fato_distribuido" and len(chunks) >= 2:
            first = rng.randrange(0, len(chunks) - 1)
            second = rng.randrange(first + 1, len(chunks))
            return [chunks[first], chunks[second]]
        return [rng.choice(chunks)]
    return []


def _ask_model(case, chunks: list[dict], document: str, model: str, base_url: str, timeout: int) -> dict:
    instruction = PROMPT_BY_CLASS.get(case.case_class, PROMPT_BY_CLASS["fato_direto"])
    blocks = "\n\n".join(
        f"[TRECHO {index} — {document}, pág. {chunk.get('page', '?')}]\n{chunk['text'][:3000]}"
        for index, chunk in enumerate(chunks, start=1)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n{blocks}"},
        ],
        "stream": False,
        "format": DRAFT_SCHEMA,
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
        return json.loads(response.json().get("message", {}).get("content", "{}"))


def _verify(proposal: dict, chunks: list[dict]) -> list[str]:
    """Conferência automática rasa — não substitui a leitura humana.

    Só pega o erro mais comum e mais caro do rascunho automático: a resposta citar
    um valor que não está no trecho. O que passa daqui ainda precisa de olho.
    """
    context = " ".join(chunk["text"] for chunk in chunks)
    problems = []
    for value in proposal.get("key_values", []):
        if value and str(value) not in context:
            problems.append(f"valor '{value}' não aparece no trecho")
    for number in re.findall(r"\d[\d.,]{1,}", proposal.get("expected_output", "")):
        if number not in context:
            problems.append(f"número '{number}' da resposta não aparece no trecho")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propõe rascunhos de casos a partir do corpus.")
    parser.add_argument("--case", help="rascunha só um case_id")
    parser.add_argument("--all", action="store_true", help="rascunha todos os casos ainda em rascunho")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)

    if not (args.case or args.all):
        parser.error("use --case <id> ou --all")

    paths.ensure_dirs()
    study = load_study_config()
    model = study.models["generation"]
    base_url = study.baseline_overrides.get("ollama_base_url") or "http://localhost:11434"
    timeout = int(study.baseline_overrides.get("generation_timeout_seconds", 300))
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))

    cases = load_cases()
    if args.case:
        cases = [case for case in cases if case.case_id == args.case]
        if not cases:
            raise SystemExit(f"case_id {args.case} não existe em golden.jsonl")
    else:
        cases = [case for case in cases if case.is_draft and case.evidence_role != "none"]

    if not cases:
        logger.info("Nenhum caso a rascunhar.")
        return 0

    rng = random.Random(args.seed)
    altered = _altered_values()
    proposals: list[dict] = []

    with sut_configuration(study.baseline_overrides) as resolved:
        # O rascunho sai dos chunks indexados, entao o indice do estudo precisa
        # existir. Construi-lo aqui evita a dependencia circular do runbook:
        # calibrar o piso de recuperacao exige casos prontos, e escrever casos
        # exige indice.
        ensure_index(resolved)
        for case in cases:
            wants_altered = altered if {"C2", "C4"} & set(case.targets) else set()
            chunks = _pick_chunks(case, manifest, rng, wants_altered)
            if not chunks:
                logger.warning("%s: nenhum chunk disponível para o papel '%s'", case.case_id, case.evidence_role)
                continue

            document = chunks[0]["chunk_id"].split("::")[0]
            try:
                proposal = _ask_model(case, chunks, document, model, base_url, timeout)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                logger.error("%s: falha ao rascunhar — %s", case.case_id, exc)
                continue

            problems = _verify(proposal, chunks)
            proposals.append(
                {
                    "case_id": case.case_id,
                    "class": case.case_class,
                    "evidence_role": case.evidence_role,
                    "targets": case.targets,
                    "question": proposal.get("question", ""),
                    "expected_output": proposal.get("expected_output", ""),
                    "key_values": proposal.get("key_values", []),
                    "evidence_hint": ", ".join(
                        f"{document} pág. {chunk.get('page', '?')}" for chunk in chunks
                    ),
                    "source_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                    "auto_check": problems or ["ok"],
                    "reviewed": False,
                }
            )
            status = "REVISAR" if problems else "ok"
            logger.info("%s [%s] %s", case.case_id, status, proposal.get("question", "")[:80])

    jsonl.write_all(DRAFTS_PATH, proposals)
    flagged = sum(1 for p in proposals if p["auto_check"] != ["ok"])
    logger.info("%d rascunhos em %s (%d com pendência automática)", len(proposals), DRAFTS_PATH, flagged)
    logger.info(
        "Próximo passo: conferir cada rascunho contra o documento, ajustar o texto e "
        "copiar para suite/golden.jsonl com \"status\": \"ready\". Depois, escrever as "
        "assertivas em suite/assertions.yaml e rodar validate_suite.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
