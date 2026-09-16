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

from tests.mutation.corpus.build_corpus import looks_like_bibliography
from tests.mutation.operators import apply as operators
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
        "terminar citando a fonte entre colchetes, com o NOME DO ARQUIVO e o NÚMERO DA "
        "PÁGINA que aparecem no cabeçalho do trecho — copie-os literalmente, não "
        "escreva a palavra \"documento\" nem \"N\"."
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


def _pick_chunks(
    case,
    manifest: dict,
    rng: random.Random,
    prefer_altered: set[str],
    used: set[str],
    role_cursor: dict[str, int],
) -> list[dict]:
    """Escolhe o(s) trecho(s) que vao embasar o rascunho.

    Tres regras que so ficaram obvias depois de olhar a primeira leva de
    rascunhos:

    1. **Nao reaproveitar trecho.** Dois slots sorteados no mesmo chunk produzem
       a mesma pergunta (aconteceu com c04/c05 e c29/c30): dois dos seis casos de
       fato direto viravam duplicata e a suite perdia cobertura sem avisar.
    2. **Rodizio entre os documentos do papel.** `recent` tem dois documentos;
       pegar sempre o primeiro da lista embaralhada deixava o segundo sem
       nenhum caso, e E2 so seria morto pela metade do que remove.
    3. **Preferir trechos densos em fatos.** Um chunk cujo unico numero e o ano
       do cabecalho da conferencia nao sustenta um caso de fato direto.
    """
    documents = _documents_for_role(case.evidence_role, manifest)
    if len(documents) > 1:
        cursor = role_cursor.get(case.evidence_role, 0)
        documents = documents[cursor % len(documents):] + documents[: cursor % len(documents)]
        role_cursor[case.evidence_role] = cursor + 1
    else:
        rng.shuffle(documents)

    for document in documents:
        chunks = [
            c for c in _chunks_of(document)
            if c["chunk_id"] not in used and not looks_like_bibliography(c["text"])
        ]
        if not chunks:
            continue

        if case.evidence_role == "secondary":
            # C3 remove os ultimos 30%: a evidencia precisa morar la.
            cut = _c3_cut_page(document)
            tail = [c for c in chunks if c["page"] and c["page"] > cut]
            tail = [c for c in tail if _factual_density(c["text"]) > 0]
            if not tail:
                # Degradacao silenciosa aqui custaria caro: o caso sai parecendo
                # normal e C3 fica sem quem o mate. Acontece quando a cauda do
                # documento `secondary` tem poucas paginas de prosa e elas ja
                # foram usadas por outro caso — o sinal e para trocar o papel,
                # nao para aceitar o rascunho.
                logger.warning(
                    "%s: nenhum trecho com fato sobrou na cauda de %s (paginas > %d) — "
                    "o rascunho vai sair de outro ponto do documento e NAO sustenta C3. "
                    "Rode tools/audit_roles.py: provavelmente o papel `secondary` precisa "
                    "de um documento com mais prosa no terco final.",
                    case.case_id, document[:40], cut,
                )
            chunks = tail or chunks

        if prefer_altered:
            with_values = [c for c in chunks if any(value in c["text"] for value in prefer_altered)]
            chunks = with_values or chunks

        if case.case_class in ("fato_direto", "filtro_condicional", "rastreabilidade"):
            chunks = sorted(chunks, key=lambda c: -_factual_density(c["text"]))[: max(3, len(chunks) // 3)]

        if case.case_class == "fato_distribuido" and len(chunks) >= 2:
            first = rng.randrange(0, len(chunks) - 1)
            second = rng.randrange(first + 1, len(chunks))
            picked = [chunks[first], chunks[second]]
        else:
            picked = [rng.choice(chunks)]

        used.update(c["chunk_id"] for c in picked)
        return picked
    return []


def _c3_cut_page(document: str) -> int:
    """Ultima pagina que C3 preserva — abaixo dela a evidencia sobrevive a truncagem."""
    import fitz

    with fitz.open(paths.CORPUS_BASE / document) as handle:
        total = handle.page_count
    return max(1, round(total * 0.7))


def _factual_density(text: str) -> int:
    from tests.mutation.tools.audit_roles import count_factual_numbers

    return count_factual_numbers(text)


def _ask_model(
    case,
    chunks: list[dict],
    document: str,
    model: str,
    base_url: str,
    timeout: int,
    altered: set[str] | None = None,
) -> dict:
    instruction = PROMPT_BY_CLASS.get(case.case_class, PROMPT_BY_CLASS["fato_direto"])

    if altered and {"C2", "C4"} & set(case.targets):
        # Escolher um trecho que CONTEM valor perturbado nao bastava: o modelo
        # via o trecho inteiro e respondia citando outro numero qualquer. C2 e C4
        # so sao observaveis se a resposta de referencia citar justamente um dos
        # valores que as variantes mudam, entao a exigencia vai explicita.
        context_text = " ".join(chunk["text"] for chunk in chunks)
        present = sorted(value for value in altered if value in context_text)
        if present:
            instruction += (
                "\n\nOBRIGATORIO: a resposta de referencia precisa citar "
                "literalmente um destes valores do trecho: "
                f"{', '.join(present[:8])}."
            )
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


PLACEHOLDER_LIKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+\[")
NUMBER_IN_TEXT = re.compile(r"\d+(?:[.,]\d+)*%?")
TEMPLATE_LEFTOVER = re.compile(r"\[documento|\[doc,|p[aá]g\.?:", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    """Numeros normalizados: sem separador decimal e sem pontuacao de borda.

    "42,6%" e "42.6%" sao o mesmo fato. O modelo escreve em pt-BR e o corpus em
    en-US, entao comparar as duas grafias como strings cruas reprovava rascunho
    correto — foi 40% do ruido da primeira rodada de conferencia.
    """
    found = set()
    for raw in NUMBER_IN_TEXT.findall(text):
        cleaned = raw.rstrip(".,").replace(",", ".")
        if cleaned:
            found.add(cleaned)
    return found


def _verify(proposal: dict, chunks: list[dict], case, altered: set[str]) -> list[str]:
    """Conferencia automatica rasa — nao substitui a leitura humana.

    A regra e reportar so o que e verificavel e acionavel. A primeira versao
    checava todo `key_values` como literal, e o modelo costuma preencher aquilo
    com conceitos ("metricas tradicionais"): 15 dos 19 rascunhos vinham marcados,
    o que treina a pessoa a ignorar o aviso — pior do que nao avisar. Restam
    quatro checagens, todas com consequencia clara:

    1. numero da resposta que nao existe no trecho — indicio de invencao;
    2. caso que mira C2/C4 sem citar valor que as variantes perturbam — os dois
       operadores sobreviveriam por construcao;
    3. sobra de gabarito ("[documento, pag. N]" literal, `snake_case_id`) — o
       modelo devolveu o molde em vez de preencher;
    4. pagina citada diferente da pagina de onde o trecho veio.
    """
    context = " ".join(chunk["text"] for chunk in chunks)
    context_numbers = _numbers(context)
    answer = proposal.get("expected_output", "")
    problems: list[str] = []

    missing = sorted(_numbers(answer) - context_numbers)
    if missing:
        problems.append(f"numero(s) da resposta ausentes do trecho: {missing}")

    if {"C2", "C4"} & set(case.targets):
        if not (_numbers(answer) & {n.rstrip("%").replace(",", ".") for n in altered}):
            problems.append(
                "nao cita valor que as variantes prev/conflict alteram — C2 e C4 "
                "sobreviveriam por construcao"
            )

    if TEMPLATE_LEFTOVER.search(answer):
        problems.append("sobra de gabarito na resposta (cite o nome real do documento)")

    for value in proposal.get("key_values", []):
        text = str(value).strip()
        if PLACEHOLDER_LIKE.match(text) and text not in context:
            problems.append(f"placeholder nao preenchido: '{text}'")

    for chunk in chunks:
        if looks_like_bibliography(chunk["text"]):
            problems.append(
                f"trecho {chunk['chunk_id']} e lista de referencias — um caso que pergunta "
                "sobre entrada de bibliografia nao testa o pipeline"
            )

    pages = {str(chunk.get("page")) for chunk in chunks}
    for cited in re.findall(r"p[aá]g\.?\s*(\d+)", answer, re.IGNORECASE):
        if cited not in pages:
            problems.append(f"cita pag. {cited}, mas o trecho veio da(s) pag. {sorted(pages)}")

    return problems


def recheck() -> int:
    """Reaplica a conferencia sobre drafts.jsonl, sem chamar o modelo.

    Separar conferir de gerar importa: melhorar a heuristica custava 10 minutos
    de GPU a cada tentativa enquanto as duas coisas estavam no mesmo comando.
    """
    if not DRAFTS_PATH.exists():
        logger.error("%s nao existe — rode --all antes.", DRAFTS_PATH)
        return 1

    from app.rag import vector_store

    study = load_study_config()
    altered = _altered_values()
    cases = {case.case_id: case for case in load_cases()}
    rows = jsonl.read(DRAFTS_PATH)

    with sut_configuration(study.baseline_overrides):
        stored = vector_store.get_collection().get(include=["documents"])
        text_by_id = dict(zip(stored["ids"], stored["documents"]))

        for row in rows:
            chunks = [
                {"chunk_id": cid, "text": text_by_id.get(cid, ""), "page": _page_of(cid)}
                for cid in row["source_chunk_ids"]
            ]
            problems = _verify(row, chunks, cases[row["case_id"]], altered)
            row["auto_check"] = problems or ["ok"]

    jsonl.write_all(DRAFTS_PATH, rows)
    flagged = [r for r in rows if r["auto_check"] != ["ok"]]
    logger.info("%d rascunhos reconferidos; %d com pendencia.", len(rows), len(flagged))
    for row in flagged:
        logger.info("  %s: %s", row["case_id"], "; ".join(row["auto_check"]))
    return 0


def _page_of(chunk_id: str) -> int | None:
    match = re.search(r"::p(\d+)::", chunk_id)
    return int(match.group(1)) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propõe rascunhos de casos a partir do corpus.")
    parser.add_argument("--case", help="rascunha só um case_id")
    parser.add_argument("--all", action="store_true", help="rascunha todos os casos ainda em rascunho")
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="so reaplica a conferencia sobre drafts.jsonl, sem chamar o modelo",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)

    if args.recheck:
        return recheck()
    if not (args.case or args.all):
        parser.error("use --case <id>, --all ou --recheck")

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
    used_chunks: set[str] = set()
    role_cursor: dict[str, int] = {}
    proposals: list[dict] = []

    with sut_configuration(study.baseline_overrides) as resolved:
        # O rascunho sai dos chunks indexados, entao o indice do estudo precisa
        # existir. Construi-lo aqui evita a dependencia circular do runbook:
        # calibrar o piso de recuperacao exige casos prontos, e escrever casos
        # exige indice.
        operators.revert()   # work/ espelha corpus/base antes de indexar
        ensure_index(resolved)
        for case in cases:
            wants_altered = altered if {"C2", "C4"} & set(case.targets) else set()
            chunks = _pick_chunks(case, manifest, rng, wants_altered, used_chunks, role_cursor)
            if not chunks:
                logger.warning("%s: nenhum chunk disponível para o papel '%s'", case.case_id, case.evidence_role)
                continue

            document = chunks[0]["chunk_id"].split("::")[0]
            proposal = None
            for attempt in (1, 2):
                try:
                    proposal = _ask_model(case, chunks, document, model, base_url, timeout, altered)
                    break
                except httpx.TimeoutException:
                    # A primeira chamada paga o carregamento do modelo na VRAM, e
                    # numa placa de 8 GB ela vem logo depois da indexacao, que
                    # deixou o modelo de embedding ocupando o lugar. Foi assim que
                    # c01 falhou nas duas primeiras rodadas: e sempre o primeiro
                    # caso da lista que morre, nunca o mesmo caso por merito.
                    if attempt == 1:
                        logger.warning("%s: timeout (modelo carregando?) — tentando de novo", case.case_id)
                        continue
                    logger.error("%s: timeout nas duas tentativas", case.case_id)
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    logger.error("%s: falha ao rascunhar — %s", case.case_id, exc)
                    break
            if proposal is None:
                continue

            problems = _verify(proposal, chunks, case, altered)
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

    # Merge, nunca substituicao: `--case c01` reescrevendo o arquivo inteiro
    # apagaria os outros 17 rascunhos ja conferidos. Reger um caso e a operacao
    # mais natural do mundo depois de olhar um rascunho ruim, e ela nao pode
    # custar o trabalho feito nos demais.
    existing = {row["case_id"]: row for row in jsonl.read(DRAFTS_PATH)}
    for proposal in proposals:
        existing[proposal["case_id"]] = proposal
    merged = [existing[key] for key in sorted(existing)]
    jsonl.write_all(DRAFTS_PATH, merged)

    proposals = merged
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
