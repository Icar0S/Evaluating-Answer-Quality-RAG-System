"""Mede quão bem cada documento serve a cada papel do corpus, e checa a escolha atual.

Os papéis (`primary`, `secondary`, `recent`) decidem quais operadores conseguem
ser mortos por algum caso. Atribuídos errado, o operador vira mutante equivalente
e o resultado aparece como "a suíte não detecta" quando na verdade é "o desenho
não permitia detectar". A atribuição automática do build_corpus.py é alfabética —
serve para destravar o pipeline, não para rodar o estudo.

O que se mede, por papel:

- **primary** (C1, C2, C4): quantidade de valores numéricos factuais no documento
  inteiro. São eles que as variantes `prev`/`conflict` perturbam e que um caso
  precisa citar para que C2 e C4 sejam observáveis.

- **secondary** (C3, truncar -30%): quanto do trecho que C3 apaga é **prosa com
  fatos**, e não bibliografia. Este é o mais traiçoeiro: em artigo de conferência
  o terço final costuma ser justamente a lista de referências, e caso nenhum cita
  bibliografia. Quando isso acontece, C3 é equivalente por construção.

- **recent** (E2): fatos no documento inteiro, com preferência por documentos
  menores — E2 tira todos os `recent` do índice de uma vez, e um documento grande
  derrubaria casos que nada têm a ver com o defeito injetado.

Uso:
    python -m tests.mutation.tools.audit_roles
    python -m tests.mutation.tools.audit_roles --suggest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.corpus.build_corpus import (
    CITATION_SPAN,
    FACTUAL_CUES,
    NUMBER_PATTERN,
    looks_like_bibliography,
)
from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger

logger = get_logger("mutation.papeis")

C3_KEEP_FRACTION = 0.7          # espelha o operador C3 do catálogo

MIN_TAIL_FACTS = 10             # abaixo disso, C3 tem pouco o que matar
MIN_TAIL_PROSE = 0.5            # metade do trecho apagado precisa ser prosa


def count_factual_numbers(text: str) -> int:
    """Numeros que um caso de teste poderia citar — ignora marcadores de citacao."""
    spans = [(m.start(), m.end()) for m in CITATION_SPAN.finditer(text)]
    total = 0
    for match in NUMBER_PATTERN.finditer(text):
        begin = match.start(1)
        if any(low <= begin < high for low, high in spans):
            continue
        window = text[max(0, begin - 60) : match.end(1) + 60].lower()
        if match.group(2) == "%" or any(cue in window for cue in FACTUAL_CUES):
            total += 1
    return total


def measure() -> tuple[list[dict], dict]:
    """Mede PAGINA A PAGINA, que e a granularidade em que C3 opera.

    A versao anterior procurava o ponto onde a bibliografia comeca e tratava tudo
    dali em diante como referencia. O modelo estava errado para este corpus:
    nestes artigos a lista de referencias fica no MEIO e e seguida de apendice
    (tabelas de resultado, checklist de submissao). Medir por pagina evita
    decidir onde a bibliografia "termina" — pergunta que nao tem resposta limpa.
    """
    import fitz

    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for document in manifest["documents"]:
        with fitz.open(paths.CORPUS_BASE / document["document"]) as handle:
            pages = [page.get_text() for page in handle]

        keep = max(1, round(len(pages) * C3_KEEP_FRACTION))
        tail = pages[keep:]
        tail_detail = [
            {
                "page": keep + offset + 1,
                "bibliography": looks_like_bibliography(text),
                "facts": count_factual_numbers(text),
            }
            for offset, text in enumerate(tail)
        ]
        prose_pages = [d for d in tail_detail if not d["bibliography"]]

        rows.append(
            {
                "document": document["document"],
                "pages": len(pages),
                "c3_deletes": f"{keep + 1}-{len(pages)}" if tail else "-",
                "facts_total": sum(
                    count_factual_numbers(text)
                    for text in pages
                    if not looks_like_bibliography(text)
                ),
                "tail_pages": len(tail_detail),
                "tail_prose_pages": len(prose_pages),
                "tail_facts": sum(d["facts"] for d in prose_pages),
                "tail_detail": tail_detail,
            }
        )
    return rows, manifest.get("roles", {})


def check(rows: list[dict], roles: dict) -> list[str]:
    by_document = {row["document"]: row for row in rows}
    problems: list[str] = []

    primary = by_document.get(roles.get("primary", ""))
    if primary is None:
        problems.append("papel `primary` não aponta para um documento do corpus")
    elif primary["facts_total"] < 40:
        problems.append(
            f"primary ({primary['document'][:40]}) tem só {primary['facts_total']} valores "
            "factuais — as variantes prev/conflict terão pouco o que perturbar, e C2/C4 "
            "ficam sem âncora para os casos citarem"
        )

    secondary = by_document.get(roles.get("secondary", ""))
    if secondary is None:
        problems.append("papel `secondary` nao aponta para um documento do corpus")
    elif secondary["tail_facts"] == 0:
        # Unico veredito automatico que se sustenta: zero fato citavel no trecho
        # que C3 apaga significa que nenhum caso pode morrer por causa de C3.
        # Acima de zero a pergunta e "esses fatos valem um caso?", que e
        # julgamento — a ferramenta mostra a evidencia e a pessoa decide.
        problems.append(
            f"secondary ({secondary['document'][:40]}): nenhum fato citavel nas paginas "
            f"{secondary['c3_deletes']}, que sao as que C3 apaga. C3 seria equivalente "
            "por construcao"
        )

    recent = roles.get("recent") or []
    recent = recent if isinstance(recent, list) else [recent]
    if not recent:
        problems.append("papel `recent` vazio — E2 não teria o que remover do índice")
    for name in recent:
        row = by_document.get(name)
        if row is None:
            problems.append(f"papel `recent` aponta para documento inexistente: {name}")
        elif row["facts_total"] < 20:
            problems.append(
                f"recent ({name[:40]}): {row['facts_total']} fatos — pouco para sustentar "
                "os casos que dependem dele (c04, c05, c29)"
            )

    overlap = {roles.get("primary"), roles.get("secondary")} & set(recent)
    if overlap:
        problems.append(f"documento acumula papéis conflitantes: {overlap}")

    return problems


def suggest(rows: list[dict]) -> dict:
    """Melhor candidato para cada papel, pelas métricas acima."""
    by_facts = sorted(rows, key=lambda r: -r["facts_total"])
    fit_for_c3 = [r for r in rows if r["tail_facts"] > 0]
    secondary = max(fit_for_c3, key=lambda r: r["tail_facts"], default=None)

    taken = {secondary["document"]} if secondary else set()
    primary = next((r for r in by_facts if r["document"] not in taken), None)
    if primary:
        taken.add(primary["document"])
    # `recent`: melhor densidade por página entre os que sobraram, favorecendo os menores.
    remaining = sorted(
        (r for r in rows if r["document"] not in taken),
        key=lambda r: -(r["facts_total"] / max(1, r["pages"])),
    )
    return {
        "primary": primary["document"] if primary else None,
        "secondary": secondary["document"] if secondary else None,
        "recent": [r["document"] for r in remaining[:2]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita a atribuição de papéis do corpus.")
    parser.add_argument("--suggest", action="store_true", help="propõe papéis a partir das métricas")
    args = parser.parse_args(argv)

    rows, roles = measure()

    header = "\n{:<50}{:>5}{:>10}{:>7}{:>13}{:>12}".format(
        "documento", "pag", "C3 apaga", "fatos", "prosa/cauda", "fatos/cauda"
    )
    print(header)
    print("-" * 100)
    for row in sorted(rows, key=lambda r: -r["tail_facts"]):
        prose = f"{row['tail_prose_pages']}/{row['tail_pages']}"
        print(
            f"{row['document'][:48]:<50}{row['pages']:>5}{row['c3_deletes']:>10}"
            f"{row['facts_total']:>7}{prose:>13}{row['tail_facts']:>12}"
        )

    # Evidencia pagina a pagina do documento que ocupa o papel `secondary`: e a
    # escolha que nenhum limiar acertou em tres tentativas, entao a ferramenta
    # mostra o material e quem decide e voce.
    secondary_name = roles.get("secondary")
    secondary = next((r for r in rows if r["document"] == secondary_name), None)
    if secondary:
        print(f"\nPaginas que C3 apagaria de `secondary` ({secondary_name[:44]}):")
        for detail in secondary["tail_detail"]:
            kind = "bibliografia" if detail["bibliography"] else "prosa"
            print(f"  pag {detail['page']:>3}  {kind:<13} {detail['facts']:>3} fatos citaveis")
        print("  -> confira se esses fatos sustentam um caso de teste (c03, c11, c28).")

    print("\nPapeis atuais:")
    for role, value in roles.items():
        print(f"  {role:<10} {value}")

    problems = check(rows, roles)
    print()
    if problems:
        logger.warning("Atribuicao atual tem %d problema(s):", len(problems))
        for problem in problems:
            logger.warning("  - %s", problem)
    else:
        logger.info("Atribuicao de papeis OK para os operadores C1-C4 e E2.")

    if args.suggest:
        print("\nSugestão a partir das métricas (confira antes de adotar):")
        for role, value in suggest(rows).items():
            print(f"  {role:<10} {value}")
        print("\nPara adotar: edite `roles` em corpus/sources.yaml e rode")
        print("  python -m tests.mutation.corpus.build_corpus --roles-only")
        print("  python -m tests.mutation.corpus.build_corpus --make-variants   # se `primary` mudou")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
