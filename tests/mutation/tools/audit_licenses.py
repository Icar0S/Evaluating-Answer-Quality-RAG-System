"""Audita a licença de cada documento do corpus e grava o resultado no manifesto.

O §6.2 do protocolo exige licença que permita redistribuição, porque o corpus
viaja no pacote de replicação. Este script lê os PDFs de origem, procura as
declarações que editoras costumam imprimir na primeira página e classifica cada
documento em `yes` / `no` / `unknown`.

**Ele não decide licença por você e nunca inventa uma.** `unknown` significa
exatamente isso: o PDF não declarou nada e a resposta está na página do editor ou
do arXiv, não no arquivo. Preencher `unknown` com um palpite seria o pior
resultado possível — um pacote de replicação publicado sem direito de
redistribuição.

Troca de corpus exige rodar isto de novo: é a única forma de o manifesto não
mentir sobre o que pode ser publicado.

Uso:
    python -m tests.mutation.tools.audit_licenses
    python -m tests.mutation.tools.audit_licenses --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger

logger = get_logger("mutation.licencas")

# Ordem importa: a primeira regra que casar decide. Licença aberta explícita vem
# antes de copyright de editora, porque muitos PDFs trazem as duas coisas (o
# autor mantém o copyright E libera sob CC).
RULES: list[tuple[str, str, str, str]] = [
    (r"CC[\s-]?BY[\s-]?4\.0|Creative Commons[^.\n]{0,40}Attribution 4", "CC-BY-4.0", "yes",
     "Creative Commons Attribution 4.0 — redistribuição permitida com atribuição"),
    (r"CC[\s-]?BY[\s-]?SA", "CC-BY-SA", "yes", "Creative Commons ShareAlike — redistribuição permitida"),
    (r"CC[\s-]?BY[\s-]?NC", "CC-BY-NC", "unknown",
     "Creative Commons NonCommercial — confira se o uso do pacote de replicação é não-comercial"),
    (r"CC[\s-]?BY[\s-]?ND", "CC-BY-ND", "no",
     "Creative Commons NoDerivatives — os operadores de corpus criam obras derivadas"),
    (r"CC[\s-]?BY", "CC-BY", "yes", "Creative Commons Attribution — redistribuição permitida"),
    (r"licensed to ACM|ACM ISBN", "ACM", "no",
     "copyright licenciado à ACM — redistribuição do PDF não é garantida"),
    (r"©\s*\d{4}\s*IEEE|IEEE\s+DOI", "IEEE", "no",
     "copyright IEEE — redistribuição do PDF não é garantida"),
    (r"All rights reserved", "proprietária", "no", "todos os direitos reservados"),
]

IDENTIFIER_PATTERNS = (
    (r"arXiv:\s*(\d{4}\.\d{4,5})", "arXiv"),
    (r"\b(10\.\d{4,9}/[^\s,;)\]]+)", "DOI"),
    (r"(ceur-ws\.org[^\s,;)\]]*)", "CEUR"),
)


def _page_texts(pdf_path: Path) -> tuple[str, str]:
    """(texto para licenca, texto para identificador).

    A busca do DOI/arXiv fica restrita a PRIMEIRA pagina de proposito: a ultima
    pagina e bibliografia, e o primeiro DOI que aparece la e de uma referencia
    citada, nao do artigo — foi assim que o qaquest ganhou o DOI de um paper de
    psicologia social de 2004. Ja a declaracao de licenca precisa das duas
    pontas: o bloco de copyright da ACM vive no rodape da pagina 1, mas avisos
    de Creative Commons as vezes so aparecem na ultima.
    """
    import fitz

    with fitz.open(pdf_path) as document:
        pages = list(document)
        license_text = "\n".join(page.get_text() for page in pages[:2] + pages[-1:])
        identifier_text = pages[0].get_text() if pages else ""

    def squash(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    return squash(license_text), squash(identifier_text)


def classify(text: str) -> tuple[str, str, str]:
    for pattern, label, redistributable, note in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return label, redistributable, note
    return (
        "DESCONHECIDA",
        "unknown",
        "o PDF não declara licença — verifique na página do editor/arXiv antes de publicar",
    )


def find_identifier(text: str) -> str | None:
    """DOI ou arXiv ID, para conferir a licença na fonte sem caçar o paper de novo."""
    for pattern, kind in IDENTIFIER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{kind}:{match.group(1)}"
    return None


def audit() -> list[dict]:
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    source_dir = Path(manifest["source_dir"])
    results: list[dict] = []

    for document in manifest["documents"]:
        original = source_dir / document["source_file"]
        if not original.exists():
            results.append(
                {
                    "document": document["document"],
                    "license": "DESCONHECIDA",
                    "redistributable": "unknown",
                    "note": f"PDF de origem não encontrado em {original}",
                    "identifier": None,
                }
            )
            continue

        license_text, identifier_text = _page_texts(original)
        label, redistributable, note = classify(license_text)
        results.append(
            {
                "document": document["document"],
                "license": label,
                "redistributable": redistributable,
                "note": note,
                "identifier": find_identifier(identifier_text),
            }
        )
    return results


def write_into_manifest(results: list[dict]) -> None:
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    by_document = {row["document"]: row for row in results}
    for document in manifest["documents"]:
        row = by_document.get(document["document"])
        if not row:
            continue
        document["license"] = row["license"]
        document["redistributable"] = row["redistributable"]
        document["license_note"] = row["note"]
        if row["identifier"]:
            document["identifier"] = row["identifier"]
    manifest["license_audit"] = {
        "yes": sum(1 for r in results if r["redistributable"] == "yes"),
        "no": sum(1 for r in results if r["redistributable"] == "no"),
        "unknown": sum(1 for r in results if r["redistributable"] == "unknown"),
    }
    paths.CORPUS_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("manifest.json atualizado com a auditoria de licenças.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita as licenças do corpus do estudo.")
    parser.add_argument("--write", action="store_true", help="grava o resultado em corpus/manifest.json")
    args = parser.parse_args(argv)

    results = audit()
    print(f"\n{'documento':<50}{'licença':<16}{'redistr.':<10}identificador")
    print("-" * 104)
    for row in sorted(results, key=lambda r: (r["redistributable"] != "yes", r["document"])):
        print(
            f"{row['document'][:48]:<50}{row['license'][:14]:<16}"
            f"{row['redistributable']:<10}{row['identifier'] or '-'}"
        )

    blocked = [r for r in results if r["redistributable"] == "no"]
    unknown = [r for r in results if r["redistributable"] == "unknown"]
    print()
    if blocked:
        logger.warning(
            "%d documento(s) NÃO podem ir no pacote de replicação: %s",
            len(blocked), ", ".join(r["document"][:40] for r in blocked),
        )
    if unknown:
        logger.warning(
            "%d documento(s) sem licença declarada no PDF — confirme na fonte (o "
            "identificador acima leva direto) e ajuste RULES se for um caso recorrente.",
            len(unknown),
        )
    if not blocked and not unknown:
        logger.info("Todos os documentos são redistribuíveis. Corpus liberado para o Zenodo.")

    if args.write:
        write_into_manifest(results)
    else:
        logger.info("Use --write para gravar em corpus/manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
