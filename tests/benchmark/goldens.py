"""Goldens v2 (plano-benchmark.md, §7): esquema, validador e sorteio do split.

O arquivo (tests/benchmark/goldens/v2.json) é uma lista, no formato que o
tests/generate_answers.py já lê (`name`, `input`, `input_en`,
`expected_output`, `context`), com os campos do §7 ao lado. Só os goldens
`revisado` entram nos vereditos; `rascunho` é candidato à revisão humana.

O validador confere o esquema, e com `--evidencia` confere que cada fragmento
do `context` é texto literal da fonte, como a versão do índice o extrai: o
fragmento tem de casar com o documento inteiro (tests/benchmark/c1.py). É o que
garante que a evidência@k mede a recuperação, e não o jeito de escrever o
golden.

Uso, da raiz:

    backend\\.venv\\Scripts\\python.exe -m tests.benchmark.goldens validar --indice pdf-v1 --evidencia
    backend\\.venv\\Scripts\\python.exe -m tests.benchmark.goldens sortear-split --semente 20261018
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from tests.benchmark import c1

ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / "tests" / "benchmark" / "goldens" / "v2.json"
# Os outros conjuntos que podem ir no mesmo arquivo de respostas: os juízes indexam por nome.
OTHER_SETS = (ROOT / "tests" / "deepeval" / "goldens" / "dataset.json",
              ROOT / "tests" / "deepeval" / "goldens" / "dataset_md.json")

QUESTION_CLASSES = ("fato_direto", "fato_distribuido", "filtro_condicional", "rastreabilidade",
                    "fora_do_corpus", "fora_de_escopo", "ambiguo")
OUT_CLASSES = ("fora_do_corpus", "fora_de_escopo")
CONTENT_TYPES = ("tabela", "numero", "secao", "texto")  # nesta ordem de precedência
OUT_STRATA = ("perto", "longe", "escopo")
SPLITS = ("selecao", "confirmacao")
SETS = ("goldens", "demanda")
STATUSES = ("rascunho", "revisado")
REQUIRED = ("name", "input", "input_en", "expected_output", "context", "source_document", "question_class",
            "content_type", "has_acronym", "names_article", "out_stratum", "split", "set", "status")
OPTIONAL = ("observacao", "evidence_page")
# Nota editorial dentro do context quebra a evidência@k: o lugar dela é `observacao`.
_EDITORIAL_NOTE = re.compile(
    r"\(\s*(?:no markdown|nota|obs\.?|observa[cç][aã]o)\b|espa[cç]os restaurados", re.IGNORECASE
)
# Abaixo disto, o fragmento não é literal: alguma palavra foi trocada, traduzida ou resumida.
LITERAL_MIN = 0.95


def load(path: Path = V2_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_out(golden: dict) -> bool:
    return golden.get("question_class") in OUT_CLASSES


def _enum(golden: dict, field: str, allowed: tuple, errors: list[str], nullable: bool = False) -> None:
    value = golden.get(field)
    if value is None and nullable:
        return
    if value not in allowed:
        errors.append(f"{golden.get('name')}: {field}={value!r} fora de {allowed}")


def validate(goldens: list[dict], other_names: set[str] = frozenset(),
             index_documents: set[str] | None = None) -> list[str]:
    """Erros de esquema e de consistência (lista vazia = válido).

    `index_documents`: os documentos da versão, pelo nome sem extensão; com ele, todo source_document
    tem de estar na versão.
    """
    errors: list[str] = []
    names = Counter(g.get("name") for g in goldens)
    errors += [f"{name}: nome repetido no arquivo" for name, n in names.items() if n > 1]
    errors += [f"{name}: nome já usado em outro conjunto de goldens" for name in names if name in other_names]
    split_of_document: dict[str, set[str]] = defaultdict(set)
    for g in goldens:
        name = g.get("name")
        missing = [f for f in REQUIRED if f not in g]
        unknown = [f for f in g if f not in REQUIRED + OPTIONAL]
        if missing:
            errors.append(f"{name}: faltam {missing}")
        if unknown:
            errors.append(f"{name}: campos fora do esquema {unknown}")
        for field in ("input", "input_en", "expected_output"):
            if not str(g.get(field) or "").strip():
                errors.append(f"{name}: {field} vazio")
        _enum(g, "question_class", QUESTION_CLASSES, errors)
        _enum(g, "split", SPLITS, errors)
        _enum(g, "set", SETS, errors)
        _enum(g, "status", STATUSES, errors)
        for field in ("has_acronym", "names_article"):
            if not isinstance(g.get(field), bool):
                errors.append(f"{name}: {field} tem de ser true ou false")
        context = g.get("context")
        if not isinstance(context, list) or not all(isinstance(item, str) for item in context):
            errors.append(f"{name}: context tem de ser uma lista de textos")
            context = []
        errors += [f"{name}: nota editorial no context (vai para observacao): {item[:60]!r}"
                   for item in context if _EDITORIAL_NOTE.search(item)]
        source = g.get("source_document")
        if source is None:
            if g.get("question_class") not in OUT_CLASSES + ("ambiguo",):
                errors.append(f"{name}: sem source_document, a classe tem de ser de fora ou ambiguo")
            if g.get("content_type") is not None:
                errors.append(f"{name}: pergunta sem documento não tem content_type")
            if is_out(g):
                _enum(g, "out_stratum", OUT_STRATA, errors)
            elif g.get("out_stratum") is not None:
                errors.append(f"{name}: out_stratum só nas perguntas de fora")
            if context:
                errors.append(f"{name}: pergunta sem documento não tem evidência no context")
        else:
            if c1.doc_stem(source) != source:
                errors.append(f"{name}: source_document sem extensão ({source!r})")
            if is_out(g):
                errors.append(f"{name}: pergunta de fora não tem source_document")
            _enum(g, "content_type", CONTENT_TYPES, errors)
            if g.get("out_stratum") is not None:
                errors.append(f"{name}: out_stratum só nas perguntas de fora")
            if not context:
                errors.append(f"{name}: pergunta de dentro precisa de evidência no context")
            if index_documents is not None and source not in index_documents:
                errors.append(f"{name}: {source!r} não está na versão do índice")
            page = g.get("evidence_page")
            if page is not None and (not isinstance(page, int) or page < 1):
                errors.append(f"{name}: evidence_page tem de ser inteiro >= 1")
            split_of_document[source].add(g.get("split"))
    errors += [f"{source}: perguntas do mesmo documento em splits diferentes {sorted(map(str, splits))}"
               for source, splits in split_of_document.items() if len(splits) > 1]
    return errors


def check_evidence(goldens: list[dict], source_text: Callable[[str], dict[int | None, str] | None]) -> list[str]:
    """Cada fragmento casa com o documento inteiro? `source_text(stem)` dá {página: texto} (Markdown: {None: texto}).

    Erro: fragmento abaixo de c1.TAU, que não contaria como evidência nem com o documento inteiro na mão.
    Aviso: abaixo de LITERAL_MIN (alguma palavra não é a da fonte). Aviso também quando evidence_page não tem
    o primeiro fragmento.
    """
    problems = []
    for g in goldens:
        source = g.get("source_document")
        if not source:
            continue
        pages = source_text(source)
        if pages is None:
            problems.append(f"ERRO {g['name']}: não achei o documento {source!r}")
            continue
        fragments = c1.split_fragments(g.get("context"))
        found = c1.evidence(fragments, ["\n".join(pages.values())])
        for fragment, fraction in zip(fragments, found["union_fractions"]):
            if fraction < c1.TAU:
                problems.append(f"ERRO {g['name']}: fragmento com {fraction:.0%} na fonte: {fragment[:70]!r}")
            elif fraction < LITERAL_MIN:
                problems.append(f"AVISO {g['name']}: fragmento com {fraction:.0%} na fonte: {fragment[:70]!r}")
        page = g.get("evidence_page")
        if page is not None and fragments:
            on_page = c1.evidence(fragments[:1], [pages.get(page, "")])["union_fractions"][0]
            if on_page < c1.TAU:
                problems.append(f"AVISO {g['name']}: o primeiro fragmento não está na pág. {page} ({on_page:.0%})")
    return problems


def _stratum(questions: list[dict]) -> str:
    """O content_type do documento: o mais comum entre as perguntas dele, com a precedência no empate."""
    counts = Counter(q.get("content_type") or q.get("out_stratum") or "?" for q in questions)
    order = {kind: i for i, kind in enumerate(CONTENT_TYPES + OUT_STRATA)}
    return min(counts, key=lambda kind: (-counts[kind], order.get(kind, 99)))


def assign_splits(goldens: list[dict], seed: int, reassign: bool = False) -> int:
    """Sorteia `split` por documento, estratificado pelo content_type do documento (§7).

    Perguntas de um mesmo documento ficam juntas; pergunta sem documento é sorteada sozinha, estratificada
    pelo out_stratum. Sem `reassign`, só sorteia quem ainda não tem split: o split se fixa antes de qualquer
    rodada e não muda depois. Devolve quantas perguntas receberam split.
    """
    pending = [g for g in goldens if reassign or g.get("split") not in SPLITS]
    units: dict[str, list[dict]] = defaultdict(list)
    for g in pending:
        units[g["source_document"] or f"__{g['name']}"].append(g)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key, questions in sorted(units.items()):
        by_stratum[_stratum(questions)].append(key)
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        keys = by_stratum[stratum]
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            for g in units[key]:
                g["split"] = SPLITS[i % 2]
    return len(pending)


def _other_names() -> set[str]:
    names: set[str] = set()
    for path in OTHER_SETS:
        if path.is_file():
            names |= {g.get("name") for g in json.loads(path.read_text(encoding="utf-8"))}
    return names


def _version_reader(index: str) -> tuple[set[str], Callable[[str], dict | None]]:
    """Os documentos da versão (sem extensão) e um leitor do texto de cada um, como a ingestão lê."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.config import RAG_INDICES, PROJECT_ROOT
    from app.rag import ingestion

    folder = PROJECT_ROOT / RAG_INDICES[index]["source_dir"]
    registered = ingestion.registered_documents(index) or {}
    by_stem = {c1.doc_stem(name): folder / name for name in registered}

    def read(stem: str) -> dict | None:
        path = by_stem.get(stem)
        return dict(ingestion.read_document(path).segments) if path and path.is_file() else None

    return set(by_stem), read


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Goldens v2: validar o arquivo ou sortear o split.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validar")
    check.add_argument("--arquivo", type=Path, default=V2_PATH)
    check.add_argument("--indice", default=None, help="versão cujos documentos os source_document têm de ser")
    check.add_argument("--evidencia", action="store_true", help="confere cada fragmento contra a fonte (com --indice)")
    split = sub.add_parser("sortear-split")
    split.add_argument("--arquivo", type=Path, default=V2_PATH)
    split.add_argument("--semente", type=int, required=True)
    split.add_argument("--refazer", action="store_true", help="sorteia de novo também quem já tem split")
    args = parser.parse_args(argv)

    goldens = load(args.arquivo)
    if args.command == "sortear-split":
        if args.refazer and any(g.get("status") == "revisado" for g in goldens):
            print("[ERRO] Há goldens revisados: o split deles já pode ter sido usado.", file=sys.stderr)
            return 1
        assigned = assign_splits(goldens, args.semente, reassign=args.refazer)
        args.arquivo.write_text(json.dumps(goldens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = Counter(g["split"] for g in goldens)
        print(f"{assigned} pergunta(s) sorteadas (semente {args.semente}): {dict(counts)}")
        return 0

    documents, reader = _version_reader(args.indice) if args.indice else (None, None)
    errors = validate(goldens, _other_names(), documents)
    problems = check_evidence(goldens, reader) if args.evidencia and reader else []
    for line in errors:
        print(f"ERRO {line}")
    for line in problems:
        print(line)
    inside = [g for g in goldens if g.get("source_document")]
    print(f"{len(goldens)} goldens: {len(inside)} de dentro ({len({g['source_document'] for g in inside})} documentos), "
          f"{sum(map(is_out, goldens))} de fora, {Counter(g.get('status') for g in goldens)}")
    return 1 if errors or any(p.startswith("ERRO") for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
