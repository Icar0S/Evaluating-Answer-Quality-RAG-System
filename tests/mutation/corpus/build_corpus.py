"""Constrói o corpus do estudo: normaliza os PDFs de origem e gera as variantes.

Por que normalizar em vez de usar os PDFs originais direto:

1. **Comparabilidade.** Os operadores de corpus (C2, C4) precisam produzir
   documentos com conteúdo alterado. Editar texto dentro de um PDF diagramado é
   inviável; gerar um PDF novo a partir do texto é trivial. Se só os mutantes
   fossem re-renderizados, a diferença de extração entre original e mutante
   entraria no resultado junto com o defeito injetado. Normalizando **todo** o
   corpus, baseline e mutantes passam pelo mesmo renderizador e a única diferença
   é o defeito.

2. **Chunking observável.** O SUT quebra chunks por página. Com páginas de
   ~500 tokens (típico de artigo em duas colunas), um chunk de 1200 tokens
   engoliria a página inteira e os operadores K1–K4 seriam inertes por
   construção. Repaginar por orçamento de tokens (--tokens-per-page, padrão
   1800) devolve sentido aos quatro.

3. **Empacotamento.** O corpus vai junto no pacote de replicação (Zenodo), então
   precisa de licença que permita redistribuição — ver sources.yaml e o campo
   `license` do manifesto.

Uso:
    python -m tests.mutation.corpus.build_corpus --from data/source_pdfs --limit 8
    python -m tests.mutation.corpus.build_corpus --make-variants
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger

logger = get_logger("mutation.corpus")

DEFAULT_TOKENS_PER_PAGE = 1800
PAGE_WIDTH, PAGE_HEIGHT = 595.0, 842.0  # A4 em pontos
MARGIN = 40.0
FONT_SIZES = (8.0, 7.0, 6.5, 6.0)

# O conjunto Base-14 do PDF representa latin-1 e mais nada. Sem tradução, todo
# símbolo fora disso virava "?" — 1372 deles no primeiro corpus montado, sendo
# 677 num único documento. Não é perda cosmética: notação estatística é
# exatamente o que os casos factuais citam, e "p = 0,377" virando "? = 0,377"
# apaga o fato que o caso ia testar.
PUNCTUATION_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
    "−": "-", "•": "-", "ﬁ": "fi", "ﬂ": "fl",
}

# Letras gregas e operadores matemáticos que a NFKC não reduz a ASCII. A grafia
# por extenso ("alpha") é preferível ao símbolo perdido: um caso de teste pode
# citar "alpha = 0,05", nunca "? = 0,05".
SYMBOL_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "λ": "lambda", "μ": "mu",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega", "κ": "kappa", "ν": "nu",
    "Δ": "Delta", "Σ": "Sigma", "Ω": "Omega", "Φ": "Phi", "Θ": "Theta",
    "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=", "≡": "==",
    "×": "x", "÷": "/", "±": "+/-", "∞": "inf", "√": "sqrt",
    "→": "->", "←": "<-", "⇒": "=>", "↔": "<->",
    "∈": " in ", "∉": " not in ", "⊂": " subset ", "∪": " union ", "∩": " inter ",
    "∑": "sum", "∏": "prod", "∫": "int", "∂": "d", "∇": "grad",
    "·": ".", "∗": "*", "⋅": ".", "°": " graus", "‰": " por mil",
    "™": "(TM)", "®": "(R)", "©": "(c)", "§": "sec.", "€": "EUR",
    # Decoração de tabela de resultados: seta para cima/baixo marcando
    # melhor/pior, círculo marcando ausência. Aparecem centenas de vezes.
    "↑": "+", "↓": "-", "○": "o", "●": "*", "◦": "o", "†": "+", "‡": "++",
    "⟨": "<", "⟩": ">", "≪": "<<", "≫": ">>", "′": "'", "″": '"', "⋆": "*",
}


def _latin1_safe(text: str) -> str:
    """Deixa o texto representável pelas fontes Base-14 do PDF, sem perder conteúdo.

    Três passadas, nesta ordem:

    1. pontuação tipográfica para o equivalente ASCII;
    2. NFKC, que resolve sozinha as classes que têm decomposição de
       compatibilidade — itálico matemático (𝑝 -> p), ligaduras, expoentes,
       largura dupla. É o que recupera a notação estatística dos artigos;
    3. tabela explícita para grego e operadores, que a NFKC não reduz.

    Acentos do português sobrevivem (estão em latin-1). O que ainda escapar vira
    "?" — visível de propósito, para a perda não passar despercebida.
    """
    for source, target in PUNCTUATION_MAP.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKC", text)
    for source, target in SYMBOL_MAP.items():
        text = text.replace(source, target)
    return "".join(_fold_to_latin1(ch) for ch in text)


def _fold_to_latin1(char: str) -> str:
    """Ultimo recurso por caractere: decompoe e larga os acentos que sobrarem.

    Recupera nomes proprios de autores — Ismayle, Gencer, Sahin viram
    representaveis em vez de "?". So o que nao tem nenhuma forma latina
    (emoji, pedaco de parentese gigante do LaTeX) ainda vira "?", e ai o "?"
    e informacao: marca onde a extracao perdeu algo de verdade.
    """
    try:
        return char.encode("latin-1").decode("latin-1")
    except UnicodeEncodeError:
        pass
    decomposed = unicodedata.normalize("NFKD", char)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    try:
        return stripped.encode("latin-1").decode("latin-1")
    except UnicodeEncodeError:
        return "?"


def _reflow(text: str) -> str:
    """Desfaz a quebra de linha da diagramação original, preservando parágrafos.

    Texto extraído de PDF vem com uma quebra por linha impressa (~70 caracteres).
    Isso tem dois efeitos ruins: cada linha curta consome uma linha inteira na
    re-renderização (o que forçaria páginas com metade do orçamento de tokens) e
    palavras hifenizadas na quebra chegam partidas ao chunk. Junta-se o parágrafo
    e remove-se a hifenização de fim de linha; linha em branco continua sendo
    fronteira de parágrafo.
    """
    text = re.sub(r"-\n(?=[a-zàáâãéêíóôõúç])", "", text)
    paragraphs = re.split(r"\n\s*\n", text)
    joined = [" ".join(line.strip() for line in p.splitlines() if line.strip()) for p in paragraphs]
    return "\n\n".join(p for p in joined if p)


def _token_pages(text: str, tokens_per_page: int) -> list[str]:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return [
        encoding.decode(tokens[start : start + tokens_per_page])
        for start in range(0, len(tokens), tokens_per_page)
    ] or [""]


def _render_pdf(pages: list[str], destination: Path, title: str) -> int:
    """Escreve um PDF com uma página por elemento de `pages`. Devolve o total de páginas."""
    import fitz

    document = fitz.open()
    rect = fitz.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)

    def fitting_fontsize(content: str) -> float | None:
        """Menor corpo de fonte em que o texto cabe na página, ou None se não couber.

        Testa num documento descartável: uma página que não coube não pode ser
        apagada do documento real (o PyMuPDF rejeita remover a página recém-criada),
        então é mais simples nunca criá-la.
        """
        probe = fitz.open()
        probe_page = probe.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        try:
            for size in FONT_SIZES:
                if probe_page.insert_textbox(rect, content, fontsize=size, fontname="helv", align=0) >= 0:
                    return size
            return None
        finally:
            probe.close()

    queue = list(pages)
    written = 0
    while queue:
        content = _latin1_safe(queue.pop(0))
        size = fitting_fontsize(content)
        if size is None:
            # Não coube nem no menor corpo: parte ao meio e tenta de novo. Melhor
            # uma página a mais do que texto perdido em silêncio.
            middle = max(1, len(content) // 2)
            queue.insert(0, content[middle:])
            queue.insert(0, content[:middle])
            continue
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_textbox(rect, content, fontsize=size, fontname="helv", align=0)
        written += 1

    document.set_metadata({"title": title, "producer": "tests/mutation/corpus/build_corpus.py"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(destination))
    document.close()
    return written


def _extract_text(pdf_path: Path) -> str:
    import fitz

    with fitz.open(pdf_path) as document:
        raw = "\n\n".join(page.get_text().strip() for page in document if page.get_text().strip())
    return _reflow(raw)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(name: str) -> str:
    stem = unicodedata.normalize("NFKD", Path(name).stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return (stem[:48] or "documento")


# ----------------------------------------------------------------- variantes

# Sinais de lista de referencias. Detectar bibliografia pelo titulo "REFERENCES"
# nao funciona neste corpus: a renormalizacao junta paragrafos e nem todo artigo
# usa o estilo [N] Autor — o rag_driven nao usa, e por isso a primeira versao
# deste detector deu a cauda dele como 100% prosa quando metade e bibliografia.
# Densidade de marcas (ano, "et al", colchete numerado, DOI/arXiv/paginacao) por
# mil caracteres separa os dois casos com folga: neste corpus, prosa fica entre
# 0 e 6, bibliografia entre 11 e 16.
REFERENCE_MARKS = (
    re.compile(r"(19|20)\d{2}"),
    re.compile(r"et al", re.IGNORECASE),
    re.compile(r"\[\d{1,3}\]"),
    re.compile(r"arXiv|doi\.org|https?://|pp\.\s*\d+|In:\s", re.IGNORECASE),
)
BIBLIOGRAPHY_DENSITY = 9.0


def reference_density(text: str) -> float:
    """Marcas de referencia por mil caracteres."""
    per_thousand = max(1, len(text)) / 1000
    return sum(len(pattern.findall(text)) for pattern in REFERENCE_MARKS) / per_thousand


def looks_like_bibliography(text: str) -> bool:
    return reference_density(text) >= BIBLIOGRAPHY_DENSITY


NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d{1,4}(?:[.,]\d{1,3})?)(%?)(?![\w])")


@dataclass
class Substitution:
    page: int
    original: str
    replacement: str
    rule: str


@dataclass
class VariantResult:
    variant: str
    document: str
    substitutions: list[Substitution] = field(default_factory=list)


def _perturb(value_text: str, is_percent: bool, variant: str) -> str | None:
    """Regra determinística de perturbação. Devolve None quando não se aplica.

    `prev` simula a versão anterior do documento (valores menores, ano anterior);
    `conflict` simula uma cópia divergente (valores maiores). As duas direções são
    opostas de propósito: assim um caso que cite o valor distingue as três versões.
    """
    normalized = value_text.replace(",", ".")
    try:
        number = float(normalized)
    except ValueError:
        return None

    decimals = len(normalized.split(".")[1]) if "." in normalized else 0

    if not is_percent and decimals == 0 and 1900 <= number <= 2099:
        shifted = number - 2 if variant == "prev" else number + 3
        return str(int(shifted))

    if is_percent:
        shifted = number - 10 if variant == "prev" else number + 15
        shifted = min(max(shifted, 0.0), 100.0)
    elif number >= 10:
        shifted = number * (0.8 if variant == "prev" else 1.35)
    else:
        return None

    if decimals == 0:
        new_value = str(int(round(shifted)))
    else:
        new_value = f"{shifted:.{decimals}f}"
        if "," in value_text:
            new_value = new_value.replace(".", ",")
    return new_value


CITATION_SPAN = re.compile(r"\[[\d\s,;+-]+\]")

# Um numero vale a pena perturbar quando alguem poderia fazer uma pergunta sobre
# ele. Estas palavras, vizinhas do numero, sao o sinal mais barato disso.
FACTUAL_CUES = (
    "accuracy", "coverage", "precision", "recall", "score", "rate", "ratio",
    "average", "mean", "median", "total", "sample", "participants", "subjects",
    "classes", "projects", "cases", "tests", "mutants", "questions", "documents",
    "tokens", "seconds", "minutes", "hours", "times", "versions", "runs",
    "threshold", "percent", "improvement", "reduction", "increase", "decrease",
    "of the", "up to", "we used", "consisted", "resulting in",
)


def _citation_spans(text: str) -> list[tuple[int, int]]:
    """Intervalos ocupados por marcadores de citacao bibliografica."""
    return [(m.start(), m.end()) for m in CITATION_SPAN.finditer(text)]


def _factual_score(text: str, start: int, end: int) -> int:
    """Quao provavel e que este numero seja um fato citavel por um caso de teste.

    Numero dentro de [12, 34] e referencia bibliografica: mudar aquilo nao cria
    defeito observavel nenhum, so estraga a bibliografia. Este foi o erro que a
    primeira versao cometia -- as 40 substituicoes caiam quase todas na
    introducao, que e onde a densidade de citacoes e maior, e C2/C4 ficavam sem
    ancora possivel na suite.
    """
    window = text[max(0, start - 60) : end + 60].lower()
    score = 0
    if "%" in text[end : end + 2]:
        score += 3
    if any(cue in window for cue in FACTUAL_CUES):
        score += 2
    following = text[end : end + 25].lstrip()
    if following[:1].isalpha():   # "16 Java classes", "30 participants"
        score += 2
    if re.search(r"(table|figure|section|fig\.|eq\.)\s*$", text[:start].lower()):
        score -= 3
    return score


def make_variant(
    pages: list[str], variant: str, max_substitutions: int
) -> tuple[list[str], list[Substitution]]:
    """Perturba valores numericos factuais, distribuindo-os pelo documento.

    Duas regras que existem para C2 e C4 serem observaveis por um caso de teste:

    1. marcadores de citacao ([13], [15, 36]) nunca sao tocados;
    2. a cota de substituicoes e dividida entre as paginas, e dentro de cada
       pagina os candidatos mais "factuais" vem primeiro. Sem isso a cota inteira
       era consumida pela primeira pagina.
    """
    per_page = max(1, max_substitutions // max(1, len(pages)))
    substitutions: list[Substitution] = []
    new_pages: list[str] = []

    for page_number, page_text in enumerate(pages, start=1):
        spans = _citation_spans(page_text)
        candidates = []
        for match in NUMBER_PATTERN.finditer(page_text):
            begin, finish = match.start(1), match.end(1)
            if any(low <= begin < high for low, high in spans):
                continue
            replacement = _perturb(match.group(1), bool(match.group(2)), variant)
            if replacement is None:
                continue
            candidates.append((_factual_score(page_text, begin, finish), begin, finish, match, replacement))

        budget = per_page if len(substitutions) + per_page <= max_substitutions else max_substitutions - len(substitutions)
        # Melhores primeiro para escolher, mas aplicados na ordem do texto para
        # que os offsets nao se invalidem entre si.
        chosen = sorted(sorted(candidates, key=lambda c: (-c[0], c[1]))[: max(0, budget)], key=lambda c: c[1])

        rebuilt = []
        cursor = 0
        for score, begin, finish, match, replacement in chosen:
            percent = match.group(2)
            rebuilt.append(page_text[cursor:begin])
            rebuilt.append(replacement)
            cursor = finish
            substitutions.append(
                Substitution(
                    page=page_number,
                    original=match.group(1) + percent,
                    replacement=replacement + percent,
                    rule=f"{variant}:score{score}",
                )
            )
        rebuilt.append(page_text[cursor:])
        new_pages.append("".join(rebuilt))

    return new_pages, substitutions


# ------------------------------------------------------------------- comandos


def build(source_dir: Path, limit: int | None, tokens_per_page: int, roles_override: dict[str, Any]) -> dict:
    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Nenhum PDF em {source_dir}")
    if limit:
        # Os maiores primeiro: documentos densos dão mais casos por documento e
        # tornam os operadores de chunking observáveis.
        pdfs = sorted(pdfs, key=lambda p: p.stat().st_size, reverse=True)[:limit]

    if paths.CORPUS_BASE.exists():
        for stale in paths.CORPUS_BASE.glob("*.pdf"):
            stale.unlink()
    paths.CORPUS_BASE.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(pdfs, key=lambda p: p.name):
        text = _extract_text(pdf_path)
        if not text.strip():
            logger.warning("%s não tem camada de texto — ignorado", pdf_path.name)
            continue
        pages = _token_pages(text, tokens_per_page)
        name = f"{_slug(pdf_path.name)}.pdf"
        destination = paths.CORPUS_BASE / name
        page_count = _render_pdf(pages, destination, title=pdf_path.stem)
        documents.append(
            {
                "document": name,
                "source_file": pdf_path.name,
                "pages": page_count,
                "approx_tokens": sum(len(p) for p in pages) // 4,
                "sha256": _sha256(destination),
                "license": "PREENCHER",
                "source_url": "PREENCHER",
            }
        )
        logger.info("%s -> %s (%d páginas)", pdf_path.name, name, page_count)

    names = [d["document"] for d in documents]
    roles = {
        "primary": names[0],
        "secondary": names[1] if len(names) > 1 else names[0],
        "recent": names[-2:] if len(names) > 2 else names[-1:],
    }
    roles.update(roles_override or {})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "tokens_per_page": tokens_per_page,
        "documents": documents,
        "roles": roles,
        "license_note": (
            "Todo documento precisa de licença que permita redistribuição antes do "
            "pacote de replicação (Zenodo). Campos 'PREENCHER' bloqueiam o empacotamento."
        ),
    }
    paths.CORPUS_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _invalidate_work()
    logger.info("manifest.json escrito com %d documentos; papéis: %s", len(documents), roles)
    return manifest


def _invalidate_work() -> None:
    """Descarta work/ e a assinatura do índice depois de reconstruir o corpus.

    Sem isto, reconstruir o corpus não chegava ao índice: `ensure_index` compara
    a assinatura com o conteúdo de work/index_src, que continuava com os PDFs
    antigos, e concluía — corretamente, dado o que via — que nada havia mudado.
    O sintoma foi um rascunho citando "(?= 0.377)" horas depois de o corpus já
    conter "(p= 0.377)": o modelo leu o índice, não o corpus.
    """
    import shutil

    for directory in (paths.WORK_CORPUS, paths.WORK_INDEX_SRC):
        if directory.exists():
            shutil.rmtree(directory)
    if paths.WORK_STATE.exists():
        paths.WORK_STATE.unlink()
    logger.info("work/ invalidado — o próximo build de índice vai reindexar.")


def apply_roles(roles_override: dict[str, Any]) -> int:
    """Regrava apenas `roles` no manifesto, validando contra os documentos existentes."""
    if not roles_override:
        logger.error("corpus/sources.yaml nao define `roles`. Nada a repinar.")
        return 1

    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    known = {document["document"] for document in manifest["documents"]}

    missing: list[str] = []
    for role, value in roles_override.items():
        for name in (value if isinstance(value, list) else [value]):
            if name not in known:
                missing.append(f"{role} -> {name}")
    if missing:
        logger.error("Papeis apontam para documentos que nao estao no corpus: %s", missing)
        return 1

    manifest["roles"] = dict(manifest.get("roles", {}))
    manifest["roles"].update(roles_override)
    paths.CORPUS_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Papeis repinados: %s", manifest["roles"])
    logger.info("Se `primary` mudou, rode --make-variants: as variantes seguem esse papel.")
    return 0


def build_variants(max_substitutions: int, tokens_per_page: int) -> dict:
    manifest = json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))
    roles = manifest["roles"]
    paths.CORPUS_VARIANTS.mkdir(parents=True, exist_ok=True)

    # Limpa variantes de builds anteriores: se o papel `primary` mudou, as
    # antigas ficariam orfas no diretorio e viajariam no pacote de replicacao.
    for stale in paths.CORPUS_VARIANTS.glob("*.pdf"):
        stale.unlink()

    plan = [(roles["primary"], "prev"), (roles["primary"], "conflict")]
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "variants": []}

    for document, variant in plan:
        base_pdf = paths.CORPUS_BASE / document
        if not base_pdf.exists():
            raise SystemExit(f"{base_pdf} não existe — rode o build primeiro")
        import fitz

        with fitz.open(base_pdf) as doc:
            pages = [page.get_text() for page in doc]

        new_pages, substitutions = make_variant(pages, variant, max_substitutions)
        destination = paths.CORPUS_VARIANTS / f"{Path(document).stem}__{variant}.pdf"
        _render_pdf(new_pages, destination, title=f"{Path(document).stem} ({variant})")
        report["variants"].append(
            {
                "document": document,
                "variant": variant,
                "file": destination.name,
                "sha256": _sha256(destination),
                "substitutions": [s.__dict__ for s in substitutions],
            }
        )
        logger.info("variante %s de %s: %d substituições", variant, document, len(substitutions))
        if not substitutions:
            logger.warning(
                "Nenhum valor numérico perturbável em %s — C2 e C4 seriam mutantes "
                "equivalentes. Escolha outro documento como papel 'primary'.",
                document,
            )

    target = paths.CORPUS_VARIANTS / "variants.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constrói o corpus normalizado do estudo de mutação.")
    parser.add_argument("--from", dest="source", default="data/source_pdfs", help="diretório com os PDFs de origem")
    parser.add_argument("--limit", type=int, default=8, help="quantos documentos usar (protocolo: 5 a 10)")
    parser.add_argument("--tokens-per-page", type=int, default=DEFAULT_TOKENS_PER_PAGE)
    parser.add_argument("--make-variants", action="store_true", help="gera apenas as variantes prev/conflict")
    parser.add_argument(
        "--roles-only",
        action="store_true",
        help="so repina os papeis do manifesto a partir de sources.yaml, sem re-renderizar",
    )
    parser.add_argument("--max-substitutions", type=int, default=40)
    args = parser.parse_args(argv)

    paths.ensure_dirs()

    if args.make_variants:
        build_variants(args.max_substitutions, args.tokens_per_page)
        return 0

    roles_override: dict[str, Any] = {}
    if paths.CORPUS_SOURCES.exists():
        sources = yaml.safe_load(paths.CORPUS_SOURCES.read_text(encoding="utf-8")) or {}
        roles_override = sources.get("roles") or {}

    if args.roles_only:
        # Repinar papeis nao muda um byte do corpus, entao re-renderizar seria so
        # trocar o sha256 de todo documento e forcar uma reindexacao inutil.
        return apply_roles(roles_override)

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = paths.PROJECT_ROOT / source_dir
    build(source_dir, args.limit, args.tokens_per_page, roles_override)
    build_variants(args.max_substitutions, args.tokens_per_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
