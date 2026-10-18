"""Deixa os dois acervos do RAG com os mesmos artigos, cada um no seu formato.

    data/source_pdfs        PDFs      (acervo do índice pdf-v1)
    data/new_papers_marks   Markdown  (acervo do índice md-v1)

Cada PDF sem Markdown correspondente vira um .md em data/new_papers_marks, e
cada .md sem PDF correspondente vira um PDF em data/source_pdfs.
Correspondente = mesmo nome de arquivo sem a extensão, ignorando maiúsculas,
espaços e pontuação. O arquivo gerado leva o nome da fonte: RAGAS.pdf ->
RAGAS.md.

Uso, da raiz do repositório (venv próprio, ver data/requirements.txt):

    data\\.venv\\Scripts\\python data\\sync_corpora.py                  # converte o que falta, nas duas direções
    data\\.venv\\Scripts\\python data\\sync_corpora.py --dry-run        # só mostra o que faria
    data\\.venv\\Scripts\\python data\\sync_corpora.py --only md2pdf --limit 2
    data\\.venv\\Scripts\\python data\\sync_corpora.py --update         # refaz o que veio de uma fonte que mudou
    data\\.venv\\Scripts\\python data\\sync_corpora.py --undo           # apaga o que este script gerou

Nada é sobrescrito. O --undo só apaga o que este script gerou e que continua
igual ao que ele gravou (SHA-256 no manifesto). Um original fica, e também um
gerado que alguém editou ou trocou pelo PDF de verdade do artigo.

O manifesto data/conversions.json registra cada arquivo gerado: a fonte, o
SHA-256 dos dois, a ferramenta e a cobertura de texto (fração do texto da
fonte que reaparece no convertido, ignorando espaços, quebras de linha e
marcação). A ingestão (backend/app/rag/ingestion.py) indexa todo .pdf e .md
das pastas dos acervos: por isso o manifesto fica fora delas, e o temporário
de cada gravação, dentro, termina em .tmp.

O que cada direção faz com o conteúdo:

- PDF -> Markdown: pymupdf4llm com o pymupdf-layout (ordem de leitura em
  colunas, títulos, tabelas), sem OCR e sem imagens. Palavras hifenizadas no
  fim da linha saem inteiras.
- Markdown -> PDF: o Markdown vira HTML e o fpdf2 diagrama em A4, com a Noto
  Sans e, para o que ela não tem, Noto Sans Math, Noto Sans Symbols e a fonte
  CJK do PyMuPDF. Cada linha de tabela vira uma linha de texto, com as células
  unidas por espaço (como a ingestão do md-v1 faz): as tabelas destes .md são,
  na maior parte, linhas do artigo que a conversão anterior partiu em células,
  e uma tabela de 20 colunas não cabe na página. Palavra mais larga que a
  linha é quebrada no meio. Caractere sem glifo em nenhuma das fontes (emoji)
  sai do PDF, e o manifesto conta quantos.

  O PyMuPDF (Story) diagramaria sem dependência nova, mas foi descartado nesta
  direção: medido em 17/10/2026, perdia o resto de um parágrafo partido entre
  páginas (14% do texto de um artigo sumiu) e gravava o código errado das
  letras matemáticas fora do plano básico do Unicode (𝑃 saía como ᵄ).

Versões do índice (backend/app/config.py, RAG_INDICES): cada versão só ingere
os arquivos da lista dela (backend/app/rag/index_versions). O pdf-v1 e o md-v1
continuam com os originais. O pdf-v2 e o md-v2 têm os 61 artigos da
sincronização de 17/10/2026. Um arquivo gerado depois disso não entra em
nenhuma versão registrada: para usá-lo, registre uma versão nova
(scripts/ingest_documents.py --register).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
PDF_DIR = DATA_DIR / "source_pdfs"
MD_DIR = DATA_DIR / "new_papers_marks"
MANIFEST = DATA_DIR / "conversions.json"
SETUP_HINT = (
    "crie o venv com: python -m venv data\\.venv && data\\.venv\\Scripts\\pip install -r data\\requirements.txt"
)
ABOUT = (
    "Arquivos gerados por data/sync_corpora.py. Cada um é a conversão de um documento do outro acervo "
    "(source = de onde veio). sha256 é o do arquivo como foi gravado: se não bate mais, alguém mexeu nele."
)

# Abaixo disto a conversão perdeu texto demais: o arquivo é gravado assim mesmo, com aviso no console e no
# manifesto, para alguém conferir.
LOW_COVERAGE = 0.97

# PDF gerado: A4, margens de ~2 cm, corpo 10 pt.
MARGIN_PT = 56
BODY_PT, TABLE_PT = 10, 9
LINE_HEIGHT = 1.25

# Para a cobertura: some o que muda entre formatos sem mudar o conteúdo (espaço, quebra de linha, hífen de
# fim de linha, hífen invisível, espaço de largura zero, marcação do Markdown, marcador de lista: o "•" de um
# PDF vira "- " no Markdown).
_INVISIBLE_AND_BULLETS = "".join(map(chr, (0x00AD, 0x200B, 0x2022, 0x2023, 0x25AA, 0x25CF, 0x25E6)))
_NOT_TEXT = re.compile(r"[\s*_#|>`\-" + _INVISIBLE_AND_BULLETS + "]+")
# Tags que o pymupdf4llm põe no Markdown (quebra de linha em célula, expoente). Só estas: um "<Operation>" do
# texto do artigo é conteúdo.
_MD_INLINE_TAG = re.compile(r"</?(?:br|sup|sub|b|i|em|strong|span|u|s|del|code)\b[^>]*>", re.IGNORECASE)
# Célula de separador de tabela ("---", ":---:") repetido no meio da tabela, que a conversão anterior deixou.
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


@dataclass
class Job:
    source: Path
    target: Path
    direction: str  # "pdf2md" ou "md2pdf"
    update: bool = False  # refazer um gerado cuja fonte mudou (sobrescreve só o que o script gravou)


# ----------------------------------------------------------------- utilidades


def doc_key(path: Path) -> str:
    """Nome sem extensão, sem maiúsculas, espaços e pontuação: 'AIWare_2025___X' casa com 'AIWare 2025 X'."""
    stem = unicodedata.normalize("NFKC", path.stem).casefold()
    return "".join(ch for ch in stem if ch.isalnum())


def scan(folder: Path, suffix: str) -> list[Path]:
    """Os documentos que a ingestão veria na pasta (mesmo critério: a extensão, sem distinguir maiúsculas)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def html_text(markup: str) -> str:
    """Texto de um HTML gerado pelo markdown-it (todo '<' do texto vem escapado, então o que casa é tag)."""
    return html.unescape(re.sub(r"<[^>]+>", " ", markup))


def text_coverage(reference: str, candidate: str, k: int = 8) -> float:
    """Fração do texto de `reference` que reaparece em `candidate` (os dois em texto puro).

    Compara trechos de k caracteres depois de tirar o que muda entre formatos sem mudar o conteúdo (ver
    _NOT_TEXT, e NFKC: a ligadura "ﬁ" de um PDF casa com "fi"). Um pedaço que sumiu derruba a cobertura;
    uma quebra de linha em outro lugar, não.
    """
    ref, cand = _squash(reference), _squash(candidate)
    windows = [ref[i:i + k] for i in range(0, len(ref) - k + 1, k)]
    if not windows:
        return 1.0
    seen = {cand[i:i + k] for i in range(len(cand) - k + 1)}
    return sum(w in seen for w in windows) / len(windows)


def _squash(text: str) -> str:
    return _NOT_TEXT.sub("", unicodedata.normalize("NFKC", text))


def write_atomic(target: Path, data: bytes) -> None:
    """Grava por um temporário na mesma pasta. O temporário termina em .tmp, que a ingestão não lê."""
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def percent(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


# ----------------------------------------------------------------- manifesto


class Manifest:
    """data/conversions.json: o que este script gerou, com caminhos relativos à pasta do manifesto."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.base = path.parent
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self.files: dict[str, dict] = data.get("files", {})

    def key(self, path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), self.base)).as_posix()

    def resolve(self, key: str) -> Path:
        return (self.base / key).resolve()

    def outputs(self) -> set[Path]:
        return {self.resolve(k) for k in self.files}

    def save(self) -> None:
        data = {"about": ABOUT, "files": dict(sorted(self.files.items()))}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------- plano


def plan(pdf_dir: Path, md_dir: Path, generated: set[Path]) -> tuple[list[Job], list[str]]:
    """O que falta em cada acervo. Arquivo gerado por este script nunca é fonte: sem isto, apagar um original
    faria o gerado voltar como PDF/Markdown novo na rodada seguinte."""
    pdfs, mds = scan(pdf_dir, ".pdf"), scan(md_dir, ".md")
    jobs: list[Job] = []
    problems: list[str] = []
    for sources, targets, target_dir, suffix, direction in (
        (pdfs, mds, md_dir, ".md", "pdf2md"),
        (mds, pdfs, pdf_dir, ".pdf", "md2pdf"),
    ):
        present = {doc_key(p) for p in targets}
        seen: dict[str, Path] = {}
        for source in sources:
            if source.resolve() in generated:
                continue
            key = doc_key(source)
            if key in seen:
                problems.append(f"{source.name} e {seen[key].name} têm o mesmo nome normalizado: pulei o segundo")
                continue
            seen[key] = source
            if key not in present:
                jobs.append(Job(source, target_dir / (source.stem + suffix), direction))
    return jobs, problems


def review_generated(manifest: Manifest) -> tuple[list[Job], list[str]]:
    """Confere os arquivos gerados antes: fonte que mudou (o --update refaz), fonte que sumiu, saída editada.

    Saída editada ou trocada (o PDF de verdade no lugar do gerado, por exemplo) deixa de ser deste script:
    sai do manifesto e passa a valer como original.
    """
    stale: list[Job] = []
    notes: list[str] = []
    for key, entry in list(manifest.files.items()):
        output, source = manifest.resolve(key), manifest.resolve(entry["source"])
        if not output.is_file():
            # Apagado à mão: a fonte, se existir, volta para o plano como qualquer documento sem par.
            del manifest.files[key]
            notes.append(f"{key}: apagado à mão (sai do manifesto)")
        elif sha256(output) != entry["sha256"]:
            del manifest.files[key]
            notes.append(f"{key}: mudou depois da conversão; agora conta como original (sai do manifesto)")
        elif not source.is_file():
            notes.append(f"{key}: a fonte {entry['source']} não existe mais (--undo apaga o gerado)")
        elif sha256(source) != entry["source_sha256"]:
            direction = "pdf2md" if output.suffix.lower() == ".md" else "md2pdf"
            stale.append(Job(source, output, direction, update=True))
    return stale, notes


# ------------------------------------------------------------ PDF -> Markdown


def pdf_to_markdown(source: Path) -> tuple[bytes, dict]:
    import pymupdf
    import pymupdf4llm

    with pymupdf.open(source) as doc:
        pages = doc.page_count
        original = "\n".join(page.get_text() for page in doc)
    # Sem OCR: os PDFs do acervo têm texto, e o OCR mudaria o resultado conforme a máquina.
    markdown = pymupdf4llm.to_markdown(str(source), use_ocr=False, show_progress=False)
    coverage = text_coverage(original, html.unescape(_MD_INLINE_TAG.sub(" ", markdown)))
    info: dict = {"pages": pages, "text_coverage": round(coverage, 4)}
    if pages and len(original.strip()) < 200 * pages:
        info["notes"] = ["pouco texto extraível no PDF (escaneado?): sem OCR, o Markdown sai quase vazio"]
    return (markdown.rstrip() + "\n").encode("utf-8"), info


def pdf_to_markdown_tool() -> str:
    import pymupdf4llm

    return f"pymupdf4llm {pymupdf4llm.__version__} (pymupdf-layout), use_ocr=False, sem imagens"


# ------------------------------------------------------------ Markdown -> PDF


def drop_layout_rows(tokens: list) -> list:
    """Tira as linhas de tabela vazias ou só de traços (separador repetido no meio da tabela).

    É o mesmo que a ingestão do md-v1 faz (normalize_markdown): são sintaxe, não conteúdo, e no PDF virariam
    linhas de "--- --- ---".
    """
    out, i = [], 0
    while i < len(tokens):
        if tokens[i].type == "tr_open":
            end = next(j for j in range(i, len(tokens)) if tokens[j].type == "tr_close")
            cells = [t.content.strip() for t in tokens[i:end] if t.type == "inline"]
            if all(not c or _SEPARATOR_CELL.match(c) for c in cells):
                i = end + 1
                continue
        out.append(tokens[i])
        i += 1
    return out


class Typesetter:
    """Markdown -> HTML (markdown-it) -> PDF (fpdf2). Prepara fontes e conversor uma vez por execução."""

    # Família no fpdf2 -> {estilo: código da fonte no pymupdf-fonts}.
    FONTS = {
        "NotoSans": {"": "notos", "B": "notosbo", "I": "notosit", "BI": "notosbi"},
        "Mono": {"": "cascadia", "B": "cascadiab", "I": "cascadiai", "BI": "cascadiabi"},
        "NotoMath": {"": "math"},
        "NotoSymbols": {"": "symbol1"},
        "NotoSymbols2": {"": "symbol2"},
    }
    # Para o caractere que a fonte do trecho não tem, nesta ordem (CJK = a fonte embutida no PyMuPDF).
    FALLBACKS = ["NotoSans", "NotoMath", "NotoSymbols", "NotoSymbols2", "CJK"]

    def __init__(self) -> None:
        import pymupdf
        import pymupdf_fonts
        from fontTools.ttLib import TTFont
        from markdown_it import MarkdownIt

        # Sem isto, o fontTools avisa a cada fonte que não sabe enxugar uma tabela opcional (TTFA), e o fpdf2
        # lista os caracteres sem glifo, que o manifesto já conta.
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.getLogger("fpdf").setLevel(logging.ERROR)

        # O fpdf2 só lê fonte de arquivo: as do pymupdf-fonts e a CJK do PyMuPDF vão para uma pasta temporária.
        self._fonts_dir = tempfile.TemporaryDirectory(prefix="sync_corpora_", ignore_cleanup_errors=True)
        folder = Path(self._fonts_dir.name)
        self.font_files: dict[tuple[str, str], Path] = {}
        for family, styles in self.FONTS.items():
            for style, code in styles.items():
                path = folder / f"{code}.ttf"
                path.write_bytes(pymupdf_fonts.fontdescriptors[code]["loader"]())
                self.font_files[family, style] = path
        cjk = folder / "cjk.ttf"
        cjk.write_bytes(pymupdf.Font("cjk").buffer)
        self.font_files["CJK", ""] = cjk
        # Caracteres que alguma fonte desenha (o corpo e as de reserva). O resto sai do PDF.
        self.drawable: set[int] = set()
        for family in self.FALLBACKS:
            self.drawable |= set(TTFont(self.font_files[family, ""]).getBestCmap())

        # html=False: o que parece tag no texto dos artigos (<Operation>, <WidgetName>, placeholders de
        # prompt) é conteúdo e sai como texto; tratado como HTML, sumiria.
        self.md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
        # Tabela -> um parágrafo com uma linha por linha da tabela, células unidas por espaço.
        flat = {
            "table_open": f'<p line-height="{LINE_HEIGHT}"><font size="{TABLE_PT}">', "table_close": "</font></p>\n",
            "thead_open": "", "thead_close": "", "tbody_open": "", "tbody_close": "",
            "tr_open": "", "tr_close": "<br>\n",
            "th_open": "<b>", "th_close": "</b> ", "td_open": "", "td_close": " ",
        }
        for rule, markup in flat.items():
            self.md.add_render_rule(rule, lambda _self, _tokens, _idx, _options, _env, markup=markup: markup)
        # Imagem: o arquivo não vem junto com o .md; fica o texto alternativo, que é conteúdo.
        self.md.add_render_rule(
            "image",
            lambda _self, tokens, idx, options, env: html.escape(
                _self.renderInlineAsText(tokens[idx].children or [], options, env), quote=False
            ),
        )

    def convert(self, source: Path) -> tuple[bytes, dict]:
        import pymupdf

        text = source.read_text(encoding="utf-8", errors="replace")
        body = self.md.renderer.render(drop_layout_rows(self.md.parse(text)), self.md.options, {})
        # Entrelinha um pouco maior que a padrão do fpdf2 (1,0) nos parágrafos e itens de lista.
        for tag in ("p", "li"):
            body = body.replace(f"<{tag}>", f'<{tag} line-height="{LINE_HEIGHT}">')
        data, pages = self.render(body, source)
        plain = html_text(body)
        with pymupdf.open("pdf", data) as doc:
            extracted = "\n".join(page.get_text() for page in doc)
        info: dict = {"pages": pages, "text_coverage": round(text_coverage(plain, extracted), 4)}
        # Não contam como caractere perdido: marcas combinantes e seletores de variação (sem desenho próprio),
        # controles, uso privado (lixo da conversão anterior) e espaços — categorias Unicode M, C e Z.
        missing = {ch for ch in plain if ord(ch) not in self.drawable and unicodedata.category(ch)[0] not in "MCZ"}
        if missing:
            count = sum(1 for ch in plain if ch in missing)
            sample = " ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else "")
            info["notes"] = [f"{count} caractere(s) sem glifo em nenhuma fonte, fora do PDF: {sample}"]
        return data, info

    def tool(self) -> str:
        import fpdf
        import markdown_it

        return (
            f"markdown-it-py {markdown_it.__version__} + fpdf2 {fpdf.__version__}, A4, Noto Sans "
            "(reservas: Noto Sans Math, Noto Sans Symbols, CJK do PyMuPDF); tabela em linhas de texto"
        )

    def render(self, body: str, source: Path) -> tuple[bytes, int]:
        from fpdf import FPDF
        from fpdf.fonts import FontFace, TextStyle

        pdf = FPDF(unit="pt", format="A4")
        pdf.set_margins(MARGIN_PT, MARGIN_PT, MARGIN_PT)
        pdf.set_auto_page_break(True, margin=MARGIN_PT)
        for (family, style), path in self.font_files.items():
            pdf.add_font(family, style=style, fname=path)
        pdf.set_fallback_fonts(self.FALLBACKS, exact_match=False)
        pdf.set_title(source.stem)
        pdf.set_subject(f"Gerado de {source.name} por data/sync_corpora.py")
        pdf.set_creator("data/sync_corpora.py")
        pdf.add_page()
        pdf.set_font("NotoSans", size=BODY_PT)
        black = "#000000"
        headings = {"h1": 16, "h2": 13, "h3": 11.5, "h4": 10.5, "h5": 10, "h6": 10}
        styles = {
            tag: TextStyle(font_size_pt=size, color=black, t_margin=8, b_margin=3) for tag, size in headings.items()
        }
        styles["blockquote"] = TextStyle(color=black, l_margin=12, t_margin=3, b_margin=3)
        styles["pre"] = TextStyle(font_family="Mono", font_size_pt=8.5, t_margin=4)
        styles["code"] = FontFace(family="Mono")
        styles["a"] = FontFace(color=black)
        pdf.write_html(body, font_family="NotoSans", li_prefix_color=black, tag_styles=styles)
        return bytes(pdf.output()), pdf.pages_count


# ----------------------------------------------------------------------- main


def undo(manifest: Manifest, dry_run: bool) -> int:
    removed, kept = [], []
    for key, entry in list(manifest.files.items()):
        output = manifest.resolve(key)
        if not output.is_file():
            del manifest.files[key]
            continue
        if sha256(output) != entry["sha256"]:
            # Editado ou trocado pelo original: não é mais o que o script gravou, então não apaga.
            kept.append(key)
            if not dry_run:
                del manifest.files[key]
            continue
        removed.append(key)
        if not dry_run:
            output.unlink()
            del manifest.files[key]
    verb = "apagaria" if dry_run else "apagou"
    print(f"--undo {verb} {len(removed)} arquivo(s) gerado(s):")
    for key in removed:
        print(f"   {key}")
    for key in kept:
        print(f"   [mantido] {key}: mudou depois da conversão (não é mais o que o script gravou)")
    if not dry_run:
        manifest.save()
    return 0


def select_jobs(args: argparse.Namespace, manifest: Manifest) -> tuple[dict[str, list[Job]], list[str]]:
    """As conversões desta execução, por direção, e os avisos sobre o que já foi gerado antes."""
    stale, notes = review_generated(manifest)
    jobs, problems = plan(args.pdf_dir, args.md_dir, manifest.outputs())
    if args.update:
        jobs += stale
    elif stale:
        notes += [f"{manifest.key(j.target)}: a fonte mudou depois da conversão (--update refaz)" for j in stale]
    if args.only:
        jobs = [j for j in jobs if j.direction == args.only]
    by_direction = {d: [j for j in jobs if j.direction == d][: args.limit] for d in ("pdf2md", "md2pdf")}
    return by_direction, problems + notes


@dataclass
class Tally:
    done: int = 0
    failures: list[str] = field(default_factory=list)
    low: list[str] = field(default_factory=list)


def load_converters(by_direction: dict[str, list[Job]]) -> dict[str, tuple]:
    """(função de conversão, descrição da ferramenta) de cada direção com trabalho. Só carrega o que vai usar:
    o pymupdf4llm traz o onnxruntime junto."""
    converters: dict[str, tuple] = {}
    if by_direction["pdf2md"]:
        converters["pdf2md"] = (pdf_to_markdown, pdf_to_markdown_tool())
    if by_direction["md2pdf"]:
        typesetter = Typesetter()
        converters["md2pdf"] = (typesetter.convert, typesetter.tool())
    return converters


def run_job(job: Job, convert, tool: str, manifest: Manifest, label: str, tally: Tally) -> None:
    """Converte um arquivo, grava e registra no manifesto. Falha num arquivo não interrompe os outros."""
    started = time.perf_counter()
    try:
        if job.target.exists() and not job.update:
            raise FileExistsError(f"{job.target.name} apareceu durante a execução; não sobrescrevo")
        data, info = convert(job.source)
        write_atomic(job.target, data)
    except Exception as exc:  # noqa: BLE001 - um arquivo com problema não derruba os outros
        tally.failures.append(job.source.name)
        size = len(str(job.target.resolve()))
        hint = f" — o caminho tem {size} caracteres, além do limite do Windows" if size >= 260 else ""
        print(f"{label}: FALHOU ({type(exc).__name__}: {exc}){hint}", flush=True)
        return
    seconds = time.perf_counter() - started
    manifest.files[manifest.key(job.target)] = {
        "source": manifest.key(job.source),
        "source_sha256": sha256(job.source),
        "sha256": sha256(job.target),
        "tool": tool,
        "converted_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        **info,
    }
    manifest.save()  # a cada arquivo: uma execução interrompida não perde o registro do que gravou
    tally.done += 1
    coverage = info["text_coverage"]
    warn = ""
    if coverage < LOW_COVERAGE:
        tally.low.append(job.target.name)
        warn = "  [AVISO] cobertura baixa"
    print(f"{label} -> {job.target.name}  ({info['pages']} pág., {seconds:.1f} s, cobertura {percent(coverage)})"
          f"{warn}", flush=True)
    for note in info.get("notes", []):
        print(f"         {note}", flush=True)


def shown(path: Path) -> str:
    """Caminho relativo à pasta atual quando dá (o normal: rodar da raiz do repositório)."""
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Converte o que falta para os dois acervos terem os mesmos artigos (PDF <-> Markdown)."
    )
    parser.add_argument("--dry-run", action="store_true", help="só mostra o que faria, sem gravar nada")
    parser.add_argument("--only", choices=("pdf2md", "md2pdf"), help="uma direção só")
    parser.add_argument("--limit", type=int, default=None, help="no máximo N conversões por direção (ensaio)")
    parser.add_argument("--update", action="store_true",
                        help="refaz os arquivos gerados cuja fonte mudou depois da conversão")
    parser.add_argument("--undo", action="store_true",
                        help="apaga os arquivos que este script gerou e que ninguém alterou")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help=f"padrão: {PDF_DIR}")
    parser.add_argument("--md-dir", type=Path, default=MD_DIR, help=f"padrão: {MD_DIR}")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help=f"padrão: {MANIFEST}")
    args = parser.parse_args()

    manifest = Manifest(args.manifest)
    if args.undo:
        return undo(manifest, args.dry_run)

    by_direction, warnings = select_jobs(args, manifest)
    for line in warnings:
        print(f"[AVISO] {line}")
    titles = {
        "pdf2md": f"PDF -> Markdown ({shown(args.pdf_dir)} -> {shown(args.md_dir)})",
        "md2pdf": f"Markdown -> PDF ({shown(args.md_dir)} -> {shown(args.pdf_dir)})",
    }
    if args.dry_run:
        for direction, todo in by_direction.items():
            print(f"\n{titles[direction]}: {len(todo)} a converter")
            for job in todo:
                print(f"   [{'refazer' if job.update else 'novo'}] {job.source.name} -> {job.target.name}")
        return 0
    if not any(by_direction.values()):
        print("Nada a converter." if args.only else "Nada a converter: os dois acervos já têm os mesmos documentos.")
        if args.manifest.is_file() or manifest.files:
            manifest.save()
        return 0

    try:
        converters = load_converters(by_direction)
    except ImportError as exc:
        print(f"[ERRO] {exc}: {SETUP_HINT}", file=sys.stderr)
        return 1
    tally = Tally()
    for direction, todo in by_direction.items():
        if todo:
            print(f"\n{titles[direction]}: {len(todo)} a converter", flush=True)
        convert, tool = converters.get(direction, (None, ""))
        for i, job in enumerate(todo, start=1):
            label = f"   [{i:>{len(str(len(todo)))}}/{len(todo)}] {job.source.name}"
            run_job(job, convert, tool, manifest, label, tally)

    print(f"\n{tally.done} arquivo(s) gerado(s), {len(tally.failures)} falha(s), {len(tally.low)} com cobertura "
          f"abaixo de {percent(LOW_COVERAGE)}. Manifesto: {shown(args.manifest)}")
    return 1 if tally.failures else 0


if __name__ == "__main__":
    sys.exit(main())
