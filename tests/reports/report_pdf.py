"""Renderiza os blocos de report_document.py em PDF (A4), com ReportLab.

Fonte: DejaVu Sans, a que vem dentro do matplotlib. As fontes embutidas do
ReportLab (Helvetica etc.) não têm glifos como ≥, ✓, ✗, † e caem em
quadrados pretos; usar a DejaVu dispensa instalar fonte no sistema e deixa PDF
e gráficos com a mesma tipografia.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph
from reportlab.platypus import PageBreak as RLPageBreak
from reportlab.platypus import SimpleDocTemplate, Spacer
from reportlab.platypus import Table as RLTable
from reportlab.platypus import TableStyle

from report_document import Block, Bullets, Callout, Figure, Heading, PageBreak, Para, Table

INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
GRID = colors.HexColor("#e1e0d9")
BASELINE = colors.HexColor("#c3c2b7")
HEADER_BG = colors.HexColor("#f0efec")
ZEBRA = colors.HexColor("#f9f9f7")
WARNING = colors.HexColor("#fab219")  # status "warning" da paleta; sempre com ícone + rótulo

PAGE_MARGIN = 2 * cm
TITLE = "Relatório de qualidade de resposta do RAG"


def _register_fonts() -> None:
    if "DejaVu" in pdfmetrics.getRegisteredFontNames():
        return
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(base / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(base / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", str(base / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique"
    )


def _styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=12.5, textColor=INK, spaceAfter=5)
    return {
        "h1": ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=17, leading=21, spaceAfter=8),
        "h2": ParagraphStyle(
            "h2", parent=body, fontName="DejaVu-Bold", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=6,
            keepWithNext=1,
        ),
        "h3": ParagraphStyle(
            "h3", parent=body, fontName="DejaVu-Bold", fontSize=10, leading=13, spaceBefore=9, spaceAfter=4,
            keepWithNext=1,
        ),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, spaceAfter=3),
        "caption": ParagraphStyle(
            "caption", parent=body, fontName="DejaVu-Oblique", fontSize=7.5, leading=9.5, textColor=INK_SECONDARY,
            spaceBefore=3, spaceAfter=8,
        ),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0),
        "cell_small": ParagraphStyle("cell_small", parent=body, fontSize=7, leading=8.8, spaceAfter=0),
        "callout_title": ParagraphStyle("callout_title", parent=body, fontName="DejaVu-Bold", spaceAfter=3),
    }


def _inline(text: str) -> str:
    """**negrito**, `código` e *itálico* -> marcação do ReportLab, com o resto escapado."""
    s = html.escape(str(text), quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r'<font face="DejaVuMono">\1</font>', s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def _aligned(style: ParagraphStyle, align: str) -> ParagraphStyle:
    alignment = {"l": TA_LEFT, "r": TA_RIGHT, "c": TA_CENTER}.get(align, TA_LEFT)
    return ParagraphStyle(f"{style.name}_{align}", parent=style, alignment=alignment)


def _table(block: Table, width: float, st: dict[str, ParagraphStyle]) -> list:
    base = st["cell_small"] if block.small else st["cell"]
    header = ParagraphStyle(f"{base.name}_h", parent=base, fontName="DejaVu-Bold", textColor=INK)
    align = block.align or ["l"] * len(block.headers)
    data = [[Paragraph(_inline(h), _aligned(header, a)) for h, a in zip(block.headers, align)]]
    data += [[Paragraph(_inline(c), _aligned(base, a)) for c, a in zip(row, align)] for row in block.rows]
    table = RLTable(data, colWidths=[w * width for w in block.widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, BASELINE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    out = [table]
    out.append(Paragraph(_inline(block.caption), st["caption"]) if block.caption else Spacer(1, 8))
    return out


def _bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_inline(i), style), leftIndent=12) for i in items],
        bulletType="bullet",
        start="•",
        bulletFontName="DejaVu",
        bulletFontSize=style.fontSize,
        leftIndent=12,
        spaceAfter=6,
    )


def _figure(block: Figure, width: float, st: dict[str, ParagraphStyle]) -> KeepTogether:
    with PILImage.open(block.path) as img:
        px_w, px_h = img.size
    w = width * block.width
    return KeepTogether([Image(str(block.path), width=w, height=w * px_h / px_w), Paragraph(_inline(block.caption), st["caption"])])


def _callout(block: Callout, width: float, st: dict[str, ParagraphStyle]) -> RLTable:
    content = [Paragraph(f"⚠ {_inline(block.title)}", st["callout_title"]), _bullets(block.items, st["small"])]
    box = RLTable([[content]], colWidths=[width], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ZEBRA),
                ("LINEBEFORE", (0, 0), (0, -1), 3, WARNING),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return box


class _Doc(SimpleDocTemplate):
    """Marca cada título como favorito do PDF (o painel de navegação do leitor)."""

    def afterFlowable(self, flowable) -> None:  # noqa: N802 - nome da API do ReportLab
        level = getattr(flowable, "_outline_level", None)
        if level is None:
            return
        key = f"h{id(flowable)}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=level > 0)


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(PAGE_MARGIN, 1.1 * cm, TITLE)
    canvas.drawRightString(A4[0] - PAGE_MARGIN, 1.1 * cm, f"página {doc.page}")
    canvas.restoreState()


def render_pdf(blocks: list[Block], path: Path) -> Path:
    _register_fonts()
    st = _styles()
    doc = _Doc(
        str(path),
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=1.8 * cm,
        bottomMargin=1.9 * cm,
        title=TITLE,
        author="tests/reports/build_report.py",
    )
    width = doc.width
    story: list = []
    # Títulos esperam o primeiro elemento do bloco seguinte e entram com ele num
    # KeepTogether: título no pé da página com o conteúdo na próxima é o defeito
    # que o keepWithNext do estilo, sozinho, não evitou diante de figuras.
    pending: list = []
    for block in blocks:
        if isinstance(block, Heading):
            p = Paragraph(_inline(block.text), st[f"h{min(block.level, 3)}"])
            p._outline_level = block.level - 1
            pending.append(p)
            continue
        if isinstance(block, Para):
            flowables = [Paragraph(_inline(block.text), st["small" if block.small else "body"])]
        elif isinstance(block, Bullets):
            flowables = [_bullets(block.items, st["small"])]
        elif isinstance(block, Table):
            flowables = _table(block, width, st)
        elif isinstance(block, Figure):
            flowables = [_figure(block, width, st)]
        elif isinstance(block, Callout):
            flowables = [_callout(block, width, st), Spacer(1, 8)]
        elif isinstance(block, PageBreak):
            story += pending + [RLPageBreak()]
            pending = []
            continue
        else:
            continue
        if pending:
            story.append(KeepTogether(pending + flowables[:1]))
            flowables, pending = flowables[1:], []
        story += flowables
    story += pending
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path
