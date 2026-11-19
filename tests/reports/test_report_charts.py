"""O gráfico do histórico (report_charts.history_chart) com mais séries do que cores livres.

Uma série é framework + índice + idioma. Em 12/11/2026, o histórico desta
máquina tinha 8 séries (uma rodada do RAGAS no pdf-v1-bgem3 entrou em 19/10), e
a sexta sem slot reservado derrubava o relatório inteiro com IndexError.
Herméticos. Rodar da raiz:
    tests\\reports\\.venv\\Scripts\\python.exe -m pytest tests/reports
"""
from __future__ import annotations

from pathlib import Path

import pytest

import report_charts
from report_data import DIMENSIONS

# As séries do histórico em que o relatório quebrou.
SERIES = ["DeepEval", "RAGAS", "DeepEval · pdf-v1", "JevEval · pdf-v1", "RAGAS · md-v1",
          "RAGAS · md-v1 · perguntas em inglês", "RAGAS · pdf-v1", "RAGAS · pdf-v1-bgem3"]


def test_cores_do_historico_usam_o_slot_reservado_que_nenhuma_serie_reclama():
    colors = report_charts.framework_colors(SERIES)

    assert len(set(colors.values())) == len(SERIES) == len(report_charts.CATEGORICAL)
    # O framework continua com a cor dele quando aparece com o nome puro.
    assert colors["DeepEval"] == report_charts.CATEGORICAL[report_charts.FRAMEWORK_SLOT["DeepEval"]]
    assert colors["RAGAS"] == report_charts.CATEGORICAL[report_charts.FRAMEWORK_SLOT["RAGAS"]]


def test_mais_series_do_que_cores_e_erro_claro_e_nao_indexerror():
    with pytest.raises(ValueError, match="no máximo 8 séries"):
        report_charts.framework_colors(SERIES + ["RAGAS · md-v2"])


def _runs(n_series: int) -> list[dict]:
    """Duas rodadas por série; a série i é mais recente que a i-1."""
    runs = []
    for i in range(n_series):
        for hour in (1, 2):
            runs.append({"name": f"RAGAS · indice-{i}", "framework": "ragas",
                         "run_generated_at": f"2026-10-{10 + i:02d}T0{hour}:00:00+00:00",
                         "means": {d.key: 0.5 for d in DIMENSIONS}})
    return runs


def test_historico_com_mais_series_do_que_cores_mostra_as_mais_recentes(tmp_path: Path, monkeypatch):
    drawn = []
    real = report_charts.framework_colors
    monkeypatch.setattr(report_charts, "framework_colors", lambda names: drawn.append(list(dict.fromkeys(names)))
                        or real(names))

    path = report_charts.history_chart(_runs(10), tmp_path / "historico.png")

    assert path is not None and path.is_file()
    assert drawn[0] == [f"RAGAS · indice-{i}" for i in range(2, 10)]  # as duas mais antigas saem do gráfico


def test_figura_mais_alta_que_a_pagina_encolhe_em_vez_de_derrubar_o_pdf(tmp_path: Path):
    # O heatmap de uma rodada com 103 perguntas (benchmark, 19/10/2026) tinha 1505 pt num quadro de 725.
    from PIL import Image as PILImage

    import report_pdf
    from report_document import Figure, Heading, Para

    tall = tmp_path / "alta.png"
    PILImage.new("RGB", (400, 1300), "white").save(tall)

    pdf = report_pdf.render_pdf([Heading(1, "Notas"), Para("Antes da figura."), Figure(tall, "Uma figura alta.")],
                                tmp_path / "relatorio.pdf")

    assert pdf.is_file() and pdf.stat().st_size > 0
