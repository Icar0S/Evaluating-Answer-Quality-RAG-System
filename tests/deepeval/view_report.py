"""Reabre o painel HTML da última rodada, sem rodar a avaliação de novo.

Equivalente ao `allure serve` (que reconstrói e abre o relatório a partir de
resultados já existentes) — aqui não tem servidor, o HTML é autocontido e
abre direto do disco.

Rodar (venv isolado, da raiz do repo):
    tests\\deepeval\\.venv\\Scripts\\python.exe tests\\deepeval\\view_report.py
"""
from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

import report  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if not json_files:
        print(
            "Nenhum resultado em tests/deepeval/results/ ainda. Rode "
            "`python tests/deepeval/run_and_export.py` primeiro.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    latest_json = json_files[-1]
    data = json.loads(latest_json.read_text(encoding="utf-8"))

    # Regenera o HTML a partir do JSON em vez de só reabrir o latest.html
    # salvo antes — assim view_report.py continua funcionando mesmo se
    # report.py mudar depois do JSON ter sido gerado (ex. layout novo).
    html_doc = report.render_html(data, source_label=f"tests/deepeval/results/{latest_json.name}")
    latest_html = RESULTS_DIR / "latest.html"
    latest_html.write_text(html_doc, encoding="utf-8")

    print(f"Abrindo {latest_html} (dados de {latest_json.name})")
    webbrowser.open(latest_html.resolve().as_uri())


if __name__ == "__main__":
    main()
