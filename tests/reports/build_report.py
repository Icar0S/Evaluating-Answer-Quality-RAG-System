"""Relatório consolidado das avaliações de qualidade do RAG (RAGAS, DeepEval...).

Lê os JSONs que as suítes gravam (tests/<framework>/results/*.json) e monta um
relatório só, separado por framework, com a comparação entre eles:

- relatorio.md   para ler no GitHub/VS Code (figuras em figuras/)
- relatorio.pdf  para anexar ou imprimir
- relatorio.xlsx os dados para análise (filtros, pivô, matrizes coloridas)
- notas.csv      formato longo, para pandas/R

Rodar da raiz do repo (venv próprio, ver README.md desta pasta):

    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\build_report.py
    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\build_report.py --formats pdf xlsx
    tests\\reports\\.venv\\Scripts\\python.exe tests\\reports\\build_report.py ^
        --input tests\\ragas\\results\\20261007T115107Z.json ^
        --input tests\\deepeval\\results\\20260827T163236Z.json

Sem --input, usa o resultado mais recente de cada framework.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_charts  # noqa: E402
import report_data  # noqa: E402
import report_history  # noqa: E402
from report_document import build_document  # noqa: E402
from report_markdown import render_markdown  # noqa: E402
from report_pdf import render_pdf  # noqa: E402
from report_sheets import write_csv, write_xlsx  # noqa: E402

FORMATS = ("md", "pdf", "xlsx", "csv")
DEFAULT_OUT = HERE / "out"


def build(
    paths: list[Path],
    formats: list[str],
    out_dir: Path,
    execution: dict | None = None,
    history_root: Path | None = None,
) -> dict[str, Path]:
    """Gera o relatório e devolve {formato: arquivo}. Separado do main() para os testes.

    `execution`: o execucao.json do tests/run_all.py (vira a seção "Execução dos
    testes"). `history_root`: a pasta que guarda o histórico (tests/reports/out);
    None = relatório avulso, fora do histórico.
    """
    runs = [report_data.load_run(p) for p in paths]
    # Ordem estável: DeepEval antes de RAGAS (slots 1 e 2 da paleta), depois por data.
    order = {"DeepEval": 0, "RAGAS": 1}
    runs.sort(key=lambda r: (order.get(r.framework, 9), r.generated_at or datetime.min.replace(tzinfo=timezone.utc)))
    comparisons = [report_data.compare(a, b) for a, b in itertools.combinations(runs, 2)]

    fig_dir = out_dir / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    if len(runs) > 1:
        figures["means"] = report_charts.means_chart(runs, fig_dir / "medias_por_dimensao.png")
    for i, run in enumerate(runs, start=1):
        slug = f"{i}_{run.framework.lower()}"
        figures[f"heatmap_{i}"] = report_charts.heatmap(run, fig_dir / f"notas_{slug}.png")

    generated_at = datetime.now()

    # Histórico = relatórios anteriores + este. As rodadas entram uma vez só
    # (report_history.unique_runs), então regerar um relatório não duplica ponto.
    history: list[dict] = []
    if history_root is not None:
        previous = report_history.load_entries(history_root)
        current = report_history.build_summary(runs, comparisons, execution, generated_at, files={})
        history = report_history.unique_runs(previous + [current])
        chart = report_charts.history_chart(history, fig_dir / "historico.png", threshold=runs[0].threshold)
        if chart:
            figures["history"] = chart

    outputs: dict[str, Path] = {}
    if "md" in formats or "pdf" in formats:
        blocks = build_document(runs, comparisons, figures, generated_at, execution=execution, history=history)
        if "md" in formats:
            outputs["md"] = render_markdown(blocks, out_dir / "relatorio.md")
        if "pdf" in formats:
            outputs["pdf"] = render_pdf(blocks, out_dir / "relatorio.pdf")
    if "xlsx" in formats:
        outputs["xlsx"] = write_xlsx(
            runs, comparisons, figures, generated_at, out_dir / "relatorio.xlsx", execution=execution, history=history
        )
    if "csv" in formats:
        outputs["csv"] = write_csv(runs, out_dir / "notas.csv")

    files = {name: path.name for name, path in outputs.items()}
    report_history.write_summary(
        out_dir, report_history.build_summary(runs, comparisons, execution, generated_at, files)
    )
    if history_root is not None:
        report_history.write_index(history_root, report_history.load_entries(history_root))
        if out_dir.resolve().parent == history_root.resolve():
            report_history.refresh_latest(history_root, out_dir)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Relatório consolidado das avaliações de qualidade do RAG.")
    parser.add_argument(
        "--input", action="append", type=Path, default=[],
        help="JSON de resultado de uma suíte (repetível). Sem isto: o mais recente de cada framework.",
    )
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS), help="padrão: todos")
    parser.add_argument("--out", type=Path, default=None, help=f"pasta de saída (padrão: {DEFAULT_OUT}/<data>)")
    parser.add_argument("--open", action="store_true", help="abre o PDF (ou o .md) ao terminar")
    parser.add_argument(
        "--execution", type=Path, default=None,
        help="execucao.json do tests/run_all.py (acrescenta a seção 'Execução dos testes')",
    )
    parser.add_argument(
        "--sem-historico", action="store_true",
        help="não entra no histórico (tests/reports/out/HISTORICO.md) nem atualiza out/ultimo",
    )
    args = parser.parse_args()

    paths = args.input or report_data.discover_latest()
    if not paths:
        print(
            "[ERRO] Nenhum resultado encontrado. Rode antes o run_and_export.py de alguma suíte "
            "(tests/ragas ou tests/deepeval), ou passe --input.",
            file=sys.stderr,
        )
        return 1
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        print(f"[ERRO] Arquivo(s) não encontrado(s): {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    out_dir = args.out or DEFAULT_OUT / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    execution = json.loads(args.execution.read_text(encoding="utf-8")) if args.execution else None
    # Só relatórios dentro de tests/reports/out entram no histórico: um --out
    # para outra pasta é relatório avulso.
    in_history = not args.sem_historico and out_dir.resolve().parent == DEFAULT_OUT.resolve()
    history_root = DEFAULT_OUT if in_history else None

    print("Rodadas:")
    for p in paths:
        print(f"  {p}")
    try:
        outputs = build([Path(p) for p in paths], args.formats, out_dir, execution=execution, history_root=history_root)
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    print("\nRelatório gerado:")
    for fmt_name, path in outputs.items():
        print(f"  {fmt_name:5s} {path}")
    if history_root is not None:
        print(f"\nHistórico: {history_root / 'HISTORICO.md'}")
        print(f"Último relatório (caminho fixo): {history_root / report_history.LATEST_DIR}")
    if args.open:
        target = outputs.get("pdf") or outputs.get("md")
        if target and hasattr(os, "startfile"):
            os.startfile(target)  # noqa: S606 - abre no visualizador padrão do Windows
    return 0


if __name__ == "__main__":
    sys.exit(main())
