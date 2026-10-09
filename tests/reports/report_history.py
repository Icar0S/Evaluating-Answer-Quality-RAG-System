"""Histórico dos relatórios: cada geração fica na sua pasta e entra num índice.

Layout em tests/reports/out/:

    20261007T111417/        um relatório (relatorio.md/.pdf/.xlsx, notas.csv,
      resumo.json           figuras/, logs/ quando vem do run_all.py) e o
                            resumo.json que alimenta o histórico
    20261008T090000/        ...
    ultimo/                 cópia do relatório mais recente (caminho fixo)
    HISTORICO.md            índice de todas as execuções, mais recente primeiro
    historico.csv           as médias de cada rodada avaliada, em formato longo

Nada é apagado: o histórico é a soma das pastas. Apagar uma pasta tira a
execução do índice na próxima geração.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from report_data import DIMENSIONS, Comparison, Run, dimension_stats, fmt

SUMMARY_FILE = "resumo.json"
LATEST_DIR = "ultimo"


def build_summary(
    runs: list[Run],
    comparisons: list[Comparison],
    execution: dict[str, Any] | None,
    generated_at: datetime,
    files: dict[str, str],
) -> dict[str, Any]:
    """O que fica de cada relatório para o histórico (só números e nomes de arquivo)."""
    return {
        "generated_at": generated_at.astimezone().isoformat(),
        "origin": "run_all" if execution else "manual",
        "execution": None
        if not execution
        else {
            "started_at": execution.get("started_at"),
            "duration_seconds": execution.get("duration_seconds"),
            "steps": [
                {k: s.get(k) for k in ("key", "name", "status", "tests", "seconds")} for s in execution.get("steps", [])
            ],
            "shared_answers": bool(execution.get("shared_answers")),
        },
        "runs": [
            {
                "framework": run.framework,
                "label": run.label,
                "run_generated_at": run.generated_at.isoformat() if run.generated_at else None,
                "generation_model": run.generation_model,
                "judge_model": run.judge_model,
                "threshold": run.threshold,
                "n_cases": len(run.cases),
                "all_passed": sum(1 for c in run.cases if c.all_passed),
                "shared_answers": bool(run.extra.get("answers_source")),
                "means": {k: s.mean for k, s in dimension_stats(run).items()},
                "passed": {k: s.n_passed for k, s in dimension_stats(run).items()},
                "source": str(run.source),
            }
            for run in runs
        ],
        "comparisons": [
            {
                "a": c.a.label,
                "b": c.b.label,
                "identical_answers": c.identical_answers,
                "identical_contexts": c.identical_contexts,
                "common_cases": len(c.common_cases),
                "agreement": {k: a.agreement for k, a in c.per_dimension.items()},
            }
            for c in comparisons
        ],
        "files": files,
    }


def write_summary(out_dir: Path, summary: dict[str, Any]) -> Path:
    path = out_dir / SUMMARY_FILE
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_entries(root: Path) -> list[dict[str, Any]]:
    """Os resumos de todas as pastas de relatório, do mais antigo ao mais recente."""
    entries = []
    if not root.is_dir():
        return entries
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != LATEST_DIR):
        path = folder / SUMMARY_FILE
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["_folder"] = folder.name
        entries.append(data)
    entries.sort(key=lambda e: e.get("generated_at") or "")
    return entries


def unique_runs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cada rodada avaliada uma vez só, pela data da AVALIAÇÃO (não do relatório).

    Gerar dois relatórios a partir do mesmo JSON não pode virar dois pontos na
    evolução das métricas.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        for run in entry.get("runs", []):
            key = (run.get("framework"), run.get("run_generated_at") or run.get("source"))
            seen.setdefault(key, run)
    return sorted(seen.values(), key=lambda r: r.get("run_generated_at") or "")


def _steps_line(entry: dict[str, Any]) -> str:
    execution = entry.get("execution")
    if not execution:
        return "— (relatório gerado à mão)"
    icons = {"passou": "✓", "concluída": "✓", "falhou": "✗", "pulada": "⊘", "erro": "⚠"}
    return " · ".join(f"{s['name']} {icons.get(s['status'], '?')}" for s in execution.get("steps", []))


def write_index(root: Path, entries: list[dict[str, Any]]) -> tuple[Path, Path]:
    """HISTORICO.md (uma linha por relatório) e historico.csv (médias por rodada avaliada)."""
    lines = [
        "# Histórico dos relatórios de qualidade",
        "",
        "Gerado por `tests/reports/build_report.py` (e `tests/run_all.py`) a cada relatório. "
        "Mais recente primeiro. Ícones: ✓ passou · ✗ falhou · ⊘ pulada · ⚠ erro.",
        "",
        "| Data | Execução | Rodadas avaliadas | Relatório |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for entry in reversed(entries):
        when = datetime.fromisoformat(entry["generated_at"]).strftime("%d/%m/%Y %H:%M")
        runs = "<br>".join(
            f"**{r['framework']}**: {r['all_passed']}/{r['n_cases']} casos com todas ≥ limiar"
            + (" (respostas compartilhadas)" if r.get("shared_answers") else "")
            for r in entry.get("runs", [])
        )
        folder = entry["_folder"]
        files = entry.get("files", {})
        links = " · ".join(f"[{fmt_name}]({folder}/{name})" for fmt_name, name in files.items())
        lines.append(f"| {when} | {_steps_line(entry)} | {runs or '—'} | {links or folder} |")
    md_path = root / "HISTORICO.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = root / "historico.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["data_avaliacao", "framework", "modelo_geracao", "juiz", "respostas_compartilhadas",
                         "dimensao", "media", "aprovadas", "casos"])
        for run in unique_runs(entries):
            for dim in DIMENSIONS:
                mean = run["means"].get(dim.key)
                writer.writerow([
                    run.get("run_generated_at"), run["framework"], run.get("generation_model"), run.get("judge_model"),
                    "sim" if run.get("shared_answers") else "não", dim.key,
                    "" if mean is None else round(mean, 4), run["passed"].get(dim.key), run["n_cases"],
                ])
    return md_path, csv_path


def refresh_latest(root: Path, out_dir: Path) -> Path:
    """Copia o relatório recém-gerado para root/ultimo (caminho que não muda).

    Cópia, não link: link simbólico no Windows pede privilégio de administrador.
    """
    target = root / LATEST_DIR
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(out_dir, target)
    return target


def history_table_rows(runs: list[dict[str, Any]], limit: int = 15) -> list[list[str]]:
    rows = []
    for run in list(reversed(runs))[:limit]:
        when = run.get("run_generated_at")
        when = datetime.fromisoformat(when).astimezone().strftime("%d/%m/%Y %H:%M") if when else "—"
        rows.append(
            [when, run["framework"]]
            + [fmt(run["means"].get(d.key)) for d in DIMENSIONS]
            + [f"{run['all_passed']}/{run['n_cases']}", "sim" if run.get("shared_answers") else "não"]
        )
    return rows
