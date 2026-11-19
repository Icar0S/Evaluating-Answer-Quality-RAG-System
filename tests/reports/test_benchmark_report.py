"""Relatório do benchmark (benchmark_report.py): formatação, marcadores, vereditos e o PDF, com rodadas falsas."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import benchmark_data as bd
import benchmark_report as br
from report_document import Bullets, Callout, Figure, Heading, Para, Table


def _delta(delta: float, low: float, high: float, n: int = 4, reference: float = 0.5) -> dict:
    return {"delta": delta, "ci_low": low, "ci_high": high, "abs_p95": abs(delta), "reference_mean": reference,
            "arm_mean": reference + delta, "n": n, "clusters": 2, "one_side_only": 0}


CARD = {
    "id": "EX", "titulo": "Teste", "pergunta": "O B ganha do A?", "hipotese": "O B ganha.", "didatica": "Meça.",
    "metrica_primaria": {"camada": "c2", "metrica": "faithfulness"},
    "bracos": {"A": {"indice": "pdf-v1", "papel": "baseline"}, "B": {"indice": "pdf-v1", "set": {"top_k": 8}}},
    "guardas": [{"nome": "latência", "par": ["A", "B"], "metrica": "cost.generation_ms", "tolerancia": 3000}],
}
PLACAR = {
    "means": {"A": {"c2.faithfulness": 0.7, "abstention.correct_abstention": 0.5, "cost.generation_ms": 2500.0},
              "B": {"c2.faithfulness": 0.85, "abstention.correct_abstention": 1.0, "cost.generation_ms": 3700.0}},
    "pairs": {"A→B": {"c2.faithfulness": _delta(0.15, 0.05, 0.25, reference=0.7),
                      "cost.generation_ms": _delta(1200.0, 600.0, 1600.0, reference=2500.0)}},
    "verdicts": {"A→B": {"metric": "c2.faithfulness", "margin": 0.05, "delta": _delta(0.15, 0.05, 0.25, reference=0.7),
                         "margin_from": "cartão", "verdict": "melhor"}},
    "margins": {"gerador_juiz": {"c2.faithfulness": 0.05, "abstention.correct_abstention": 0.0,
                                 "cost.generation_ms": 200.0}},
    "guards": [{"nome": "latência", "par": ["A", "B"], "metrica": "cost.generation_ms", "tolerance": 3000,
                "delta": _delta(1200.0, 600.0, 1600.0), "passed": True}],
    "summary": {"A": {"judge": {"judge_model": "qwen3:8b"}}},
}
CLUSTERS = {"q1": "d1", "q2": "d1", "q3": "d2", "q4": "d2", "f1": "f1", "f2": "f2"}
MEDIDAS = {
    "A": {"clusters": CLUSTERS, "metrics": {"abstention.correct_abstention": {"f1": 1.0, "f2": 0.0},
                                            "c2.faithfulness": {"q1": 0.6, "q2": 0.8, "q3": 0.7, "q4": 0.7}}},
    "B": {"clusters": CLUSTERS, "metrics": {"abstention.correct_abstention": {"f1": 1.0, "f2": 1.0},
                                            "c2.faithfulness": {"q1": 0.9, "q2": 0.8, "q3": 0.9, "q4": 0.8}}},
}


def _run_dir(root: Path, name: str, placar: dict = PLACAR, medidas: dict = MEDIDAS) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "cartao.yaml").write_text(yaml.safe_dump(CARD, allow_unicode=True), encoding="utf-8")
    (path / "placar.json").write_text(json.dumps(placar, ensure_ascii=False), encoding="utf-8")
    (path / "medidas.json").write_text(json.dumps(medidas), encoding="utf-8")
    (path / "parametros.json").write_text(json.dumps({"limite": None}), encoding="utf-8")
    (path / "execucao.json").write_text(json.dumps({"runs": [
        {"started_at": "2026-10-19T01:46:02-03:00", "duration_seconds": 9000.0, "so_placar": False},
        {"started_at": "2026-10-19T05:00:00-03:00", "duration_seconds": 2.0, "so_placar": True}]}), encoding="utf-8")
    for arm, outputs in (("A", ["Resposta.", "Outra."]), ("B", ["Resposta.", "Mudou."])):
        answers = [{"name": f"q{i}", "actual_output": o} for i, o in enumerate(outputs, 1)]
        (path / f"respostas_{arm}.json").write_text(json.dumps({"answers": answers}), encoding="utf-8")
    return path


def _report(tmp_path: Path, body: str, meta: dict | None = None) -> tuple[br.Report, br.Section]:
    runs = tmp_path / "out"
    if not (runs / "EX_1").exists():
        _run_dir(runs, "EX_1")
    meta = {"id": "EX", "titulo": "Teste", "rodadas": {"r": "EX_1"}, **(meta or {})}
    text_path = tmp_path / "EX.md"
    text_path.write_text(f"---\n{yaml.safe_dump(meta, allow_unicode=True)}---\n{body}\n", encoding="utf-8")
    report = br.Report(tmp_path, runs, tmp_path / "figuras")
    section = br.Section(report, br.read_text(text_path))
    report.sections = [section]
    return report, section


def test_numeros_no_padrao_brasileiro():
    assert bd.num(1234.5, 1) == "1.234,5"
    assert bd.num(-0.067) == "−0,07"
    assert bd.signed(0.0) == "+0,00" and bd.signed(-0.2) == "−0,20"
    assert bd.pct(0.633) == "63%"
    assert bd.fmt_value("cost.generation_ms", 2580.5) == "2,6 s"
    assert bd.fmt_value("c1.retrieval_ms", 246.9) == "247 ms"
    assert bd.fmt_value("cost.prompt_tokens", 3286.2) == "3.286"
    assert bd.fmt_delta("c2.faithfulness", _delta(0.367, 0.2, 0.533)) == "+0,37 [+0,20; +0,53]"
    assert bd.fmt_delta("cost.generation_ms", _delta(1155.2, 639.7, 1616.1)) == "+1,2 s [+0,6; +1,6]"
    assert bd.fmt_delta("c1.retrieval_ms", _delta(113.0, 80.0, 150.0)) == "+113 ms [+80; +150]"


def test_marcadores_preenchem_com_o_placar_e_as_medidas(tmp_path: Path):
    report, section = _report(tmp_path, "")

    text = section.resolve("De {{media r A c2.faithfulness}} para {{media r B c2.faithfulness}} "
                           "({{delta r A→B c2.faithfulness}}, {{veredito r A→B}}); de fora: "
                           "{{contagem r B abstention.correct_abstention}}; margem {{margem r c2.faithfulness}}; "
                           "latência {{dif r A→B cost.generation_ms}}.")

    assert text == ("De 0,70 para 0,85 (+0,15 [+0,05; +0,25], melhor); de fora: 2 de 2; margem 0,050; "
                    "latência +1,2 s.")
    assert report.warnings == []


def test_marcador_que_nao_existe_vira_aviso_e_nao_quebra(tmp_path: Path):
    report, section = _report(tmp_path, "")

    text = section.resolve("{{media r Z c2.faithfulness}} e {{inventado r}}")

    assert text == f"{br.MISSING} e {br.MISSING}"
    assert len(report.warnings) == 2 and "media r Z" in report.warnings[0]


def test_veredito_que_mudou_avisa_na_secao(tmp_path: Path):
    expected = {"r A→B": "pior", "r A→B cost.generation_ms": "melhor"}
    report, section = _report(tmp_path, "Texto.", {"vereditos": expected})

    blocks = section.blocks()

    callout = next(b for b in blocks if isinstance(b, Callout))
    assert callout.tone == "aviso" and "outro resultado" in callout.title
    assert "“pior”" in callout.items[0] and "“melhor”" in callout.items[0]
    # Métrica fora do veredito: a regra do placar, com a margem do cartão (+1,2 s, menos é melhor: pior).
    assert any("latência da geração" in w and "“pior”" in w for w in report.warnings)


def test_corpo_markdown_vira_blocos(tmp_path: Path):
    body = "\n".join([
        "### Um título", "", "Um parágrafo", "em duas linhas.", "",
        "1. primeiro", "2. segundo", "", "- item", "  continua", "",
        "> **Para a turma**", "> Uma frase.", "",
        "> [!aviso] Cuidado", "> - um", "> - dois", "",
        "| Termo | O que é |", "|---|---:|", "| a | b |", "",
        "[[cartao r]]", "[[vereditos r]]", "[[guardas r]]",
    ])
    report, section = _report(tmp_path, body)

    blocks = section.blocks()

    kinds = [type(b).__name__ for b in blocks]
    assert kinds[:3] == ["PageBreak", "Heading", "Heading"]
    assert any(isinstance(b, Para) and b.text == "Um parágrafo em duas linhas." for b in blocks)
    numbered = [b for b in blocks if isinstance(b, Bullets)]
    assert numbered[0].numbered and numbered[0].items == ["primeiro", "segundo"]
    assert not numbered[1].numbered and numbered[1].items == ["item continua"]
    notes = [b for b in blocks if isinstance(b, Callout)]
    assert (notes[0].tone, notes[0].title, notes[0].items) == ("nota", "Para a turma", ["Uma frase."])
    assert (notes[1].tone, notes[1].title, notes[1].items) == ("aviso", "Cuidado", ["um", "dois"])
    tables = [b for b in blocks if isinstance(b, Table)]
    assert tables[0].headers == ["Termo", "O que é"] and tables[0].align == ["l", "r"]
    assert tables[1].rows[0][1] == "O B ganha do A?"  # o cartão
    assert tables[3].rows[0][-2:] == ["melhor", "4"]  # os vereditos
    assert tables[4].rows[0][-1] == "✓ passa"  # as guardas
    assert report.warnings == []


def test_figuras_saem_em_png_com_a_tabela(tmp_path: Path):
    meta = {"figuras": {
        "dif": {"tipo": "diferencas", "rodada": "r", "pares": ["A→B"], "metricas": ["c2.faithfulness"],
                "legenda": "A diferença.", "como_ler": "Leia assim."},
        "medias": {"tipo": "medias", "rodada": "r", "bracos": ["A", "B"],
                   "metricas": ["c2.faithfulness", "abstention.correct_abstention"]},
        "custo": {"tipo": "custo_qualidade", "rodada": "r", "bracos": ["A", "B"], "y": ["c2.faithfulness"]},
    }}
    report, section = _report(tmp_path, "[[figura dif]]\n[[figura medias]]\n[[figura custo]]", meta)

    blocks = section.blocks()

    figures = [b for b in blocks if isinstance(b, Figure)]
    assert len(figures) == 3 and all(f.path.is_file() for f in figures)
    assert any(isinstance(b, Callout) and b.title == "Como ler este gráfico" for b in blocks)
    medias = [b for b in blocks if isinstance(b, Table)][1]
    assert medias.rows[1][1:] == ["0,50 (1 de 2)", "1,00 (2 de 2)"]
    assert report.warnings == []


def test_comparacao_entre_rodadas_e_respostas_que_mudam(tmp_path: Path):
    other = {**MEDIDAS, "B": {"clusters": CLUSTERS, "metrics": {"abstention.correct_abstention": {"f1": 0.0, "f2": 0.0}}}}
    _run_dir(tmp_path / "out", "EX_2", medidas=other)
    meta = {"rodadas": {"r": "EX_1", "s": "EX_2"}, "analises": {
        "preco": {"tipo": "comparacao", "de": "r B", "para": "s B", "metricas": ["abstention.correct_abstention"]},
        "mudou": {"tipo": "mudancas", "rodada": "r", "de": "A", "para": "B"}}}
    report, section = _report(tmp_path, "", meta)

    text = section.resolve("{{comparacao preco abstention.correct_abstention}}; "
                           "{{valor mudou changed int}} de {{valor mudou total int}}")

    assert text == "−1,00 [−1,00; −1,00]; 1 de 2"


def test_main_gera_pdf_e_markdown(tmp_path: Path):
    _run_dir(tmp_path / "out", "EX_1")
    texts = tmp_path / "textos"
    texts.mkdir()
    (texts / "resumo.md").write_text("---\ntitulo: Resumo\nquebra: false\n---\n[[resumo]]\n\n[[ajustes]]\n",
                                     encoding="utf-8")
    meta = {"id": "EX", "titulo": "Teste", "rodadas": {"r": "EX_1"}, "vereditos": {"r A→B": "melhor"},
            "resumo": "Sobe {{dif r A→B c2.faithfulness}}.", "recomendacao": "Ligar.",
            "ajuste": [{"variavel": "TOP_K", "valor": "8", "motivo": "porque sim"}],
            "figuras": {"dif": {"tipo": "diferencas", "rodada": "r", "legenda": "Δ."}}}
    (texts / "EX.md").write_text(f"---\n{yaml.safe_dump(meta, allow_unicode=True)}---\n[[cartao r]]\n\n"
                                 "[[figura dif]]\n", encoding="utf-8")
    manifest = tmp_path / "relatorio.yaml"
    manifest.write_text(yaml.safe_dump({"titulo": "Relatório de teste", "pasta_rodadas": "out", "avisos": ["Rascunho."],
                                        "textos": ["textos/resumo.md", "textos/EX.md"]}, allow_unicode=True),
                        encoding="utf-8")
    out = tmp_path / "saida"

    assert br.main(["--manifesto", str(manifest), "--out", str(out)]) == 0

    markdown = (out / "relatorio-benchmark.md").read_text(encoding="utf-8")
    assert "Sobe +0,15." in markdown and "`TOP_K`" in markdown and "## EX — Teste" in markdown
    assert "Nenhum aviso" in markdown and "2 h 30" in markdown  # o apêndice: a duração sem o --so-placar
    assert (out / "relatorio-benchmark.pdf").stat().st_size > 10_000
    assert (out / "figuras" / "ex-dif.png").is_file()


def test_manifesto_com_rodada_que_nao_existe_falha_com_mensagem(tmp_path: Path, capsys):
    (tmp_path / "EX.md").write_text("---\nid: EX\nrodadas: {r: NAO_EXISTE}\n---\nnada\n", encoding="utf-8")
    manifest = tmp_path / "relatorio.yaml"
    manifest.write_text("titulo: x\npasta_rodadas: .\ntextos: [EX.md]\n", encoding="utf-8")

    assert br.main(["--manifesto", str(manifest), "--out", str(tmp_path / "saida"), "--formatos", "md"]) == 1
    assert "NAO_EXISTE" in capsys.readouterr().err


@pytest.mark.parametrize("path", sorted((bd.PROJECT_ROOT / "tests" / "benchmark" / "relatorio" / "textos").glob("*.md")))
def test_textos_do_repositorio_tem_cabecalho_valido(path: Path):
    text = br.read_text(path)
    assert text.body, path.name
    for figure in (text.meta.get("figuras") or {}).values():
        assert hasattr(br.Section, f"_f_{figure['tipo']}"), figure
    for table in (text.meta.get("tabelas") or {}).values():
        assert hasattr(br.Section, f"_t_{table['tipo']}"), table
