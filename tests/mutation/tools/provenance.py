"""Gera results/PROVENANCE.md: o que cada log de execução é, de onde veio e por quê.

Roda com o venv do estudo, a partir da raiz do repositório.
"""
import hashlib
import json
import pathlib
import datetime
import statistics as st
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
R = ROOT / "tests/mutation/results"

ENTRIES = [
    ("pilot/runs.jsonl", "pilot/verdicts.jsonl", "e3fae9e", "2026-09-16",
     "Piloto da semana 5: 3 casos (c01, c07, c16) x (baseline, R1, P2) x 2 repetições.",
     "Rodou com o prompt TRUNCADO (ver §4.13) e com as perguntas anteriores de c01. "
     "Serve como evidência do piloto e da calibração do O4, não como medida do SUT."),
    ("archive_v1_truncated/runs_baseline_v1.jsonl", "archive_v1_truncated/verdicts_baseline_v1.jsonl",
     "bfcd105", "2026-09-21 17:21–19:41",
     "Baseline v1 completo, 30 casos x 10 repetições.",
     "INVÁLIDO: as 300 invocações rodaram sem `num_ctx` declarado e tiveram o prompt cortado "
     "em 2.050 tokens (288 avisos `truncating input prompt` no log do Ollama). O system prompt "
     "não chegou ao modelo. Mantido porque é a evidência do achado §4.13 e da ameaça do §9."),
    ("archive_v1_truncated/runs_partial_v2.jsonl", "archive_v1_truncated/verdicts_partial_v2.jsonl",
     "859a8b9", "2026-09-21 20:22–20:29",
     "Tentativa parcial depois da reescrita de 8 perguntas, ainda sem `num_ctx`.",
     "INVÁLIDA pelo mesmo motivo. 8 invocações."),
    ("archive_v2_fixed_order/runs.jsonl", "archive_v2_fixed_order/verdicts.jsonl",
     "ed20e49", "2026-09-21 21:08–23:23",
     "Baseline v2: primeiro baseline sem truncagem. 30 casos x 10 repetições.",
     "VÁLIDO como medida, SUPERSEDIDO como amostra: os casos rodaram em ordem fixa, e a "
     "ordem fixa faz r2..r10 repetirem o mesmo predecessor no cache do servidor (respostas "
     "byte a byte iguais). |S| = O1 25, O4 23, O2 21, O5 21. É o par de comparação do v3."),
    ("runs.jsonl", "verdicts.jsonl", "4590c49", "2026-09-22 00:04 em diante",
     "Baseline v3 (ordem embaralhada por repetição) e, na sequência, a campanha dos 18 mutantes.",
     "Baseline v3 COMPLETO (300 invocações, 0 erros depois do retry) e portão verde: "
     "|S| = O1 26, O4 22, O2 20, O5 20. Campanha dos 18 mutantes em curso desde 22/09 10:51. "
     "Três anomalias tratadas: `baseline-c18-r1` atravessou uma suspensão da máquina (26.463 s "
     "e 58,4 Wh — custo inválido, resposta válida); execuções de c14 bateram o teto de geração "
     "com resposta vazia (4 dentro das 5 repetições do nível L2) e ficam fora dos vereditos; "
     "`baseline-c23-r10` recebeu 500 do servidor quando outra aplicação tomou a VRAM, foi "
     "removida com `--retry-errors` e refeita."),
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(path: pathlib.Path) -> dict:
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs or "answer" not in runs[0]:
        return {"linhas": len(runs)}
    wall = [r["wall_ms"] / 1000 for r in runs]
    return {
        "linhas": len(runs),
        "casos": len({r["case_id"] for r in runs}),
        "mutantes": len({r["mutant_id"] for r in runs}),
        "erros": sum(1 for r in runs if r.get("error")),
        "wall_mediana_s": round(st.median(wall), 1),
        "wh_total": round(sum(r["wh"] for r in runs if r.get("wh")), 1),
    }


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    out = [
        "# Proveniência dos logs de execução",
        "",
        "Cada campanha deste estudo deixou um log. Alguns foram invalidados por defeitos de",
        "instrumentação descobertos depois, e **nenhum foi apagado**: um log inválido é a",
        "evidência da ameaça à validade que ele revelou, e o artigo o cita como tal.",
        "Este arquivo diz o que cada um é, com que código foi produzido e por que foi",
        "superado. É o índice do que vai para o Zenodo.",
        "",
        "Os logs em si **não são versionados** (`.gitignore`: `tests/mutation/results/*`) —",
        "são grandes e regeneráveis pelos comandos do README. O que o repositório versiona é",
        "este índice, com os hashes, mais o código, a configuração, a suíte e a calibração.",
        "",
        f"Gerado em {datetime.datetime.now():%Y-%m-%dT%H:%M}",
        f"com o repositório em `{head}`.",
        "",
    ]
    for runs_rel, verdicts_rel, commit, when, what, why in ENTRIES:
        runs_path = R / runs_rel
        out.append(f"## `{runs_rel}`")
        out.append("")
        if not runs_path.exists():
            out.append("_ausente no disco._")
            out.append("")
            continue
        info = stats(runs_path)
        out.append(f"- **Quando:** {when}  ")
        out.append(f"- **Código:** `{commit}`  ")
        out.append(f"- **O que é:** {what}  ")
        out.append(f"- **Estado:** {why}  ")
        out.append(f"- **Conteúdo:** {info}  ")
        out.append(f"- **sha256 (runs):** `{sha256(runs_path)}`  ")
        verdicts_path = R / verdicts_rel
        if verdicts_path.exists():
            v = stats(verdicts_path)
            out.append(f"- **Vereditos:** `{verdicts_rel}`, {v['linhas']} linhas, "
                       f"sha256 `{sha256(verdicts_path)}`  ")
        else:
            out.append(f"- **Vereditos:** `{verdicts_rel}` ainda não gerado  ")
        out.append("")

    out += [
        "## Como refazer cada um",
        "",
        "```bash",
        "# v1 (truncado) — só para reproduzir o achado §4.13:",
        "#   remova generation_num_ctx/generation_num_predict de config/study.yaml",
        "#   e observe `truncating input prompt` no log do servidor Ollama.",
        "# v2 (ordem fixa):",
        "#   shuffle_cases_per_repetition: false em config/study.yaml",
        "python -m tests.mutation.run_campaign --baseline",
        "# v3 (ordem embaralhada, configuração atual):",
        "python -m tests.mutation.run_campaign --baseline",
        "python -m tests.mutation.run_campaign --campaign",
        "python -m tests.mutation.evaluate",
        "```",
        "",
        "Cada etapa é retomável: `run_id` já presente em `runs.jsonl` é pulado.",
        "",
    ]
    (R / "PROVENANCE.md").write_text("\n".join(out), encoding="utf-8")
    print("escrito:", R / "PROVENANCE.md")


if __name__ == "__main__":
    main()
