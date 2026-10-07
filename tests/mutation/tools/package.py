"""Empacota o que o artigo analisou: dados, instrumentos e código, com hashes.

O `Data availability` do artigo promete o pacote de replicação. Este script o
produz a partir do estado atual do repositório e dos logs, de forma que o que
vai para o Zenodo seja gerado por um comando e não montado à mão --- pacote
montado à mão é pacote que diverge do que a análise usou.

Inclui, e a distinção importa:

  dados      o log bruto de execução e os vereditos, mais os logs históricos
             que foram invalidados (eles são a evidência das ameaças que o
             artigo declara, e por isso não são descartados);
  análise    tabelas, figuras e o resumo, exatamente como `analyze.py` os
             produziu;
  instrumentos  suíte, assertivas, catálogo de operadores com o relatório de
             fidelidade, pares de calibração com a revisão, τ*, codebook;
  código     tudo sob `tests/mutation/` e o commit do repositório;
  hashes     sha256 de cada arquivo incluído, no MANIFEST.

Uso:
    python -m tests.mutation.tools.package
    python -m tests.mutation.tools.package --out caminho/do/pacote.zip
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner.console import get_logger  # noqa: E402

logger = get_logger("mutation.package")

ROOT = Path(__file__).resolve().parents[3]
MUT = ROOT / "tests" / "mutation"

# (origem, destino no pacote, obrigatório)
ITENS: list[tuple[Path, str, bool]] = [
    (MUT / "results" / "runs.jsonl", "dados/runs.jsonl", True),
    (MUT / "results" / "verdicts.jsonl", "dados/verdicts.jsonl", True),
    (MUT / "results" / "cost.jsonl", "dados/cost.jsonl", False),
    (MUT / "results" / "coherence.jsonl", "dados/coherence.jsonl", False),
    (MUT / "results" / "PROVENANCE.md", "dados/PROVENANCE.md", True),
    (MUT / "results" / "summary.json", "analise/summary.json", False),
    (MUT / "config" / "study.yaml", "instrumentos/study.yaml", True),
    (MUT / "config" / "codebook.yaml", "instrumentos/codebook.yaml", True),
    (MUT / "suite" / "golden.jsonl", "instrumentos/golden.jsonl", True),
    (MUT / "suite" / "assertions.yaml", "instrumentos/assertions.yaml", True),
    (MUT / "operators" / "catalog.yaml", "instrumentos/catalog.yaml", True),
    (MUT / "operators" / "fidelity_report.json", "instrumentos/fidelity_report.json", False),
    (MUT / "calibration" / "pairs.jsonl", "instrumentos/calibration_pairs.jsonl", True),
    (MUT / "calibration" / "tau.json", "instrumentos/tau.json", True),
    (MUT / "calibration" / "oracle_diagnostic.json", "instrumentos/oracle_diagnostic.json", False),
    (MUT / "corpus" / "manifest.json", "instrumentos/corpus_manifest.json", True),
    (MUT / "corpus" / "sources.yaml", "instrumentos/corpus_sources.yaml", False),
    (MUT / "README.md", "codigo/README.md", True),
    (MUT / "requirements.txt", "codigo/requirements.txt", True),
]

DIRETORIOS: list[tuple[Path, str, str]] = [
    (MUT / "results" / "tables", "analise/tabelas", "*"),
    (MUT / "results" / "figures", "analise/figuras", "*"),
    (MUT / "results" / "pilot", "dados/historico/piloto", "*.jsonl"),
    (MUT / "results" / "archive_v1_truncated", "dados/historico/v1_truncado", "*.jsonl"),
    (MUT / "results" / "archive_v2_fixed_order", "dados/historico/v2_ordem_fixa", "*.jsonl"),
    (MUT, "codigo", "**/*.py"),
]

EXCLUIR = (".venv", "__pycache__", "work", "results", "corpus/base", "corpus/variants")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for bloco in iter(lambda: handle.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return resultado.stdout.strip()


def coletar() -> list[tuple[Path, str]]:
    arquivos: list[tuple[Path, str]] = []
    for origem, destino, obrigatorio in ITENS:
        if origem.exists():
            arquivos.append((origem, destino))
        elif obrigatorio:
            logger.error("Ausente e obrigatório: %s", origem.relative_to(ROOT))
        else:
            logger.warning("Ausente (opcional): %s", origem.relative_to(ROOT))

    for base, destino, padrao in DIRETORIOS:
        if not base.exists():
            continue
        for origem in sorted(base.glob(padrao)):
            if not origem.is_file():
                continue
            # A exclusão vale sobre o caminho RELATIVO à base: o caminho
            # absoluto contém "results", o que descartaria as próprias tabelas
            # e figuras que queremos empacotar.
            relativo = origem.relative_to(base).as_posix()
            if any(parte in relativo for parte in EXCLUIR):
                continue
            arquivos.append((origem, f"{destino}/{relativo}"))
    return arquivos


def leiame(arquivos: list[tuple[Path, str]], commit: str, quando: str) -> str:
    total = sum(origem.stat().st_size for origem, _ in arquivos)
    return f"""# Pacote de replicação --- adequação de suítes para assistentes RAG

Gerado em {quando} a partir do repositório em `{commit}`.
{len(arquivos)} arquivos, {total / 1e6:.1f} MB descompactados.

## O que tem aqui

- `dados/runs.jsonl` --- o log bruto: uma linha por invocação do assistente,
  com a pergunta, os trechos recuperados e suas similaridades, a resposta
  íntegra, contagens de tokens, tempo de parede, GPU-segundo e watt-hora
  medidos, semente, temperatura e configuração de recuperação vigente.
- `dados/verdicts.jsonl` --- um veredito por (nível, mutante, caso, oráculo),
  com o veredito por execução, se o par é estável, se é avaliável e o custo do
  próprio oráculo. É append-only: ao reavaliar, vale o último de cada chave.
- `dados/historico/` --- os logs que foram invalidados durante o estudo, com o
  motivo em `PROVENANCE.md`. Não são descartados porque são a evidência das
  ameaças à validade que o artigo declara.
- `analise/` --- tabelas e figuras como `analyze.py` as produziu, mais o
  `summary.json` com todas as estatísticas.
- `instrumentos/` --- a suíte, as assertivas, o catálogo de operadores com o
  relatório de fidelidade, os pares de calibração, o τ* e o codebook congelado.
- `codigo/` --- o arcabouço completo.

## Como refazer a análise sem reexecutar nada

Julgar é separado de executar: os oráculos leem o log, nunca o sistema. Com
`dados/runs.jsonl` no lugar de `tests/mutation/results/runs.jsonl`:

    python -m tests.mutation.evaluate --level L2 --force
    python -m tests.mutation.analyze

Isso recalcula vereditos e refaz tabelas e figuras. Trocar τ*, o backend do
oráculo O3 ou o modelo juiz muda o resultado sem custar uma invocação do
assistente.

## Como reexecutar a campanha

Ver `codigo/README.md`. O corpus não vai neste pacote por licença dos PDFs de
origem; `instrumentos/corpus_manifest.json` traz os hashes de conteúdo e a
lista de documentos, e o construtor do corpus reconstrói a versão normalizada a
partir dos originais.

## Integridade

`MANIFEST.sha256` traz o sha256 de cada arquivo. Para conferir:

    sha256sum -c MANIFEST.sha256
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Empacota dados, instrumentos e código do estudo.")
    parser.add_argument("--out", help="caminho do .zip (padrão: results/replication-<data>.zip)")
    args = parser.parse_args(argv)

    quando = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    commit = git("rev-parse", "--short", "HEAD") or "sem-git"
    sujo = bool(git("status", "--porcelain"))
    if sujo:
        logger.warning("Árvore de trabalho com mudanças não commitadas: o pacote "
                       "não corresponde exatamente a %s.", commit)

    destino = Path(args.out) if args.out else (
        MUT / "results" / f"replication-{datetime.date.today().isoformat()}.zip")
    destino.parent.mkdir(parents=True, exist_ok=True)

    arquivos = coletar()
    if not arquivos:
        logger.error("Nada a empacotar.")
        return 1

    logger.info("Calculando hashes de %d arquivos...", len(arquivos))
    manifest = [f"# pacote gerado em {quando} do repositório {commit}"]
    for origem, nome in arquivos:
        manifest.append(f"{sha256(origem)}  {nome}")

    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zip_:
        for origem, nome in arquivos:
            zip_.write(origem, nome)
        zip_.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
        zip_.writestr("LEIAME.md", leiame(arquivos, commit, quando))

    tamanho = destino.stat().st_size
    bruto = sum(origem.stat().st_size for origem, _ in arquivos)
    logger.info("Pacote: %s", destino)
    logger.info("%d arquivos, %.1f MB descompactados, %.1f MB compactados (%.0f%%).",
                len(arquivos), bruto / 1e6, tamanho / 1e6, 100 * tamanho / bruto)
    logger.info("sha256 do pacote: %s", sha256(destino))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
