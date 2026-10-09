# Relatório consolidado das avaliações de qualidade

Junta os resultados das suítes de qualidade de resposta em **um relatório
só**, separado por framework e com a comparação entre eles. Hoje as suítes são
[RAGAS](../ragas/README.md) e [DeepEval](../deepeval/README.md). O gerador
só lê os JSONs que cada suíte já gravou: não chama o Ollama, não importa
`ragas`/`deepeval` e não precisa do backend no ar.

| Formato | Arquivo | Para quê |
|---|---|---|
| Markdown | `relatorio.md` + `figuras/` | Ler no GitHub ou no VS Code, colar em issue/PR |
| PDF | `relatorio.pdf` | Anexar, imprimir, mandar para quem não abre código |
| Planilha | `relatorio.xlsx` | Filtrar, pivotar, refazer contas |
| CSV | `notas.csv` | pandas/R (uma linha por rodada × caso × dimensão) |

O Markdown e o PDF têm o mesmo conteúdo, montado uma vez só
(`report_document.py`).

## Setup (uma vez)

```powershell
python -m venv tests\reports\.venv
tests\reports\.venv\Scripts\pip install -r tests\reports\requirements.txt
```

O venv é próprio e leve (`matplotlib`, `openpyxl`, `reportlab`), e não
conflita com os das suítes.

## Rodar

Depois de rodar o `run_and_export.py` de pelo menos uma suíte, a partir da
raiz do repositório:

```powershell
# o resultado mais recente de cada framework, nos quatro formatos
tests\reports\.venv\Scripts\python.exe tests\reports\build_report.py

# só alguns formatos, abrindo o PDF ao terminar
tests\reports\.venv\Scripts\python.exe tests\reports\build_report.py --formats pdf xlsx --open

# rodadas específicas (de qualquer framework; pode repetir --input)
tests\reports\.venv\Scripts\python.exe tests\reports\build_report.py `
    --input tests\ragas\results\20261007T115107Z.json `
    --input tests\deepeval\results\20260827T163236Z.json

# outra pasta de saída
tests\reports\.venv\Scripts\python.exe tests\reports\build_report.py --out C:\temp\relatorio
```

A saída vai para `tests/reports/out/<data>/`, que não é versionada (mesma
regra das pastas `results/` das suítes).

Para rodar **todas as suítes e gerar o relatório no fim**, use
`scripts\run_all_tests.bat` (ou `tests\run_all.py`), descrito no README da
raiz, em "Rodar tudo de uma vez". O relatório gerado assim ganha a seção
**Execução dos testes**: o resultado de cada suíte, a duração, o motivo de
cada uma pulada e o caminho do log.

## Histórico

Nenhum relatório apaga o anterior. Em `tests/reports/out/`:

```
20261007T130627/        uma execução: relatorio.md/.pdf/.xlsx, notas.csv, figuras/,
                        resumo.json, e (vindo do run_all) execucao.json,
                        respostas.json, logs/ e junit/
20261008T090000/        a próxima...
ultimo/                 cópia da mais recente: o caminho que não muda
HISTORICO.md            índice de todas, mais recente primeiro, com links
historico.csv           as médias de cada rodada avaliada, em formato longo
```

- O relatório traz uma seção **Histórico**, com a evolução da média de cada
  dimensão por framework ao longo das execuções (gráfico e tabela).
- Cada rodada de avaliação conta **uma vez só**, pela data da avaliação. Gerar
  dois relatórios a partir do mesmo resultado não duplica o ponto.
- Apagar uma pasta tira aquela execução do histórico na próxima geração.
- `--sem-historico` gera um relatório sem entrar no índice. Um `--out` para
  fora de `tests/reports/out/` também fica fora do histórico.
- `--execution caminho\execucao.json` acrescenta a seção de execução a um
  relatório gerado à mão.

## O que o relatório traz

1. **Resumo.** Para cada rodada: quantos casos passam em todas as dimensões, a
   melhor e a pior dimensão, notas não medidas e abstenções do assistente.
   Traz também os **avisos de comparabilidade** entre rodadas (abaixo).
2. **Rodadas avaliadas.** Framework, data, modelo de geração, juiz, limiar,
   arquivo de origem e os metadados próprios de cada uma (versão do ragas,
   chamadas e falhas do juiz, alterações de prompt).
3. **Comparação entre frameworks.** Gráfico e tabela com a média de cada
   dimensão, a concordância de veredito por dimensão e a lista de
   **divergências** (casos que um framework aprova e o outro reprova).
4. **Resultados por framework.** Estatísticas por dimensão, heatmap caso ×
   dimensão (escala divergente centrada no limiar), tabela de notas e as notas
   que não foram medidas, com o motivo.
5. **Detalhe por caso.** Pergunta, referência e, por rodada, a resposta
   avaliada, as notas e a justificativa do juiz (quando o framework grava
   uma, como o DeepEval).
6. **Como ler.** A equivalência entre as métricas dos dois frameworks e as
   regras de leitura.

As cinco **dimensões** são as que as duas suítes medem:

| Dimensão | RAGAS | DeepEval |
|---|---|---|
| Fidelidade ao contexto | `Faithfulness` | `FaithfulnessMetric` |
| Relevância da resposta | `AnswerRelevancy` | `AnswerRelevancyMetric` |
| Precisão do contexto | `ContextPrecisionWithReference` | `ContextualPrecisionMetric` |
| Recall do contexto | `ContextRecall` | `ContextualRecallMetric` |
| Relevância do contexto | `ContextRelevance` | `ContextualRelevancyMetric` |

### Avisos de comparabilidade

Uma diferença entre RAGAS e DeepEval só é diferença entre *métodos* se o resto
for igual. O relatório confere e avisa quando não for:

- mesmos casos;
- mesmo modelo de geração;
- mesmo juiz;
- mesmo limiar;
- **mesmas respostas e mesmos trechos recuperados.** A geração não é
  determinística, então duas rodadas quase nunca julgam o mesmo texto.

Com os resultados de hoje (DeepEval de 27/08, RAGAS de 07/10), modelo de
geração e juiz são os mesmos, mas nenhuma resposta e nenhum conjunto de
trechos coincide. A recuperação mudou entre as duas datas:

- 20 dos 39 PDFs de `data/source_pdfs/` têm data posterior à rodada de agosto;
- em `metarag-definicao`, o primeiro trecho recuperado em agosto era do artigo
  do MetaRAG, e hoje nenhum dos quatro é.

Para isolar o efeito do framework, o próximo passo é julgar **as mesmas
respostas** com os dois.

### Regras de leitura

- **Veredito** = nota ≥ limiar da rodada.
- **n/d** = a métrica não produziu nota (erro, timeout ou NaN). A nota fica
  fora das médias, mas nas contagens "≥ limiar" conta como não aprovada.
- **†** = o assistente se absteve ("não encontrei…"). A frase vem de
  `backend/app/rag/generation.py::NOT_FOUND_MARKER`.
- Os gráficos seguem a paleta validada do projeto: a cor é fixa por framework
  (DeepEval azul, RAGAS laranja) e o heatmap vai de vermelho (abaixo do
  limiar) a azul (acima), passando por cinza no limiar. Toda figura tem a
  tabela com os números logo abaixo.

## Adicionar outro framework ou metodologia

Para entrar no relatório, um novo avaliador precisa gravar o JSON no mesmo
formato das suítes atuais (`cases[]` com `name`, `input`, `actual_output`,
`expected_output`, `retrieval_context` e `metrics[]` com `name`, `score`,
`success`, `reason` e `error`). Em `report_data.py`:

1. ensine `detect_framework()` a reconhecê-lo;
2. mapeie os nomes das métricas dele para as dimensões em `_METRIC_TO_DIMENSION`
   (e em `NATIVE_METRIC`);
3. se ele tiver uma pasta `results/` própria, acrescente-a em `RESULTS_DIRS`.

O resto (gráficos, comparação par a par, os quatro formatos) sai de graça. Em
`report_charts.py`, `FRAMEWORK_SLOT` fixa a cor dele. Sem isso, ele pega o
próximo slot livre da paleta.

## Testes do gerador

Herméticos (JSONs sintéticos, sem Ollama):

```powershell
tests\reports\.venv\Scripts\python.exe -m pytest tests/reports
```

| Arquivo | Papel |
|---|---|
| `build_report.py` | Linha de comando e orquestração |
| `report_data.py` | Leitura dos JSONs, modelo comum, estatísticas e comparação |
| `report_document.py` | O conteúdo do relatório, como blocos (fonte do MD e do PDF) |
| `report_markdown.py` / `report_pdf.py` | Renderização dos blocos |
| `report_sheets.py` | Planilha `.xlsx` e CSV |
| `report_charts.py` | Gráficos (PNG) |
| `report_history.py` | `resumo.json`, `HISTORICO.md`, `historico.csv` e a cópia em `ultimo/` |
| `test_run_all.py` | Testes do orquestrador (`tests/run_all.py`) e do histórico |
