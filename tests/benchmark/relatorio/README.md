# O relatório do benchmark

Um PDF didático (e o mesmo conteúdo em Markdown) com os resultados de todos os experimentos, gerado a partir das
rodadas em `tests/benchmark/out/`. O código fica em `tests/reports/benchmark_report.py`; o **conteúdo** fica
aqui, e é aqui que se ajusta o tom.

## Gerar

Da raiz, com o venv do relatório (`tests/reports/.venv`, ver `tests/reports/README.md`):

```powershell
tests\reports\.venv\Scripts\python.exe tests\reports\benchmark_report.py           # PDF + Markdown
tests\reports\.venv\Scripts\python.exe tests\reports\benchmark_report.py --abrir   # e abre o PDF
```

A saída vai para `tests/benchmark/out/relatorio_<data>/`: `relatorio-benchmark.pdf`, o `.md` e `figuras/`. O
gerador só lê os arquivos das rodadas; não chama o Ollama nem o juiz, e leva segundos.

**Depois de rodar um experimento de novo:** troque a pasta da rodada no cabeçalho do texto dele (`rodadas:`) e
gere de novo. Os números do texto se atualizam sozinhos. Se um veredito mudar, o relatório avisa na própria
seção e no apêndice, e aí é o texto que precisa ser revisto.

## Onde está cada coisa

| Arquivo | O que é |
|---|---|
| `relatorio.yaml` | o título, os avisos da capa e a ordem dos textos |
| `textos/resumo.md` | o resumo para quem mantém o RAG: a tabela dos experimentos e o `.env` recomendado |
| `textos/como-testamos.md` | como o teste foi feito, para a turma: o funil, as regras, como ler um veredito |
| `textos/E*.md` | uma seção por experimento |
| `textos/limitacoes.md`, `textos/glossario.md` | as ressalvas e o glossário |

## O formato de um texto

Um Markdown com um cabeçalho YAML entre duas linhas `---`.

```markdown
---
id: E9                      # sem id, o texto é uma seção comum (resumo, glossário)
titulo: O gerador
rodadas:                    # apelido: pasta em tests/benchmark/out/
  e9: E9_20261019T014602
vereditos:                  # o que o texto afirma; se o placar disser outra coisa, o relatório avisa
  e9 gemma→llama: inconclusivo                               # o veredito da métrica primária do par
  e9 gemma→llama abstention.correct_abstention: melhor       # o de outra métrica, pela mesma regra
resumo: Uma frase para a tabela do resumo, com {{marcadores}}.
recomendacao: O que fazer, em uma frase.
ajuste:                     # entra na tabela do .env do resumo
  - {variavel: GENERATION_MODEL, valor: llama3.1:8b, motivo: "por quê"}
figuras: {...}              # ver abaixo
tabelas: {...}
analises: {...}
---
O corpo, em Markdown.
```

O corpo aceita títulos `###`, parágrafos, listas (`-` e `1.`), `**negrito**`, `` `código` ``, tabelas com `|`,
e duas caixas:

```markdown
> **Para a turma**                (caixa didática, ℹ)
> O que a turma deve perceber.

> [!aviso] Uma ressalva           (caixa de aviso, ⚠)
> - um item
```

### Marcadores: os números

Todo número do texto deve vir de um marcador, nunca digitado. Um marcador que não se resolve vira `⟨?⟩` e entra
nos avisos.

| Marcador | Sai | Exemplo |
|---|---|---|
| `{{media R BRAÇO MÉTRICA}}` | a média do braço | `{{media e9 llama c2.faithfulness}}` → 0,78 |
| `{{contagem R BRAÇO MÉTRICA}}` | acertos de uma métrica de 0 ou 1 | → 30 de 30 |
| `{{delta R A→B MÉTRICA}}` | a diferença com o IC de 95% | → +0,37 [+0,20; +0,53] |
| `{{dif R A→B MÉTRICA}}` | só a diferença | → +1,2 s |
| `{{veredito R A→B [MÉTRICA]}}` | o veredito (da métrica primária, ou da dada) | → melhor |
| `{{margem R MÉTRICA}}` ou `{{margem R A→B}}` | a margem de ruído | → 0,048 |
| `{{ruido R GRUPO MÉTRICA}}` | a margem de um grupo A/A do cartão (E0) | `{{ruido e0 juiz c2.faithfulness}}` |
| `{{buraco R BRAÇO}}` | a fração do buraco fechado (E1) | → 132% |
| `{{auc R BRAÇO}}`, `{{limiar R BRAÇO}}` | a AUC do primeiro trecho e o limiar recalibrado (E6) | |
| `{{categoria R BRAÇO CATEGORIA}}` | as respostas numa categoria de citação (E3) | `{{categoria e3 hdr fabricada}}` |
| `{{comparacao ANÁLISE MÉTRICA [veredito]}}` | a diferença entre rodadas (análise `comparacao`) | |
| `{{valor ANÁLISE caminho/no/resultado [pct\|int\|N]}}` | um valor de uma análise | `{{valor nomic off_source_mass pct}}` |

As métricas se escrevem como no placar (`c1.evidence_at_k`, `c2.faithfulness_answered`,
`abstention.with_evidence`...). O `[N]` é o número de casas.

### Diretivas: os blocos

Sozinhas numa linha: `[[cartao R]]` (o cartão como foi escrito, e os braços), `[[vereditos R]]`,
`[[guardas R]]`, `[[figura ID]]`, `[[tabela ID]]`; no resumo, `[[resumo]]` e `[[ajustes]]`.

### Figuras (`figuras:` no cabeçalho)

Toda figura sai com a tabela dos números embaixo (`tabela: false` tira). `legenda` e `como_ler` (uma caixa
didática antes da figura) aceitam marcadores.

| `tipo` | O que desenha | Campos |
|---|---|---|
| `diferencas` | a diferença de cada par, com o IC e a faixa de ruído | `rodada`, `pares`, `metricas`, `rotulos`; ou `linhas` (rodada, par, métrica, rótulo) para juntar rodadas |
| `por_grupo` | a diferença de cada par por classe de pergunta | `rodada`, `pares`, `grupos` (rótulo de cada classe) |
| `medias` | uma barra por braço, um painel por métrica | `rodada`, `bracos`, `metricas`, `rotulos` |
| `margens` | a margem de ruído por métrica e por grupo A/A (E0) | `rodada`, `grupos`, `metricas` |
| `curva` | evidência × tokens de contexto, uma linha por braço (E6) | `rodada`, `bracos`, `marca_braco`, `marca_k`, `marca_texto` |
| `lorenz` | a concentração das fontes (E5) | `analises` (do tipo `e5`), `rotulos` |
| `roc` | a separação das perguntas de fora pela nota do primeiro trecho (E8) | `rodada`, `curvas` (braço, nota, rótulo) |
| `categorias` | para onde aponta a citação de cada resposta (E3) | `rodada`, `bracos`, `rotulos` |
| `custo_qualidade` | um ponto por braço: latência × qualidade (E9) | `rodada`, `bracos`, `y` (métricas), `x` |

### Tabelas (`tabelas:`) e análises (`analises:`)

Tabelas: `medias`, `pares` (como as figuras), `comparacao` (`analise`), `e5` (`analises`), `imas` (`analise`) e
`guardas` (`rodada`).

Análises: `e5` (`rodada`, `braco`: a concentração das fontes na C1 do braço; precisa do manifesto da coleção em
`data/vector_store/manifests/`), `comparacao` (`de`, `para`: "apelido braço", `metricas`: o Δ pareado entre
rodadas diferentes, pelas medidas por pergunta) e `mudancas` (`rodada`, `de`, `para`: quantas respostas mudam).
