# Benchmark de estratégias de RAG — estado da implementação

**Plano:** [plano-benchmark.md](../plano-benchmark.md) · **Branch:** `feat/indices-v2`
(um commit por etapa) · **Última atualização:** 18/10/2026 (E0, E1, E5 e E7 concluídos; seleção do E2 e
do E6 feita; confirmações e E3 rodando)

Este documento registra o que já foi construído do plano, etapa por etapa: o que
mudou, onde, como usar, os testes que rodaram no fim de cada etapa e as decisões
tomadas no caminho.

> **A lista de etapas é provisória.** O plano chegou cortado no meio da tabela
> de prioridade: faltam as seções 9 a 14, inclusive a lista de passos da §14
> ("Fase A", "passo 26"). As etapas abaixo saíram das §5 a §8 e da ordem de
> prioridade (E0, E1, E6, E5, E2). Vão ser conferidas com a §14 quando ela
> chegar. As que dependem das seções que faltam estão marcadas.

---

## 1. Onde está

| Etapa | Entrega | Estado |
|---|---|---|
| **A1** | Versões derivadas (`--register --like`), parâmetros de chunking por versão, sobreposição efetiva no manifesto | **concluída** |
| **A2** | `vector_store.get_all()` (BM25 do E7, diagnóstico de FP1) | **concluída** |
| **A3** | C0 · Acervo: métricas por documento no manifesto, e cálculo para as versões já ingeridas | **concluída** |
| **A4** | C1 · Recuperação: métricas sem juiz e o diagnóstico estendido | **concluída** |
| **A5** | Goldens v2: esquema, validador, sorteio do split por documento, migração dos casos existentes como `rascunho` | **concluída** |
| **A6** | C2/C4 no `generate_answers.py`: proveniência do braço, custo por resposta, ordem embaralhada com semente, abstenção | **concluída** |
| **A7** | Goldens v2: núcleo de ≥ 60 perguntas em ≥ 20 PDFs e ≥ 30 de fora, como `rascunho` para a revisão humana | **concluída** (falta a revisão humana) |
| **B0** | Runner de experimentos e cartões em YAML | **concluída** (formato provisório até a §9) |
| **B1** | E0 — rodada A/A | **concluída** (margem de ruído em `tests/benchmark/out/E0_20261018T082340/margem.json`) |
| **B2** | E1 — idioma (`pdf-v1-bgem3`, tradução da pergunta) | **concluída**: o bge-m3 vence (evidência@4 de 0,13 para 0,66 em português; fidelidade +0,15), ao preço da abstenção correta (0,83 → 0,63) |
| **B3** | E6 — parâmetros de recuperação (C1) | **seleção feita** em português (nomic e bge-m3) e em inglês: o limiar recalibrado separa quando a recuperação funciona; o MMR nunca ajudou; a confirmação (`top_k` 8 e o limiar, no bge-m3) vem a seguir |
| **B4** | E5 — concentração das fontes | **concluída**: o cruzamento com a C2 (as respostas apoiadas nos ímãs são piores), o H5a (os ímãs somem com o bge-m3) e o H5b (a bibliografia não é ímã); falta a demanda real das aulas |
| **B5** | E2 — formato: PDF × Markdown | **concluída** (só a C1, em inglês): a conversão do Docling perde 8 de 39 perguntas para o PDF; o chunker por seção, com orçamento igual, só empata com o PDF |
| **B6** | E3 — cabeçalho contextual e atribuição errada | **construído** (cabeçalho no embedding ou no trecho, títulos em rascunho, regra de citação v2, atribuição errada, cartão); roda depois do E1 |
| **B7** | E7 — busca híbrida (BM25 + RRF) | **concluída**: +0,20 de evidência@4 em inglês, sem a interação com a sigla; com o bge-m3 em português, **piora** (−0,11): o BM25 traz o ímã de idioma de volta |
| **B8** | E8 — reranker (o LLM via Ollama) | **concluída**: a precisão@4 sobe em português (+0,08), a nota separa as perguntas de fora pior que o cosseno, e a busca fica 40× mais lenta (~11 s): não recomendado |
| B9 | E4, E9 | aguardam (o E4 precisa dos distratores e de perguntas revisadas por degrau; o E9 é opcional) |
| C | Placar (regra de veredito, Holm, interação) e as seções novas do relatório | aguarda a §11 |

### Portões, estado atual

Rodados no fim de cada etapa. Nenhum precisa de Ollama, GPU ou rede.

```
tests/api       (pytest, backend/.venv)                     VERDE — 78 testes (35 antes do benchmark)
tests/benchmark (pytest, backend/.venv; entra no CI)        VERDE — 67 testes
tests/reports   (pytest, tests/reports/.venv)               VERDE — 44 testes
tests/jeveval   (pytest -m "not live", tests/jeveval/.venv) VERDE — 63 testes
goldens v2      (validar --indice pdf-v1 --evidencia)       VERDE — 103 goldens, 0 erro, 0 aviso
```

O `pytest` da raiz (o do CI) roda `tests/api` e `tests/benchmark` juntos: 145
testes.

---

## 2. Decisões tomadas

**Nomes das versões.** O plano (§8.2) previa os braços do E2 como `md2-p4llm`,
`md2-dl` e `md2-dl-sec`, e dizia que esses nomes substituíam o `md-v2` que o
README anunciava. Depois do commit em que o plano foi escrito (`fcf039a`), o
`md-v2` e o `pdf-v2` passaram a existir como versões de verdade (os 61 artigos
sincronizados), e `md2-*` passaria a parecer um pedaço do `md-v2`. Ficou assim:

- versão de acervo continua com `-vN` (`pdf-v1`, `md-v2`, `md-v3`...), como no
  README;
- braço do benchmark é a versão-base mais o que mudou (`pdf-v1-bgem3`,
  `pdf-v1-cont`, `pdf-hdr`, `pdf-v1-noref`, `esc-dif-030`), como no plano;
- os braços do E2 viram `md-p4llm`, `md-dl` e `md-dl-sec` (e `md-dl-sec-hdr` no
  E3). Todos cabem no limite de ~13 caracteres da aba da planilha.

**O que mudou no repositório depois de `fcf039a`** e muda o que o plano diz:

- A identidade do documento pelo nome sem extensão (§6, C1) já existia no
  `tests/retrieval_diagnostic.py` e no `demo_e2e.source_rank`. Faltava nas
  contas do E5, que agora a usam (`tests/benchmark/e5.py`, por `c1.doc_stem`).
- As duas pastas de acervo têm 61 arquivos, e cada versão fixa a lista dos seus
  documentos (`backend/app/rag/index_versions/`). Todo braço derivado do
  `pdf-v1` ou do `md-v1` precisa da lista da base: é a etapa A1.
- A conversão pymupdf4llm dos 39 PDFs (degrau 2 do E2) já existe: são os 39
  arquivos que o `md-v2` acrescentou a `data/new_papers_marks`. Só a conversão
  do Docling é nova, numa pasta própria (os nomes colidiriam).
- Toda entrada nova em `RAG_INDICES` precisa de `origins`: o `/health` lê o
  campo sem valor padrão.

**Achado que pede decisão: `localhost` custa 2 s por chamada ao Ollama nesta
máquina.** O Ollama escuta só em `127.0.0.1:11434` (IPv4). Com
`OLLAMA_BASE_URL=http://localhost:11434`, que é o valor do `.env` e o padrão do
`config.py`, cada conexão tenta primeiro o IPv6 (`::1`) e espera ~2 s a recusa.
E o backend abre um cliente novo por chamada.

| | Embedding (média de 5) | Diagnóstico, 4 índices × 10 perguntas × 2 línguas |
|---|---|---|
| `http://localhost:11434` | 2.289 ms | 6 min 12 s |
| `http://127.0.0.1:11434` | 250 ms | 44 s, com as mesmas listas ranqueadas |

Medi só o embedding. A geração abre o cliente do mesmo jeito, no mesmo
endereço, e deve pagar o mesmo atraso: no chat, são pelo menos duas conexões
por pergunta. Isso vale para a latência medida até hoje e para o custo do C4:
o braço de tradução do E1 pagaria uma conexão a mais. As rodadas do benchmark usam
`OLLAMA_BASE_URL=http://127.0.0.1:11434` no ambiente, sem mexer no `.env`. A
correção (trocar o `.env`, o padrão do `config.py` e do `.env.example`, ou
reaproveitar a conexão) muda o assistente e ficou para você decidir.

**Achado que pede decisão: a busca vetorial muda de um processo para outro.**
O Chroma (1.5.9) só grava em disco o grafo HNSW de uma coleção depois de 1.000
inserções (`sync_threshold`). As coleções de 894 chunks (`pdf-v1`, `pdf-v1-aa`,
`pdf-v1-bgem3`) e a de 857 (`md-v1`) nunca chegam lá. As de 1.632 e 1.977
gravam só os primeiros ~1.000. Cada processo reconstrói o resto do grafo a
partir do log, com inserção em paralelo, e o grafo sai diferente a cada vez. A
busca é aproximada, e o que ela perde muda junto. Medido com 60 consultas
fixas, sem Ollama, em dois processos:

| Coleção | Listas top-4 que mudam entre os 2 processos | recall@4 contra a busca exata (processo 1 · 2) |
|---|---|---|
| `pdf-v1` | 1 de 60 | 0,996 · 1,000 |
| `pdf-v1-aa` | 2 de 60 | 0,988 · 0,996 |

Na C1 do E0, isso apareceu numa pergunta. A c28 teve evidência no A5 e não no
A1, porque a busca do A1 perdeu um trecho de similaridade 0,6666, maior que a
do primeiro trecho que ela devolveu (0,6575). Um diagnóstico anterior no mesmo
`pdf-v1` tinha achado esse trecho.

- **O ruído da C1 não é ~0**, como o plano supunha: é ~1 pergunta em 70 na
  evidência@4. A margem do E0 absorve isso.
- **O A5 não mede só a reingestão:** qualquer processo novo varia.

As opções mudam a recuperação do assistente; ficam para você decidir:

1. **Busca exata nas coleções pequenas:** o cosseno contra todos os vetores,
   que até alguns milhares de chunks custa pouco. É determinística e não
   perde trecho. Um campo opcional, ligado só nos cartões do benchmark,
   deixaria o assistente como está.
2. **Gravar o grafo:** criar as coleções com `sync_threshold` baixo. Fica
   igual entre processos, mas a perda de recall continua (sempre a mesma). Pede
   reingerir todas as versões.
3. **Deixar como está** e usar a margem do E0. É o que as rodadas atuais
   fazem.

---

## 3. Etapas concluídas

### A1 — Versões derivadas e parâmetros de chunking por versão (17/10)

**Por quê.** Quase todo braço do plano é uma versão derivada: os mesmos
documentos do `pdf-v1` e uma mudança só (`pdf-v1-bgem3`, `pdf-v1-cont`,
`pdf-hdr`, `pdf-v1-noref`, os chunks de 400 e 1200 do E6). O `--register`
grava a lista com o que está na pasta, e `data/source_pdfs` tem hoje 61 PDFs,
22 deles diagramados a partir do Markdown. Registrado assim, um `pdf-v1-bgem3`
indexaria 61 artigos em silêncio, e a comparação com o `pdf-v1` mediria o
tamanho e o formato do acervo junto com o embedding.

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/ingestion.py` | `register_documents(..., like=)` copia a lista da versão-base, na mesma ordem, e recusa se a pasta não tem algum documento dela ou se ele mudou (SHA-256). O cabeçalho da lista nova registra a base. `effective_overlap()` sai do `chunk_text` e o manifesto grava `chunk_overlap_tokens_effective` ao lado do pedido |
| `backend/app/config.py` | `INDEX_SETTINGS_KEYS` (`chunk_size_tokens`, `chunk_overlap_tokens`): chaves opcionais da versão, copiadas pelo validador por cima do `.env`, como o modelo de embedding |
| `scripts/ingest_documents.py` | `--like BASE`, só com `--register` |
| `README.md` | parágrafo sobre versão derivada, na seção das versões do índice |
| `tests/api/test_rag_indices.py` | 5 testes novos |

**Como usar.**

```powershell
# 1. a entrada em backend/app/config.py (RAG_INDICES), com coleção própria e origins
# 2. a lista, copiada da base:
backend\.venv\Scripts\python scripts\ingest_documents.py --index pdf-v1-bgem3 --register --like pdf-v1
# 3. a ingestão, como qualquer versão:
backend\.venv\Scripts\python scripts\ingest_documents.py --index pdf-v1-bgem3
```

**Testes novos.** A lista copiada é igual à da base, com a mesma ordem, mesmo
com outro arquivo na pasta. A cópia é recusada se um documento da base sumiu ou
mudou, e também se a base não tem lista. Uma versão com `chunk_size_tokens`
ganha do `CHUNK_SIZE_TOKENS` do ambiente, e uma versão sem a chave segue o
ambiente, como antes. O manifesto grava 150 pedido e 25 efetivo para chunk de
100. Toda versão registrada tem os campos que o backend lê, as coleções não se
repetem, e nenhuma pede sobreposição que seria trocada sem aviso.

**O que ficou para depois.** A chave `chunker` (`token` ou `section`) que o
plano lista com as duas acima entra com o chunker por seção, no E2. Antes
disso, ela seria um valor que não faz nada.

### A2 — Leitura da coleção inteira (17/10)

`vector_store.get_all(include_embeddings=False, batch_size=5000)` devolve todos
os chunks da coleção ativa: ids, textos e metadados, e os vetores se pedidos.
Lê em lotes com `limit`/`offset`, porque o degrau de mil artigos do E4 passa de
30 mil chunks. É o que o BM25 do E7 indexa e o que dá, no E5, a fração de
chunks de cada documento no índice (o "esperado ao acaso" do ímã).

Testes novos em `tests/api/test_vector_store.py` (2), com uma coleção falsa:
paginação, ordem, vetores convertidos em lista e coleção vazia. Conferido
também no índice real do `pdf-v1`: 894 chunks, igual ao `count()` do Chroma,
dos 39 documentos.

### A3 — C0 · Acervo no manifesto (17/10)

**O que mede.** As métricas da §6 (C0), por documento, sobre o mesmo texto que
o chunker recebe. Ficam no manifesto ao lado de `file` e `sha256`, e um resumo
da versão fica em `c0`, no topo:

| Campo | Definição |
|---|---|
| `pages_total`, `pages_with_text` | PDF; a ingestão descarta página sem texto, sem aviso |
| `tokens`, `chunk_tokens` | cl100k do texto inteiro; média, p10 e p90 dos tokens por chunk |
| `glued_word_rate` | fração das letras em tokens alfabéticos de 18+ letras fora do dicionário |
| `table_rows`, `headings` | Markdown cru: linhas de tabela (sem a separadora) e títulos `#`. No PDF, `None` e 0 |
| `references_found`, `references_share` | o bloco de referências foi achado? que fração dos tokens ele tem? |

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/corpus_quality.py` (novo) | as métricas, puras, sem dependência nova; `C0_VERSION = "c0-v1"` |
| `backend/app/rag/c0_long_words.txt` (novo) | 2.774 palavras de 18+ letras (248 em inglês, 2.526 em português), das listas do pyspellchecker 0.8.3 (MIT) |
| `backend/app/rag/ingestion.py` | `read_document()` e `chunk_document()`: a leitura sai do chunker, e o C0 usa a mesma extração, sem ler o PDF duas vezes. `build_chunk_records()` mantém a assinatura (o `verify_mutants.py` do estudo de mutação usa). `markdown_structure()`. `measure_corpus_quality()` |
| `scripts/ingest_documents.py` | `--c0`: calcula o C0 de uma versão já ingerida e grava no manifesto, sem reingerir |
| `tests/api/test_corpus_quality.py` (novo) | 9 testes |

**O bloco de referências.** Começa no último título "References"/"Referências"
seguido de linhas com cara de referência (ano, "et al.", "Proceedings", doi,
lista de autores, numeração `[12]`). Termina no que vier primeiro:

- um título de apêndice;
- no Markdown, qualquer outro título;
- no PDF, um título de letra ("A Experimental Setup", ou "A" numa linha e o
  título na seguinte) seguido de linhas sem cara de referência;
- a densidade de linhas com marca abaixo de 15% numa janela de 25 linhas.

A regra foi conferida à mão nos 100 documentos (os 39 PDFs, as 39 conversões
do pymupdf4llm e os 22 Markdown originais), olhando onde cada bloco acaba. A
primeira versão cortava cedo as referências em duas colunas, em que os autores
quebram em linhas sem ano (RAGAS, ARES, RAGChecker). Ela ganhou as marcas de
lista de autores e o título de apêndice partido em duas linhas.

**Como usar.**

```powershell
backend\.venv\Scripts\python scripts\ingest_documents.py --index pdf-v1 --c0
```

A ingestão grava o C0 sozinha. O `--c0` serve para as versões já ingeridas.
Ele confere o SHA-256 de cada arquivo e o número de chunks contra o
manifesto, e deixa de fora, em `c0.not_measured`, o que não bate, porque o
texto desse documento não é o que está no índice.

**Resultado nas quatro versões** (17/10, manifestos copiados antes):

| Versão | Docs | Tokens | Chunk: média / p10 | Grude: média · docs > 5% | Bibliografia · docs sem bloco |
|---|---|---|---|---|---|
| `pdf-v1` | 39 | 524.705 | 647 / 263 | 0,1% · 0 | 20,9% · 1 |
| `md-v1` | 22 | 576.171 | 789 / 800 | 39,1% · 18 | 20,4% · 2 |
| `pdf-v2` | 61 | 1.116.033 | 623 / 268 | 14,1% · 18 | 15,5% · 13 |
| `md-v2` | 61 | 1.095.445 | 787 / 800 | 14,2% · 18 | 20,9% · 3 |

O que a tabela já diz:

- **O corte por página encolhe o chunk do PDF.** Um em cada dez chunks do
  `pdf-v1` tem 263 tokens ou menos, contra os 800 pedidos. No Markdown, o
  p10 é 800. É a confusão que o degrau 1b do E2 (`pdf-v1-cont`) isola.
- **O grude está todo nos 22 Markdown originais.** 18 deles passam de 5% das
  letras, e a média é 39%. Nas 39 conversões do pymupdf4llm, o máximo é 1,15%,
  e nos 39 PDFs, 1,19%. O README dizia "16 de 22, de 3% a 25% das palavras",
  em outra unidade e com método não registrado.
- **Um quinto do índice é bibliografia.** São 20,9% dos tokens no `pdf-v1`
  (de 6,5% a 39,6% por artigo), o que dá peso à hipótese H5b do E5 (trechos de
  referência como ímã).
- **O `2025_01_RETORCH_ICTSS24_ResearchGate.pdf` tem uma página só**, a de
  resumo do ResearchGate, e por isso um chunk só. Não serve de documento-fonte
  nos goldens v2.
- **Os 22 PDFs gerados perderam o título "REFERENCES".** O fpdf2 reflui o
  texto, e o título cai no meio da linha ("...IEEE REFERENCES"). Daí os 13
  documentos sem bloco no `pdf-v2` e os 15,5% contra 20,9% do `md-v2`.

**Limitações.**

- O `glued_word_rate` ainda não foi conferido contra uma amostra rotulada à
  mão, como o plano pede (200 linhas de 6 arquivos). Sobrenome e identificador
  de código longos contam como grude: é o piso de ~1% dos PDFs.
- Nos Markdown originais com colunas intercaladas, o bloco de referências
  mistura texto do artigo: o `TestingLLM.md` sai com 60%. Dois dos 22 não têm
  título de referências nenhum.

### A4 — C1 · Recuperação sem juiz (18/10)

**O que mede.** As métricas da §6 (C1), por pergunta e braço, em
`tests/benchmark/c1.py` (só biblioteca padrão):

- Hit@1, Hit@k e Hit@10, o rank de sempre e o MRR@10 pelo rank do documento
  entre os distintos;
- precisão@k: fração dos trechos que são da fonte ou que, sozinhos, casam um
  fragmento da evidência;
- cobertura@k e evidência@k: fragmentos do `context` casados por 25-gramas,
  τ = 60%, sobre a união dos k trechos, com o texto normalizado;
- similaridade do primeiro trecho, sem limiar, e o bloqueio pelo limiar do
  braço.

Cada média sai com IC de 95% por bootstrap **por documento**, porque as
perguntas de um mesmo documento erram juntas (§7). A abstenção pelo limiar sai
como taxa nas perguntas de fora, falso bloqueio nas de dentro, e AUC da
similaridade com IC.

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `tests/benchmark/c1.py` (novo) | as métricas e os agregados; `C1_VERSION = "c1-v1"` |
| `tests/retrieval_diagnostic.py` | grava, por pergunta e braço, a lista ranqueada que o gerador recebe (`top_k` e MMR do braço) e a de k = 10, cada uma da sua própria busca. Também grava as métricas e o resumo em `c1`. `--set CHAVE=VALOR` aplica a configuração de consulta de um braço (MMR, limiar, `top_k`) sem mexer no ambiente. Golden com `source_document: null` vai para `out_rows`; pergunta cujo documento a versão não tem conta como de fora nela |
| `pytest.ini` | `tests/benchmark` entra no `pytest` da raiz, e no CI |
| `tests/benchmark/test_c1.py`, `test_retrieval_diagnostic.py`, `conftest.py` (novos) | 13 testes, herméticos: recuperação falsa e configuração isolada do `.env` |

**Compatível com o que já existia.** `rows[].ranks` continua com o mesmo
formato, que o relatório lê, e a linha `índice: PT x/10 · EN y/10`, que o
`compare_indices.py` resume, continua igual. Golden sem a chave
`source_document`, como os 10 de `dataset.json`, continua de fora.

**Conferido nos índices reais** com os 10 goldens dos Markdown
(`dataset_md.json`). Os ranks repetem os números do README: `md-v1` PT 2/10 e
EN 10/10, `pdf-v2` EN 5/10, `md-v2` EN 7/10. Em inglês:

| Índice | Hit@4 | MRR@10 | Precisão@4 | Cobertura@4 | Evidência@4 | Cobertura@10 |
|---|---|---|---|---|---|---|
| `md-v1` | 1,00 [1,00; 1,00] | 0,90 | 0,57 [0,42; 0,75] | 0,18 [0,03; 0,41] | 0,10 [0,00; 0,33] | 0,42 |
| `pdf-v2` | 0,50 [0,18; 0,89] | 0,47 | 0,28 | 0,18 | 0,10 | 0,32 |
| `md-v2` | 0,70 [0,44; 0,90] | 0,59 | 0,38 | 0,18 | 0,10 | 0,28 |

**O documento certo não basta.** No `md-v1`, em inglês, o documento certo vem
entre os 4 trechos em 10 de 10 perguntas, mas só 18% dos fragmentos da
evidência chegam ao gerador, e só uma pergunta recebe a evidência inteira. O
caso dos "sete pontos de falha" mostra o porquê. O artigo vem em 1º, mas pelos
chunks `c2`, `c4` e `c0`. As descrições de FP1 a FP7 estão nos chunks `c5` e
`c6`, e nenhum dos dois vem. Conferido à mão: contra todos os chunks do
artigo, os 7 fragmentos casam com 100%. O Hit@k do diagnóstico antigo media
"o artigo veio", não "a resposta veio".

**Cuidados com estes números.**

- São 10 perguntas de 9 documentos: os ICs vão de 0 a 0,33 na evidência.
- O `context` de 5 desses goldens traz notas editoriais ("No Markdown, este
  trecho está com as palavras grudadas..."), que não estão na fonte e puxam a
  cobertura para baixo. Os goldens v2 (A5) levam as notas para `observacao`.
- O τ = 60% ainda não foi calibrado à mão com 20 pares, como o plano pede.
  Nos Markdown com colunas intercaladas, há fragmentos com 55% a 57% dos
  gramas presentes, logo abaixo do τ.

### A5 — Goldens v2: esquema, validador, split e migração (18/10)

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `tests/benchmark/goldens.py` (novo) | o esquema do §7, o validador e o sorteio do split, com uma CLI (`validar`, `sortear-split`) |
| `tests/benchmark/goldens/v2.json` (novo) | os 30 casos do estudo de mutação migrados, todos `rascunho` |
| `tests/retrieval_diagnostic.py` | pergunta `ambiguo` fica fora da C1: não tem documento nem é de fora, e o certo é pedir esclarecimento |
| `tests/benchmark/test_goldens.py` (novo) | 8 testes, entre eles um que valida o `v2.json` do repositório no CI |

**O validador** confere:

- os campos do §7, sem campo a mais nem a menos, e os valores de cada
  enumeração;
- `name` único no arquivo e entre os outros conjuntos (`dataset.json`,
  `dataset_md.json`), porque os juízes indexam as respostas por nome;
- nota editorial no `context` ("No Markdown...", "Nota:", "espaços
  restaurados"), que vai para `observacao`;
- `source_document` sem extensão, e de uma versão do índice com `--indice`;
- a coerência entre documento, classe, `out_stratum`, `content_type` e
  evidência;
- as perguntas do mesmo documento no mesmo split.

Com `--evidencia`, cada fragmento tem de casar com o documento inteiro, como a
versão o extrai. Abaixo de τ é erro, porque nem com o documento todo na mão
contaria como evidência. Abaixo de 95% é aviso, porque alguma palavra não é a
da fonte. O primeiro fragmento também tem de estar na `evidence_page`.

**O split** é sorteado por documento e estratificado pelo `content_type` mais
comum do documento. Pergunta de fora é sorteada sozinha, estratificada pelo
`out_stratum`. Sem `--refazer`, só sorteia quem não tem split. Com goldens
`revisado` no arquivo, o `--refazer` é recusado, porque o split deles já pode
ter sido usado.

**A migração.** A pergunta e a resposta esperada saem do estudo palavra por
palavra. O resto é desta migração:

- a pergunta em inglês;
- os fragmentos, reencontrados no PDF original pelas âncoras das assertivas do
  estudo (`assertions.yaml`), e não copiados do corpus re-renderizado, cujas
  páginas são outras (a dica "pág. 12" do c14 é a pág. 23 do original);
- a página original;
- os rótulos (`content_type`, `has_acronym`, `names_article`, `out_stratum`).

Nos 4 casos de rastreabilidade, a citação da resposta esperada passou a
apontar o arquivo e a página originais. O `observacao` de cada golden diz de
qual caso ele veio e o que mudou.

**O que entrou.** 19 perguntas de dentro, de 6 documentos (8 delas do mesmo
artigo), 8 de fora (5 `perto`, 2 `escopo`, 1 `longe`) e 3 ambíguas.
`validar --indice pdf-v1 --evidencia`: nenhum erro e nenhum aviso. Todo
fragmento está pelo menos 95% na fonte, e todo primeiro fragmento está na
página declarada.

**Split.** Com 6 documentos o sorteio saía desequilibrado. Foi sorteado de
novo no fim da A7, com o núcleo completo (ver lá).

**Cotas do §7.** Ficaram longe com os casos migrados (6 documentos, 19 de
dentro, 3 de tabela, 3 sem sigla, 8 de fora). A A7 completou.

**Primeira C1 do `pdf-v1`.** O diagnóstico antigo não rodava nos índices de
PDF, porque os goldens dos PDFs não têm `source_document`. Com estes 19 de
dentro e 8 de fora:

| Pergunta em | Hit@4 | MRR@10 | Precisão@4 | Evidência@4 | Cobertura@10 | AUC (primeiro trecho) |
|---|---|---|---|---|---|---|
| português | 0,53 [0,22; 0,68] | 0,45 | 0,32 | 0,32 [0,00; 0,52] | 0,34 | 0,81 [0,61; 0,96] |
| inglês | 0,89 [0,82; 1,00] | 0,80 | 0,67 | 0,42 [0,00; 0,69] | 0,71 | 0,95 [0,85; 1,00] |

- **O idioma pesa também no `pdf-v1`.** O Hit@4 cai de 17/19 em inglês para
  10/19 em português.
- **O limiar de 0,428 do estudo de mutação não barra nada aqui.** Nenhuma das
  8 perguntas de fora fica abaixo dele. No `pdf-v1`, até a receita de bolo tem
  similaridade 0,594 em português. As de fora vão de 0,59 a 0,70 e as de dentro
  de 0,63 a 0,78, de modo que as faixas se sobrepõem. É o que o E6 vai
  recalibrar, com a regra de seleção e confirmação do plano.
- Com 6 documentos, os ICs ainda são largos.

### A6 — C2/C4 no `generate_answers.py` (18/10)

**O que cada resposta grava agora**, além do que já gravava:

| Campo | De onde vem | Para quê |
|---|---|---|
| `retrieval_ms`, `generation_ms` | `retrieve()` e `generate_answer()` (antes descartados) | C4: latência |
| `prompt_tokens`, `completion_tokens` | `generate_answer()` | C4: custo; tokens de raciocínio contam aqui |
| `truncated` | `prompt_tokens >= prompt_token_limit() - 2`, a regra do `sut.py` | o Ollama corta o início do prompt sem avisar |
| `source_documents`, `chunk_ids`, `similarity_scores` | os trechos que chegaram | E5 (concentração), sem tirar o "(pág. N)" de `sources` |
| `abstained` | `tests/benchmark/abstention.py` | C2: abstenção correta e indevida |
| `position` | a ordem em que a pergunta foi feita | com `--shuffle-seed`, a ordem embaralhada |

**No topo do arquivo:**

- `arm`: a configuração efetiva do braço, gerada por
  `tests/benchmark/arm.py`. Traz o índice, a coleção, o modelo de embedding e
  o manifesto que respondeu (data, chunks, parâmetros, versão do C0), a
  recuperação (top_k, MMR, limiar), a geração (provider, modelo, temperatura,
  semente, `num_ctx` resolvido, `num_predict`, `think`), o prompt e as
  sobrescritas aplicadas;
- `shuffle_seed`;
- `goldens_sha256`, o hash de cada arquivo de goldens.

O diagnóstico de recuperação grava o mesmo `arm` em `c1.arms`.

**Opções novas:**

- `--set CHAVE=VALOR`: sobrescritas de `Settings`, aplicadas com
  `set_settings_overrides`, sem mexer no `.env` nem no ambiente;
- `--shuffle-seed N`: embaralha a ordem das perguntas. O cache de prefixo do
  Ollama faz a resposta depender da pergunta anterior;
- `--local`: força o provider local, porque o remoto ignora temperatura,
  semente, `num_ctx` e `think`.

O `--index` agora também vai pelas sobrescritas. A CLI de sempre não mudou, e
o `compare_indices.py` e o `run_all.py` chamam do mesmo jeito.

**A recusa.** A frase de abstenção é comparada ignorando acento, caixa,
pontuação e espaço. "Nao encontrei essa informacao..." passa a contar como
recusa. A lista de variantes (`VARIANTS`) fica vazia até o E0, porque o plano
manda revisá-la à mão nas respostas de verdade. O relatório ainda usa a busca
antiga (`report_data.Case.abstained`) e passa a usar esta depois do E0. Até
lá, as duas podem divergir numa recusa sem acento.

**Testes novos** em `tests/benchmark/test_generate_answers.py` (5), com
recuperação e geração falsas, e mais 1 no diagnóstico. Cobrem a recusa sem
acento, caixa ou pontuação e a frase que não é recusa; os campos de custo e de
documentos; a truncagem no limite (4.096 − 512); a falha que vira registro
depois de duas tentativas; a ordem embaralhada reproduzível; o braço com as
sobrescritas convertidas; e a pergunta ambígua fora da C1.

**Fumaça com o modelo de verdade.** Três perguntas dos goldens v2 no `pdf-v1`,
com a geração fixada pelo §8.2: `gemma3:4b`, temperatura 0, semente 42,
`num_ctx=auto`, `num_predict=1024`, `think=false` e provider local.

- O `gemma3:4b` aceita `think=false`, sem erro, mesmo sem ter modo de
  raciocínio. O `num_ctx=auto` resolve para 6.144.
- Nenhuma truncagem. Os prompts tiveram de 2.226 a 3.716 tokens.
- A pergunta c01, em português sem acento, recuperou só os artigos em
  português do acervo, e o modelo se absteve, com `abstained = true`. É o ímã
  de idioma do E1, ao vivo.
- Na c03, o CV veio certo (3,79%), mas a "precisão média" veio errada: 0,935 ±
  0,0305 é o F1 da mesma linha da tabela, e a precisão é 0,9410 ± 0,0357. É o
  tipo de erro de leitura de tabela que o E2 testa.

### A7 — Goldens v2: núcleo completo e perguntas de fora (18/10)

**O que entrou.** São 51 perguntas de dentro novas, escritas a partir dos PDFs
originais: 3 em cada um de 15 artigos novos, e mais 2 em cada um dos três
artigos que tinham ficado com uma só (MeTMaP, RAGChecker, QAQuest). Entraram
também 22 perguntas de fora novas. Todas são `rascunho`.

Em cada artigo, uma pergunta sai de uma tabela, uma não depende de sigla nem
de nome próprio, e uma sai do texto ou de uma seção. As classes variam entre
fato direto, fato distribuído, filtro condicional e rastreabilidade. A
rastreabilidade tem a citação esperada no formato da `CITATION_RULE`.

**As perguntas de fora** só usam termos ausentes dos 122 arquivos das duas
pastas, conferido termo a termo, porque os goldens v2 valem também para o
`pdf-v2` e o `md-v2`. Na conferência, três casos migrados mudaram:

- **c20:** o FALAAI, que o estudo de mutação tratava como referência
  inventada, existe. É o sistema descrito no `Testing LLM-Based Systems...`,
  um dos 22 Markdown ("FALAAI/Cagece, 53 casos, 96,22% aprovação"). Nas
  versões com os Markdown, a pergunta deixaria de ser de fora. O estudo virou
  QualiBot, ausente dos 122 arquivos, e o motivo está no `observacao`.
- **c18 e c19:** o Appium e o JaCoCo aparecem em artigos do `pdf-v1`, sem o
  fato perguntado. Continuam de fora (tema presente, resposta ausente), com
  nota.
- Termos descartados porque aparecem no acervo: Cypress, Stryker, SonarQube,
  TruLens, LangSmith, HumanEval, Robot Framework, Mockito.

**Composição final** (103 goldens; `validar --indice pdf-v1 --evidencia`, sem
erro e sem aviso):

| Cota do §7 | Pedido | Agora |
|---|---|---|
| Documentos do núcleo, com 3 perguntas cada | ≥ 20 | 21, todos com 3 ou mais |
| Perguntas de dentro | ≥ 60 | 70 |
| `content_type = tabela` | ≥ 15 | 31 |
| `has_acronym = false` | ≥ 15 | 34 |
| `names_article = false` | ≥ 15 | 63 |
| Perguntas de fora | ≥ 30 | 30 (16 `perto`, 6 `longe`, 8 `escopo`) |

Há ainda 3 perguntas ambíguas, que ficam fora da C1.

**O split** foi sorteado de novo, com `--refazer`, permitido porque tudo
ainda é rascunho (semente 20261018):

| | `selecao` | `confirmacao` |
|---|---|---|
| Documentos | 10 | 11 |
| Perguntas de dentro | 31 | 39 |
| De tabela | 15 | 16 |
| De fora | 15 | 15 |

A regra do sorteio ganhou um ajuste: a alternância continua de um estrato para
o outro. Antes, estrato com número ímpar de documentos dava a sobra sempre a
`selecao`, e o primeiro sorteio completo saiu 12 contra 9 documentos.

**Convenção de rótulo.** `has_acronym` marca a pergunta que traz uma sigla ou
um nome próprio distintivo (RAGAS, Qwen, tullibee, MeTMaP), daqueles em que a
busca por palavra-chave do E7 se agarra. Siglas genéricas no acervo inteiro
(RAG, LLM) não contam.

**Primeira C1 do `pdf-v1` com os goldens completos** (só embeddings, 1 min 46
s; 70 de dentro, de 21 documentos, e 30 de fora):

| Pergunta em | Hit@4 | Hit@10 | MRR@10 | Precisão@4 | Evidência@4 | Cobertura@10 | AUC |
|---|---|---|---|---|---|---|---|
| português | 0,23 [0,09; 0,38] | 0,29 | 0,23 | 0,16 | 0,13 [0,03; 0,24] | 0,18 | 0,79 [0,69; 0,89] |
| inglês | 0,80 [0,69; 0,89] | 0,90 | 0,79 | 0,61 | 0,53 [0,40; 0,66] | 0,69 | 0,97 [0,94; 0,99] |

- **O buraco do idioma é o maior efeito medido até aqui.** Em português, o
  documento certo vem entre os 4 trechos em 16 de 70 perguntas; em inglês, em
  56 de 70. Os ICs não se encostam. É a régua do E1: "teto" em inglês e
  baseline em português, nos mesmos 70.
- **Em inglês, a similaridade do primeiro trecho separa bem as perguntas de
  dentro das de fora** (AUC 0,97). Um limiar pode funcionar em inglês; em
  português (0,79), muito menos. O E6 decide.
- São números exploratórios: os goldens ainda são rascunho.

**O que falta, e é humano:**

- revisar as 103 perguntas e passar para `revisado` as aprovadas, porque só
  elas entram nos vereditos;
- a rotulagem dupla do `content_type` em 20 perguntas, com o kappa no
  relatório;
- calibrar o τ à mão com 20 pares;
- validar o `glued_word_rate` com 200 linhas rotuladas.

Os candidatos foram escritos por um LLM (eu), como o plano prevê, e conferidos
contra a fonte pelo validador. A pergunta, a resposta esperada e os rótulos
são julgamento meu e pedem a revisão.

### B0 — Runner de experimentos e cartões em YAML (18/10)

> **Formato provisório.** A §9 do plano, que especifica o runner e o YAML, não
> chegou. O que está aqui segue as §4, §5 e §8 e o `tests/compare_indices.py`,
> que o plano dá como modelo. Quando a §9 chegar, o formato se ajusta a ela.

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `tests/benchmark/run_experiment.py` (novo) | o runner: lê o cartão, mostra no terminal, roda o funil (índices → C1 → respostas → RAGAS → placar) e grava tudo numa pasta por rodada. Retomável |
| `tests/benchmark/placar.py` (novo) | Δ pareado com IC 95% por bootstrap de documentos (2.000 reamostras, semente fixa), McNemar exato, Holm, a margem de ruído (P95 de \|Δ\|) e o veredito provisório |
| `tests/benchmark/cartoes/E0.yaml` (novo) | o primeiro cartão |
| `backend/app/config.py`, `index_versions/pdf-v1-aa.sha256` | a versão `pdf-v1-aa`: a lista do `pdf-v1` numa coleção nova, para o ruído da reingestão (A5 do E0). Ingerida: 894 chunks, em 32 s |
| `tests/retrieval_diagnostic.py` | grava a latência de cada busca (`retrieval_ms`) e aceita `--langs` |
| `.gitignore` | as rodadas em `tests/benchmark/out/` ficam fora do git (só o `.gitkeep` entra) |
| `tests/benchmark/test_placar.py`, `test_run_experiment.py` (novos) | 4 + 6 testes, com saídas falsas; um deles valida os cartões do repositório |

**O cartão.** Um YAML por experimento, em `tests/benchmark/cartoes/`:

| Chave | O que é |
|---|---|
| `id`, `titulo`, `pergunta`, `hipotese`, `didatica` | obrigatórias; o runner mostra no terminal, e o resumo traz o cartão inteiro |
| `goldens` | o arquivo de goldens (v2) |
| `fixos` | sobrescritas de `Settings` para todos os braços: no E0 e no E1, a geração do §8.2 |
| `juiz` | `modelo` e `num_ctx` do RAGAS |
| `bracos` | cada braço: `indice`, `lingua` (pt ou en), `camadas` (c1, c2), `ordem` (semente que embaralha as perguntas), `set` (sobrescritas só dele) e `papel` (baseline, candidato, ruido, referencia). Ou `rejulgar: <braço>` com `ordem_juiz`: julga de novo as respostas de outro braço |
| `pares` | os pares comparados, Δ = segundo − primeiro. Sem a chave, o baseline contra cada braço |
| `margem` | só no E0: os grupos de pares A/A (`gerador_juiz`, `juiz`, `c1`); o placar grava `margem.json` |
| `teto` | o braço-régua da fração do buraco fechado (E1) |
| `metrica_primaria` | `camada` e `metrica`: a do veredito |
| `guardas` | `nome`, `par`, `metrica` (`grupo.metrica`, como `c1.retrieval_ms`) e `tolerancia` (absoluta) ou `tolerancia_perguntas` (n perguntas, que vira n / N) |
| `ollama_base_url` | opcional; o padrão é `http://127.0.0.1:11434`, pelo achado do `localhost` |

**A rodada.** Cada passo grava o seu arquivo em
`tests/benchmark/out/<id>_<data>/` e é pulado se o arquivo já está completo:

| Arquivo | Passo |
|---|---|
| `cartao.yaml`, `parametros.json` | o cartão como foi escrito; retomar com outro cartão ou outro `--limite` é recusado |
| `c1_<braço>.json` | o diagnóstico de recuperação, na língua do braço, com os fixos e o `set` |
| `respostas_<braço>.json` | `generate_answers.py --local`, com os fixos, o `set` e a semente da ordem |
| `respostas_<braço>_dentro.json`, `ragas_<braço>.json` | só as perguntas de dentro do índice vão ao juiz. O processo do juiz não herda o índice nem o embedding do braço (§4, princípio 8) |
| `placar.json`, `resumo.md` | médias, pares, margem ou veredito, guardas e buraco fechado |
| `execucao.json` | os passos, com tempo e log, de cada chamada: a original, cada retomada, cada `--so-placar` |

Antes de cada passo medido, o runner carrega no Ollama o embedding do braço e
o gerador. Sem isso, a primeira pergunta de cada braço paga o carregamento: na
fumaça, 6,0 s contra ~1,5 s das outras, porque o juiz tira o gerador da
memória entre um braço e outro.

**O veredito, provisório até a §11.** Vale para a métrica primária, com a
margem do E0 completo mais recente: o grupo `c1` para métrica da C1, e o
`gerador_juiz` para métrica da C2.

- |Δ| ≤ margem: "sem diferença relevante";
- |Δ| > margem e o IC sem o zero: "melhor" ou "pior";
- o resto: "inconclusivo". Sem E0 completo: "sem margem do E0".

O McNemar exato e o Holm estão no `placar.py`, mas o placar ainda não os usa:
a família do Holm e o papel do McNemar são da §11.

**Como usar**, da raiz:

```
backend\.venv\Scripts\python.exe tests\benchmark\run_experiment.py tests\benchmark\cartoes\E0.yaml
backend\.venv\Scripts\python.exe tests\benchmark\run_experiment.py tests\benchmark\cartoes\E0.yaml --limite 3
backend\.venv\Scripts\python.exe tests\benchmark\run_experiment.py tests\benchmark\cartoes\E0.yaml --retomar tests\benchmark\out\E0_<data>
backend\.venv\Scripts\python.exe tests\benchmark\run_experiment.py tests\benchmark\cartoes\E0.yaml --retomar tests\benchmark\out\E0_<data> --so-placar
```

**Fumaça do E0** (`--limite 3`: 3 perguntas de dentro e 3 de fora, nos 5
braços). Os 10 passos passaram, em ~9 min.

- O RAGAS leva ~35 s por caso com o `qwen3:8b`. A rodada completa, com 70
  perguntas de dentro e 4 julgamentos, deve levar perto de 3 h.
- Nas 3 perguntas, A1, A2 e A3 deram as mesmas respostas, token a token, e o
  rejulgamento (A4) deu as mesmas notas: Δ = 0 em toda métrica de qualidade.
  Três perguntas não dizem nada do ruído, só que o encanamento funciona.
- O custo varia: a latência da geração difere em até 4,5 s entre braços A/A.
  Foi isso que levou ao aquecimento acima.
- A c01, em português, recuperou só os artigos em português, e o modelo se
  absteve: a abstenção indevida de A1 é 1 em 3.

### B1 — E0: a rodada A/A (18/10)

Rodada completa em `tests/benchmark/out/E0_20261018T082340/`, com os 103 goldens
v2 (70 de dentro do `pdf-v1`, 30 de fora, 3 ambíguas), em 3 h. O RAGAS levou de
40 a 42 min por julgamento de 70 casos.

**A margem de ruído.** É o P95 de |Δ| por bootstrap de documentos, e os
vereditos dos outros experimentos a usam.

| Métrica | Gerador + juiz (A1 × A2 × A3) | Só o juiz (A1 × A4) | C1 (A1 × A5) |
|---|---|---|---|
| fidelidade | 0,068 | 0,054 | — |
| recall de contexto | 0,046 | 0,046 | — |
| relevância da resposta | 0,006 | 0,006 | — |
| precisão e relevância de contexto | 0 | 0 | — |
| abstenção indevida e correta | 0 | — | — |
| evidência@4, Hit@4, MRR@10 | — | — | 0,042 |
| precisão@4 | — | — | 0,010 |
| latência da geração | 338 ms | — | — |
| latência da busca | 57 ms | — | 7 ms |

**De onde vem o ruído:**

- **Gerador: quase nada.** Com temperatura 0 e semente 42, A1 e A3 deram as
  mesmas 103 respostas, letra por letra. A2, com outra ordem das perguntas,
  mudou 2. É o cache de prefixo do Ollama.
- **Juiz: quase tudo.** As mesmas 70 respostas, julgadas de novo, mudam de
  fidelidade em 7 a 9 casos, muitas vezes de 0 para 1. O juiz já roda com
  temperatura 0 e sem raciocínio; a variação vem do próprio Ollama, com as
  chamadas em paralelo e o cache.
- **C1: uma pergunta.** A c28 mudou entre A1 e A5 por causa do HNSW (o achado
  da seção 2). A margem de 0,042, que dá ~3 perguntas em 70, sai dela: no
  bootstrap, o documento da c28 entra 0, 1, 2 ou 3 vezes.

**A lição do cartão, medida.** A1 e A2 são a mesma configuração. A fidelidade
de A2 saiu 0,030 abaixo, com IC [−0,068; −0,005], que não contém o zero. Só
pelo IC, seria uma piora "significativa". A regra do veredito põe a margem
antes: 0,030 ≤ 0,068, sem diferença relevante.

**O baseline** (A1: `pdf-v1`, perguntas em português, `gemma3:4b`):

| | |
|---|---|
| evidência@4 (C1) | 0,11 (8 de 70) |
| abstenção indevida | 0,56 (39 das 70 perguntas de dentro recebem "não encontrei") |
| abstenção correta | 0,83 (25 de 30 de fora) |
| fidelidade · relevância da resposta | 0,73 · 0,35 |
| precisão · recall · relevância de contexto | 0,28 · 0,39 · 0,55 |
| aprovadas em todas as dimensões (limiar 0,7) | 7 de 70 |
| latência da busca · da geração | 361 ms · 2,36 s |
| tokens no prompt (média) | 2.687, sem truncagem |

A abstenção indevida é o ímã de idioma visto do outro lado: o trecho certo não
chega (evidência@4 de 0,11), e o gerador obedece à regra de não inventar.

**A recusa, revisada à mão** (o plano manda fazer nos casos do E0). Nas 103
respostas do A1, duas recusas passavam como resposta: "Com base nos documentos
fornecidos, não encontrei essa informação." (a frase sem o fim) e "O trecho
fornecido não contém informações sobre ...". As duas entraram em `VARIANTS`
(`tests/benchmark/abstention.py`). A resposta parcial ("o documento não
especifica o número exato", depois de responder o resto) continua resposta. O
placar agora recalcula a recusa pelo texto da resposta, e não pelo `abstained`
gravado. Assim, uma variante nova vale também para as rodadas antigas. O
relatório (`tests/reports/report_data.py`) passou a usar o mesmo detector.
Com ele, a diferença de abstenção entre A1 e A2 sumiu: a resposta que mudou
era uma dessas recusas.

**Uma falha do juiz:** no A1, a precisão de contexto da c10 não foi medida
(HTTP 500 do Ollama). Por isso o n é 69 nos pares dessa métrica.

**O cruzamento concentração × qualidade (E5), no A1.**

| Documento (os quatro em português) | Participação | Excesso de ímã | Respostas que o usam | Reprovadas |
|---|---|---|---|---|
| Teste de Mutação em System Prompts | 45% | +41 | 55 | 93% |
| Utilizando a IA Generativa | 18% | +35 | 38 | 100% |
| EEVEE | 13% | +21 | 26 | 96% |
| RAG Corporativo com HyDE | 10% | +14 | 17 | 94% |

66 das 70 respostas se apoiam só no top 20% dos documentos. Contra as outras 4:

- fidelidade 0,72 × 0,94, Δ −0,22 [−0,37; −0,05];
- relevância de contexto 0,53 × 0,88, Δ −0,34 [−0,57; −0,04];
- aprovação 9% × 25%, com IC largo (só 4 respostas do outro lado).

O 80/20 do português não é só descritivo: as respostas apoiadas nos ímãs são
piores.

### B2 — E1: o que o experimento precisava (18/10)

O plano (§8, E1) pede três construções, e as três estão prontas. A rodada vem
depois do E0, porque o veredito usa a margem dele.

| Arquivo | Mudança |
|---|---|
| `backend/app/config.py`, `index_versions/pdf-v1-bgem3.sha256` | a versão `pdf-v1-bgem3`: a lista do `pdf-v1` (`--register --like pdf-v1`), coleção própria e `embedding_model: bge-m3`. Ingerida: 39 documentos e 894 chunks, como o `pdf-v1`, em 57 s. O C0 dá o mesmo do `pdf-v1` |
| `backend/app/rag/translation.py` (novo) | a tradução da pergunta para o inglês, só na busca. Com cache, lê a tradução gravada; fora dele, faz uma chamada ao Ollama local (temperatura 0, sem raciocínio) |
| `backend/app/rag/retrieval.py` | `retrieve()` embeda `translation.query_for_retrieval(question)`, dentro do tempo medido. Quem chama continua com a pergunta original, e o gerador também |
| `backend/app/config.py`, `.env.example` | `RETRIEVAL_QUERY_TRANSLATION` (desligada por padrão), `RETRIEVAL_TRANSLATION_MODEL` (vazio = o gerador) e `RETRIEVAL_TRANSLATION_CACHE` |
| `tests/benchmark/arm.py` | a proveniência do braço grava a tradução: a versão do prompt, o modelo, o cache e o SHA-256 dele |
| `tests/benchmark/translate_questions.py` (novo) | traduz as 103 perguntas uma vez e grava `tests/benchmark/goldens/v2_traducoes.json`, com o tempo de cada tradução e marcas automáticas de defeito |
| `docker-compose.yml` | o `bge-m3` entra na lista `MODELS` do serviço `models` |
| `tests/benchmark/run_experiment.py` | avalia as guardas, calcula a fração do buraco fechado até o `teto` e soma o tempo gravado da tradução à latência da busca do braço |
| `tests/benchmark/cartoes/E1.yaml` (novo) | o cartão, com os cinco braços do plano e as guardas |
| `tests/api/test_translation.py`, `tests/benchmark/test_translate_questions.py` (novos) | 4 + 3 testes |

**As guardas, como ficaram no cartão.** Não-inferioridade em inglês:
`bgem3-en` ≥ `nomic-en` − 1 pergunta na evidência@4 (tolerância de 1/N). A
latência ≤ +1 s por pergunta vale para os dois candidatos, contra o
`nomic-pt`. No braço da tradução, a latência soma o tempo gravado da tradução,
porque a busca com o cache não pagou a chamada.

**A tradução, medida.** Feita com o `gemma3:4b`, o gerador do benchmark, numa
chamada por pergunta. Ela não é corrigida à mão: corrigida, viraria o
`input_en` escrito à mão, que é o braço de teto.

| Prompt | Com marca de defeito | Delas, de dentro | Tempo por tradução (média · máx.) |
|---|---|---|---|
| v1: só a instrução | 10 de 103 | 5 | 666 ms · 1,2 s |
| v2: regras explícitas e três exemplos neutros | 4 de 103 | 0 | 700 ms · 1,2 s |

- **O v1 respondia em vez de traduzir.** "Qual é a fórmula química da água?"
  virou "H2O". A pergunta sobre os seeds do MetaRAG virou "...and 10 different
  random seeds foram usadas.": um dado inventado e meio em português. A
  pergunta de injeção devolveu o próprio prompt do tradutor. E uma pergunta
  perdeu o nome do método (MeTMaP), justo a palavra que a busca precisava.
- **O v2 fica.** As 70 perguntas de dentro saem sem marca. Os exemplos do
  prompt não se parecem com nenhum golden. Sobram defeitos em 4 perguntas fora
  de escopo: o pedido de poema e o de opinião voltam sem tradução, a da Copa
  vira "Brazil" e a de culinária perde o "você é um assistente de culinária".
  A pergunta de injeção vira uma recusa, sem marca. Tudo isso só pesa na
  abstenção e na AUC do braço, não na evidência.
- **As marcas** (`flags` no arquivo): `curta` (menos da metade do tamanho da
  pergunta), `sem_pergunta` (a pergunta tem "?" e a tradução não),
  `portugues` (duas ou mais palavras que só existem em português) e `prompt`
  (repete a instrução do tradutor).
- **A guarda de latência fica justa.** A busca custa ~250 ms; com os ~700 ms
  da tradução, o Δ esperado fica perto de +0,7 s, contra a tolerância de 1 s.

**A rodada** (`tests/benchmark/out/E1_20261018T141628/`, 2 h 38 min, com os
103 goldens v2).

| Braço | evidência@4 | Hit@4 | MRR@10 | precisão@4 | latência da busca |
|---|---|---|---|---|---|
| `nomic-pt` (baseline) | 0,13 | 0,23 | 0,23 | 0,16 | 351 ms |
| `bgem3-pt` | **0,66** | 0,90 | 0,87 | 0,71 | 465 ms |
| `traducao` | 0,56 | 0,83 | 0,78 | 0,62 | 1.110 ms (com a tradução gravada) |
| `nomic-en` (teto) | 0,53 | 0,80 | 0,79 | 0,61 | 347 ms |
| `bgem3-en` | 0,63 | 0,93 | 0,88 | 0,73 | 459 ms |

- **Os dois candidatos passam do teto.** O buraco fechado é de 132% com o
  bge-m3 e de 107% com a tradução. O "teto" (o inglês escrito à mão, com o
  `nomic-embed-text`) não era o limite: o bge-m3 é melhor também em inglês,
  +0,10 [+0,02; +0,18]. A guarda de não-inferioridade em inglês passa com
  folga.
- **Veredito na C1** (margem de 0,042 do E0): `bgem3-pt` melhor, +0,53
  [+0,41; +0,66]; `traducao` melhor, +0,43 [+0,28; +0,59]. Os dois com Holm
  abaixo de 0,001.
- **Latência:** o bge-m3 custa +113 ms por busca; a tradução, +759 ms (os
  ~700 ms da tradução). As duas guardas passam.

**A C2** (as 70 perguntas de dentro julgadas, e as 103 respondidas):

| | `nomic-pt` | `bgem3-pt` | `traducao` |
|---|---|---|---|
| fidelidade | 0,70 | **0,85** | 0,83 |
| relevância da resposta | 0,35 | 0,76 | 0,74 |
| precisão · recall · relevância de contexto | 0,29 · 0,41 · 0,55 | 0,84 · 0,95 · 0,97 | 0,75 · 0,85 · 0,97 |
| aprovadas em todas as dimensões (0,7) | 7 de 70 | **35 de 70** | 23 de 70 |
| abstenção indevida | 0,56 | **0,03** | 0,04 |
| **abstenção correta** | **0,83** | 0,63 | 0,73 |
| tokens no prompt · latência da geração | 2.687 · 2,39 s | 3.291 · 2,97 s | 2.868 · 2,50 s |

- **Nos finalistas, a fidelidade** (margem de 0,068 do E0): `bgem3-pt` ×
  `nomic-pt`, +0,15 [+0,01; +0,30], melhor; `traducao` × `nomic-pt`, +0,13
  [−0,00; +0,27], inconclusivo (passa da margem, mas o IC encosta no zero);
  `bgem3-pt` × `traducao`, +0,02 [−0,07; +0,11], sem diferença relevante. O
  recall de contexto do bge-m3 é maior: +0,10 [+0,03; +0,19].
- **O preço: a abstenção correta cai.** Com o trecho certo chegando, o gerador
  também acha o que dizer nas perguntas de fora: com o bge-m3, 6 das 30 que
  antes eram recusadas passam a ter resposta (0,83 → 0,63; IC [−0,40; 0,00]).
  O cartão não tinha guarda para isso. É o trabalho do limiar, e o
  `cartoes/E6-bgem3.yaml` o testa.

**O vencedor é o bge-m3.** É a melhor C1, a fidelidade empata com a tradução, o
recall de contexto é maior, e a busca não chama o LLM. Os experimentos que
dependem do idioma passam a usar o `pdf-v1-bgem3`: o E4, como o plano manda, e
o E6 em português (`E6-bgem3`).

### B3 — E6: o que a varredura precisava (18/10)

O plano (§8, E6) varre um fator por vez a partir do baseline, em `selecao`, e
tira o veredito do escolhido em `confirmacao`.

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/ingestion.py` | `CHUNK_ACROSS_PAGES`: no PDF, a janela de tokens atravessa a página (as páginas viram uma sequência só, separadas por uma quebra de linha) e o chunk leva a página onde começa. As janelas saem de `_windows()`, que o `chunk_text` também usa; um teste prova que o corte de sempre não mudou. O manifesto grava a opção |
| `backend/app/config.py` | `chunk_across_pages` (falso por padrão) entra nas chaves que a versão pode fixar. As versões `pdf-v1-cont` (800/120), `pdf-cont-400`, `pdf-cont-1200` e `pdf-cont-o0` (800/0), todas com a lista do `pdf-v1` |
| `tests/benchmark/c1.py` | por pergunta: `context_tokens` (os tokens que o gerador recebe) e a curva `evidence_curve` × `tokens_curve` dos k primeiros trechos da lista de 10, numa passada só |
| `tests/retrieval_diagnostic.py` | conta os tokens de cada trecho (cl100k, o do chunker) |
| `tests/benchmark/run_experiment.py` | `split` no cartão (ou `--split`); `min_similarity_score: recalibrado`, calibrado na C1 do baseline da própria rodada (`limiar_<braço>.json`, com a validação deixando um documento de fora); a abstenção pelo limiar por pergunta (`c1.correct_abstention`, `c1.false_block`); métrica primária por braço; McNemar exato com Holm na família dos pares; Δ por classe de pergunta; a curva por braço |
| `tests/benchmark/cartoes/E6.yaml` (novo) | os 11 braços da tabela do plano |
| `tests/benchmark/limiar.py` | a regra do estudo de mutação, avaliada fora da amostra (commit anterior) |

**Os braços** (todos só C1, em `selecao`: 31 perguntas de dentro e 15 de fora):

| Tipo | Braços | Métrica primária | Contra |
|---|---|---|---|
| `top_k` | `k2`, `k8` | evidência@k | `base` (`pdf-v1`, k = 4) |
| MMR | `mmr-05`, `mmr-07` | cobertura@k | `base` |
| Limiar | `lim-0428` (referência: calibrado em outro índice), `lim-recal` | abstenção correta pelo limiar; guarda: falso bloqueio ≤ 1 pergunta | `base` |
| Chunk | `cont-400` (k = 8), `cont-1200` (k = 3), `cont-o0` | evidência@k, com orçamento de ~3.200 tokens | `cont` (`pdf-v1-cont`, 800/120, k = 4) |

O par `base` × `cont` mede o efeito de deixar a janela atravessar a página.

**Duas limitações, antes de rodar:**

- **O MMR tem pouca pergunta onde a hipótese mora.** São 4 perguntas
  `fato_distribuido` em `selecao` e 5 em `confirmacao`. O placar dá o Δ por
  classe, mas com n = 4 ele é só ilustrativo.
- **O limiar recalibrado é avaliado na mesma amostra em `selecao`.** O falso
  bloqueio ali dá zero por construção. O número honesto é o da validação
  deixando um documento de fora (no `limiar_lim-recal.json`) e o da
  confirmação.

**A varredura em português** (`tests/benchmark/out/E6_20261018T135338/`, 9
min com a ingestão das quatro versões novas). Está toda no chão. O baseline
acha a evidência em 2 das 31 perguntas da seleção, e nenhum braço passa de 3:

| Braço | evidência@k | Veredito (exploratório) |
|---|---|---|
| `base`, `k2`, `lim-0428`, `lim-recal`, `cont`, `cont-1200` | 2 de 31 | sem diferença relevante |
| `k8`, `mmr-05`, `mmr-07`, `cont-400` | 3 de 31 | sem diferença relevante |
| `cont-o0` | 1 de 31 | sem diferença relevante |

- **O limiar não separa em português.** A AUC da similaridade do primeiro
  trecho é 0,66 [0,47; 0,83]: o IC contém o acaso. O limiar recalibrado
  (0,558, maior F1 com margem) barra só 1 das 15 de fora. O 0,428 do estudo de
  mutação não barra nenhuma: as de fora chegam com similaridade de 0,59 a 0,70.
- **O ímã de idioma manda na varredura toda.** Nenhum parâmetro de recuperação
  conserta uma pergunta que puxa o documento errado. Por isso a mesma
  varredura roda também com as perguntas em inglês (`cartoes/E6-en.yaml`), como
  o plano faz no E4 ("ou as perguntas em EN"). Depois do E1, o E6 em português
  se refaz com o vencedor.

**A varredura em inglês** (`tests/benchmark/out/E6-en_20261018T140825/`, 9
min). Sem o ímã, o baseline acha a evidência em 16 das 31 perguntas da seleção.

| Braço | evidência@k | Métrica primária: Δ [IC 95%] | Veredito (exploratório) | Holm |
|---|---|---|---|---|
| `k2` | 12 de 31 | evidência −0,13 [−0,27; −0,03] | pior | 0,88 |
| `k8` | 19 de 31 | evidência +0,10 [+0,03; +0,19] | melhor | 1,00 |
| `mmr-05` | 12 de 31 | cobertura −0,15 [−0,34; +0,03] | inconclusivo | — |
| `mmr-07` | 13 de 31 | cobertura −0,11 [−0,21; −0,03] | pior | — |
| `lim-0428` | 16 de 31 | abstenção correta +0,07 (1 de 15) | inconclusivo | 1,00 |
| `lim-recal` (0,593) | 16 de 31 | abstenção correta +0,67 [+0,47; +0,87] (10 de 15) | **melhor** | **0,016** |
| `cont` (contra `base`) | 16 de 31 | evidência 0,00 [−0,30; +0,30] | sem diferença relevante | 1,00 |
| `cont-400` (contra `cont`) | 15 de 31 | evidência −0,03 | sem diferença relevante | 1,00 |
| `cont-1200` (contra `cont`) | 13 de 31 | evidência −0,10 | inconclusivo | 1,00 |
| `cont-o0` (contra `cont`) | 14 de 31 | evidência −0,06 | inconclusivo | 1,00 |

- **O limiar recalibrado é o único que sobrevive ao Holm.** Em inglês, a AUC
  do primeiro trecho é 0,96 [0,91; 1,00], contra 0,66 em português. O limiar
  de 0,593 barra 10 das 15 de fora sem barrar nenhuma de dentro, nem deixando
  um documento de fora por vez (a guarda passa). Ele foi calibrado em perguntas
  em inglês e não serve para as em português: o limiar depende da língua e do
  embedding.
- **O MMR piora, ao contrário da hipótese.** Nas 4 perguntas `fato_distribuido`,
  a cobertura cai 0,375 nos dois λ. Com n = 4, é um indício, não uma conclusão,
  e a confirmação tem 5.
- **`k8` ganha por construção.** Sem MMR, a lista de 8 contém a de 4, e a
  evidência só pode subir. O plano manda a decisão do `top_k` para a C2 e para
  o custo: o prompt cresce de ~2.600 para ~5.400 tokens.
- **Atravessar a página não muda o total, mas muda quem acerta.** `cont` empata
  com `base` em 16, com um IC de ±0,30: ganha nas perguntas de
  `rastreabilidade` (+2 de 3) e perde nas `fato_distribuido` (−2 de 4).
- **Por token, o chunk por página é o mais eficiente.** Na curva, `base` chega
  a 0,52 com ~2.600 tokens de contexto. `cont-400` precisa de ~4.000 para o
  mesmo, e `cont-1200`, de ~5.900. O chunk curto do fim da página, que o C0
  aponta como defeito (p10 de 263 tokens), não atrapalha a recuperação aqui.

**O que vai para a confirmação** (rodada uma vez, depois do E1): o `top_k` 8
contra o 4, com a C2 e o custo; e o limiar recalibrado, na língua e no
embedding do vencedor do E1, porque o de 0,593 só vale para o inglês com o
`nomic-embed-text`. O MMR e os chunks não passam da seleção.

**A varredura em português com o bge-m3** (`cartoes/E6-bgem3.yaml`,
`tests/benchmark/out/E6-bgem3_20261018T170702/`, 4 min), depois do E1. O
baseline acha a evidência em 21 das 31 perguntas da seleção.

| Braço | evidência@k | Métrica primária: Δ [IC 95%] | Veredito (exploratório) | Holm |
|---|---|---|---|---|
| `k2` | 18 de 31 | evidência −0,10 [−0,23; 0,00] | inconclusivo | 0,25 |
| `k8` | 25 de 31 | evidência +0,13 [+0,03; +0,23] | melhor (por construção) | 0,25 |
| `mmr-05` | 16 de 31 | cobertura −0,18 [−0,35; 0,00] | inconclusivo | — |
| `mmr-07` | 16 de 31 | cobertura −0,16 [−0,34; +0,03] | inconclusivo | — |
| `lim-recal` (0,436) | 21 de 31 | abstenção correta +0,33 [+0,13; +0,60] (5 de 15) | melhor | 0,19 |

- **Com o bge-m3, o limiar volta a separar em português:** a AUC do primeiro
  trecho é 0,91 [0,81; 0,97], contra 0,66 com o `nomic-embed-text`. O limiar
  recalibrado barra 4 das 5 perguntas fora de escopo e 1 das 10 fora do
  acervo, sem falso bloqueio (nem deixando um documento de fora). É a peça
  que falta ao E1, que perdeu abstenção correta (0,83 → 0,63).
- **O MMR piora de novo,** agora em português: nas `fato_distribuido`, a
  cobertura cai 0,50 (n = 4). Nas três varreduras (português com nomic, inglês,
  português com bge-m3), o MMR nunca ajudou.
- **A confirmação:** `top_k` 4 × 8 e o limiar de 0,436, no `pdf-v1-bgem3`, em
  `confirmacao`, com a C2.

### B5 — E2: a escada até o degrau 2 (18/10)

O plano (§8, E2) pede uma escada em que cada degrau muda uma coisa do
anterior. O degrau 1b é o `pdf-v1-cont` do E6.

| Degrau | Versão | O que muda |
|---|---|---|
| 1 | `pdf-v1` | baseline: PyMuPDF, janela de 800/120 dentro da página |
| 1b | `pdf-v1-cont` | só o corte: a janela atravessa a página |
| 2 | `md-p4llm`, `md-dl` | o formato e a extração (o Markdown já corta o documento inteiro) |
| 3 | `md-<melhor>-sec` | só o chunker: corte nos títulos; sai da seleção |

| Arquivo | Mudança |
|---|---|
| `data/convert_docling.py`, `data/requirements-docling.txt` (novos) | converte os 39 PDFs do `pdf-v1` com o Docling 2.133.0 (o mais novo com duas semanas), sem OCR, sem imagens e com o TableFormer, para `data/md_dl/`. Venv próprio (`data/.venv-docling`, 1,4 GB, torch só CPU). O manifesto `data/conversions_docling.json` grava o SHA-256 da fonte e do convertido, as páginas, o tempo e as versões |
| `backend/app/rag/ingestion.py` | `register_documents(..., by_stem=True)`: os mesmos artigos da versão-base pelo nome sem a extensão, em outro formato (`--register --like pdf-v1 --by-stem`). O chunker por seção (`chunker="section"`): o Markdown normalizado é cortado nas linhas `#` (`markdown_sections()`) e a janela corre dentro de cada seção, com `chunk_id` `{stem}::s{i}::c{j}`. As ferramentas do estudo de mutação leem `::p(\d+)::` e não o veem. `ChunkRecord.section` e os metadados do Chroma (só quando há seção, como a página) |
| `backend/app/config.py` | `CHUNKER` (`token` ou `section`, padrão `token`) entra nas chaves que a versão pode fixar. As versões `md-p4llm` (a conversão pymupdf4llm que o `md-v2` já trouxe) e `md-dl` |
| `backend/app/rag/retrieval.py`, `backend/app/schemas.py` | a seção no dicionário do `retrieve()` e no `SourceChunk`; sem isso, o pydantic descartaria o campo na resposta do `/chat` |
| `tests/benchmark/run_experiment.py` | `classe_por` no cartão: o Δ por grupo sai por outro campo do golden (no E2, `content_type`) |
| `tests/benchmark/cartoes/E2.yaml` (novo) | a seleção: os degraus 1, 1b e 2, só C1, com as confusões do plano declaradas |
| `.gitignore` | `data/md_dl/`, o manifesto e o venv do Docling |

**A conversão.** 18 min para os 38 PDFs em CPU (de 5 a 135 s por arquivo; o
RAGChecker, com 29 páginas, levou 135 s). As tabelas saem com as colunas
inteiras: a linha da configuração ID 19 do MetaRAG traz "0.9410 ± 0.0357" na
coluna da precisão.

**Uma limitação, antes de rodar:** o `content_type` dos goldens quase não tem
`secao` (0 pergunta na seleção, 2 na confirmação) e tem pouco `texto` (4 e 8).
A interação que dá para medir é `tabela` × `numero` × `texto`.

**A C0 por degrau** (a C0 explica, não decide):

| Versão | Chunks | Tokens | Chunk médio · p10 | Bibliografia | Linhas de tabela | Títulos |
|---|---|---|---|---|---|---|
| `pdf-v1` | 894 | 524.705 | 647 · 263 | 20,9% | — | 0 |
| `pdf-v1-cont` | 785 | 524.705 | 782 · 799 | 20,9% | — | 0 |
| `md-p4llm` | 775 | 519.274 | 784 · 800 | 21,5% | 1.531 | 929 |
| `md-dl` | 716 | 476.454 | 779 · 800 | 22,8% | 1.244 | 956 |

- **O corte por página faz chunk pequeno.** No `pdf-v1`, o p10 é de 263
  tokens: o fim de cada página vira um chunk curto. Atravessando a página, o
  p10 vai a 799.
- **O Docling tem 9% menos tokens, mas não perde texto.** Dos 25-gramas
  normalizados do PDF, 96,9% reaparecem no Docling (o pior artigo, 89,6%) e
  96,0% no pymupdf4llm (o pior, 84,1%). A diferença é marcação e cabeçalho
  repetido, que o Docling tira.
- **Nenhuma das duas cola palavras:** 0,17% das letras, contra 39% nos
  originais do `md-v1`.

**A seleção em português** (`tests/benchmark/out/E2_20261018T140452/`): os
quatro degraus acham a evidência nas mesmas 2 das 31 perguntas. Como no E6, o
ímã de idioma esconde o formato. A escada roda também com as perguntas em
inglês (`cartoes/E2-en.yaml`), depois do E1.

**O degrau 3, pronto para a confirmação.** As versões `md-p4llm-sec` e
`md-dl-sec` (a lista do degrau 2 e `chunker: section`) estão registradas; a
seleção diz qual vai. O corte nos títulos muda a distribuição de tamanhos, como
o plano previu:

| Versão | Chunks | Tokens por chunk: média · p10 · mediana | Chunks com menos de 50 tokens |
|---|---|---|---|
| `md-p4llm` (janela) | 775 | 784 · 800 · 800 | 0 |
| `md-p4llm-sec` | 1.276 | 438 · 54 · 401 | 118 |
| `md-dl` (janela) | 716 | 779 · 800 · 800 | 0 |
| `md-dl-sec` | 1.243 | 411 · 55 · 356 | 119 |

Os chunks de menos de 50 tokens são, na maior parte, títulos sem texto (uma
seção que só abre subseções). O corte segue o plano ao pé da letra. Por isso a
confirmação do degrau 3 compara também com orçamento de tokens igual: k = 8
com ~420 tokens por chunk dá os ~3.200 do k = 4 com 800.

**A seleção em inglês** (`tests/benchmark/out/E2-en_20261018T165730/`, 2 min).
O Markdown não ganha do PDF:

| Degrau | evidência@4 | Δ contra o degrau anterior [IC 95%] | Veredito |
|---|---|---|---|
| 1 · `pdf` | 16 de 31 | — | — |
| 1b · `pdf-cont` | 16 de 31 | 0,00 [−0,30; +0,30] | sem diferença relevante |
| 2 · `md-p4llm` | 11 de 31 | −0,16 [−0,36; +0,03] | inconclusivo |
| 2 · `md-dl` | 12 de 31 | −0,13 [−0,32; +0,07] | inconclusivo |

- **Atravessar a página empata no total e troca quem acerta:** ganha em
  `texto` (+1 de 4) e `numero` (+1 de 12), perde em `tabela` (−2 de 15).
- **As duas conversões perdem do PDF** em todos os tipos de evidência, mais em
  `numero` (−2 de 12) e `texto` (−1 de 4) do que em `tabela` (−1 de 15 no
  Docling). A hipótese ("o MD ganha mais em tabela") não aparece: o MD perde
  menos em tabela.
- **A ferramenta:** o Docling fica 1 pergunta acima do pymupdf4llm (+0,03
  [−0,16; +0,23]), com a diferença em `tabela`. Pela regra do plano, a
  escolha sai da seleção: o degrau 3 da confirmação é o `md-dl-sec`.
- Na curva, o `md-dl` alcança o PDF com 9 trechos (~7.100 tokens): a
  evidência está no Markdown, mas chega mais tarde na lista.

**A confirmação** (`cartoes/E2c.yaml`, `tests/benchmark/out/E2c_20261018T201049/`,
em inglês, as 39 perguntas de dentro de `confirmacao`, 4 min):

| Degrau | evidência@4 | tokens de contexto | Δ [IC 95%] | Veredito |
|---|---|---|---|---|
| 1 · `pdf` | 21 de 39 | 2.575 | — | — |
| 1b · `pdf-cont` (contra 1) | 23 de 39 | 3.180 | +0,05 [−0,06; +0,18] | inconclusivo |
| 2 · `md-dl` (contra 1b) | 15 de 39 | 3.187 | −0,21 [−0,37; −0,06] | **pior** |
| 3 · `md-dl-sec`, k = 4 (contra 2) | 15 de 39 | 1.700 | 0,00 [−0,24; +0,18] | sem diferença relevante |
| 3 · `md-dl-sec`, k = 8 (contra 2) | 22 de 39 | 3.400 | +0,18 [−0,03; +0,37] | inconclusivo |
| 3 · `md-dl-sec`, k = 8 (contra 1) | — | — | +0,03 [−0,15; +0,18] | sem diferença relevante |

- **O Markdown perde do PDF neste acervo.** A conversão do Docling custa 8
  perguntas contra o `pdf-cont`, mais em `numero` (−5 de 13) e `tabela` (−3 de
  16). O chunker por seção, com o mesmo orçamento de tokens, devolve o que a
  conversão tirou e empata com o PDF, sem passar dele.
- **Conferido: não é viés da medida.** A evidência dos goldens foi copiada do
  texto do PDF, e o Docling reorganiza tabelas. Pelo validador
  (`goldens validar --indice md-dl --evidencia`), só 2 dos 86 fragmentos não
  casam no texto do Docling, contra 0 no PDF; a perda de 8 perguntas não vem
  daí. No pymupdf4llm são 14 de 86: o resultado da seleção dele tem parte de
  medida.
- **A leitura para a turma:** "o Markdown é melhor" não vale aqui. O que o MD
  permite (o corte por seção) só compensa o que a conversão perde, e a escada
  mostra os dois efeitos separados.

### B6 — E3: o cabeçalho contextual e a atribuição errada (18/10)

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/titles.json` (novo), `tests/benchmark/draft_titles.py` (novo) | o título de cada um dos 39 artigos, rascunhado pelos metadados do PDF (33) ou pelo texto de maior fonte da primeira página (6). Três rascunhos saíram errados e foram corrigidos: "123" (um número solto na página), "R C : A Fine-grained…" e "RAGA : Automated…" (versalete perdido; são o RAGChecker e o RAGAS). **Todos ficam com `revisado: false` até a revisão humana**, como o plano pede |
| `backend/app/rag/titles.py` (novo) | a tabela e o cabeçalho "Artigo: <título> · pág. N" |
| `backend/app/rag/ingestion.py`, `backend/app/config.py` | `CHUNK_HEADER`: `embedding` põe o cabeçalho só no texto embedado (o guardado, o prompt e os juízes não mudam); `text`, no próprio trecho, fora da janela de 800 tokens. O manifesto grava o modo e o SHA-256 da tabela de títulos. As versões `pdf-hdr-emb` e `pdf-hdr` |
| `backend/app/rag/generation.py`, `backend/app/config.py` | a regra de citação v2 (§6): cita `[arquivo]` quando o trecho não tem página. Atrás de `PROMPT_CITATION_RULE=v2`; a v1 continua o padrão, literal, porque o estudo de mutação a lê |
| `tests/benchmark/attribution.py` (novo) | as categorias da §6: recusa, (a) sem citação, (b) citação fabricada (arquivo ou página que não veio), (c) cita outro documento recebido, (d) cita a fonte. Atribuição errada = (b) nas de dentro, mais a resposta não abstida que cita algo numa pergunta de fora; o denominador é o total de respostas |
| `tests/benchmark/run_experiment.py` | o grupo `attribution` no placar, só quando a proveniência da rodada diz que a citação era obrigatória; a camada `respostas` (as respostas sem o juiz, para a A/A da atribuição); a margem do A/A do próprio cartão valendo para as métricas que ele mediu, e a do E0 para o resto |
| `tests/benchmark/cartoes/E3.yaml` (novo) | `pdf-v1` (baseline), `pdf-hdr-emb` e `pdf-hdr`, os três com a citação v2, e a A/A curta com citação |

**Uma limitação, antes de rodar:** só 7 perguntas de dentro nomeiam o artigo
(`names_article`). A interação do cabeçalho com o nome do artigo fica só
ilustrativa.

### B7 — E7: a busca híbrida (18/10)

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/bm25.py` (novo) | BM25 (Okapi, k1 = 1,5, b = 0,75) em Python puro, com índice invertido, montado com o texto da coleção (`get_all`) e guardado por coleção e contagem de chunks. Tokens: letras e dígitos, sem acento e sem caixa |
| `backend/app/rag/retrieval.py` | `RETRIEVAL_HYBRID`: a lista do vetor e a do BM25 (cada uma com top_k × `RETRIEVAL_FETCH_MULTIPLIER`) fundidas por RRF (k = 60, `RETRIEVAL_RRF_K`). O trecho que só o BM25 achou ganha o cosseno pelo vetor guardado, porque o limiar e o MMR usam o `similarity_score`. Cada trecho diz de onde veio (`found_by`). O índice BM25 é montado fora do tempo medido. Com a flag desligada, a busca é a de sempre |
| `backend/app/rag/vector_store.py` | `get_by_ids()`, para o vetor dos trechos do BM25 |
| `tests/benchmark/c1.py` | as listas da C1 guardam o `found_by` |
| `tests/benchmark/run_experiment.py` | guarda de subgrupo (`grupo: {campo, valor}`), para "sem sigla não piora" |
| `tests/benchmark/cartoes/E7.yaml` (novo) | `pdf-v1` × híbrida, em português e em inglês, com o Δ por `has_acronym` |
| `tests/api/test_hybrid.py` (novo) | 5 testes: tokens, ranking, cache, a fusão com o cosseno do trecho do BM25, e a busca de sempre sem a flag |

Em português, a sigla e o nome próprio são as palavras que a pergunta e o
acervo em inglês têm em comum. O BM25 pode achar o documento que o embedding
não acha, e é isso que o E7 mede no `has_acronym`.

**A rodada** (`tests/benchmark/out/E7_20261018T170323/`, 4 min, as 70
perguntas de dentro):

| Par | evidência@4 | Δ [IC 95%] | Veredito | Δ sem sigla · com sigla |
|---|---|---|---|---|
| `base` → `hibrida` (português) | 8 → 11 de 70 | +0,04 [−0,03; +0,10] | inconclusivo | +0,06 (n = 34) · +0,03 (n = 36) |
| `base-en` → `hibrida-en` | 37 → **51** de 70 | **+0,20 [+0,09; +0,31]** | **melhor** (Holm 0,001) | +0,21 · +0,19 |

- **A hipótese da interação cai:** a híbrida não ganha mais nas perguntas com
  sigla. Em inglês, ela ganha 0,2 nas duas metades.
- **O efeito principal, em inglês, é grande.** 51 de 70 é mais que o bge-m3
  sozinho (44 em inglês, 46 em português). A combinação bge-m3 + híbrida fica
  para testar.
- **Em português com o nomic, quase nada:** o BM25 acha a sigla, mas o resto da
  lista continua puxado pelo ímã de idioma.
- **As guardas passam:** sem sigla não piora (+0,06), e a latência não muda
  (o BM25 custa menos de 1 ms; o índice é montado uma vez, fora do tempo
  medido).

**A híbrida com o bge-m3** (`cartoes/E7-bgem3.yaml`,
`tests/benchmark/out/E7-bgem3_20261018T200351/`, exploratório, 6 min): **em
português, ela piora.**

| Par | evidência@4 | precisão@4 | Δ evidência [IC 95%] | Veredito | Holm |
|---|---|---|---|---|---|
| bge-m3 → + híbrida (português) | 46 → 38 de 70 | 0,71 → 0,44 | −0,11 [−0,22; −0,02] | **pior** | 0,02 |
| bge-m3 → + híbrida (inglês) | 44 → 53 de 70 | 0,73 → 0,80 | +0,13 [+0,05; +0,23] | melhor | 0,008 |

- **A guarda "sem sigla não piora" não passa** em português: −6 das 34
  perguntas sem sigla.
- **O porquê:** o BM25 casa as palavras da pergunta em português com os quatro
  artigos em português do acervo, e o RRF põe esses trechos de volta na lista.
  O ímã de idioma, que o bge-m3 tinha resolvido no vetor, volta pelo canal
  lexical. Em inglês, onde a pergunta e o acervo falam a mesma língua, a
  híbrida soma de novo.
- **A lição** é a do cartão: duas melhorias que funcionam sozinhas não somam
  por obrigação. Para perguntas em português, a combinação recomendada é o
  bge-m3 sem a híbrida; a híbrida só vale com a pergunta na língua do acervo.

### B8 — E8: o reranker (18/10)

O plano propõe um cross-encoder (`bge-reranker-v2-m3`) e dá uma "alternativa
sem dependência nova: usar o LLM como reranker via Ollama, que é mais lento".
Foi a escolhida: o cross-encoder pediria o torch no venv do backend.

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/rerank.py` (novo) | cada candidato vira uma chamada ao Ollama local que pede uma nota de 0 a 10 para "o trecho ajuda a responder a pergunta?", com temperatura 0, sem raciocínio e 4 tokens de saída. Resposta sem número vale 0. A nota já pedida fica em cache no processo |
| `backend/app/rag/retrieval.py`, `backend/app/config.py` | `RETRIEVAL_RERANK`: 20 candidatos (`RETRIEVAL_RERANK_CANDIDATES`, os 4 × 5 do plano), ordenados pela nota (empate: o maior cosseno), e os k primeiros seguem. A nota vai em `rerank_score`; o cosseno continua em `similarity_score`. Com o reranker, o MMR não roda. `RETRIEVAL_RERANK_MODEL` vazio = o gerador |
| `tests/benchmark/c1.py` | a nota do primeiro trecho por pergunta e a AUC dela, ao lado da do cosseno |
| `tests/benchmark/cartoes/E8.yaml` (novo) | `pdf-v1` × reranker, em português e em inglês |
| testes | 5 em `tests/api/test_rerank.py` e 1 em `test_c1.py` |

**O custo:** 20 chamadas de LLM por pergunta. O número fixo de candidatos e o
cache existem para a lista de 10 do diagnóstico reusar as notas da lista do
braço, em vez de pedir 50 notas a mais.

**A rodada** (`tests/benchmark/out/E8_20261018T171129/`, as 70 perguntas de
dentro). Um HTTP 500 passageiro do Ollama derrubou o braço em português na
primeira tentativa; o reranker ganhou três tentativas por chamada, e a rodada
foi retomada.

| Par | precisão@4 | evidência@4 | Hit@1 | MRR@10 | AUC das de fora: cosseno · nota | latência da busca |
|---|---|---|---|---|---|---|
| `base` → `rerank` (português) | 0,16 → **0,23** | 9 → 15 de 70 | 0,19 → 0,37 | 0,23 → 0,37 | 0,80 · **0,67** | 0,26 → **11,7 s** |
| `base-en` → `rerank-en` | 0,61 → 0,64 | 37 → 45 de 70 | 0,71 → 0,77 | 0,79 → 0,83 | 0,97 · **0,63** | 0,27 → **10,0 s** |

- **Veredito na precisão@4** (margem de 0,010 do E0): em português, melhor,
  +0,08 [+0,04; +0,12]; em inglês, inconclusivo, +0,03 [−0,02; +0,08]. O
  reranker ajuda mais onde o cosseno é pior: em português, o Hit@1 dobra.
- **A segunda hipótese cai:** a nota do LLM separa as perguntas de fora *pior*
  que o cosseno (0,67 contra 0,80 em português; 0,63 contra 0,97 em inglês).
  Uma nota de 0 a 10 do `gemma3:4b` é grossa demais para virar portão.
- **O custo inviabiliza o uso interativo:** 40 vezes a latência da busca. As
  guardas passam (o MRR sobe; a latência ficou dentro dos 15 s da guarda, que
  foi folgada de propósito), mas a recomendação é não ligar o reranker com o
  LLM no assistente. O cross-encoder que o plano propõe custaria uma chamada
  em lote, e não 20 sequenciais; fica como continuação, porque pede o torch no
  venv do backend.

### B4 — E5: a análise de concentração e a primeira leitura (18/10)

O E5 não é um braço: é uma análise sobre o diagnóstico de qualquer braço. Ela
não depende do runner e roda já.

**O que mudou.**

| Arquivo | Mudança |
|---|---|
| `tests/benchmark/e5.py` (novo) | as métricas da tabela do E5: participação observada e esperada, massa fora da fonte, TVD, N80, o teste do enunciado (observado e esperado), Gini e Gini@1, ímã, morto, chunk ímã com a marca de bibliografia, e a curva de Lorenz. CLI: `python -m tests.benchmark.e5 diag.json --indice pdf-v1 --lingua pt --bibliografia` |
| `tests/benchmark/c1.py` | cada pergunta grava também `documents_at_10`, que o "morto" usa |
| `tests/benchmark/test_e5.py` (novo) | 4 testes, num caso feito à mão |

**Duas definições que o plano deixa em aberto**, e como ficaram:

- **Morto:** documento que é fonte de alguma pergunta e não aparece nos 10
  primeiros trechos de nenhuma das perguntas dele.
- **Chunk de bibliografia:** metade ou mais dos 25-gramas dele estão no bloco
  de referências do documento (`references_block` do C0). No `pdf-v1`, são
  183 dos 894 chunks (20,5%), o que bate com os 20,9% de tokens do C0.

**Primeira leitura**, no baseline (`pdf-v1`, 70 perguntas de dentro, k = 4):

| | Massa fora da fonte | TVD | N80 (esperado) | Todas as fontes no top 20% (esperado) | Gini (esperado) |
|---|---|---|---|---|---|
| português | 84% | 0,82 | 4 de 39 (17) | 94% (44%) | 0,90 (0,51) |
| inglês | 39% | 0,22 | 15 de 39 (17) | 47% (44%) | 0,59 (0,51) |

- **Em português, o 80/20 é viés do sistema, não da demanda.** 94% das
  perguntas recebem só trechos dos 20% de documentos mais recuperados. O
  desenho das perguntas daria 44%.
- **Os ímãs são os quatro artigos em português do acervo:**
  - *Teste de Mutação em System Prompts*: 52 perguntas alheias, contra 10,7 ao
    acaso;
  - *Utilizando a IA Generativa*: 38, contra 2,8;
  - *EEVEE*: 26, contra 4,7;
  - *RAG Corporativo com HyDE*: 17, contra 2,5.

  É o H5a, confirmado no baseline. Dez documentos em inglês ficam mortos: as
  perguntas sobre eles, em português, nunca os trazem nos 10 primeiros.
- **Em inglês, a concentração é quase a do desenho**, e não há documento ímã.
  O único morto é o artigo em português: as perguntas dele, em inglês, nunca o
  trazem. É o mesmo efeito, na direção contrária.
- **Contra o H5b: bibliografia não é ímã nas perguntas de dentro.** Dos trechos
  recuperados para elas, só 0,7% (português) e 2,5% (inglês) são de
  bibliografia, contra 20,5% do índice. Nenhum chunk ímã é de bibliografia.
- **A favor de outra hipótese:** nas perguntas de fora, em inglês, 16,7% dos
  trechos recuperados são de bibliografia. A lista de referências, com o
  título de outros artigos, parece servir de resposta quando o acervo não tem
  uma. Pode ser o caminho da "fonte errada" da §3. Fica para o E3 e para o
  `pdf-v1-noref`, que agora tem uma previsão: cortar a bibliografia mexe pouco
  na evidência das perguntas de dentro e mais na abstenção das de fora.

**O cruzamento concentração × qualidade** está pronto em código
(`e5.py --respostas ... --ragas ...`, com 2 testes novos). Por documento, ele
põe a participação e o excesso de ímã ao lado da taxa de reprovação das
respostas que o usam. Reprovada é a resposta que o RAGAS não aprova em todas
as dimensões (limiar 0,7), a regra do relatório. Ele também compara as
respostas apoiadas só no top 20% com as demais, com IC por bootstrap dos
documentos-fonte. Os trechos são os que o gerador recebeu na rodada julgada. A
atribuição errada precisa de citação e fica para o E3.

**O H5a, com o E1: os ímãs somem com o bge-m3.** A mesma análise sobre a C1 dos
três braços em português do E1:

| Braço | Massa fora da fonte | TVD | N80 (esperado 17) | Todas as fontes no top 20% (esperado 44%) | Ímãs | Mortos |
|---|---|---|---|---|---|---|
| `nomic-pt` | 84% | 0,82 | 4 de 39 | 94% | os 4 em português | 10 |
| `bgem3-pt` | 29% | 0,16 | 15 de 39 | 40% | nenhum | nenhum |
| `traducao` | 38% | 0,22 | 14 de 39 | 44% | nenhum | 1 (o artigo em português) |

Com o bge-m3, a concentração fica a do desenho das perguntas: o 80/20 do
português era viés do embedding, e ele some. Com a tradução, o efeito se
inverte no único artigo em português: as perguntas dele, traduzidas, não o
trazem mais, como no `nomic-en`.

**O que falta no E5:**

- ~~rodar o cruzamento na C2 do E0~~: feito, na seção do B1;
- ~~o H5a~~: confirmado, acima;
- ~~o H5b~~: refutado (abaixo);
- a demanda real das aulas. Para começar a coletar, basta um `LOGS_DIR` por
  aula no backend local (§8, E5).

**O H5b, rodado: a bibliografia não é ímã** (`cartoes/E5-noref.yaml`,
`tests/benchmark/out/E5-noref_20261018T165925/`, 4 min).

| Par | evidência@4 | Hit@4 | AUC do primeiro trecho (de dentro × de fora) |
|---|---|---|---|
| `pdf-v1` → `pdf-v1-noref` (português) | 0,13 → 0,13 | 0,23 → 0,24 | 0,79 → 0,80 |
| `pdf-v1` → `pdf-v1-noref` (inglês) | 0,53 → 0,53 | 0,80 → 0,81 | 0,971 → 0,972 |

Os ímãs são os mesmos com e sem o corte (os quatro artigos em português, nas
perguntas em português), e nenhum chunk ímã era de bibliografia. Cortar 17% do
texto dos artigos não mudou a evidência nem a separação das perguntas de fora.
Os 16,7% de trechos de bibliografia que as perguntas de fora recebiam, em
inglês, só trocam de lugar com outros trechos de similaridade parecida. O
ajuste "cortar a bibliografia" não se paga.

**O H5b, construído.**

| Arquivo | Mudança |
|---|---|
| `backend/app/rag/ingestion.py` | `strip_references()`: o bloco de referências que a C0 acha (`corpus_quality.references_block`) é cortado antes do chunking, página a página; página que fica sem texto sai. O manifesto grava quantas linhas saíram de cada documento |
| `backend/app/config.py` | `STRIP_REFERENCES` (falso por padrão) entra nas chaves que a versão fixa; a versão `pdf-v1-noref`, com a lista do `pdf-v1` |
| `tests/benchmark/noref_check.py` (novo) | a conferência que o plano pede antes de rodar: fragmento de evidência dos goldens que casa no texto inteiro e não casa sem o bloco seria evidência perdida |
| `tests/benchmark/cartoes/E5-noref.yaml` (novo) | `pdf-v1` × `pdf-v1-noref`, em português e em inglês, só C1 |
| testes | 1 de corte (`test_rag_indices.py`) e 1 da conferência (`test_noref_check.py`) |

A conferência passou: **nenhum dos 86 fragmentos de evidência dos goldens cai
no bloco cortado**. O corte tira 5.207 linhas, 17% do texto dos 39 artigos, e
38 páginas que eram só bibliografia. Um artigo não tem bloco (o RETORCH, o
resumo de uma página).
