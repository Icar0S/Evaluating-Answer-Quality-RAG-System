# Plano — Benchmark didático de estratégias de RAG

> Escrito em 17/10/2026 sobre o estado do repositório no commit `fcf039a`.
> Caminhos, flags, campos de JSON e números citados foram conferidos no código
> e nos READMEs nessa data. O que ainda não existe está marcado **(novo)**.
> Ferramentas externas citadas (conversores, modelos de embedding, rerankers)
> devem ser reconfirmadas na versão disponível na hora de implementar.
>
> **Nome do arquivo:** salvar como `plano-benchmark.md` (na raiz ou em
> `docs/`). O `plano.md` da raiz é o plano do DeepEval, citado abaixo, e não
> deve ser sobrescrito.

## 1. Objetivo

Transformar a arquitetura de testes num **banco de ensaio de decisões**. Cada
experimento responde a uma pergunta de engenharia ("PDF ou Markdown?",
"10 ou 1000 documentos?", "vale pôr um reranker?") e termina com um veredito e
**o ajuste recomendado em forma aplicável**: as linhas de `.env` ou a entrada
de `RAG_INDICES` do braço vencedor, com o tamanho do efeito e o intervalo de
confiança ao lado.

Dois públicos, o mesmo pipeline:

- **A turma**, que assiste ao pipeline rodando e precisa entender *como* o
  teste está sendo feito: o que muda, o que fica fixo, o que cada camada mede
  e por que a conclusão vale.
- **Quem mantém o RAG**, que precisa de um relatório que aponte o ajuste
  certo, e não só uma média.

### Perguntas de pesquisa

| # | Pergunta | Experimento |
|---|---|---|
| P1 | Com **os mesmos artigos**, Markdown recupera e responde melhor que PDF? Em que tipo de pergunta? | E2 |
| P2 | Pergunta em português sobre acervo em inglês: embedding multilíngue ou tradução da pergunta resolvem? | E1 |
| P3 | Quanto acervo um chatbot de suporte deve ter? Quanto cada degrau de documentos custa em precisão e ganha em cobertura? | E4 |
| P4 | As respostas se apoiam sempre nos mesmos poucos documentos (80/20)? Isso é demanda real ou viés do sistema? E as respostas apoiadas nesses documentos são piores ou melhores? | E5 |
| P5 | Quais estratégias (cabeçalho no chunk, busca híbrida, reranker, parâmetros) pagam o próprio custo neste acervo? | E3, E6, E7, E8 (e a lista da §8.1) |

## 2. Onde isso entra no que já existe

Quase tudo de que o benchmark precisa já existe em alguma forma. O plano é
**ligar as peças**, não reescrever:

| Peça existente | O que já faz | Papel no benchmark |
|---|---|---|
| `RAG_INDICES` (`backend/app/config.py`) | Versão do índice = acervo + coleção + modelo de embedding | Cada braço de formato, idioma ou escala vira uma versão |
| `scripts/ingest_documents.py` (`--index`, `--list`, `--source`, `--no-reset`, `--if-empty`) | Ingere e grava o manifesto (`data/vector_store/manifests/<coleção>.json`, com SHA-256 por arquivo) | O manifesto é a proveniência de cada braço |
| `set_settings_overrides()` (`backend/app/config.py`) | Sobrescreve a configuração em memória e limpa o cache | Troca de braço em processo, sem reindexar e sem mexer em `os.environ` |
| `tests/retrieval_diagnostic.py` | Posição do documento certo, pergunta em PT e EN, só embeddings | Semente da Camada 1 |
| `tests/generate_answers.py` | Gera as respostas uma vez; todos os juízes julgam as mesmas | Camada 2, um arquivo de respostas por braço |
| `tests/compare_indices.py` | Orquestra N versões: ingestão, diagnóstico, respostas, RAGAS, relatório | Modelo do runner de experimentos |
| `tests/run_all.py` (`Runner`, `run_logged`, `Step`) | Etapas, logs, pulos com motivo, `execucao.json` | Reaproveitado pelo runner |
| `tests/reports/` | Relatório MD, PDF, XLSX e CSV, com histórico | Ganha as seções novas |
| `tests/mutation/stats.py` | `bootstrap_ci`, `mcnemar` (exato), `holm`, `wilson_ci`, `cohen_kappa` | Referência da estatística do placar (ver §10, linha de dependências) |
| `tests/mutation/runner/sut.py` e `cost.py` | Configuração por braço, detecção de prompt truncado, custo (GPU-s, Wh) | Padrão para braços em processo e para o custo |
| `tests/mutation/suite/golden.jsonl` | 30 casos com classe (`fato_direto`, `fato_distribuido`, `filtro_condicional`, `rastreabilidade`, `fora_do_corpus`, `fora_de_escopo`, `ambiguo`) e `evidence_hint` | Taxonomia de perguntas e semente dos goldens v2 (com a ressalva da §7) |
| `tests/mutation/tools/calibrate_retrieval_gate.py` (`choose_threshold`) | Regra de calibração do limiar de similaridade, sem LLM | A regra é copiada para o runner (§8, E6). O script não roda nos goldens v2 |
| `frontend/tests/e2e/fixtures.ts` (faixa `#e2e-demo-banner`) e `helpers.showSidePanel` | Faixa e painéis na tela durante a demonstração | Cartão de hipótese ao vivo |

## 3. O que já se sabe (e motiva cada experimento)

Achados registrados no README, no `docs/ARCHITECTURE.md` e nos arquivos de
calibração. Cada um vira hipótese de algum experimento:

- **Idioma.** No `md-v1`, com o `nomic-embed-text`, o documento certo veio
  entre os 4 trechos em 2 de 10 perguntas em português e em 10 de 10 em
  inglês. A pergunta em português puxa o único documento em português do
  acervo. No `pdf-v1`, o mesmo efeito tirou o MetaRAG da recuperação. → E1,
  E5.
- **O `md-v1` não é o PDF em outro formato.** São 22 outros artigos, nenhum
  entre os 39 PDFs. Além disso, 16 dos 22 foram degradados pela conversão
  (de 3% a 25% das palavras grudadas, segundo o README). → E2 precisa de
  conversões novas.
- **Fonte errada.** Nas perguntas de fora do acervo, as respostas atribuíram ao
  artigo perguntado o conteúdo de outro: os "sete pontos de falha" saíram de
  outros artigos, e os "12 estudos" eram 102 (o 12 era de outra revisão).
  → E3 e a métrica de atribuição errada (§6).
- **O juiz de contexto não ordena índices.** O `context_recall` deu 1,00 a
  trechos sem os fatos da referência. → A recuperação é decidida por
  métricas sem juiz (Camada 1). As métricas de contexto do juiz ficam como
  leitura secundária.
- **Precisão de contexto baixa.** 0,40 (RAGAS) no `pdf-v1`. → E6, E7, E8.
- **Quase nunca se abstém fora do acervo.** O `pdf-v1` se absteve em 1 de 10
  perguntas de fora; o `md-v1`, em 2 de 10. O limiar calibrado no estudo de
  mutação (0,428, `tests/mutation/config/retrieval_gate.json`) **não separa**
  as classes: os positivos vão de 0,49 a 0,63 e os negativos de 0,42 a 0,61.
  Ele barra só 2 dos 8 casos negativos. Esse limiar foi calibrado em outro
  índice (8 PDFs re-renderizados, chunk 1200/200, MMR). → E6 e E8.
- **Truncagem silenciosa.** Sem `GENERATION_NUM_CTX`, o Ollama usa 4096 e
  corta o início do prompt (288 truncagens no baseline do estudo de mutação).
  → Todo braço fixa `GENERATION_NUM_CTX=auto` e `GENERATION_NUM_PREDICT`.
- **Temperatura 0 não basta para repetir.** `GENERATION_TEMPERATURE` e
  `GENERATION_SEED` vêm vazios por padrão. Mesmo com temperatura 0, a resposta
  depende do prompt anterior no servidor, por causa do cache de prefixo do
  Ollama. Em ordem fixa, as repetições saem idênticas, e a variação só aparece
  com a ordem embaralhada (`tests/mutation/config/study.yaml`,
  `shuffle_cases_per_repetition`). → E0, e a ordem embaralhada com semente em
  todo braço.

## 4. Princípios do desenho

1. **Uma variável por vez.** Cada braço é o baseline com uma mudança. Se um
   experimento precisa de duas mudanças, ele vira uma escada de braços que
   separa os efeitos (ver E2).
2. **Comparação pareada.** Todos os braços respondem às mesmas perguntas, com
   o mesmo gerador, o mesmo juiz e a mesma temperatura. A ordem das perguntas
   é embaralhada com semente registrada.
3. **Funil: barato em tudo, caro só nos finalistas.** A Camada 1 roda em todos
   os braços (segundos a minutos). O juiz LLM roda no baseline e em até 2
   candidatos.
4. **Recuperação se decide sem juiz; resposta se decide com juiz.** O juiz já
   mostrou que não ordena índices pela recuperação (§3).
5. **Ruído medido antes do sinal.** A rodada A/A (E0) define a margem abaixo
   da qual uma diferença não é relevante.
6. **Regras escritas antes de rodar.** Cada cartão traz **uma** métrica
   primária, as métricas de guarda com a sua margem, a regra de veredito e o
   critério dos finalistas. O relatório mostra o cartão como foi escrito.
7. **Proveniência de ponta a ponta.** O braço (índice + configuração efetiva)
   vai das respostas ao relatório. Hoje a identidade de uma rodada é só
   `rag_index` + idioma (§10).
8. **O instrumento não muda entre braços.** O juiz é o mesmo em todos os
   frameworks e em todos os braços, com a mesma janela de contexto. O
   processo do juiz nunca herda o índice nem o embedding do braço: a
   `AnswerRelevancy` do RAGAS usa o `embedding_model` das settings do processo
   de avaliação (§9).
9. **Provider local sempre.** O provider remoto ignora temperatura, semente,
   `num_ctx` e `think`. O harness de mutação já força `local`.

## 5. O funil de camadas

| Camada | Pergunta que responde | Quem roda | Métricas principais | Custo | Portão para a próxima |
|---|---|---|---|---|---|
| **Sanidade** | O sistema está de pé? | pytest (`tests/api`) e Playwright (interface) | testes herméticos e E2E | ~3 min | Vermelho = não roda o benchmark |
| **C0 · Acervo** | O texto que entra no índice presta? | `ingest_documents.py` (manifesto) | palavras grudadas, páginas sem texto, tabelas, cabeçalhos, tokens por chunk, % de bibliografia | segundos (sem LLM) | Não barra. Explica a C1 de antemão |
| **C1 · Recuperação** | O trecho certo chega ao gerador? | `retrieval_diagnostic.py` estendido (sem framework) | Hit@k, MRR, evidência@k, precisão@k, abstenção pelo limiar, concentração | segundos a minutos (só embeddings) | Por experimento, no cartão (`finalistas` no YAML, §9). Braço cortado = "não avaliado na C2", nunca "perdeu" |
| **C2 · Resposta** | A resposta usa bem o que chegou? | RAGAS sobre as respostas do `generate_answers.py` | 5 dimensões do RAGAS, abstenção correta e indevida, atribuição errada | ~70 s de julgamento por resposta com GPU, mais a geração | Placar (§11) |
| **C3 · Confiança** | Os juízes concordam entre si e com o humano? | DeepEval numa amostra (JevEval opcional), revisão humana | kappa de Cohen entre os vereditos, sinal do Δ | horas, só antes da aula | Divergência alta (§6) → veredito "inconclusivo" |
| **C4 · Custo e decisão** | Qual configuração adotar, pelo custo? | `build_report.py` | latência, tokens, tamanho do índice, tempo de ingestão; fronteira custo × qualidade | — | Veredito + ajuste aplicável |

## 6. Métricas: definições exatas

### C0 · Acervo **(novo)**

Calculadas na ingestão e gravadas no manifesto, por documento (hoje cada
entrada de `documents[]` tem só `file`, `format`, `bytes`, `sha256` e
`chunks`):

- `pages_total` e `pages_with_text`. Hoje o `extract_pdf_pages` descarta as
  páginas vazias em silêncio.
- `tokens` (cl100k, o mesmo tokenizador do chunker) e a distribuição de tokens
  por chunk (média, p10, p90). O PDF é cortado por página, e o tamanho
  efetivo do chunk pode ficar bem abaixo do pedido.
- `glued_word_rate`: fração das **letras** que estão em tokens alfabéticos com
  18 ou mais letras fora do dicionário (inglês + português). Medir em letras
  não depende de quantas palavras cada token grudado engoliu. É um limite
  inferior: grudes curtos, como "ofthe", escapam. Validar contra uma amostra
  rotulada à mão (por exemplo, 200 linhas de 6 arquivos). A faixa do README
  (3% a 25% das palavras) é só ordem de grandeza, porque o método dela não
  está registrado.
- `table_rows`: linhas de tabela. No MD, as linhas com `|` antes da
  normalização. No PDF, "n/d".
- `headings`: no MD, as linhas que começam com `#`. No PDF, 0, porque o
  `page.get_text()` não traz cabeçalhos.
- `references_share`: fração dos tokens do **bloco de referências**. O bloco
  vai do título "References"/"Referências" até o próximo título de apêndice
  ("Appendix", "Supplementary"...) ou até as linhas deixarem de parecer
  referência. Não usar "tudo depois de References": em artigo do arXiv, os
  apêndices costumam vir depois.

### C1 · Recuperação (estende `tests/retrieval_diagnostic.py`)

- **Lista ranqueada** por pergunta e braço: `chunk_id`, `document`,
  `similarity_score`. Ela é medida **com o `top_k` e o MMR do próprio braço**
  (com MMR, o conjunto de candidatos depende do k) e, à parte, com k = 10 para
  o Hit@10. Hoje o diagnóstico guarda só a posição.
- **Hit@k** (k = 1, 4, 10): o documento-fonte aparece entre os **k primeiros
  trechos**. É o `rank` de hoje (posição do primeiro trecho do documento), e
  vale para o corte que o gerador recebe.
- **MRR@10**: pelo rank deduplicado do documento, entre os documentos
  distintos presentes nos 10 primeiros trechos.
- **Precisão@k (novo)**: fração dos k trechos que são do documento-fonte ou
  contêm a evidência. É a alternativa sem juiz à precisão de contexto do
  RAGAS.
- **Evidência@k (novo)**: a evidência do golden chegou ao gerador? Ela não
  depende de página, de `chunk_id` nem de formato.
  1. **Fragmentos.** Cada item da lista `context` do golden já é um
     fragmento. Dentro de um item, quebrar também nos `[...]` e `...`.
  2. **Normalização** do fragmento e dos trechos:
     `unicodedata.normalize("NFKC", ...)` (desfaz ligaduras como "ﬁ", comuns
     no texto do PyMuPDF), minúsculas e remoção de espaços, pontuação e hífen
     suave. Assim, palavras grudadas e hifenização deixam de importar.
  3. **Casamento.** O fragmento casa se pelo menos τ = 60% dos seus 25-gramas
     de caracteres aparecem na **união** dos k trechos. Um fragmento partido
     entre dois trechos recuperados conta. O relatório registra à parte se
     ele coube num trecho só.
  4. **Cobertura@k** = fração dos fragmentos que casam. **Evidência@k** =
     cobertura@k igual a 1. Em `fato_distribuido`, todos os fragmentos
     contam, e não só o mais longo.
  
  Calibrar τ à mão com 20 pares. O `context` precisa conter só texto literal
  da fonte (§7).
- **Identidade de documento:** comparar por `Path(x).stem` dos dois lados
  (`artigo.pdf` ↔ `artigo.md`) em `retrieval_diagnostic.py`, em
  `demo_e2e.retrieval_diagnostic` e nas contas do E5. Hoje os dois comparam o
  nome exato, com extensão. No E2 (os mesmos artigos em PDF e MD), sem isso
  nenhum acerto cruza os formatos. Uma mesma pasta, porém, nunca mistura dois
  formatos do mesmo artigo, senão a identidade por stem os junta.
- **Abstenção pelo limiar:** nas perguntas de fora, a fração que fica sem
  nenhum trecho acima de `MIN_SIMILARITY_SCORE`. Nas de dentro, a fração
  barrada sem dever (falso bloqueio). A separação entre as classes é medida
  pela AUC dos scores, com IC por bootstrap.

### C2 · Resposta

- **As 5 dimensões do RAGAS**, como hoje, só nas perguntas de dentro do
  acervo.
- **Abstenção correta** (pergunta de fora → recusa) e **abstenção indevida**
  (pergunta de dentro → recusa). Não precisam de juiz.
  - Hoje a recusa é uma busca do `NOT_FOUND_MARKER` sem diferenciar
    maiúsculas, em `generation.generate_answer` (`grounded`) e em
    `report_data.Case.abstained`. É esta última que conta no relatório.
  - Uma recusa com outra frase passa como resposta. **(novo)** Normalizar a
    detecção com uma lista pequena de variantes, revisada à mão nos casos do
    E0.
- **Atribuição errada (novo)**, medida nos braços com citação obrigatória.
  Cada resposta cai numa categoria:

  | Categoria | Definição | Conta como |
  |---|---|---|
  | (a) sem citação | Não cita nenhum trecho | Não conformidade, com taxa reportada por braço |
  | (b) citação fabricada | Cita arquivo (ou arquivo + página) que não está entre os trechos recuperados | **Atribuição errada** |
  | (c) cita outro documento | Cita só trechos recuperados de outro documento | Erro de recuperação, já contado na C1 |
  | (d) cita a fonte | Cita o documento do golden, e o trecho citado é dele | Correto |

  Também conta como **atribuição errada** a resposta não abstida a uma
  pergunta de fora que nomeia ou cita o artigo perguntado. É exatamente o caso
  dos "sete pontos" e dos "12 estudos". O denominador é o total de respostas,
  não só as com citação.
  
  A `CITATION_RULE` de hoje pede `[arquivo, pág. N]`, e o MD não tem página.
  Por isso, uma regra de citação **v2 (novo)** cita `[arquivo]` quando o
  trecho não tem página. Ela fica atrás de flag, desligada por padrão, porque
  o estudo de mutação depende da `CITATION_RULE` literal (baseline com
  `prompt_require_citation: true`, operador P3 e assertiva `must_cite_page`).
  O cabeçalho do trecho fixado em `tests/api/test_rag_indices.py` não muda.
  `PROMPT_VERSION` muda à mão.

### C3 · Confiança **(novo)**

- **DeepEval numa amostra** de 10 a 15 perguntas por finalista, só na rodada
  feita antes da aula. Com até 18 min por golden medidos no README, rodar
  tudo não cabe.
- **Concordância:** kappa de Cohen (`cohen_kappa`, o mesmo de
  `tests/mutation/stats.py`) entre os vereditos RAGAS e DeepEval, por
  dimensão.
- **Divergência alta** (definida no cartão): os juízes discordam no **sinal**
  do Δ da métrica primária, ou o kappa fica abaixo de 0,4 nessa dimensão.
  Nesse caso, o veredito sai como **inconclusivo**.
- Os casos de divergência são os primeiros da fila de revisão humana (Fase 3
  do README).

### C4 · Custo e decisão

- **Por resposta:** latência de recuperação (devolvida por `retrieve()`),
  latência de geração e `prompt_tokens`/`completion_tokens` (devolvidos por
  `generate_answer()`). Hoje o `generate_answers.py` descarta esses números e
  guarda só `seconds` e `sources`.
- **Por índice:** `n_chunks` e `duration_ms` do manifesto.
- **Opcional:** GPU-s e Wh, com `tests/mutation/runner/cost.measure()` (NVML).
- **Juiz:** `wall_seconds` e `judge_calls`. O RAGAS já grava; o DeepEval
  passa a gravar.
- **`GENERATION_THINK` fixo** em todos os braços, porque os tokens de
  raciocínio contam em `completion_tokens` (no `qwen3:8b`, 727 → 224 tokens
  com `think=false`).

## 7. O conjunto de perguntas (goldens v2) **(novo)**

### Por que um conjunto novo

- `tests/deepeval/goldens/dataset.json` (os 10 dos PDFs) não tem
  `source_document` nem `input_en`. Sem isso, a C1 não roda nos índices de
  PDF.
- `dataset_md.json` é um rascunho para revisão humana e é sobre outro acervo.
  Além disso, ele traz notas editoriais dentro do `context` (por exemplo,
  "(No Markdown, este trecho está com as palavras grudadas...)"), que
  quebrariam a evidência@k.
- Com 10 perguntas, a meia-largura do IC de 95% de **uma** proporção perto de
  0,5 é de ~±31 p.p. A diferença entre dois braços exige mais (§11).

### Esquema

Um arquivo `tests/benchmark/goldens/v2.json`, compatível com o
`generate_answers.py` (`name`, `input`, `input_en`, `expected_output` e
`context` mantêm o nome de hoje):

| Campo | Para quê |
|---|---|
| `name` | **Único em todos os conjuntos**: os juízes indexam as respostas por nome, e nomes repetidos colidem em silêncio |
| `input`, `input_en` | PT e EN em toda pergunta, para o fator idioma rodar em qualquer experimento |
| `expected_output` | Como hoje |
| `context` | Só **texto literal da fonte**, um fragmento por item. O validador rejeita notas entre parênteses ("No Markdown...", "Nota:...") |
| `observacao` | Onde vão as notas editoriais |
| `source_document` | Stem do documento-fonte, sem extensão. `null` nas perguntas de fora |
| `evidence_page` | Opcional, só PDF |
| `question_class` | A taxonomia de `tests/mutation/suite/golden.jsonl` |
| `content_type` | Onde a evidência mora. Com precedência `tabela` > `numero` > `secao` > `texto` (um número dentro de uma tabela é `tabela`) |
| `has_acronym` | A pergunta depende de sigla ou nome próprio (RAGAS, ARES, Defects4J...). Recorte do E7 |
| `names_article` | A pergunta nomeia o artigo. Recorte do E3 |
| `out_stratum` | Só nas perguntas de fora: `perto` (mesma área, documento ausente), `longe` ou `escopo` (receita de bolo, injeção de prompt...) |
| `split` | `selecao` ou `confirmacao`, sorteado **por documento** e estratificado por `content_type`, antes de qualquer rodada (§11) |
| `set` | `goldens` ou `demanda` (E5) |
| `status` | `rascunho` ou `revisado`. Só os `revisado` entram nos vereditos |

### Tamanho e composição

- **Núcleo: pelo menos 20 dos 39 PDFs, com 3 perguntas cada (≥ 60).** O n
  efetivo de uma comparação está entre o número de documentos e o de
  perguntas, porque as perguntas de um mesmo documento erram juntas (§11).
- **Cotas, escritas de propósito, para os recortes usados em hipóteses:**
  - ≥ 15 perguntas com a evidência em `tabela`;
  - ≥ 15 com `has_acronym = false` (as 10 de hoje nomeiam todas RAGAS, ARES,
    MetaRAG, RAGChecker ou Visagio);
  - ≥ 15 com `names_article = false`.
  
  Um recorte com menos de 15 perguntas sai como **descritivo**, sem veredito.
- **Fora do acervo: ≥ 30 perguntas**, divididas nos três `out_stratum`. São
  baratas de escrever, a C1 é barata, e a abstenção não precisa de juiz. Com
  só 8 a 10, nenhuma comparação de abstenção chega a veredito.
- **Rotulagem dupla.** Duas pessoas rotulam `content_type` em 20 perguntas, e
  o kappa vai para o relatório.
- **Candidatos gerados por LLM e revisados à mão**, como o `plano.md`
  (DeepEval) previa para o Synthesizer.
  - **Ponto de partida:** os 30 casos de `tests/mutation/suite/golden.jsonl`.
  - **Atenção:** o `evidence_hint` deles usa os nomes curtos (48 caracteres)
    e as páginas do corpus re-renderizado do estudo de mutação (1800 tokens
    por página).
  - **Na migração:** mapear o documento pelo `source_file` de
    `tests/mutation/corpus/manifest.json` e reencontrar a página e o
    fragmento no PDF original. Não copiar a página.

## 8. Experimentos

### 8.1 Estratégias consideradas

A resposta a "quais outras estratégias valem entrar no benchmark" tem um
critério: **atacar um defeito já observado (§3), custar pouco para construir e
caber numa aula**.

| Estratégia | Defeito que ataca | Custo de construir | Cabe numa aula? | Entra? |
|---|---|---|---|---|
| Embedding multilíngue (`bge-m3`) | Idioma (2/10 × 10/10) | baixo: uma versão | sim | **E1** |
| Tradução PT→EN só na busca | Idioma | baixo | sim | **E1** |
| HyDE / multi-query | Idioma, vocabulário | médio | sim | Depois, como rival da tradução no E1 |
| Formato e conversão (PDF × MD) | Extração | médio: conversões | sim | **E2** |
| Chunker por seção | Corte no meio do assunto | médio | sim | **E2** |
| Cabeçalho contextual no chunk | Fonte errada | baixo/médio: títulos | sim | **E3** |
| Tamanho do acervo | Distração × cobertura | médio/alto: documentos | sim | **E4** |
| `top_k`, MMR, limiar, chunk e overlap | Precisão baixa, sem abstenção | só config | sim | **E6** |
| Corte da bibliografia | Trechos ímã (hipótese) | baixo | sim | **E5** (H5b) |
| Busca híbrida BM25 + vetorial | Siglas e nomes próprios | médio | sim | **E7** |
| Reranker *cross-encoder* | Precisão baixa, sem abstenção | médio/alto | sim | **E8** |
| Filtro por metadado / roteamento pelo artigo citado na pergunta | Fonte errada | médio | sim | **Candidato a E10**, depois do E3 |
| Parent-document / small-to-big | Contexto partido (FP3/FP4) | médio | sim | Depois do E2 |
| GraphRAG, RAPTOR, RAG agêntico, ajuste fino do embedding | — | alto | não | Fora (§13) |

### 8.2 Fixos e convenções

Fixos em **todos** os braços, salvo quando o fator é o próprio fixo:

- **Gerador `gemma3:4b`.**
- **Juiz `qwen3:8b` em todos os frameworks.** O runner define
  `RAGAS_JUDGE_MODEL=qwen3:8b` e `DEEPEVAL_JUDGE_MODEL=qwen3:8b` no processo do
  juiz. O `.env.example` traz `DEEPEVAL_JUDGE_MODEL=gemma3:4b`, que é o
  gerador (*self-grading*). Como juiz, o `gemma3:4b` reprova nos controles do
  RAGAS.
- **Janela do juiz igual para todos.** `RAGAS_JUDGE_NUM_CTX` é dimensionado
  pelo maior braço do experimento, por exemplo 16384. Com o padrão de 8192,
  braços com `TOP_K=8` ou chunk 1200 estouram a janela, o juiz falha, e o
  relatório compara só os casos medidos dos dois lados (perda diferencial).
- **Geração:**
  - `GENERATION_TEMPERATURE=0` e `GENERATION_SEED` fixo;
  - `GENERATION_NUM_CTX=auto` e `GENERATION_NUM_PREDICT` explícito;
  - `GENERATION_THINK=false`;
  - provider `local`.
- **Recuperação:** chunk 800/120, `TOP_K=4`, MMR desligado e limiar 0,0 (os
  defaults do repo).
- **Perguntas:** os goldens v2 `revisado`, em ordem embaralhada com semente
  registrada.

**Formato do cartão** (o mesmo no terminal, na tela e no relatório):

- Pergunta;
- Hipótese;
- Braços (candidatos e referências);
- Fixos (só o que difere do §8.2);
- Métrica primária, métricas de guarda (com margem) e portão dos finalistas;
- Precisa construir;
- Saída no relatório;
- **Didática:** o que a turma deve perceber, em uma frase.

**Nomes das versões:** a versão-base mais o que mudou, com **no máximo ~13
caracteres**. A aba da planilha é `"Matriz " + framework + " · " + versão` e
o Excel corta em 31. Exemplos: `pdf-v1-bgem3`, `pdf-v1-cont`, `md2-dl`,
`md2-p4llm`, `md2-dl-sec`, `pdf-hdr`, `pdf-hdr-emb`, `pdf-v1-noref`,
`esc-dif-030`. Esses nomes **substituem** o que o README anuncia hoje
(`md-v2` e `*-v2` com embedding multilíngue). README e ARCHITECTURE passam a
seguir esta convenção.

### E0 — Linha de base e ruído (rodada A/A)

- **Pergunta:** quanto duas rodadas idênticas diferem entre si?
- **Hipótese:** a C1 tem ruído ~0, porque o embedding é determinístico (a
  reconstrução do índice pode variar um pouco). A C2 tem dois ruídos
  separáveis:
  - **o do gerador**, que vem do cache de prefixo do Ollama mesmo com
    temperatura 0 (§3);
  - **o do juiz.**
- **Braços:**
  - A1: baseline;
  - A2 e A3: baseline com a ordem das perguntas **embaralhada** (semente = nº
    da repetição), como o estudo de mutação faz. Em ordem fixa, as rodadas
    sairiam idênticas e o ruído medido seria zero por construção;
  - A4: as respostas de A1 rejulgadas em outra ordem (só o ruído do juiz);
  - A5: o mesmo acervo reingerido numa coleção nova, comparando as listas
    ranqueadas (ruído da C1).
- **Saída:** a margem de ruído `m_E0` por métrica, igual ao percentil 95 de
  |Δ| entre os pares A/A, por bootstrap por documento. Ela entra na regra de
  veredito (§11). É **refeita** quando o prompt ou o gerador muda (E3 com
  citação, E9).
- **Didática:** uma diferença de 0,03 entre dois braços pode ser só sorte, e
  dá para dizer de onde vem a sorte.

### E1 — Idioma da pergunta

- **Pergunta:** pergunta em PT sobre acervo em EN: o que resolve?
- **Hipótese:**
  - `bgem3-pt` fecha a maior parte do buraco entre o PT e o teto EN, sem
    piorar o EN;
  - `traducao` fecha parecido, ao custo de uma chamada de LLM por pergunta.
- **Braços candidatos:**

  | Braço | Versão | Pergunta |
  |---|---|---|
  | `nomic-pt` (baseline) | `pdf-v1` | PT |
  | `bgem3-pt` | `pdf-v1-bgem3` | PT |
  | `traducao` | `pdf-v1` | PT; só a busca usa a tradução para EN |
- **Braços de referência** (régua da C1; não disputam vaga nem vão à C2):

  | Braço | Versão | Pergunta |
  |---|---|---|
  | `nomic-en` (teto) | `pdf-v1` | `input_en`, escrito à mão |
  | `bgem3-en` | `pdf-v1-bgem3` | `input_en` |
- **Métrica primária:** evidência@4 em PT (C1), reportada também como
  **fração do buraco fechado** = (braço − baseline) / (teto − baseline).
  Nos finalistas: fidelidade (C2).
- **Guardas:**
  - não-inferioridade em EN: `bgem3-en` ≥ `nomic-en` − 1 pergunta;
  - latência de recuperação ≤ +1 s por pergunta.
- **Precisa construir:**
  - **A versão `pdf-v1-bgem3`.** Uma entrada em `RAG_INDICES` com o mesmo
    `source_dir`, coleção própria e `embedding_model: bge-m3`. Não basta
    `EMBEDDING_MODEL=bge-m3` no ambiente: com `RAG_INDEX` definido, o
    validador sobrescreve.
  - **A tradução.** Um passo antes do `embed_texts([question])` em
    `retrieval.retrieve()`, atrás de `RETRIEVAL_QUERY_TRANSLATION`.
    - O gerador continua recebendo a pergunta original.
    - A tradução é feita **uma vez**, antes da aula, e gravada por pergunta.
      C1 e C2 leem a mesma tradução, e a C1 ao vivo continua só com
      embeddings.
  - **O demo.** O `bge-m3` entra na lista `MODELS` do serviço `models` do
    `docker-compose.yml`.
- **Variações opcionais:**
  - **"nomic com prefixos"** (`search_query:`/`search_document:`, que o
    `nomic-embed-text` espera e o `embed_texts` não envia). Custa uma versão
    nova (reindexar) e um parâmetro de papel em `embed_texts`.
  - **Repetir no `md-v1`.** Com `dataset_md.json` limpo das notas e uma
    versão `md-v1-bgem3`, marcado como exploratório até a revisão.
- **Saída:** tabela braço × idioma, fração do buraco fechado e a concentração
  por braço (liga com o H5a do E5).
- **Didática:** o teto em inglês vira régua. A turma vê quanto do buraco cada
  estratégia fecha, e não só "subiu ou desceu".

### E2 — Formato: PDF × Markdown, com os mesmos artigos

- **Pergunta:** com os mesmos 39 artigos, Markdown é melhor que PDF? Em que
  tipo de pergunta?
- **Hipótese:**
  - o ganho do MD sobre o PDF é maior nas perguntas com evidência em `tabela`
    e `secao` do que em `texto`. Isso é um teste de **interação**, ver §11;
  - a ferramenta de conversão pesa tanto quanto o formato;
  - o maior ganho vem do chunker por seção, que só o MD permite.
- **Preparação:**
  - Converter os 39 PDFs offline com duas ferramentas (por exemplo
    pymupdf4llm e Docling; alternativa: Marker) para `data/md2_p4llm/` e
    `data/md2_dl/`.
  - **Pôr as pastas no `.gitignore`.** Hoje, das pastas de acervo, ele cobre
    só `data/source_pdfs/*` e `data/new_papers_marks/*`, e o acervo tem
    restrição de licença.
  - **Escolher a ferramenta só com o `split: selecao`.** O degrau 3 e o
    veredito saem na `confirmacao`.
- **Braços: uma escada, não um 2×2.** O texto do PyMuPDF não tem cabeçalhos,
  então um "PDF com chunker por seção" teria de sair da conversão para MD, e
  aí é MD. A escada separa os efeitos:

  | Degrau | Versão | O que muda em relação ao anterior |
  |---|---|---|
  | 1 | `pdf-v1` | baseline: texto do PyMuPDF, janela de 800/120 **dentro de cada página** |
  | 1b | `pdf-v1-cont` **(obrigatório)** | só o corte: as páginas do mesmo texto são concatenadas antes da janela |
  | 2 | `md2-p4llm`, `md2-dl` | o formato e a extração, com a mesma janela contínua |
  | 3 | `md2-<melhor>-sec` | só o chunker: corte nos cabeçalhos, com a janela dentro de cada seção |

- **Confusões a declarar no cartão:**
  - **Cabeçalhos e rodapés.** Os conversores costumam remover cabeçalho e
    rodapé de página e carimbos do arXiv. O PyMuPDF mantém.
  - **Tabelas.** O `MD_NORMALIZATION` v1 transforma tabela em células
    separadas por espaço. "MD ganha em tabela" testa a ordem de leitura, não
    a estrutura. Uma variação opcional, `md2-<melhor>-tab`, normaliza linhas
    de tabela como "coluna: valor".
  - **Tamanho dos chunks no degrau 3.** O chunker por seção muda a
    distribuição de tamanhos (seção curta, chunk curto). Comparar também com
    **orçamento de tokens igual**: k ajustado para ~3200 tokens de contexto.
- **Precisa construir:**
  - **O degrau 1b** (`pdf-v1-cont`): uma flag de ingestão que concatena as
    páginas antes do `chunk_text`.
  - **Chunker por seção.** Em `build_chunk_records()`, no ramo `.md`:
    dividir o texto normalizado nas linhas `#`, que o `normalize_markdown`
    preserva, e aplicar o `chunk_text` dentro de cada seção. É controlado por
    um campo de settings `chunker: token|section`.
  - **`chunk_id` novo, só nas versões novas:** `{stem}::s{i}::c{j}`. Ferramentas
    do estudo de mutação fazem parse de `::p(\d+)::`.
  - **Seção ponta a ponta.** `section` no `ChunkRecord`, nos metadados do
    Chroma (só quando não for `None`, como `page`), no dicionário devolvido por
    `retrieve()` e no `SourceChunk`. Sem isso, o pydantic descarta o campo na
    resposta do `/chat`.
  - **Parâmetros do chunker na versão:** chaves opcionais `chunker`,
    `chunk_size_tokens` e `chunk_overlap_tokens` em `RAG_INDICES`, copiadas
    pelo validador como as três de hoje (§14, Fase A).
- **Métrica primária:** evidência@4 (C1), com o teste de interação por
  `content_type`. Nos finalistas: fidelidade (C2). A C0 **explica**, não
  decide. O recall de contexto do juiz é só descritivo.
- **Guardas:** abstenção indevida ≤ baseline + 1 pergunta.
- **Saída:**
  - tabela da C0 por versão;
  - gráfico da escada (evidência@4 por degrau, uma linha por `content_type`);
  - veredito por degrau.
- **Didática:** a escada mostra **de onde vem** o ganho: do corte por página,
  do formato ou do chunker. "MD é melhor" vira "isto aqui no MD é melhor".

### E3 — Cabeçalho contextual no chunk

- **Pergunta:** colocar "Artigo: \<título\> · pág. N" antes de cada trecho
  reduz a resposta com fonte errada?
- **Hipótese:**
  - **no embedding**, o cabeçalho sobe o Hit@4 nas perguntas com
    `names_article = true`, e não nas outras (interação);
  - **no texto que o gerador lê**, ele reduz a atribuição errada.
- **Braços:** os três rodam **com citação obrigatória**, que é um fixo do E3
  e não parte do tratamento:
  - `pdf-v1` + citação (baseline);
  - `pdf-hdr-emb`: cabeçalho **só no texto embedado**. O texto armazenado, o
    prompt e o que os juízes leem não mudam. É o efeito puro na recuperação;
  - `pdf-hdr`: cabeçalho **no texto do chunk** (embedding + prompt + juízes).
- **Fixos:** `PROMPT_REQUIRE_CITATION=true`, com a regra de citação v2 (§6) e
  o mesmo `PROMPT_VERSION` nos três. Uma rodada A/A curta com citação refaz a
  `m_E0`.
- **Métrica primária:** taxa de atribuição errada (C2, categoria (b) mais as
  perguntas de fora).
- **Guardas:** evidência@4 ≥ baseline − 1 pergunta; taxa de "sem citação"
  sem diferença relevante entre os braços.
- **Precisa construir:**
  - **Títulos reais.** Uma tabela `titles.json` por stem, tirada dos
    metadados do PDF ou da primeira página e revisada à mão. Hoje o gerador vê
    só o nome do arquivo (`[Trecho i - arquivo (pág. N)]`), e um dos MD tem
    nome trocado.
  - **Cabeçalho no embedding:** no loop de lotes do `ingest_documents()`,
    embedar `header + text` e manter `documents=[text]`.
  - **Orçamento de tokens.** O cabeçalho fica fora da janela de 800 tokens,
    então o prompt cresce. Conferir a truncagem com a regra do `sut.py`
    (`prompt_tokens >= prompt_token_limit() - 2`).
  - **Seção no cabeçalho:** no `pdf-v1` não existe. Ela é testada num braço
    opcional `md2-dl-sec-hdr` ("Artigo · Seção"), se o E2 terminar antes.
- **Saída:**
  - tabela das categorias (a)–(d) por braço;
  - Hit@4 por `names_article`;
  - exemplos de respostas que mudaram de categoria.
- **Didática:** a resposta pode estar certa e a fonte errada. O teste mostra
  as duas coisas separadas.

### E4 — Escada de acervo: 10 ou 1000 documentos?

- **Pergunta:** quanto acervo um chatbot de suporte deve ter?
- **Hipótese:**
  - distratores fáceis (fora do tema) quase não afetam a recuperação;
  - distratores difíceis (mesmo tema) derrubam a evidência@4 do núcleo a
    partir de algum degrau;
  - a cobertura da demanda cresce até ser atendida e depois para de crescer;
  - a curva tem um ponto ótimo, que híbrida e reranker (E7, E8) empurram para
    a direita.
- **Braços:** degraus **encaixados**, em que cada um contém o anterior:

  | Versão (difícil / fácil) | Acervo |
  |---|---|
  | `esc-000` | só o núcleo (os ≥ 20 documentos com perguntas) |
  | `esc-dif-010` / `esc-fac-010`, `-030` | + 10 / + 30 distratores |
  | `-100`, `-300`, `-1000` | + 100 / + 300 / + 1000 |

  - **Difícil:** só PDFs do mesmo tema, com o mesmo extrator do núcleo (o
    restante dos 39 e artigos do arXiv sobre avaliação de RAG e LLM).
  - **Fácil:** PDFs de outra área.
  - **Os 22 MD do `md-v1` ficam fora.** Eles trazem outro formato, 16
    documentos degradados e o único documento em português (ímã conhecido).
    O degrau em que entrassem cairia por idioma e formato, não por distração.
- **Fixos:** o embedding vencedor do E1 (ou as perguntas em EN), para a
  escada não medir o ímã de idioma.
- **Preparação:**
  - **Uma pasta plana e uma coleção por degrau.** A ingestão não é recursiva e
    recria a coleção. Não montar degraus com `--no-reset`, porque o manifesto
    listaria só o último lote. As pastas entram no `.gitignore`.
  - **Ordem de entrada sorteada com 3 sementes.** O relatório mostra a curva
    média e a faixa; uma ordem só é uma realização ao acaso.
  - **Perguntas por degrau.** Antes de rodar, cada degrau difícil ganha
    perguntas `revisado` sobre uma amostra dos documentos que acrescenta
    (por exemplo, 1 pergunta a cada 10 documentos novos), ou usa a demanda
    rotulada do E5. Sem isso, a cobertura não cresce depois do núcleo por
    construção, e a regra escolheria sempre o menor degrau.
  - **Revisão por *pooling*.** Em cada degrau, os documentos novos que
    aparecem no top-4 das perguntas do núcleo são revisados à mão. Se contêm
    a resposta (um survey pode citar o RAGAS), passam a contar como
    relevantes.
  - **Licença.** Só importa se o acervo for redistribuído. Nesse caso,
    auditar reaproveitando as regras (`RULES`) de
    `tests/mutation/tools/audit_licenses.py` num script que receba a pasta do
    degrau. O script atual só audita o corpus do estudo de mutação.
- **Métricas:** duas curvas sobre o mesmo eixo de degraus, mais o custo:
  - **Precisão sob distração:** evidência@4 só nas perguntas do núcleo.
  - **Cobertura:** evidência@4 nas perguntas do maior degrau (ou na demanda
    rotulada). No sabor fácil, só se mede a precisão.
  - **Custo:** tempo de ingestão, `n_chunks` e latência de recuperação por
    degrau.
- **Regra de decisão** (X e Y escritos no cartão antes de rodar): **o menor
  degrau que cobre ≥ X% da demanda sem perder mais que Y de evidência@4 no
  núcleo**. A C2 (com abstenção e atribuição errada) roda só no degrau
  escolhido e no maior.
- **Saída:** o gráfico das duas curvas, com a faixa das 3 ordens e o degrau
  escolhido marcado.
- **Didática:** "mais documentos" não é melhor nem pior em si. A turma vê as
  duas forças (cobertura e distração) e o ponto em que elas se cruzam.

### E5 — Concentração das fontes (80/20)

Não é um braço: é uma **análise que roda sobre os resultados dos outros
experimentos**. Ela só pede rodada nova no H5b.

- **Pergunta:** 80% das respostas se apoiam em só 20% dos documentos? Isso é
  demanda real ou viés do sistema? E as respostas apoiadas nesses documentos
  são piores ou melhores?
- **Dados:**
  - **Goldens** (`set: goldens`): os trechos devolvidos em cada resposta. O
    `generate_answers.py` passa a gravar `source_documents` limpo. Hoje dá
    para derivar de `sources`, tirando o " (pág. N)".
  - **Demanda** (`set: demanda`): perguntas reais da turma, de
    `logs/interactions.jsonl` (`sources[].document`), filtradas por
    `metadata.rag_index`.
    - **Separar o tráfego de teste.** O E2E também passa pelo `/chat` e é
      logado sem marca. No backend local, basta um `LOGS_DIR` por aula. Na
      demonstração em Docker, o `LOGS_DIR` está fixo em `/data/logs`, nos
      volumes `rag-logs`/`rag-logs-md`. Muda para
      `LOGS_DIR: /data/logs/${AULA:-padrao}` no `x-backend-env`, mais um passo
      que copia o `interactions.jsonl` do volume para
      `tests/benchmark/demanda/<aula>.jsonl`.
    - **Começar a coletar já na próxima aula.** No backend local basta o
      `LOGS_DIR` por aula, e isso pode vir antes da Fase B. A mudança no
      compose e a rotulagem são o passo 26 (§14).
    - **Rotular uma amostra.** No mínimo 40 perguntas distintas (deduplicadas
      por similaridade) rotuladas à mão com o documento que as responde, ou
      `null`. Sem rótulo, a demanda mostra só a concentração observada.
    - **Por braço:** as perguntas da demanda são reexecutadas na C1 de cada
      braço, porque o log tem só o braço que serviu a aula.
- **Métricas.** Só entram perguntas com `source_document`; as de fora vão
  para a abstenção. Cada pergunta tem massa 1, e n_q é o número de trechos que
  ela de fato recebeu (o limiar pode devolver menos que k):

  | Métrica | Definição |
  |---|---|
  | **Participação observada** o_d | média, sobre as perguntas, de n_{q,d} / n_q |
  | **Participação esperada** e_d | fração das perguntas cujo `source_document` é d (o que uma recuperação perfeita daria) |
  | **Massa fora da fonte** | média de 1 − n_{q,fonte} / n_q. É a medida principal de viés |
  | **Distância de variação total** | ½ · Σ \|o_d − e_d\| |
  | **N80** | menor número de documentos cuja participação acumulada (em ordem decrescente) chega a 80%. Reportado como N80/N, ao lado do N80 esperado |
  | **O teste do enunciado** | fração das respostas cujas fontes estão **todas** no top 20% dos documentos (o enunciado literal), ao lado da fração com pelo menos uma. Os dois valores também são calculados para e_d, para separar o que vem do desenho das perguntas. Nos braços com citação, contar também pelos documentos **citados** |
  | **Gini** | Gini de o (e de e, na mesma unidade), e o Gini@1 (documento no rank 1). Descritivo; só se compara entre braços com o mesmo N no manifesto |
  | **Ímã** | documento com ≥ 3 aparições indevidas (por pergunta, só nas de dentro) acima do esperado ao acaso (fração de chunks do documento no índice × k × Q) |
  | **Morto** | documento com pergunta que nunca aparece nos 10 primeiros trechos |
  | **Chunk ímã** | a mesma conta por `chunk_id`, separando os trechos de bibliografia |
  | **Concentração × qualidade** | por documento: participação e índice de ímã × taxa de reprovação (C2) e de atribuição errada das respostas que o usam; e a nota média das respostas apoiadas só no top 20% contra as demais, com IC por bootstrap |

- **Leitura** (a primeira coluna só existe para a demanda rotulada; nos
  goldens, a concentração das perguntas é a do desenho):

  | As perguntas se concentram? | A recuperação se concentra além disso? | Leitura | Ajuste |
  |---|---|---|---|
  | sim | não | Pareto natural da demanda | Acervo enxuto (E4) |
  | não | não | Saudável: a recuperação acompanha as perguntas | Nenhum ajuste de concentração |
  | qualquer | sim (massa fora da fonte alta, ímãs) | Viés do sistema | Embedding (E1), cabeçalho (E3), corte da bibliografia (H5b) |
  | sim | espalha (o_d < e_d nos documentos procurados) | A recuperação dilui o que devia focar | `top_k`, limiar, reranker (E6, E8) |

- **Hipóteses testáveis:**
  - **H5a:** no `pdf-v1`, os documentos que aparecem nas perguntas em PT sem
    serem a fonte (o efeito que tirou o MetaRAG da busca) são ímãs, e deixam
    de ser no `pdf-v1-bgem3` (cruza com o E1).
  - **H5b:** os chunks de bibliografia aparecem como ímã acima do esperado.
    Teste: a versão `pdf-v1-noref` **(novo)** corta só o bloco de referências
    (§6, C0). Antes de rodar, conferir pelo casamento de 25-gramas que
    nenhuma evidência dos goldens cai no trecho removido. Se o ímã some e a
    evidência@4 sobe, o ajuste é cortar a bibliografia.
- **Saída:**
  - curva de Lorenz com **três linhas** (igualdade, esperada e observada);
  - tabela das métricas acima e ímãs com exemplos;
  - o cruzamento concentração × qualidade.
- **Didática:** a curva "esperada" impede a conclusão errada. Parte do 80/20
  vem de como as perguntas foram escritas; só o que passa disso é
  comportamento do sistema.

### E6 — Parâmetros de recuperação (varredura exploratória)

- **Pergunta:** qual combinação de `top_k`, MMR, limiar e chunk serve a este
  acervo?
- **Hipótese:**
  - MMR melhora a cobertura@4 em `fato_distribuido`;
  - um limiar calibrado melhora a abstenção correta com falso bloqueio
    pequeno;
  - chunk menor melhora a precisão@k e piora a cobertura.
- **Braços** (um fator por vez a partir do baseline; interações só entre os
  2 melhores; o cartão marca o E6 como **exploratório**):

  | Tipo | Braços | Sobre | Reindexa? |
  |---|---|---|---|
  | Consulta | `TOP_K` ∈ {2, 8}; MMR com λ ∈ {0,5; 0,7}; `MIN_SIMILARITY_SCORE` ∈ {0,428 (só referência: calibrado em outro índice); recalibrado} | `pdf-v1` | não (`set_settings_overrides`) |
  | Índice | chunk ∈ {400, 1200}; overlap 0 | `pdf-v1-cont` | sim (uma coleção por valor) |

  Os braços de índice rodam sobre `pdf-v1-cont`, e o baseline deles é o próprio
  `pdf-v1-cont` com 800/120. No `pdf-v1`, a janela não atravessa a página, e
  "chunk 1200" fica perto de "um chunk por página".
- **Métricas por tipo de braço:**
  - **`TOP_K`:** evidência@K e precisão@K do próprio braço. Sem MMR, a ordem
    da recuperação não muda com K, então o recall sobe com K por construção.
    A decisão vai para a C2 e para o custo.
  - **Chunk:** comparar com **orçamento de tokens igual** (K = 3200 / chunk:
    8, 4 e 3 trechos), com a curva evidência × tokens de contexto.
  - **Limiar:** abstenção correta (primária) com falso bloqueio ≤ 1 pergunta
    (guarda).
  - **Recalibração do limiar:** uma cópia da `choose_threshold()` (maior F1
    com margem de 0,02 abaixo do menor positivo), aplicada às similaridades
    da C1. Ela é calibrada em `selecao` e avaliada só em `confirmacao` (ou por
    validação cruzada deixando um documento de fora). Calibrada e avaliada
    nos mesmos dados, o falso bloqueio dá zero por construção. Não rodar o
    `calibrate_retrieval_gate.py`: ele só lê a suíte do estudo de mutação e
    sobrescreve `tests/mutation/config/retrieval_gate.json`.
- **Seleção × confirmação:** a varredura usa só `selecao`. O Δ e o veredito
  do braço escolhido saem de `confirmacao`, rodada uma vez. A família do Holm
  é **todos** os braços da varredura (§11).
- **Fixos:** `RAGAS_JUDGE_NUM_CTX` dimensionado pelo braço `TOP_K=8`.
- **Cuidados:**
  - **`num_ctx` com `top_k` maior.** O `auto` usa o `settings.top_k`, e não o
    `top_k` da requisição. O braço muda o `TOP_K` da configuração.
  - **Overlap inválido.** Um overlap ≥ chunk é trocado em silêncio por
    chunk // 4, e o manifesto grava o valor pedido. O manifesto passa a gravar
    também o efetivo.
- **Saída:**
  - tabela da varredura em `selecao`;
  - o veredito do escolhido em `confirmacao`;
  - curva evidência × tokens;
  - AUC do limiar.
- **Didática:** testar 10 configurações e escolher a melhor nos mesmos dados
  engana. A turma vê o vencedor "encolher" na confirmação.

### E7 — Busca híbrida (BM25 + vetorial, fusão RRF)

- **Pergunta:** somar a busca por palavra-chave à busca vetorial ajuda nas
  perguntas com sigla e nome próprio?
- **Hipótese:** o ganho da híbrida é maior com `has_acronym = true` (RAGAS,
  ARES, MetaRAG, Defects4J) do que sem (interação).
- **Braços:** `pdf-v1` (baseline) × `pdf-v1` + híbrida.
- **Métrica primária:** evidência@4, com o teste de interação por
  `has_acronym`.
- **Guardas:**
  - evidência@4 em `has_acronym = false` ≥ baseline − 1 pergunta;
  - latência de recuperação ≤ +200 ms.
- **Precisa construir (novo):**
  - **Ler a coleção inteira.** Um `vector_store.get_all()`, porque hoje não há
    como buscar todos os documentos da coleção (entra na Fase A, §14, porque o
    diagnóstico de FP1 também usa).
  - **Índice BM25 em cache** por coleção, com chave nome + contagem.
  - **Fusão RRF** por `chunk_id`, com k = 60.
  - **Score dos trechos que só o BM25 achou.** Calcular o cosseno pelos
    embeddings (`get(ids=..., include=["embeddings"])`), porque o limiar e o
    MMR dependem do `similarity_score`.
  - **BM25 em Python puro**, para não somar dependência aos quatro arquivos de
    requirements que importam o pipeline em processo (backend, eval, ragas,
    mutation).
- **Saída:** evidência@4 por `has_acronym` e exemplos de trechos que só o BM25
  trouxe.
- **Didática:** busca vetorial entende sentido; BM25 acha a palavra exata.
  Cada uma falha num tipo de pergunta.

### E8 — Reranker

- **Pergunta:** reordenar os candidatos com um modelo que lê a pergunta e o
  trecho juntos melhora o que chega ao gerador e a abstenção?
- **Hipótese:**
  - buscar 20 trechos, reordenar com um *cross-encoder* (por exemplo
    `bge-reranker-v2-m3`) e ficar com 4 sobe a precisão@4 e o MRR;
  - o score do reranker separa melhor as perguntas de fora do que o cosseno,
    que não separou (§3).
- **Braços:** `pdf-v1` (baseline) × `pdf-v1` + reranker.
- **Fixos:** `RETRIEVAL_FETCH_MULTIPLIER=5` com `TOP_K=4` (20 candidatos; o
  padrão 3 daria 12).
- **Métrica primária:** precisão@4 (C1, sem juiz).
- **Guardas e secundárias:**
  - MRR@10;
  - AUC do `rerank_score` para as perguntas de fora, com o portão calibrado
    fora da amostra (como no E6);
  - latência por pergunta (C4).
  
  A `context_precision` do RAGAS é só leitura secundária nos finalistas.
- **Precisa construir (novo):**
  - **Flags** `RETRIEVAL_RERANK` e `RETRIEVAL_RERANK_MODEL`.
  - **Candidatos:** `fetch_k = top_k × RETRIEVAL_FETCH_MULTIPLIER` também com
    o MMR desligado.
  - **Score do reranker:** um campo `rerank_score` novo, mantendo o cosseno em
    `similarity_score`.
  - **Import preguiçoso** (sentence-transformers/torch), para os testes de API
    continuarem herméticos e a imagem Docker leve.
  - **Alternativa sem dependência nova:** usar o LLM como reranker via
    Ollama, que é mais lento.
- **Saída:**
  - precisão@4 e MRR;
  - curva ROC das perguntas de fora (cosseno × reranker);
  - custo em latência.
- **Didática:** recuperar muito e escolher bem é diferente de recuperar pouco.

### E9 (opcional) — Gerador

- **Pergunta:** quanto da qualidade vem do gerador, e a que custo?
- **Braços:** `gemma3:4b` × um gerador que **não seja o juiz** (por exemplo
  `llama3.1:8b`), com o juiz `qwen3:8b` mantido. Se o `qwen3:8b` entrar como
  gerador, o juiz desse braço é ele mesmo (*self-grading*, que o repo evita),
  e o E9 sai como descritivo.
- **Métrica primária:** fidelidade (C2), com a `m_E0` refeita para cada
  gerador.
- **Guarda:** latência ≤ 2× o baseline.
- **Saída:** o ponto de cada gerador no painel custo × qualidade.
- **Didática:** nem todo problema de qualidade se resolve trocando o modelo.

### Prioridade

| Ordem | Experimento | Por quê | Custo de preparar |
|---|---|---|---|
| 1 | E0 | Sem ele, nenhum veredito se sustenta | baixo |
| 2 | E1 | Efeito enorme, já medido, só uma versão nova | baixo |
| 3 | E6 (só C1) | Varredura barata, já suportada pela config | baixo |
| 4 | E5 | Roda sobre E1 e E6; só o H5b pede a versão `pdf-v1-noref` | médio (relatório, rotular a demanda) |
| 5 | E2 | A pergunta principal da turma | médio (conversões, chunker) |

> **Texto incompleto.** A colagem foi cortada aqui, no limite de 50 mil
> caracteres da mensagem. Faltam o resto da tabela de prioridade e as seções
> 9 a 14, citadas acima.
