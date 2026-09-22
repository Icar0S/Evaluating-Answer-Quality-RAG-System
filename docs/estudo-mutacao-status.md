# Estudo de mutação para RAG — relatório de execução

**Artigo:** *Mutation-Based Adequacy Assessment of Test Suites for Retrieval-Augmented Assistants*
**Destino:** Information and Software Technology, special issue VSI:EQUISA · submissão 13/12/2026
**Branch:** `journal-ist` · **Última atualização:** 21/09/2026 (semana 6 em curso — baseline v3 e campanha rodando; proveniência dos logs indexada)

Este documento registra o que foi construído e executado até aqui, as decisões de
projeto com suas razões, e o que já dá para escrever do artigo. Ele existe porque
boa parte das decisões tomadas durante a construção **é conteúdo do artigo** — e
decisão não registrada vira, três meses depois, uma escolha que ninguém sabe
justificar para um revisor.

---

## 1. Onde o estudo está

| Semana | Entrega prevista | Estado |
|---|---|---|
| 1 | Corpus escolhido e versionado; esqueleto de `tests/mutation/` | **concluída** |
| 2 | 30 casos + `golden.jsonl` + `assertions.yaml` | **concluída** |
| 3 | `calibration/` gerado; τ* calibrado; O1 e O2 prontos | **concluída** — τ* = 0,60, e a calibração virou resultado (§4.7) |
| 4 | `apply.py` + catálogo testado (GO/NO-GO) | **concluída** — 18/18 revertem sem resíduo E injetam o defeito declarado (§3.5) |
| 5 | O3, O4, O5 prontos; `run_campaign.py` validado | **concluída** — O4 estável; O3 inviável com juízes locais (§4.9); três decisões abertas (§6) |
| 6 | Baseline (300 inv.) + campanha (2.700 inv.) | **em curso** — baseline v3 fechado (|S| = 26/22/20/20, portão verde); campanha dos 18 mutantes desde 22/09 10:51, 21% em 16:45 |
| 7–13 | Coerência, RQ3, escrita, submissão | não iniciadas |

O cronograma está **adiantado**: o portão GO/NO-GO da semana 4 fechou junto com a
semana 2, porque calibrar o piso de recuperação (pré-requisito do operador R4)
exigia os casos prontos e os casos ficaram prontos antes do previsto.

### Portões, estado atual

```
validate_suite (sem --allow-draft)   VERDE — 30 casos, 0 rascunhos
apply verify                         VERDE — 18/18 operadores sem resíduo
verify_mutants (fidelidade)          VERDE — 18/18 injetam o defeito declarado
validate_suite --check-retrieval     VERDE — 19/19 factuais recuperam o documento da evidência
assertivas vs. própria referência    VERDE — nenhuma se auto-reprova
tests/api (regressão do SUT)         VERDE — 18 testes
```

---

## 2. O que foi construído

56 arquivos versionados em `tests/mutation/`: ~7.850 linhas de Python e ~2.000 de
configuração e dados. Mais 264 linhas de mudança no SUT (`backend/app/`).

### 2.1 Mudanças no SUT

Os operadores de mutação mutacionam **configuração**, e parte dos parâmetros não
existia no pipeline. Todos entraram com defaults que preservam o comportamento
anterior: o system prompt v1 sai byte a byte idêntico e os 18 testes de
`tests/api/` continuam passando.

| Arquivo | Mudança | Default | Serve a |
|---|---|---|---|
| `config.py` | overlay de configuração em memória | vazio | todo o harness |
| `retrieval.py` | `min_similarity_score` passa a ser aplicado (era knob morto) | `0.0` | R4 |
| `retrieval.py` | MMR opcional na seleção final | desligado | R3 |
| `generation.py` | system prompt montado de blocos ligáveis | = prompt v1 | P1, P2, P3 |
| `generation.py` | `generation_temperature` / `generation_seed` | `None` | P4, reprodutibilidade |
| `ingestion.py` | `chunk_boundary_offset_tokens` | `0` | K4 |
| `ingestion.py` | `embed_texts(model=...)` | segue configuração | O1 (espaço fixo) |
| `vector_store.py` | `reset_client()`, `query(include_embeddings=)` | — | troca de índice, MMR |

A decisão de fundo: **reconfigurar o SUT sem editá-lo**. O harness aplica um
delta sobre `Settings` em memória e descarta no fim; nenhum arquivo de
`backend/app/` é reescrito durante a campanha. Isso elimina a classe de falha
mais cara de diagnosticar depois — o mutante que continuou aplicado após o
revert.

### 2.2 O harness

```
config/study.yaml        configuração do estudo (o "Study Context" executável)
config/codebook.yaml     códigos de coerência, a congelar antes de ver resultado
corpus/                  normalização dos PDFs, variantes prev/conflict, manifesto
operators/catalog.yaml   os 18 operadores com camada, ponto de falha e patch
operators/apply.py       aplica/reverte idempotente + portão `verify`
suite/                   30 casos, assertivas, validação (com --check-retrieval)
  draft_cases.py           rascunha casos a partir do corpus indexado; --recheck sem GPU
  promote_drafts.py        promove rascunho conferido para golden.jsonl, com bloqueios
  build_assertions.py      deriva assertivas e RODA O2 contra a própria referência
calibration/             pares rotulados por construção, varredura de τ, conferência
  review_pairs.py          apoio à conferência visual de 100% dos pares EQUIV (§6.3)
  diagnose_o1.py           o que o cosseno vê e não vê, por tipo de erro e por modelo
oracles/                 O1 cosseno, O2 assertivas, O3 RAGAS/deepeval, O4 juiz, O5 conjunção
runner/                  ponte com o SUT, custo (GPU/Wh por NVML), JSONL, caminhos
tools/
  audit_roles.py           papéis do corpus servem aos operadores C1-C4 e E2?
  audit_licenses.py        o corpus pode ir para o Zenodo?
  verify_mutants.py        cada mutante injeta o defeito declarado? (§10, executável)
  calibrate_retrieval_gate.py  piso de similaridade do baseline
  make_synthetic_results.py    dados falsos para validar a análise sem GPU
run_campaign.py          baseline + mutantes -> runs.jsonl
evaluate.py              oráculos sobre o log -> verdicts.jsonl
coherence.py             classificação de coerência + κ intra-avaliador
analyze.py               10 tabelas + 4 figuras + cost.jsonl
stats.py                 Wilson, bootstrap, Cochran Q, McNemar, KW, Dunn, Holm, κ
```

Três separações estruturais, todas com consequência:

**Executar ≠ julgar.** `run_campaign.py` só produz `runs.jsonl`; os vereditos
saem de `evaluate.py`, sobre o log. Recalibrar τ*, trocar o backend do O3 ou
reexecutar o juiz não custa uma hora de GPU a mais. É também o que permite
responder "o que L1 teria detectado?" sem reexecutar o SUT.

**Corpus ≠ índice.** O harness mantém `work/corpus` (o que o corpus é) e
`work/index_src` (o que foi indexado). Só o operador E2 os faz divergir — é
exatamente o defeito que ele injeta.

**Medir ≠ decidir.** As ferramentas de auditoria imprimem evidência e só reprovam
o caso inequívoco. Ver §4.3.

---

## 3. O que foi executado

### 3.1 Corpus (semana 1)

8 documentos, 96 páginas, ~164 mil tokens, construídos a partir de
`data/source_pdfs/`.

O construtor **re-renderiza todos os documentos** a partir do texto extraído e
repagina por orçamento de tokens (~1.800/página). As duas decisões têm razão
metodológica:

- se só os mutantes fossem re-renderizados, a diferença de extração entre
  original e mutante entraria no resultado junto com o defeito injetado;
- o SUT quebra chunks por página; com páginas de ~500 tokens (artigo em duas
  colunas), um chunk de 1.200 engoliria a página inteira e **K1–K4 seriam
  inertes por construção**. Com a repaginação, o índice tem 184 chunks para 96
  páginas — 1,9 chunk por página, e os quatro operadores de chunking passam a ter
  efeito observável.

Papéis fixados em `corpus/sources.yaml`:

| Papel | Documento | Critério |
|---|---|---|
| `primary` | does_prompt_engineering | 275 valores factuais, 40 perturbados pelas variantes |
| `secondary` | metarag | C3 apaga as páginas 7-9, e as três são prosa com 65 fatos |
| `recent` | evaluator_bias + qaquest | os dois menores com fato suficiente |

Variantes `prev` e `conflict` do documento `primary`: 40 substituições cada,
distribuídas pelas 10 páginas, do tipo `16 Java classes`, `32% de cobertura`,
`Mutation Score de 89.5%`.

### 3.2 Suíte de 30 casos (semana 2)

| Classe | Casos | Papel da evidência |
|---|---|---|
| fato direto | 6 | primary (2), secondary (1), recent (2), qualquer (1) |
| fato distribuído | 5 | primary (1), secondary (1), qualquer (3) |
| filtro condicional | 4 | primary (2), qualquer (2) |
| fora do corpus | 5 | nenhuma — deve se abster |
| fora de escopo | 3 | nenhuma — deve recusar |
| ambíguo | 3 | nenhuma — deve pedir esclarecimento |
| rastreabilidade | 4 | primary, secondary, recent, qualquer |

Cobertura por operador (quantos casos miram cada um): C1=4, C2=2, C3=3, C4=1,
K1=5, K2=4, K3=3, K4=5, E1=7, E2=3, R1=10, R2=7, R3=4, R4=5, P1=11, P2=7, P3=4,
P4=5. Nenhum operador fica sem caso capaz de matá-lo.

Cada resposta de referência foi conferida contra o trecho de origem no índice.

### 3.3 Calibração do piso de recuperação

`min_similarity_score = 0.428`, gravado em `config/study.yaml`. Com ele, `apply
verify` passa nos 18 operadores — R4 deixa de ser mutante equivalente por
construção.

### 3.4 Calibração de τ* (semana 3)

Conjunto final: **247 pares** (111 EQUIV, 136 ERRO) sobre 22 casos. Os oito casos
cuja referência é a frase fixa de abstenção ficaram de fora: eles não produzem
ERRO — não existe versão "errada plausível" de uma abstenção — e suas paráfrases
seriam variantes de uma única sentença, pares quase idênticos que mediriam a
capacidade do cosseno de reconhecer uma frase fixa, não de julgar uma resposta.

Conferência visual de 100% dos 115 pares EQUIV, como o §6.3 exige: **111
aprovados, 4 rejeitados**. As rejeições não foram de fato perdido, e sim de
degeneração (`"O Pitest (Pitest) foi adotado"`) e de relação alterada
(`"estável entre sementes"` virando `"estável quando comparado com sementes"`).

τ* = 0,60, e o resultado da calibração virou achado — ver §4.7.

Limitação a declarar, espelho da que o protocolo já previa: as paráfrases saíram
conservadoras (troca de verbo e conectivo, mesma estrutura). Somada aos erros
programáticos da classe ERRO, **τ\* é limite superior pelos dois lados**, não só
por um.

### 3.5 Fidelidade dos mutantes (semana 4)

`apply verify` prova que os 18 operadores aplicam e revertem sem resíduo. Não
prova que o mutante **é** o que o catálogo diz. O §10 (validade interna) pede
revisão manual de 100% dos 18 antes da campanha; `tools/verify_mutants.py` é
essa revisão em forma executável — aplica cada operador, observa o efeito no
ponto do pipeline que ele deveria alterar, e grava `operators/fidelity_report.json`.

| Op. | Evidência observada |
|---|---|
| C1 | `primary` ausente de `work/corpus` e de `work/index_src`; 7 documentos restantes |
| C2 | 40/40 substituições da variante `prev` visíveis no texto; índice recebe a variante |
| C3 | `secondary` 9 → 6 páginas, no corpus e no índice |
| C4 | `__dup` presente nos dois diretórios, original preservado, conteúdo divergente |
| K1 | chunk 400 → 790 chunks (baseline 184) |
| K2 | chunk 3000 → 96 chunks = 1 por página |
| K3 | sobreposição na fronteira: 913 caracteres no baseline, **0** no mutante |
| K4 | tamanho dos 3 primeiros chunks: [6206, 4512, 6316] → [3131, 6333, 2630] |
| E1 | dimensão do vetor 768 → 384 |
| E2 | os 2 `recent` presentes no corpus e **ausentes** do índice |
| R1 | 1 chunk devolvido |
| R2 | 12 chunks devolvidos |
| R3 | MMR desligado |
| R4 | pergunta abaixo do piso: baseline devolve **0** chunks, mutante devolve 4 (mín. 0,3995) |
| P1 | regra de ancoragem ausente, abstenção preservada |
| P2 | regra de abstenção ausente, ancoragem preservada |
| P3 | regra de citação ausente, ancoragem preservada |
| P4 | `temperature: 0.8` nas opções enviadas ao Ollama |

Duas sondas precisaram ser endurecidas depois da primeira rodada, e o motivo é
o mesmo achado de sempre — checagem que passa pelo motivo errado é pior que
nenhuma:

- **R4** usava a primeira pergunta "fora do corpus" da lista, cujo melhor chunk
  (0,492) já passa do piso de 0,428: baseline e mutante devolviam 4 chunks e a
  checagem passava sem observar nada. A sonda agora é a pergunta negativa de
  menor similaridade (0,418), onde o baseline devolve zero.
- **K3** era conferido por contagem de chunks, que não muda ao tirar a
  sobreposição — o que muda é o conteúdo. Agora mede quantos caracteres do fim
  do 1º chunk reaparecem no início do 2º.

### 3.6 Piloto (semana 5)

Baseline + R1 + P2 sobre três casos (c01 factual/primary, c07 distribuído, c16
abstenção), 2 execuções cada: 18 invocações, 45 vereditos, cinco oráculos
ativos. O caminho `run_campaign → evaluate` funciona de ponta a ponta com custo
medido por NVML (GPU-s e Wh por invocação). Quatro achados:

**Perguntas que não recuperam a evidência.** `"Quantas classes foram
selecionadas no total?"` (c01) trouxe quatro documentos e nenhum era o `primary`.
Num corpus onde todo artigo fala de classes, pergunta genérica recupera o
vizinho errado; o caso reprovaria no baseline e sairia de S — e é a âncora de
C2/C4. `validate_suite --check-retrieval` confere isso para cada caso factual
(uma chamada de embedding por caso) e achou mais dois. Os três foram reescritos
com termos que só o documento tem; **19/19 recuperam**.

**O juiz tratava a referência como máximo.** O4 reprovava o baseline de c07 por
"adiciona informações não presentes na referência". O juiz não vê o documento,
então não tem como saber se um detalhe a mais é verdadeiro — só pode julgar
contradição. O prompt agora diz que a referência é o mínimo. Controles depois
da mudança: número errado e mutador trocado continuam reprovando, resposta curta
correta passa. c07 continua reprovando, e relendo, com razão: a pergunta pede
*quais mutantes* e o SUT respondeu com categorias de fraqueza sem nomear nenhum.

**O SUT alucina fora do corpus.** c16 (WCAG, deveria se abster) recebeu
`"A WCAG 2.2 define 14 critérios de sucesso no nível AA"` com contexto que não
menciona WCAG. Não é defeito da suíte — é o sistema não-mutado violando a
própria regra de ancoragem. Pelo §7.1 o caso sai de S. Se os cinco "fora do
corpus" reprovarem no baseline, P2 fica sem caso avaliável não porque a suíte é
fraca, mas porque **a abstenção já está quebrada antes de qualquer mutação**.
Decisão pendente: aceitar e reportar como achado sobre o SUT, ou afastar as
perguntas do corpus (o que testaria menos o que um usuário real perguntaria).

**O3 é inviável com juiz de 4B.** `ragas` não parseia a saída do gemma3:4b em 6
de 9 avaliações. `deepeval` avalia tudo, mas dá faithfulness **0,0** para a
resposta ancorada de c07 e **1,0** para a alucinação de c16 — sintoma de zero
afirmações extraídas, que a métrica lê como verdade vazia. O problema não é o
backend: métrica RAGAS exige extrair afirmações e verificar cada uma contra o
contexto, e um modelo de 4B não sustenta isso.

Diagnóstico com juiz de 8B (qwen3, só para isolar a causa): faithfulness
**1,0 para a alucinação de c16** de novo, e 4 a 9 minutos por avaliação. Não é
tamanho. É a definição da métrica no `deepeval` — "afirmações que não
*contradizem* o contexto" — pela qual inventar algo sobre um tópico ausente
não conta como infidelidade. O `ragas` é mais estrito (a afirmação precisa ser
*inferível* do contexto), mas não parseia a saída de nenhum juiz local
disponível.

**Decisão pendente sobre O3**, entre três opções:

1. Aplicar a contingência do §9 e rodar com quatro oráculos (O1, O2, O4, O5). O
   protocolo já previa essa saída para O4; o caso é simétrico.
2. Baixar um juiz de outra família sem modo de raciocínio (ex.: llama3.1:8b) e
   retestar o `ragas`. Custa um download de ~5 GB e um novo piloto.
3. Manter O3 via `deepeval` e reportá-lo como degenerado. Diferente do O1, a
   degeneração aqui é da implementação disponível, não do método — o que
   enfraquece o que se pode afirmar.

**Custo real.** 27 s por invocação no baseline, com 812 tokens de saída em média
— a maioria é raciocínio do qwen3 antes da resposta. Projeção da campanha
completa: **22,6 h** de parede e ~1,1 kWh, contra 8,5 h estimadas no protocolo.
Com a regra de corte do §9 (20 casos): 15 h. O harness é retomável, então três
noites resolvem; mas é decisão de cronograma.

**O4 é estável.** 5/5 de concordância em todos os controles e em todos os pares
do piloto, a temperatura 0. O ponto de decisão da semana 5 (§9) está respondido:
o juiz fica, e é ele que pega o que o O1 não vê — número trocado sai como
`fail / wrong_value`.

### 3.7 Semana 6 — decisões e início da campanha (21/09)

As três decisões do §6 foram tomadas antes da primeira invocação, e registradas
onde o harness as lê:

| Decisão | Tomada | Onde está |
|---|---|---|
| O3 | **Contingência do §9**: campanha com O1, O2, O4, O5. Como executar ≠ julgar, o log fica íntegro e O3 pode ser rejulgado sobre ele se um juiz viável aparecer (`evaluate.py --oracles O3`) | `config/study.yaml`, `oracles.enabled`, com o motivo em comentário |
| Classe "fora do corpus" | **Aceitar**: decidido com dados (abaixo), não por preferência | §3.7 e artigo §6.7/§7.1 |
| Duração | **Campanha completa**, sem a regra de corte | — |

Antes de qualquer resultado: o piloto foi arquivado em `results/pilot/` (o run de
c01 usava a pergunta antiga e teria sido pulado pela retomada) e o codebook foi
congelado (`frozen_at: 2026-09-21`).

**Baseline v1 (300 invocações, 21/09 17:21–19:41) — inválido, e por quê.**
0 erros de execução; 28 s e 875 tokens de saída por invocação; 22,8 GPU-s e
0,475 Wh. Conjunto avaliável: |S| = O1 23, O4 11, O2 6, O5 6. Seis casos sob o
oráculo primário eram um alarme, e a leitura dos vereditos apontava para o SUT:
"o texto não menciona" com a evidência no contexto (c01, c08, c14), respostas
por conhecimento paramétrico nos 5 casos "fora do corpus", nenhuma citação nos
4 de rastreabilidade, nenhum pedido de esclarecimento nos 3 ambíguos.

A causa real estava no log do servidor Ollama, não nos vereditos: **288 avisos
`truncating input prompt limit=2050 prompt=5255 keep=4`**. Sem `num_ctx`
explícito o servidor usa 4.096 e trunca o prompt em `num_ctx/2` tokens,
mantendo os 4 primeiros e os 2.046 últimos — ou seja, descarta o **system
prompt inteiro** (ancoragem, abstenção, citação) e o começo do contexto. Todas
as 300 invocações do baseline, e as 18 do piloto, rodaram assim. Nada falhou,
nada avisou fora do log do servidor. Confirmado com um prompt de controle: a
regra "comece com SENTINELA" no system prompt some sem `num_ctx` e volta com
ele.

Consequências, na ordem em que importam:

- **P1, P2 e P3 seriam inertes por construção** — removem regras que o modelo
  nunca via. R2 e K2 (mais contexto) apenas truncariam mais. A campanha inteira
  estaria confundida.
- As conclusões sobre o SUT tiradas do v1 estão **retiradas**: com o contexto
  íntegro, c01 responde "16 classes, 8 do Defects4J e 8 do TestBench"; c16
  abstém com o marcador exato; c27 dá H = 4,223 e p = 0,377. O que restava era
  formulação da pergunta e a instrução de citação.
- A decisão "aceitar a classe de abstenção como está" foi tomada sobre dados
  truncados e **volta a ser aberta**: decide-se com o baseline v2.

**Correção (SUT, parametrizável, default preserva o comportamento):**
`generation_num_ctx: auto` dimensiona a janela pela configuração de recuperação
de cada mutante — `ceil_1024(top_k × chunk_size × 1,25 + 1.024 + num_predict)`,
com `generation_num_predict: 4096` explícito porque o limite de prompt do
servidor é `num_ctx − num_predict` (maior `tokens_out` observado: 3.124). Um
`num_ctx` fixo que coubesse o pior caso (R2: 12 chunks, 15.250 tokens medidos)
forçaria offload para CPU em todas as invocações; dimensionar por mutante mantém
o baseline a 11.264 (1–2 camadas na CPU, ~25 tok/s) e paga o offload só em K2
(20.480) e R2 (23.552, ~200 s por invocação). O harness agora converte
`prompt_tokens ≥ limite` em erro auditável (`context_truncated`). Flash
attention e KV q8_0 foram testados e descartados: prompt eval 5× mais lento e
OOM nesta GPU (6,5 GiB disponíveis dos 8).

**Duas outras correções antes do v2**, ambas de instrumento e ambas antes de
qualquer mutante:

- **Portão de recuperação em nível de página.** Diagnóstico só com embeddings
  (independente da truncagem): em 10 dos 19 casos factuais o documento certo
  vinha no top-4 e a página da evidência não; com k = 12, 15/19; com MMR
  desligado, 9/19 — é formulação. 8 perguntas reescritas com termos do trecho
  (c02, c03, c06, c07, c09, c10, c11, c12; pergunta anterior em `notes`);
  `validate_suite --check-retrieval` passa a exigir a página. c04, c05 e c15 não
  têm reformulação natural que vença as páginas vizinhas do mesmo documento:
  `accepted_gap: retrieval-page`, saem de S pelo §7.1.
- **Instrução de citação.** Com "[documento, pág. N]" o modelo copiava a palavra
  "documento" literalmente — regra que o baseline não cumpre torna P3
  equivalente por construção (o mesmo princípio do piso e do MMR). A regra, que
  não faz parte do prompt v1 de produção, passou a nomear o arquivo e dar um
  exemplo; c24/c27/c28 citam `[arquivo.pdf, pág. N]`.

Tudo arquivado (`results/archive_v1_truncated/`) e refeito. Baseline v2
iniciado às 21:08 com a configuração final.

**Baseline v2 (300 invocações, 21/09 21:08–23:23) — válido.** 0 erros, 0
truncagens; 27 s por invocação (mediana 22,7), 4.151 tokens de prompt em média
(máx. 5.780, limite 7.168), 523 de saída (o v1 truncado dava 875: com as
regras na frente o modelo é mais conciso), 16,3 GPU-s e 0,368 Wh. Portão
`--check-baseline` verde. Conjunto avaliável, depois de três ajustes de
instrumento feitos sobre os vereditos e antes de qualquer mutante:

| Oráculo | \|S\| v1 (truncado) | \|S\| v2 |
|---|---|---|
| O1 cosseno | 23 | **25** |
| O4 juiz | 11 | **23** |
| O2 assertivas | 6 | **21** |
| O5 = O1 ∧ O2 | 6 | **21** |

Os 9 casos fora de S sob O5, com o motivo de cada um: c04, c05, c15 (página da
evidência não vem — lacunas aceitas), c14 (o RAGChecker tem dez tabelas quase
idênticas e o chunk com 93,7 não vence as outras — `accepted_gap:
retrieval-chunk`), c07 (o modelo lista dois dos três mutadores — incompleto,
FP7, é o SUT), c24–c26 (ambíguos: o SUT **abstém** em vez de pedir
esclarecimento — o prompt não tem regra de esclarecimento; a classe sai de S e
isso é achado sobre o SUT), c28 (copiou o exemplo `[relatorio_x.pdf, pág. 7]`
da regra de citação — o SUT; c27, c29, c30 citam certo e P3 tem 3 alvos).

Os três ajustes: (i) `normalize()` do O2 passou a igualar separador decimal
entre dígitos — c03 respondia "3.79%" e reprovava contra "3,79%"; (ii) as
assertivas de c13 usavam a flexão exata ("configuração" não casa em
"configurações") e a regex de condição não tinha "com base nos" — radicais e
alternativas adicionados; (iii) c02 recuperava a página certa e o **chunk
errado** (p5::c1) e abstinha 10/10 — pergunta reescrita de novo e o portão
passou a conferir o **valor** no contexto recuperado, não só a página.

**A classe de abstenção resolveu-se com dados:** c16–c20 abstêm 10/10 com o
marcador exato. A classe fica, P2 e R4 são avaliáveis. A decisão do §6 está
fechada.

**Repetição não é amostra independente.** Em 29 casos, 14 têm as 10 respostas
idênticas e 15 têm r2 = r3 = … = r10 byte a byte com r1 diferente — e a
diferença não é ruído numérico: são reescritas (razão de similaridade 0,38 a
0,80, mesmos fatos). Sondagem com 6 invocações extras: a semente não muda nada
(seeds 1, 1, 2 e 7 idênticas); dentro de uma mesma carga do modelo, prompt
idêntico → resposta idêntica; e a resposta depende do **prompt anterior no slot
do servidor** (reuso de prefixo do cache KV): em ordem fixa, r2..r10 de um caso
têm sempre o mesmo predecessor. Dez repetições em ordem fixa amostram **dois**
estados. Correção antes da campanha: os casos são embaralhados por repetição
(semente = índice da repetição), para que cada invocação tenha um predecessor
diferente. O baseline foi refeito assim (v3, iniciado 22/09 00:05) e a v2 ficou
em `results/archive_v2_fixed_order/` para comparar |S| entre as duas ordens —
um dado para o §7.1 e para Atil et al.

**Para o artigo:** a truncagem é o achado §4.13 — FP3 por construção, sem
erro, sem aviso fora do log do servidor, e invisível a todos os cinco oráculos
(eles julgam a resposta, não o prompt). É o argumento mais forte do estudo para
"verifique o instrumento contra o baseline antes de medir": três semanas de
instrumentação cuidadosa e o defeito estava no parâmetro que ninguém declarou.

### 3.8 Semana 6 — a noite de 21 para 22/09 (auditoria de replicação)

A cadeia `baseline v3 → evaluate → portão → campanha → evaluate` foi lançada às
00:05. Estado às 09:40 de 22/09: **baseline v3 em 219/300**, 0 erros. O atraso
não é do harness — a máquina **suspendeu** às 00:41 e voltou às 08:01 (o log do
Ollama salta de 00:00:58 para 08:01:17). Windows conta ociosidade por entrada do
usuário, não por carga de GPU, e o plano de energia tinha suspensão em 30 min no
AC. Corrigido para "nunca" enquanto a campanha roda (`powercfg /change
standby-timeout-ac 0`; restaurar com `1800`).

Duas anomalias de medição ficaram registradas, e as duas viraram guarda no
código:

- **`baseline-c18-r1` atravessou a suspensão**: 26.463 s de relógio de parede e
  58,4 Wh medidos por NVML (contra 27 s e 0,37 Wh típicos) para uma resposta de
  203 tokens. A resposta é válida; a medição de custo, não. `runner/sut.py`
  passa a marcar `wall_clock_anomaly` quando o relógio passa 1,2× o timeout do
  cliente — só a suspensão produz isso, porque o httpx aborta no timeout.
- **6 execuções de c14 bateram o teto de geração** (`num_predict` 4.096) com
  **resposta vazia**: o qwen3 gastou todos os tokens no raciocínio e não chegou
  a responder. Nenhuma ocorrência no baseline v2 (ordem fixa) — foi o
  embaralhamento que levou c14 a essa trajetória, o que é mais uma evidência do
  achado sobre repetição. Reprovar uma resposta vazia atribuiria ao mutante um
  efeito do **nosso teto**: `evaluate.py` passa a excluir do julgamento as
  execuções com resposta vazia E `tokens_out ≥ num_predict`, reportando quantas
  foram; `runner/sut.py` as marca como `generation_cap` no próprio log.

**Desfecho do baseline v3 (10:20).** 300 invocações, 616 min de relógio (dos
quais 7,3 h de suspensão). Portão verde depois de duas correções de contagem:

| Oráculo | v2 (ordem fixa) | v3 (embaralhado) |
|---|---|---|
| O1 cosseno | 25 | **26** |
| O4 juiz | 23 | **22** |
| O2 assertivas | 21 | **20** |
| O5 = O1 ∧ O2 | 21 | **20** |

As duas ordens dão |S| a um caso de distância — a diferença está em quais casos,
não em quantos, e é material para o §7.1.

Mais duas correções de instrumento, ambas sobre execuções que **não são
comportamento do SUT**:

- **500 do servidor** em `baseline-c23-r10`: outra aplicação tomou a VRAM
  (o log do Ollama mostra 625 MiB livres) e o `llama-server` não subiu. O caso
  c23 (recusa) passou a parecer "respondeu 1/10" porque o portão contava a
  execução com erro. `check_baseline` passa a excluí-las, como o `invoke` já
  documentava, e avisa quantas foram.
- **Execução com erro nunca era refeita**: a retomada pula `run_id` já
  presente, o que é certo para uma campanha interrompida e errado para uma
  invocação que falhou. `run_campaign --retry-errors` remove do log as
  execuções com erro e as refaz — explícito por opção, nunca automático, e o
  que ele apaga aparece no log da campanha.

**Campanha dos 18 mutantes iniciada às 10:51** (2.700 invocações), com
`--retry-errors` e `evaluate` encadeados.

**Índice de proveniência.** Os logs de execução não são versionados (são
grandes e regeneráveis), mas até aqui também não havia registro de qual log era
qual. `results/PROVENANCE.md` (versionado, gerado por
`tools/provenance.py`) passa a dizer, para cada um: quando rodou, com que
commit, o que é, por que foi superado, quantas linhas, energia total e
**sha256**. Cinco logs indexados: piloto, baseline v1 (truncado), parcial v1,
baseline v2 (ordem fixa) e o v3 em curso. Nenhum foi apagado — um log inválido é
a evidência da ameaça que ele revelou.

### 3.9 Campanha — andamento e primeiro achado (22/09, 16:45)

**569/2.700 invocações (21%)** desde as 10:51, 10 com erro — todas
`generation_cap` (resposta vazia no teto de 4.096 tokens), nenhuma de
infraestrutura. Ordem: os mutantes de configuração primeiro (reaproveitam o
índice do baseline), os de corpus no fim.

| Mutante | Invocações | tokens de prompt (mediana) | s/invocação | total |
|---|---|---|---|---|
| E1 | 150/150 | 302 | 11 | 39 min |
| K1 | 150/150 | 1.997 | 16 | 57 min |
| K2 | 150/150 | 8.227 | 64 | 3,6 h |
| K3 | 118/150 | 4.400 | 22 | ~59 min |

O tempo por invocação é quase constante até o prompt caber na VRAM (manda a
geração da resposta) e dispara depois — o joelho está entre 4.600 e 8.200
tokens, onde o modelo passa a processar parte do prompt na CPU. Isso projeta
**R2 (top-k 12) em ~130 s por invocação, 5,4 h sozinho**, e **17,4 h para o
resto da campanha**, mais ~3,3 h de avaliação. `tools/progress.py` recalcula
isso a qualquer momento, sem tocar na GPU.

**E1 não mede o que o catálogo diz — e isso é achado.** Trocar
`nomic-embed-text` por `all-minilm` deveria degradar a recuperação; o que
acontece é que ela **para**: 95 das 150 invocações recuperam **zero chunks**
(prompt mediano de 302 tokens, contra 4.646 do baseline). A causa é o piso de
similaridade: 0,428 foi calibrado para a escala do `nomic`, e a do `all-minilm`
é outra. O piso não é propriedade do corpus, é do **par corpus/modelo**, e uma
troca de modelo de embedding — atualização de rotina em produção — invalida o
piso silenciosamente.

Para o artigo isto tem duas consequências. Primeira: E1 será morto por quase
todo caso, e o mecanismo precisa ser reportado, senão o leitor entende
"degradação de recall" onde o que houve foi piso descalibrado. Segunda, e mais
forte: é o mesmo defeito do §4.2 visto do outro lado — lá o piso não barrava
pergunta fora do corpus, aqui barra evidência legítima. **Um piso global de
similaridade é frágil nas duas direções**, e a fragilidade só aparece quando se
muda uma peça que parece não ter relação com ele.

---

## 4. Achados metodológicos — material para o artigo

Esta seção é a mais importante deste documento. São descobertas feitas durante a
construção, **não previstas no protocolo**, e várias são transferíveis para quem
replicar o estudo.

### 4.1 Operador pode ser equivalente por estrutura do corpus, não por desenho

O operador C3 trunca 30% do documento `secondary`. Em artigo de conferência, o
terço final é quase sempre a lista de referências — e **nenhum caso de teste cita
bibliografia**. Dos 8 documentos do corpus, a medição inicial encontrou apenas um
com prosa no trecho que C3 apaga.

Isso generaliza: a observabilidade de um operador de corpus depende da estrutura
interna do documento em que ele opera, e essa dependência não aparece no
catálogo. Um catálogo de operadores para RAG precisa vir acompanhado de um
critério de adequação do corpus, sob pena de reportar "a suíte não detecta"
quando o correto é "o desenho não permitia detectar".

**Onde entra:** §4 (Mutation Operators for RAG), subseção de equivalência.

### 4.2 Corpus homogêneo derrota o piso de relevância

Medida a similaridade do melhor chunk recuperado para os 30 casos: **5 dos 8
casos negativos** (fora do corpus / fora de escopo) ficam **acima** de qualquer
piso que preserve todos os positivos.

O caso mais claro foi diagnóstico: uma pergunta sobre "a taxa de mutantes
equivalentes que o PIT reporta no Defects4J" tinha a **maior similaridade de toda
a suíte** (0,656, acima de todos os positivos), porque o documento `primary` é
exatamente sobre PIT, Defects4J e mutantes sobreviventes. A pergunta é sobre um
valor ausente, mas o *tópico* está densamente coberto.

Consequência para o estudo: a classe "fora do corpus" exercita muito mais P2
(instrução de abstenção) do que R4 (piso de relevância). Consequência para a
prática: **num corpus topicamente homogêneo, um piso global de similaridade não
distingue "coberto" de "não coberto"** — é propriedade do par corpus/modelo de
embedding, não da formulação das perguntas.

**Onde entra:** §7.2 (Results — oracle bias) como hipótese a confirmar, §8
(Discussion) como implicação prática, §10 (Threats) como ameaça de constructo.

### 4.3 Os dois erros de calibração não custam o mesmo

Ao escolher o piso de similaridade, a primeira implementação pegava "o maior
limiar com F1 máximo" — e ele caiu **exatamente** sobre o positivo mais baixo
(0,488 contra um caso em 0,488).

Os erros são assimétricos:

- piso que barra um positivo derruba o caso no **baseline**, e caso que reprova
  no baseline sai do conjunto avaliável (§7.1): some do denominador das três RQs
  sem quebrar nada e sem avisar ninguém;
- piso que deixa passar um negativo apenas transfere o trabalho para P2, o que é
  **observável no resultado**.

A escolha passou a manter margem de 0,02 abaixo do menor positivo, mesmo custando
F1. O mesmo raciocínio vale para τ* e deve ser declarado.

**Onde entra:** §6.3 (calibração como instrumento), §6.4 (oráculos).

### 4.4 Assertiva que reprova a própria referência

Ao derivar as assertivas determinísticas das respostas de referência, três casos
ambíguos tinham assertivas que **falhavam na própria resposta de referência** — a
resposta descrevia o pedido de esclarecimento em vez de ser um. Uma assertiva
assim reprova o baseline, e o caso sai do conjunto avaliável em silêncio.

Foi adicionada uma verificação: o gerador roda o oráculo O2 contra a referência e
descarta a assertiva que ela não cumpre. Também apareceu um falso positivo do
tipo oposto — a assertiva de condição do caso c13 passava porque `"se "` casava
dentro de `"base em"`, agora corrigido com fronteira de palavra.

**Lição transferível:** num estudo de mutação com oráculo escrito à mão, o
oráculo precisa ser validado contra o baseline antes da campanha. Assertiva que
aprova por acaso é tão ruim quanto a que reprova por acaso.

**Onde entra:** §6.4 (O2) e §7.4 (procedimento de validade).

### 4.5 Rascunho automático inventa justificativa, não número

Os 19 casos dependentes do corpus foram rascunhados com o modelo local a partir
dos chunks indexados. Conferidos um a um contra a fonte: **nenhum número
inventado** em quatro rodadas. O vício é outro — o modelo anexa justificativa
própria a um fato correto ("93,7, *pois sua configuração otimiza a fidelidade*",
onde só o 93,7 está no documento).

Isso importa porque a resposta de referência **é o oráculo**: claim inventado ali
penaliza resposta correta do SUT. Seis casos foram reescritos para conter apenas
o literal.

Um caso ilustra o limite do automático: a frase-fonte sobre a definição de
Pareto-ótimo está **truncada pela legenda de uma figura** no corpus normalizado,
e o modelo completou a definição de cabeça. Números conferiam; a afirmação não
estava no documento.

**Onde entra:** §6.1 (suíte) como procedimento de construção, §10 (Threats).

### 4.6 Falhas silenciosas de pipeline

Três bugs que não quebravam nada e só apareciam ao ler a saída:

1. **Índice obsoleto.** Reconstruir o corpus não invalidava o índice, porque a
   assinatura compara o conteúdo de `work/index_src`, que não era atualizado. Um
   rascunho citou `(?= 0.377)` horas depois de o corpus já conter `(p= 0.377)`.
   Ironia útil: é a mesma falha que o operador E2 injeta de propósito.
2. **Normalização destruindo o corpus.** As fontes Base-14 do PDF só representam
   latin-1; 1.372 símbolos viravam `?`, sendo 677 num único documento. Notação
   estatística é justamente o que os casos factuais citam. Recuperados 93% com
   NFKC, tabela de grego/operadores e decomposição por caractere.
3. **Perturbação na bibliografia.** As 40 substituições das variantes caíam quase
   todas em marcadores de citação (`[13]`, `[15, 36]`), porque a cota era
   consumida pela primeira página, onde a densidade de citações é maior. C2 e C4
   ficariam sem âncora possível na suíte.

**Onde entra:** §3 (Study Context) para a normalização do corpus; o resto como
nota de replicação no pacote do Zenodo.

### 4.7 O oráculo de cosseno ordena pelo avesso

Este é o achado mais forte até aqui, e ele apareceu **antes da campanha**, na
calibração de τ* (§6.3).

O procedimento do protocolo — varrer τ ∈ [0,60; 0,95] e escolher o F1 máximo —
devolveu τ* = 0,60, o mínimo da grade, com **AUC = 0,485**. Abaixo de 0,5 é pior
que o acaso: o F1 máximo é obtido aceitando tudo. Nenhum limiar separa paráfrase
correta de erro injetado.

Não é defeito de medição. Controles diretos:

| controle | cosseno |
|---|---|
| idêntico | 1,0000 |
| paráfrase correta | 0,9907 |
| **frase negada** (sentido oposto) | **0,9912** |
| **número trocado** (56,58% → 85,00%) | **0,9184** |
| assunto alheio | 0,5973 |
| outro idioma | 0,4165 |

O embedding separa *tópico* com folga e é quase cego a *valor* e *polaridade*.
Uma frase negada pontua acima de uma paráfrase legítima.

A quebra por tipo de erro mostra que o AUC agregado esconde regimes opostos:

| tipo de erro | n | média | AUC vs. paráfrase (nomic) | AUC (all-minilm) |
|---|---|---|---|---|
| condição invertida | 20 | 0,9947 | **0,091** | **0,174** |
| número trocado | 53 | 0,9876 | **0,248** | **0,140** |
| *paráfrase correta* | *111* | *0,9702* | — | — |
| entidade substituída | 63 | 0,9418 | 0,808 | 0,752 |

O oráculo funciona para entidade trocada e **se inverte** para número e condição:
nesses dois o erro pontua mais alto que a paráfrase correta em 75% e 91% das
comparações. O padrão replica em dois modelos de embedding de famílias
diferentes, então o achado é sobre o **método** — cosseno contra referência — e
não sobre a escolha de modelo deste estudo.

A razão é estrutural: o erro injetado é a referência com **um token alterado**,
textualmente quase idêntica; a paráfrase é uma **reescrita honesta**, com outro
verbo e outra ordem. O cosseno mede proximidade de superfície, não equivalência
semântica, e por isso prefere a corrupção mínima à reformulação legítima.

Consequência para o artigo: é mais forte que a limitação que o protocolo
antecipava ("τ* é limite superior"). Neste par corpus/modelo **não existe τ*
útil**, e o oráculo mais barato e mais comum na prática é justamente o que não
enxerga o defeito que mais importa em RAG — um número errado numa frase correta.

Reproduzível por `python -m tests.mutation.calibration.diagnose_o1`, com saída em
`calibration/oracle_diagnostic.json`.

**Onde entra:** §6.3 (calibração como instrumento), §7.2 (Results — oracle bias)
como resultado central, §8 (Discussion) como implicação para a prática de
avaliação de RAG, §10 (Threats) para a validade de constructo.

### 4.8 O juiz lê a referência como teto, não como piso

Apareceu no piloto (§3.6). O prompt do O4 pedia que a resposta não tivesse
"afirmação não sustentada", e o juiz leu *não sustentada* como *ausente da
referência*: uma resposta correta e mais detalhada que a referência — os mesmos
três percentuais, cada um descrito pela causa em vez do nome do mutador —
reprovava com "adiciona informações".

O ponto é de desenho, não de prompt: o juiz recebe pergunta, referência e
resposta, e **não vê o documento**. Ele não tem como saber se um detalhe a mais
é verdadeiro. O que ele pode julgar é contradição. Um juiz de referência tem,
portanto, um viés estrutural contra respostas *melhores* que a referência — e
esse viés só aparece quando o SUT é bom. O prompt agora declara que a
referência é o mínimo; os controles depois da mudança (número errado reprova,
mutador trocado reprova, resposta curta correta passa) estão em §3.6.

**Onde entra:** §5.4 (desenho do O4, com a instrução explícita), §7.2 (é um
viés de oráculo medido *antes* da campanha, como o do O1), §9.

### 4.9 A métrica de fidelidade do RAGAS não vê alucinação sobre tópico ausente

O O3 é o oráculo que o protocolo nomeia e o que a literatura usa como padrão.
Com os juízes locais disponíveis ele não funciona, e o motivo tem duas camadas
(§3.6):

1. **Implementação.** `ragas` exige que o juiz devolva JSON num formato
   específico; gemma3:4b falha em 6 de 9 avaliações. `deepeval` tolera a
   saída, mas com 4B extrai zero afirmações e devolve faithfulness = 0 para
   resposta ancorada e = 1 para alucinação — verdade vazia.
2. **Definição.** Com juiz de 8B (qwen3, 4–9 min por avaliação) a alucinação de
   c16 — `"a WCAG 2.2 define 14 critérios"` com contexto que não menciona WCAG —
   continua com faithfulness **1,0**. A métrica do `deepeval` conta afirmações
   que *não contradizem* o contexto; inventar sobre um tópico ausente não
   contradiz nada. O `ragas` é mais estrito (a afirmação precisa ser
   *inferível*), mas é o que não parseia.

Isso importa para o artigo além da decisão operacional: **o oráculo padrão da
área é, por definição, cego ao defeito que o operador P2 injeta** (abstenção
removida). Se O3 ficar, ele entra como oráculo com ponto cego declarado; se
sair, a saída é a contingência do §9 e o motivo é documentado. Ver §6.

**Onde entra:** §5.3, §7.2 (viés por construção), §8, §9 (contingência
exercida).

### 4.10 A classe de abstenção pressupõe um baseline que sabe se abster

c16 (WCAG, fora do corpus) recebeu resposta com número inventado no sistema
**não-mutado**, com contexto que não menciona o tópico. Não é defeito da suíte:
é o SUT violando a própria regra de ancoragem. Pelo §7.1 o caso sai de S para
todos os oráculos.

A consequência é sobre o desenho do estudo, não sobre um caso: os operadores de
prompt que removem abstenção (P2) só são observáveis em casos onde o baseline
se abstém. Se o baseline já alucina, **P2 sobrevive por construção, e o
sobrevivente parece lacuna da suíte quando é defeito do sistema**. O conjunto
avaliável S protege o resultado, mas esvazia a classe; a alternativa — afastar
as perguntas do corpus até o SUT se abster — mede menos o que um usuário real
perguntaria. Decisão pendente (§6).

**Onde entra:** §6.1 (por que a classe existe e o que ela pressupõe), §7.1 (S
por oráculo, com o motivo de cada exclusão), §8, §9.

### 4.11 O custo da campanha é 3× o estimado, e a causa é o raciocínio do modelo

27 s por invocação com 812 tokens de saída em média, quando a resposta útil tem
~150. O resto é o modo de raciocínio do qwen3:8b antes de responder. Projeção:
22,6 h de parede e ~1,1 kWh contra 8,5 h no protocolo. O custo foi medido por
NVML por invocação (GPU-s e Wh), então a RQ3 já tem a linha de base do SUT.

Vale como observação para a RQ3: em um SUT com modelo de raciocínio, o custo do
*sistema sob teste* domina o custo dos *oráculos* — inclusive do juiz. A
proporção que o protocolo esperava (O4 caro, O1 barato) continua verdadeira
entre oráculos, mas ambos são pequenos diante da geração.

**Onde entra:** §6.5 (custo), §7.3 (RQ3), §9 (cronograma e regra de corte).

### 4.12 Uma verificação que não pode falhar não verifica

Duas sondas de fidelidade passavam pelo motivo errado (§3.5): R4 usava uma
pergunta cuja similaridade já passava do piso, então baseline e mutante
devolviam o mesmo; K3 contava chunks, que não mudam ao tirar a sobreposição.
As duas foram trocadas por sondas que **falham no baseline e passam no
mutante** — o critério que toda checagem de fidelidade deveria satisfazer, e
que o protocolo pedia como "revisão manual de 100%" sem dizer como.

**Onde entra:** §4 (fidelidade dos operadores, com a tabela do §3.5), §10
(validade interna: a revisão manual virou executável e versionada).

### 4.13 O servidor truncava o prompt e ninguém viu

Trezentas invocações de baseline, dezoito de piloto, cinco oráculos, três
portões verdes — e o system prompt não chegava ao modelo. O Ollama, sem
`num_ctx` explícito, usa 4.096 e corta o prompt em `num_ctx/2`, preservando os 4
primeiros tokens e os 2.046 últimos. O corte cai exatamente sobre as regras de
ancoragem, abstenção e citação, que ficam no início. O servidor avisa apenas no
próprio log (`truncating input prompt`, 288 vezes); a API responde 200 e
devolve `prompt_eval_count = 2050` — um número que ninguém compara com nada.

Por que os oráculos não viram: todos julgam a resposta. Uma resposta fluente
por conhecimento paramétrico, sem citação e sem abstenção, é indistinguível de
"o SUT ignora as regras" — que foi a conclusão errada tirada por duas horas.
Só um controle de prompt (uma regra sentinela no system prompt) distingue.

Por que importa além deste estudo: é o FP3 de Barnett et al. ("recuperado mas
não consolidado no contexto") acontecendo **por construção, no sistema
não-mutado**, e é o tipo de defeito que a mutação em RAG deveria detectar — mas
que nenhuma suíte de saída detecta se o baseline já o tem. Dois instrumentos
saíram daqui: `num_ctx` dimensionado pela configuração (não pelo pior caso, que
força offload para CPU) e a guarda `prompt_tokens ≥ num_ctx − num_predict →
erro auditável`.

**Onde entra:** §3.5 (instrumentação), §6.7 (desvio), §8 (o parâmetro que
ninguém declara), §9 (validade interna — e a razão de o baseline v1 estar
arquivado, não apagado).

---

## 5. Seções do artigo que já dão para escrever

O que muda em relação à versão anterior deste documento: τ* está calibrado, o
piloto rodou, e dois dos cinco oráculos (O1 e O3) já têm resultado sobre viés
**antes** da campanha. Isso move o §7.2 de "bloqueado" para "parcial".

### Podem ser escritas agora, na íntegra

**§3 Study Context.** Material completo: o SUT e sua configuração exata
(`config/study.yaml`), o corpus caracterizado (8 documentos, 96 páginas, 184
chunks, 1,9 chunk/página) com a justificativa da normalização (§4.6 e §3.1
acima), e a suíte de 30 casos com as cotas por classe, a cobertura por operador
e a verificação de que cada pergunta factual recupera o documento da sua
evidência (§3.6).

**§4 Mutation Operators for RAG.** Os 18 operadores estão no catálogo com camada,
ponto de falha e patch; `apply verify` demonstra aplicação e reversão sem
resíduo e `verify_mutants` demonstra que cada um injeta o defeito declarado,
com a evidência observada por operador (tabela em §3.5). A subseção de
equivalência ganha o achado §4.1 e a regra do conjunto avaliável; a de
fidelidade ganha o §4.12.

**§5 Oracles under Study.** O1–O5 implementados, com a formalização de O5 como
conjunção. O τ* = 0,60 está calibrado e a calibração é resultado (§4.7). O
desenho do O4 inclui a instrução "referência é o mínimo" com a justificativa do
§4.8. O O3 é descrito com o ponto cego do §4.9 — a redação final depende da
decisão de mantê-lo (§6).

**§6 Study Design.** Todos os parâmetros estão fixados e versionados: baseline de
30×10, campanha de 18×30×5, `stop on kill` desligado, temperatura 0,0 exceto P4,
seed = índice da repetição. As regras estatísticas do §7.7 estão implementadas em
`stats.py` e fixadas antes da coleta. O §6.3 (calibração) já tem os números
finais: 247 pares, 111 EQUIV conferidos um a um, 4 rejeitados, AUC por tipo de
erro em dois modelos de embedding. O §6.5 (custo) tem a medição por NVML e o
custo real por invocação.

### Podem ser escritas agora, com lacuna declarada

**§1 Introduction** e **§2.5 Gap.** Não dependem de resultado. O parágrafo de
relação com o SAST 2026 já estava previsto no protocolo. A introdução pode
antecipar o §4.7 como motivação: o oráculo mais barato da prática não vê o
defeito mais importante em RAG.

**§7.2 Results — RQ2 (oracle bias).** Parcial. O viés do O1 (inverte para
número e condição, §4.7) e o ponto cego do O3 (§4.9) são resultados de
calibração e piloto, independentes da campanha. O que falta é o κ entre
oráculos e a comparação de taxas de morte por oráculo, que só a campanha dá.

**§9 Threats to Validity.** A maior parte já está no protocolo e ganha ameaças
*empiricamente demonstradas*, o que vale mais que ameaça antecipada: §4.2
(corpus homogêneo derrota o piso), §4.4 (oráculo determinístico não validado
contra o baseline), §4.5 (referência com justificativa do modelo), §4.8 (juiz
sem acesso ao documento), §4.10 (classe de abstenção pressupõe baseline que se
abstém), e a limitação de que τ* é limite superior pelos dois lados (§3.4).

### Dependem de leitura, não do estudo

**§2.1–2.4** (RAG failure points, oracle problem, mutation testing para sistemas
LLM, avaliação de assistentes). Bloqueadas pelo tempo de leitura, não pela
execução. A semana 6 é quase toda de máquina rodando — é a janela natural para
elas, como o próprio protocolo sugere. O §2.2 (oracle problem) tem agora dois
exemplos próprios para ancorar a leitura: §4.7 e §4.9.

### Bloqueadas pela campanha

**§7.1, §7.3, §7.4** (Results — adequação por camada, custo, coerência) e
**§10** (Conclusion). Precisam de `runs.jsonl` e `verdicts.jsonl` completos. O
`analyze.py` já gera as 10 tabelas e 4 figuras e foi validado com dados
sintéticos — quando a campanha rodar, a análise sai no mesmo dia.

**§8 Discussion.** Parcial: as implicações para praticantes dos achados §4.1 a
§4.12 já podem ser escritas; as que dependem de qual camada sobrevive mais, não.

> **Recomendação de ordem:** escrever §3, §4, §5 e §6 agora, enquanto as decisões
> estão frescas — são as seções onde a memória do que foi decidido e por quê se
> perde mais rápido, e as que um revisor de IST lê com mais atenção num artigo
> de método. Em seguida o §7.2 parcial e o §9, que já têm os números. A
> campanha da semana 6 pode rodar enquanto isso é escrito.

---

## 6. O que ainda falta

### Decisões da semana 6 — tomadas (ver §3.7)

O3 fora da campanha (contingência do §9), campanha completa, classe de abstenção
mantida (c16–c20 abstêm 10/10 no baseline v2). Nenhuma é desvio do pré-registro.
Desvios declarados (artigo §6.7): janela de contexto, portão de recuperação em
nível de página/valor com 9 perguntas reescritas e 4 lacunas aceitas, regra de
citação com exemplo, normalização decimal do O2, assertivas de c13,
embaralhamento por repetição.

### Bloqueadores para publicação (não para a campanha)

| Item | Estado | Impacto |
|---|---|---|
| **Licenças do corpus** | 1 redistribuível, 2 bloqueados (ACM), 5 sem declaração | Bloqueia o Zenodo. Trocar documento **depois** da campanha significa refazer tudo — resolver antes da semana 6 ou aceitar publicar só o manifesto com hashes |
| **Codebook congelado** | `frozen_at: null` | Precisa ser fechado **antes** de ver qualquer resultado da campanha (§7.4). É uma linha em `config/codebook.yaml`, mas é o compromisso do pré-registro |

### Revisões recomendadas antes da campanha

- **caso c09** — os dois fatos estão no mesmo chunk, então ele não exercita a
  recuperação distribuída que a classe pressupõe. A limitação está anotada no
  próprio caso.
- **caso c14** — membro mais fraco da classe `filtro_condicional`: o trecho é uma
  tabela de resultados e não enuncia condição.
- **c07** — reprova no baseline com razão (pede *quais mutantes*, o SUT responde
  com categorias). Sai de S por mérito; se a classe `multi_hop` ficar com poucos
  avaliáveis, é a primeira pergunta a reformular.

### Depois da campanha

Classificação de coerência (§7.4, semana 7) com o codebook congelado; κ entre
dois classificadores; RQ3 sobre os custos já medidos; escrita e submissão
(semanas 8–13).

---

## 7. Como reproduzir o que está feito

```bash
# ambiente
python -m venv tests/mutation/.venv && tests/mutation/.venv/Scripts/activate
pip install -r tests/mutation/requirements.txt
ollama pull qwen3:8b nomic-embed-text gemma3:4b all-minilm

# corpus e índice (semana 1)
python -m tests.mutation.corpus.build_corpus --from data/source_pdfs --limit 8
python -m tests.mutation.tools.audit_licenses --write
python -m tests.mutation.tools.audit_roles --suggest
python -m tests.mutation.tools.calibrate_retrieval_gate --index-only

# suíte (semana 2) — os rascunhos já foram promovidos; isto só confere
python -m tests.mutation.suite.build_assertions
python -m tests.mutation.suite.validate_suite --check-retrieval

# calibração de τ* (semana 3) — pares e revisão estão versionados
python -m tests.mutation.calibration.calibrate
python -m tests.mutation.calibration.diagnose_o1

# fidelidade dos mutantes (semana 4)
python -m tests.mutation.operators.apply verify
python -m tests.mutation.tools.verify_mutants

# piloto (semana 5): 3 casos × (baseline + R1 + P2) × 2 repetições
python -m tests.mutation.run_campaign --baseline --repetitions 2 --cases c01,c07,c16
python -m tests.mutation.run_campaign --campaign --operators R1,P2 --repetitions 2 --cases c01,c07,c16
python -m tests.mutation.evaluate --mutants baseline,R1,P2

# análise sobre dados sintéticos (valida o caminho sem gastar GPU)
python -m tests.mutation.tools.make_synthetic_results
python -m tests.mutation.analyze --results-dir tests/mutation/results/synthetic
```

Os arquivos que cada etapa produz e que estão versionados: `corpus/manifest.json`,
`corpus/variants/variants.json`, `suite/golden.jsonl`, `suite/assertions.yaml`,
`calibration/pairs.jsonl`, `calibration/tau.json`,
`calibration/oracle_diagnostic.json`, `operators/fidelity_report.json`. Os
resultados do piloto (`results/runs.jsonl`, `results/verdicts.jsonl`) não são
versionados — são regenerados pelos comandos acima.

Passo a passo completo em [tests/mutation/README.md](../tests/mutation/README.md).

---

## 8. Histórico de commits

| Commit | Conteúdo |
|---|---|
| `b312412` | parametriza o pipeline do SUT para o estudo |
| `5df5f24` | harness: operadores, oráculos, corpus, suíte, runner, campanha |
| `6f87eed` | análise estatística, tabelas e figuras |
| `e9abd99` | runbook e integração no README raiz |
| `a4054eb` | variantes perturbam fatos, não referências bibliográficas |
| `8faecb8` | auditoria de papéis e de licenças do corpus |
| `168f1cb` | recupera 93% dos símbolos perdidos na normalização |
| `23bef00` | detecta bibliografia por densidade; para de dar veredito sobre papel |
| `1a8797c` | reconstruir o corpus passa a invalidar o índice |
| `24c672a` | semana 2: 30 casos prontos e assertivas conferidas |
| `0d4fec6` | casos "fora do corpus" que recuperavam o corpus; piso com margem |
| `4b4fae1` | este relatório (primeira versão) |
| `89a3f3a` | conjunto de calibração: exclui abstenções e equilibra as classes |
| `0bb9376` | referências curtas demais para servir de alvo ao O1 |
| `1cf45c1` | conferência visual dos 115 pares EQUIV concluída (§6.3) |
| `589483f` | τ* calibrado — e a calibração é o primeiro resultado da RQ2 |
| `48780f3` | ferramenta de diagnóstico do O1 e registro do achado (§4.7) |
| `f4c2248` | O1 embeda em lote — medir o custo certo, não desempenho |
| `a721190` | remove `scores.json` do versionamento |
| `248cb27` | semana 4: fidelidade dos 18 mutantes verificada |
| `e3fae9e` | semana 5: venv próprio, NVML na RQ3, casos do piloto |
| `bf434a7` | piloto: perguntas que não recuperavam a evidência; juiz calibrado |
| `40fb63e` | relatório: piloto da semana 5 (§3.6) |
| `1d7bc66` | relatório: diagnóstico do O3 com juiz de 8B e decisão pendente |
