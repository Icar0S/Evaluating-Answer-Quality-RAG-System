# Estudo de mutação para RAG — relatório de execução

**Artigo:** *Mutation-Based Adequacy Assessment of Test Suites for Retrieval-Augmented Assistants*
**Destino:** Information and Software Technology, special issue VSI:EQUISA · submissão 13/12/2026
**Branch:** `journal-ist` · **Última atualização:** 17/09/2026 (semana 5 concluída; três decisões pendentes para a semana 6)

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
| 6 | Baseline (300 inv.) + campanha (2.700 inv.) | **aguardando decisões** — O3, c16, duração (§6) |
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

### Decisões que bloqueiam a semana 6

São decisões de desenho, não de código; o harness está pronto para qualquer
uma das saídas.

| Decisão | Opções | O que muda |
|---|---|---|
| **O3 (RAGAS)** — inviável com os juízes locais (§4.9) | (a) contingência do §9: campanha com O1, O2, O4, O5; (b) baixar llama3.1:8b e retestar `ragas` num novo piloto; (c) manter via `deepeval` e reportar como degenerado | (a) é a saída pré-registrada e o motivo está documentado; (b) custa ~5 GB e mais um piloto; (c) enfraquece a RQ2 porque a degeneração é da implementação, não do método |
| **c16 e a classe de abstenção** — o baseline alucina (§4.10) | (a) aceitar: os casos saem de S e o artigo reporta como achado sobre o SUT; (b) afastar as cinco perguntas "fora do corpus" até o baseline se abster | (a) pode deixar P2 sem caso avaliável; (b) mede menos o que um usuário perguntaria. Aceitar exige rodar o baseline dos 5 casos antes para saber quantos sobram |
| **Duração da campanha** — 22,6 h vs. 8,5 h estimadas (§4.11) | (a) três noites, harness retomável; (b) regra de corte do §9: 20 casos, ~15 h | (b) reduz o poder das comparações por classe (cotas caem de 5 para ~3) |

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
