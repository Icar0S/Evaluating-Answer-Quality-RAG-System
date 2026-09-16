# Estudo de mutação para RAG — relatório de execução

**Artigo:** *Mutation-Based Adequacy Assessment of Test Suites for Retrieval-Augmented Assistants*
**Destino:** Information and Software Technology, special issue VSI:EQUISA · submissão 13/12/2026
**Branch:** `journal-ist` · **Última atualização:** 16/09/2026

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
| 3 | `calibration/` gerado; τ* calibrado; O1 e O2 prontos | parcial — piso de recuperação calibrado, τ* pendente |
| 4 | `apply.py` + catálogo testado (GO/NO-GO) | **concluída antecipadamente** |
| 5 | O3, O4, O5 prontos; `run_campaign.py` validado | código pronto, piloto não executado |
| 6 | Baseline (300 inv.) + campanha (2.700 inv.) | não iniciada |
| 7–13 | Coerência, RQ3, escrita, submissão | não iniciadas |

O cronograma está **adiantado**: o portão GO/NO-GO da semana 4 fechou junto com a
semana 2, porque calibrar o piso de recuperação (pré-requisito do operador R4)
exigia os casos prontos e os casos ficaram prontos antes do previsto.

### Portões, estado atual

```
validate_suite (sem --allow-draft)   VERDE — 30 casos, 0 rascunhos
apply verify                         VERDE — 18/18 operadores sem resíduo
assertivas vs. própria referência    VERDE — nenhuma se auto-reprova
tests/api (regressão do SUT)         VERDE — 18 testes
```

---

## 2. O que foi construído

49 arquivos versionados em `tests/mutation/`: ~6.900 linhas de Python e ~1.800 de
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
suite/                   30 casos, assertivas, geração de rascunho, promoção, validação
calibration/             300 pares rotulados por construção, varredura de τ
oracles/                 O1 cosseno, O2 assertivas, O3 RAGAS, O4 juiz, O5 conjunção
runner/                  ponte com o SUT, custo (GPU/Wh por NVML), JSONL, caminhos
tools/                   auditoria de papéis e licenças, calibração do piso, dados sintéticos
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

---

## 5. Seções do artigo que já dão para escrever

### Podem ser escritas agora, na íntegra

**§3 Study Context.** Material completo: o SUT e sua configuração exata
(`config/study.yaml`), o corpus caracterizado (8 documentos, 96 páginas, 184
chunks, 1,9 chunk/página) com a justificativa da normalização (§4.6 e §3.1
acima), e a suíte de 30 casos com as cotas por classe e a cobertura por operador.

**§4 Mutation Operators for RAG.** Os 18 operadores estão no catálogo com camada,
ponto de falha e patch, e `apply verify` demonstra aplicação e reversão sem
resíduo. A subseção de equivalência ganha material novo: o achado §4.1
(equivalência por estrutura do corpus) e a regra do conjunto avaliável.

**§6 Study Design.** Todos os parâmetros estão fixados e versionados: baseline de
30×10, campanha de 18×30×5, `stop on kill` desligado, temperatura 0,0 exceto P4,
seed = índice da repetição. As regras estatísticas do §7.7 estão implementadas em
`stats.py` e foram fixadas antes da coleta — o que é exatamente o que dá força ao
pré-registro.

**§5 Oracles under Study.** O1–O5 implementados, com a formalização de O5 como
conjunção (e não como oráculos em paralelo). Falta só o valor de τ*; o desenho, a
justificativa de custo (O3/O4 sobre a resposta modal) e a assimetria entre
oráculos já estão descritos.

### Podem ser escritas agora, com lacuna declarada

**§1 Introduction** e **§2.5 Gap.** Não dependem de resultado. O parágrafo de
relação com o SAST 2026 já estava previsto no protocolo.

**§9 Threats to Validity.** A maior parte já está escrita no protocolo, e ganha
três ameaças novas e *empiricamente demonstradas*: §4.2 (corpus homogêneo derrota
o piso), §4.4 (oráculo determinístico não validado contra o baseline) e §4.5
(resposta de referência com justificativa do modelo). Ameaça demonstrada vale
mais que ameaça antecipada.

### Dependem de leitura, não do estudo

**§2.1–2.4** (RAG failure points, oracle problem, mutation testing para sistemas
LLM, avaliação de assistentes). Bloqueadas pelo tempo de leitura, não pela
execução. A semana 6 é quase toda de máquina rodando — é a janela natural para
elas, como o próprio protocolo sugere.

### Bloqueadas pela campanha

**§7.1–7.4** (Results) e **§10** (Conclusion). Precisam de `runs.jsonl` e
`verdicts.jsonl`. O `analyze.py` já gera as 10 tabelas e 4 figuras, e foi
validado com dados sintéticos — quando a campanha rodar, a análise sai no mesmo
dia.

**§8 Discussion.** Parcial: as implicações para praticantes dos achados §4.1 a
§4.4 já podem ser escritas; as que dependem de qual camada sobrevive mais, não.

> **Recomendação de ordem:** escrever §3, §4 e §6 agora, enquanto as decisões
> estão frescas. São as três seções onde a memória do que foi decidido e por quê
> se perde mais rápido, e são justamente as que um revisor de IST lê com mais
> atenção num artigo de método.

---

## 6. O que ainda falta

### Bloqueadores reais

| Item | Estado | Impacto |
|---|---|---|
| **Licenças do corpus** | 1 redistribuível, 2 bloqueados (ACM), 5 sem declaração | Bloqueia o Zenodo. Trocar documento depois da campanha significa refazer tudo |
| **τ\* do oráculo O1** | não calibrado | Bloqueia `evaluate.py` e, portanto, todos os vereditos |
| **300 pares de calibração** | não gerados | Pré-requisito de τ*; inclui ~2 h de conferência visual dos 150 pares EQUIV |
| **Codebook congelado** | `frozen_at: null` | Precisa ser fechado **antes** de ver qualquer resultado (§7.4) |

### Revisões recomendadas antes da campanha

- **caso c09** — os dois fatos estão no mesmo chunk, então ele não exercita a
  recuperação distribuída que a classe pressupõe. A limitação está anotada no
  próprio caso.
- **caso c14** — membro mais fraco da classe `filtro_condicional`: o trecho é uma
  tabela de resultados e não enuncia condição.
- **revisão manual dos 18 mutantes** antes da campanha (§10, validade interna).

### Sem bloqueio

Piloto de 2 mutantes (semana 5) e validação do O4. O código está pronto; falta
executar.

---

## 7. Como reproduzir o que está feito

```bash
# ambiente
python -m venv tests/mutation/.venv && tests/mutation/.venv/Scripts/activate
pip install -r tests/mutation/requirements.txt
ollama pull qwen3:8b nomic-embed-text gemma3:4b all-minilm

# corpus e índice
python -m tests.mutation.corpus.build_corpus --from data/source_pdfs --limit 8
python -m tests.mutation.tools.audit_licenses --write
python -m tests.mutation.tools.audit_roles --suggest
python -m tests.mutation.tools.calibrate_retrieval_gate --index-only

# portões
python -m tests.mutation.suite.validate_suite
python -m tests.mutation.operators.apply verify

# análise sobre dados sintéticos (valida o caminho sem gastar GPU)
python -m tests.mutation.tools.make_synthetic_results
python -m tests.mutation.analyze --results-dir tests/mutation/results/synthetic
```

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
