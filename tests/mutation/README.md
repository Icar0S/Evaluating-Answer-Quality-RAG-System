# Estudo de mutação para RAG — runbook

Implementação do protocolo **Mutation-Based Adequacy Assessment of Test Suites
for Retrieval-Augmented Assistants** (IST / VSI:EQUISA, submissão 13/12/2026).

Este arquivo é o passo a passo operacional: o que rodar, em que ordem, o que
precisa estar verde antes de seguir, e o que fazer quando não estiver. O
protocolo responde *por quê*; aqui está o *como*.

- **SUT:** este repositório (`backend/app/`), invocado in-process.
- **RQ1** — que camadas do pipeline produzem defeitos que a suíte não detecta.
- **RQ2** — quanto o escore de adequação depende do oráculo escolhido.
- **RQ3** — quanto custa cada nível de rigor, por mutante morto.

---

## 1. Preparo (uma vez)

### 1.1 Ambiente Python

A camada de mutação vive num venv **próprio**, separado do `backend/.venv`, pelo
mesmo motivo que `tests/deepeval/` já era separado: `ragas` e `deepeval` exigem
`pydantic` mais novo que o pin de `backend/requirements.txt`, e o pip não resolve
os dois juntos.

```bash
python -m venv tests/mutation/.venv
tests/mutation/.venv/Scripts/activate        # Windows
# source tests/mutation/.venv/bin/activate   # Linux/macOS
pip install -r tests/mutation/requirements.txt
```

Todos os comandos abaixo rodam a partir da **raiz do repositório**.

### 1.2 Modelos no Ollama

| Modelo | Papel | Por quê |
|---|---|---|
| `qwen3:8b` | geração do SUT | definido no protocolo |
| `nomic-embed-text` | embeddings do SUT **e** do oráculo O1 | idem |
| `gemma3:4b` | juiz (O4), paráfrases da calibração | família distinta da do SUT — mitiga autopreferência e circularidade |
| `all-minilm` | operador E1 | embedding alternativo, menor e de outra dimensionalidade |

```bash
ollama pull qwen3:8b && ollama pull nomic-embed-text
ollama pull gemma3:4b && ollama pull all-minilm
```

---

## 2. Mapa dos arquivos

```
tests/mutation/
├── config/
│   ├── study.yaml          # o "Study Context" do artigo, em forma executável
│   └── codebook.yaml       # códigos de coerência, congelados A PRIORI (§7.4)
├── corpus/
│   ├── sources.yaml        # manifesto: licença, URL, sha256, papéis
│   ├── build_corpus.py     # normaliza os PDFs e gera as variantes prev/conflict
│   ├── base/               # corpus do estudo (gerado)
│   └── variants/           # variantes + variants.json com o que mudou
├── operators/
│   ├── catalog.yaml        # os 18 operadores, com camada, FP e patch
│   └── apply.py            # aplica/reverte (idempotente) + portão `verify`
├── suite/
│   ├── golden.jsonl        # 30 casos de referência
│   ├── assertions.yaml     # assertivas determinísticas por caso (oráculo O2)
│   ├── draft_cases.py      # propõe rascunhos a partir do corpus
│   └── validate_suite.py   # portão da suíte
├── calibration/
│   ├── build_pairs.py      # 300 pares rotulados por construção
│   └── calibrate.py        # varredura de τ, F1, AUC, IC bootstrap
├── oracles/                # o1_cosine, o2_assertions, o3_ragas, o4_judge, o5_conjunctive
├── runner/                 # ponte com o SUT, custo (GPU/Wh), JSONL, caminhos
├── tools/
│   ├── verify_mutants.py             # cada mutante injeta o defeito declarado?
│   ├── audit_roles.py                # papéis do corpus servem aos operadores C1-C4 e E2?
│   ├── audit_licenses.py             # o corpus pode ir para o Zenodo?
│   ├── calibrate_retrieval_gate.py   # calibra o piso de similaridade do baseline
│   └── make_synthetic_results.py     # dados falsos para testar a análise sem GPU
├── run_campaign.py         # baseline + mutantes -> runs.jsonl
├── evaluate.py             # oráculos sobre o log -> verdicts.jsonl
├── coherence.py            # classificação de coerência + κ intra-avaliador
├── analyze.py              # todas as tabelas e figuras + cost.jsonl
├── stats.py                # Wilson, bootstrap, Cochran Q, McNemar, KW, Dunn, Holm, κ
└── results/                # runs.jsonl, verdicts.jsonl, cost.jsonl, tables/, figures/
```

---

## 3. Passo a passo, semana a semana

O cronograma é o do §8 do protocolo. Cada passo termina num **portão**: se ele
não passar, não siga adiante — o custo de descobrir o problema depois da campanha
de 8,5 h é alto demais.

### Semana 1 — corpus

```bash
# 1. Escolha os documentos e registre licença/URL/sha256 em corpus/sources.yaml.
# 2. Construa o corpus normalizado (e as variantes prev/conflict):
python -m tests.mutation.corpus.build_corpus --from data/source_pdfs --limit 8
```

O construtor **re-renderiza todos os documentos** a partir do texto extraído e
repagina por orçamento de tokens (~1800/página). As duas decisões têm motivo:

- se só os mutantes fossem re-renderizados, a diferença de extração entre
  original e mutante entraria no resultado junto com o defeito injetado;
- com páginas de ~500 tokens (artigo em duas colunas), um chunk de 1200 engoliria
  a página inteira e **K1–K4 seriam inertes por construção** — o SUT quebra
  chunks por página.

Os papéis (`primary`, `secondary`, `recent`) nascem atribuídos por ordem
alfabética, o que destrava o pipeline mas não serve para rodar o estudo: eles
decidem quais operadores conseguem ser mortos por algum caso. Meça e escolha:

```bash
python -m tests.mutation.tools.audit_roles --suggest
# adote a escolha editando `roles` em corpus/sources.yaml e:
python -m tests.mutation.corpus.build_corpus --roles-only
python -m tests.mutation.corpus.build_corpus --make-variants   # se `primary` mudou
```

O papel que mais erra na atribuição automática é o `secondary`, alvo de C3
(truncar -30%): em artigo de conferência o terço final costuma ser a
bibliografia, e caso nenhum cita bibliografia — C3 viraria equivalente por
construção. `audit_roles` mede quanto do trecho que C3 apaga é prosa com fatos e
reprova a escolha quando não é.

```bash
python -m tests.mutation.tools.audit_licenses --write
```

Classifica cada documento em `yes` / `no` / `unknown` a partir das declarações
impressas no PDF e grava no manifesto. **`unknown` nunca vira `yes` por
conveniência**: o script extrai o DOI/arXiv da primeira página justamente para
você confirmar na fonte. Um pacote de replicação publicado sem direito de
redistribuição é o pior desfecho possível deste passo.

**Portão:** `audit_roles` sem problemas, `audit_licenses` sem nenhum documento
`no` ou `unknown`, e `corpus/manifest.json` com 5–10 documentos.

### Semana 2 — os 30 casos

Este é o item de maior tempo humano irredutível. O script de rascunho corta a
parte mecânica, não o julgamento:

```bash
# Constrói o índice do estudo (alguns minutos — embeda o corpus inteiro):
python -m tests.mutation.tools.calibrate_retrieval_gate --index-only
python -m tests.mutation.suite.draft_cases --all
```

> A calibração do piso de recuperação **não** roda aqui: ela precisa de casos
> factuais prontos para saber o que deve passar do piso, e esses casos são o
> produto desta semana. Por isso `--index-only` agora e a calibração completa na
> semana 3.

As propostas saem em `suite/drafts.jsonl` com uma conferência automática rasa
(valores citados que não aparecem no trecho). **Confira cada uma contra o
documento**, ajuste o texto, e copie para `suite/golden.jsonl` com
`"status": "ready"`. Depois escreva as assertivas em `suite/assertions.yaml`.

Onze casos já vêm escritos: os que não dependem do corpus (5 fora do corpus, 3
fora de escopo, 3 ambíguos). Reconfira os 5 "fora do corpus" contra o corpus
definitivo — se algum documento cobrir o assunto, troque a pergunta.

```bash
python -m tests.mutation.suite.validate_suite
```

**Portão:** validação verde. Ela checa as cotas das 7 classes, que todo operador
tem ao menos um caso alvo, e que algum caso cita um valor que as variantes
`prev`/`conflict` alteram (sem isso, C2 e C4 sobrevivem por construção).

### Semana 3 — calibração

```bash
# Piso de similaridade do baseline (sem ele, R4 é mutante equivalente):
python -m tests.mutation.tools.calibrate_retrieval_gate --write

# 300 pares rotulados por construção:
python -m tests.mutation.calibration.build_pairs

# >>> CONFERÊNCIA VISUAL DOS 150 PARES EQUIV (~2 h, §6.3) <<<
#     marque "reviewed": true nos que preservam todos os fatos

python -m tests.mutation.calibration.calibrate --write
```

A conferência visual não é burocracia: uma paráfrase que perdeu um fato é um
rótulo errado, e rótulo errado desloca τ*. `calibrate.py` se recusa a rodar com
pares não conferidos (`--allow-unreviewed` só para testar o pipeline).

**Portão:** `calibration/tau.json` com τ*, F1, AUC e IC bootstrap. IC de τ* mais
largo que 0,15 é sinal de que EQUIV e ERRO se sobrepõem demais — reveja as
paráfrases antes de seguir.

### Semana 4 — operadores (GO / NO-GO)

```bash
python -m tests.mutation.operators.apply list
python -m tests.mutation.operators.apply verify
python -m tests.mutation.tools.verify_mutants
```

`verify` aplica e reverte os 18 operadores conferindo que `work/` volta ao estado
original byte a byte. `verify_mutants` faz a pergunta seguinte, que o `verify`
não responde: **o mutante é o que o catálogo diz?** Para cada operador, aplica,
observa o efeito no ponto do pipeline que ele deveria alterar (arquivos,
contagem de chunks, sobreposição na fronteira, dimensão do embedding, chunks
devolvidos, system prompt) e grava `operators/fidelity_report.json`. É a
"revisão manual de 100% dos 18 mutantes" do §10, em forma executável. Aplicar é sempre "reconstruir do zero e então mutar", nunca
"desfazer a operação inversa" — é o que torna a chamada idempotente e elimina a
classe de falha mais cara de diagnosticar: o mutante que continuou aplicado
depois do revert.

**Portão (GO/NO-GO do protocolo):** `verify` e `verify_mutants` verdes. Se algum operador não passar,
a regra de corte manda remover os 6 marcados `cuttable: true` e rodar com 12:

```bash
python -m tests.mutation.run_campaign --campaign --exclude-cuttable
```

### Semana 5 — oráculos e piloto

```bash
# Piloto em 2 mutantes, 3 casos — valida o caminho inteiro:
python -m tests.mutation.run_campaign --baseline --repetitions 2 --cases c01,c07,c16
python -m tests.mutation.run_campaign --campaign --operators R1,P2 --repetitions 2 --cases c01,c07,c16
python -m tests.mutation.evaluate --mutants baseline,R1,P2
```

A escolha dos casos não é arbitrária. Dois factuais (c01, c07) recuperam
contexto e exercitam os cinco oráculos — um caso de abstenção não tem contexto e
deixa O3 sem denominador. E cada mutante tem um caso que deveria matá-lo: R1
(top-k = 1) mira c07, que precisa de dois trechos; P2 (sem instrução de
abstenção) mira c16, que deve se abster. Se nenhuma dessas mortes acontecer no
piloto, o problema é do harness, não da suíte.

Valide a análise **antes** da campanha, com dados falsos, para não descobrir um
caso de borda depois de queimar a noite de GPU:

```bash
python -m tests.mutation.tools.make_synthetic_results
python -m tests.mutation.analyze --results-dir tests/mutation/results/synthetic
```

**Portão:** O4 estável no piloto (os 5 julgamentos concordam ou divergem de forma
explicável). Se o juiz não estabilizar, o §9 prevê rodar com 4 oráculos —
`oracles.enabled` em `config/study.yaml` sem `O4`; a RQ2 sobrevive com O1, O2,
O3 e O5.

### Semana 6 — campanha (máquina, não pesquisador)

```bash
python -m tests.mutation.run_campaign --baseline     # 30 x 10 = 300 invocações
python -m tests.mutation.run_campaign --campaign     # 18 x 30 x 5 = 2.700
python -m tests.mutation.run_campaign --l3           # 10 x 30 = 300 (RQ3)
```

Retomável: `run_id` já presente em `runs.jsonl` é pulado, então uma campanha
interrompida na hora 6 continua de onde parou. Reindexação acontece só quando a
assinatura de índice muda (parâmetros de chunking, modelo de embedding ou
conteúdo de `work/index_src`).

```bash
python -m tests.mutation.suite.validate_suite --check-baseline
```

Confere que o baseline se comporta como os casos declaram — se um caso "fora do
corpus" já é respondido pelo sistema não-mutado, ou a pergunta está no corpus ou
o piso de recuperação está frouxo.

**Portão:** campanha dentro do orçamento de GPU. Se estourar, o §9 manda reduzir
a suíte de 30 para 20 casos preservando as 7 classes.

### Semana 7 — vereditos e coerência

```bash
# Congele o codebook ANTES de olhar qualquer resultado:
python -m tests.mutation.coherence --freeze-codebook

python -m tests.mutation.evaluate                    # nível L2 (padrão)
python -m tests.mutation.coherence --classify
python -m tests.mutation.coherence --summary
```

`evaluate.py` roda sobre o log, nunca sobre o SUT: rejulgar não custa GPU de
geração. É isso que permite recalibrar τ*, trocar o backend de O3 ou reexecutar o
juiz sem repetir as 3.000 invocações.

A classificação é por morte `(mutante, caso)`, não por oráculo: a causa é
propriedade da resposta, não do instrumento que a reprovou.

### Semana 8 — RQ3 e análise

```bash
python -m tests.mutation.evaluate --level L1         # veredito pontual
python -m tests.mutation.evaluate --level L3
python -m tests.mutation.analyze
```

Sai em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `tables/T0_conjunto_avaliavel` | \|S\| e flakiness por oráculo |
| `tables/T1_taxa_morte_por_operador` | taxa de morte por operador, IC de Wilson |
| `tables/T2_camadas` + `T2b_dunn` | agregação por camada, Kruskal-Wallis, Dunn/Holm |
| `tables/T3*` | MS por oráculo, swing, McNemar/Holm, Cohen κ, divergência por classe |
| `tables/T4_sobreviventes` | mutantes sobreviventes e requisitos de teste não cobertos |
| `tables/T5_custo_por_nivel` | custo por nível e oráculo, normalizado por morte |
| `figures/F1..F4` | PNG (300 dpi) e PDF vetorial |
| `cost.jsonl`, `summary.json` | dados brutos da RQ3 e resumo dos testes |

Cada tabela sai em CSV (para reanálise) e `.tex` com `booktabs` (para o artigo).

### Semana 11 — κ intra-avaliador

```bash
# >= 7 dias depois da primeira classificação:
python -m tests.mutation.coherence --blind
python -m tests.mutation.coherence --kappa
```

O intervalo é o que torna a reclassificação de fato cega; `coherence.py` se recusa
a rodar antes disso.

---

## 4. Esquemas de dados

`results/runs.jsonl` — uma linha por invocação (§4.1), mais campos que o
protocolo previa: `retrieval_context` (insumo do O3), `wh` (energia medida por
NVML, não estimada por TDP) e `error` (falha vira registro auditável, não derruba
a campanha).

`results/verdicts.jsonl` — uma linha por `(nível, mutante, caso, oráculo)`.
`coherent` e `coherence_code` nascem `null` e são preenchidos por `coherence.py`.

`results/cost.jsonl` — uma linha por `(nível, oráculo)`, derivada por
`analyze.py` a partir dos outros dois.

---

## 5. Decisões de implementação e desvios do protocolo

Tudo aqui precisa aparecer no artigo. Nenhuma destas escolhas é silenciosa.

**Configuração do estudo ≠ defaults do repositório.** `config/study.yaml` fixa
chunk 1200/200, MMR ligado, piso de similaridade calibrado, regra de citação
ligada e temperatura 0,0. Os defaults do assistente em produção continuam os de
antes; o overlay é aplicado em memória e descartado no fim. Sem esse baseline,
R3 não teria MMR para desligar, R4 não teria limiar para remover e P3 não teria
exigência de citação para retirar.

**Conjunto avaliável exige `pass` no baseline.** O §7.1 define caso avaliável
como "veredito único nas N execuções do baseline". Um caso que já reprova no
sistema não-mutado reprovaria em todo mutante e inflaria a taxa de mortes sem
carregar informação sobre o defeito. `evaluable_requires_baseline_pass: true`
exige que o veredito único seja aprovação; `analyze.py` reporta instáveis e
estavelmente-reprovados em colunas separadas.

**Duas leituras de "MS", nunca misturadas.** Com um mutante por operador, MS por
operador seria 0 ou 1 e o IC de Wilson não diria nada. A Tabela 1 reporta *taxa
de morte* = casos de S que reprovam / |S| (proporção com IC interpretável, e a
observação que entra no Kruskal-Wallis entre camadas). MS_bruto e MS_coerente
ficam no nível do oráculo, como o §7.1 define, e é sobre eles que o swing da RQ2
é calculado.

**O2 olha só a saída.** Assertivas determinísticas nunca assertam sobre os chunks
recuperados. A H1 pressupõe que "a suíte típica testa a saída e não a
recuperação" — uma suíte já instrumentada com assertivas de recuperação
responderia a RQ1 por construção.

**E2 vs. C1.** Os dois tiram evidência do alcance do sistema, mas por caminhos
diferentes: C1 remove o arquivo do corpus (reparo: reobter o documento); E2 deixa
os arquivos do papel `recent` no corpus e fora do índice (reparo: `POST /ingest`).
O harness mantém dois diretórios — `work/corpus` (o que o corpus é) e
`work/index_src` (o que foi indexado) — e só E2 os faz divergir.

**Níveis L1/L2/L3 por rejulgamento.** `--level` limita quantas execuções entram
no veredito, e os vereditos saem etiquetados. A mesma campanha responde "o que L1
teria detectado?" sem reexecutar o SUT; só o custo dos oráculos muda entre
níveis, e ele é medido, não estimado.

**Temperatura do juiz em 0,0.** Os 5 julgamentos de O4 medem o não-determinismo
residual do runtime, não variância de amostragem. Se os 5 concordarem sempre,
isso é um dado sobre o juiz e a maioria simples degenera em julgamento único sem
custo metodológico. Subir a temperatura mede outra coisa e precisaria ser
declarado.

**Backend do O3 é escolha explícita.** `ragas` é o instrumento nomeado no
protocolo; `deepeval` existe como alternativa para o ponto de decisão da semana
5. Não há fallback automático: um O3 que "passa" porque não conseguiu medir
contaminaria a RQ2 inteira.

**Energia é medida, não estimada.** `runner/cost.py` amostra utilização e
potência via NVML a 10 Hz e integra. Sem NVIDIA/pynvml, os campos voltam `null` —
zero seria indistinguível de "GPU ociosa" e contaminaria a média da RQ3.

**Erro de digitação no protocolo.** O §6.1 diz que a classe "fora de escopo" é
alvo de "P1 e P5", mas o catálogo do §5 tem 18 operadores e vai só até P4. Aqui a
classe está mapeada para P1 (remoção da ancoragem no contexto, que é o que
governa responder fora do domínio) e P2.

### Mudanças no SUT (`backend/app/`)

Os operadores mutacionam configuração, e alguns dos parâmetros não existiam. Foram
adicionados com **defaults que preservam o comportamento anterior** — o prompt v1
sai byte a byte idêntico e os 18 testes de `tests/api/` continuam passando:

| Onde | O quê | Default |
|---|---|---|
| `config.py` | overlay de configuração em memória (`set_settings_overrides`) | vazio |
| `retrieval.py` | `min_similarity_score` aplicado de fato (era knob morto) | `0.0` = sem filtro |
| `retrieval.py` | MMR opcional (`retrieval_use_mmr`, `retrieval_mmr_lambda`) | desligado |
| `generation.py` | system prompt montado a partir de blocos ligáveis | = prompt v1 |
| `generation.py` | `generation_temperature` / `generation_seed` | `None` = não envia `options` |
| `ingestion.py` | `chunk_boundary_offset_tokens` (operador K4) | `0` |
| `ingestion.py` | `embed_texts(model=...)` — espaço de embedding fixo para o O1 | segue a configuração |
| `vector_store.py` | `reset_client()` e `query(include_embeddings=)` | — |

---

## 6. O que ainda depende de você

1. **19 casos de `golden.jsonl`** e suas assertivas (semana 2).
2. **Licenças do corpus.** A auditoria de 15/09/2026 sobre o corpus montado a
   partir de `data/source_pdfs` encontrou **1 documento redistribuível
   (CC-BY-4.0), 2 bloqueados (ACM) e 5 sem licença declarada no PDF**. Como
   está, o corpus não pode ir para o Zenodo. Confirme os 5 `unknown` na fonte
   (o manifesto guarda o DOI/arXiv de cada um) e troque os bloqueados por
   documentos de licença aberta — proceedings CEUR-WS (CC BY 4.0), preprints
   arXiv com licença CC declarada e documentação técnica sob licença livre são
   os caminhos mais rápidos. Depois de trocar, rode de novo `audit_licenses`,
   `audit_roles` e `build_corpus`.
3. **Calibrações** — τ* e o piso de recuperação. Enquanto o piso for 0,0,
   `apply verify` recusa R4 de propósito.
4. **Congelar o codebook** antes de olhar resultado.
5. **Conferência visual dos 150 pares EQUIV.**
6. **Revisão manual dos 18 mutantes** antes da campanha (§10, validade interna).

---

## 7. Problemas comuns

| Sintoma | Causa e saída |
|---|---|
| `apply verify` falha em R4 | `min_similarity_score` ainda é 0,0 — rode `tools/calibrate_retrieval_gate.py --write` |
| `evaluate.py` aborta dizendo que τ* é nulo | rode `calibration/calibrate.py --write` |
| `UnicodeEncodeError` no console | os scripts já reconfiguram stdout; se persistir, `set PYTHONUTF8=1` |
| Campanha lenta demais | `--exclude-cuttable` (12 operadores) ou reduza a suíte preservando as 7 classes |
| `matplotlib` ausente | `analyze.py --skip-figures` gera só as tabelas |
| Índice reconstruído toda hora | confira `work/state.json`; a assinatura muda com chunking, embedding ou conteúdo de `index_src` |
| Um operador de corpus nunca mata ninguém | quase sempre é papel mal atribuído — rode `tools/audit_roles.py` |

`tests/mutation/work/` é descartável: apague à vontade, `apply.py` reconstrói a
partir de `corpus/base/`. O índice do estudo fica em `work/vector_store` e nunca
toca `data/vector_store` — rodar a campanha não invalida o assistente do dia a dia.

---

## 8. Pacote de replicação

Vão para o Zenodo: `config/`, `corpus/` (com licenças resolvidas), `operators/`,
`suite/`, `calibration/pairs.jsonl` + `tau.json`, `oracles/`, `results/` completo
e este README. `work/` não vai — é derivável.
