# Avaliação de qualidade (JevEval)

Terceira camada de qualidade do RAG, ao lado do [DeepEval](../deepeval/README.md)
e do [RAGAS](../ragas/README.md). Usa o **JevEval**, métrica do `deepeval`
(4.3+) em que a decisão semântica fica com o **Jev**, o modelo "System One" da
TypeSafe AI. O Jev não gera texto: responde perguntas fechadas com
probabilidades. A nota sai de uma conta fixa, feita fora do modelo: a média
ponderada das respostas.

A camada mede as **mesmas cinco dimensões** das outras duas (fidelidade,
relevância da resposta, precisão, recall e relevância do contexto) e mais uma
que só ela tem: a **correção** da resposta em relação à referência.

Ela tem três partes:

1. **Testes herméticos.** Um Jev falso (`_fake_jev.py`) valida as perguntas, os
   pesos, a agregação, o N/A e o formato do export. Rodam em todo PR no CI, sem
   rede e sem chave.
2. **Validação do instrumento.** Controles com resposta conhecida, uma checagem
   de estabilidade e a validade de critério sobre os **247 pares rotulados à
   mão** do estudo de mutação. Rodam antes de qualquer nota ser lida.
3. **Avaliação do RAG.** Os goldens são julgados sobre as respostas
   compartilhadas do `tests/run_all.py`, o mesmo texto que DeepEval e RAGAS
   julgam. Como análise exploratória, o oráculo **O6** aplica a correção ao log
   da campanha de mutação, sem reexecutar o assistente.

> **Os dados saem da máquina.** As outras camadas usam um juiz local, via
> Ollama. Aqui, cada medição manda pergunta, resposta, referência e trechos para
> `https://api.typesafe.ai`. O acervo é de artigos públicos, mas a camada só
> roda com `TYPESAFE_API_KEY` definida. Sem a chave, as partes 2 e 3 se
> auto-pulam, e só a parte 1 roda.

**Estado (16/10/2026):** implementada e testada com o Jev falso (63 testes
herméticos), e os scripts rodaram de ponta a ponta sobre os artefatos reais
(log da campanha, os 247 pares, respostas compartilhadas). Lendo o
`verdicts.jsonl` da campanha, o resumo do O6 reproduz o escore oficial do O4
(16/18, |S| = 22, tabela T3 do estudo). A validação com o Jev de verdade (controles, critério e O6) ainda
não rodou, porque a máquina não tinha chave da TypeSafe. Até ela rodar, não há
número do JevEval para ler.

## Setup (uma vez)

```powershell
python -m venv tests\jeveval\.venv
tests\jeveval\.venv\Scripts\pip install -r tests\jeveval\requirements.txt
```

E, para as partes 2 e 3, a chave no `.env` da raiz (ver `.env.example`):

```
TYPESAFE_API_KEY=...
```

O venv é próprio pelo mesmo motivo das outras camadas: o `deepeval` exige um
`pydantic` mais novo que o pinado em `backend/requirements.txt`. E não é o venv
de `tests/deepeval`, que fica no 4.2.0. O JevEval só existe a partir do 4.3, e
trocar a versão de lá mudaria o instrumento de uma camada que já tem
resultados registrados.

A camada não importa o backend. Ela julga respostas já geradas e lê os artefatos
do estudo de mutação, e por isso o venv não leva chromadb nem PyMuPDF.

## Rodar

Sempre a partir da raiz do repositório:

```powershell
# 1. herméticos (Jev falso; o mesmo comando do CI)
tests\jeveval\.venv\Scripts\python.exe -m pytest tests/jeveval -m "not live"

# 2a. controles e estabilidade com o Jev de verdade (~25 chamadas)
tests\jeveval\.venv\Scripts\python.exe -m pytest tests/jeveval -m live

# 2b. validade de critério nos 247 pares rotulados (~247 chamadas)
tests\jeveval\.venv\Scripts\python.exe tests\jeveval\criterion_validity.py

# 3a. avaliação dos goldens sobre respostas já geradas
backend\.venv\Scripts\python.exe tests\generate_answers.py --out respostas.json
tests\jeveval\.venv\Scripts\python.exe tests\jeveval\run_and_export.py --answers respostas.json

# 3b. O6 sobre o log da campanha de mutação (exploratório, 570 chamadas no L2)
tests\jeveval\.venv\Scripts\python.exe tests\jeveval\o6_mutation.py
tests\jeveval\.venv\Scripts\python.exe tests\jeveval\o6_mutation.py --summary-only
```

Pelo `scripts\run_all_tests.bat`, os herméticos rodam sempre (inclusive no
`--rapido`). Os controles e a avaliação rodam depois do DeepEval, sobre as
respostas compartilhadas, se houver chave. Sem chave, saem como "pulada" com o
motivo. O relatório consolidado mostra o JevEval como terceiro framework.

O custo é baixo: o Jev cobra US$ 0,042 por milhão de tokens de entrada, e a
saída é grátis (`deepeval/models/system_one/constants.py`). Cada export
registra as chamadas e o custo.

## O instrumento

As perguntas estão em [`_jev_shared.py`](_jev_shared.py). Cada métrica manda ao
Jev só os campos que declara: a fidelidade, por exemplo, não vê a referência.

| Métrica | Campos enviados | Origem das perguntas | N/A? |
|---|---|---|---|
| `faithfulness` | resposta, trechos | system_one do `FaithfulnessMetric`, **adaptado** | sim |
| `answer_relevancy` | pergunta, resposta | system_one do `AnswerRelevancyMetric`, com N/A | sim |
| `context_precision` | pergunta, trechos, referência | system_one do `ContextualPrecisionMetric`, sem alteração | não |
| `context_recall` | referência, trechos | system_one do `ContextualRecallMetric`, sem alteração | não |
| `context_relevance` | pergunta, trechos | system_one do `ContextualRelevancyMetric`, sem alteração | não |
| `correctness` | pergunta, referência, resposta | própria, com a semântica do O4 | não |

"system_one" são as perguntas que o próprio deepeval manda ao Jev quando uma
métrica nativa roda nesse modo
(`deepeval/metrics/<métrica>/templates/_experimental_system_one_questions.txt`).
As três de contexto são cópia literal, e um teste hermético compara a cópia
com o pacote instalado. Se uma atualização do deepeval mudar as perguntas, o
teste falha, em vez de o instrumento mudar sozinho.

As adaptações, cada uma motivada por um problema que o projeto já documentou:

- **A fidelidade perdeu a pergunta "nenhuma afirmação contradiz o contexto".**
  É o ponto cego do
  [§4.9 do estudo de mutação](../../docs/estudo-mutacao-status.md): "a WCAG 2.2
  define 14 critérios", com contexto que nem cita WCAG, levou faithfulness 1,0
  no DeepEval, porque inventar sobre um tópico ausente não contradiz nada. As
  perguntas que ficaram ganharam a regra do RAGAS: afirmação sobre o que os
  trechos não mencionam não tem apoio.
- **Fidelidade e relevância da resposta ganharam a opção N/A.** Quando o
  assistente responde "não encontrei", a FaithfulnessMetric dá 0,0 à recusa, e
  o certo é N/A: recusa não tem afirmação para ser fiel. Para isso, cada Noul
  virou um Choice {No, Yes, N/A}, e cada Score virou um Choice com os mesmos
  níveis e o mesmo crédito por nível. Os pesos foram mantidos. Se a abstenção
  foi certa ou errada, quem diz é a correção.

A **correção** é um Choice só, com seis opções. Só "correta" tem crédito 1, e
por isso a nota é a probabilidade que o Jev dá à resposta estar certa. As
outras opções nomeiam o defeito (omite, troca ou contradiz um fato, não faz o
que a referência faz, perde a citação, responde outra pergunta). Isso dá a
causa provável de graça, no vocabulário do O4 (`cause` no export). A
referência é o **mínimo**, não o máximo, a mesma calibração do prompt do O4 no
piloto: detalhe a mais não reprova.

As perguntas estão em inglês, com o conteúdo em português. É a mesma condição
dos prompts do DeepEval e do RAGAS neste projeto. Mudar o idioma só aqui
misturaria idioma do instrumento com método na comparação.

### N/A, e por que ele não é nota 1

Quando nenhuma pergunta se aplica, o JevEval devolve **1,0 e `success=True`**
(`EMPTY_SCORE` em `deepeval/metrics/jev_eval/utils.py`). É a mesma verdade
vazia que deu 1,0 a uma alucinação no estudo de mutação, só que do outro lado.
A camada transforma isso em N/A explícito (`score: null`, `applicable: false`,
sem erro). Uma pergunta sai da conta quando a opção N/A concentra 50% ou mais da
probabilidade.

Uma nota pode, então, estar em três estados, que o relatório consolidado
distingue:

| Estado | No JSON | Nas médias | Em "≥ limiar" |
|---|---|---|---|
| medida | `score` com a nota | entra | aprova ou reprova |
| não medida (erro do Jev, geração falhou) | `score: null`, `error` | fora | **reprova** |
| N/A (nada se aplica) | `score: null`, `applicable: false` | fora | fora |

### Limiares e versão

- **0,7** nas cinco dimensões, o mesmo do DeepEval e do RAGAS
  (`JEVEVAL_THRESHOLD`). Com limiares iguais, a comparação de vereditos no
  relatório é de método, não de régua. É um ponto de partida, não um valor
  medido.
- **0,5** na correção (`JEVEVAL_CORRECTNESS_THRESHOLD`), que é uma
  probabilidade: "mais provável certa do que errada". Fixado **antes** de olhar
  os pares rotulados. A validade de critério mostra a varredura, mas não
  escolhe o limiar.
- **`jev-1.13.0`**, e não o alias `jev-latest` (`JEVEVAL_MODEL`). O alias passa
  a apontar para outro modelo quando a TypeSafe publica uma versão nova.
- Cada export grava as perguntas usadas (`question_sets`) e uma impressão
  digital delas (`instrument_sha`). Rodadas com `instrument_sha` diferente não
  usaram o mesmo instrumento. O O6 se recusa a misturá-las.

## Validação do instrumento

Mesma regra de `tests/ragas/test_judge_controls.py`: **se um controle reprova,
as notas da avaliação medem o instrumento, não o RAG.**
[`test_jev_controls.py`](test_jev_controls.py) amarra cada problema documentado
a um caso com resposta conhecida:

| Problema | Controle |
|---|---|
| Alucinação sobre tópico ausente: DeepEval deu 1,0 (§4.9) | fidelidade ≤ 0,3 para "a WCAG 2.2 define 14 critérios" |
| Recusa: a FaithfulnessMetric dá 0,0 | fidelidade e relevância saem **N/A**, sem erro |
| O O1 não separa paráfrase de erro (AUC 0,48) | correção aprova a paráfrase e reprova o número trocado do **mesmo** par (`c01-eq1` × `c01-err1`) |

Também há controles de relevância (resposta fora do assunto ≤ 0,3), de contexto
(trecho de outro assunto ≤ 0,3) e de abstenção (a correção aprova a recusa
quando a referência se abstém e reprova a resposta inventada). Os textos são os
mesmos dos controles do RAGAS, então dá para comparar os dois instrumentos
diretamente.

A **estabilidade** julga o mesmo caso três vezes. A variação máxima tolerada é
0,02. O Jev devolve probabilidades em vez de amostrar texto, então o esperado é
nota idêntica. A folga cobre arredondamento.

A **validade de critério** ([`criterion_validity.py`](criterion_validity.py))
julga a correção nos 247 pares que calibraram o O1. São 111 paráfrases que
preservam os fatos (EQUIV) e 136 respostas com erro injetado (ERRO): troca de
número, troca de entidade ou condição invertida. O conjunto passa pelo mesmo
filtro de `tests/mutation/calibration/calibrate.py`. O relatório traz:

- a AUC, com IC bootstrap de 95%, ao lado da AUC 0,4846 do O1
  (`tau.json`, sobre os mesmos pares);
- a classificação no limiar 0,5, na convenção do `tau.json` (positivo = EQUIV);
- o acerto por regra de construção, isto é, qual tipo de erro o Jev deixa
  passar, e a causa que ele atribui a cada um;
- a varredura de limiares, para comparação com o τ\* do O1. Ela não escolhe o
  limiar.

A ressalva do `tau.json` vale aqui também: os erros foram gerados por regra e
podem ser mais fáceis de detectar que os erros naturais de um LLM. O desempenho
medido é um limite superior.

## O6: correção sobre o log da campanha (exploratório)

[`o6_mutation.py`](o6_mutation.py) aplica a correção à resposta modal de cada
par (mutante, caso) de `tests/mutation/results/runs.jsonl`, sem reexecutar o
assistente: executar não é julgar. As regras são as do `evaluate.py`, replicadas
em [`_jev_mutation.py`](_jev_mutation.py) e testadas contra um log sintético:

- execuções com erro ou cortadas pelo teto de geração ficam de fora;
- o O6 julga a resposta modal, como O3 e O4, com o mesmo `modal_answer`;
- o conjunto avaliável S contém os casos cujo baseline foi medido e passou;
- um mutante morre quando ao menos um caso de S reprova.

**É exploratório porque o O6 nasceu depois da campanha.** Os vereditos ficam em
`tests/jeveval/results/o6/` e **não entram em
`tests/mutation/results/verdicts.jsonl`**. Levar um número daqui para o artigo
exige declará-lo como análise post hoc, na lista de desvios do §6.7. O resumo
compara o O6 com o O4 da campanha no mesmo nível: os dois escores de mutação,
com IC de Wilson, e a concordância caso a caso, com kappa de Cohen. Ele também
traz uma amostra das divergências, com a justificativa do O4 e a causa do O6.

A rodada é retomável. Cada julgamento é gravado assim que sai, e
`--summary-only` refaz vereditos e resumo sem chamar o Jev. O log da campanha
não é versionado: ele vem do pacote de replicação
(`tests/mutation/results/replication-*.zip`).

## Formato do export

`results/<timestamp>.json` segue o formato do DeepEval e do RAGAS: `cases[]`,
com `name`, `input`, `actual_output`, `expected_output`, `retrieval_context` e
`metrics[]`. Assim o relatório consolidado cruza os três frameworks pelo nome do
caso. Cada métrica acrescenta:

- `applicable`: false = N/A;
- `confidence`: a resposta menos decidida do Jev entre as perguntas;
- `breakdown`: probabilidades e valor de cada pergunta;
- `cause`: só na correção.

No topo do arquivo vêm `instrument_sha`, `question_sets`, `question_sources`,
`jev_calls`, `jev_failures` e `cost_usd`. O `results/<timestamp>.csv` tem uma
linha por caso × métrica.

A correção não é uma das cinco dimensões do relatório consolidado. Ela fica no
JSON e no CSV daqui.

## Limitações

- **Ainda não há medida real.** Ver "Estado", no topo.
- **O Jev é um serviço externo e fechado.** Não há como inspecionar o modelo,
  só o comportamento. Os controles e a validade de critério são a única base
  para confiar nele.
- **A correção é uma pergunta só.** Isso dá uma leitura limpa (a nota é a
  probabilidade de estar certa), mas não tem a média de várias perguntas para
  amortecer um erro. Se a estabilidade ou o critério forem ruins, o próximo
  passo é acrescentar perguntas, e não trocar o limiar.
- **10 goldens** é pouco para afirmar algo estatístico, a mesma ressalva das
  outras camadas. A validade de critério, com 247 pares, é a parte com
  tamanho de amostra.

## Atualizar o deepeval

O JevEval tinha dias quando esta camada foi escrita, e a API pode mudar. Ao
subir a versão em `requirements.txt`, rode os herméticos. Se as perguntas do
system_one mudarem, `test_perguntas_de_contexto_iguais_as_do_deepeval` falha, e
cabe decidir se o instrumento acompanha. Se acompanhar, o `instrument_sha` muda
e as rodadas antigas deixam de ser comparáveis. Registre a mudança aqui.
