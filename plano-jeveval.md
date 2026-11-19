# Plano: JevEval (Jev-as-a-Judge) como camada de testes de qualidade do RAG

> **Fontes.** Post de lançamento: <https://deepeval.com/blog/introducing-jev-as-a-judge>,
> com as páginas relacionadas `docs/evaluation-eval-modes`,
> `integrations/models/typesafe-ai` e `blog/introducing-jev-in-deepeval`.
> O domínio `deepeval.com` estava bloqueado na rede da sessão em que este plano foi
> escrito (17/10/2026), então o conteúdo do post entrou só pelo resumo da busca. **Toda
> afirmação técnica abaixo foi conferida no código-fonte do `deepeval==4.3.1`** (PyPI,
> publicado em 16/10/2026: `deepeval/metrics/jev_eval/`,
> `deepeval/metrics/conversational_jev_eval/`, `deepeval/models/system_one/`,
> `deepeval/config/eval_mode.py`) e nos metadados do `typesafe-sdk==0.7.5`. Os trechos de
> código das seções 6 e 7.1 foram executados contra essa versão com um Jev falso (sem
> rede). Antes de implementar, reconfira tudo na doc oficial: o recurso tem dias e a API
> pode mudar.

## 0. Resumo

O JevEval é uma métrica do DeepEval. Nela a decisão semântica fica com o **Jev**, o
modelo "System One" da TypeSafe AI. O Jev não gera texto: responde perguntas fechadas
com probabilidades calibradas. A nota sai de uma conta fixa, feita fora do modelo. O
plano adiciona uma **terceira camada de qualidade**, `tests/jeveval/`, ao lado do
DeepEval com juiz local e do RAGAS. Ela é dividida em três partes:

1. **Testes herméticos.** Um Jev falso valida as perguntas, os pesos, a agregação e o
   formato do export. Rodam no CI do GitHub em todo PR, sem rede e sem chave.
2. **Validação do instrumento.** Os controles com resposta conhecida rodam antes de
   qualquer nota ser lida, como já acontece em `tests/ragas/test_judge_controls.py`.
   Somam-se uma checagem de estabilidade e uma validade de critério sobre os **247
   pares rotulados à mão** do estudo de mutação.
3. **Avaliação do RAG.** Os goldens são julgados sobre as respostas compartilhadas de
   `tests/run_all.py`, com o mesmo texto que DeepEval e RAGAS julgam. Depois, como
   análise exploratória, o oráculo **O6** é aplicado sobre o log da campanha de
   mutação, sem reexecutar o SUT.

Três problemas que o projeto já documentou motivam a camada. O JevEval deve ser testado
contra cada um deles (seção 2):

- Na alucinação sobre tópico ausente (§4.9 do estudo de mutação), a `FaithfulnessMetric`
  com juiz LLM dá **1,0**.
- O oráculo O1 (cosseno) tem **AUC 0,48** nos pares rotulados
  (`tests/mutation/calibration/tau.json`): não separa paráfrase de erro.
- Quando o assistente responde "não encontrei", a Faithfulness pontua a recusa como
  **0,0**, quando o certo seria N/A.

## 1. Como o JevEval funciona (conferido no código 4.3.1)

### 1.1 Peças

| Peça | Onde | O que faz |
|---|---|---|
| `JevEval` | `deepeval.metrics` | Métrica single-turn. Recebe `name`, `evaluation_params`, `questions`, `threshold` (padrão `0.5`), `strict_mode`, `system_one_model`, `include_reason`, `async_mode` |
| `ConversationalJevEval` | `deepeval.metrics` | Versão multi-turn, sobre `ConversationalTestCase`. `ROLE`/`CONTENT` entram sempre no estado |
| `Noul`, `Score`, `Choice` | `deepeval.metrics.jev_eval` (**não** em `deepeval.metrics`) | Os três tipos de pergunta |
| `TypeSafeModel` | `deepeval.models` | Cliente do Jev. Lê `TYPESAFE_API_KEY` e `TYPESAFE_MODEL_NAME`. Exige `pip install typesafe-sdk` |
| `DeepEvalBaseSystemOneModel` | `deepeval.models` | Classe base. Uma subclasse com `decide()`/`a_decide()` próprios substitui o Jev, e é assim que entra o Jev falso dos testes herméticos |
| `eval_mode` | métricas nativas (`FaithfulnessMetric` etc.) | `llm` (padrão): o LLM decide tudo. `hybrid`: o LLM extrai e o Jev decide. `system_one`: o Jev faz tudo numa chamada, sem LLM |

### 1.2 Os três tipos de pergunta e como viram número

| Tipo | Uso | Valor `v` em [0,1] | Restrições |
|---|---|---|---|
| `Noul(statement, true=?, false=?, weight=1)` | Uma proposição verdadeira ou falsa | `v = P(true)` | Texto não vazio. `true`/`false` são critérios opcionais que dizem o que cada lado significa |
| `Score(question, levels=[...], weight=1)` | Um juízo holístico em níveis **do pior para o melhor** | `v = nível esperado / (n_níveis − 1)` | De 2 a 10 níveis, únicos |
| `Choice(question, options={nome: crédito}, weight=1)` | Uma escolha num conjunto fechado | Média dos créditos ponderada pelas probabilidades | Pelo menos 2 opções, crédito em [0,1] ou `None`. `None` significa "não se aplica": se essas opções somam ≥ 0,5 de probabilidade, a pergunta sai da conta. **O Jev só vê os nomes das opções, nunca os créditos** |

- **Agregação:** `score = Σ(wᵢ·vᵢ) / Σwᵢ`, só sobre as perguntas aplicáveis. Se nenhuma
  se aplica, o score é **1,0** (`EMPTY_SCORE`). Ver o cuidado C2 na seção 6.0.
- **`strict_mode=True`:** o limiar vira 1 e o score é 0 ou 1. Cada pergunta aplicável
  precisa passar: num `Noul`, `P(true) ≥ 0,5`; num `Score`, o nível mais provável tem
  de ser o último; num `Choice`, a opção mais provável tem de ter crédito 1,0.
- **`metric.confidence`:** é a **menor** confiança entre as perguntas. Num `Noul` vale
  `|2p − 1|`; em `Choice` e `Score`, vem da API.
- **`metric.score_breakdown`:** lista uma linha por pergunta, com `value`, `applicable`,
  `probabilities`, `confidence` e `passed`. É o que torna cada decisão auditável.
- **`metric.reason`:** é um texto **determinístico** montado a partir das respostas. No
  4.3.1, nenhum LLM é chamado em nenhum ponto do JevEval.
- **Estado enviado ao Jev:** `{"test_case": {<campo>: <valor>}}`, só com os campos de
  `evaluation_params` que estão preenchidos. Por convenção (os templates do próprio
  deepeval fazem assim), as perguntas citam os campos pelo nome entre crases:
  `` `actual_output` ``, `` `retrieval_context` ``.
- **Sem `evaluation_params`:** o estado passa a ser o *trace* (`@observe`), e a métrica
  avalia a trajetória de um agente.

### 1.3 Limites operacionais

- **Contexto.** O estado mais a pergunta mais longa cabem em 32k tokens, e a requisição
  inteira em 64k. A conta é estimada antes de chamar a API, com 3 caracteres por token,
  e falha com `SystemOneContextLimitError`.
- **Só texto.** O Jev não recebe imagem.
- **Modelos:** `jev-1.13.0`, `jev-latest` (padrão) e `jev-preview`. Hoje todos os
  aliases apontam para o 1.13.
- **Preço** (tabela embutida no deepeval): US$ 0,042 por milhão de tokens de entrada. A
  saída não é cobrada.
- **Versões.** O JevEval existe a partir do `deepeval 4.2.9`. **O `deepeval==4.2.0`
  pinado hoje em `backend/requirements-eval.txt` não tem a métrica.**
- **SDK.** `typesafe-sdk` exige `pydantic>=2.12` e Python 3.10 ou mais novo.
- **Sem chave ou sem SDK, o JevEval falha já no construtor** (`DeepEvalError`). Isso
  ajuda: dá para decidir o skip antes de coletar os testes.

## 2. Por que isso cabe neste projeto

| Achado já documentado no repo | Onde | O que o JevEval permite testar |
|---|---|---|
| A `FaithfulnessMetric` (juiz LLM) dá 1,0 para alucinação sobre tópico ausente, porque conta afirmações que *não contradizem* o contexto | `docs/ARCHITECTURE.md` (RAGAS, out/2026); `tests/mutation/config/study.yaml` (O3 fora da campanha) | Uma pergunta escrita como "**pode ser inferida** de `retrieval_context`" em vez de "não contradiz". O próprio template de Faithfulness em `system_one` já pergunta "*supported by*" com peso 2 |
| O3 saiu da campanha à espera de "um juiz viável", e o comentário em `study.yaml` diz que ele pode ser rejulgado sobre o log | `study.yaml`, seção `oracles` | O6 (Jev) rejulgado sobre `runs.jsonl`, sem tocar GPU de geração |
| Recusa ("não encontrei") pontuada como Faithfulness 0,0, quando deveria ser N/A | ARCHITECTURE, calibração inicial (`ares-tres-dimensoes`) | Um `Choice` com a opção "se absteve" com crédito `None` |
| O O1 (cosseno) tem AUC 0,48 nos 247 pares EQUIV/ERRO rotulados à mão: não distingue número trocado, entidade trocada nem condição invertida | `tests/mutation/calibration/tau.json` | Um teste de validade de critério barato: são só chamadas ao Jev, sem SUT |
| `gemma3:4b` reprovado como juiz. O juiz local leva minutos por caso, devolve JSON truncado, entra em loop de *thinking* e precisa de timeout por thread | `tests/deepeval/_shared.py`, `run_and_export.py`, `tests/ragas/README.md` | Uma requisição por métrica, sem parse de texto gerado, sem GPU e com custo de centavos |
| O O4 faz 5 julgamentos e maioria simples por causa da instabilidade do juiz | `tests/mutation/oracles/o4_judge.py` | Medir se o Jev é estável. Se for, 1 chamada basta, o que é um dado de custo para a RQ3 |
| A regra "métricas de qualidade nunca vêm do CI" existe porque elas precisam de GPU | ARCHITECTURE, "Testes determinísticos vs. testes que exercitam o modelo" | A primeira camada de julgamento que **pode** rodar no CI, sobre respostas congeladas, sem GPU (seção 8.3) |

## 3. Decisões que o plano assume (e o que pode mudar nelas)

**D1. Exceção ao "100% local", restrita e opt-in.** O Jev é um serviço hospedado: o
texto do caso (pergunta, resposta e trechos) sai da máquina. Por isso:

- a camada vem **desligada por padrão** e se auto-pula sem `TYPESAFE_API_KEY`;
- só julga o **corpus público** (os PDFs acadêmicos de `data/source_pdfs/` e
  `tests/mutation/corpus/`), **nunca** dados da organização parceira;
- a exceção é registrada em `docs/ARCHITECTURE.md`, como os outros trade-offs do projeto.

**D2. Venv próprio, `tests/jeveval/.venv`, com `deepeval==4.3.1` e `typesafe-sdk==0.7.5`.**
**Não** atualizar `tests/deepeval` (4.2.0) agora. Uma troca de versão mexe nos prompts
das métricas e quebra a série histórica do relatório consolidado. A comparação "juiz
LLM × Jev" é feita **dentro do mesmo venv e da mesma versão** (seção 7.4), para isolar
o efeito do juiz do efeito da versão. Esta é a mesma regra de isolamento das outras
suítes, já justificada pelo conflito de `pydantic`.

**D3. Modelo pinado.** `TYPESAFE_MODEL_NAME=jev-1.13.0`, não `jev-latest`, porque o alias
anda. A versão vai em todo export (`judge_model`). Só se troca de versão rodando de novo
os controles da seção 7.2.

**D4. `eval_mode` sempre explícito em cada métrica nativa, e nunca `DEEPEVAL_EVAL_MODE`
no `.env`.** `resolve_eval_mode()` lê essa variável global. Se ela vazar para o `.env`,
qualquer métrica construída sem `eval_mode` muda de juiz em silêncio. Isso vira teste
(seção 7.1, U9).

**D5. Idioma das perguntas decidido por experimento, não por palpite.** O corpus e as
respostas são em português. Não há garantia de que o Jev responda igual a perguntas em
PT e em EN. Os controles da seção 7.2 rodam nas duas versões, e a que passar com folga
maior vira o padrão (registrado no export: `questions_lang`). A D5 só vale para a trilha
B. As perguntas das 5 métricas nativas em `system_one` (trilha A) são fixas e em inglês,
nos templates `_experimental_system_one_questions.txt` do deepeval.

**D6. O O6 no estudo de mutação é análise pós-hoc e exploratória.** O protocolo 1.0 e o
MS publicado não mudam. O O6 entra numa tabela à parte, declarada como tal.

## 4. Escopo

**Dentro:**

- As métricas JevEval do domínio, M1 a M6 (seção 6).
- As 5 métricas RAG nativas em `eval_mode="system_one"` (trilha A). Elas dão uma coluna
  "Jev" comparável, dimensão por dimensão, com DeepEval-LLM e RAGAS.
- Testes herméticos (CI), controles do instrumento, estabilidade e validade de critério.
- Avaliação dos goldens, export JSON, integração em `run_all.py` e no relatório
  consolidado.
- O oráculo O6 pós-hoc sobre o log da campanha de mutação.

**Fora** (citado, não implementado agora):

- `ConversationalJevEval`. O `/chat` não tem contrato multi-turn: o histórico só existe
  no `localStorage` do frontend, a mesma razão que já tirou o multi-turn do `plano.md`.
- JevEval sobre *trace* (`@observe`). O pipeline é fixo (recupera e depois gera), sem
  escolha de ferramenta. O ganho sobre M1 e M2 seria pequeno. Fica como extensão se o
  assistente virar agente.
- `eval_mode="hybrid"` nas métricas nativas. Mistura juiz local (extração) com Jev
  (decisão) e embaralha a atribuição de causa. Só entra se a trilha A der resultado
  ambíguo.
- Expor nota do Jev em `GET /stats`. É opcional, ver a Fase 7.

## 5. Estrutura de arquivos

```
tests/jeveval/
├── README.md                 o que roda, como, por quê, custo e a exceção D1
├── requirements.txt          deepeval==4.3.1, typesafe-sdk==0.7.5, pytest + o que o SUT importa
│                             (chromadb, httpx, pydantic-settings, PyMuPDF, tiktoken; sem pin de pydantic)
├── requirements-unit.txt     só deepeval==4.3.1 + pytest (CI hermético, sem SDK do Jev de propósito)
├── conftest.py               mínimo: NÃO importa `app` (os testes unitários rodam sem o backend)
├── _jev_metrics.py           registro único das métricas M1–M6 e da trilha A (perguntas, pesos, limiares)
├── _jev_shared.py            build_jev_model(), jev_status() (motivo de skip), constantes, NOT_FOUND_MARKER
├── _fake_jev.py              FakeJev(DeepEvalBaseSystemOneModel) roteirizado, sem rede
├── _jev_export.py            métricas → dicionário do export (formato da seção 9)
├── unit/                     herméticos, rodam no CI
│   ├── test_questions.py
│   ├── test_scoring.py
│   ├── test_state_budget.py
│   ├── test_export_contract.py
│   └── test_no_network.py
├── live/                     chamam o Jev de verdade; skip sem chave
│   ├── conftest.py
│   ├── test_jev_controls.py      controles com resposta conhecida (sem SUT)
│   ├── test_jev_stability.py     mesma entrada N vezes
│   ├── test_jev_criterion.py     247 pares EQUIV/ERRO (validade de critério)
│   └── test_rag_quality_jev.py   goldens × métricas (stack real ou respostas congeladas)
├── fixtures/
│   └── respostas_congeladas.json  uma rodada de tests/generate_answers.py, versionada (corpus público)
├── run_and_export.py         análogo aos das outras suítes, aceita --answers / --limit / --track
└── results/.gitkeep
```

`pytest.ini` continua com `testpaths = tests/api`. A suíte nova roda com o caminho
explícito, como `tests/deepeval` e `tests/ragas`. No `.gitignore`, acrescentar
`tests/jeveval/results/*` e `!tests/jeveval/results/.gitkeep`.

## 6. Catálogo de métricas

### 6.0 Cuidados de desenho (vieram do protótipo, não da doc)

- **C1. Num `Noul`, "verdadeiro" sempre significa "bom".** O valor é `P(true)`. Um
  `Noul("A resposta inventa fatos")` inverte a nota sem erro nenhum. Toda proposição é
  escrita no sentido desejável, e um teste unitário (U2) cobra isso pelo registro.
- **C2. N/A não pode virar 1,0.** Quando nenhuma pergunta se aplica, o JevEval devolve
  `score = 1.0`. Uma abstenção julgada só por `Choice` com opção `None` passaria com
  nota cheia. O export marca o caso como **N/A** (`score = null`, `not_applicable = true`)
  quando `all(not o["applicable"] for o in score_breakdown)`. Teste U5.
- **C3. Um `Choice` N/A não anula os `Noul` da mesma métrica.** No protótipo, uma
  abstenção derrubou o `Choice`, mas o `Noul` "toda afirmação pode ser inferida" ficou
  vacuamente verdadeiro e a nota foi 0,93. Por isso a abstenção em M1 é decidida
  **antes**, de forma determinística: se a resposta contém `NOT_FOUND_MARKER`, M1 é
  N/A e nem chama o Jev. O `Choice` com `None` vira a segunda linha de defesa, para
  abstenções reformuladas. Isso implementa a "próxima iteração" que o ARCHITECTURE já
  apontava ("excluir respostas `grounded=False` da amostra de Faithfulness").
- **C4. Métrica nativa sem `model`/`eval_mode` cai na OpenAI** e falha sem
  `OPENAI_API_KEY`. Toda métrica nativa da suíte recebe `eval_mode` explícito (D4).

### 6.1 Trilha A: métricas nativas em `system_one`

As mesmas 5 dimensões do relatório consolidado, agora decididas pelo Jev. Os nomes no
export continuam `Faithfulness`, `Answer Relevancy` e assim por diante, e o campo
`framework: "jev"` distingue a rodada (seção 9).

```python
from deepeval.metrics import (
    AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric,
    ContextualRelevancyMetric, FaithfulnessMetric,
)

def native_system_one(jev, threshold: float) -> list:
    kw = dict(eval_mode="system_one", system_one_model=jev, threshold=threshold, async_mode=False)
    return [
        FaithfulnessMetric(**kw),
        AnswerRelevancyMetric(**kw),
        ContextualPrecisionMetric(**kw),
        ContextualRecallMetric(**kw),
        ContextualRelevancyMetric(**kw),
    ]
```

Em `system_one`, a Faithfulness nativa pergunta: (1) "*Every factual claim … is
supported by …*", com peso 2; (2) "*No claim … contradicts …*"; (3) um `Score` de 4
níveis. Para a alucinação do §4.9, a hipótese é (1) ≈ 0, (2) ≈ 1 e (3) ≈ 0, o que dá
nota em torno de 0,25. Fica perto do limite de 0,3 usado nos controles. É exatamente o
que o controle K2 (seção 7.2) mede, e é a razão de existir a M1 customizada (sem a
pergunta de contradição).

### 6.2 Trilha B: JevEval do domínio

Em `_jev_metrics.py`, cada métrica é uma função `build_mX(jev) -> JevEval`, com
`async_mode=False` (sequencial, igual ao `run_and_export.py` do DeepEval). Limiar
inicial de **0,7**, alinhado às outras suítes. É ponto de partida, a ser recalibrado
na Fase 3.

**M1. Fundamentação** (dimensão `faithfulness`; mata P1 e parte de C2/C3).
Params: `INPUT`, `ACTUAL_OUTPUT`, `RETRIEVAL_CONTEXT`. Não se aplica a abstenções (C3).

```python
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.test_case import SingleTurnParams as P

ABSTEVE = "se absteve, dizendo que não encontrou a informação"

def build_m1(jev) -> JevEval:
    return JevEval(
        name="Fundamentação",
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
        questions=[
            Noul("Toda afirmação factual de `actual_output` pode ser inferida dos trechos em `retrieval_context`.",
                 weight=3),
            Choice("Como `actual_output` usa `retrieval_context`?", options={
                "responde só com fatos presentes em `retrieval_context`": 1.0,
                "mistura fatos de `retrieval_context` com fatos ausentes dele": 0.3,
                "responde com fatos ausentes de `retrieval_context`": 0.0,
                ABSTEVE: None,
            }, weight=2),
            Score("Quanto de `actual_output` está apoiado em `retrieval_context`?",
                  levels=["nada", "pouco", "a maior parte", "tudo"]),
        ],
        threshold=0.7, system_one_model=jev, async_mode=False,
    )
```

**M2. Comportamento de abstenção** (sem dimensão equivalente hoje; mata P2, R1, R4, C1,
E2). Params: `INPUT`, `ACTUAL_OUTPUT`, `RETRIEVAL_CONTEXT`. `strict_mode=True`: o
comportamento ou está certo ou está errado.

```python
Choice("Qual alternativa descreve o que o assistente fez, dado `retrieval_context`?", options={
    "respondeu, e `retrieval_context` contém a informação usada": 1.0,
    "se absteve, e `retrieval_context` de fato não contém a resposta": 1.0,
    "se absteve, mas `retrieval_context` contém a resposta": 0.0,       # falsa recusa (qwen3, ares-tres-dimensoes)
    "respondeu, mas `retrieval_context` não contém a resposta": 0.0,    # deveria ter se abstido
})
```

**M3. Correção contra a referência** (sem dimensão equivalente; é o papel do O4, agora
sem LLM gerativo). Params: `INPUT`, `ACTUAL_OUTPUT`, `EXPECTED_OUTPUT`. O texto segue a
lição de calibração do O4: **a referência é o mínimo, não o máximo**, e detalhe a mais
não reprova.

```python
[
    Noul("`actual_output` contém os mesmos valores numéricos, entidades e condições essenciais de `expected_output`.",
         weight=3),
    Noul("Nenhuma afirmação de `actual_output` contradiz `expected_output`.", weight=2),
    Score("Quanto de `expected_output` está coberto por `actual_output`?",
          levels=["nada", "parte", "o essencial", "tudo"]),
]
```

**M4. Relevância da resposta** (dimensão `answer_relevancy`). Params: `INPUT`,
`ACTUAL_OUTPUT`.

```python
[
    Noul("`actual_output` responde diretamente à pergunta em `input`.", weight=2),
    Choice("Em que idioma está `actual_output`?",
           options={"português": 1.0, "misto": 0.5, "inglês": 0.0, "outro": 0.0}),
]
```

**M5. Citação** (mata P3; **só com `PROMPT_REQUIRE_CITATION=true`**, que é o caso do
estudo de mutação; nos goldens, em que a regra vem desligada, a métrica é pulada e
registrada como N/A). Params: `ACTUAL_OUTPUT`, `RETRIEVAL_CONTEXT`.

```python
Choice("Como `actual_output` cita a fonte?", options={
    "cita documento e página que aparecem em `retrieval_context`": 1.0,
    "cita o documento, sem página": 0.5,
    "cita uma fonte que não está em `retrieval_context`": 0.0,
    "não cita fonte": 0.0,
    ABSTEVE: None,
})
```

**M6. Qualidade da recuperação** (dimensões `context_recall` e `context_relevance`; mata
K1–K4, E1, R1–R3). Params: `INPUT`, `RETRIEVAL_CONTEXT`, `EXPECTED_OUTPUT`.

```python
[
    Score("Quanto da informação de `expected_output` está presente em `retrieval_context`?",
          levels=["nada", "parte", "o essencial", "tudo"], weight=2),
    Noul("O primeiro trecho de `retrieval_context` é relevante para a pergunta em `input`."),
    Choice("Qual fração dos trechos de `retrieval_context` é relevante para `input`?",
           options={"todos": 1.0, "a maioria": 0.75, "metade": 0.5, "a minoria": 0.25, "nenhum": 0.0}),
]
```

### 6.3 Mapa métrica × mutante (hipóteses do O6, seção 7.6)

| Família | Mutantes | Métricas que deveriam detectar |
|---|---|---|
| Corpus | C1 remover doc, C2 desatualizado, C3 truncar, C4 duplicata conflitante | M2 (abstenção certa), M3 (valor errado), M6 |
| Chunking | K1–K4 | M6, M3 |
| Embedding/índice | E1 trocar modelo, E2 índice obsoleto | M6, M2 |
| Recuperação | R1 top-k=1, R2 top-k×3, R3 sem MMR, R4 sem limiar | M6, M2 (R4: deveria abster e respondeu) |
| Prompt | P1 sem ancoragem, P2 sem abstenção, P3 sem citação, P4 temperatura 0,8 | M1, M2, M5, M3 (P4) |

## 7. Plano de testes

### 7.1 Herméticos (`unit/`, CI em todo PR, sem rede)

Base: um `FakeJev` roteirizado. Ele já foi validado contra o 4.3.1 no protótipo deste
plano: o `JevEval` e a `FaithfulnessMetric(eval_mode="system_one")` aceitam a subclasse
sem SDK e sem chave.

```python
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion, SystemOneAnswers,
)

class FakeJev(DeepEvalBaseSystemOneModel):
    """Responde com probabilidades roteirizadas por texto de pergunta; nunca toca a rede."""

    def __init__(self, script: dict | None = None):
        self.script, self.calls = script or {}, []
        super().__init__("fake-jev")

    def load_model(self):
        return None

    def decide(self, state, questions):
        self.calls.append((state, questions))
        answers = SystemOneAnswers()
        for key, q in questions.items():
            s = self.script.get(q.instructions)
            if isinstance(q, NoulQuestion):
                answers.nouls[key] = NoulAnswer(probability=0.9 if s is None else s)
            elif isinstance(q, ChoiceQuestion):
                probs = s or {name: 1 / len(q.options) for name in q.options}
                answers.choices[key] = ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs, confidence=0.8)
            elif isinstance(q, ScoreQuestion):
                probs = s or {len(q.levels) - 1: 1.0}
                answers.scores[key] = ScoreAnswer(score=sum(l * p for l, p in probs.items()),
                                                  probabilities=probs, confidence=0.8)
        return answers, None

    async def a_decide(self, state, questions):
        return self.decide(state, questions)

    def get_model_name(self):
        return "fake-jev"
```

| # | Teste | Verifica |
|---|---|---|
| U1 | `test_questions::test_registro_constroi` | Todas as métricas M1–M6 e a trilha A são construídas com o `FakeJev`, sem `ValueError` de validação (níveis 2–10, ≥ 2 opções, crédito em [0,1]) |
| U2 | `test_questions::test_noul_no_sentido_desejavel` | Cada `Noul` do registro tem um metadado `desejavel=True` revisado à mão, e a lista de exceções está vazia. Uma revisão de PR que adicione `Noul` sem o metadado quebra o teste (C1) |
| U3 | `test_questions::test_campos_citados_existem` | Todo `` `campo` `` citado no texto de uma pergunta está em `evaluation_params` da métrica. Sem isso o Jev recebe pergunta sobre um campo que não está no estado |
| U4 | `test_scoring::test_media_ponderada` | Probabilidades roteirizadas dão o score calculado à mão (ex.: M1 com Noul 0,9 × 3, Choice N/A e Score "tudo" × 1 → 0,925) |
| U5 | `test_scoring::test_tudo_na_vira_na_no_export` | Com todas as perguntas N/A, o JevEval devolve 1,0 e o export grava `score=null, not_applicable=true` (C2) |
| U6 | `test_scoring::test_abstencao_pula_m1` | Uma resposta com `NOT_FOUND_MARKER` não chama o Jev em M1 (`fake.calls == []`) e sai N/A (C3) |
| U7 | `test_scoring::test_strict_m2` | Em M2, falsa recusa dá 0 e abstenção correta dá 1 (`strict_mode`) |
| U8 | `test_state_budget::test_pior_caso_cabe` | O estado do pior caso de configuração (K2: chunk 3000 × top-k 4; R2: top-k 12 × 1200) passa em `check_context_budget` com folga de pelo menos 30% |
| U9 | `test_no_network::test_eval_mode_explicito` | Com `DEEPEVAL_EVAL_MODE=llm` no ambiente, toda métrica nativa do registro continua em `system_one` (D4) |
| U10 | `test_no_network::test_sem_sdk_no_ci` | `typesafe_sdk` não está instalado no job hermético, então qualquer caminho que tente construir `TypeSafeModel` quebra o teste em vez de chamar a rede |
| U11 | `test_export_contract::test_relatorio_le_o_export` | Um export gerado com o `FakeJev` é lido por `tests/reports/report_data.load_run` como `framework="Jev"`, com as 5 dimensões preenchidas (seção 9). O import de `report_data` não puxa dependências pesadas, como já documentado lá |
| U12 | `test_export_contract::test_breakdown_serializa` | `score_breakdown`, `confidence` e `judge_model` aparecem no JSON e voltam com `json.loads` |

### 7.2 Controles do instrumento (`live/test_jev_controls.py`)

Mesmo princípio de `tests/ragas/test_judge_controls.py`: os casos têm resposta certa
conhecida e **não passam pelo SUT**. Se um controle reprova, as notas da seção 7.5 não
são lidas como medida de qualidade. Os textos reaproveitam os de lá (golden
`ragas-definicao`), para que os três frameworks sejam comparados no mesmo controle.

| # | Caso | Métrica | Esperado |
|---|---|---|---|
| K1 | Resposta que só repete o trecho | M1, Faithfulness `system_one` | ≥ 0,7 |
| K2 | **Alucinação sobre tópico ausente** ("A WCAG 2.2 define 14 critérios…") | M1 | ≤ 0,3. **Registrar também** a nota da Faithfulness nativa em `system_one` (hipótese ~0,25; a seção 6.1 explica) |
| K3 | Resposta fora do assunto ("A capital da França…") | M4 | ≤ 0,3, e distância de pelo menos 0,4 para a resposta relevante (o Jev não tem o piso de cosseno do RAGAS) |
| K4 | Trecho de outro assunto | M6, Contextual Relevancy `system_one` | ≤ 0,3 |
| K5 | Falsa recusa: abstenção com o trecho certo em `retrieval_context` | M2 | 0 |
| K6 | Abstenção correta: abstenção com trecho irrelevante | M2 | 1 |
| K7 | Respondeu sem apoio (o trecho não tem a resposta) | M2 | 0 |
| K8 | Abstenção julgada por M1 | M1 | N/A (C3), não 0 nem 1 |
| K9 | Número trocado ("16 classes" → "61 classes") | M3 | ≤ 0,3 |
| K10 | Resposta correta e **mais longa** que a referência | M3 | ≥ 0,7 (a lição de calibração do O4) |
| K11 | Os controles K1–K10 com as perguntas em EN | todas | Mesmo veredito. A diferença de nota entre PT e EN é registrada, e é ela que decide D5 |

Cada controle grava `score`, `confidence` e `score_breakdown` num JSON de execução. A
leitura do instrumento fica rastreável, não só o pass/fail.

### 7.3 Estabilidade (`live/test_jev_stability.py`)

- O mesmo caso, 5 vezes, para K1, K2, K5 e K9: a maior diferença absoluta de
  probabilidade por pergunta fica **≤ 0,02**.
- Se passar, a avaliação usa 1 chamada por métrica. É um resultado de custo para a RQ3,
  em contraste com os 5 julgamentos do O4.
- Se não passar, o resultado é registrado, e o `run_and_export.py` ganha `--repeticoes N`
  com mediana.

### 7.4 Validade de critério e concordância

**Validade de critério** (`live/test_jev_criterion.py`): M3 sobre os 247 pares revisados
de `tests/mutation/calibration/pairs.jsonl` (111 EQUIV, 136 ERRO; regras `paraphrase`,
`swap_number`, `replace_entity`, `invert_condition`). Os 4 pares não revisados ficam de
fora, como no `tau.json`.

- Mapeamento: `input` = pergunta do caso (`golden.jsonl`), `expected_output` =
  `reference`, `actual_output` = `candidate`.
- Métricas: AUC, F1 e acurácia balanceada, com IC por bootstrap (o mesmo método de
  `calibration/calibrate.py`), **por regra**.
- **Critério de aceite pré-registrado**, antes de rodar: AUC ≥ 0,80 e acurácia
  balanceada ≥ 0,75. Referência: o O1 tem AUC 0,48 e acurácia 0,45 nesse conjunto.
- O teste pytest só confere o critério. Os números completos vão para
  `results/criterion_<ts>.json`.

**Concordância entre juízes** (no relatório, não em pytest), sobre as **mesmas
respostas** (`--answers`):

- Jev (trilha A) × DeepEval-LLM × RAGAS, por dimensão: Spearman nas notas e Cohen κ no
  pass/fail ao limiar. `report_data.compare()` hoje compara dois runs; com três, faz os
  três pares.
- **Controle de versão (D2):** a trilha A roda também em `eval_mode="llm"` com o mesmo
  juiz local (`_NoThinkOllamaModel`, `gemma3:4b`/`qwen3:8b`) **no venv 4.3.1**. A
  diferença "4.3.1-LLM × 4.3.1-Jev" é o efeito do juiz. A diferença "4.2.0-LLM ×
  4.3.1-LLM" é o efeito da versão, e fica registrada para quem um dia atualizar
  `tests/deepeval`.

### 7.5 Avaliação dos goldens (`live/test_rag_quality_jev.py` + `run_and_export.py`)

- **pytest:** é parametrizado pelos goldens, como `tests/deepeval/test_rag_quality.py`.
  A fonte da resposta vem em ordem de preferência: (1) `JEVEVAL_ANSWERS=<arquivo>`; (2)
  pipeline in-process, com o mesmo `run_pipeline` e o mesmo `stack_status()` de lá,
  se a stack estiver no ar; (3) `fixtures/respostas_congeladas.json`. Por isso a suíte
  roda **sem GPU**.
- **Asserção:** `assert_test(test_case, metrics)` com M1–M4 e M6. M5 só entra com a
  citação ligada. Casos N/A (C2/C3) são pulados com `pytest.skip`, nunca contados como
  aprovados.
- **`run_and_export.py`:** usa as mesmas flags das outras suítes (`--answers`,
  `--limit`, `--no-open`) e acrescenta `--track A|B|AB` (padrão `AB`). Mede caso a caso,
  sequencialmente, e regrava o JSON a cada caso, com o mesmo padrão resiliente de
  `tests/deepeval/run_and_export.py`. Erros viram registro, não derrubam a rodada. O teto
  por chamada é de **60 s**: o Jev é uma API, não um juiz local, e 300 s esconderia
  problema de rede.

### 7.6 O6: Jev no estudo de mutação (pós-hoc, D6)

- `tests/mutation/oracles/o6_jev.py` segue o contrato de `base.Verdict`:
  - `score` é a nota agregada;
  - `components` traz o `score_breakdown` por métrica (M1, M2, M3, M5, M6);
  - `evaluable=False` se a chave ou a API faltarem. Assim a falha de infraestrutura não
    infla mortes, a mesma regra do O5.
- Veredito: `fail` se alguma métrica ficar abaixo do limiar (conjunção, como o O5).
  Os vereditos por métrica ficam guardados, para a análise por família.
- Roda **só sobre a resposta modal**, como O3 e O4 (`MODAL_ORACLES`), via
  `python -m tests.mutation.evaluate --oracles O6`, sobre o `runs.jsonl` existente.
  **Não reexecuta o SUT.**
- Dependências: o O6 importa `deepeval` e `typesafe_sdk` de forma preguiçosa, como o
  backend `deepeval` do O3. Roda a partir de `tests/jeveval/.venv` com `PYTHONPATH`
  apontando para a raiz, sem mexer em `tests/mutation/requirements.txt`.
- Saída: `results/verdicts_o6.jsonl`, separado, mais uma tabela à parte com:
  - o MS do O6;
  - o swing contra O4 (McNemar e κ);
  - a matriz de mortes por família (seção 6.3);
  - o custo medido (tempo e tokens) contra o O4.
- `study.yaml` **não** muda: `enabled` continua `[O1, O2, O4, O5]`. O O6 é ligado por
  flag de linha de comando.

### 7.7 Confiança e fila humana (ponte para a Fase 3, HITL)

- Os casos com `confidence < 0,3` em qualquer métrica são marcados `indeciso` no export.
- `run_and_export.py --fila-humana` grava esses casos em
  `tests/human_review/annotations/fila_<ts>.jsonl` (hoje esse diretório é só um
  placeholder). O formato traz a pergunta, a resposta, os trechos, o `score_breakdown` e
  campos vazios para o rótulo humano.
- O valor aqui é a pergunta exata que o Jev achou difícil, não só "o caso teve nota
  baixa".

## 8. Integração no pipeline

### 8.1 `tests/run_all.py`

- `SUITES` ganha `"jeveval"`, `PY["jeveval"]` e `SETUP_HINT["jeveval"]`.
- Novo passo **depois** do DeepEval: "Jev: controles" (pytest `live/test_jev_controls.py`)
  e depois "Jev: avaliação" (`run_and_export.py --answers <respostas compartilhadas>`).
  Os três frameworks julgam o mesmo texto.
- Motivos de pulo explícitos: `TYPESAFE_API_KEY ausente`, `typesafe-sdk não instalado`,
  `venv tests/jeveval/.venv não existe`. **Pular não é falhar.**
- `--rapido` pula o Jev, como as outras avaliações. `--pular jeveval` funciona como os
  demais.
- O passo de relatório passa a buscar também o resultado mais recente de
  `tests/jeveval/results`.
- `tests/reports/test_run_all.py` ganha casos para o pulo sem chave e para o resultado
  novo detectado.

### 8.2 Relatório consolidado (`tests/reports/`)

- `report_data.RESULTS_DIRS["Jev"] = tests/jeveval/results`.
- `detect_framework` checa `data.get("framework") == "jev"` **antes** do teste por nome
  de métrica. Na trilha A, os nomes são os mesmos do DeepEval (`Faithfulness`…) e o
  export seria lido como DeepEval por engano.
- `_METRIC_TO_DIMENSION["Jev"]` e `NATIVE_METRIC["Jev"]` mapeiam as 5 nativas. M1, M4 e
  M6 entram como "variante do domínio" na mesma dimensão, numa linha própria.
- Seção nova, "**Decisões do Jev**": por caso e por métrica, cada pergunta com
  probabilidade e confiança (o `score_breakdown`), além da contagem de N/A e de
  `indeciso`.
- Comparação de três vias (seção 7.4) e os casos de maior divergência entre os três
  juízes.
- `test_build_report.py` ganha uma fixture JSON de um run Jev sintético.

### 8.3 CI (GitHub Actions)

**Job hermético em `ci.yml`**, em todo push e PR:

```yaml
  jeveval-unit:
    name: JevEval — testes herméticos (pytest)
    runs-on: ubuntu-latest
    env:
      DEEPEVAL_TELEMETRY_OPT_OUT: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: tests/jeveval/requirements-unit.txt
      - name: Instalar dependencias (sem typesafe-sdk, de proposito)
        run: pip install -r tests/jeveval/requirements-unit.txt
      - name: Rodar pytest
        run: pytest tests/jeveval/unit
```

**Workflow separado, `.github/workflows/jeveval-live.yml`.** Fica fora de `ci.yml` para
que o agendamento não rode API e E2E de novo à toa.

- Gatilhos: `workflow_dispatch` e `schedule` semanal.
- Roda: controles (7.2), estabilidade (7.3) e as `respostas_congeladas.json` contra M1–M6.
- **Não testa o SUT.** Testa se o **instrumento** mudou: o alias do modelo, o SDK, a
  calibração do Jev.
- `TYPESAFE_API_KEY` vem de `secrets`, e `TYPESAFE_MODEL_NAME=jev-1.13.0`.
- O primeiro passo checa se a chave existe. Sem ela, emite `::warning::` e encerra
  verde, o que acontece por exemplo em fork, que não recebe secrets.
- Nunca roda em `pull_request`: custo e secret ficam fora de PR de terceiros.

A seção "Testes determinísticos vs. testes que exercitam o modelo" do ARCHITECTURE ganha
um terceiro grupo: **"julgamento por API sobre texto congelado"**. Não é determinístico
no sentido do pytest, mas também não depende de GPU nem de geração. Por isso pode rodar
agendado.

### 8.4 `.env.example`

```ini
# --- Avaliação JevEval (tests/jeveval/, opcional — ver tests/jeveval/README.md) ---
# Serviço HOSPEDADO (TypeSafe AI): o texto julgado sai da máquina. Só para o corpus
# público. Vazio = a suíte se auto-pula. NUNCA commite a chave real.
TYPESAFE_API_KEY=
# Pinado: jev-latest anda. Trocar exige rodar os controles de novo.
TYPESAFE_MODEL_NAME=jev-1.13.0
JEVEVAL_THRESHOLD=0.7
# NÃO defina DEEPEVAL_EVAL_MODE aqui: ele troca o juiz de toda métrica sem eval_mode explícito.
```

## 9. Formato do export (`tests/jeveval/results/<AAAAMMDDTHHMMSSZ>.json`)

Mesmo esqueleto dos outros dois, para que o `report_data.load_run` leia sem caso
especial além do `framework`:

```json
{
  "framework": "jev",
  "generated_at": "2026-10-20T14:03:11+00:00",
  "generation_model": "qwen3:8b",
  "answers_source": "tests/reports/out/2026-10-20/respostas.json",
  "judge_model": "jev-1.13.0 (TypeSafe AI)",
  "deepeval_version": "4.3.1",
  "typesafe_sdk_version": "0.7.5",
  "questions_lang": "pt",
  "tracks": ["A", "B"],
  "threshold": 0.7,
  "n_cases": 10,
  "cases_done": 10,
  "metrics_avg": {"Faithfulness": 0.81, "Fundamentação [JevEval]": 0.77},
  "usage": {"input_tokens": 431200, "cost_usd": 0.018, "seconds": 74.2},
  "cases": [
    {
      "name": "ragas-definicao",
      "input": "...", "expected_output": "...", "actual_output": "...", "retrieval_context": ["..."],
      "metrics": [
        {
          "name": "Fundamentação [JevEval]", "track": "B",
          "score": 0.93, "success": true, "not_applicable": false, "indeciso": false,
          "confidence": 0.8, "reason": "...", "error": null,
          "score_breakdown": [{"question": "...", "type": "noul", "weight": 3.0, "value": 0.9,
                               "applicable": true, "probabilities": {"true": 0.9, "false": 0.1},
                               "confidence": 0.8, "passed": null}]
        }
      ]
    }
  ]
}
```

`metrics_avg` exclui N/A e erros, e o relatório mostra quantos ficaram de fora de cada
média.

## 10. Fases (cada item cabe num commit; DoD = critério de pronto)

| Fase | Entregas | DoD | Estimativa |
|---|---|---|---|
| **0. Preparação** | Conta e chave TypeSafe; reconferir a doc oficial contra a seção 1 deste plano; `tests/jeveval/requirements*.txt`; `.env.example`; `.gitignore` | No venv: `import deepeval, typesafe_sdk`; um `decide()` de fumaça com um `Noul` trivial devolve probabilidade | 0,5 dia |
| **1. Núcleo hermético** | `_jev_metrics.py`, `_fake_jev.py`, `_jev_export.py`, `unit/` (U1–U12), job `jeveval-unit` | CI verde; nenhum teste de `unit/` toca a rede (U10) | 1–2 dias |
| **2. Validação do instrumento** | `live/test_jev_controls.py` (K1–K11), `test_jev_stability.py`, `test_jev_criterion.py` | K1–K10 passam com `jev-1.13.0`; D5 decidida e registrada; resultado de estabilidade e de critério (AUC por regra) gravados, **passando ou não** | 1–2 dias |
| **3. Goldens + export** | `run_and_export.py`, `test_rag_quality_jev.py`, `fixtures/respostas_congeladas.json`, passo em `run_all.py` | `run_all.py` produz o JSON Jev sobre as respostas compartilhadas; pulo limpo sem chave; limiar recalibrado com os números reais e documentado | 1–2 dias |
| **4. Relatório** | `report_data`/`report_document`: 3ª coluna, "Decisões do Jev", concordância de três vias | `build_report.py` gera MD/PDF/XLSX com os três frameworks; `test_build_report.py` verde | 1 dia |
| **5. O6 pós-hoc** | `oracles/o6_jev.py`, `--oracles O6` no `evaluate.py`, tabela T-O6 | MS do O6, swing contra O4, matriz por família e custo, declarados como exploratórios | 2–3 dias |
| **6. CI agendado + docs** | `jeveval-live.yml`; `tests/jeveval/README.md`; ARCHITECTURE (exceção D1, terceiro grupo de testes, decisões); README raiz ("Rodar tudo de uma vez") | Workflow manual verde com secret e verde com aviso sem secret | 0,5–1 dia |
| **7. Opcionais** | Fila humana (7.7); `GET /stats` com `jev_faithfulness_avg`; `hybrid` se a trilha A for ambígua | Conforme cada um | — |

## 11. Riscos e mitigações

| Risco | Efeito | Mitigação |
|---|---|---|
| API do JevEval muda (recurso com dias de vida) | Quebra de import ou de assinatura | Pin exato (`deepeval==4.3.1`); U1 quebra cedo no CI; Fase 0 reconfere a doc |
| Jev pior em português | Notas enviesadas | D5 decidida por K11; idioma registrado no export |
| `jev-latest` muda por baixo | Série histórica incomparável | D3: modelo pinado; workflow agendado detecta deriva |
| Abstenção pontuada como 1,0 (`EMPTY_SCORE`) ou vacuamente alta (C2/C3) | Falso positivo silencioso | Porta determinística por `NOT_FOUND_MARKER`; U5, U6 e K8 |
| `Noul` escrito no sentido negativo (C1) | Nota invertida sem erro | Metadado `desejavel` + U2 + revisão |
| Limite de contexto em mutantes K2/R2 | `SystemOneContextLimitError` | U8 com o pior caso; erro vira registro (`evaluable=False` no O6) |
| Rate limit ou falha de rede | Rodada parcial | Retry do próprio deepeval (`retry_typesafe`); export regravado por caso; teto de 60 s |
| Dado sensível enviado à API | Violação do princípio do projeto | D1: só corpus público; README e `.env.example` dizem isso; nenhum caminho de código lê o corpus da organização parceira |
| Comparar Jev com DeepEval 4.2.0 mistura juiz e versão | Conclusão errada | Controle de versão da seção 7.4 |

## 12. Custo estimado (preço embutido no deepeval: US$ 0,042 / 1M tokens de entrada)

- **Goldens:** 10 casos × 11 métricas (trilhas A+B) × ~4k tokens ≈ 0,44M tokens ≈
  **US$ 0,02 por rodada**.
- **Validade de critério:** 247 pares × ~0,5k tokens ≈ 0,12M tokens, **menos de
  US$ 0,01**.
- **O6:** 19 configurações (baseline + 18 mutantes) × 30 casos × 5 métricas × ~6k
  tokens ≈ 17M tokens ≈ **US$ 0,75**. Do lado do SUT, o custo é zero: só lê o log.
- O custo real (tokens e segundos) vai em todo export (`usage`) e na tabela de custo da
  RQ3. A estimativa acima não substitui a medida.

## 13. Perguntas em aberto (para decidir antes da Fase 1)

1. **A exceção D1 é aceitável para o artigo e para o grupo?** Se não for, o plano para
   na Fase 1 (só herméticos) e o resto vira trabalho futuro.
2. **Quem paga e onde fica a chave?** Conta pessoal ou do grupo de pesquisa, e um secret
   no repositório só se a Fase 6 for feita.
3. **O O6 entra no artigo ou só no pacote de replicação?** O plano assume que ele entra
   como análise exploratória declarada (D6).
4. **Versionar `fixtures/respostas_congeladas.json`?** Ele contém trechos dos PDFs
   públicos. Hoje `tests/mutation/corpus/` está fora do repositório até as licenças serem
   resolvidas (`.gitignore`). Se a mesma cautela valer aqui, a fixture fica de fora e o
   workflow agendado só roda os controles (que usam os textos já versionados em
   `tests/ragas/test_judge_controls.py` e `tests/deepeval/goldens/dataset.json`).
