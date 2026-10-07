# Avaliação de qualidade (RAGAS)

Suíte [RAGAS](https://docs.ragas.io/) que avalia a *qualidade da resposta* do
RAG com as mesmas cinco dimensões da suíte [DeepEval](../deepeval/README.md):
fidelidade ao contexto, relevância da resposta e precisão, recall e relevância
da recuperação. Ela usa **os mesmos goldens**
(`tests/deepeval/goldens/dataset.json`), de propósito: o objetivo é comparar os
dois frameworks sobre o mesmo conjunto.

Ela segue as mesmas regras da suíte DeepEval:

- Chama a stack real (Ollama com geração, embeddings e juiz, e o vector store
  populado).
- É lenta: cerca de 2 min por golden.
- **Nunca roda no CI.** Sem a stack no ar, a suíte inteira se auto-pula.

## Setup (uma vez)

```powershell
python -m venv tests\ragas\.venv
tests\ragas\.venv\Scripts\pip install -r tests\ragas\requirements.txt
ollama pull qwen3:8b      # juiz (RAGAS_JUDGE_MODEL)
```

Também é preciso o vector store populado
(`python scripts\ingest_documents.py`, ver o README da raiz).

O venv é próprio pelo mesmo motivo da suíte DeepEval: o `ragas` exige um
`pydantic` mais novo que o pinado em `backend/requirements.txt`. O
`requirements.txt` daqui também fixa o LangChain na linha 0.3, porque o
`ragas` 0.4.3 não limita a versão, e com o LangChain 1.x o `import ragas`
quebra.

## Rodar

Sempre a partir da raiz do repositório:

```powershell
# testes (pass/fail): controles do juiz + goldens x métricas
tests\ragas\.venv\Scripts\python.exe -m pytest tests/ragas

# só os controles do juiz (~1 min), ao trocar de juiz
tests\ragas\.venv\Scripts\python.exe -m pytest tests/ragas/test_judge_controls.py

# números para análise: JSON + CSV em tests/ragas/results/
tests\ragas\.venv\Scripts\python.exe tests\ragas\run_and_export.py
tests\ragas\.venv\Scripts\python.exe tests\ragas\run_and_export.py --limit 2
tests\ragas\.venv\Scripts\python.exe tests\ragas\run_and_export.py --only ragas-definicao
```

O JSON tem o mesmo formato do export do DeepEval: `cases[]` com `name`,
`input`, `actual_output`, `expected_output`, `retrieval_context` e
`metrics[]`. Assim, as duas avaliações podem ser cruzadas pelo nome do caso. O
JSON acrescenta:

- a versão do `ragas`;
- `prompt_overrides`: o que foi alterado nos prompts de fábrica;
- `judge_calls` e `judge_failures`;
- quantas notas de cada métrica foram de fato medidas.

O JSON é regravado a cada caso, então uma rodada interrompida não perde o que
já mediu. O CSV tem uma linha por caso × métrica (separador `;`, abre direto no
Excel).

## O que é avaliado

As métricas são de `ragas.metrics.collections` (a API atual do ragas 0.4) e
são julgadas pelo juiz local. Limiar inicial: `0.7` (`RAGAS_THRESHOLD`), o
mesmo do DeepEval. É um ponto de partida, não um valor medido.

| Métrica (ragas) | O que audita | Equivalente no DeepEval |
|---|---|---|
| `Faithfulness` | Cada afirmação da resposta é *inferível* dos trechos recuperados? | `FaithfulnessMetric` |
| `AnswerRelevancy` | A resposta responde à pergunta feita? | `AnswerRelevancyMetric` |
| `ContextPrecisionWithReference` | Os trechos úteis vêm primeiro no ranking? | `ContextualPrecisionMetric` |
| `ContextRecall` | A recuperação trouxe o que a resposta de referência precisa? | `ContextualRecallMetric` |
| `ContextRelevance` | Os trechos recuperados têm a ver com a pergunta? | `ContextualRelevancyMetric` |

Os testes são de dois tipos:

- **`test_judge_controls.py`**: casos com resposta certa conhecida, sem o
  SUT. Validam o *instrumento* (juiz + métrica) antes de confiar nas notas.
  Por exemplo, a alucinação real do SUT no estudo de mutação ("a WCAG 2.2
  define 14 critérios…", com contexto que nem cita WCAG) precisa receber
  faithfulness baixa. **Se um controle reprova, as notas da outra suíte não
  medem o RAG, medem o juiz.**
- **`test_rag_quality.py`**: um teste por par golden × métrica (50 no total).
  A resposta do SUT é gerada uma vez por golden e reaproveitada nas 5
  métricas. Métrica não medida (erro do juiz ou NaN) **reprova** com o motivo,
  em vez de passar.

## O juiz local, e por que ele não usa a integração padrão

O estudo de mutação concluiu que o `ragas` "não parseia a saída de nenhum juiz
local". Ver [docs/estudo-mutacao-status.md](../../docs/estudo-mutacao-status.md),
§3.6 e §4.9: com o gemma3:4b, a saída não foi parseada em 6 de 9 avaliações. O
que faltava não era um juiz maior. Faltavam três ajustes que a suíte DeepEval
já fazia:

1. **`num_ctx` explícito (8192).** Sem ele, o Ollama usa 4096 e, quando o
   prompt passa disso, **descarta o início**: as instruções e o schema de
   saída. Com 4 trechos de ~800 tokens, os prompts de faithfulness e de
   context recall passam disso.
2. **Saída restrita ao JSON schema** (`format` do `/api/chat`). O Ollama
   decodifica sob a gramática do schema, então o juiz não consegue devolver
   JSON inválido.
3. **`think: false`.** Com raciocínio ligado, o qwen3 gasta minutos antes de
   cada veredito.

A integração padrão do ragas (`llm_factory`) fala com o Ollama pelo endpoint
compatível com OpenAI, onde não dá para passar `num_ctx`. Por isso o juiz é
uma implementação própria da interface que as métricas aceitam
(`_ragas_shared.py::OllamaJudge`), sobre o `/api/chat` nativo.

O juiz falha **alto** quando o prompt pode ter sido truncado ou quando a saída
não bate com o schema. Nota sobre prompt cortado é pior que nota nenhuma.

**Resultado:** zero falhas de parse em todas as chamadas do piloto, com os dois
juízes.

### Qual juiz usar

| Juiz | Controles | Observação |
|---|---|---|
| `qwen3:8b` (padrão) | 5/5 | ~10–40 s por métrica numa GPU de 8 GB |
| `gemma3:4b` | 4/5 | Marca respostas diretas como evasivas (`noncommittal: 1`), o que zera `answer_relevancy` até para a resposta de referência |

O juiz precisa ser diferente do `GENERATION_MODEL`; caso contrário, ele avalia
a si mesmo (*self-grading*), e a suíte avisa. Par validado nesta máquina:
geração com `gemma3:4b` e juiz `qwen3:8b`. Para comparar com o DeepEval, use o
mesmo juiz nos dois (`DEEPEVAL_JUDGE_MODEL`).

## Diferenças em relação ao ragas de fábrica

Há duas, ambas em `AnswerRelevancy`, e registradas no JSON de cada rodada:

- **`strictness=1`, em vez de 3.** Com o juiz em temperatura 0, as 3 perguntas
  geradas saíam idênticas. Isso triplicava o custo sem mudar a nota.
- **Instrução de idioma.** O prompt do ragas é todo em inglês, então o juiz
  gerava a pergunta em inglês, e ela era comparada por embedding com a
  pergunta original em português. Com a instrução *"Write the question in the
  same language as the response"*, uma resposta correta foi de 0,68 para 0,83;
  a resposta fora do assunto ficou em 0,55 nos dois casos. O `prompt.adapt()`
  do ragas traduziria os exemplos usando o próprio juiz a cada rodada,
  gerando um prompt diferente a cada execução.

Além disso, a telemetria do ragas (que envia eventos de uso para fora da
máquina) fica desligada pelo próprio código (`RAGAS_DO_NOT_TRACK`), como a do
DeepEval.

## Limitações conhecidas

- **`answer_relevancy` tem piso alto.** A nota é o cosseno entre embeddings do
  `nomic-embed-text`, e duas perguntas em português sem relação nenhuma ficam
  perto de 0,55. A escala útil vai de ~0,55 a 1. Por isso o controle dessa
  métrica é relativo (relevante − fora do assunto ≥ 0,15).
- **O `context` do golden é um trecho abreviado** (com "..."). Ele não entra
  nas métricas, que usam o que o retriever trouxe de fato. Mas uma resposta de
  referência que diga mais do que o trecho leva faithfulness baixa contra ele,
  e isso é correto pela definição do ragas.
- **10 goldens** é pouco para afirmar algo estatístico. Esta é uma base para
  evoluir, não um resultado.

## Para evoluir

- **Comparar com o DeepEval:** cruzar `results/*.json` daqui com
  `tests/deepeval/results/*.json` pelo `name` do caso e medir a concordância
  por métrica (com o mesmo juiz dos dois lados).
- **Rejulgar o O3 do estudo de mutação:** o oráculo saiu da campanha porque o
  ragas não funcionava com juiz local (contingência do §9). Como o log da
  campanha guarda pergunta, resposta e trechos, o O3 pode ser rejulgado sem
  reexecutar o assistente (`tests/mutation/evaluate.py --oracles O3`). Antes,
  é preciso portar `tests/mutation/oracles/o3_ragas.py`, que ainda usa o
  ragas 0.2.14 com `ChatOllama`, para o juiz daqui.
- **Mais métricas:** `ResponseGroundedness` e `FactualCorrectness` (resposta ×
  referência) já estão em `ragas.metrics.collections` e usam o mesmo juiz.
- **Calibrar o limiar** por métrica com os números das primeiras rodadas, em
  vez do `0.7` único.
