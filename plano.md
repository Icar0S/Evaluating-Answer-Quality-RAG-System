# Plano — DeepEval como camada de avaliação de qualidade do RAG

> Baseado em https://deepeval.com/docs/introduction e nas páginas filhas
> (`getting-started-rag`, `metrics-introduction`, `evaluation-test-cases`,
> `evaluation-datasets`, `synthesizer-introduction`, `guides-using-custom-llms`,
> `command-line-interface`, `environment-variables`, `integrations/models/ollama`),
> consultadas em 26/08/2026. Comandos/APIs citados abaixo são os documentados
> nessas páginas — reconfirmar contra a versão do pacote no momento da
> implementação, framework evolui rápido.

## 1. Onde isso entra no que já existe

O projeto já tem duas camadas de teste rodando (`docs/ARCHITECTURE.md`, seção
"Fase 2"):

| Camada | O que cobre | Determinístico? |
|---|---|---|
| `tests/api/` (pytest) | Roteamento por provider, métricas do monitor, log — sem LLM real | Sim |
| `frontend/tests/e2e/` (Playwright) | UI, e um grupo (`chat-flow.spec.ts`) que exercita geração real | Parcial — o grupo de geração real se auto-pula sem backend/Ollama |

Falta a terceira: **qualidade da resposta** (fidelidade, relevância,
alucinação). O README já reserva o nome "Fase 2 — RAGAS" pra isso, mas nunca
foi implementada — `tests/ragas/results/.gitkeep` e `tests/human_review/annotations/.gitkeep`
são só placeholders, e `StatsResponse.ragas_faithfulness_avg`
(`backend/app/schemas.py:122`) já existe na API com a nota "Disponível após a
Fase 2 (avaliação RAGAS)" mas sempre retorna `null`.

**Decisão proposta neste plano**: implementar essa camada com **DeepEval no
lugar do RAGAS** cogitado antes (nenhuma linha de RAGAS existe no repo, então
não há nada pra migrar) — não como "mais uma", literalmente a implementação
da Fase 2. Motivo: DeepEval cobre o mesmo espaço de métricas (fidelidade,
relevância, precisão/recall de contexto) e soma o que o RAGAS não tem pronto:
suporte nativo a Ollama como juiz (`deepeval.models.OllamaModel`), integração
`pytest`/CI de fábrica (`deepeval test run`), cache de resultado, e
gerenciamento de dataset/golden versionável em JSON local — tudo sem exigir
nenhum serviço de nuvem (Confident AI é *opcional*, nunca obrigatório).

Isso também mantém a distinção já registrada no artigo: o oráculo
determinístico do sistema original (similaridade por cosseno, limiar 0,85,
`spaCy pt_core_news_lg`) continua existindo e é *outra coisa* — a heurística
`grounded` do backend (regex sobre `NOT_FOUND_MARKER`,
`backend/app/rag/generation.py`) é o guardrail rápido de produção. DeepEval
entra como o **LLM-as-judge** que audita esse guardrail com um julgamento
real, exatamente o papel que o RAGAS ocupava no design original.

## 2. Escopo

**Dentro do escopo desta implementação:**
- Métricas RAG: `FaithfulnessMetric`, `AnswerRelevancyMetric`,
  `ContextualPrecisionMetric`, `ContextualRecallMetric`,
  `ContextualRelevancyMetric`.
- Juiz local via Ollama (`OllamaModel`), sem chave de API nenhuma.
- Dataset (`Golden`/`EvaluationDataset`) pequeno, público, versionado no repo.
- Execução via `pytest` + `deepeval test run`, local, com auto-skip se
  Ollama/backend não estiverem no ar (mesmo padrão do Playwright).
- Persistência dos resultados em JSON, consumida pelo `/stats`.

**Fora do escopo (citado, não implementado agora):**
- Confident AI (`deepeval login`, `dataset.push`) — projeto é 100% local por
  princípio (ver README, "código/golden dataset/corpus indisponíveis").
- Métricas de agente, multimodais, e de segurança (Bias/Toxicity/PII) — não
  se aplicam a este domínio (RAG de pergunta-resposta, não agente com
  tool-calling nem geração de imagem).
- `ConversationSimulator`/multi-turn — o `/chat` avalia uma pergunta por vez;
  sessões multi-turno existem só no `localStorage` do frontend, não têm
  contrato de API ainda.
- Juiz via provider remoto (`smartdatatest`) — não é Ollama-compatível, exigiria
  escrever um `DeepEvalBaseLLM` customizado. Citado na seção 6 como extensão
  futura, não faz parte da entrega inicial.

## 3. Decisão de arquitetura: o juiz

DeepEval precisa de um LLM para julgar cada métrica (ele gera *reasoning* +
score, não é regex). Duas opções documentadas:

1. **`deepeval set-ollama --model=qwen3:8b --base-url=http://localhost:11434`**
   — configura via CLI, persiste em `.env`/keystore local do deepeval.
2. **Programático** (preferido para reprodutibilidade — não depende de estado
   de CLI que quem clona o repo não tem):

   ```python
   from deepeval.models import OllamaModel

   judge_model = OllamaModel(
       model=settings.generation_model,       # reaproveita GENERATION_MODEL do .env
       base_url=settings.ollama_base_url,      # reaproveita OLLAMA_BASE_URL
       temperature=0,
   )
   ```

   Passado explicitamente em cada métrica: `FaithfulnessMetric(model=judge_model, threshold=0.7)`.

**Trade-off documentado, não escondido**: usar o mesmo `qwen3:8b` que gera a
resposta como juiz da própria resposta é um risco conhecido de "self-grading
bias" (o modelo tende a validar seu próprio estilo). A alternativa —modelo
juiz maior/diferente— exigiria mais VRAM do que os 8GB disponíveis (ver
`docs/ARCHITECTURE.md`, tabela de hardware) rodando ao lado do modelo de
geração. Este plano aceita o mesmo modelo como juiz por restrição de
hardware, e documenta a limitação no README/ARCHITECTURE (mesmo padrão já
usado no projeto pra outros trade-offs, ex. GPU vs. modo servidor). Se um
juiz maior via provider remoto (`smartdatatest`) vier a ser desejado depois,
o ponto de extensão é um `DeepEvalBaseLLM` chamando `POST /v1/chat` (mesmo
contrato de `generation.py::_generate_smartdatatest`).

## 4. Dependências

Novo grupo de dependências, isolado do que é necessário pra *rodar* o
projeto (mesma filosofia de `requirements-dev.txt` e do Node da homepage —
ver README, "Requirements"):

- `backend/requirements-eval.txt`:
  ```
  -r requirements.txt
  deepeval==<pin na hora de implementar>
  ```
- `.env.example`: adicionar `DEEPEVAL_TELEMETRY_OPT_OUT=1` — opt-out de
  telemetria anônima por padrão num projeto acadêmico, sem motivo pra deixar
  ligado (`environment-variables`, seção Telemetria).
- Nunca setar `CONFIDENT_API_KEY` — sua ausência é o que mantém tudo local
  (mesma doc, seção Cloud).

## 5. Estrutura de arquivos nova

```
tests/deepeval/
├── README.md              análogo a tests/api/README.md: o que roda, como, e por quê
├── conftest.py            skip automático (Ollama/backend fora do ar), fixture do judge_model
├── goldens/
│   └── dataset.json        Golden[] público — ver seção 7
├── test_rag_quality.py     os testes parametrizados (assert_test)
└── results/
    └── .gitkeep             saída JSON de cada execução (gitignored, exceto .gitkeep)
```

`tests/ragas/` (placeholder vazio, nunca implementado) é substituído por
`tests/deepeval/` — ver seção 9, "decisões em aberto", antes de apagar.

## 6. Pipeline: como o teste chama a stack real

Reaproveita as mesmas funções que `POST /chat` usa
(`backend/app/main.py:110-198`), chamadas **in-process** — sem precisar subir
o `uvicorn` nem gerenciar processo de servidor no teste (diferença
deliberada do Playwright, que testa a API via HTTP porque está testando a
camada HTTP; aqui o alvo é a qualidade da resposta, não o protocolo):

```python
from app.rag import retrieval, generation

def run_pipeline(question: str, top_k: int = 4):
    chunks, _ = retrieval.retrieve(question, top_k)
    result = generation.generate_answer(question, chunks)
    return result["answer"], [c["text"] for c in chunks]
```

Mapeamento pro `LLMTestCase` (`evaluation-test-cases`):

| Campo do teste | Origem |
|---|---|
| `input` | `golden.input` (a pergunta) |
| `actual_output` | `answer` retornado pelo pipeline |
| `retrieval_context` | textos dos `chunks` recuperados (o que o retriever *de fato* achou) |
| `expected_output` | `golden.expected_output`, curado manualmente |
| `context` | `golden.context` — trecho(s) "verdade fundamental" que a pergunta deveria recuperar, usado por `ContextualPrecision`/`ContextualRecall` |

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,
)

@pytest.mark.parametrize("golden", load_goldens())  # tests/deepeval/goldens/dataset.json
def test_rag_quality(golden, judge_model):
    answer, retrieval_context = run_pipeline(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        context=golden.context,
        retrieval_context=retrieval_context,
    )
    assert_test(test_case, [
        FaithfulnessMetric(model=judge_model, threshold=0.7),
        AnswerRelevancyMetric(model=judge_model, threshold=0.7),
        ContextualPrecisionMetric(model=judge_model, threshold=0.7),
        ContextualRecallMetric(model=judge_model, threshold=0.7),
        ContextualRelevancyMetric(model=judge_model, threshold=0.7),
    ])
```

Roda com `deepeval test run tests/deepeval/test_rag_quality.py` (ou `pytest`
puro — `assert_test` funciona nos dois; `deepeval test run` só soma o
relatório rico e `deepeval inspect` depois).

## 7. Dataset (goldens) — por que não é o dataset privado

O golden dataset real (perguntas de saneamento básico da CAGECE) é
justamente um dos três itens que o README já declara indisponíveis por
conter informação operacional da organização parceira. Pra este artefato
público, a avaliação precisa de um dataset próprio, pequeno e público:

- **Fonte**: os mesmos PDFs acadêmicos já usados como corpus de teste do
  artefato (`data/source_pdfs/`, o corpus sobre teste de LLM/RAG, não o
  regulatório).
- **Tamanho inicial**: 10–20 perguntas — o suficiente pra exercitar as 5
  métricas sem virar um segundo golden dataset a manter.
- **Curadoria**: manual, versionada em `tests/deepeval/goldens/dataset.json`
  (`input` + `expected_output` + `context` escritos/revisados à mão — mais
  confiável que gerar tudo via LLM).
- **Synthesizer como acelerador, não como fonte de verdade**
  (`synthesizer-introduction`): script auxiliar opcional
  (`scripts/generate_deepeval_goldens.py`, fora da entrega inicial) rodando
  `Synthesizer(model=judge_model).generate_goldens_from_docs(...)` sobre
  `data/source_pdfs/` pra gerar candidatos — que exigem revisão humana antes
  de entrar no `dataset.json` versionado (a própria doc do deepeval avisa que
  a saída do Synthesizer passa por filtragem, não é golden pronto).

## 8. CI e execução local

Mesma regra já estabelecida pra `chat-flow.spec.ts` e o RAGAS original
(`docs/ARCHITECTURE.md`, "Testes determinísticos vs. testes que exercitam o
modelo"): isso não roda no CI do GitHub. `conftest.py` faz skip automático
(`pytest.mark.skipif`, checando Ollama/backend acessível, mesmo padrão de
`isBackendUp()` do Playwright) — falha vermelha no CI continua significando
regressão real, não ausência de GPU no runner.

Comando local (adicionar ao README, seção "Testes"):

```powershell
cd backend
.venv\Scripts\pip install -r requirements-eval.txt   # uma vez
cd ..
.venv\Scripts\deepeval test run tests/deepeval/test_rag_quality.py
```

Não entra em `scripts/start_dev.bat` — mesma decisão já tomada pra RAGAS no
design original: roda sob demanda, não faz parte de "subir tudo".

## 9. Persistência e exposição em `/stats`

`evaluate()`/`deepeval test run` devolvem um `EvaluationResult` com score e
reasoning por métrica e por caso. Um passo extra no teste (ou um script
separado que chama `evaluate()` diretamente em vez de `assert_test`)
serializa isso em `tests/deepeval/results/<timestamp>.json`: médias por
métrica + detalhe por pergunta.

Esse arquivo alimenta dois lugares, ambos já com o "encaixe" pronto no
código:

1. **`GET /stats`** (`backend/app/schemas.py:117-123`,
   `backend/app/main.py:219-240`) — o campo hoje é
   `ragas_faithfulness_avg: float | None`, sempre `null`, com a nota
   "Disponível após a Fase 2 (avaliação RAGAS)". Como a Fase 2 passa a ser
   DeepEval (seção 1), renomear para `deepeval_faithfulness_avg` (e ajustar a
   nota) é mais correto que manter um nome que aponta pra um framework nunca
   implementado — ver decisão em aberto na seção 10. `get_stats()` passa a
   ler o JSON mais recente de `tests/deepeval/results/` (`asyncio.to_thread`,
   mesmo padrão dos outros campos).
2. **Homepage** (`frontend/home/src/StatsBoard.jsx`) — já tem o texto "nota
   explícita (disponível após a Fase 2)" em vez de número inventado
   (`docs/ARCHITECTURE.md`, seção Homepage); troca pra mostrar o número real
   assim que `/stats` parar de retornar `null`.

## 10. Passo a passo sugerido (cada item cabe num commit)

1. `backend/requirements-eval.txt` + `DEEPEVAL_TELEMETRY_OPT_OUT` no `.env.example`.
2. Estrutura `tests/deepeval/` (README, `conftest.py` com skip automático e fixture do `judge_model`).
3. `goldens/dataset.json` — curadoria manual das 10–20 perguntas.
4. `test_rag_quality.py` com as 5 métricas.
5. Rodar localmente, calibrar thresholds (0.7 é ponto de partida, não dado medido), documentar os números encontrados no `ARCHITECTURE.md` (mesmo padrão do resto do doc).
6. Serialização dos resultados em `results/*.json`.
7. `StatsResponse`/`main.py::get_stats` lendo o JSON mais recente; `StatsBoard.jsx` mostrando o número.
8. Atualizar `README.md` (seção "Fase 2", hoje "Em construção") e `docs/ARCHITECTURE.md` (registro da decisão, seguindo o padrão do resto do arquivo).
9. *(Stretch, opcional)* `scripts/generate_deepeval_goldens.py` via `Synthesizer`.
10. *(Stretch, opcional)* uma métrica `GEval` customizada de domínio (ex. "a resposta cita a fonte/norma corretamente"), demonstrando extensibilidade além das 5 métricas padrão.

## 11. Decisões em aberto — resolvidas na implementação (26/08/2026)

Implementado na íntegra nesta mesma sessão, com a stack real local (Ollama +
vector store já populados na máquina) — não ficou só no papel. Como as 4
perguntas abaixo bloqueavam a implementação e o ambiente permitia validar
cada uma na hora, decidi e segui em vez de esperar confirmação:

- **Nome do campo**: renomeado para `deepeval_faithfulness_avg` (era
  `ragas_faithfulness_avg`, sempre `null`, nunca teve valor real gravado —
  renomear não quebra nada em produção).
- **`tests/ragas/`**: removido (`git rm`), substituído por `tests/deepeval/`.
- **Tamanho do dataset**: 10 perguntas, curadas manualmente a partir de 5 PDFs
  do corpus (RAGAS.pdf, ARES, MetaRAG, RAGChecker, o paper da Visagio) — ver
  `tests/deepeval/goldens/dataset.json`.
- **Juiz**: **não** o mesmo `qwen3:8b` — usei `gemma3:4b`, que já estava
  baixado localmente nesta máquina, evitando o self-grading bias sem exigir
  download novo. Documentado em `docs/ARCHITECTURE.md`, seção "Avaliação de
  qualidade (DeepEval)".

Rodei a suíte completa (10/10 perguntas) contra a stack real pra validar —
não ficou só implementado, ficou testado. Números e observações da primeira
calibração estão em `docs/ARCHITECTURE.md`.
