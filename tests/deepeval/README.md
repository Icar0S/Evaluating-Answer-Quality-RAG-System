# Avaliação de qualidade (DeepEval) — Fase 2

Suíte [DeepEval](https://deepeval.com/docs/introduction) que audita a
*qualidade da resposta* do RAG: fidelidade ao contexto, relevância da
resposta, e precisão/recall/relevância da recuperação. Complementa
`tests/api/` (mecânica do sistema, sem LLM real) e
`frontend/tests/e2e/chat-flow.spec.ts` (fluxo de UI com geração real) — este
é o terceiro pilar descrito em `docs/ARCHITECTURE.md`, "Fase 2 — Arquitetura
de testes".

**É o oposto de `tests/api/` em isolamento de propósito**: em vez de mockar
tudo, esta suíte chama a stack real — Ollama (geração, embeddings e o
modelo-juiz) e o vector store populado. Por isso não é hermética, é lenta
(minutos, não segundos) e **nunca roda no CI do GitHub** — mesma regra dos
testes E2E de geração real (`docs/ARCHITECTURE.md`, "Testes determinísticos
vs. testes que exercitam o modelo"). Sem a stack no ar, a suíte inteira se
auto-pula (`conftest.py::_skip_without_real_stack`) em vez de falhar.

## Por que um venv separado

`backend/requirements.txt` pina `pydantic==2.10.4`; o `deepeval` exige uma
versão mais nova (conflito de resolução testado, não hipotético). Por isso
esta suíte vive num venv próprio (`tests/deepeval/.venv/`, gitignored),
igual ao `node_modules` separado entre `frontend/home/` e `frontend/tests/`
— instalar isto nunca deve afetar o venv que roda a API.

## Setup (uma vez)

```powershell
python -m venv tests\deepeval\.venv
tests\deepeval\.venv\Scripts\pip install -r backend\requirements-eval.txt
```

Requer, além disso, no Ollama local: `GENERATION_MODEL`/`EMBEDDING_MODEL` do
`.env` baixados (já exigidos pra rodar o projeto — ver README raiz), **mais**
o modelo-juiz (`DEEPEVAL_JUDGE_MODEL`, padrão `gemma3:4b`):

```powershell
ollama pull gemma3:4b
```

E o vector store populado (`python scripts\ingest_documents.py`, ver README raiz).

## Rodar

```powershell
tests\deepeval\.venv\Scripts\python.exe -m pytest tests/deepeval
```

Da raiz do repo, sempre — é de lá que `pytest.ini` aponta `pythonpath = backend`,
necessário pra `from app.rag import ...` funcionar.

### Bug conhecido: `deepeval test run` no Windows

Use pytest puro (acima), **não** `deepeval test run tests/deepeval/test_rag_quality.py`.
A CLI liga o cache local do deepeval (via env var `DEEPEVAL=1`), que depende do
`portalocker` pra lock compartilhado — no Windows, sem o extra `pywin32`, isso cai num
fallback (`msvcrt`) que não sabe fazer shared lock de verdade, e a chamada de cache
quebra com `AttributeError: 'NoneType' object has no attribute 'test_cases_lookup_map'`
**depois** que a avaliação real já rodou (geração + julgamento reais, tempo e créditos
gastos à toa). Rodando com pytest puro, `DEEPEVAL` fica unset e o cache nunca liga —
`run_and_export.py` também desliga o cache explicitamente (`CacheConfig(write_cache=False)`)
pelo mesmo motivo.

Pra gerar o JSON que alimenta `GET /stats` (`deepeval_faithfulness_avg`) e a
seção de resultados do artigo, em vez do `assert_test` (que só dá pass/fail):

```powershell
tests\deepeval\.venv\Scripts\python.exe tests\deepeval\run_and_export.py
```

Salva `results/<timestamp>.json` e `results/<timestamp>.html` (mesmo nome),
mais um `results/latest.html` estável; `backend/app/main.py::get_stats` lê
sempre o JSON mais recente. Ao terminar, abre o HTML sozinho no navegador
padrão — nenhum desses três arquivos é versionado (mesma regra de
`data/vector_store/`: reproduzível, não commitado).

## Ver o painel

O HTML é autocontido (sem servidor, sem build, `file://` funciona) — arquivo
de verdade em `tests/deepeval/results/`, não uma página hospedada em outro
domínio. `run_and_export.py` já abre sozinho ao terminar; pra reabrir depois
sem rodar a avaliação de novo (equivalente ao `allure serve`, que reconstrói
o relatório a partir de resultados já existentes em vez de rodar os testes):

```powershell
tests\deepeval\.venv\Scripts\python.exe tests\deepeval\view_report.py
```

Mostra, por métrica: a média com uma barra até o limiar; uma grade com todos
os casos × as 5 métricas pra escanear rápido (score e PASS/FAIL, clicável);
e cada caso expansível com a pergunta, resposta esperada vs. gerada, e o
motivo que o juiz deu pra cada score — inclusive um aviso quando a resposta
foi uma recusa (`NOT_FOUND_MARKER`), caso em que Faithfulness não tem uma
recusa pra julgar como fiel/infiel e o score vira um artefato da métrica.
Geração do HTML fica em `report.py` (lido a partir do JSON, sem nada
hardcoded de uma rodada específica — continua funcionando conforme o
dataset crescer).

## O que é avaliado

5 métricas RAG do DeepEval, todas julgadas pelo `DEEPEVAL_JUDGE_MODEL` local
(nunca OpenAI/nuvem — ver `_shared.py::build_judge_model`), limiar inicial
`0.7` (`_shared.py::THRESHOLD`, ponto de partida, não dado medido — calibrar
com os números reais de cada rodada):

| Métrica | O que audita |
|---|---|
| `FaithfulnessMetric` | A resposta está fundamentada no contexto recuperado? (audita a heurística `grounded` do backend com um julgamento de fato, não regex) |
| `AnswerRelevancyMetric` | A resposta é relevante pra pergunta feita? |
| `ContextualPrecisionMetric` | Os chunks recuperados são focados (sem ruído irrelevante)? |
| `ContextualRecallMetric` | A recuperação trouxe tudo que era relevante? |
| `ContextualRelevancyMetric` | Os chunks recuperados batem com a pergunta? |

## O dataset (`goldens/dataset.json`)

10 perguntas curadas manualmente sobre os PDFs acadêmicos já versionados em
`data/source_pdfs/` (o corpus de teste do artefato, não o corpus regulatório
da CAGECE — esse é confidencial, ver README raiz). Cada entrada tem `input`
(pergunta), `expected_output` (resposta de referência) e `context` (o trecho
real do PDF de onde a resposta vem — carregado no teste por documentação e
rastreabilidade; as 5 métricas acima usam `retrieval_context`, o que o
retriever *de fato* trouxe, não este campo).

Pra expandir o dataset, adicione entradas ao JSON (mesmo formato) ou use
`deepeval.synthesizer.Synthesizer` sobre novos PDFs — mas trate a saída como
rascunho: o Synthesizer gera candidatos, curadoria manual antes de virar
golden versionado continua sendo o caminho confiável.

## Juiz local: por que um modelo diferente do de geração

`DEEPEVAL_JUDGE_MODEL` (`gemma3:4b`) é deliberadamente diferente de
`GENERATION_MODEL` (`qwen3:8b`): usar o mesmo modelo pra gerar e julgar a
própria resposta é um viés conhecido ("self-grading"). Os dois cabem na
mesma GPU de 8GB porque o Ollama troca os modelos em sequência — geração
termina e descarrega antes do juiz carregar — não ficam os dois na VRAM ao
mesmo tempo. Trade-off documentado: 4B é um juiz mais fraco que um modelo
maior daria; aceito aqui por restrição de hardware (mesma lógica de outras
decisões de tier em `docs/ARCHITECTURE.md`).
