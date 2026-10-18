# Testes E2E do frontend

Suíte Playwright para pegar regressões visuais/funcionais do frontend rapidamente
durante iteração — sem precisar validar tudo manualmente (ou eu escrever um script
descartável) a cada mudança. Cobre os bugs que já apareceram uma vez:

- `composer.spec.ts` — sem botão de emoji, botão de enviar habilita/desabilita
  corretamente, `Shift+Enter` não envia, contador de caracteres, modelo no rodapé.
- `monitor.spec.ts` — painel de Monitor troca dentro da sidebar sem colapsar a
  largura do chat (regressão do bug de CSS Grid), sessões ↔ monitor alternam.
- `chat-flow.spec.ts` — envia pergunta(s) reais via Ollama, resposta aparece,
  scroll e digitação continuam funcionando após várias mensagens (regressão do
  bug de `min-height`). **É a suíte mais lenta** (~1-2 min) porque espera geração
  real do LLM local.
- `homepage.spec.ts` — HUD não bloqueia o clique no CTA (regressão do bug de
  `pointer-events`), números da homepage carregam do backend.
- `providers.spec.ts` — seletor Local/Servidor do monitor: abas renderizam com o
  status de cada provider, clicar troca o provider ativo no backend, e as métricas
  de CPU/RAM/GPU viram "indisponível" no modo remoto em vez de mostrar os números
  da máquina local. **Mocka o backend inteiro** (HTTP + WebSocket via
  `page.route`/`page.routeWebSocket`), então roda sem API no ar — é a suíte que dá
  cobertura real no CI, onde todo o resto se auto-pula.
- `judge-vs-generator.spec.ts` — **quem responde não é quem julga**
  (LLM-as-a-judge). O teste:
  1. confere que o juiz é outro modelo; se for o mesmo do gerador, falha
     (*self-grading*);
  2. faz uma pergunta pelo chat;
  3. confere na tela que quem respondeu foi o gerador;
  4. entrega pergunta, trechos recuperados e resposta ao juiz, que diz se a
     resposta é fiel aos trechos;
  5. mostra os dois papéis e o veredito num painel na página.

  O veredito não é assertado, porque é opinião de um LLM; o que o teste
  garante é a separação dos papéis. O juiz vem de `RAG_JUDGE_MODEL` (padrão
  `qwen3:8b`), num Ollama em `RAG_JUDGE_OLLAMA_URL` (padrão
  `http://localhost:11434`). Sem esse Ollama, ou sem o modelo baixado, o teste
  se auto-pula.
- `goldens.spec.ts`: as **perguntas de referência** (goldens dos PDFs e dos
  Markdown), feitas pelo chat a cada RAG. Cada resposta, com os trechos que o
  chat devolveu, vai para um arquivo que o RAGAS julga depois
  (`tests/demo_e2e.py`). Só roda na demonstração com relatório
  (`scripts\demo_e2e.bat`); o `npm test` e o `npm run demo` o deixam de fora.

## Setup (uma vez)

```powershell
cd frontend/tests
npm install
npx playwright install chromium
```

## Rodar

```powershell
npm test
```

Isso já sobe o servidor estático do frontend (`python -m http.server 5510`)
automaticamente. **A API precisa estar rodando à parte** em `http://localhost:8010`
(`uvicorn app.main:app` no backend, com o Ollama no ar) — sem ela, os testes que
dependem de resposta real do backend são pulados automaticamente (não falham),
com uma mensagem indicando o motivo.

Para depurar visualmente: `npm run test:headed`. Para ver o último relatório:
`npm run report`.

## Modo demonstração (ver os testes rodando ao vivo)

Os mesmos testes, para uma turma acompanhar
(`playwright.demo.config.ts`):

```powershell
npm run demo          # navegador visível, ações em câmera lenta
npm run demo:ui       # modo UI do Playwright: lista de testes, linha do tempo, DOM a cada passo
npm run demo:report   # relatório HTML com vídeo e trace de cada teste
```

A demonstração testa **dois RAGs**, o dos PDFs e o dos Markdown (os dois
backends do `docker-compose.yml`), em três projetos do Playwright:

| Projeto | Specs | Backend |
|---|---|---|
| `interface` | `composer`, `homepage`, `monitor`, `providers` | `http://localhost:8010` |
| `RAG pdf-v1 (PDFs)` | `chat-flow`, `goldens`, `judge-vs-generator` | `http://localhost:8010` |
| `RAG md-v1 (Markdown)` | os mesmos três | `http://localhost:8011` |

Com `DEMO_PDF_INDEX=pdf-v2` e `DEMO_MD_INDEX=md-v2`, os projetos viram
`RAG pdf-v2 (PDFs)` e `RAG md-v2 (Markdown)`. Em `goldens.spec.ts`, o painel
diz se o certo é responder ou se abster a partir do `/health` do backend
(`rag_index_origins`, os acervos cujos artigos o índice contém).

Num projeto de RAG, antes de cada teste, `e2e/fixtures.ts`:

- pula o teste se o backend daquele RAG estiver fora do ar ou com o índice
  vazio;
- confere no `/health` que o backend serve mesmo aquele índice, porque uma
  porta trocada mediria o RAG errado;
- aponta o chat da página para ele (`window.__RAG_API_BASE_URL__`).

Em modo demonstração, cada página ganha uma faixa no canto com o teste em
execução, o RAG e os dois modelos: o que **responde** (lido do `/health` do
backend) e o que **julga**. Fora dele, a faixa não aparece. A faixa ignora o
mouse e não muda o layout, então não interfere nos testes.

| Variável | Para quê | Padrão |
|---|---|---|
| `DEMO_SLOWMO` | Milissegundos entre ações | `400` |
| `RAG_JUDGE_MODEL` | Modelo juiz | `qwen3:8b` |
| `RAG_JUDGE_OLLAMA_URL` | Ollama do juiz | `http://localhost:11434` |
| `DEMO_PDF_API_URL` / `DEMO_MD_API_URL` | API de cada RAG | `http://localhost:8010` / `8011` |
| `DEMO_PDF_INDEX` / `DEMO_MD_INDEX` | Versão do índice que cada RAG deve servir (conferida no `/health`) | `pdf-v1` / `md-v1` |
| `DEMO_ANSWERS_DIR` | Onde `goldens.spec.ts` grava as respostas; sem ela, o spec fica de fora | — |
| `DEMO_GOLDENS` | Só as N primeiras perguntas de referência de cada acervo | todas |

Para subir o sistema inteiro em Docker e rodar a demonstração num comando só,
use `scripts\demo_e2e.bat` (README da raiz, "Demonstração E2E ao vivo").
