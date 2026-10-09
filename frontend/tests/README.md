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

Em modo demonstração, cada página ganha uma faixa no canto com o teste em
execução e os dois modelos: o que **responde** (lido do `/health` do backend)
e o que **julga** (`e2e/fixtures.ts`). Fora dele, a faixa não aparece. A
faixa ignora o mouse e não muda o layout, então não interfere nos testes.

| Variável | Para quê | Padrão |
|---|---|---|
| `DEMO_SLOWMO` | Milissegundos entre ações | `400` |
| `RAG_JUDGE_MODEL` | Modelo juiz | `qwen3:8b` |
| `RAG_JUDGE_OLLAMA_URL` | Ollama do juiz | `http://localhost:11434` |

Para subir o sistema inteiro em Docker e rodar a demonstração num comando só,
use `scripts\demo_e2e.bat` (README da raiz, "Demonstração E2E ao vivo").
