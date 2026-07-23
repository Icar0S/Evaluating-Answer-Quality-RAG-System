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

Isso já sobe o servidor estático do frontend (`python -m http.server 5500`)
automaticamente. **A API precisa estar rodando à parte** em `http://localhost:8000`
(`uvicorn app.main:app` no backend, com o Ollama no ar) — sem ela, os testes que
dependem de resposta real do backend são pulados automaticamente (não falham),
com uma mensagem indicando o motivo.

Para depurar visualmente: `npm run test:headed`. Para ver o último relatório:
`npm run report`.
