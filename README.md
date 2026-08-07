# Evaluating Answer Quality of a RAG System

Assistente de RAG 100% local, especializado em teste de sistemas baseados em LLM/RAG, com arquitetura de testes (RAGAS + Playwright) construída ao redor dele. Base experimental de um projeto de pesquisa sobre avaliação de qualidade de resposta em RAG usando LLM-as-a-Judge com Human-in-the-Loop.

Veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para as decisões técnicas e o racional por trás delas.

## Pré-requisitos

- [Ollama](https://ollama.com/download) instalado — **sempre necessário**, mesmo
  gerando pelo servidor remoto: a busca embeda a pergunta a cada consulta e a API
  remota não expõe rota de embeddings. No modo servidor ele carrega só o
  `nomic-embed-text` (~274MB, roda em CPU).
- Python 3.11+
- GPU com pelo menos 8GB de VRAM — **só para geração local** (`qwen3:8b`).
  Sem GPU livre, use o modo servidor (`scripts\start_dev_remote.bat`, ver Fase 1.5).
- Node.js 18+ — para os testes E2E (Playwright) e para alterar a homepage
  (ilha React). Não é preciso apenas para *rodar* o projeto.

## Instalação

### Fase 0 — Setup do Ollama

1. Defina onde os modelos serão salvos (por padrão, fora do disco C: para não lotar o SO):

   ```powershell
   setx OLLAMA_MODELS "D:\modelosLLM\models"
   ```

   Reinicie o serviço do Ollama (ou a sessão do terminal) após definir a variável.

2. Baixe os modelos e valide o setup:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_ollama.ps1
   ```

   O script detecta o Ollama, baixa `qwen3:8b` (geração) e `nomic-embed-text` (embeddings), e testa os dois com um prompt simples. Ajuste os modelos via parâmetros `-GenerationModel` / `-EmbeddingModel` se seu hardware pedir outra faixa (veja a tabela em `docs/ARCHITECTURE.md`).

### Fase 1 — Rodando o RAG local

1. Instale as dependências do backend:

   ```powershell
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copie `.env.example` para `.env` na raiz do projeto e ajuste se necessário:

   ```powershell
   copy .env.example .env
   ```

3. Coloque os PDFs de referência em `data/source_pdfs/`.

4. Suba a API:

   ```powershell
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

5. Ingira os documentos (pode ser feito via API ou script):

   ```powershell
   # Via API (com o servidor já rodando)
   curl.exe -X POST http://localhost:8000/ingest

   # Ou via script, sem precisar do servidor:
   python scripts\ingest_documents.py
   ```

6. Sirva a pasta `frontend/` com um servidor estático (necessário para o WebSocket do monitor e para evitar problemas de CORS com `file://`), por exemplo:

   ```powershell
   cd frontend
   python -m http.server 5500
   ```

   Abra `http://localhost:5500/index.html` — a homepage leva ao chat (`chat.html`), que tem um trilho lateral com as views **Chat** e **Monitor do LLM**.

   Nenhuma etapa de build é necessária para *rodar* o projeto: a homepage é
   servida a partir de `frontend/index.html` e `frontend/home-assets/`, que são
   versionados. Só é preciso Node para **mudar** a homepage — ver
   [Frontend](#frontend).

### Fase 1.5 — Provider remoto (opcional)

Além do Ollama local, `/chat` pode gerar através da API do Mac mini em
`https://llm.smartdatatest.com` — ver [docs/llm-api-referencia.md](docs/llm-api-referencia.md)
para o contrato completo (rotas, auth, limites). **Não é um Ollama**: é uma API
própria, com autenticação Bearer, fila com 429/503 documentados, e sem rota de
embeddings — por isso a recuperação (retrieval) continua sempre local mesmo com
o provider remoto ativo; só a geração muda de lugar. (O antigo acesso direto via
Tailscale/Ollama bruto, em [docs/SETUP-LLM-LOCAL.md](docs/SETUP-LLM-LOCAL.md), foi
substituído por este gateway autenticado.)

Preencha no `.env` (deixe em branco para manter 100% do comportamento local de hoje;
**nunca commite a chave real**):

```
REMOTE_API_BASE_URL=https://llm.smartdatatest.com
REMOTE_API_KEY=sk-...
REMOTE_GENERATION_MODEL=<tag-do-modelo>   # ver GET /v1/models; vazio usa o default do servidor
REMOTE_LABEL=Servidor (Mac mini)
ACTIVE_PROVIDER=local   # ou remote — só define o padrão ao iniciar o backend
```

A troca entre "Local" e "Servidor" acontece em runtime (sem reiniciar o backend) pelas
abas no topo do painel de Monitor, ou via `POST /providers/active {"name": "local|remote"}`.
CPU/RAM/GPU no monitor só existem para o provider local — não há como o backend ler o
hardware de uma máquina remota sem um agente rodando lá, então essas métricas aparecem
como "indisponível" quando o Servidor está ativo (tokens/s, latência e status continuam
reais nos dois casos, vêm da própria resposta do provider).

Trocar o modelo **local** continua igual a antes: editar `GENERATION_MODEL`/
`EMBEDDING_MODEL` no `.env` e reiniciar o backend.

#### Subindo tudo já no modo servidor

```powershell
scripts\start_dev_remote.bat
```

Sobe backend + frontend com `ACTIVE_PROVIDER=remote` e **sem carregar o modelo de
geração local** — útil em máquina sem GPU livre, com bateria, ou quando o
`qwen3:8b` não cabe na VRAM. Antes de subir, ele valida em ordem: chaves do
servidor no `.env`, Ollama no ar **com o modelo de embeddings** (o de geração não
é tocado), servidor acessível em `/v1/ready`, provider ativo efetivamente em
`remote`, e índice vetorial não vazio. Qualquer uma dessas falhando, ele para e
diz o que fazer.

O que este modo **não** faz é dispensar o Ollama: a recuperação embeda a pergunta
a cada consulta e a API remota não tem rota de embeddings. Em resumo —
**embeddings: local (leve) · geração: servidor (pesado)**.

Para o ambiente 100% local com a suíte E2E, continue usando
`scripts\start_dev.bat`.

## Frontend

A homepage é uma ilha React (Vite); o chat (`chat.html`) continua HTML/CSS/JS
puro. O racional dessa divisão está em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Mexendo na homepage

A fonte fica em `frontend/home/` e a saída de build em `frontend/index.html` +
`frontend/home-assets/` (ambos versionados, por isso rodar o projeto não exige
Node):

```powershell
cd frontend/home
npm install        # uma vez
npm run build      # regenera frontend/index.html e frontend/home-assets/
```

**Não edite `frontend/index.html` diretamente** — ele é gerado a partir de
`frontend/home/index.html` e será sobrescrito no próximo build. O CI tem um job
(`homepage-build`) que rebuilda e falha se o artefato commitado estiver
desatualizado.

O chat não tem etapa de build: editar `frontend/components/*.js` e
`frontend/styles/*.css` e recarregar a página basta.

### Desligando as animações

Útil para testes determinísticos, máquinas fracas ou preferência pessoal. Qualquer
uma destas desliga todo o movimento das duas páginas:

| Como | Escopo |
|---|---|
| `?motion=off` na URL | Só naquele carregamento |
| `localStorage.setItem('motion', 'off')` | Persistente, por navegador |
| `VITE_DISABLE_MOTION=true npm run build` | Permanente, no artefato gerado |
| `prefers-reduced-motion: reduce` no SO | Automático, respeitado sem configuração |

`?motion=on` força ligado, inclusive por cima da preferência do sistema. Com o
movimento desligado, os blocos que entram por scroll renderizam no estado final
imediatamente — é o que mantém a suíte headless determinística.

## Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/chat` | Envia uma pergunta, retorna resposta + fontes + metadados |
| `POST` | `/ingest` | Reprocessa os PDFs de `data/source_pdfs/` |
| `GET` | `/health` | Status da API e do provider ativo (Ollama + vector store) |
| `GET` | `/providers` | Lista os providers configurados (local/remoto) e reachability de cada um |
| `POST` | `/providers/active` | Troca o provider ativo em runtime (`{"name": "local"\|"remote"}`) |
| `GET` | `/metrics` | Snapshot pontual de CPU/RAM/GPU + última geração (polling) |
| `WS` | `/ws/metrics` | Mesmo snapshot, em push a cada ~1s (usado pelo painel de Monitor) |
| `GET` | `/stats` | Números agregados de `logs/interactions.jsonl`, usados na homepage |

## Testes

Duas suítes, rodadas em paralelo pelo CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
a cada push na `main` e em todo pull request.

**API (pytest)** — roteamento por provider, métricas do monitor e log estruturado.
Herméticos: sem Ollama, sem GPU, sem rede. Rodam em ~1s. Ver
[tests/api/README.md](tests/api/README.md):

```powershell
cd backend
.venv\Scripts\pip install -r requirements-dev.txt   # uma vez
cd ..
.\backend\.venv\Scripts\python -m pytest
```

**E2E (Playwright)** — chat, monitor, homepage e seletor de provider. Ver
[frontend/tests/README.md](frontend/tests/README.md):

```powershell
cd frontend/tests
npm install && npx playwright install chromium   # uma vez
npm test
```

Os testes E2E que dependem de geração real do LLM se auto-pulam quando a API não
está no ar (é o caso do CI); os demais mockam o backend e rodam em qualquer lugar.
Para rodar tudo de uma vez localmente — subindo backend, checando os providers e
executando as duas suítes — use `scripts\start_dev.bat`. Para subir o ambiente
gerando pelo servidor, sem carregar o LLM local, use `scripts\start_dev_remote.bat`
(não roda a suíte E2E, para não gastar fila do servidor a cada execução).

Todas as interações são logadas em `logs/interactions.jsonl` (JSON Lines), uma linha por interação, com pergunta, contexto recuperado, resposta, métricas e metadados — isso alimenta a Fase 2 (RAGAS) e a escrita do artigo.

## Fase 2 — Arquitetura de testes

Em construção. Vai cobrir: ingestão de corpus de teste via ZIP, geração de dataset sintético + curadoria manual, avaliação com RAGAS contra a API real, testes de API (pytest) e testes E2E (Playwright/TypeScript). Detalhes em `docs/ARCHITECTURE.md` conforme forem implementados.

## Fase 3 — Human-in-the-loop (proposta)

Camada de revisão humana e relatório consolidado comparando notas do RAGAS com avaliação humana — a implementar mediante confirmação.

## Estrutura do projeto

```
backend/app/              API FastAPI e pipeline de RAG (ingestion, retrieval, generation, vector_store)
backend/app/providers.py  Registro local/remoto e qual está ativo (embeddings são sempre locais)

frontend/chat.html        Chat + Monitor — HTML/CSS/JS puro, sem build
frontend/components/      chat.js, monitor.js e motion/ (guard de animação, indicador de abas)
frontend/styles/          Tokens compartilhados + CSS por página
frontend/assets/fonts/    IBM Plex Sans e Mono auto-hospedadas
frontend/home/            Fonte da homepage (React + Vite) — só isto precisa de build
frontend/index.html       GERADO por frontend/home/ — não editar à mão
frontend/home-assets/     GERADO por frontend/home/ — JS/CSS com hash
frontend/tests/           Suíte E2E (Playwright)

data/source_pdfs/         PDFs de produção (não versionados)
data/vector_store/        Índice ChromaDB persistido (não versionado)
scripts/                  Setup do Ollama, ingestão via CLI e os .bat de ambiente
scripts/start_dev.bat        sobe tudo em modo local + roda as duas suítes
scripts/start_dev_remote.bat sobe tudo gerando pelo servidor, sem LLM local
tests/                    API, RAGAS, revisão humana (Fase 2/3)
logs/                     interactions.jsonl (log estruturado, não versionado)
docs/ARCHITECTURE.md      Decisões técnicas e racional
```
```