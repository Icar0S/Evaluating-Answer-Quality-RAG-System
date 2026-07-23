# Evaluating Answer Quality of a RAG System

Assistente de RAG 100% local, especializado em teste de sistemas baseados em LLM/RAG, com arquitetura de testes (RAGAS + Playwright) construída ao redor dele. Base experimental de um projeto de pesquisa sobre avaliação de qualidade de resposta em RAG usando LLM-as-a-Judge com Human-in-the-Loop.

Veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para as decisões técnicas e o racional por trás delas.

## Pré-requisitos

- [Ollama](https://ollama.com/download) instalado
- Python 3.11+
- Node.js 18+ (para os testes E2E com Playwright, Fase 2)
- GPU com pelo menos 8GB de VRAM recomendada (testado em RTX 4060 Laptop 8GB)

## Fase 0 — Setup do Ollama

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

## Fase 1 — Rodando o RAG local

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

### Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/chat` | Envia uma pergunta, retorna resposta + fontes + metadados |
| `POST` | `/ingest` | Reprocessa os PDFs de `data/source_pdfs/` |
| `GET` | `/health` | Status da API, do Ollama e do vector store |
| `GET` | `/metrics` | Snapshot pontual de CPU/RAM/GPU + última geração (polling) |
| `WS` | `/ws/metrics` | Mesmo snapshot, em push a cada ~1s (usado pelo painel de Monitor) |
| `GET` | `/stats` | Números agregados de `logs/interactions.jsonl`, usados na homepage |

### Testes E2E do frontend

`frontend/tests/` tem uma suíte Playwright que cobre chat, monitor e homepage —
ver [frontend/tests/README.md](frontend/tests/README.md). Roda em ~1-2min e serve
para pegar regressões de UI rapidamente a cada mudança, sem validação manual:

```powershell
cd frontend/tests
npm install && npx playwright install chromium   # uma vez
npm test
```

Todas as interações são logadas em `logs/interactions.jsonl` (JSON Lines), uma linha por interação, com pergunta, contexto recuperado, resposta, métricas e metadados — isso alimenta a Fase 2 (RAGAS) e a escrita do artigo.

## Fase 2 — Arquitetura de testes

Em construção. Vai cobrir: ingestão de corpus de teste via ZIP, geração de dataset sintético + curadoria manual, avaliação com RAGAS contra a API real, testes de API (pytest) e testes E2E (Playwright/TypeScript). Detalhes em `docs/ARCHITECTURE.md` conforme forem implementados.

## Fase 3 — Human-in-the-loop (proposta)

Camada de revisão humana e relatório consolidado comparando notas do RAGAS com avaliação humana — a implementar mediante confirmação.

## Estrutura do projeto

```
backend/app/          API FastAPI e pipeline de RAG (ingestion, retrieval, generation, vector_store)
frontend/              Chat HTML/CSS/JS puro
data/source_pdfs/      PDFs de produção (não versionados)
data/vector_store/     Índice ChromaDB persistido (não versionado)
scripts/                Setup do Ollama e ingestão via CLI
tests/                  API, RAGAS, E2E, revisão humana (Fase 2/3)
logs/                   interactions.jsonl (log estruturado, não versionado)
docs/ARCHITECTURE.md    Decisões técnicas e racional
```
