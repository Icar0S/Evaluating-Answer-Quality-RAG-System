# Decisões de Arquitetura

Este documento registra as decisões técnicas tomadas na construção do projeto e o racional por trás delas, servindo de insumo direto para a seção de metodologia do artigo-alvo.

## Fase 0 — Escolha do modelo local

### Kimi K3 descartado

O modelo inicialmente cogitado, Kimi K3 (Moonshot AI), foi descartado antes de qualquer tentativa de download pelos seguintes motivos:

- ~2,8T de parâmetros (MoE, ~50B ativos, 1M de contexto); pesos completos não publicados até o momento (previsão da Moonshot: até 27/07/2026).
- Mesmo após publicado, o download estimado (1,4–1,7TB) é inviável fora de clusters de GPU corporativos.
- Os modelos "menores" da mesma família (Kimi K2 a K2.7, ~1T de parâmetros) exigem 240–350GB de RAM+VRAM somados mesmo em quantização agressiva (1–2 bit) — incompatível com hardware de estação de trabalho/notebook.
- A tag `kimi-k2.6:cloud` do Ollama roda nos servidores da Moonshot, não localmente, o que viola o requisito de execução 100% local.

### Hardware detectado e modelo escolhido

Detecção via `nvidia-smi` e `Get-CimInstance Win32_ComputerSystem`:

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8GB VRAM |
| RAM | ~16GB |
| Espaço livre em D: | ~430GB |

Isso posiciona o hardware no tier "GPU ≤ 8GB VRAM" da tabela de referência do projeto. Modelos escolhidos:

- **Geração:** `qwen3:8b` — já presente localmente em `D:\modelosLLM\models`, validado com prompt de teste via `/api/generate`.
- **Embeddings:** `nomic-embed-text` — leve (274MB), embeddings de 768 dimensões, validado via `/api/embeddings`.

`OLLAMA_MODELS` já estava configurado para `D:\modelosLLM\models` (variável de ambiente de usuário) antes do início do projeto; confirmado que os blobs dos modelos residem em `D:\modelosLLM\models\blobs`.

## Fase 1 — RAG local

### Decisões confirmadas com o usuário

| Decisão | Escolha | Racional |
|---|---|---|
| Framework de orquestração | Pipeline manual (sem LangChain/LlamaIndex) | Transparência total do fluxo (chunking → embedding → retrieval → geração), essencial para instrumentar logging fino e reprodutibilidade granular exigidos pelo artigo. Evita abstrações que escondem o comportamento real do sistema, que é justamente o que está sendo medido. |
| Vector store | ChromaDB (persistência local em `data/vector_store/`) | Simplicidade de setup, persistência em disco nativa, sem necessidade de serviço externo (ao contrário de Qdrant, que exigiria Docker). Suficiente para a escala de um corpus de artigos acadêmicos. |
| Modelo de geração | `qwen3:8b` | Ver seção acima. |
| Modelo de embeddings | `nomic-embed-text` | Ver seção acima. |
| Backend | FastAPI | Integração natural com chamadas assíncronas ao Ollama, schemas tipados via Pydantic, e testabilidade (pytest + TestClient) direta para a Fase 2. |
| Frontend | HTML/CSS/JS puro, sem framework | Mantém o frontend desacoplado do backend, facilitando tanto testes de API isolados quanto testes E2E via Playwright na Fase 2. |

### Chunking

- Chunking feito **por página** do PDF (via PyMuPDF), não pelo documento inteiro — isso permite atribuir a página exata como fonte em cada chunk, importante para os testes de fidelidade/citação da Fase 2.
- Contagem de tokens via `tiktoken` (encoding `cl100k_base`) como aproximação genérica — não é o tokenizer real do Qwen3, mas dá uma medida consistente e configurável de tamanho de chunk independente do modelo usado.
- Tamanho e overlap configuráveis via `.env` (`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`), padrão 800/120 tokens (~15% de overlap).

### Grounding e prevenção de alucinação

O prompt de sistema (`backend/app/rag/generation.py`) instrui o modelo a:
1. Responder somente com base no contexto recuperado.
2. Usar uma frase-âncora fixa (`NOT_FOUND_MARKER`) quando o contexto não cobrir a pergunta, em vez de inventar uma resposta.

O campo `grounded` na resposta da API é derivado heuristicamente: `True` quando há chunks recuperados **e** a resposta não contém a frase-âncora. Essa heurística é propositalmente simples — a avaliação real de fidelidade/alucinação fica a cargo do RAGAS na Fase 2, que usa um LLM-juiz para julgar a resposta contra o contexto, não apenas um regex.

### Logging estruturado

Cada chamada a `/chat` grava uma linha em `logs/interactions.jsonl` com: id da interação, timestamp UTC, pergunta, resposta, chunks recuperados (com score), flag `grounded`, e metadados (modelo, versão do prompt, latências de retrieval/geração, contagem de tokens). Esse formato foi desenhado para ser lido diretamente por pandas na Fase 2/3 sem parsing adicional, e para preservar histórico suficiente para a seção de resultados do artigo.

### Versão do prompt

`PROMPT_VERSION` no `.env` (padrão `v1`) é registrado em cada interação. Deve ser incrementado manualmente sempre que o `SYSTEM_PROMPT_TEMPLATE` mudar de forma material, para que comparações longitudinais de métricas do RAGAS possam ser segmentadas por versão de prompt.

### Nota técnica: ChromaDB no Windows + Python 3.12

`chromadb==0.5.23` (versão inicialmente fixada) depende de `chroma-hnswlib`, que só publica wheels pré-compiladas para Windows até o Python 3.11 — no 3.12 o pip tenta compilar a extensão C++ localmente e falha sem o Visual C++ Build Tools instalado. Resolvido fixando `chromadb==1.5.9`: a partir da série 1.x o índice HNSW foi reescrito em Rust e embutido no próprio pacote `chromadb`, publicado como wheel `abi3-win_amd64` (compatível com qualquer Python ≥3.9), eliminando a dependência de compilação local. Sem impacto na API usada (`PersistentClient`, `get_or_create_collection`, `add`, `query`).

### Validação end-to-end da Fase 1

- Ingestão rodada com o corpus real do usuário: 19 PDFs acadêmicos sobre teste de LLM/RAG → 536 chunks indexados (`python scripts/ingest_documents.py`).
- `/health` confirmou `vector_store_documents: 536` e `ollama_reachable: true`.
- `/chat` testado com pergunta real ("O que é RAGAS e quais métricas ele avalia?"): resposta corretamente fundamentada no PDF `RAGAS.pdf` (página 7), `grounded: true`, fontes com score de similaridade.
- `/chat` testado sem contexto disponível (antes da ingestão): resposta usou corretamente a frase-âncora de "não encontrado", `grounded: false` — validando a defesa contra alucinação antes mesmo da Fase 2.
- Frontend validado visualmente via Playwright headless (servido em `http://localhost:5500`, API em `http://localhost:8000`): mensagem do usuário aparece, indicador de carregamento é exibido, resposta do assistente renderiza com seção de fontes expansível e metadados (modelo/latência), indicador de status fica verde. Nenhum erro de console além de um 404 de favicon (inofensivo).

## Redesign do frontend + monitor de recursos + homepage

Feito entre a Fase 1 e a Fase 2, a pedido do usuário, para dar uma base visual mais
madura antes de começar a instrumentar testes E2E sobre ela.

### Estrutura de arquivos

O frontend continua HTML/CSS/JS puro (decisão da Fase 1), agora com duas páginas
navegáveis por link normal (sem router JS):

```
frontend/
├── index.html   → homepage/landing
├── home.css / home.js
├── chat.html     → chat (era o antigo index.html)
├── chat.css / chat.js
├── monitor.css / monitor.js   → painel de monitoramento, embutido em chat.html
└── shared.css     → tokens de design (cores, tipografia, sombras) usados pelas duas páginas
```

Dentro de `chat.html`, "Chat" e "Monitor" são painéis (`.view`) alternados via JS no
trilho de ícones lateral — não são rotas separadas, para não introduzir um router
num projeto propositalmente sem build step.

### Histórico de sessões (client-side)

Como o RAG é uma sessão única (não multi-contato), a lista lateral do chat vira
histórico de conversas salvas em `localStorage` (`rag_sessions_v1` / `rag_active_session_v1`),
não no backend — cada sessão guarda suas mensagens, reações (emoji, decorativas) e
flag de favorito. Trade-off consciente: histórico não é compartilhado entre
navegadores/dispositivos nem sobrevive a um "limpar dados do site". Se isso vier a
importar para o artigo (ex. precisar correlacionar sessão ↔ métricas do RAGAS na
Fase 2), migrar para persistência no backend é a próxima etapa natural.

### Monitor de LLM em tempo real

- `backend/app/metrics.py`: `psutil` para CPU/RAM; `nvidia-ml-py` (pacote `pynvml`)
  para GPU/VRAM, escolhido em vez de `GPUtil` por ser mantido ativamente. Detecção
  de GPU é best-effort — qualquer falha na inicialização do NVML deixa `gpu.available=False`
  em vez de derrubar a API.
- Tokens/s e latências não exigem instrumentação própria: vêm de `eval_count`,
  `eval_duration`, `prompt_eval_duration` e `total_duration`, já retornados pelo
  Ollama em `/api/chat`. `tokens_per_second = completion_tokens / (eval_duration / 1e9)`.
- Status idle/processing rastreado por um contador simples em memória
  (`mark_generation_start`/`mark_generation_end`), incrementado/decrementado ao
  redor da chamada de geração em `/chat`.
- Transporte: `WS /ws/metrics` (push a cada 1s, principal) com fallback automático
  no frontend para polling via `GET /metrics` a cada 1.5s se o WebSocket cair.
- Sparklines de CPU/GPU (bônus do pedido original) implementadas com `<canvas>` puro,
  sem lib de gráficos — buffer das últimas 30 amostras mantido em `metrics.py`.

### Homepage

Números da seção "case-study" vêm de `GET /stats` (lê `logs/interactions.jsonl` +
contagem do vector store) — nada é hardcoded. `ragas_faithfulness_avg` fica `null`
com nota explícita ("disponível após a Fase 2") em vez de um número inventado, já
que a avaliação RAGAS ainda não existe. O HUD do hero reaproveita `GET /metrics`
via polling simples (a homepage não precisa da latência de WebSocket).

### Bug de CSS Grid encontrado e corrigido

Ao alternar para a view do Monitor, a sidebar recebe `display:none`. Isso a remove
do fluxo do grid inteiramente — como `.app-shell` não tinha `grid-column` explícito
em cada filho, o auto-placement do CSS Grid reindexava `.main-panel` para a 2ª
coluna (que tínhamos zerado), deixando-o com largura 0. Corrigido fixando
`grid-column: 1/2/3` explicitamente em `.icon-rail`/`.sidebar`/`.main-panel`, para
que a ausência de um item do fluxo não desloque os demais. Achado via Playwright
headless + `getBoundingClientRect()` — não seria óbvio só lendo o CSS.

### Validação

Redesign, monitor e homepage testados via Playwright headless (Chromium), navegando
de fato pelas páginas (não só carregando-as): envio de pergunta com sessão salva,
troca para o painel de Monitor com sparklines populadas em tempo real, criação de
nova sessão, e navegação da homepage para o chat pelo CTA. Screenshots conferidos
visualmente a cada etapa.

## Fase 2 — Arquitetura de testes (a documentar conforme implementado)

## Fase 3 — Human-in-the-loop (a documentar caso confirmado)
