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

## Fase 2 — Arquitetura de testes (a documentar conforme implementado)

## Fase 3 — Human-in-the-loop (a documentar caso confirmado)
