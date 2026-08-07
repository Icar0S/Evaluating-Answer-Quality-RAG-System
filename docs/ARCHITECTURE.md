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
| Frontend | HTML/CSS/JS puro, sem framework | Mantém o frontend desacoplado do backend, facilitando tanto testes de API isolados quanto testes E2E via Playwright na Fase 2. **Revisado depois** — ver "Camada de movimento e ilha React" mais abaixo: a homepage passou a ter uma ilha React, o chat continua puro. |

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

### Ajustes de UX pós-redesign

Duas rodadas de correção a pedido do usuário, após uso real da interface:

- **Scroll do chat travando após múltiplas mensagens**: bug clássico de flexbox —
  `.chat-log` tinha `flex: 1` mas não `min-height: 0`, então em vez de encolher e
  ativar seu próprio `overflow-y: auto`, o elemento crescia para caber todo o
  conteúdo e empurrava a barra de input para fora da área visível. Corrigido
  adicionando `min-height: 0` em `.chat-log` e `.main-panel` (mesma classe de bug
  se propagaria para `.monitor-grid`, corrigido preventivamente).
- **Monitor ocupando a tela inteira**: redesenhado para caber na sidebar (troca de
  painel, não de rota) — clicar no ícone de Monitor substitui a lista de sessões
  pelos cards compactos de métricas, mas o chat permanece visível e utilizável ao
  lado, permitindo monitorar consumo enquanto conversa. Isso eliminou de quebra o
  hack de `grid-template-columns` dinâmico que tinha causado o bug de CSS Grid
  documentado acima — o layout do `main-panel` agora é sempre o mesmo.
- **Emoji picker removido** do compositor de mensagens (desnecessário para um chat
  com um assistente LLM). As reações em emoji nas respostas do assistente (👍 👎 💡
  ❓, feedback rápido) foram mantidas — é uma funcionalidade diferente.
- **Input redesenhado com inspiração no Claude Code**: borda com glow sutil ao
  focar (`:focus-within`), rodapé com o nome do modelo ativo e atalhos de teclado
  em estilo `<kbd>` ("Enter enviar · Shift+Enter nova linha"), contador de
  caracteres que só aparece perto do limite de 4000, e botão de enviar
  desabilitado (visualmente neutro, não só opaco) enquanto o campo está vazio.

## Provider Ollama local/remoto

Motivação: manter o Ollama local rodando (`qwen3:8b`, GPU de 8GB) só para poder testar
a aplicação passou a ser o principal atrito no dia a dia de desenvolvimento. Já existia
um segundo Ollama rodando 24/7 num Mac mini M4 da rede doméstica (setup documentado em
`SETUP-LLM-LOCAL.md`, originalmente feito para uso como backend de agente de código via
Aider), acessível via Tailscale. Objetivo: permitir que `/chat`/`/ingest` apontem para
esse servidor sem tocar no fluxo local existente.

### Desenho

Um "provider" é um alvo Ollama completo (`base_url` + modelo de geração + modelo de
embeddings) — `local` (as variáveis `OLLAMA_*`/`GENERATION_MODEL`/`EMBEDDING_MODEL` de
sempre, inalteradas) e `remote` (novo, opcional — só existe se `REMOTE_OLLAMA_BASE_URL`
estiver preenchido no `.env`). `backend/app/providers.py` mantém qual está ativo como
estado em memória (protegido por lock, mesmo padrão de `app/metrics.py`), inicializado
a partir de `ACTIVE_PROVIDER` do `.env` e trocável em runtime via
`POST /providers/active` — decisão deliberada de não exigir restart do backend nem
edição de `.env` para alternar durante os testes manuais do dia a dia. `generation.py`,
`ingestion.py` (embeddings) e `metrics.py` resolvem o provider ativo a cada chamada em
vez de ler `Settings` diretamente.

### Métricas de hardware são só do provider local

`psutil`/`pynvml` (ver seção "Monitor de LLM em tempo real" acima) só enxergam a
máquina onde o backend roda — não há como ler CPU/RAM/GPU do Mac mini pela rede sem um
agente de métricas rodando lá. Avaliadas duas opções: (a) construir esse agente
(script Python + LaunchAgent no Mac mini, no mesmo estilo do `backup-db.sh` que já
existe lá, exposto por HTTP), ou (b) aceitar que hardware remoto fica fora do alcance e
mostrar isso explicitamente na UI. Optado por (b) — decisão do usuário, para não
adicionar mais um serviço 24/7 a manter no Mac mini por ora. `MetricsSnapshot` ganhou
`host_metrics_available: bool`; quando `false` (provider remoto ativo), CPU/RAM/GPU
aparecem como "indisponível" no painel de Monitor em vez de números da máquina local
(que estariam tecnicamente corretos, mas seriam enganosos — não refletem a máquina que
está de fato gerando a resposta). Tokens/s, latência e status idle/processando
continuam reais nos dois modos, porque vêm da própria resposta do Ollama
(`eval_count`/`eval_duration`/etc.), não de instrumentação local — o mesmo mecanismo já
descrito na seção de monitor acima, só que agora provider-aware.

### UI

O painel de Monitor ganhou um seletor Local/Servidor (`#provider-tabs`, reaproveitando
o componente visual de pill-tabs já usado nos filtros de sessão/mensagens) no topo,
abaixo do indicador de status. Trocar de aba chama `POST /providers/active` de verdade
— não é só uma troca de visualização, redireciona `/chat`/`/ingest` para o provider
escolhido.

### Migração: de Ollama remoto (Tailscale) para a API smartdatatest

O desenho acima assumia que "remoto" seria sempre um segundo Ollama completo,
espelhando o local (mesma rota `/api/chat`, mesmo endpoint de embeddings). Isso
mudou quando o Mac mini passou a expor uma API própria e autenticada em
`https://llm.smartdatatest.com` (ver [llm-api-referencia.md](llm-api-referencia.md),
gerado automaticamente do OpenAPI dela) em vez do Ollama cru na porta 11434 via
Tailscale — o motivo é segurança: `SETUP-LLM-LOCAL.md` já registrava que "Ollama
não tem autenticação — quem alcançar a porta 11434 usa seus modelos", e o gateway
resolve isso com Bearer auth, fila com 429/`Retry-After` e 503 documentados, e
governor de recursos (a máquina divide 16GB com uma aplicação em produção).

Essa API **não é Ollama-compatível**, o que forçou duas mudanças de design, não
só de endereço:

1. **`Provider` ganhou `kind: "ollama" | "smartdatatest"` e `api_key`.**
   `generation.py` ramifica entre `_generate_ollama()` (inalterado) e
   `_generate_smartdatatest()` (`POST /v1/chat`, `Authorization: Bearer`, sem
   `corpus_id` — o RAG usado continua sendo o nosso, retrieval local + ChromaDB,
   não o deles). A API remota não separa prompt-eval de geração como o Ollama
   (só `duration_s` + `queue_wait_s`), então `prompt_eval_latency_ms` fica `null`
   nesse modo — mostrado como "—" no monitor em vez de um número inventado.
   `check_reachable()` também ramifica: `smartdatatest` usa `GET /v1/ready`
   (documentado como propositalmente sem autenticação, "é o alvo do monitor
   externo") em vez de `/api/tags`.

2. **Embeddings deixaram de seguir o provider ativo — agora são sempre locais.**
   Era uma premissa do design anterior (fazia sentido só porque um Ollama remoto
   *poderia* ter `nomic-embed-text` instalado também); a API smartdatatest **não
   expõe rota de embeddings** (só usa `nomic-embed-text` internamente para o
   RAG dela, via `/v1/corpora` — que este projeto não usa, para não delegar a
   avaliação a um retrieval de terceiros). `ingestion.py::embed_texts()` agora
   ignora `providers.get_active_provider()` de propósito e sempre usa
   `settings.ollama_base_url`. Isso também corrige um risco que já existia no
   design anterior mesmo se o remoto fosse Ollama de verdade: misturar espaços
   de embedding entre chamadas — cosine similarity entre vetores de modelos
   diferentes não tem significado — teria corrompido a busca silenciosamente,
   sem erro, só respostas piores.

Variáveis renomeadas no `.env`/`Settings` para não sugerir Ollama onde não há mais
um (`REMOTE_OLLAMA_BASE_URL` → `REMOTE_API_BASE_URL`, `REMOTE_OLLAMA_LABEL` →
`REMOTE_LABEL`, `+REMOTE_API_KEY`, `-REMOTE_EMBEDDING_MODEL`). O modelo padrão do
provider remoto (`qwen3:8b`) foi escolhido igual ao local deliberadamente — mesmo
modelo, hardware/hospedagem diferentes, para permitir comparação metodológica
local-vs-remoto mais adiante no artigo, em vez do default da API (`qwen3:4b`,
mais leve).

`tests/api/conftest.py::LlmBackendMock` mocka os dois protocolos (renomeado de
`OllamaMock`); `test_chat.py` tem um teste dedicado confirmando que trocar para o
remoto redireciona *só* a geração — embeddings continuam batendo no host local —
e outro para o 429 com `Retry-After`. Validado também contra a API real (não
mockada) antes de fechar a mudança: `/v1/ready`, `/v1/models` (catálogo real,
confirmou `qwen3:8b` disponível) e um `/chat` completo da nossa API com o
provider remoto ativo, ponta a ponta.

## Fase 2 — Arquitetura de testes (a documentar conforme implementado)

### Testes determinísticos vs. testes que exercitam o modelo

O sistema tem duas classes de teste com propósitos diferentes, e misturá-las
degradaria as duas:

- **Determinísticos** (`tests/api/`, `frontend/tests/e2e/providers.spec.ts`) —
  verificam a *mecânica* do sistema: roteamento por provider, formato dos
  metadados, o que é gravado no log, o que a UI mostra em cada estado. Não
  envolvem o LLM: o Ollama é substituído por `httpx.MockTransport` no backend e
  por `page.route`/`page.routeWebSocket` no frontend. Rodam em segundos, não
  flutuam e não exigem hardware.
- **Não-determinísticos** (`frontend/tests/e2e/chat-flow.spec.ts` e, na Fase 2, a
  avaliação RAGAS) — exercitam a geração real. São lentos (dezenas de segundos por
  pergunta), dependem de GPU e do modelo carregado, e sua saída varia entre
  execuções. Só fazem sentido rodando localmente contra a stack de verdade.

O CI do GitHub roda os dois grupos, mas só o primeiro produz asserções lá: os
testes que precisam do LLM se auto-pulam via `test.skip(!isBackendUp(...))` em vez
de falhar. Isso mantém o pipeline honesto — falha vermelha significa regressão de
verdade, não ausência de GPU no runner.

Consequência prática para o artigo: métricas de *qualidade de resposta* (fidelidade,
alucinação) nunca vêm do CI — vêm da execução local do RAGAS registrada em
`logs/interactions.jsonl`. O CI cobre a infraestrutura que produz esses números,
não os números.

### Isolamento dos testes de API

Três estados globais precisaram ser neutralizados por fixture, todos documentados
em `tests/api/README.md`: o `@lru_cache` de `get_settings()`, o provider ativo
(estado de módulo em `providers.py`) e o `.env` da máquina — que, sem
sobrescrita por variável de ambiente, faria o resultado do teste depender de quem
está rodando. Os endereços de Ollama usados nos testes apontam para hosts
inexistentes de propósito: se um mock falhar, o teste quebra com erro de conexão
em vez de acertar silenciosamente o Ollama local e passar por acidente.

Esse desenho já se pagou na primeira execução: o teste de listagem de providers
pegou uma variável de ambiente documentada como `REMOTE_OLLAMA_LABEL` mas lida
pelo backend como `REMOTE_LABEL` (o campo em `Settings` tinha nome divergente).
O `.env` era silenciosamente ignorado e o bug era invisível na UI, porque o valor
default coincidia com o texto configurado.

## Fase 3 — Human-in-the-loop (a documentar caso confirmado)

## Camada de movimento e ilha React na homepage

### O que mudou em relação à decisão original

A Fase 1 registrou "HTML/CSS/JS puro, sem framework" para o frontend inteiro.
Isso continua valendo para `chat.html`, que é a tela de trabalho e onde a suíte
E2E tem mais superfície. A **homepage** passou a ser híbrida:

- `frontend/home/` — fonte React + Vite (só a homepage).
- `frontend/index.html` e `frontend/home-assets/` — **saída de build, versionada**.
- `chat.html`, `components/` e `styles/` — inalterados quanto a framework.

O build ser versionado é o que preserva a propriedade que motivou a decisão
original: `python -m http.server` (README, `playwright.config.ts`, CI) continua
servindo o projeto sem nenhuma etapa de build. Rodar o projeto não exige Node;
só *mudar a homepage* exige (`cd frontend/home && npm run build`).

A homepage é uma **ilha**, não uma aplicação: nav, hero, narrativa, stack e rodapé
são HTML estático. O React é dono de dois pontos, os dois que têm estado — o HUD
(`#hud-root`, polling de `/metrics`) e o painel de números (`#stats-root`,
`/stats`). Isso foi medido, não estimado: com a página inteira em React o LCP era
536 ms porque nada pintava antes do JS executar; com o hero estático, voltou para
68 ms (o original era 88 ms).

### Por que o gsap foi removido

Dois componentes vieram do React Bits (MIT + Commons Clause), copiados para dentro
do projeto e adaptados: `SplitFlapText` e `AnimatedContent`. O `AnimatedContent`
original anima via GSAP + ScrollTrigger. Medido neste projeto, gsap + ScrollTrigger
custavam **46,2 KB gzipped — 48% do bundle da homepage** — para fazer um bloco
entrar com fade e 24 px de deslocamento. Trocado por `IntersectionObserver` + uma
transition CSS, que produz o mesmo resultado por ~0,3 KB. A API e o comportamento
do componente não mudaram; reverter é reinstalar o gsap e trocar o motor.

O `SplitFlapText` mantém o motor original (requestAnimationFrame + CSS 3D). Foi
adaptado de carrossel (`words[]` ciclando num timer) para um valor único que vira
quando o dado chega do backend, que é o que existe neste domínio.

### Cor semântica

A paleta deixou de ser decorativa. Três cores têm papel fixo e não aparecem fora
dele — é o que faz a interface dizer, sem texto, o que o sistema sabe:

| Token | Papel |
|---|---|
| `--cite-*` (âmbar) | Evidência: fonte citada, resposta fundamentada, leitura vinda do backend |
| `--probe-*` (ciano) | Telemetria do instrumento: CPU/RAM/GPU/VRAM, tok/s, latência |
| `--clay-*` (argila) | Ausência: contexto não encontrado, recusa do modelo, erro |

Todo o resto é monocromático. Afordância de interação (foco, aba ativa, botão
primário) se resolve com contraste, nunca com um quarto acento. Os contrastes
foram verificados com axe-core: zero violações WCAG 2.1 AA nas duas páginas.

### Desligar o movimento

`frontend/components/motion/guard.js` é a fonte única de verdade, carregada como
script bloqueante no `<head>` das duas páginas para definir `<html data-motion>`
antes do primeiro paint. Desliga com qualquer uma destas:

| Como | Para quê |
|---|---|
| `?motion=off` na URL | Testes E2E determinísticos, depuração |
| `localStorage.motion = 'off'` | Preferência persistente do usuário |
| `VITE_DISABLE_MOTION=true` no build | Artefato permanentemente sem movimento |
| `prefers-reduced-motion: reduce` | Preferência do sistema operacional |

Com o movimento desligado, componentes de entrada por scroll renderizam no estado
final imediatamente, sem `IntersectionObserver` — é o que impede que a suíte
headless dependa de timing de animação.

