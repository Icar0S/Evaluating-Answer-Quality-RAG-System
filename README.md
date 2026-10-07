# Evaluating Answer Quality of a RAG System

> **Artefato associado ao artigo:** "Relato de Experiência: Arquitetura de Testes
> e Validação Semântica para um Sistema RAG no Setor Público", aceito no
> SAST 2026 (11th Brazilian Symposium on Systematic and Automated Software
> Testing), CBSoft 2026, São Paulo, SP.
>
> Este repositório é uma **implementação de referência aberta** dos princípios
> descritos no artigo. O código, o golden dataset e o corpus regulatório do
> sistema original não podem ser publicados por conterem informações
> operacionais internas da organização parceira.
>
> 📄 **Camera-ready:** [docs/paper.pdf](docs/paper.pdf)

Assistente de RAG 100% local, especializado em teste de sistemas baseados em LLM/RAG, com arquitetura de testes (RAGAS + Playwright) construída ao redor dele. Base experimental de um projeto de pesquisa sobre avaliação de qualidade de resposta em RAG usando LLM-as-a-Judge com Human-in-the-Loop.

Veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para as decisões técnicas e o racional por trás delas.

## Início rápido

Pré-requisitos: [Ollama](https://ollama.com/download) instalado e aberto, e
Python 3.11+. Todos os comandos abaixo rodam em PowerShell, **a partir da raiz
do projeto**. Não é preciso ativar a venv nem entrar em subpastas.

```powershell
# 1. Modelos (uma vez). Escolha o de geração pela tabela logo abaixo.
ollama pull nomic-embed-text
ollama pull gemma3:4b

# 2. Backend (uma vez)
python -m venv backend\.venv
backend\.venv\Scripts\pip install -r backend\requirements.txt
copy .env.example .env
#    -> abra o .env e ajuste GENERATION_MODEL conforme a tabela abaixo

# 3. Documentos: coloque os PDFs em data\source_pdfs\ e indexe
#    (uma vez, e de novo sempre que mudar os PDFs ou o EMBEDDING_MODEL)
backend\.venv\Scripts\python scripts\ingest_documents.py

# 4. Dia a dia: sobe backend + frontend em janelas próprias e abre o navegador
scripts\start_dev.bat
```

Se os testes E2E (Node) estiverem instalados, o `start_dev.bat` também os
executa. Se não estiverem, ele só avisa e pula essa etapa. Para subir sem o
script, use dois terminais:

```powershell
# terminal 1: API em http://localhost:8010
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8010

# terminal 2: frontend em http://localhost:5510
cd frontend
..\backend\.venv\Scripts\python -m http.server 5510
```

Depois abra **http://localhost:5510/index.html**. Não abra o arquivo direto
(`file://`), porque o navegador bloqueia as chamadas à API. Para conferir se
está tudo no ar, rode `curl.exe http://localhost:8010/health`.

> **Antes da primeira pergunta**, confira a escolha do modelo na tabela a
> seguir. Com o `.env.example` sem ajuste e uma máquina sem GPU dedicada, o chat
> fica "pensando" até dar timeout. Ver [Problemas comuns](#problemas-comuns).

### Qual modelo de geração usar

O modelo de geração é o que mais pesa no tempo de resposta. O de embeddings
(`nomic-embed-text`) é leve e é sempre o mesmo. Ajuste no `.env` e reinicie o
backend:

| Sua máquina | No `.env` | Tempo por resposta* |
|---|---|---|
| GPU NVIDIA com 8 GB+ de VRAM | `GENERATION_MODEL=qwen3:8b` (padrão) | ~15–55 s com raciocínio; ~10–20 s com `GENERATION_THINK=false` |
| GPU com menos de 8 GB, ou notebook sem GPU dedicada | `GENERATION_MODEL=gemma3:4b` | ~5–30 s em GPU; em CPU, pode passar de 1 min |
| Só CPU, máquina modesta | `GENERATION_MODEL=gemma3:4b` + `GENERATION_TIMEOUT_SECONDS=600` | minutos, mas responde |
| Sem GPU e com acesso ao servidor (chave de API) | modo servidor: `scripts\start_dev_remote.bat` | depende da fila do servidor ([Fase 1.5](#fase-15--provider-remoto-opcional)) |

\* Medido numa RTX 4060 de 8 GB com o prompt real do pipeline (4 trechos,
~2.400 tokens de entrada). O valor mais alto de cada faixa inclui o
carregamento do modelo, que só acontece na primeira pergunta depois de subir.

## Problemas comuns

### O chat fica "pensando" até dar timeout (mas a ingestão funcionou)

**Sintoma:** backend e frontend sobem, a ingestão informa os chunks indexados
(ex.: "192 chunks"), mas a pergunta fica carregando por uns 2 minutos e termina
em *"Tempo limite excedido ao gerar a resposta."* (HTTP 504).

**Causa:** se a ingestão funcionou, o Ollama e os embeddings estão OK. O que
estoura o tempo é a **geração**, que tem um limite de
`GENERATION_TIMEOUT_SECONDS` (120 s). O `.env.example` vem com `qwen3:8b`, um
modelo que *raciocina antes de responder*: são uns 500–700 tokens de
"thinking" antes da resposta. Em GPU isso leva segundos. Quando o modelo não
cabe na VRAM e roda em CPU, no todo ou em parte, passa fácil de 2 minutos.

**Como confirmar:** com uma pergunta em andamento, rode em outro terminal:

```powershell
ollama ps
```

Olhe a coluna `PROCESSOR`. `100% GPU` está ok. Já `100% CPU` ou uma divisão
como `45%/55% CPU/GPU` indica que o modelo não coube na VRAM, e esse é o
problema. Para ver a velocidade bruta, rode
`ollama run qwen3:8b "oi" --verbose` e olhe o `eval rate`. Abaixo de
~10 tokens/s, o `qwen3:8b` com raciocínio não cabe em 120 s.

**Correção:** escolha uma das opções, da mais simples para a mais lenta. Em
todas, edite o `.env` e **reinicie o backend**, porque ele só lê o `.env` ao
subir.

1. **Desligar o raciocínio**, mantendo o `qwen3:8b`: `GENERATION_THINK=false`.
   Gera ~3x menos tokens (727 → 224) e responde ~5x mais rápido (54 s → 10 s)
   na medição acima. Requer Ollama
   recente (0.9+); confira com `ollama --version`.
2. **Trocar por um modelo mais leve**: `ollama pull gemma3:4b` e
   `GENERATION_MODEL=gemma3:4b`.
3. **Dar mais tempo**: `GENERATION_TIMEOUT_SECONDS=600`. Esta opção resolve o
   timeout, mas não a lentidão.
4. **Gerar no servidor**, se você tiver a chave de API: `scripts\start_dev_remote.bat`.

### O backend não sobe

- `ModuleNotFoundError` ou `uvicorn` não encontrado: a venv não foi criada ou
  o comando não está usando ela. Rode o passo 2 do
  [Início rápido](#início-rápido) e suba com
  `.venv\Scripts\python -m uvicorn ...`, de dentro de `backend\`.
- Porta 8010 ocupada: veja o próximo item.

### "Porta ocupada por OUTRO serviço" (ou erros estranhos de JSON e 404)

O projeto usa a **8010** (API) e a **5510** (frontend). Até out/2026 eram a
8000 e a 5500, que colidem com qualquer Django/FastAPI em Docker e com o Live
Server do VS Code. Nesse caso, o `start_dev.bat` achava que o backend "já
estava no ar", mas quem respondia era o outro serviço. O resultado era um
`JSONDecodeError` no passo 2 e `Timed out waiting ... from config.webServer` no
Playwright. Hoje o script confere se quem responde é este projeto e, se não
for, para com uma mensagem. Para descobrir quem ocupa a porta:

```powershell
netstat -ano | findstr LISTENING | findstr :8010
tasklist /fi "PID eq <pid>"
docker ps    # se for um container
```

Feche o processo ou, se não puder, troque a porta deste projeto. Ela aparece
em `scripts\start_dev*.bat`, `CORS_ORIGINS` no `.env`,
`frontend\components\chat.js`/`home.js`, `frontend\home\src\api.js` (depois
`npm run build`) e `frontend\tests\playwright.config.ts`/`e2e\helpers.ts`.

### O chat responde "Não encontrei essa informação nos documentos fornecidos"

Para tudo, até para perguntas que estão nos PDFs: o índice está vazio ou foi
montado com outro modelo de embeddings. Confira se os PDFs estão em
`data\source_pdfs\` e rode a ingestão de novo (passo 3). Ela recria o índice do
zero. Isso também é obrigatório depois de trocar o `EMBEDDING_MODEL`.

### Erro 404 / "model not found" ao perguntar

O modelo do `GENERATION_MODEL` não está baixado nesse Ollama. Confira com
`ollama list` e baixe com `ollama pull <modelo>`. Se você rodou o
`setup_ollama.ps1`, lembre que ele muda a pasta dos modelos (`OLLAMA_MODELS`)
para `D:\modelosLLM\models`. Se o Ollama não foi reiniciado depois disso, os
modelos podem ter sido baixados na pasta antiga e "sumir" na reinicialização.
Em máquina sem disco `D:`, passe outro caminho:
`-ModelsPath "C:\caminho\que\existe"`.

### O frontend abre, mas o indicador de status fica vermelho

O backend não está respondendo em `http://localhost:8010`. Rode
`curl.exe http://localhost:8010/health` e veja a janela do backend. Confira
também se a página foi aberta por `http://localhost:5510` e não por `file://`.

## Instalação detalhada

O [Início rápido](#início-rápido) cobre o caminho comum. Esta seção explica cada
peça e as alternativas.

### Requisitos

- [Ollama](https://ollama.com/download) instalado. É **sempre necessário**,
  mesmo gerando pelo servidor remoto: a busca embeda a pergunta a cada consulta,
  e a API remota não expõe rota de embeddings. No modo servidor, o Ollama
  carrega só o `nomic-embed-text` (~274MB, roda em CPU).
- Python 3.11+
- GPU com pelo menos 8GB de VRAM, **só para gerar localmente com o
  `qwen3:8b`**. Com menos, use o `gemma3:4b` ou o modo servidor (ver
  [Qual modelo de geração usar](#qual-modelo-de-geração-usar)).
- Node.js 18+, para os testes E2E (Playwright) e para alterar a homepage
  (ilha React). Não é necessário só para *rodar* o projeto.

### Fase 0 — Setup do Ollama (opcional)

O passo 1 do Início rápido (`ollama pull`) já basta. O script abaixo faz o
mesmo e ainda guarda os modelos fora do disco C: e testa os dois modelos com um
prompt simples:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ollama.ps1 -GenerationModel gemma3:4b -ModelsPath "D:\modelosLLM\models"
```

Sem parâmetros, ele usa `qwen3:8b` e `D:\modelosLLM\models`. **Reinicie o
Ollama** depois da primeira execução, porque a variável `OLLAMA_MODELS` só vale
para processos novos. A tabela de modelos por faixa de hardware está em
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Fase 1 — Rodando o RAG local

Este é o mesmo fluxo do Início rápido, com as alternativas:

- **Ingestão pela API**, em vez do script: com o backend no ar, rode
  `curl.exe -X POST http://localhost:8010/ingest`.
- **Backend com reload automático** ao editar o código:
  `.venv\Scripts\python -m uvicorn app.main:app --reload --port 8010`
  (de dentro de `backend\`).
- O frontend precisa ser servido por HTTP, e não aberto como `file://`, por
  causa do WebSocket do monitor e do CORS. A homepage leva ao chat
  (`chat.html`), que tem um trilho lateral com as views **Chat** e **Monitor do
  LLM**.

Nenhuma etapa de build é necessária para *rodar* o projeto: a homepage é
servida a partir de `frontend/index.html` e `frontend/home-assets/`, que são
versionados. Só é preciso Node para **mudar** a homepage. Ver
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
| `GET` | `/stats` | Números agregados de `logs/interactions.jsonl` + última avaliação DeepEval, usados na homepage |

## Testes

Três suítes. As duas primeiras rodam em paralelo pelo CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)) a cada push na `main`
e em todo pull request; a terceira (DeepEval) roda só localmente — ver abaixo.

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

**Qualidade da resposta ([DeepEval](https://deepeval.com/docs/introduction))** — fidelidade ao
contexto, relevância da resposta, e precisão/recall/relevância da recuperação, julgados por um
LLM local via Ollama (nunca OpenAI/nuvem). Exercita a stack real (geração + juiz + vector store
populado); nunca roda no CI, mesma regra dos testes E2E de geração real. Ver
[tests/deepeval/README.md](tests/deepeval/README.md):

```powershell
python -m venv tests/deepeval/.venv
tests\deepeval\.venv\Scripts\pip install -r backend\requirements-eval.txt
ollama pull gemma3:4b   # modelo-juiz, diferente do de geração
tests\deepeval\.venv\Scripts\python.exe -m pytest tests/deepeval
```

(pytest puro, não `deepeval test run` — ver "Bug conhecido" em
[tests/deepeval/README.md](tests/deepeval/README.md).)

Pra ver os resultados com scores, limiares e o motivo de cada julgamento —
não só pass/fail no terminal — use `tests\deepeval\run_and_export.py` (gera
`results/*.json` + `results/*.html` e abre o painel sozinho) ou
`tests\deepeval\view_report.py` pra reabrir sem rodar a avaliação de novo.
HTML autocontido, arquivo de verdade em `tests/deepeval/results/`, sem
servidor nem domínio externo.

Todas as interações são logadas em `logs/interactions.jsonl` (JSON Lines), uma linha por interação, com pergunta, contexto recuperado, resposta, métricas e metadados — isso alimenta a suíte DeepEval e a escrita do artigo.

## Fase 2 — Arquitetura de testes

Testes de API (pytest), testes E2E (Playwright/TypeScript) e avaliação de qualidade de
resposta ([DeepEval](https://deepeval.com/docs/introduction), ver seção "Testes" acima e
[tests/deepeval/README.md](tests/deepeval/README.md)) — implementados. Em aberto: ingestão de
corpus de teste via ZIP e geração de dataset sintético assistida por LLM (o dataset atual é
curado manualmente, ver `tests/deepeval/goldens/dataset.json`). Detalhes em `docs/ARCHITECTURE.md`.

## Fase 4 — Estudo de mutação (pesquisa)

O repositório é também o SUT de um estudo de teste de mutação para pipelines RAG,
submetido à special issue **VSI:EQUISA** do *Information and Software Technology*
(*Mutation-Based Adequacy Assessment of Test Suites for Retrieval-Augmented
Assistants*). O estudo mutaciona as cinco camadas do pipeline — corpus, chunking,
índice, recuperação e prompt — com 18 operadores, e mede quanto do veredito de
adequação depende do oráculo escolhido.

Tudo vive em [tests/mutation/](tests/mutation/), com venv próprio e índice
próprio: **rodar a campanha não toca `data/vector_store` nem muda o
comportamento do assistente**. Os parâmetros que os operadores mutacionam foram
adicionados ao pipeline com defaults que preservam o comportamento anterior (o
system prompt v1 sai byte a byte idêntico).

Passo a passo completo — preparo, calibração, campanha, análise e o que ainda
depende de decisão humana — em **[tests/mutation/README.md](tests/mutation/README.md)**.

## Fase 3 — Human-in-the-loop (proposta)

Camada de revisão humana e relatório consolidado comparando notas do DeepEval com avaliação
humana — a implementar mediante confirmação.

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
tests/mutation/           Estudo de mutação para RAG (Fase 4) — operadores, oráculos, campanha e análise
logs/                     interactions.jsonl (log estruturado, não versionado)
docs/ARCHITECTURE.md      Decisões técnicas e racional
```
```