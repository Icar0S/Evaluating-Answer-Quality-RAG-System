# LLM API — Mac mini — Referência

> **Gerado automaticamente** do OpenAPI em 2026-08-05 21:37 UTC.
> Não edite à mão — rode `~/srv/bin/gerar-doc-api.py` após mudar rotas ou schemas.
> Versão da API: `0.1.0` · OpenAPI `3.1.0`

**Swagger interativo:** <https://llm.smartdatatest.com/docs>

---

API sobre modelos de linguagem rodando **localmente** num Mac mini M4.

### Como usar

1. `GET /v1/models` — veja os modelos e quais estão disponíveis agora
2. `POST /v1/corpora` — registre uma pasta sua para consulta (opcional)
3. `POST /v1/chat` — pergunte, com ou sem `corpus_id`

### Por que existe fila

O servidor tem 16 GB de memória unificada e divide a máquina com uma
aplicação em produção. Uma inferência roda por vez.

- Fila cheia devolve **429** com `Retry-After`
- Servidor sob pressão devolve **503**
- Os headers `X-Queue-Position` e `X-Estimated-Wait` acompanham cada resposta

Isso é deliberado: recusar rápido é melhor que aceitar e derrubar o servidor.

### Sobre seus documentos

O RAG lê a fonte que **você** indica — bucket S3, WebDAV, ou uma pasta servida
pela sua própria máquina via Tailscale. No modo `ephemeral` (padrão) o índice
vive apenas em memória e é descartado por inatividade: **nenhum byte do seu
conteúdo é gravado no disco do servidor**.

---

## Índice de rotas

| Método | Rota | O que faz |
|---|---|---|
| `POST` | [`/v1/chat`](#post-v1-chat) | Gera uma resposta de texto |
| `POST` | [`/v1/corpora`](#post-v1-corpora) | Registra uma fonte de documentos e indexa |
| `GET` | [`/v1/corpora`](#get-v1-corpora) | Lista seus corpora |
| `GET` | [`/v1/corpora/{corpus_id}`](#get-v1-corpora-corpus_id) | Estado de um corpus |
| `DELETE` | [`/v1/corpora/{corpus_id}`](#delete-v1-corpora-corpus_id) | Apaga o índice e as credenciais |
| `POST` | [`/v1/corpora/{corpus_id}/refresh`](#post-v1-corpora-corpus_id-refresh) | Relê a fonte e reconstrói o índice |
| `GET` | [`/v1/health`](#get-v1-health) | Liveness — responde 200 enquanto o processo estiver de pé |
| `GET` | [`/v1/models`](#get-v1-models) | Lista os modelos e o que está disponível agora |
| `GET` | [`/v1/ready`](#get-v1-ready) | Readiness — 200 se está saudável, 503 se algo está degradado |
| `GET` | [`/v1/status`](#get-v1-status) | Estado de recursos, fila e pressão |
| `POST` | [`/v1/vision`](#post-v1-vision) | Faz uma pergunta sobre uma imagem |

---

## Rotas

### `POST` /v1/chat
<a id="post-v1-chat"></a>

**Gera uma resposta de texto**

Enfileira a requisição e devolve quando o modelo terminar.

**Sobre `corpus_id`:** informe-o para que a resposta consulte os seus documentos em vez de depender do que o modelo memorizou. É o que faz um modelo de 4B responder de forma útil sobre assuntos específicos. As fontes usadas voltam em `sources` — confira-as.

**Sobre `allow_fallback`:** com o servidor em `yellow`, um pedido ao tier pesado seria recusado. Com fallback ligado, ele é atendido por um modelo mais leve e a resposta indica isso em `fallback_applied`.

**Corpo da requisição**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `messages` | lista de Message | ✔ | Histórico da conversa, em ordem cronológica. |
| `model` | texto (opcional) |  | Modelo a usar. Omitido, usa o default (`qwen3:4b`). Consulte /v1/models para as opções e o estado de cada uma. |
| `corpus_id` | texto (opcional) |  | Corpus registrado em /v1/corpora. Quando presente, os trechos mais relevantes dos seus documentos são recuperados e injetados como contexto, e a resposta traz as fontes usadas.  É isto que faz um modelo leve responder bem: sem contexto ele chuta, com contexto ele consulta. |
| `temperature` | número |  |  *(padrão: `0.7`)* |
| `max_tokens` | inteiro (opcional) |  | Teto de tokens gerados. Omitido, o servidor escolhe um valor adequado ao modelo.  **Atenção com modelos de raciocínio** (veja `thinking` em /v1/models): eles gastam ~1000 tokens pensando *antes* de escrever a resposta. Um teto baixo faz o orçamento acabar durante o raciocínio e a resposta sair vazia. Valores abaixo do mínimo do modelo são elevados automaticamente. |
| `web_search` | booleano |  | Busca na internet antes de responder e injeta os resultados como contexto, com as fontes em `sources`.  Devolve **501** se o servidor não tiver provedor configurado. Resultados de busca são conteúdo de terceiros — o modelo é instruído a tratá-los como indício, e você deve conferir as fontes. *(padrão: `False`)* |
| `allow_fallback` | booleano |  | Se o modelo pedido não for admitido pelo governador, tenta um mais leve em vez de devolver 503. Desligue para exigir exatamente o modelo pedido. *(padrão: `True`)* |

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `content` | texto | ✔ | — |
| `thinking` | texto (opcional) |  | Raciocínio interno, quando o modelo é do tipo que pensa antes de responder. Devolvido separado de `content` para você poder auditar como o modelo chegou à conclusão — útil justamente em modelos leves, que erram mais. |
| `model` | texto | ✔ | Modelo que de fato respondeu. |
| `requested_model` | texto | ✔ | Modelo pedido — difere se houve fallback. |
| `fallback_applied` | booleano | ✔ | — |
| `max_tokens_adjusted` | booleano |  | Verdadeiro quando o `max_tokens` pedido foi elevado ao mínimo que o modelo precisa para conseguir responder. *(padrão: `False`)* |
| `sources` | lista de Source |  | Trechos usados quando `corpus_id` foi informado. Confira-os: modelos leves acertam mais com contexto, mas não são infalíveis. |
| `tokens_in` | inteiro (opcional) |  | — |
| `tokens_out` | inteiro (opcional) |  | — |
| `duration_s` | número | ✔ | — |
| `queue_wait_s` | número | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `404` | Modelo desconhecido ou ainda não baixado. |
| `422` | Validation Error |
| `429` | Fila cheia. Respeite o header `Retry-After`. |
| `503` | Servidor sob pressão ou espera na fila excedida. |

---

### `POST` /v1/corpora
<a id="post-v1-corpora"></a>

**Registra uma fonte de documentos e indexa**

Lê a fonte que você indicar, extrai o texto, fatia e monta um índice **em memória**.

**O que fica no servidor:** nada em disco. Trechos e vetores vivem em RAM e são descartados após 30 minutos sem uso, ou imediatamente no `DELETE`. As credenciais do conector ficam só em memória, nunca em log.

**O que não fica sob nosso controle:** o conteúdo passa pela RAM para poder ser processado, e RAM pode ir para o swap — que é cifrado no macOS, mas ainda é memória em disco. Se você tem requisito regulatório estrito, considere isso antes de indexar material sensível.

Formatos: `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.html`, `.rst`, `.json`.

**Corpo da requisição**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `kind` | `local` · `tailscale` · `s3` · `webdav` | ✔ | De onde os documentos são lidos. **A fonte permanece com você** — o servidor lê sob demanda e não guarda cópia.  - `tailscale`: você serve a pasta read-only na sua máquina e entra no tailnet. O arquivo nunca sai do seu computador para ficar aqui. - `s3`: seu bucket (R2, S3, Backblaze, MinIO) com credencial read-only. - `webdav`: Nextcloud, ownCloud, NAS. - `local`: pasta dentro do sandbox do servidor. Só para o dono da máquina. |
| `config` | objeto | ✔ | Campos por conector:  - `local`: `path` (relativo à sua raiz; caminhos fora dela são recusados) - `tailscale`: `url` (ex.: `http://100.x.y.z:8000/docs`) - `webdav`: `url`, `username`, `password` - `s3`: `bucket`, `prefix`, `endpoint_url`, `access_key`, `secret_key`, `region`  Credenciais ficam apenas em memória, nunca em disco nem em log, e somem no DELETE. |

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `id` | texto | ✔ | — |
| `kind` | texto | ✔ | — |
| `status` | `pending` · `indexing` · `ready` · `error` | ✔ | — |
| `documents` | inteiro | ✔ | — |
| `chunks` | inteiro | ✔ | — |
| `memory_mb` | inteiro | ✔ | RAM ocupada pelo índice. |
| `idle_seconds` | inteiro | ✔ | — |
| `ttl_seconds` | inteiro | ✔ | Inatividade que descarta o índice. Depois disso, nada seu permanece no servidor. |
| `error` | texto (opcional) |  | — |

**Erros**

| Código | Significado |
|---|---|
| `400` | Fonte inalcançável, vazia ou configuração inválida. |
| `403` | Endereço bloqueado (rede interna do servidor). |
| `422` | Validation Error |

---

### `GET` /v1/corpora
<a id="get-v1-corpora"></a>

**Lista seus corpora**

Só os seus. O isolamento é aplicado em cada acesso, não apenas na borda.

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `corpora` | lista de CorpusInfo | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `GET` /v1/corpora/{corpus_id}
<a id="get-v1-corpora-corpus_id"></a>

**Estado de um corpus**

`idle_seconds` versus `ttl_seconds` diz quanto falta para o índice ser descartado por inatividade.

**Parâmetros**

| Nome | Em | Obrig. | Descrição |
|---|---|:--:|---|
| `corpus_id` | path | ✔ | — |

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `id` | texto | ✔ | — |
| `kind` | texto | ✔ | — |
| `status` | `pending` · `indexing` · `ready` · `error` | ✔ | — |
| `documents` | inteiro | ✔ | — |
| `chunks` | inteiro | ✔ | — |
| `memory_mb` | inteiro | ✔ | RAM ocupada pelo índice. |
| `idle_seconds` | inteiro | ✔ | — |
| `ttl_seconds` | inteiro | ✔ | Inatividade que descarta o índice. Depois disso, nada seu permanece no servidor. |
| `error` | texto (opcional) |  | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `DELETE` /v1/corpora/{corpus_id}
<a id="delete-v1-corpora-corpus_id"></a>

**Apaga o índice e as credenciais**

Remove trechos, vetores e a configuração do conector — incluindo qualquer credencial que você tenha informado.

**Parâmetros**

| Nome | Em | Obrig. | Descrição |
|---|---|:--:|---|
| `corpus_id` | path | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `POST` /v1/corpora/{corpus_id}/refresh
<a id="post-v1-corpora-corpus_id-refresh"></a>

**Relê a fonte e reconstrói o índice**

Use depois de alterar os documentos na origem. O índice anterior é descartado antes da releitura.

**Parâmetros**

| Nome | Em | Obrig. | Descrição |
|---|---|:--:|---|
| `corpus_id` | path | ✔ | — |

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `id` | texto | ✔ | — |
| `kind` | texto | ✔ | — |
| `status` | `pending` · `indexing` · `ready` · `error` | ✔ | — |
| `documents` | inteiro | ✔ | — |
| `chunks` | inteiro | ✔ | — |
| `memory_mb` | inteiro | ✔ | RAM ocupada pelo índice. |
| `idle_seconds` | inteiro | ✔ | — |
| `ttl_seconds` | inteiro | ✔ | Inatividade que descarta o índice. Depois disso, nada seu permanece no servidor. |
| `error` | texto (opcional) |  | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `GET` /v1/health
<a id="get-v1-health"></a>

**Liveness — responde 200 enquanto o processo estiver de pé**

Deliberadamente burro: não consulta Ollama, fila nem governador. Serve para o supervisor saber se o processo morreu. Para saber se o servidor consegue *atender*, use /v1/status.

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `status` | texto | ✔ | — |

---

### `GET` /v1/models
<a id="get-v1-models"></a>

**Lista os modelos e o que está disponível agora**

`available` reflete o estado do servidor **neste momento**, não uma capacidade fixa. Um modelo do tier `heavy` aparece indisponível sempre que o governador não estiver em `green` — por pressão de memória, temperatura ou porque a aplicação em produção que divide esta máquina começou a responder devagar.

`loaded` indica quem está residente: uma requisição a um modelo já carregado começa a responder bem mais rápido, porque pula o carregamento dos pesos.

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `models` | lista de ModelInfo | ✔ | — |
| `default` | texto | ✔ | — |
| `default_vision` | texto | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `GET` /v1/ready
<a id="get-v1-ready"></a>

**Readiness — 200 se está saudável, 503 se algo está degradado**

**Sem autenticação, de propósito**: é o alvo do monitor externo.

Diferente de `/v1/health`, que só prova que o processo está vivo, este reflete duas coisas:

1. O estado do governador de recursos
2. O último resultado do `healthcheck.sh` — disco, backup, containers

É o que permite um único monitor externo cobrir a máquina inteira. Um monitor que só olhasse `/v1/health` diria 'está no ar' com o disco a 95% e o backup parado há três dias.

O corpo é deliberadamente pobre — `ready` ou `degraded`, sem números. Sendo público, não deve revelar memória, swap nem estado da infraestrutura; para isso existe `/v1/status`, que exige chave.

**Erros**

| Código | Significado |
|---|---|
| `503` | Degradado. Consulte /v1/status com chave para o detalhe. |

---

### `GET` /v1/status
<a id="get-v1-status"></a>

**Estado de recursos, fila e pressão**

Antes de mandar um lote de requisições, olhe aqui: `state` diz se o servidor aceita carga, e `queue.estimated_wait_s` diz quanto você esperaria.

`platform_ok` acompanha a aplicação em produção que divide os mesmos 16 GB. Quando ela degrada, o LLM recua sozinho — a inferência é o que cede, não a plataforma.

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `state` | `green` · `yellow` · `red` | ✔ | `green`: todos os tiers liberados. `yellow`: tiers pesados bloqueados. `red`: inferência recusada com 503 até a pressão passar. |
| `reasons` | lista de texto | ✔ | Por que o estado não é `green`. |
| `free_mb` | inteiro | ✔ | — |
| `swap_mb` | inteiro | ✔ | — |
| `thermal_pressure` | texto (opcional) | ✔ | — |
| `platform_ok` | booleano | ✔ | Saúde da aplicação em produção que divide a máquina. Quando degrada, o LLM recua antes de o usuário dela perceber. |
| `platform_latency_s` | número (opcional) | ✔ | — |
| `loaded_model` | texto (opcional) | ✔ | — |
| `corpus_mb` | inteiro | ✔ | — |
| `queue` | QueueStatus | ✔ | — |
| `ollama_version` | texto | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `422` | Validation Error |

---

### `POST` /v1/vision
<a id="post-v1-vision"></a>

**Faz uma pergunta sobre uma imagem**

Envie a imagem e a pergunta; o modelo descreve, extrai texto ou analisa o conteúdo.

A imagem é processada **em memória e descartada** — mesma política do RAG, nada é gravado em disco.

Formatos: JPEG, PNG, GIF, WebP. Limite de 12 MB. O tipo é verificado pelos bytes do arquivo, não pela extensão.

Com o servidor fora de `green`, o modelo de visão pesado dá lugar ao `moondream`, mais leve — a resposta indica isso em `fallback_applied`.

**Corpo da requisição**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `image` | texto | ✔ | A imagem. |
| `prompt` | texto |  | O que você quer saber sobre a imagem. *(padrão: `Descreva esta imagem em detalhes.`)* |
| `model` | texto (opcional) |  | Modelo de visão. Omitido, usa o padrão. |

**Resposta**

| Campo | Tipo | Obrig. | Descrição |
|---|---|:--:|---|
| `content` | texto | ✔ | — |
| `model` | texto | ✔ | — |
| `requested_model` | texto | ✔ | — |
| `fallback_applied` | booleano | ✔ | — |
| `duration_s` | número | ✔ | — |

**Erros**

| Código | Significado |
|---|---|
| `400` | Arquivo não é uma imagem suportada, ou excede o limite. |
| `404` | Modelo de visão não baixado neste servidor. |
| `422` | Validation Error |
| `503` | Servidor sob pressão. |

---
