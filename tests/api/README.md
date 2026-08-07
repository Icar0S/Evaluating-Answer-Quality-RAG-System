# Testes de API

Suíte pytest sobre a API FastAPI, cobrindo o roteamento por provider (local
Ollama ↔ remoto smartdatatest, ver `../../llm-api-referencia.md`), as métricas
do monitor e o log estruturado de interações.

**São herméticos de propósito**: nenhuma chamada real, nenhuma GPU, nenhum acesso
à rede, nenhuma escrita em `logs/` ou `data/`. É o que permite rodá-los no CI do
GitHub, onde nada disso existe — diferente da suíte E2E, cujos testes de geração
real se auto-pulam quando a API não está no ar.

## Rodar

```powershell
cd backend
.venv\Scripts\pip install -r requirements-dev.txt   # uma vez
cd ..
.\backend\.venv\Scripts\python -m pytest
```

Roda em ~1s. A configuração (`pytest.ini` na raiz) já aponta `pythonpath` para
`backend/`, então não é preciso instalar o pacote nem mexer no `PYTHONPATH`.

## Como o isolamento funciona (`conftest.py`)

| Risco | Como é neutralizado |
|---|---|
| `get_settings()` é `@lru_cache` | `cache_clear()` antes e depois de cada teste |
| O `.env` real da máquina venceria os defaults | Variáveis de ambiente sobrescritas por fixture (têm precedência no pydantic-settings) |
| Provider ativo é estado de módulo | `providers._active_name` resetado a cada teste |
| Chamadas HTTP a Ollama ou à API remota | `httpx.MockTransport` (`LlmBackendMock`, cobre os dois protocolos) instalado em `httpx.Client`/`AsyncClient` por fixture `autouse` |
| ChromaDB | `vector_store.collection_count`/`query` stubados |
| Escrita em `logs/` e `data/` | `LOGS_DIR`/`VECTOR_STORE_DIR` apontados para `tmp_path` |

As URLs de teste usam hosts inexistentes de propósito (`ollama-local-de-teste`):
se algum mock deixar passar uma chamada, o teste falha com erro de conexão em vez
de acertar silenciosamente um serviço real da máquina de quem está rodando.
