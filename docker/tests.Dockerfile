# O container dos testes (serviço `tests` do docker-compose.yml, perfil "tests";
# quem chama é scripts/docker_tests.bat ou .sh). Tudo o que os testes pedem na
# máquina vem aqui dentro: Python, os 5 venvs, Node, o Playwright e o Chromium.
# Ele testa o RAG como caixa-preta, pela API: não monta os volumes do Chroma.
#
# A base é a imagem oficial do Playwright, com o Chromium e as dependências de
# sistema já instalados. A versão TEM de ser a do @playwright/test do
# frontend/tests/package-lock.json (1.61.1): com outra, o Playwright procura um
# navegador que não está na imagem.
#
# O código é copiado, não montado: assim os venvs e o node_modules ficam onde o
# tests/run_all.py espera, sem um bind mount escondê-los e sem colidir com os
# venvs do Windows na pasta. As camadas dos requirements vêm antes do código:
# mudar um teste não reinstala nada. O .env nunca entra (ver
# tests.Dockerfile.dockerignore); a chave da TypeSafe chega por variável no run.
#
# Sem a linha `# syntax=`: ela baixaria a cada build um frontend com tag
# flutuante (docker/dockerfile:1). O frontend embutido do Docker já aceita o
# RUN --mount=type=cache usado abaixo.
FROM mcr.microsoft.com/playwright:v1.61.1-noble

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    RAGAS_DO_NOT_TRACK=true \
    RAG_TESTS_IN_DOCKER=1

# python-is-python3: o webServer do playwright.config.ts chama `python`.
# socat: encaminha o localhost do container para os serviços (tests-entrypoint.sh).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python-is-python3 socat \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Os 5 venvs, nos caminhos que tests/run_all.py espera (fora do Windows, <venv>/bin/python). Separados como na
# máquina: o pydantic de cada um não se resolve junto com o dos outros (docs/ARCHITECTURE.md).
COPY backend/requirements.txt backend/requirements-dev.txt backend/requirements-eval.txt backend/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements-dev.txt
COPY tests/reports/requirements.txt tests/reports/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv tests/reports/.venv && tests/reports/.venv/bin/pip install -r tests/reports/requirements.txt
COPY tests/ragas/requirements.txt tests/ragas/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv tests/ragas/.venv && tests/ragas/.venv/bin/pip install -r tests/ragas/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv tests/deepeval/.venv && tests/deepeval/.venv/bin/pip install -r backend/requirements-eval.txt
COPY tests/jeveval/requirements.txt tests/jeveval/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv tests/jeveval/.venv && tests/jeveval/.venv/bin/pip install -r tests/jeveval/requirements.txt

# O que as libs baixam ou montam no primeiro uso, feito aqui: o tiktoken baixa a codificação da internet, e o
# app.rag.ingestion a carrega no import (os testes de API, o RAGAS e o DeepEval importam o app); o matplotlib
# monta o cache de fontes na primeira figura do relatório.
RUN backend/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && tests/reports/.venv/bin/python -c "import matplotlib.font_manager"

# Os navegadores vêm da base: sem `playwright install`.
COPY frontend/tests/package.json frontend/tests/package-lock.json frontend/tests/
RUN --mount=type=cache,target=/root/.npm \
    cd frontend/tests && npm ci --no-audit --no-fund

COPY . .
COPY docker/tests-entrypoint.sh /usr/local/bin/tests-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/tests-entrypoint && chmod +x /usr/local/bin/tests-entrypoint

ENTRYPOINT ["tests-entrypoint"]
