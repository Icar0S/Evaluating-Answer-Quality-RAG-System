# Backend FastAPI do RAG (também usado pelo serviço `ingest` do docker-compose).
#
# Só o código do backend e o script de ingestão entram na imagem. PDFs, índice
# vetorial e logs vêm de volumes (docker-compose.yml), e a configuração vem de
# variáveis de ambiente: o .env da raiz NÃO é copiado (tem a chave da API remota).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# O tiktoken baixa a codificação da internet no primeiro uso, e o backend a
# carrega no import (app/rag/ingestion.py). Baixada aqui, a imagem sobe offline.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY backend/app backend/app
COPY scripts/ingest_documents.py scripts/ingest_documents.py

WORKDIR /app/backend
EXPOSE 8010
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
