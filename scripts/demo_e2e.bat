@echo off
setlocal enabledelayedexpansion

REM Demonstracao E2E ao vivo: sobe o RAG inteiro em containers, com DOIS RAGs
REM (o dos PDFs de data\source_pdfs e o dos Markdown de data\new_papers_marks;
REM mesmo gerador, mesmo juiz, acervos diferentes), roda os testes Playwright
REM com o navegador VISIVEL nos dois, para uma turma acompanhar, e gera o
REM relatorio que compara a qualidade dos dois RAGs.
REM
REM Uso:
REM   scripts\demo_e2e.bat           testes ao vivo nos dois RAGs + relatorio (RAGAS)
REM   scripts\demo_e2e.bat rapido    so os testes ao vivo (sem perguntas de referencia nem relatorio)
REM   scripts\demo_e2e.bat ui        modo UI do Playwright (lista, linha do tempo, DOM)
REM   scripts\demo_e2e.bat subir     so sobe os containers (para abrir o chat a mao)
REM   scripts\demo_e2e.bat parar     derruba os containers (indices e modelos ficam)
REM
REM Variaveis opcionais: DEMO_GENERATION_MODEL (padrao gemma3:4b),
REM DEMO_JUDGE_MODEL (padrao qwen3:8b), DEMO_SLOWMO (ms entre acoes, padrao 400),
REM DEMO_GOLDENS (so as N primeiras perguntas de referencia de cada acervo),
REM OLLAMA_MODELS_DIR (pasta de modelos do Ollama a reaproveitar).

cd /d "%~dp0.."
chcp 65001 >nul
set "MODE=%~1"

docker version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O Docker nao esta no ar. Abra o Docker Desktop e tente de novo.
    goto :erro
)

set "COMPOSE=docker compose -f docker-compose.yml"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "COMPOSE=!COMPOSE! -f docker-compose.gpu.yml"
    echo GPU NVIDIA detectada: o Ollama do container vai usa-la.
) else (
    echo Sem GPU NVIDIA: tudo em CPU ^(gerador e juiz ficam bem mais lentos^).
)

if /i "%MODE%"=="parar" (
    !COMPOSE! down
    exit /b !errorlevel!
)

if not defined DEMO_GENERATION_MODEL set "DEMO_GENERATION_MODEL=gemma3:4b"
if not defined DEMO_JUDGE_MODEL set "DEMO_JUDGE_MODEL=qwen3:8b"
if not defined DEMO_EMBEDDING_MODEL set "DEMO_EMBEDDING_MODEL=nomic-embed-text"

REM O relatorio (modo padrao) roda em Python na maquina: o orquestrador so usa
REM a biblioteca padrao, e o RAGAS e o relatorio usam os venvs de tests\.
set "PY="
if /i not "%MODE%"=="rapido" if /i not "%MODE%"=="ui" if /i not "%MODE%"=="subir" (
    if exist "backend\.venv\Scripts\python.exe" set "PY=backend\.venv\Scripts\python.exe"
    if not defined PY (
        where py >nul 2>&1
        if not errorlevel 1 set "PY=py -3"
    )
    if not defined PY (
        where python >nul 2>&1
        if not errorlevel 1 set "PY=python"
    )
    if not defined PY (
        echo [ERRO] Python nao encontrado. O relatorio precisa dele: instale o Python 3.10+
        echo        ou rode "scripts\demo_e2e.bat rapido" para so os testes ao vivo.
        goto :erro
    )
)

REM Modelos: no volume do Docker, que carrega bem mais rapido que uma pasta do
REM Windows montada no container (medido: ~44 s contra ~1 min 50 s para o
REM gemma3:4b). Se a maquina ja tem os modelos no Ollama dela, copia so os tres
REM da demonstracao para o volume (uma vez; nas outras, so confere). O que nao
REM estiver na maquina, o servico "models" do compose baixa.
REM Para montar a pasta da maquina direto (sem copiar), defina OLLAMA_MODELS_DIR.
set "OLLAMA_IMAGE=ollama/ollama:0.35.0"
set "MODELS_VOLUME=rag-test-specialist_ollama-models"
if defined OLLAMA_MODELS_DIR (
    echo Modelos do Ollama: montados de !OLLAMA_MODELS_DIR!
) else (
    set "MODELS_SRC="
    if defined OLLAMA_MODELS if exist "!OLLAMA_MODELS!\manifests" set "MODELS_SRC=!OLLAMA_MODELS!"
    if not defined MODELS_SRC if exist "%USERPROFILE%\.ollama\models\manifests" set "MODELS_SRC=%USERPROFILE%\.ollama\models"
    REM Com as etiquetas do compose, ele reconhece o volume como dele (sem aviso).
    docker volume create --label com.docker.compose.project=rag-test-specialist --label com.docker.compose.volume=ollama-models !MODELS_VOLUME! >nul
    if defined MODELS_SRC (
        echo.
        echo == Copiando os modelos da maquina para o volume do Docker ^(so o que falta^) ==
        docker run --rm --entrypoint sh -v "!MODELS_SRC!:/src:ro" -v !MODELS_VOLUME!:/dst -v "%CD%\docker:/scripts:ro" !OLLAMA_IMAGE! /scripts/seed_models.sh /src /dst !DEMO_EMBEDDING_MODEL! !DEMO_GENERATION_MODEL! !DEMO_JUDGE_MODEL!
        if errorlevel 1 (
            echo [AVISO] A copia falhou; o servico "models" vai baixar o que faltar.
        )
    ) else (
        echo Modelos do Ollama: volume do Docker ^(o primeiro "subir" baixa ~9 GB^).
    )
)

REM As portas sao as do ambiente local: se o start_dev.bat estiver rodando, conflita.
set "RUNNING="
for /f %%c in ('!COMPOSE! ps -q --status running backend 2^>nul') do set "RUNNING=1"
if not defined RUNNING (
    curl.exe -s -o nul --max-time 3 http://localhost:8010/health
    if not errorlevel 1 (
        echo [ERRO] Ja ha um backend na porta 8010 fora do Docker ^(start_dev.bat?^).
        echo        Feche a janela "RAG backend" e rode de novo.
        goto :erro
    )
)

echo.
echo == Subindo os containers ==
echo    Na primeira vez: build do backend, modelos e indexacao dos PDFs e dos Markdown ^(alguns minutos^).
!COMPOSE! up -d --build
if errorlevel 1 (
    echo [ERRO] O docker compose falhou. Veja: !COMPOSE! logs
    goto :erro
)

echo.
echo == Esperando os dois RAGs ficarem prontos ==
set "READY="
for /l %%i in (1,1,180) do (
    if not defined READY (
        curl.exe -s --max-time 5 http://localhost:8010/health 2>nul | findstr /c:"ollama_reachable" >nul
        if not errorlevel 1 (
            set "READY=1"
        ) else (
            ping -n 6 127.0.0.1 >nul
        )
    )
)
if not defined READY (
    echo [ERRO] O backend nao respondeu em 15 min. Veja: !COMPOSE! logs models ingest backend
    goto :erro
)
REM O RAG dos Markdown sobe em paralelo; se nao vier, a demonstracao segue so
REM com o dos PDFs (os testes do md-v1 se auto-pulam).
set "READY_MD="
for /l %%i in (1,1,60) do (
    if not defined READY_MD (
        curl.exe -s --max-time 5 http://localhost:8011/health 2>nul | findstr /c:"ollama_reachable" >nul
        if not errorlevel 1 (
            set "READY_MD=1"
        ) else (
            ping -n 6 127.0.0.1 >nul
        )
    )
)
echo    RAG dos PDFs ^(pdf-v1^):     http://localhost:8010   chat: http://localhost:5510/chat.html
if defined READY_MD (
    echo    RAG dos Markdown ^(md-v1^): http://localhost:8011   chat: http://localhost:5510/chat.html?api=http://localhost:8011
    curl.exe -s --max-time 5 http://localhost:8011/health 2>nul | findstr /r /c:"vector_store_documents.:0[,}]" >nul
    if not errorlevel 1 (
        echo [AVISO] O indice md-v1 esta vazio: faltam os .md em data\new_papers_marks
        echo         ^(o acervo nao e versionado^). Os testes do RAG dos Markdown vao se auto-pular.
    )
) else (
    echo [AVISO] O RAG dos Markdown nao respondeu em 5 min; os testes dele vao se auto-pular.
    echo         Veja: !COMPOSE! logs ingest-md backend-md
)

REM Carrega o gerador na memoria antes da plateia ver o primeiro teste.
curl.exe -s -o nul --max-time 300 http://localhost:11435/api/generate -d "{\"model\":\"!DEMO_GENERATION_MODEL!\",\"keep_alive\":\"30m\"}"

if /i "%MODE%"=="subir" (
    echo.
    echo Containers no ar. Abra os enderecos de chat acima.
    echo Para parar: scripts\demo_e2e.bat parar
    exit /b 0
)

if not exist "frontend\tests\node_modules" (
    echo.
    echo == Instalando o Playwright ^(uma vez^) ==
    pushd frontend\tests
    call npm install
    call npx playwright install chromium
    popd
)

set "RAG_JUDGE_MODEL=!DEMO_JUDGE_MODEL!"
set "RAG_JUDGE_OLLAMA_URL=http://localhost:11435"

echo.
echo == Rodando os testes E2E ao vivo nos dois RAGs ==
echo    responde: !DEMO_GENERATION_MODEL!   julga: !DEMO_JUDGE_MODEL!
if /i "%MODE%"=="ui" (
    pushd frontend\tests
    call npm run demo:ui
    set "CODE=!errorlevel!"
    popd
) else if /i "%MODE%"=="rapido" (
    pushd frontend\tests
    call npm run demo
    set "CODE=!errorlevel!"
    popd
) else (
    echo    Depois dos testes, o RAGAS julga as respostas dos dois RAGs ^(~1 min por
    echo    resposta com GPU^) e o relatorio que compara os dois abre sozinho.
    !PY! tests\demo_e2e.py --abrir
    set "CODE=!errorlevel!"
)

echo.
echo Video e trace de cada teste: cd frontend\tests ^&^& npm run demo:report
echo Os containers continuam no ar ^(enderecos de chat acima^).
echo Para parar: scripts\demo_e2e.bat parar
exit /b !CODE!

:erro
echo.
pause
exit /b 1
