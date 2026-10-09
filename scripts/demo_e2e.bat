@echo off
setlocal enabledelayedexpansion

REM Demonstracao E2E ao vivo: sobe o RAG inteiro em containers (Ollama com
REM embeddings, gerador e juiz; ingestao; backend; frontend) e roda os testes
REM Playwright com o navegador VISIVEL, para uma turma acompanhar.
REM
REM Uso:
REM   scripts\demo_e2e.bat           navegador visivel, acoes em camera lenta
REM   scripts\demo_e2e.bat ui        modo UI do Playwright (lista, linha do tempo, DOM)
REM   scripts\demo_e2e.bat subir     so sobe os containers (para abrir o chat a mao)
REM   scripts\demo_e2e.bat parar     derruba os containers (indice e modelos ficam)
REM
REM Variaveis opcionais: DEMO_GENERATION_MODEL (padrao gemma3:4b),
REM DEMO_JUDGE_MODEL (padrao qwen3:8b), DEMO_SLOWMO (ms entre acoes, padrao 400),
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
echo    Na primeira vez: build do backend, modelos e indexacao dos PDFs ^(alguns minutos^).
!COMPOSE! up -d --build
if errorlevel 1 (
    echo [ERRO] O docker compose falhou. Veja: !COMPOSE! logs
    goto :erro
)

echo.
echo == Esperando o backend ficar pronto ==
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
echo    Backend: http://localhost:8010   Chat: http://localhost:5510/chat.html

REM Carrega o gerador na memoria antes da plateia ver o primeiro teste.
curl.exe -s -o nul --max-time 300 http://localhost:11435/api/generate -d "{\"model\":\"!DEMO_GENERATION_MODEL!\",\"keep_alive\":\"30m\"}"

if /i "%MODE%"=="subir" (
    echo.
    echo Containers no ar. Abra http://localhost:5510/chat.html
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
echo == Rodando os testes E2E ao vivo ==
echo    responde: !DEMO_GENERATION_MODEL!   julga: !DEMO_JUDGE_MODEL!
pushd frontend\tests
if /i "%MODE%"=="ui" (
    call npm run demo:ui
) else (
    call npm run demo
)
set "CODE=!errorlevel!"
popd

echo.
echo Relatorio com video e trace de cada teste: cd frontend\tests ^&^& npm run demo:report
echo Os containers continuam no ar ^(chat em http://localhost:5510/chat.html^).
echo Para parar: scripts\demo_e2e.bat parar
exit /b !CODE!

:erro
echo.
pause
exit /b 1
