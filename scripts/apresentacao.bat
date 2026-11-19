@echo off
setlocal enabledelayedexpansion

REM A apresentacao gravada: um comando sobe a aplicacao, roda os testes locais, o E2E
REM e os tres avaliadores de qualidade (RAGAS, DeepEval e JevEval), numa pagina que
REM conta a rotina (o palco), e grava essa pagina num MP4
REM (tests\apresentacao.py; README, "Apresentacao gravada").
REM
REM Uso:
REM   scripts\apresentacao.bat                               tudo, 1 pergunta de referencia por acervo
REM   scripts\apresentacao.bat --goldens 2                   2 perguntas por acervo (video mais longo)
REM   scripts\apresentacao.bat --avaliadores ragas jeveval   sem o DeepEval, o mais lento
REM   scripts\apresentacao.bat --assistir                    abre o palco no navegador, para acompanhar
REM   scripts\apresentacao.bat --sem-video                   ensaio, sem gravar
REM   scripts\apresentacao.bat --abrir                       abre o video no fim
REM   ... --sem-pausa                                        nao espera uma tecla no fim
REM
REM So a pagina do palco e gravada, num navegador sem janela; o E2E tambem roda sem
REM janela e aparece dentro do palco. Nada da sua tela entra no video, e o computador
REM fica livre durante a rotina.

cd /d "%~dp0.."
chcp 65001 >nul

REM Os argumentos: --sem-pausa e daqui; o resto vai para o tests\apresentacao.py.
set "ARGS="
set "NOPAUSE="
set "SEMVIDEO="
:args
if "%~1"=="" goto :args_fim
if /i "%~1"=="--sem-pausa" (
    set "NOPAUSE=1"
) else (
    if /i "%~1"=="--sem-video" set "SEMVIDEO=1"
    set "ARGS=!ARGS! %1"
)
shift
goto :args
:args_fim

docker version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O Docker nao esta no ar. Abra o Docker Desktop e tente de novo.
    goto :erro
)

if not defined SEMVIDEO (
    where ffmpeg >nul 2>&1
    if errorlevel 1 (
        echo [ERRO] ffmpeg nao encontrado: e ele que monta o video.
        echo        Instale com: winget install Gyan.FFmpeg   e abra um terminal novo.
        echo        Para so ensaiar, sem gravar: scripts\apresentacao.bat --sem-video
        goto :erro
    )
)

REM As variaveis que o docker compose tambem le do .env (DEMO_*, OLLAMA_SEED_DIR,
REM OLLAMA_MODELS_DIR, COMPOSE_*), quando nao vierem do ambiente: no compose o
REM ambiente ganha do .env, e sem isto os padroes abaixo passariam por cima dele.
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /c:"DEMO_" /c:"OLLAMA_SEED_DIR=" /c:"OLLAMA_MODELS_DIR=" /c:"COMPOSE_FILE=" /c:"COMPOSE_PATH_SEPARATOR=" ".env"`) do (
        if not defined %%a set "%%a=%%b"
    )
)
if not defined DEMO_GENERATION_MODEL set "DEMO_GENERATION_MODEL=gemma3:4b"
if not defined DEMO_JUDGE_MODEL set "DEMO_JUDGE_MODEL=qwen3:8b"
if not defined DEMO_EMBEDDING_MODEL set "DEMO_EMBEDDING_MODEL=nomic-embed-text"
if not defined DEMO_PDF_INDEX set "DEMO_PDF_INDEX=pdf-v1"
if not defined DEMO_MD_INDEX set "DEMO_MD_INDEX=md-v1"

REM GPU: o tests\apresentacao.py roda "docker compose up" sem -f, entao a GPU entra
REM pelo COMPOSE_FILE (a mesma regra do demo_e2e.bat: com nvidia-smi, junta o
REM docker-compose.gpu.yml).
if not defined COMPOSE_FILE (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "COMPOSE_PATH_SEPARATOR=:"
        set "COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml"
        echo GPU NVIDIA detectada: o Ollama do container vai usa-la.
    ) else (
        echo Sem GPU NVIDIA: tudo em CPU ^(gerador e juiz ficam bem mais lentos^).
    )
)

REM Modelos: a mesma deteccao do demo_e2e.bat. Com a pasta do Ollama da maquina, o
REM servico models-seed copia os modelos para o volume em vez de baixar ~9 GB.
if not defined OLLAMA_MODELS_DIR if not defined OLLAMA_SEED_DIR (
    if defined OLLAMA_MODELS if exist "!OLLAMA_MODELS!\manifests" set "OLLAMA_SEED_DIR=!OLLAMA_MODELS!"
    if not defined OLLAMA_SEED_DIR if exist "%USERPROFILE%\.ollama\models\manifests" set "OLLAMA_SEED_DIR=%USERPROFILE%\.ollama\models"
)
if defined OLLAMA_SEED_DIR set "OLLAMA_SEED_DIR=!OLLAMA_SEED_DIR:\=/!"

REM As portas sao as do ambiente local: se o start_dev.bat estiver rodando, conflita.
set "RUNNING="
for /f %%c in ('docker compose ps -q --status running backend 2^>nul') do set "RUNNING=1"
if not defined RUNNING (
    curl.exe -s -o nul --max-time 3 http://localhost:8010/health
    if not errorlevel 1 (
        echo [ERRO] Ja ha um backend na porta 8010 fora do Docker ^(start_dev.bat?^).
        echo        Feche a janela "RAG backend" e rode de novo.
        goto :erro
    )
)

REM O orquestrador so usa a biblioteca padrao; os testes usam os venvs de cada suite.
set "PY="
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
    echo [ERRO] Python nao encontrado. Instale o Python 3.10+ ^(README, "Inicio rapido"^).
    goto :erro
)

if not exist "frontend\tests\node_modules" (
    echo.
    echo == Instalando o Playwright ^(uma vez^) ==
    pushd frontend\tests
    call npm install
    call npx playwright install chromium
    popd
)

echo.
echo == Apresentacao gravada ==
echo    Grava so a pagina do palco, sem janela: pode usar o computador. Acompanhar: --assistir
echo    Responde: !DEMO_GENERATION_MODEL!   julga: !DEMO_JUDGE_MODEL!
!PY! tests\apresentacao.py !ARGS!
set "CODE=!errorlevel!"

echo.
if "!CODE!"=="0" (
    echo Tudo certo. O video e o relatorio estao na pasta indicada acima.
) else (
    echo Alguma etapa falhou ou o video nao foi gerado - veja o resumo acima.
)
if not defined NOPAUSE pause
exit /b !CODE!

:erro
echo.
if not defined NOPAUSE pause
exit /b 1
