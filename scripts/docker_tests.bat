@echo off
setlocal enabledelayedexpansion

REM Os testes do RAG num container, contra a stack do docker compose (README,
REM "Inicio rapido com Docker"). So Docker na maquina: nada de Python, Node ou venvs.
REM
REM Uso:
REM   scripts\docker_tests.bat                      todas as suites, contra o RAG no ar (como o run_all_tests.bat)
REM   scripts\docker_tests.bat --rapido             so os hermeticos e o E2E (minutos)
REM   scripts\docker_tests.bat --limite 2           qualidade em 2 goldens (fumaca)
REM   scripts\docker_tests.bat --pular e2e deepeval --abrir
REM   scripts\docker_tests.bat demo [--goldens N]   E2E nos dois RAGs, RAGAS e o relatorio que compara os dois
REM   scripts\docker_tests.bat demo-rapido          so os testes ao vivo da demonstracao
REM   scripts\docker_tests.bat hermeticos           API, relatorio e JevEval com o Jev falso (sem a stack)
REM   scripts\docker_tests.bat relatorio-e2e        o relatorio HTML do Playwright em http://localhost:9323
REM   scripts\docker_tests.bat reindexar            reindexa os dois RAGs (depois de trocar os documentos)
REM   scripts\docker_tests.bat shell                bash no container, para depurar
REM   ... --sem-pausa                               nao espera uma tecla no fim
REM
REM O script e fino: escolhe os arquivos do compose, decide se mexe na stack e
REM chama o container (docker/tests-entrypoint.sh), que faz o resto. Relatorio e
REM resultados ficam nas pastas de sempre (tests\reports\out\<data>\ e
REM tests\*\results\). --abrir e tratado aqui: abre o PDF do relatorio novo.

cd /d "%~dp0.."
chcp 65001 >nul

docker version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O Docker nao esta no ar. Abra o Docker Desktop e tente de novo.
    set "CODE=1"
    goto :fim
)

REM Os mesmos arquivos do "docker compose up": sem isso, um run que sobe as
REM dependencias recriaria o ollama sem a GPU. COMPOSE_FILE no ambiente ou no
REM .env vale como esta; senao, a regra do demo_e2e.bat: com nvidia-smi, junta o
REM docker-compose.gpu.yml.
set "COMPOSE=docker compose"
set "FROM_ENV="
if defined COMPOSE_FILE set "FROM_ENV=1"
if not defined FROM_ENV if exist ".env" (
    findstr /b /c:"COMPOSE_FILE=" ".env" >nul 2>&1
    if not errorlevel 1 set "FROM_ENV=1"
)
if not defined FROM_ENV (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "COMPOSE=docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
        echo GPU NVIDIA detectada: o Ollama do container vai usa-la.
    )
)

REM O modo (primeiro argumento) e as opcoes. --abrir e --sem-pausa sao daqui; o resto vai para o container.
set "MODE="
for %%m in (demo demo-rapido hermeticos relatorio-e2e reindexar shell) do if /i "%~1"=="%%m" set "MODE=%%m"
if defined MODE shift
set "ARGS="
set "OPEN="
set "NOPAUSE="
:args
if "%~1"=="" goto :args_fim
if /i "%~1"=="--abrir" (
    set "OPEN=1"
) else if /i "%~1"=="--sem-pausa" (
    set "NOPAUSE=1"
) else (
    set "ARGS=!ARGS! %1"
)
shift
goto :args
:args_fim

if /i "!MODE!"=="reindexar" (
    REM Os backends estao com o banco aberto: param durante a ingestao e voltam no fim.
    echo == Reindexando os dois RAGs ^(os backends param durante a ingestao^) ==
    !COMPOSE! stop backend backend-md
    !COMPOSE! run --rm ingest python scripts/ingest_documents.py
    set "CODE=!errorlevel!"
    if "!CODE!"=="0" (
        !COMPOSE! run --rm ingest-md python scripts/ingest_documents.py
        set "CODE=!errorlevel!"
    )
    !COMPOSE! up -d
    goto :fim
)

REM Decide se mexe na stack: no ar, --no-deps (nao toca em nada); fora do ar, o
REM depends_on a sobe. Os modos que nao precisam dela nunca a sobem.
set "NODEPS="
if /i "!MODE!"=="hermeticos" set "NODEPS=--no-deps"
if /i "!MODE!"=="relatorio-e2e" set "NODEPS=--no-deps"
if /i "!MODE!"=="shell" set "NODEPS=--no-deps"
if not defined NODEPS (
    set "RUNNING="
    for /f %%c in ('!COMPOSE! ps -q --status running backend 2^>nul') do set "RUNNING=1"
    if defined RUNNING (
        set "NODEPS=--no-deps"
    ) else (
        echo A stack nao esta no ar: o docker compose vai subi-la antes dos testes.
        echo Na primeira vez, isso inclui o build, os modelos ^(~9 GB^) e a indexacao: de minutos a meia hora.
        echo Para acompanhar em outro terminal: docker compose logs -f models ingest ingest-md
    )
)

set "PORTS="
if /i "!MODE!"=="relatorio-e2e" (
    set "PORTS=-p 9323:9323"
    echo Relatorio do Playwright em http://localhost:9323 ^(Ctrl+C para fechar^).
)

REM Criadas aqui; as que faltarem, o Docker criaria como root (no Linux; aqui, por simetria).
for %%d in (tests\reports\out tests\ragas\results tests\deepeval\results tests\jeveval\results frontend\tests\playwright-report frontend\tests\test-results) do if not exist "%%d" mkdir "%%d"

REM O fuso da maquina, para as pastas tests\reports\out\<data> e o HISTORICO.md
REM baterem com as execucoes feitas fora do container. Formato POSIX (o glibc o
REM entende sem tzdata): UTC-3 vira HOST3:00.
set "TZ_ARG="
set "TZ_MIN="
for /f %%z in ('powershell -NoProfile -Command "[int][TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes" 2^>nul') do set "TZ_MIN=%%z"
if defined TZ_MIN (
    set /a "TZ_W=-TZ_MIN"
    set "TZ_SIGN="
    if !TZ_W! lss 0 (
        set "TZ_SIGN=-"
        set /a "TZ_W=-TZ_W"
    )
    set /a "TZ_H=TZ_W/60, TZ_M=TZ_W%%60"
    set "TZ_M=0!TZ_M!"
    set "TZ_ARG=-e TZ=HOST!TZ_SIGN!!TZ_H!:!TZ_M:~-2!"
)

REM O relatorio mais novo antes de rodar (as pastas tem a data no nome): --abrir so abre um que surgir agora.
set "BEFORE="
for /f "delims=" %%d in ('dir /b /ad /o:n "tests\reports\out" 2^>nul') do if exist "tests\reports\out\%%d\relatorio.pdf" if /i not "%%d"=="ultimo" set "BEFORE=%%d"

!COMPOSE! --profile tests run --rm --build !NODEPS! !PORTS! !TZ_ARG! tests !MODE! !ARGS!
set "CODE=!errorlevel!"

if defined OPEN (
    set "AFTER="
    for /f "delims=" %%d in ('dir /b /ad /o:n "tests\reports\out" 2^>nul') do if exist "tests\reports\out\%%d\relatorio.pdf" if /i not "%%d"=="ultimo" set "AFTER=%%d"
    if defined AFTER if not "!AFTER!"=="!BEFORE!" (
        echo Abrindo tests\reports\out\!AFTER!\relatorio.pdf
        start "" "%CD%\tests\reports\out\!AFTER!\relatorio.pdf"
        goto :fim
    )
    echo Nenhum relatorio novo para abrir.
)

:fim
echo.
REM O veredito so faz sentido nos modos que testam: o relatorio-e2e e o shell terminam com Ctrl+C ou exit.
set "VERDICT=1"
if /i "!MODE!"=="relatorio-e2e" set "VERDICT="
if /i "!MODE!"=="shell" set "VERDICT="
if defined VERDICT (
    if "!CODE!"=="0" (
        echo Tudo certo.
    ) else (
        echo Alguma suite falhou ou o relatorio nao foi gerado - veja o resumo acima.
    )
)
if not defined NOPAUSE pause
exit /b !CODE!
