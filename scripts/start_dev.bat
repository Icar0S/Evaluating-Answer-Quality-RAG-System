@echo off
setlocal enabledelayedexpansion

REM Sobe o ambiente de desenvolvimento completo e valida que esta tudo no ar:
REM   1. Backend FastAPI  (http://localhost:8000)
REM   2. Providers de geracao (Ollama local + API do servidor)
REM   3. Frontend estatico (http://localhost:5500)
REM   4. Suite de testes E2E (Playwright)
REM
REM Backend e frontend sobem em janelas proprias -- feche-as (ou Ctrl+C) para parar.
REM Re-rodar o script e seguro: servicos que ja estiverem no ar sao reaproveitados.
REM
REM Uso: scripts\start_dev.bat

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"
set "VENV_PY=%PROJECT_ROOT%\backend\.venv\Scripts\python.exe"
set "BACKEND_URL=http://localhost:8000"
set "FRONTEND_URL=http://localhost:5500"
set "TESTS_STATUS=nao executados"
set "REMOTE_STATUS=nao verificado"

echo.
echo =====================================================
echo   RAG Test Specialist -- ambiente de desenvolvimento
echo =====================================================

if not exist "%VENV_PY%" (
    echo.
    echo [ERRO] venv do backend nao encontrada em:
    echo        %VENV_PY%
    echo.
    echo        Crie com:
    echo          cd backend
    echo          python -m venv .venv
    echo          .venv\Scripts\pip install -r requirements.txt
    goto :erro
)

REM ---------------------------------------------------------------- 1/4 backend
echo.
echo == 1/4  Backend ==
curl.exe -s -o nul --max-time 3 "%BACKEND_URL%/health"
if !errorlevel! equ 0 (
    echo    Ja estava no ar em %BACKEND_URL%
) else (
    echo    Subindo em %BACKEND_URL% ...
    start "RAG backend" /d "%PROJECT_ROOT%\backend" cmd /k ""%VENV_PY%" -m uvicorn app.main:app --port 8000"
    set BACKEND_OK=0
    for /l %%i in (1,1,30) do (
        if !BACKEND_OK! equ 0 (
            ping -n 2 127.0.0.1 >nul
            curl.exe -s -o nul --max-time 3 "%BACKEND_URL%/health"
            if !errorlevel! equ 0 set BACKEND_OK=1
        )
    )
    if !BACKEND_OK! equ 0 (
        echo.
        echo    [ERRO] O backend nao respondeu em ~30s.
        echo           Veja o traceback na janela "RAG backend".
        goto :erro
    )
    echo    OK
)

REM -------------------------------------------------------------- 2/4 providers
echo.
echo == 2/4  Providers de geracao ==
curl.exe -s --max-time 20 "%BACKEND_URL%/providers" > "%TEMP%\rag_providers.json"
"%VENV_PY%" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); [print('   [' + ('ONLINE ' if p['reachable'] else 'OFFLINE') + '] ' + p['label'].ljust(22) + p['base_url'].ljust(30) + 'modelo: ' + p['generation_model']) for p in d['providers']]; print('   Provider ativo: ' + d['active']); sys.exit(0 if any(p['name'] == 'remote' and p['reachable'] for p in d['providers']) else 1)" "%TEMP%\rag_providers.json"
if !errorlevel! equ 0 (
    set "REMOTE_STATUS=online"
) else (
    set "REMOTE_STATUS=offline"
    echo.
    echo    [AVISO] A API do servidor nao respondeu.
    echo            /v1/ready e publico, entao isto e conectividade ou o servidor
    echo            esta fora do ar -- nao e a chave. Ver docs\llm-api-referencia.md.
    echo            O modo Local segue funcionando normalmente.
    echo            Para subir ja no modo servidor: scripts\start_dev_remote.bat
)
del "%TEMP%\rag_providers.json" >nul 2>&1

REM --------------------------------------------------------------- 3/4 frontend
echo.
echo == 3/4  Frontend ==
curl.exe -s -o nul --max-time 3 "%FRONTEND_URL%/chat.html"
if !errorlevel! equ 0 (
    echo    Ja estava no ar em %FRONTEND_URL%
) else (
    echo    Subindo em %FRONTEND_URL% ...
    start "RAG frontend" /d "%PROJECT_ROOT%\frontend" cmd /k ""%VENV_PY%" -m http.server 5500"
    set FRONTEND_OK=0
    for /l %%i in (1,1,15) do (
        if !FRONTEND_OK! equ 0 (
            ping -n 2 127.0.0.1 >nul
            curl.exe -s -o nul --max-time 3 "%FRONTEND_URL%/chat.html"
            if !errorlevel! equ 0 set FRONTEND_OK=1
        )
    )
    if !FRONTEND_OK! equ 0 (
        echo.
        echo    [ERRO] O frontend nao respondeu em ~15s.
        echo           Veja a janela "RAG frontend".
        goto :erro
    )
    echo    OK
)

REM ------------------------------------------------------------------ 4/4 testes
echo.
echo == 4/4  Testes E2E ^(Playwright^) ==
if not exist "%PROJECT_ROOT%\frontend\tests\node_modules" (
    echo    [AVISO] node_modules nao encontrado -- testes pulados.
    echo            Instale com: cd frontend\tests ^&^& npm install ^&^& npx playwright install chromium
    set "TESTS_STATUS=pulados (falta npm install)"
) else (
    echo    Rodando ^(~1-2 min, inclui geracao real do LLM^)...
    echo.
    pushd "%PROJECT_ROOT%\frontend\tests"
    call npm test
    if !errorlevel! equ 0 (
        set "TESTS_STATUS=PASSARAM"
    ) else (
        set "TESTS_STATUS=FALHARAM"
    )
    popd
)

REM ---------------------------------------------------------------------- resumo
echo.
echo =====================================================
echo   Ambiente no ar
echo =====================================================
echo   Backend .......... %BACKEND_URL%
echo   Frontend ......... %FRONTEND_URL%/index.html
echo   Servidor ......... !REMOTE_STATUS!
echo   Testes E2E ....... !TESTS_STATUS!
echo.
echo   Para parar: feche as janelas "RAG backend" e "RAG frontend".
echo =====================================================
echo.
start "" "%FRONTEND_URL%/index.html"
pause
exit /b 0

:erro
echo.
pause
exit /b 1
