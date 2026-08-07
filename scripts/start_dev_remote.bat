@echo off
setlocal enabledelayedexpansion

REM Sobe o ambiente usando o SERVIDOR para gerar, sem carregar o LLM de geracao
REM na maquina local. Util em notebook sem GPU livre, com bateria, ou quando o
REM qwen3:8b nao cabe na VRAM.
REM
REM O QUE ISTO NAO FAZ: dispensar o Ollama por completo.
REM   A recuperacao (retrieval) embeda a pergunta a cada consulta, e a API do
REM   servidor nao expoe rota de embeddings (ver docs\llm-api-referencia.md).
REM   Por isso o Ollama local continua obrigatorio -- mas SO com o de embeddings
REM   (nomic-embed-text, ~274MB, roda em CPU). O modelo de geracao (qwen3:8b,
REM   ~5GB de VRAM) nao e carregado em momento nenhum neste modo.
REM
REM   Em resumo:  embeddings = local (leve)   |   geracao = servidor (pesado)
REM
REM Diferencas para scripts\start_dev.bat:
REM   - forca ACTIVE_PROVIDER=remote em vez de usar o default do .env
REM   - exige que a API do servidor esteja acessivel (aqui ela nao e opcional)
REM   - confere que o Ollama tem o modelo de embeddings, e nao o de geracao
REM   - nao roda a suite E2E (chat-flow gastaria fila do servidor a cada execucao)
REM
REM Uso: scripts\start_dev_remote.bat

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"
set "VENV_PY=%PROJECT_ROOT%\backend\.venv\Scripts\python.exe"
set "BACKEND_URL=http://localhost:8000"
set "FRONTEND_URL=http://localhost:5500"
set "CFG=%TEMP%\rag_cfg.txt"
set "JSON=%TEMP%\rag_remote.json"

echo.
echo =========================================================
echo   RAG Test Specialist -- modo SERVIDOR (sem LLM local)
echo =========================================================

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

REM ------------------------------------------------------------ 1/5 configuracao
echo.
echo == 1/5  Configuracao do .env ==

REM Saida em CHAVE=valor, e nao uma linha por campo na ordem: `for /f` pula
REM linhas em branco, entao um valor vazio (o caso comum de .env nao preenchido)
REM deslocaria todos os campos seguintes e a validacao acusaria o erro errado.
pushd "%PROJECT_ROOT%\backend"
"%VENV_PY%" -c "from app.config import get_settings; s=get_settings(); print('OLLAMA_URL=' + s.ollama_base_url); print('EMBED_MODEL=' + s.embedding_model); print('REMOTE_URL=' + (s.remote_api_base_url or '')); print('HAS_KEY=' + ('sim' if s.remote_api_key else 'nao')); print('REMOTE_MODEL=' + (s.remote_generation_model or '(default do servidor)'))" > "%CFG%" 2>nul
set CFG_OK=!errorlevel!
popd

if !CFG_OK! neq 0 (
    echo    [ERRO] Nao consegui ler a configuracao. As dependencias do backend
    echo           estao instaladas na venv?
    goto :erro
)

set "OLLAMA_URL="
set "EMBED_MODEL="
set "REMOTE_URL="
set "HAS_KEY="
set "REMOTE_MODEL="
for /f "usebackq tokens=1,* delims==" %%A in ("%CFG%") do set "%%A=%%B"
del "%CFG%" >nul 2>&1

if "!REMOTE_URL!"=="" (
    echo    [ERRO] REMOTE_API_BASE_URL nao esta definida no .env.
    echo.
    echo           Este script depende do servidor para gerar. Preencha na raiz
    echo           do projeto, no arquivo .env:
    echo.
    echo             REMOTE_API_BASE_URL=https://llm.smartdatatest.com
    echo             REMOTE_API_KEY=sk-...
    echo.
    echo           Detalhes do contrato em docs\llm-api-referencia.md.
    echo           Para rodar 100%% local, use scripts\start_dev.bat.
    goto :erro
)

if "!HAS_KEY!"=="nao" (
    echo    [ERRO] REMOTE_API_KEY nao esta definida no .env.
    echo           A API do servidor usa autenticacao Bearer; sem a chave toda
    echo           geracao volta 401. Nunca commite a chave real.
    goto :erro
)

echo    Servidor .......... !REMOTE_URL!
echo    Modelo remoto ..... !REMOTE_MODEL!
echo    Embeddings ........ !EMBED_MODEL! ^(local, obrigatorio^)

REM ---------------------------------------------------- 2/5 ollama (so embeddings)
echo.
echo == 2/5  Ollama local ^(so para embeddings^) ==

curl.exe -s --max-time 5 "!OLLAMA_URL!/api/tags" > "%JSON%" 2>nul
if !errorlevel! neq 0 (
    echo    [ERRO] O Ollama nao respondeu em !OLLAMA_URL!
    echo.
    echo           Mesmo gerando pelo servidor, o Ollama local e necessario: a
    echo           busca embeda a pergunta a cada consulta e a API do servidor
    echo           nao tem rota de embeddings.
    echo.
    echo           Inicie o Ollama. Ele so vai carregar o !EMBED_MODEL!
    echo           ^(~274MB, CPU^) -- o modelo de geracao nao sera tocado.
    del "%JSON%" >nul 2>&1
    goto :erro
)

"%VENV_PY%" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); alvo=sys.argv[2]; tags=[m.get('name','') for m in d.get('models',[])]; ok=any(t==alvo or t.split(':')[0]==alvo.split(':')[0] for t in tags); print('   Modelos presentes: ' + (', '.join(tags) if tags else '(nenhum)')); sys.exit(0 if ok else 1)" "%JSON%" "!EMBED_MODEL!"
if !errorlevel! neq 0 (
    echo.
    echo    [ERRO] O modelo de embeddings "!EMBED_MODEL!" nao esta instalado.
    echo           Baixe com:  ollama pull !EMBED_MODEL!
    del "%JSON%" >nul 2>&1
    goto :erro
)
del "%JSON%" >nul 2>&1
echo    OK -- modelo de geracao local NAO sera carregado neste modo.

REM ------------------------------------------------------------- 3/5 servidor
echo.
echo == 3/5  API do servidor ==
echo    Testando !REMOTE_URL!/v1/ready ...
curl.exe -s -o nul --max-time 15 "!REMOTE_URL!/v1/ready"
if !errorlevel! neq 0 (
    echo.
    echo    [ERRO] O servidor nao respondeu.
    echo           /v1/ready e publico ^(nao precisa de chave^), entao isto e
    echo           conectividade ou o servidor esta fora do ar -- nao e a chave.
    echo           Confira sua internet e o status do Mac mini.
    echo.
    echo           Para trabalhar sem o servidor, use scripts\start_dev.bat.
    goto :erro
)
echo    OK -- servidor acessivel.

REM -------------------------------------------------------------- 4/5 backend
echo.
echo == 4/5  Backend ==

REM O processo filho herda o ambiente deste script; variavel de ambiente tem
REM precedencia sobre o .env no pydantic-settings, entao o backend ja sobe com
REM o provider remoto ativo.
set "ACTIVE_PROVIDER=remote"

curl.exe -s -o nul --max-time 3 "%BACKEND_URL%/health"
if !errorlevel! equ 0 (
    echo    Ja estava no ar em %BACKEND_URL%
    echo    ^(vou trocar o provider ativo em runtime, sem reiniciar^)
) else (
    echo    Subindo em %BACKEND_URL% com ACTIVE_PROVIDER=remote ...
    start "RAG backend (remoto)" /d "%PROJECT_ROOT%\backend" cmd /k ""%VENV_PY%" -m uvicorn app.main:app --port 8000"
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
        echo           Veja o traceback na janela "RAG backend (remoto)".
        goto :erro
    )
    echo    OK
)

REM Garante o provider remoto tambem quando o backend ja estava rodando em local.
curl.exe -s -o nul --max-time 15 -X POST "%BACKEND_URL%/providers/active" -H "Content-Type: application/json" -d "{\"name\":\"remote\"}"
if !errorlevel! neq 0 (
    echo    [ERRO] Nao consegui ativar o provider remoto.
    goto :erro
)

curl.exe -s --max-time 20 "%BACKEND_URL%/providers" > "%JSON%"
"%VENV_PY%" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); [print('   [' + ('ONLINE ' if p['reachable'] else 'OFFLINE') + '] ' + p['label'].ljust(22) + p['base_url'].ljust(34) + 'modelo: ' + p['generation_model']) for p in d['providers']]; print('   Provider ativo: ' + d['active']); sys.exit(0 if d['active']=='remote' else 1)" "%JSON%"
if !errorlevel! neq 0 (
    echo    [ERRO] O provider ativo nao ficou em "remote".
    del "%JSON%" >nul 2>&1
    goto :erro
)
del "%JSON%" >nul 2>&1

REM Indice vetorial: sem ele o chat responde a frase-ancora de "nao encontrado".
curl.exe -s --max-time 20 "%BACKEND_URL%/health" > "%JSON%"
"%VENV_PY%" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); n=d.get('vector_store_documents',0); print('   Indice vetorial: ' + str(n) + ' chunks'); sys.exit(0 if n>0 else 1)" "%JSON%"
if !errorlevel! neq 0 (
    echo.
    echo    [AVISO] O indice esta vazio -- toda pergunta vai cair na resposta de
    echo            "nao encontrei no contexto". Ingira os PDFs com:
    echo              "%VENV_PY%" scripts\ingest_documents.py
    echo            ^(a ingestao usa so embeddings; nao precisa do LLM de geracao^)
)
del "%JSON%" >nul 2>&1

REM ------------------------------------------------------------- 5/5 frontend
echo.
echo == 5/5  Frontend ==
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

REM ---------------------------------------------------------------------- resumo
echo.
echo =========================================================
echo   Ambiente no ar -- gerando pelo SERVIDOR
echo =========================================================
echo   Backend .......... %BACKEND_URL%
echo   Frontend ......... %FRONTEND_URL%/index.html
echo   Geracao .......... !REMOTE_URL!  ^(modelo: !REMOTE_MODEL!^)
echo   Embeddings ....... !OLLAMA_URL!  ^(!EMBED_MODEL!^)
echo.
echo   No painel de Monitor, a aba "Servidor" aparece ativa e CPU/RAM/GPU
echo   ficam como indisponiveis -- e o esperado: sao metricas de hardware e
echo   o backend nao le o hardware de uma maquina remota. Tokens/s, latencia
echo   e status continuam reais, vem da resposta do proprio servidor.
echo.
echo   Para voltar ao modo local: clique na aba "Local" no Monitor, ou rode
echo   scripts\start_dev.bat com o backend parado.
echo.
echo   Testes E2E nao rodam aqui ^(gastariam fila do servidor^). Se quiser:
echo     cd frontend\tests ^&^& npm test
echo.
echo   Para parar: feche as janelas "RAG backend (remoto)" e "RAG frontend".
echo =========================================================
echo.
start "" "%FRONTEND_URL%/index.html"
pause
exit /b 0

:erro
echo.
pause
exit /b 1
