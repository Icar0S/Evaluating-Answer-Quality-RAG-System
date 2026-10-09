@echo off
setlocal

REM Roda todas as suites de teste e gera o relatorio consolidado (tests\run_all.py).
REM
REM Uso:
REM   scripts\run_all_tests.bat                       tudo (~40 min a horas: o DeepEval domina)
REM   scripts\run_all_tests.bat --rapido              so API, gerador de relatorio e E2E
REM   scripts\run_all_tests.bat --limite 2            qualidade em 2 goldens (fumaca)
REM   scripts\run_all_tests.bat --pular e2e deepeval  pula suites
REM   scripts\run_all_tests.bat --abrir               abre o PDF do relatorio no fim
REM
REM Relatorio e logs em tests\reports\out\<data>\ ; historico em
REM tests\reports\out\HISTORICO.md ; o ultimo sempre em tests\reports\out\ultimo\.

cd /d "%~dp0.."
chcp 65001 >nul

REM O orquestrador so usa biblioteca padrao: o Python do venv do backend basta,
REM e ele existe em qualquer maquina com o projeto configurado.
set "PY=%CD%\backend\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" tests\run_all.py %*
set "CODE=%errorlevel%"

echo.
if "%CODE%"=="0" (
    echo Tudo certo.
) else (
    echo Alguma suite falhou ou o relatorio nao foi gerado - veja o resumo acima e os logs.
)
pause
exit /b %CODE%
