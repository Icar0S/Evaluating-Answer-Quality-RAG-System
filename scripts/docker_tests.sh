#!/bin/sh
# Os testes do RAG num container, contra a stack do docker compose (README,
# "Início rápido com Docker"). Só Docker na máquina: nada de Python, Node ou venvs.
#
# Uso:
#   ./scripts/docker_tests.sh                      todas as suítes, contra o RAG no ar (como o run_all_tests.bat)
#   ./scripts/docker_tests.sh --rapido             só os herméticos e o E2E (minutos)
#   ./scripts/docker_tests.sh --limite 2           qualidade em 2 goldens (fumaça)
#   ./scripts/docker_tests.sh --pular e2e deepeval --abrir
#   ./scripts/docker_tests.sh demo [--goldens N]   E2E nos dois RAGs, RAGAS e o relatório que compara os dois
#   ./scripts/docker_tests.sh demo-rapido          só os testes ao vivo da demonstração
#   ./scripts/docker_tests.sh hermeticos           API, relatório e JevEval com o Jev falso (sem a stack)
#   ./scripts/docker_tests.sh relatorio-e2e        o relatório HTML do Playwright em http://localhost:9323
#   ./scripts/docker_tests.sh reindexar            reindexa os dois RAGs (depois de trocar os documentos)
#   ./scripts/docker_tests.sh shell                bash no container, para depurar
#
# O script é fino: escolhe os arquivos do compose, decide se mexe na stack e
# chama o container (docker/tests-entrypoint.sh), que faz o resto. Relatório e
# resultados ficam nas pastas de sempre (tests/reports/out/<data>/ e
# tests/*/results/). --abrir é tratado aqui: abre o PDF do relatório novo.
set -u
cd "$(dirname "$0")/.."

if ! docker version >/dev/null 2>&1; then
  echo "[ERRO] O Docker não está no ar. Abra o Docker Desktop (ou inicie o serviço docker) e tente de novo." >&2
  exit 1
fi

# Os mesmos arquivos do `docker compose up`: sem isso, um `run` que sobe as
# dependências recriaria o ollama sem a GPU. COMPOSE_FILE no ambiente ou no .env
# vale como está; senão, a regra do demo_e2e.bat: com nvidia-smi, junta o
# docker-compose.gpu.yml.
if [ -z "${COMPOSE_FILE:-}" ] && ! grep -q '^COMPOSE_FILE=' .env 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
    COMPOSE_PATH_SEPARATOR=:
    export COMPOSE_FILE COMPOSE_PATH_SEPARATOR
    echo "GPU NVIDIA detectada: o Ollama do container vai usá-la."
  fi
fi

MODE=""
case "${1:-}" in
  demo|demo-rapido|hermeticos|relatorio-e2e|reindexar|shell) MODE=$1; shift ;;
esac

if [ "$MODE" = reindexar ]; then
  # Os backends estão com o banco aberto: param durante a ingestão e voltam no fim.
  echo "== Reindexando os dois RAGs (os backends param durante a ingestão) =="
  docker compose stop backend backend-md
  docker compose run --rm ingest python scripts/ingest_documents.py \
    && docker compose run --rm ingest-md python scripts/ingest_documents.py
  code=$?
  docker compose up -d
  exit $code
fi

# --abrir é daqui (no container não há visualizador); o resto vai para o container.
OPEN=""
n=$#
while [ "$n" -gt 0 ]; do
  arg=$1
  shift
  n=$((n - 1))
  if [ "$arg" = "--abrir" ]; then OPEN=1; else set -- "$@" "$arg"; fi
done

# Decide se mexe na stack: no ar, --no-deps (não toca em nada); fora do ar, o
# depends_on a sobe. Os modos que não precisam dela nunca a sobem.
NODEPS=""
case "$MODE" in
  hermeticos|relatorio-e2e|shell) NODEPS="--no-deps" ;;
  *)
    if [ -n "$(docker compose ps --status running -q backend 2>/dev/null)" ]; then
      NODEPS="--no-deps"
    else
      echo "A stack não está no ar: o docker compose vai subi-la antes dos testes."
      echo "Na primeira vez, isso inclui o build, os modelos (~9 GB) e a indexação: de minutos a meia hora."
      echo "Para acompanhar em outro terminal: docker compose logs -f models ingest ingest-md"
    fi
    ;;
esac

PORTS=""
if [ "$MODE" = relatorio-e2e ]; then
  PORTS="-p 9323:9323"
  echo "Relatório do Playwright em http://localhost:9323 (Ctrl+C para fechar)."
fi

# Criadas aqui, pelo usuário da máquina: as que faltarem, o Docker criaria como root (Linux).
mkdir -p tests/reports/out tests/ragas/results tests/deepeval/results tests/jeveval/results \
  frontend/tests/playwright-report frontend/tests/test-results

# O fuso da máquina, para as pastas tests/reports/out/<data> e o HISTORICO.md
# baterem com as execuções feitas fora do container. Formato POSIX (o glibc o
# entende sem tzdata): UTC-3 vira HOST3:00.
TZ_ARG=""
off=$(date +%z 2>/dev/null)
case "$off" in
  [+-][0-9][0-9][0-9][0-9])
    sign=$(printf %s "$off" | cut -c1)
    hh=$(printf %s "$off" | cut -c2-3)
    mm=$(printf %s "$off" | cut -c4-5)
    if [ "$sign" = "-" ]; then psign=""; else psign="-"; fi
    TZ_ARG="HOST${psign}${hh#0}:${mm}"
    ;;
esac

newest_report() {
  ls -d tests/reports/out/2*/relatorio.pdf 2>/dev/null | sort | tail -n 1
}
BEFORE=$(newest_report)

# shellcheck disable=SC2086  # NODEPS, PORTS e MODE vazios não viram argumento
docker compose --profile tests run --rm --build $NODEPS $PORTS ${TZ_ARG:+-e TZ=$TZ_ARG} tests $MODE "$@"
code=$?

if [ -n "$OPEN" ]; then
  AFTER=$(newest_report)
  if [ -n "$AFTER" ] && [ "$AFTER" != "$BEFORE" ]; then
    echo "Abrindo $AFTER"
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$AFTER" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then open "$AFTER"
    else echo "(sem xdg-open nem open: abra o arquivo à mão)"
    fi
  else
    echo "Nenhum relatório novo para abrir."
  fi
fi
exit $code
