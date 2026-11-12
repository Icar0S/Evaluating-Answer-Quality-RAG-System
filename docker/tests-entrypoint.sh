#!/bin/sh
# Entrypoint do container de testes (docker/tests.Dockerfile). Quem chama é o
# scripts/docker_tests.bat|.sh: docker compose --profile tests run tests [modo] [opções].
#
#   (sem modo, ou só opções)  tests/run_all.py com as respostas pela API do RAG no ar (RUN_ALL_API_URL)
#   demo [--goldens N]        tests/demo_e2e.py: E2E nos dois RAGs, RAGAS e o relatório comparativo
#   demo-rapido               a demonstração ao vivo, sem perguntas de referência nem relatório
#   hermeticos                API, gerador de relatório e JevEval com o Jev falso (não precisa da stack)
#   relatorio-e2e             o relatório HTML do Playwright em http://localhost:9323
#   shell                     bash, para depurar
#
# Passos: (1) encaminha o localhost do container para os serviços; (2) espera o
# backend; (3) imprime o preflight; (4) roda o modo como processo filho, e não
# com exec, para (5) na saída devolver ao dono da pasta no host os arquivos
# criados nas pastas montadas.
set -u
cd /app

MODE=run_all
case "${1:-}" in
  demo|demo-rapido|hermeticos|relatorio-e2e|shell) MODE=$1; shift ;;
  run_all) shift ;;
esac

# --- 5. Na saída ----------------------------------------------------------------------------------------------
# No Linux, o container roda como root, e o que ele grava no bind mount ficaria de root na pasta do host. No
# Docker Desktop (Windows e macOS), o compartilhamento de arquivos já mapeia o dono, e isto não muda nada.
OUTPUTS="tests/reports/out tests/ragas/results tests/deepeval/results tests/jeveval/results
frontend/tests/playwright-report frontend/tests/test-results"
give_back() {
  for dir in $OUTPUTS; do
    [ -d "$dir" ] || continue
    owner=$(stat -c %u:%g "$dir" 2>/dev/null) || continue
    [ "$owner" = "0:0" ] && continue
    find "$dir" \( ! -uid "${owner%:*}" -o ! -gid "${owner#*:}" \) -exec chown -h "$owner" {} + 2>/dev/null || true
  done
}
trap give_back EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- 1. O localhost do container chega aos serviços --------------------------------------------------------
# O frontend, os testes e o run_all têm localhost:8010/8011/5510 fixos, e a origem do navegador continua
# http://localhost:5510: com o encaminhamento, o CORS dos backends não muda e nenhum arquivo do frontend muda.
# O Chromium e o Node podem resolver localhost para ::1; por isso, o mesmo no loopback IPv6, se existir.
has_ipv6_loopback() {
  grep -q '^00000000000000000000000000000001 ' /proc/net/if_inet6 2>/dev/null
}
# O log do socat vai para /tmp/socat-<porta>.log (modo shell, para depurar): na tela, cada conexão que o
# Chromium fecha virava uma linha de aviso (18 num --rapido). Um encaminhamento quebrado continua aparecendo,
# porque a espera do passo 2 passa por ele.
forward() {  # porta local, destino host:porta
  socat "TCP4-LISTEN:$1,bind=127.0.0.1,fork,reuseaddr" "TCP:$2" 2>>"/tmp/socat-$1.log" &
  if has_ipv6_loopback; then
    socat "TCP6-LISTEN:$1,bind=[::1],fork,reuseaddr,ipv6only=1" "TCP:$2" 2>>"/tmp/socat-$1.log" &
  fi
}

# --- 2. Espera o backend ------------------------------------------------------------------------------------
# O mesmo critério do run_all.backend_is_ours: o /health deste projeto, que traz ollama_reachable.
wait_backend() {
  limit=${TESTS_WAIT_SECONDS:-600}
  start=$(date +%s)
  while :; do
    if python3 -c 'import sys, urllib.request as u
sys.exit(0 if "ollama_reachable" in u.urlopen("http://127.0.0.1:8010/health", timeout=5).read().decode() else 1)' \
        2>/dev/null; then
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$limit" ]; then
      echo "[ERRO] O backend não respondeu em http://localhost:8010 (backend:8010) em ${limit} s." >&2
      echo "       Veja: docker compose ps   e   docker compose logs backend" >&2
      return 1
    fi
    sleep 3
  done
}

hermetic_suites() {
  failed=0
  echo "--> API e benchmark (pytest da raiz: tests/api e tests/benchmark)"
  backend/.venv/bin/python -m pytest -p no:cacheprovider || failed=1
  echo "--> Gerador de relatório (pytest)"
  tests/reports/.venv/bin/python -m pytest tests/reports -p no:cacheprovider || failed=1
  echo "--> JevEval: testes herméticos (Jev falso)"
  tests/jeveval/.venv/bin/python -m pytest tests/jeveval -m "not live" -p no:cacheprovider || failed=1
  return $failed
}

echo "== Testes no container (modo: $MODE) =="
case "$MODE" in
  run_all|demo|demo-rapido|shell)
    forward 8010 backend:8010
    forward 8011 backend-md:8010
    forward 5510 frontend:80
    ;;
esac
case "$MODE" in
  run_all|demo|demo-rapido)
    wait_backend || exit 1
    ;;
esac

# --- 3. Preflight -------------------------------------------------------------------------------------------
python3 scripts/stack_status.py --preflight "$MODE" "$@"

# --- 4. O modo, como processo filho -------------------------------------------------------------------------
case "$MODE" in
  run_all)       python3 tests/run_all.py "$@" ;;
  demo)          python3 tests/demo_e2e.py "$@" ;;
  demo-rapido)   (cd frontend/tests && npx playwright test -c playwright.demo.config.ts "$@") ;;
  hermeticos)    hermetic_suites ;;
  relatorio-e2e) (cd frontend/tests && npx playwright show-report --host 0.0.0.0 --port 9323) ;;
  shell)         bash "$@" ;;
esac
exit $?
