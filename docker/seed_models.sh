#!/bin/sh
# Copia, de uma pasta de modelos do Ollama (a da máquina), SÓ os modelos pedidos
# para outra (o volume do Docker): o manifesto de cada um e os blobs que ele
# referencia. Evita baixar de novo o que a máquina já tem e evita copiar a pasta
# inteira (que costuma ter outros modelos, de outros projetos).
#
# Por que copiar em vez de montar a pasta do Windows direto no container: lida
# através da montagem, a carga de um modelo ficou em ~35 MB/s (gemma3:4b levou
# 1 min 52 s para a primeira resposta); do volume, dentro da VM do Docker, é
# ordem de grandeza mais rápida. Numa demonstração o gerador é recarregado a
# cada vez que o juiz ocupa a GPU, então isso pesa.
#
# Quem chama é o serviço models-seed do docker-compose.yml, antes do ollama:
#   sh /scripts/seed_models.sh /src /dst modelo1 modelo2 ...
# com /src = OLLAMA_SEED_DIR (só leitura) e /dst = o volume dos modelos.
#
# Nunca derruba a stack: sem pasta de origem, não há nada a copiar; um modelo
# que falta na máquina, ou cuja cópia falha, fica para o serviço "models", que
# baixa o que faltar. Por isso o script sai com 0 nesses casos.
set -u

SRC=$1
DST=$2
shift 2

if [ ! -d "$SRC/manifests" ]; then
  echo "Sem pasta de modelos da máquina para copiar (OLLAMA_SEED_DIR não definida, ou sem manifests/):"
  echo "o serviço 'models' baixa o que faltar."
  exit 0
fi

seed_one() {
  spec=$1
  name=${spec%%:*}
  tag=${spec#*:}
  [ "$tag" = "$spec" ] && tag=latest
  manifest="manifests/registry.ollama.ai/library/$name/$tag"
  if [ ! -f "$SRC/$manifest" ]; then
    echo "  $spec: não está na pasta da máquina (o serviço 'models' baixa)"
    return 0
  fi
  mkdir -p "$DST/$(dirname "$manifest")" "$DST/blobs" || return 1
  for digest in $(grep -o 'sha256:[0-9a-f]\{64\}' "$SRC/$manifest" | sort -u); do
    blob="blobs/sha256-${digest#sha256:}"
    if [ ! -f "$SRC/$blob" ]; then
      echo "  $spec: o manifesto cita $blob, que não está na pasta da máquina"
      return 1
    fi
    size=$(stat -c %s "$SRC/$blob") || return 1
    if [ -f "$DST/$blob" ] && [ "$size" = "$(stat -c %s "$DST/$blob")" ]; then
      continue
    fi
    mb=$((size / 1048576))
    if [ "$mb" -gt 0 ]; then echo "  $spec: copiando $mb MB"; fi
    cp "$SRC/$blob" "$DST/$blob.part" || return 1
    mv "$DST/$blob.part" "$DST/$blob" || return 1
  done
  # Manifesto por último: o Ollama só enxerga o modelo depois que os blobs estão lá.
  # Igual ao que já está no destino (a mesma pasta montada nos dois lados, ou
  # uma subida seguinte), não copia: o cp de um arquivo sobre ele mesmo falha.
  if cmp -s "$SRC/$manifest" "$DST/$manifest"; then
    echo "  $spec: ok (já estava no volume)"
    return 0
  fi
  cp "$SRC/$manifest" "$DST/$manifest.part" || return 1
  mv "$DST/$manifest.part" "$DST/$manifest" || return 1
  echo "  $spec: ok"
}

for spec in "$@"; do
  seed_one "$spec" || echo "  [AVISO] $spec: a cópia falhou; o serviço 'models' baixa o que faltar."
done
exit 0
