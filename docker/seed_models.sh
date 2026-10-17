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
# Uso (scripts/demo_e2e.bat chama assim):
#   docker run --rm --entrypoint sh -v <pasta-da-maquina>:/src:ro -v <volume>:/dst \
#     -v <repo>/docker:/scripts:ro ollama/ollama:<versão> /scripts/seed_models.sh /src /dst modelo1 modelo2 ...
set -eu

SRC=$1
DST=$2
shift 2

for spec in "$@"; do
  name=${spec%%:*}
  tag=${spec#*:}
  [ "$tag" = "$spec" ] && tag=latest
  manifest="manifests/registry.ollama.ai/library/$name/$tag"
  if [ ! -f "$SRC/$manifest" ]; then
    echo "  $spec: não está na pasta da máquina (o serviço 'models' baixa)"
    continue
  fi
  mkdir -p "$DST/$(dirname "$manifest")" "$DST/blobs"
  for digest in $(grep -o 'sha256:[0-9a-f]\{64\}' "$SRC/$manifest" | sort -u); do
    blob="blobs/sha256-${digest#sha256:}"
    if [ -f "$DST/$blob" ] && [ "$(stat -c %s "$SRC/$blob")" = "$(stat -c %s "$DST/$blob")" ]; then
      continue
    fi
    mb=$(( $(stat -c %s "$SRC/$blob") / 1048576 ))
    if [ "$mb" -gt 0 ]; then echo "  $spec: copiando $mb MB"; fi
    cp "$SRC/$blob" "$DST/$blob.part"
    mv "$DST/$blob.part" "$DST/$blob"
  done
  # Manifesto por último: o Ollama só enxerga o modelo depois que os blobs estão lá.
  cp "$SRC/$manifest" "$DST/$manifest"
  echo "  $spec: ok"
done
