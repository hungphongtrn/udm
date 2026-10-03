#!/usr/bin/env bash
# Build every source (tasksource, samatv, samatv_clean50k, openjev) into $OUT.
# Env: OUT (default build/scrm_out), CACHE (HF download cache), WORKERS (default nproc).
# Extra args are passed to scrm_data.build (e.g. --delete-source to free disk as source files finish,
# --existing-keys to also check train rows against existing udm-massive-typed val/test rows).
set -euo pipefail
source "$(dirname "$0")/_env.sh"
py -m scrm_data.build --source all --out "$OUT" --cache-dir "$CACHE" --workers "$WORKERS" "$@"
py -m scrm_data.report --out "$OUT" > /dev/null
echo "built into $OUT ; next: scripts/data/validate.sh && scripts/data/push.sh"
