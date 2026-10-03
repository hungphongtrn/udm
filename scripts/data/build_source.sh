#!/usr/bin/env bash
# Usage: build_source.sh <tasksource|samatv|samatv_clean50k|openjev> [extra build args]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
SRC="${1:?usage: build_source.sh <tasksource|samatv|samatv_clean50k|openjev> [args]}"; shift
py -m scrm_data.build --source "$SRC" --out "$OUT" --cache-dir "$CACHE" --workers "$WORKERS" "$@"
