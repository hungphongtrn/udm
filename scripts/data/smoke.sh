#!/usr/bin/env bash
# Smoke test: run every converter on tiny REMOTE slices (HTTPS range reads, no big downloads), then validate.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
SMOKE="${SMOKE:-$(mktemp -d)}"
N="${N:-2000}"
python -m scrm_data.build --source tasksource samatv samatv_clean50k openjev --out "$SMOKE" --workers "${WORKERS:-4}" \
  --limit-rows "$N" --unit-filter '(-00000-of-|-00001-of-00013|-00001-of-00011|citation|mailroom|ir-control|sponsor)' --shard-bytes 100000000
python -m scrm_data.validate --out "$SMOKE" --workers "${WORKERS:-4}"
python -m scrm_data.report --out "$SMOKE" | head -60
echo "smoke output in $SMOKE (delete when done)"
