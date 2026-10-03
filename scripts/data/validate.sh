#!/usr/bin/env bash
# Validate all shards in $OUT/data (schema vs. an existing remote shard footer, JSON, tiers, hashes, unique ids).
set -euo pipefail
source "$(dirname "$0")/_env.sh"
py -m scrm_data.validate --out "$OUT" --workers "$WORKERS" "$@"
