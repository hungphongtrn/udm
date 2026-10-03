#!/usr/bin/env bash
# Download the dataset snapshot (parquet only) to a local dir so training starts immediately / offline.
#   scripts/train/prefetch_data.sh [local_dir=data_cache/udm] [repo=hungphongtrn/udm-massive-typed]
# then train with: data.local_dir=data_cache/udm
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DEST="${1:-$REPO_ROOT/data_cache/udm}"
REPO="${2:-hungphongtrn/udm-massive-typed}"
py - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:3]
p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, allow_patterns=["data/*.parquet", "README.md"],
                      max_workers=8)
print("snapshot at", p)
PY
echo "Use: scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml data.local_dir=$DEST"
