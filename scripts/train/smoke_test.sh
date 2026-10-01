#!/usr/bin/env bash
# Tiny-model end-to-end smoke test (CPU or GPU, no downloads, no W&B): train -> eval -> rank.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
export WANDB_MODE=disabled
OUT="${1:-outputs/smoke}"
python -m scrm.synth "$OUT/synth"
python -m scrm.train --config configs/debug_tiny.yaml data.local_dir="$OUT/synth" output_dir="$OUT/run" "${@:2}"
python -m scrm.evaluate --ckpt "$OUT/run/best" --split test --out "$OUT/test_metrics.json" data.local_dir="$OUT/synth"
python - "$OUT/run/best" <<'PY'
import sys
from scrm.model import load_scrm
m = load_scrm(sys.argv[1])
print(m.rank("Pick the best option", {"k": "v"}, ["alpha", "beta", "gamma"]))
PY
echo "smoke test OK"
