#!/usr/bin/env bash
# One-time Decision Index 0.2.1 suite build (the suite is not redistributed: rebuilt from pinned public sources,
# ~7 GB downloads, ~17 GB work space; accept the cais/hle terms on the Hub and be logged in first), plus the fixed
# stratified sample used by the in-training eval (benchmarks.decision_index).
#   scripts/train/dindex_setup.sh [DI_DIR=../decision-index] [N=1000]
# The rebuild extras (pandas, scipy, mido, python-chess, tiktoken) go into a separate venv ($DI_DIR/.venv) so the
# pinned training env is untouched.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DI_REV=87d4650b42b377c0291a89c1f1a879f9b31082bf
DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
N="${N:-1000}"
[ -d "$DI_DIR" ] || git clone https://github.com/apolinario/decision-index "$DI_DIR"
cd "$DI_DIR" && git checkout -q "$DI_REV"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[rebuild]"
export HF_HUB_DISABLE_XET=1
OUT=work/artifacts/benchmark-suite/release-v2-rebuilt
[ -f "$OUT/added-rows.jsonl.gz" ] || .venv/bin/python -m decision_index suite rebuild --work work
[ -f suite-0.2/manifest.json ] || .venv/bin/python -m decision_index suite import --dir suite-0.2 \
    --rows "$OUT/selected-rows.jsonl.gz" --added-rows "$OUT/added-rows.jsonl.gz"
.venv/bin/python -m decision_index suite verify --dir suite-0.2
.venv/bin/python -m decision_index suite sample --dir suite-0.2 --n "$N" --out "sample-$N.jsonl.gz"
echo "set: benchmarks.decision_index={enabled: true, suite_dir: $DI_DIR/suite-0.2, rows: $DI_DIR/sample-$N.jsonl.gz}"
