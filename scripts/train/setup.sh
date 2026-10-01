#!/usr/bin/env bash
# One-time setup: venv + deps (+ optional flash-attn / causal-conv1d) + HF / W&B login.
#   scripts/train/setup.sh [--flash-attn] [--no-login]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
FLASH=0; LOGIN=1
for a in "$@"; do case "$a" in --flash-attn) FLASH=1;; --no-login) LOGIN=0;; esac; done
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
# install torch matching your CUDA first if the default wheel is not right, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
if [ "$FLASH" = 1 ]; then
  pip install ninja packaging
  pip install flash-attn --no-build-isolation || echo "flash-attn install failed; falling back to sdpa"
  # fast causal conv for Qwen3.5 Gated DeltaNet layers (optional; torch fallback otherwise)
  pip install causal-conv1d --no-build-isolation || echo "causal-conv1d install failed; using torch fallback"
fi
if [ "$LOGIN" = 1 ]; then
  (huggingface-cli login || hf auth login) || echo "HF login skipped"
  wandb login || echo "wandb login skipped (set WANDB_MODE=disabled to train without it)"
fi
echo "setup done. Next: scripts/train/prefetch_data.sh && scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml"
