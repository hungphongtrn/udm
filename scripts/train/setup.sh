#!/usr/bin/env bash
# One-time GPU-box setup: venv + pinned CUDA stack + deps (+ optional flash-attn) + HF / W&B login.
#   scripts/train/setup.sh [--flash-attn] [--no-login]
#
# Pinned stack (all prebuilt wheels, nothing compiled):
#   Python 3.12 | PyTorch 2.10.0 + CUDA 12.8 (cu128 wheel index; needs NVIDIA driver >= 570)
#   causal-conv1d 1.7.0 prebuilt wheel cu12 / torch2.10 / cxx11abi=TRUE / cp312 (Qwen3.5 Gated DeltaNet conv)
# Override with PYTHON=python3.12 TORCH_INDEX=https://download.pytorch.org/whl/cu128 if needed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
FLASH=0; LOGIN=1
for a in "$@"; do case "$a" in --flash-attn) FLASH=1;; --no-login) LOGIN=0;; esac; done

PYTHON="${PYTHON:-python3.12}"
TORCH_VERSION="2.10.0"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CAUSAL_CONV1D_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

command -v "$PYTHON" >/dev/null || { echo "need $PYTHON (e.g. apt install python3.12-venv, or uv python install 3.12)"; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "Python 3.12 required (prebuilt causal-conv1d wheel is cp312)"; exit 1; }
"$PYTHON" -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
pip install -r requirements-train.txt
pip install --no-deps "$CAUSAL_CONV1D_WHEEL"
python - <<'PY'
import torch, causal_conv1d
assert torch.__version__.startswith("2.10.0"), torch.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      "| causal_conv1d", causal_conv1d.__version__)
PY
if [ "$FLASH" = 1 ]; then
  pip install ninja packaging
  pip install flash-attn --no-build-isolation || echo "flash-attn install failed; falling back to sdpa"
fi
if [ "$LOGIN" = 1 ]; then
  (hf auth login || huggingface-cli login) || echo "HF login skipped"
  wandb login || echo "wandb login skipped (set WANDB_MODE=disabled to train without it)"
fi
echo "setup done. Next: scripts/train/prefetch_data.sh && scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml"
