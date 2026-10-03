#!/usr/bin/env bash
# One-time GPU-box setup with uv: venv + pinned CUDA stack + deps (+ optional flash-attn) + HF / W&B login.
#   scripts/train/setup.sh [--flash-attn] [--no-login]
#
# Pinned stack (all prebuilt wheels, nothing compiled):
#   Python 3.12 | PyTorch 2.10.0 + CUDA 13.0 (cu130 wheel index; needs NVIDIA driver >= 580)
#   causal-conv1d 1.7.0 prebuilt wheel cu13 / torch2.10 / cxx11abi=TRUE / cp312 (Qwen3.5 Gated DeltaNet conv)
# uv is installed to ~/.local/bin if missing; uv also provides Python 3.12 if the system has none.
# Override with TORCH_INDEX=https://download.pytorch.org/whl/cu130 if needed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
FLASH=0; LOGIN=1
for a in "$@"; do case "$a" in --flash-attn) FLASH=1;; --no-login) LOGIN=0;; esac; done

TORCH_VERSION="2.10.0"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
CAUSAL_CONV1D_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 --seed .venv
. .venv/bin/activate
# torch first from the CUDA index; the requirements below then keep this build (torch==2.10.0 is already satisfied)
uv pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
uv pip install -r requirements-train.txt
uv pip install --no-deps "$CAUSAL_CONV1D_WHEEL"
python - <<'PY'
import torch, causal_conv1d
assert torch.__version__.startswith("2.10.0"), torch.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      "| causal_conv1d", causal_conv1d.__version__)
PY
if [ "$FLASH" = 1 ]; then
  uv pip install ninja packaging setuptools wheel   # build deps: --no-build-isolation uses this env
  uv pip install flash-attn --no-build-isolation || echo "flash-attn install failed; falling back to sdpa"
fi
if [ "$LOGIN" = 1 ]; then
  (hf auth login || huggingface-cli login) || echo "HF login skipped"
  wandb login || echo "wandb login skipped (set WANDB_MODE=disabled to train without it)"
fi
echo "setup done. Next: scripts/train/prefetch_data.sh && scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml"
