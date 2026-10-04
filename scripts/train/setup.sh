#!/usr/bin/env bash
# One-time GPU-box setup: installs uv if missing, syncs the uv project env (.venv) from uv.lock, checks the CUDA stack,
# then HF / W&B login.
#   [CU=128|130] scripts/train/setup.sh [--no-login]
# Locked stack (prebuilt wheels, nothing compiled): Python 3.12 | torch 2.10.0 + causal-conv1d 1.7.0 for
#   CU=128 (default): CUDA 12.8, NVIDIA driver >= 570    CU=130: CUDA 13.0, NVIDIA driver >= 580
# Every other script reads the same CU, so export it when not using the default.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
LOGIN=1
for a in "$@"; do case "$a" in --no-login) LOGIN=0;; esac; done
case "$CU" in 128|130) ;; *) echo "CU must be 128 or 130" >&2; exit 1;; esac

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked --project "$REPO_ROOT" --no-default-groups --group data --group train --group "cu$CU"
py - <<'PY'
import os, torch, causal_conv1d, fla, tilelang   # tilelang: fla's Hopper backward backend (fla issue #640)
assert torch.__version__.startswith("2.10.0"), torch.__version__
assert torch.version.cuda.replace(".", "") == os.environ["CU"], (torch.version.cuda, os.environ["CU"])
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      "| causal_conv1d", causal_conv1d.__version__, "| fla", fla.__version__, "| tilelang", tilelang.__version__)
from fla.utils import IS_NVIDIA_HOPPER
from fla.utils import has_usable_nvcc
# fla refuses Hopper training with triton 3.6 unless its TileLang backend is usable, which needs nvcc (fla #640)
if IS_NVIDIA_HOPPER and not has_usable_nvcc():
    raise SystemExit("Hopper GPU but no nvcc for TileLang (fla #640): use CU=130 scripts/train/setup.sh (nvcc wheel, "
                     "driver >= 580), or install a CUDA 12.x toolkit and export CUDA_HOME=/usr/local/cuda-12.x")
print("hopper", IS_NVIDIA_HOPPER, "| nvcc for tilelang", has_usable_nvcc())
PY
if [ "$LOGIN" = 1 ]; then
  uvr hf auth login || echo "HF login skipped"
  uvr wandb login || echo "wandb login skipped (set WANDB_MODE=disabled to train without it)"
fi
echo "setup done. Next: scripts/train/prefetch_data.sh && scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml"
