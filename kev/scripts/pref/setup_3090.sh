#!/usr/bin/env bash
# One-time setup on the 3090 training machine. Mirrors modal_app.py's image (fla, triton, causal-conv1d) for the Qwen3.5 hybrid (Gated DeltaNet) base.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
# Everything (torch 2.10, causal-conv1d, flash-linear-attention, flash-attn) comes from pyproject.toml's extras; no `uv pip install`.
# TORCH_EXTRA selects the build: cu128 (default, driver >= 570) or cu130 (driver >= 580).
# Ignore any activated venv / conda env: everything goes into kev/.venv, recreated if it was built with another Python.
TORCH_EXTRA=${TORCH_EXTRA:-cu128}
unset VIRTUAL_ENV CONDA_PREFIX UV_PYTHON
uv python install 3.12
if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
  echo "kev/.venv is not Python 3.12: recreating it"; rm -rf .venv
fi
[ -f uv.lock ] || uv lock
uv sync --python 3.12 --extra "$TORCH_EXTRA"
PY=.venv/bin/python
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; print("python", sys.version.split()[0], sys.executable)'

# sanity: eyeball the rendered chat prompt (the chat format reads the reward at its last token; tokenizer only, no weights)
"$PY" - <<'PY'
import torch, triton, fla, causal_conv1d, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, "fla", fla.__version__, "causal_conv1d", causal_conv1d.__version__, "flash_attn", flash_attn.__version__)
print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "NONE")
from transformers import AutoTokenizer
from kev.data import load_records, materialize
from kev.model import chat_prompt, chat_template_digest, chat_template_parts
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", revision="1001bb4d826a52d1f399e183466143f4da7b741b")
head, tail = chat_template_parts(tok)   # raises if the Base tokenizer has no chat template
print("chat template head:", repr(head), "tail:", repr(tail), "sha256:", chat_template_digest(tok))
rec = materialize(load_records("evals/v7/decision-v7/development.jsonl")[0])
text, n = chat_prompt(tok, rec, q=0, k=0)
print(f"--- rendered chat prompt, decision-v7 development record 0, question 0, option 0 ({n} tokens; reward read at the last token) ---")
print(text)
print("--- end ---")
PY
echo "setup done (extra: $TORCH_EXTRA). If kev/uv.lock was just generated, commit it from this machine: git add kev/uv.lock && git commit -m 'kev: lock py3.12 / torch 2.10'"
