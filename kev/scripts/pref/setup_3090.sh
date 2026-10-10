#!/usr/bin/env bash
# One-time setup on the 3090 training machine. Mirrors modal_app.py's image (fla, triton, causal-conv1d) for the Qwen3.5 hybrid (Gated DeltaNet) base.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
CAUSAL_CONV1D="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
uv sync
# fla pinned: kev.fused_qwen35 patches its kernel launches; triton>=3.7.1 because torch 2.8 pins 3.4, which fla refuses on some GPUs
uv pip install "flash-linear-attention==0.5.2" "triton>=3.7.1"
# --no-deps: resolving its torch requirement would put back torch's pinned triton. The wheel is cp313/torch2.8: uv's python must be 3.13.
uv pip install --no-deps "$CAUSAL_CONV1D"

# sanity: eyeball the rendered chat prompt (the chat format reads the reward at its last token; tokenizer only, no weights)
uv run --no-sync python - <<'PY'
import torch, fla, causal_conv1d
print("torch", torch.__version__, "cuda", torch.version.cuda, "fla", getattr(fla, "__version__", "?"), "causal_conv1d", getattr(causal_conv1d, "__version__", "?"))
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
