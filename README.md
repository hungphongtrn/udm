# udm: Set-Conditioned Reward Model (SCRM)

A reward model that scores every candidate **in the context of the whole candidate set**:
`r_i = f(x, C, c_i)`.

- A Qwen3.5-4B text decoder (LoRA, Liger kernels) reads plain chat-format text with no new tokens. For each graded
  option there is one sequence: the user turn shows the state, the instruction, **all** options (`Option 1: …`,
  `Option 2: …`) and then `Grade this choice: Option k: …`; the sequence stops at the assistant header
  (`<|im_end|>\n<|im_start|>assistant\n`). The LLM sees every option but grades one.
- The prompt shared by all options of a set is encoded once and its KV / linear-attention state is reused for every graded
  option (exact, gradients flow through it). Context length: 8k tokens (24 GB config), 16k (40 GB).
- The hidden state of that last token is the option's embedding; it is projected 2560 → 768.
- A 2-layer bidirectional Set Transformer (no positional embeddings) runs over those vectors.
- An MLP head gives one scalar reward per candidate.
- Training uses Bradley–Terry loss over tier pairs (`tier_i < tier_j`; same-tier pairs are never trained).

The work is split over two machines that only share a Hugging Face dataset repo,
[`hungphongtrn/udm-massive-typed`](https://huggingface.co/datasets/hungphongtrn/udm-massive-typed):

```
Machine 1 (many CPUs)                         Machine 2 (1 GPU, 24-40 GB)
  build tasksource / samatv / Open-Jev   ──▶   HF dataset repo   ──▶   prefetch parquet → train → eval
  validate → push (append shards)              (existing + new shards, same schema)
```

| Doc | What it covers |
|---|---|
| [`docs/CONTRACT.md`](docs/CONTRACT.md) | The data ↔ trainer contract: row semantics and how each label kind becomes tiers |
| [`docs/DATA_SPEC.md`](docs/DATA_SPEC.md) | Full data spec: schema, hash recipes, per-source mapping, metadata recovery for score/noul, drop reasons |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Model design, memory budget, configs, data mixing, metrics, checkpoints, inference |

---

## Machine 1: build the data and push it to HF (CPU box)

Requirements:
- [uv](https://docs.astral.sh/uv/) (the setup script installs it to `~/.local/bin` if missing; it also provides Python 3.12).
- Many cores. The build is parallel over source files; `WORKERS` defaults to `nproc`.
- Disk: about 25 GB of output plus about 22 GB of source downloads. Pass `--delete-source` or `--stream` to avoid keeping the downloads.
- An HF token with write access to `hungphongtrn/udm-massive-typed`.

```bash
git clone https://github.com/hungphongtrn/udm.git && cd udm
git checkout claude/set-conditioned-reward-model-hvsb9s

scripts/data/setup.sh                       # uv: creates .venv-data with requirements-data.txt
source .venv-data/bin/activate
export HF_TOKEN=hf_...                      # write token (or: hf auth login)

# 0) optional, ~1 min: convert tiny remote slices of every source and validate them
scripts/data/smoke.sh

# 1) full build of all sources into $OUT (default build/scrm_out)
#    --existing-keys also removes train rows that duplicate existing validation/test rows in the repo
OUT=build/scrm_out WORKERS=$(nproc) scripts/data/build_all.sh --existing-keys --delete-source

# 2) validate every shard: exact schema vs. the remote shards, canonical JSON, tiers, hashes, unique ids
OUT=build/scrm_out scripts/data/validate.sh

# 3) push: prints the dry-run plan (new files + README section), asks, then uploads (append only)
OUT=build/scrm_out scripts/data/push.sh     # YES=1 to skip the prompt
```

How the build behaves:
- **Sources.** It converts `tasksource/tasksource-jev-typed-decisions`, `samatv256/jev-decisions-v1` (`default` and
  `general-clean-50k`) and every `ZefanCai/Open-Jev` config. Upstream revisions are pinned.
- **Classification framing.** Every label kind becomes tiers:
  - choice: one-hot or soft;
  - score: ordinal distance from the true level;
  - noul: a `no`/`yes` pair;
  - samatv: tool/route choice.
- **Original labels are kept.** The raw supervision stays recoverable in `source_label_or_null` and `probabilities_json`
  (score levels, expected level, raw `p_yes`, …). See DATA_SPEC §6.
- **Splits are best effort.** Each source's own split is used where it exists, mapped to train / validation / test. `source_split` keeps the raw name (e.g. `ood`, `calibration`).
- **Leakage and duplicates.** Train rows whose content appears in held-out rows are removed. Every drop is counted in `$OUT/receipts`, and `python -m scrm_data.report --out $OUT` prints the counts.
- **Append only.**
  - New files are `data/{split}-{source}-NNNNN-of-MMMMM.parquet`, with exactly the existing 26-column schema.
  - Existing files are never touched.
  - The existing `data/train-*`, `data/validation-*` and `data/test-*` globs pick up the new files automatically.
- **Resumable.** Finished units are skipped on re-run; `--force` rebuilds them.
- **One source at a time:** `scripts/data/build_source.sh <name>`.

## Machine 2: pull and train (GPU box)

Requirements:
- One NVIDIA GPU with 24 GB (e.g. 4090, L4, A10) or 40 GB+ (A100-40G, A6000, L40S).
- NVIDIA driver ≥ 580 (CUDA 13.0 runtime). uv is installed by the setup script if missing and provides Python 3.12.
- The setup script (uv) pins **Python 3.12 + PyTorch 2.10.0 (cu130) + causal-conv1d 1.7.0**, all prebuilt wheels with nothing compiled.
- About 40 GB of disk for the parquet snapshot plus the base model.

```bash
git clone https://github.com/hungphongtrn/udm.git && cd udm
git checkout claude/set-conditioned-reward-model-hvsb9s

# uv: creates .venv with Python 3.12, installs torch==2.10.0 from the cu130 index, requirements-train.txt
# (transformers>=5.18, peft, liger-kernel, flash-linear-attention, wandb, ...) and the prebuilt causal-conv1d wheel
# (cu13 / torch2.10 / cp312), then runs HF + wandb login. Optional --flash-attn also tries to install flash-attn
# (it is only used by the 1-in-4 full-attention layers; sdpa is fine without it).
scripts/train/setup.sh

# 0) optional: CPU smoke test (tiny random Qwen3.5, synthetic data, no downloads)
scripts/train/smoke_test.sh

# 1) download the dataset snapshot (parquet only); training then reads local files
scripts/train/prefetch_data.sh data_cache/udm

# 2) train (24 GB card)
scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml data.local_dir=data_cache/udm
#    40 GB card: longer sequences and bigger micro-batches
scripts/train/train.sh configs/scrm_qwen3_5_4b_40gb.yaml data.local_dir=data_cache/udm
#    resume after an interruption
scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml --resume auto data.local_dir=data_cache/udm

# 3) evaluate the best checkpoint (lowest validation loss) on test, and on the OOD slice
scripts/train/eval.sh outputs/scrm_qwen3_5_4b_24gb/best --split test --out test.json data.local_dir=data_cache/udm
scripts/train/eval.sh outputs/scrm_qwen3_5_4b_24gb/best --split test --filter source_split=ood --out ood.json data.local_dir=data_cache/udm
```

Notes for the GPU box:
- **Model.**
  - `Qwen/Qwen3.5-4B` is loaded via `AutoModel.from_pretrained`. The vision tower is dropped and only the text decoder is kept.
  - Liger kernels (fused RMSNorm and SwiGLU, through Liger's HF integration) are applied before loading.
  - LoRA is `all-linear` with rsLoRA (r=64, alpha=16, giving a scale of 2.0).
  - The SCRM set block and head are a plain `torch.nn.Module` wrapped around the backbone.
- **Memory (estimated, not yet measured).**
  - The 24 GB config should peak around 14–17 GB.
  - If it OOMs, add `model.quantize_4bit=true` (QLoRA, about 9–11 GB) or lower `data.batch.max_tokens_per_batch` and raise `train.grad_accum`.
  - Install `flash-linear-attention` (it is in the requirements). Without it, the Gated DeltaNet layers use a slow torch fallback.
- **Data mixing.**
  - Mixing is set per source in the config's `data.groups`. The weight is the sampling share and `max_rows` is a cap; MASSIVE and samatv are capped.
  - Every row is kept on the Hub; caps and mixing happen only at train time.
- **Monitoring.**
  - Training logs to W&B project `scrm`: losses, LR, grad norms, reward histograms, and validation metrics (pair accuracy, top-1, MRR, NDCG, Kendall τ). Each metric is reported overall and per family/source.
  - Set `WANDB_MODE=disabled` to run without W&B. `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` and `WANDB_TAGS` are honoured.
- **Overrides.** Any config value can be overridden on the command line with dotted keys, e.g. `train.max_steps=5000 model.lora.r=32`.
- **Streaming.** Instead of prefetching, `data.streaming=true` reads from `hf://` directly. The prefetch above is faster and exact.

Inference:

```python
from scrm.model import load_scrm
m = load_scrm("outputs/scrm_qwen3_5_4b_24gb/best", "cuda")
m.rank(instruction, state, ["candidate a", "candidate b", "candidate c"])  # [{index, reward}, ...] best first
```

## Repository layout

```
src/scrm_data/   data pipeline (sources/, tiers, canonical JSON + hashes, build, validate, push, report)
src/scrm/        model, losses, rendering/collation, data loading + mixing, train, evaluate, export, wandb utils
configs/         scrm_qwen3_5_4b_24gb.yaml, scrm_qwen3_5_4b_40gb.yaml, debug_tiny.yaml, ablations
scripts/data/    Machine 1 entry points        scripts/train/   Machine 2 entry points
tests/           pytest: test_data_* (per-source conversion formats, tiers, build) and test_scrm_* (model, losses, data, train)
```

Tests run offline on CPU in about 15 s:

```bash
uv venv --python 3.12 .venv-test && uv pip install --python .venv-test/bin/python torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-test/bin/python -r requirements-data.txt "transformers>=5.18" "peft>=0.14" "datasets>=3.0" \
  safetensors pyyaml
PYTHONPATH=src .venv-test/bin/python -m pytest -q tests/
```
