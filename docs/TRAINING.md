# SCRM training guide

Set-Conditioned Reward Model: a Qwen3.5-4B text backbone (+LoRA, Liger kernels) scores a decision set with
plain text in the pretrained chat format and **no new tokens**. For each graded option k there is one sequence: the user
turn shows the state, the instruction and **all** options, then asks to grade one of them; the sequence stops at the
assistant header:

```
<|im_start|>user
State: {state}

Instruction: {instruction}

Options:
Option 1: {candidate 1}
Option 2: {candidate 2}
...

Grade this choice: Option k: {candidate k}<|im_end|>
<|im_start|>assistant
```

So the LLM sees all available options but grades exactly one: option k's embedding is the hidden state of the last
token (the `\n` ending the assistant header, i.e. where the model would start answering). Empty state / instruction
blocks are omitted; options are numbered in display order (shuffled every time in training, canonical in eval).
Rows of the same set share everything up to "Grade this choice:" and differ only in the graded option.
Knobs: `data.render.chat_template`, `system_prompt`, `state_label`, `instruction_label`, `options_header`, `option_label`, `grade_prompt`, `max_graded`
(options graded per set in training; tier-0 and one per other tier always kept; the prompt still lists every shown
option), `eval_max_graded` (default all; `rank()` always grades all).
The per-option embeddings of a set are projected (d -> 768), passed through a 2-layer bidirectional
pre-LN transformer encoder **without positional embeddings** (so scores do not depend on candidate order), and an MLP head
returns one unbounded scalar reward per candidate. Training uses Bradley-Terry over tier pairs
(`tier_i < tier_j`, never same-tier). Data contract: `docs/CONTRACT.md`.

Code: `src/scrm/` (`model.py`, `losses.py`, `render.py`, `collator.py`, `data.py`, `metrics.py`, `train.py`,
`evaluate.py`, `export.py`, `wandb_utils.py`). Everything is plain PyTorch, single GPU.

## Quick start

```bash
scripts/train/setup.sh [--flash-attn]          # py3.12 venv, torch 2.10 cu130, deps, causal-conv1d wheel, hf + wandb login
scripts/train/prefetch_data.sh data_cache/udm  # snapshot parquet (hf_transfer) -> training starts immediately / offline
scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml data.local_dir=data_cache/udm
scripts/train/train.sh configs/scrm_qwen3_5_4b_24gb.yaml --resume auto data.local_dir=data_cache/udm
scripts/train/eval.sh outputs/scrm_qwen3_5_4b_24gb/best --split test --out test.json data.local_dir=data_cache/udm
scripts/train/eval.sh outputs/.../best --split test --filter source_split=ood --out ood.json   # filtered eval
scripts/train/smoke_test.sh                    # tiny random model, CPU ok, no downloads
```

Config = `configs/*.yaml` merged over defaults in `src/scrm/config.py`; YAML may use `base: other.yaml`. Any value can be
overridden on the CLI with dotted keys (`train.max_steps=500 model.lora.r=32 data.streaming=true`). `WANDB_MODE=disabled`
turns W&B off; env `WANDB_PROJECT/ENTITY/RUN_NAME/TAGS` are honoured.

Programmatic inference:

```python
from scrm.model import load_scrm
m = load_scrm("outputs/.../best", "cuda")
m.rank(instruction, state, ["cand a", "cand b"])   # -> [{index, reward}, ...] sorted by reward desc, index = input position
m.pairwise_probability(r_i, r_j, tau=1.0)         # sigmoid((r_i - r_j)/tau)
```
`python -m scrm.export --ckpt DIR --out OUT [--merge]` copies the checkpoint (or merges LoRA into a bf16 backbone) for serving.

## Design choices

* **Backbone = Qwen3.5-4B text decoder.** `Qwen/Qwen3.5-4B` is a vision-language checkpoint (`Qwen3_5Model` = `visual` +
  `language_model`). It is loaded with `AutoModel.from_pretrained`, the vision tower (0.33B) is deleted and only the
  `Qwen3_5TextModel` (4.21B, hidden 2560, 32 layers) is kept. Layers are hybrid: 3 of every 4 are Gated DeltaNet
  (linear attention, projections `in_proj_qkv`, `in_proj_z`, `out_proj`), every 4th is gated full attention
  (`q/k/v/o_proj`). LoRA uses `target_modules: all-linear` (every `nn.Linear` of the decoder: both layer kinds incl.
  `in_proj_a/b`, plus the MLP) with **rsLoRA** (`use_rslora: true`, scaling = alpha/sqrt(r); r=64, alpha=16 -> 2.0). Install `flash-linear-attention`
  (in `requirements-train.txt`) and `causal-conv1d` (prebuilt wheel pinned in `setup.sh`: Python 3.12, torch 2.10.0 + cu130); without them transformers
  falls back to a slow, memory-hungry torch implementation of the delta rule. Text-only inputs use plain 1D positions.
* **Liger kernels.** `model.liger_kernel=true` (default) applies Liger through its HF integration
  (`liger_kernel.transformers`, the same `apply_liger_kernel_to_qwen3_5` HF Trainer's `use_liger_kernel` calls), before the
  model is built: fused RMSNorm (incl. q/k norms) and SwiGLU MLP. Liger RoPE is not available for Qwen3.5 and
  fused-linear-cross-entropy is irrelevant (no LM head). CUDA only; silently skipped on CPU. The SCRM wrapper
  (`SCRM`, a plain `torch.nn.Module`) holds the patched HF backbone.
* **No new tokens.** The tokenizer and embedding matrix are used exactly as pretrained (nothing added or resized,
  no extra trainable embeddings); only LoRA + the set block/head train.
* **No LM head.** Only the text decoder is used, so the 248k-vocab logits never exist (saves several GB).
* **Precision.** Backbone bf16, LoRA weights fp32 (peft default), set block + head fp32 (autocast disabled there).
* **Input LayerNorm.** A `LayerNorm(d)` precedes the d->768 projection (`model.input_norm=false` to drop it) to
  tame Qwen's large-magnitude hidden dims.
* Causal attention means candidate *k* only sees candidates `< k` inside the backbone; the set transformer provides
  full bidirectional interaction. Candidate order is reshuffled every time in training.

## Memory: 24 GB vs 40 GB

Context: `data.render.max_len` is the per-sequence context (prompt + graded option): **8192** on 24 GB, **16384** on
40 GB (state up to 6k / 12k tokens).

Shared-prompt KV cache (`model.prefix_cache: true`, default; `src/scrm/prefix_cache.py`): all graded options of a set
share the prompt up to "Grade this choice: ". It is encoded once; its per-layer states (K/V after RoPE for the
full-attention layers, last conv inputs + final recurrent state for the Gated DeltaNet layers) are then reused by every
graded option, whose short suffixes run as one batch. A set costs ~ prompt + n_graded x suffix tokens instead of
n_graded x prompt. It is exact (tested against re-encoding every full sequence, forward and LoRA gradients, with
gradient checkpointing) and trains end to end: gradients from every option flow back into the shared prompt. HF's own
cache classes update buffers in place and are dropped under gradient checkpointing, so two small side-effect-free cache
objects are used instead. The suffix pass needs explicit 4D masks, so full attention uses SDPA (`attn_implementation:
auto` picks it). `data.render.max_graded` (8 on 24 GB, 12 on 40 GB) bounds options graded per set in training;
`data.batch.max_tokens_per_batch` is the token budget per micro-batch (suffix rows are chunked to fit it).

The Qwen3.5-4B text decoder has 4.21B params (8.4 GB in bf16). LoRA r=64 all-linear = ~130M params: fp32 weights + grads + Adam
= ~2 GB. Gradient checkpointing keeps ~0.16 MB/token of layer inputs (32 layers x 2560 x bf16) plus one layer of
recompute activations (8k tokens: ~1.3 GB saved inputs + a few hundred MB per recomputed layer). With an 8k-token
prompt per micro-batch the estimated peak is roughly 16-20 GB (NOT measured on a GPU; SDPA does not materialise
attention matrices).

| config | max_len | tokens / micro-batch | est. peak | notes |
|---|---|---|---|---|
| `scrm_qwen3_5_4b_24gb.yaml` | 8192 | 8192 | ~16-20 GB | bf16 base + LoRA, prefix cache |
| same + `model.quantize_4bit=true` | 8192 | 8192 | ~11-14 GB | QLoRA (nf4), a bit slower, needs bitsandbytes |
| `scrm_qwen3_5_4b_40gb.yaml` | 16384 | 16384 | ~26-34 GB | |
| `ablation_frozen.yaml` | 8192 | see config | <16 GB | no grads through backbone |

Knobs if you hit OOM: lower `data.batch.max_tokens_per_batch` (micro-batch) and raise `train.grad_accum`; lower
`data.render.max_len` / `cand_max_tokens`; `model.quantize_4bit=true`. Batches are token-budgeted (examples sorted by length within a
`bucket_size` pool), so micro-batch example counts vary; the loss is normalised by the number of examples with a trainable pair
over the whole accumulation window, so this does not bias the gradient. Install flash-attn (`setup.sh --flash-attn`) for
extra speed only with `model.prefix_cache=false` (the prefix-cache suffix pass needs SDPA's explicit masks).

## Data: filters, mixing, truncation

`data.local_dir` (a snapshot with `data/{train,validation,test}-*.parquet`) or `data.repo` (read through `hf://`). Only the
needed columns are read; `candidate_rows` is read only for eval (canonical order). Two modes:

* `data.streaming=false` (recommended; run `prefetch_data.sh`): arrow-backed, vectorised filtering, exact per-group caps,
  true shuffling. Costs extra disk for the arrow cache.
* `data.streaming=true`: no arrow cache; per-group filtering rescans the parquet files (use `files:` per group to restrict a
  group to e.g. `data/train-<source_slug>-*.parquet`), shuffle buffer `data.shuffle_buffer`, caps are approximate (first
  `max_rows` after the shuffle buffer).

Filters (`data.filters` for train, `data.eval_filters` for validation; keys `source_id, family, label_kind, source_split`):

```yaml
data:
  filters: {include: {label_kind: [source_dataset_label, ordinal_score_distance_tiers]}, exclude: {family: [foo]}}
  eval_filters: {exclude: {source_split: [ood]}}      # default: keep ood out of validation
```

Mixing groups (first match wins; `weight` is the per-example sampling probability weight; unmatched rows form group `other`
with `other_weight`; `drop_unmatched: true` drops them). Groups are re-iterated (new shuffle) when exhausted, so small groups
are up-sampled according to their weight:

```yaml
data:
  groups:
    - {name: massive,  match: {source_id: [AmazonScience/massive]}, max_rows: 150000, weight: 0.15}   # 60 candidates/row: cap it
    - {name: agent,    match: {family: [agent_decisions]},           weight: 0.25}
    - {name: typed,    match: {label_kind: [ordinal_score_distance_tiers]}, weight: 0.2}
  other_weight: 0.4
```

Rendering/truncation (`data.render`): `max_len` tokens per sequence = prompt + graded option (8192 for 24 GB, 16384 for 40 GB); `instr_max_tokens`,
`cand_max_tokens` (head-truncated per candidate), `state_max_tokens`, `state_truncate: middle|left|right` (middle keeps head+tail
and inserts `...`), `min_state_tokens`. If the candidates do not fit, unprotected candidates are dropped first (all tier-0 items
and one item per other tier are protected), then per-candidate tokens shrink; examples with no trainable pair left are dropped.
More than `max_candidates` (64): subsample keeping all tier-0 items and at least one per other tier. `instruction_json`:
string as-is, or object -> `instructions` + `Criteria:` lines. `state_json`: string as-is, object -> compact JSON.

## Loss

Default: Bradley-Terry only, `softplus(-(r_i - r_j - margin_ij)/tau)` over pairs with `tier_i < tier_j`, **mean over the pairs
of each example, then mean over examples** (no bias to sets with many pairs). Options (all off by default):
`loss.margin_alpha` (margin = alpha*|tier_i - tier_j|), `loss.tau`, `loss.w_listwise` (softmax CE on tier-0 set),
`loss.w_plackett_luce` (tier-wise Plackett-Luce), `loss.w_center` (mean reward^2, pins the arbitrary offset),
`loss.w_perm` (MSE between rewards of two shuffles of the same set; costs a second forward, `loss.perm_detach=true` runs it
without grad).

## Optimisation

AdamW, groups: LoRA `lr_lora` (1e-4), set block/head `lr_head` (5e-4); linear warmup
+ cosine to `min_lr_ratio`; clip 1.0; bf16 autocast; `grad_accum`. Length via `train.max_steps` or `train.epochs`
(epochs = passes over the capped group sizes; needs non-streaming; steps are estimated from the average batch size).
Checkpoints every `save_every` steps (`step_XXXXXXX/`, last `keep_last` kept) and `best/` (lowest validation loss; resumable, saved with optimizer state):
`adapter/` (LoRA), `scrm_head.pt` (set block, head), `tokenizer/`, `scrm_config.json`,
`trainer_state.pt` (optimizer, scheduler, step). `--resume auto|DIR` restores weights, optimizer, scheduler and step; the data
stream restarts with a different seed (it is not replayed exactly).

## Metrics (train log, `metrics.jsonl`, W&B)

Train: `train/loss` (+ parts), `lr/*`, `grad_norm`, `tokens_per_s` (real / padded), `pairs_per_step`, `examples_per_step`, running
`pair_acc_lastmb`, `reward_mean/std`, `sys/gpu_mem_alloc_gb`, `train/reward_hist` (W&B).
Eval (`eval/{all,family/<f>,source/<s>}/*`, plus a `eval/by_group` W&B table), computed on a deterministic capped validation
sample (`eval_max_rows`, `eval_max_rows_per_source`), canonical candidate order:

* `pair_acc`: per-example fraction of tier pairs with higher reward for the better tier (ties = 0.5), averaged over examples;
  `pair_acc_micro`: pooled over all pairs. Random = 0.5.
* `top1`: the top-scored candidate is in tier 0 (ties share credit). `mrr`: 1/rank of the best-ranked tier-0 item.
* `ndcg`: gains `2^(max_tier - tier) - 1`, log2 discount. `kendall_tau`: tau-b between reward and tier order, only for sets with
  >= 3 tiers (ordinal score rows).
* `loss`: BT loss. `eval/perm/*` (small subset, two shuffles): `score_std` / `abs_diff` (score change under reordering),
  `rank_agree` (pairwise order agreement), `top1_agree`. Should approach 0 / 1 as training progresses.

## Ablations

* `configs/ablation_frozen.yaml`: frozen Qwen3.5 (no LoRA); only set block + head learn.
* `configs/ablation_no_set.yaml`: `model.set_layers=0` -> per-candidate scoring, no interaction.
* Loss variants: `loss.w_listwise=1 loss.w_bt=0`, `loss.margin_alpha=0.5`, `loss.w_plackett_luce=...`.
* Other: `model.d_set`, `model.lora.r`, `data.render.max_candidates`, MASSIVE cap/weights.
