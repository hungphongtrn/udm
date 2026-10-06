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
(`tier_i < tier_j`, never same-tier). `model.head: linear` (configs/peek_qwen3_5_4b_h200.yaml) replaces the set encoder
with a single shared `nn.Linear(hidden, 1)` applied per candidate (`r_i = w^T h_i + b`, raw rewards, no
cross-candidate interaction in the head); head weights in `scrm_head.pt` are keyed by module name (`set_encoder.` /
`reward_head.`) and the head type is rebuilt from the saved `scrm_config.json`. Data contract: `docs/CONTRACT.md`.

Code: `src/scrm/` (`model.py`, `losses.py`, `render.py`, `collator.py`, `data.py`, `gradcache.py`, `metrics.py`, `train.py`,
`evaluate.py`, `export.py`, `wandb_utils.py`). Everything is plain PyTorch; single GPU by default, multi-GPU via
`torchrun` (manual all-reduce, no `DistributedDataParallel`), with optional gradient caching on large-memory cards.

## Quick start

```bash
scripts/train/setup.sh [--no-login]           # uv sync (data+train+cu128 groups; CU=130 for cu130), stack check, hf + wandb login; re-run after a pull that changes uv.lock
scripts/train/prefetch_data.sh data_cache/udm  # snapshot parquet (hf_xet) -> training starts immediately / offline
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
`uv run python -m scrm.export --ckpt DIR|hf://... --out OUT [--merge]` copies the checkpoint (or merges LoRA into a bf16 backbone) for serving.

## Design choices

* **Backbone = Qwen3.5-4B text decoder.** `Qwen/Qwen3.5-4B` is a vision-language checkpoint (`Qwen3_5Model` = `visual` +
  `language_model`). It is loaded with `AutoModel.from_pretrained`, the vision tower (0.33B) is deleted and only the
  `Qwen3_5TextModel` (4.21B, hidden 2560, 32 layers) is kept. Layers are hybrid: 3 of every 4 are Gated DeltaNet
  (linear attention, projections `in_proj_qkv`, `in_proj_z`, `out_proj`), every 4th is gated full attention
  (`q/k/v/o_proj`). LoRA uses `target_modules: all-linear` (every `nn.Linear` of the decoder: both layer kinds incl.
  `in_proj_a/b`, plus the MLP) with **rsLoRA** (`use_rslora: true`, scaling = alpha/sqrt(r); r=64, alpha=16 -> 2.0). Install `flash-linear-attention`
  (`train` dependency group) and `causal-conv1d` (prebuilt wheel locked in `uv.lock`: Python 3.12, torch 2.10.0 + cu128, or cu130 with `CU=130`); without them transformers
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

Context: `data.render.max_len` is the per-sequence context (prompt + graded option): **4096** on 24 GB, **16384** on
40 GB. Nothing is ever truncated: a set whose longest sequence (state + instruction + all options + one graded option)
exceeds `max_len` is dropped whole.

Every graded option is its own full sequence (`[prompt][Grade this choice: Option k: ...]<assistant header>`), so a set
costs ~ n_graded x (prompt + option) tokens; `data.render.max_graded` (8 on 24 GB, 12 on 40 GB) bounds options graded
per set in training (eval / `rank()` / Decision Index grade every option).

Packing (padding-free, `src/scrm/packing.py`): whole sets are best-fit-decreasing packed into micro-batches of at most
`data.batch.max_tokens_per_batch` tokens and `max_batch_size` sets (a set never spans two micro-batches; a set larger
than the budget is a micro-batch of its own). All sequences of a micro-batch are concatenated into one row with
per-sequence `position_ids`; on CUDA with `flash-linear-attention` + `causal-conv1d` the backbone gets
`cu_seq_lens` / `seq_idx`, so full attention is block-diagonal and the Gated DeltaNet conv/recurrent state restarts at
each sequence boundary (no leakage; tested against the padded batch). The block-diagonal attention needs `flash-attn`
(in the `cu128` / `cu130` groups; `model.attn_implementation: auto` picks it): under `sdpa`, HF materialises a dense
`[T, T]` mask per chunk (a 256k-token chunk = 64 GiB) and computes full `T^2` attention. `model.embed` runs the row in chunks of at most
`max_tokens_per_batch` tokens, cutting only between sequences (a sequence is never split); the set encoder then sees
all options of a set together. Without those kernels (CPU) the same sequences run as a right-padded batch.

The Qwen3.5-4B text decoder has 4.21B params (8.4 GB in bf16). LoRA r=64 all-linear = ~130M params: fp32 weights + grads + Adam
= ~2 GB. Gradient checkpointing keeps ~0.16 MB/token of layer inputs (32 layers x 2560 x bf16) plus one layer of
recompute activations.

| config | max_len | tokens / micro-batch | notes |
|---|---|---|---|
| `scrm_qwen3_5_4b_24gb.yaml` | 4096 | 4096 | bf16 base + LoRA (8192 OOMs on a 4090) |
| same + `model.quantize_4bit=true` | 4096 | 4096 | QLoRA (nf4), a bit slower, needs bitsandbytes |
| `scrm_qwen3_5_4b_40gb.yaml` | 16384 | 16384 | |
| `scrm_qwen3_5_4b_h200.yaml` | 32768 | 262144 group | 2x H200, gradient caching (8192 chunks, selective checkpointing), DDP |
| `ablation_frozen.yaml` | see config | see config | no grads through backbone |

Knobs if you hit OOM: lower `data.batch.max_tokens_per_batch` (micro-batch) and raise `train.grad_accum`; lower
`data.render.max_len` (more sets dropped); `model.quantize_4bit=true`. Micro-batch set counts vary; the loss is normalised
by the number of examples with a trainable pair over the whole accumulation window, so this does not bias the gradient.

## Gradient caching (`train.grad_cache`)

Off by default; the standard path above is unchanged. With `train.grad_cache=true` a loader micro-batch is a "group":
`data.batch.max_tokens_per_batch` becomes the group budget (whole sets, can be large) and the backbone activation memory
is bounded by the new knob `train.grad_cache_chunk_tokens` (default 16384). Each group is a three-pass step
(`src/scrm/gradcache.py`):

1. **Pass 1** — the backbone runs under `torch.no_grad()` (same autocast as always) over chunks of at most
   `grad_cache_chunk_tokens` tokens (`pack_bfd` over the sequences; a sequence longer than the budget gets its own
   chunk). Only the last-token embedding of each sequence is kept, scattered (fp32) into `e[M, d]` in original sequence
   order. The RNG state (CPU + CUDA) is recorded before each chunk.
2. **Pass 2** — `e.detach().requires_grad_()` feeds the set block/head (`SCRM.head`); the loss is exactly the standard one
   (`compute_loss` + `reduce_loss` over the global number of valid sets) and `loss.backward()` yields the set-head grads
   and `g = e.grad`.
3. **Pass 3** — if the backbone is trainable, each chunk is re-encoded **with grad** after restoring its pass-1 RNG state,
   then `torch.autograd.backward(h_chunk, g[chunk])`. The graph is freed per chunk. After pass 3 the RNG state is restored
   to the value after pass 2, so the stream advances exactly as one standard forward would.

With a chunk budget >= the whole group this is bit-for-bit the standard path (single chunk, original order). Chunking
never changes the result: sequences are independent, so regrouping only changes which tensors share a forward (tested,
including dropout replay). `loss.perm_detach=true` (default) keeps the second-shuffle forward under `no_grad` as before;
`loss.perm_detach=false` caches both shuffle packs through all three passes. A frozen backbone (`model.freeze_backbone`
/ `ablation_frozen.yaml`) simply skips pass 3. Under DDP the group budget is per rank and the loss is divided by the
globally summed valid-set count.

**Selective checkpointing (`train.grad_cache_act_tokens`).** A sequence longer than the chunk budget gets its own
chunk, so the chunk never bounds memory below `data.render.max_len`, and turning `model.gradient_checkpointing` off
outright OOMs (a 16k-token chunk of Qwen3.5-4B with LoRA all-linear stores > 136 GB, ~8.3 MB/token). Instead keep
checkpointing on and set `grad_cache_act_tokens`: in pass 3 a chunk of T tokens runs `floor(L * act_tokens / T)` of its
L decoder layers (evenly spaced, so the linear/full attention mix is kept) without checkpointing and recomputes only the
rest, so stored activations stay ~`act_tokens` tokens' worth for any chunk length. With `grad_cache_chunk_tokens` =
`act_tokens`, every packed chunk runs with no recompute and only sequences longer than the budget are partly
checkpointed. Pass 1, evaluation and the Decision Index are unaffected (all layers return to checkpointed after pass 3).
Grads are identical to the standard path (tested with LoRA dropout replay). Tune by `mem=` in the `[train]` line
(`torch.cuda.max_memory_allocated`, rank 0): raise `act_tokens` while the peak leaves headroom.

**Cross-rank balancing (DDP).** Set cost scales with graded candidates × prompt length, so per-rank groups are very
uneven and one rank idles at the gradient all-reduce. In grad-cache mode the backbone work is therefore balanced at
sequence granularity: each rank's pack token ids / positions / lengths are all-gathered into one pool, every sequence is
assigned longest-first to the least-loaded rank by tokens (`balance_plan`; ties keep it on its owner), each rank encodes
its assigned sequences in chunks, and the embeddings are exchanged with one `all_reduce(SUM)` over a zero-filled fp32
`[pool, d]` buffer. Pass 2 runs on the owning rank (its own sets only); the embedding grads go back the same way and
pass 3 re-encodes on the encoding rank. Backbone grads thus accumulate on whichever rank encoded a sequence, and the
existing SUM all-reduce of the trainable grads yields the same gradient (tested against a single process, including a
skewed split). The set head itself and `perm_detach=true` second-shuffle scoring stay rank-local.

## Multi-GPU (`torchrun`)

`NPROC=2 scripts/train/train.sh configs/scrm_qwen3_5_4b_h200.yaml ...` launches
`python -m torch.distributed.run --standalone --nproc_per_node 2 -m scrm.train ...`; `NPROC` unset/1 runs the plain
single-process command (identical behaviour). DDP is manual — no `DistributedDataParallel`, since grad caching calls the
backbone directly and backwards many times:

* Process group from `WORLD_SIZE`/`RANK`/`LOCAL_RANK` (nccl on CUDA, gloo on CPU), device `cuda:LOCAL_RANK`, timeout
  `train.ddp_timeout_min` (default 240) because rank 0 alone runs eval / test / Decision Index while the others wait.
* After the model is built (and after a resume load) rank 0 broadcasts all trainable weights (LoRA + set block/head).
* Each rank accumulates local grads; before clipping one coalesced `all_reduce(SUM)` over all trainable params' grads
  (flattened per dtype; params without a grad contribute zeros) makes every rank's grads identical. Each rank's loss is
  divided by the global valid-set count (all-reduced before the backward passes), so SUM-reduced grads are the gradient
  of the mean over all ranks' valid sets.
* Data is sharded: non-streaming groups split their indices by global worker id `rank*num_workers + wid` with
  `world*num_workers` shards; streaming groups `shard(world, rank)` before HF's per-worker sharding. `train.epochs`
  estimates steps over the global dataset (divided by world size).
* Rank 0 only: eval/test/Decision Index, checkpoint saves + HubSync, W&B, `metrics.jsonl`, prints. `train/loss`,
  `pairs_per_step`, `examples_per_step`, `tokens_per_s` are SUM-reduced across ranks; per-source and loss-part metrics
  stay rank-local. "Data exhausted" and the early-stop decision are agreed with `all_reduce` (MIN / MAX) so no rank hangs.
* `grad_cache` and the standard path both work under DDP; only grad cache balances backbone work across ranks.

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

Rendering (`data.render`): `max_len` tokens per sequence = prompt + graded option. No truncation of state, instruction or
options: an over-long set is dropped. More than `max_candidates` (64) options: subsample keeping all tier-0 items and at
least one per other tier (training/validation sets only; inference grades every option). Training sets need >= 2 tiers among
the graded options. `instruction_json`: string as-is, or object -> `instructions` + `Criteria:` lines. `state_json`: string
as-is, object -> compact JSON.

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
`trainer_state.pt` (optimizer, scheduler, step). `--resume auto|DIR|hf://...` restores weights, optimizer, scheduler and step; the data
stream restarts with a different seed (it is not replayed exactly). Validation also runs at step 0 (`train.eval_at_start`,
default on) as the untrained baseline.

HF backup (`hub.repo_id`, set in the 24 GB config to the private `hungphongtrn/scrm-qwen3_5-4b`): after every save the
checkpoint is uploaded in a background thread to `<run>/best` or `<run>/last` (`<run>` = `hub.run` or the basename of
`output_dir`), replacing that folder; each upload is one commit, so older ones stay reachable as
`hf://repo@<commit>/<run>/best`. Only the LoRA adapter, set block/head, config and tokenizer are uploaded (the base model
is re-downloaded from `model.name_or_path` on load); `hub.include_optimizer=true` adds `trainer_state.pt` for
`--resume hf://...`. The next save waits for the previous upload; failures are printed, never raised.

## Metrics (train log, `metrics.jsonl`, W&B)

Train: `train/loss` (+ parts), `lr/*`, `grad_norm`, `tokens_per_s` (packed, no padding), `pairs_per_step`, `examples_per_step`, running
`pair_acc_lastmb`, `reward_mean/std`, `sys/gpu_mem_alloc_gb`, `train/reward_hist` (W&B), and per source
`train/source/<s>/{loss,n_sets}` (mean per-set loss over the sets seen since the previous log line).
Eval (`eval/{all,family/<f>,source/<s>}/*`, plus a `eval/by_group` W&B table), computed on a deterministic validation
sample: `eval_max_rows_per_source` rendered sets drawn at random (seeded) from every source in the split, rows that do not
render (over `max_len`) replaced by further draws (up to 10x the quota), then at most `eval_max_rows` in total. Only the
filter columns of every split file are scanned for the draw, so all sources are covered whatever the file order; the
log line `[eval] set:` shows sets per source and dropped draws. Canonical candidate order:

* `pair_acc`: per-example fraction of tier pairs with higher reward for the better tier (ties = 0.5), averaged over examples;
  `pair_acc_micro`: pooled over all pairs. Random = 0.5.
* `top1`: the top-scored candidate is in tier 0 (ties share credit). `mrr`: 1/rank of the best-ranked tier-0 item.
* `ndcg`: gains `2^(max_tier - tier) - 1`, log2 discount. `kendall_tau`: tau-b between reward and tier order, only for sets with
  >= 3 tiers (ordinal score rows).
* Calibration of the set softmax (the probabilities `rank()` and the Decision Index use; tracking only, not trained
  on): `conf` = mean top-1 probability, `overconf` = `conf - top1` (> 0: overconfident), `ece_top1` = expected
  calibration error of top-1 probability vs top-1 correctness (10 equal-width bins), `brier_top1`, `p_best` =
  probability mass on the best tier, `nll_best` = `-log p_best`.
* `loss`: BT loss. `eval/perm/*` (small subset, two shuffles): `score_std` / `abs_diff` (score change under reordering),
  `rank_agree` (pairwise order agreement), `top1_agree`. Should approach 0 / 1 as training progresses.
* `skill` (per group) = `clip((top1 - chance) / (1 - chance), 0, 1)`, `chance` = mean fraction of options in the best
  tier (top-1 of a random pick): the Decision Index's chance correction. `eval/index` = 100 x mean `skill` over sources
  (every source weighted equally, like the Decision Index's areas), `eval/raw_index` = 100 x mean source `top1`.

Checkpoint selection (`best/`) and early stopping run on validation at every eval, `train.best_metric`: `val_index`
(default, max `eval/index`) or `val_loss` (min `eval/loss`); `early_stop_patience` counts evals. A checkpoint resumes only
under the `best_metric` it was trained with. `train.keep_best: k` (> 0) also keeps the top-k evals as `best_step_N`
(weights only; `best` stays the resumable top-1). At save steps (and at the end) two test-only evaluations are logged,
never used for selection:

* `test/*`: the same metrics on `benchmarks.test_split` (default `test`, same filters and row caps as validation), incl.
  `test/index`, `test/raw_index`.
* `dindex/*`: [Decision Index](https://github.com/apolinario/decision-index) 0.2.1 on a fixed stratified sample of its
  suite (`benchmarks.decision_index`; enabled in the Qwen configs, off in the defaults): `index` (chance-corrected, the
  board's headline), `raw_index`, `answered_frac`, `area/*` and `bench/*` skill (x100), scored with the kit's own
  scorers restricted to the sampled requests. Each question is one SCRM set (instruction + state + every option; `noul`
  -> `no`/`yes`), probabilities = softmax of the rewards; requests longer than `max_len` (16384) are `unsupported`
  (= wrong; none in the 1000-row sample). Runs only at save steps divisible by `every` (2000) and at the end: the
  1000-row sample is ~40M tokens (every option is a full sequence containing all options; POP909 alone is 39%).
  Results per step in `<output_dir>/decision_index/step-N/`. Setup once: `scripts/train/dindex_setup.sh` (rebuilds the
  suite into `../decision-index`, ~7 GB of downloads, needs the cais/hle terms accepted; writes
  `sample-1000.jsonl.gz`, the paths the configs expect). The sample index is an estimate; for a number comparable with
  the board run the full suite: `scripts/train/dindex_eval.sh <ckpt>` (official runner, `scrm.dindex:SCRMEngine`,
  prints the rank among the board entrants).

## Ablations

* `configs/ablation_frozen.yaml`: frozen Qwen3.5 (no LoRA); only set block + head learn.
* `configs/ablation_no_set.yaml`: `model.set_layers=0` -> per-candidate scoring, no interaction.
* `configs/peek_qwen3_5_4b_h200.yaml`: `model.head=linear` (Peek: shared `nn.Linear(hidden, 1)`, no set block) with the
  v2 H200 recipe minus the listwise softmax (BT only) — a one-variable-group comparison against
  `configs/scrm_qwen3_5_4b_h200.yaml` (v2).
* Loss variants: `loss.w_listwise=1 loss.w_bt=0`, `loss.margin_alpha=0.5`, `loss.w_plackett_luce=...`.
* Other: `model.d_set`, `model.lora.r`, `data.render.max_candidates`, MASSIVE cap/weights.
