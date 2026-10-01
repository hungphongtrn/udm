# SCRM training guide

Set-Conditioned Reward Model: a Qwen3-4B backbone (+LoRA) reads one packed sequence
`<|state_start|> instruction + state <|state_end|><|candidate_set_start|> <|candidate_start|> text <|candidate_end|> ... <|candidate_set_end|>`.
The hidden state at every `<|candidate_end|>` is projected (d -> 768), passed through a 2-layer bidirectional
pre-LN transformer encoder **without positional embeddings** (so scores do not depend on candidate order), and an MLP head
returns one unbounded scalar reward per candidate. Training uses Bradley-Terry over tier pairs
(`tier_i < tier_j`, never same-tier). Data contract: `docs/CONTRACT.md`.

Code: `src/scrm/` (`model.py`, `losses.py`, `render.py`, `collator.py`, `data.py`, `metrics.py`, `train.py`,
`evaluate.py`, `export.py`, `wandb_utils.py`). Everything is plain PyTorch, single GPU.

## Quick start

```bash
scripts/train/setup.sh [--flash-attn]          # venv, deps, hf + wandb login
scripts/train/prefetch_data.sh data_cache/udm  # snapshot parquet (hf_transfer) -> training starts immediately / offline
scripts/train/train.sh configs/scrm_qwen3_4b_24gb.yaml data.local_dir=data_cache/udm
scripts/train/train.sh configs/scrm_qwen3_4b_24gb.yaml --resume auto data.local_dir=data_cache/udm
scripts/train/eval.sh outputs/scrm_qwen3_4b_24gb/best --split test --out test.json data.local_dir=data_cache/udm
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

* **Special tokens.** Six markers are added with `tokenizer.add_special_tokens`. Qwen3's embedding matrix has 151936 rows but
  the tokenizer only 151669 tokens, so the new ids (151669-151674) land in existing unused rows: **no embedding resize**.
  Instead of `modules_to_save` (which would copy and train the full 151936x2560 matrix: ~0.8 GB + fp32 grads/Adam),
  the six rows are replaced by a small trainable `[6, d]` fp32 parameter via a forward hook on `embed_tokens`
  (init = mean embedding + small seeded noise). It works with LoRA, frozen backbone and 4-bit loading, costs ~15k params,
  and is saved in `scrm_head.pt`. If a tokenizer has fewer rows than ids, the embeddings are resized automatically.
* **No LM head.** `AutoModel` (Qwen3Model) is used, so the 151k-vocab logits never exist (saves several GB).
* **Precision.** Backbone bf16, LoRA weights fp32 (peft default), set block + head fp32 (autocast disabled there).
* **Input LayerNorm.** A `LayerNorm(d)` precedes the d->768 projection (`model.input_norm=false` to drop it) to
  tame Qwen's large-magnitude hidden dims.
* Causal attention means candidate *k* only sees candidates `< k` inside the backbone; the set transformer provides
  full bidirectional interaction. Candidate order is reshuffled every time in training.

## Memory: 24 GB vs 40 GB

Qwen3-4B has 4.02B params (8.0 GB in bf16). LoRA r=64 on all 7 projections = 132M params: fp32 weights + grads + Adam
= ~2.1 GB. Gradient checkpointing keeps ~0.2 MB/token of layer inputs (36 layers x 2560 x bf16 = 0.18 MB) plus one layer of
recompute activations. Hence at `max_tokens_per_batch=4096` (padded tokens per micro-batch) the estimated peak is
roughly 14-17 GB (not measured on a GPU in this repo's CI; sdpa/flash-attn does not materialise attention matrices).

| config | max_len | tokens / micro-batch | est. peak | notes |
|---|---|---|---|---|
| `scrm_qwen3_4b_24gb.yaml` | 2048 | 4096 | ~14-17 GB | bf16 base + LoRA |
| same + `model.quantize_4bit=true` | 2048 | 4096 | ~9-11 GB | QLoRA (nf4), a bit slower, needs bitsandbytes |
| `scrm_qwen3_4b_40gb.yaml` | 4096 | 8192 | ~26-32 GB | |
| `ablation_frozen.yaml` | 2048 | 16384 | <14 GB | no grads through backbone |

Knobs if you hit OOM: lower `data.batch.max_tokens_per_batch` (micro-batch) and raise `train.grad_accum`; lower
`data.render.max_len` / `cand_max_tokens`; `model.quantize_4bit=true`. Batches are token-budgeted (examples sorted by length within a
`bucket_size` pool), so micro-batch example counts vary; the loss is normalised by the number of examples with a trainable pair
over the whole accumulation window, so this does not bias the gradient. Install flash-attn (`setup.sh --flash-attn`) for
extra speed (`model.attn_implementation=auto` picks it up).

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

Rendering/truncation (`data.render`): `max_len` total tokens (2048 for 24 GB, 4096 for 40 GB); `instr_max_tokens`,
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

AdamW, groups: LoRA `lr_lora` (1e-4), set block/head `lr_head` (5e-4), marker embeddings `lr_special` (5e-4); linear warmup
+ cosine to `min_lr_ratio`; clip 1.0; bf16 autocast; `grad_accum`. Length via `train.max_steps` or `train.epochs`
(epochs = passes over the capped group sizes; needs non-streaming; steps are estimated from the average batch size).
Checkpoints every `save_every` steps (`step_XXXXXXX/`, last `keep_last` kept) and `best/` (by validation `pair_acc`):
`adapter/` (LoRA), `scrm_head.pt` (set block, head, marker embeddings), `tokenizer/`, `scrm_config.json`,
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

* `configs/ablation_frozen.yaml`: frozen Qwen3 (no LoRA, no marker-embedding training); only set block + head learn.
* `configs/ablation_no_set.yaml`: `model.set_layers=0` -> per-candidate scoring, no interaction.
* Loss variants: `loss.w_listwise=1 loss.w_bt=0`, `loss.margin_alpha=0.5`, `loss.w_plackett_luce=...`.
* Other: `model.d_set`, `model.lora.r`, `data.render.max_candidates`, MASSIVE cap/weights.
