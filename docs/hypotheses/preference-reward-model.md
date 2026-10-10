# Hypothesis: preference reward model on Kev's recipe

Branch `hyp/preference-reward-model`. Code: `kev/kev/pref.py`, `kev/kev/model.py` (ScalarHead, chat format),
`kev/kev/train.py`, `kev/scripts/pref/`, `kev/tests/test_pref.py`. Fork provenance: [`kev/UPSTREAM.md`](../../kev/UPSTREAM.md).
Everything below (arms, thresholds, rule) is fixed before any run.

## Claim

On Kev-4B's recipe, replacing the softmax pointer head + CE with an unbounded scalar reward head trained by
Bradley-Terry (BT) preferences (optionally regularized, then optionally adding BCE) yields a decision model at least as
accurate as Kev, with better-behaved reward scale / calibration and permutation robustness.

Prior: our earlier BT-only LoRA on our own SCRM stack scored 27.61 DI vs 36.33 for a frozen sigmoid head, and strong
peers use CE/Brier/softmax ([studies/README.md](../studies/README.md), [external_comparison.md](../studies/external_comparison.md)).
That run differed from Kev in backbone setup, head and data; this study tests BT in a controlled setting on Kev's recipe.

Sub-claims, one per comparison:

| Comparison | Isolates | Sub-claim |
|---|---|---|
| A vs B | loss (CE vs pref), pointer head, Kev format | BT over label-derived pairs matches CE top1 on the pointer head |
| B vs C | head + input format | scalar head on the chat format matches the pointer head on Kev format (C0 splits head from format) |
| C vs D / E | reward regularization | BSR / L2 bound the reward scale without costing accuracy |
| C vs F | Stage-2 absolute supervision | adding BCE makes sigmoid(r) a calibrated absolute probability without costing ranking |
| G | post-hoc calibration | a temperature / Platt fit on the calibration split fixes ECE without retraining |

## Arms

| ID | What changes | Flags |
|---|---|---|
| A | Kev pointer + CE (baseline, Kev recipe on bf16 weights) | `--head pointer --input_format kev --loss ce` |
| B | pointer head, pref loss | `--head pointer --input_format kev --loss pref` |
| C0 | (optional) scalar head on Kev format, reads `</opt>` | `--head scalar --input_format kev --loss pref` |
| C | scalar head + pref, chat format | `--head scalar --input_format chat --loss pref` |
| D | C + BSR (lambda 1e-3) | C + `--reg bsr --reg_w 1e-3` |
| E | C + L2 (lambda 1e-3) | C + `--reg l2 --reg_w 1e-3` |
| F | Stage 2: warm-start from C, pref + BCE | C + `--bce_w 0.5 --epochs 1` (init from C's final checkpoint; fresh OneCycleLR at lr 5e-5) |
| G | post-hoc temperature / Platt on the calibration split | no training; `scripts/calibrate_checkpoint.py` on A and C |
| H / I | isolated candidates | later, not in wave 1 |

Exact flag strings per arm live in `kev/scripts/pref/arms.sh` (source of truth if it differs from this table; update here).

### Chat input format (arms C-F)

No special tokens beyond what the tokenizer's own chat template emits. Per (question, option k) one user message, rendered with
the Base tokenizer's chat template (`add_generation_prompt=True`, `enable_thinking=False`):

```
State:
{state}

Instruction: {instruction}

Options:
- {o_1}
- {o_2}
...

Grade this option: {o_k}
```

followed by the template's tail (for ChatML-style templates `<|im_end|>\n<|im_start|>assistant\n`, plus whatever the
Qwen3.5 template adds). The reward is the scalar head applied to the hidden state of the LAST token of that prompt.

Prefix caching: everything up to and including `Grade this option: ` is identical for the K options and runs once
(`kev.shared_prefix.branch_hidden`, exact with gradients); option k's branch is `tokens(o_k + template tail)`, positions
continuing the prefix. The pieces are tokenized separately (template head + `State:\n`, state tokens cut to the state limit,
the rest of the prefix, each branch), so the prefix ids are byte-identical across options in training and evaluation. The head/tail
strings come from rendering the template once around a sentinel; their sha256 is recorded in the checkpoint
(`chat_template_sha256`) and a mismatch at load is refused. A tokenizer without a chat template raises; none is invented.
`setup_3090.sh` prints the rendered example prompt and its token count to verify on the training machine. Limits: records are
admitted by Kev's own limits (same population in every arm); a chat row (prefix + branch) may exceed the Kev row limit by
`CHAT_ROW_SLACK` = 128 tokens. Arms A and B (and C0) keep Kev's own delimiter format: that is the baseline.

## Fixed across arms

- Base `Qwen/Qwen3.5-4B-Base` rev `1001bb4d826a52d1f399e183466143f4da7b741b`; LoRA r16, alpha 32, dropout 0.05.
- Data: decision-v7 train (12,576 records); epochs 2; lr 5e-5; batch 4 x accum 2; OneCycleLR; AdamW wd 0.01; grad clip;
  bf16 autocast; gradient checkpointing; `p_none_pair` 0.25; option permutation re-drawn each epoch.
- `--weights_dtype bf16` for every arm (deviation from the released recipe, for 24 GB on a 3090). So A is "Kev recipe on
  bf16 weights", not the released checkpoint.
- Final checkpoint only, no checkpoint selection. Seeds 0 first, then 1, 2.

## Defaults chosen (not specified by the requester)

- Hard label y: pairs (y > j) for all j != y with p = 1, i.e. `softplus(r_j - r_y)`.
- Soft target t: all pairs i < j with `p = t_i / (t_i + t_j)` (skips pairs with t_i + t_j = 0).
- Pref loss averaged per question, then aggregated like Kev's CE (per-record share).
- BSR: `lambda * mean(cat(rw, rl))^2`, lambda 1e-3, per micro-batch over pair members (a chosen option counts once per pair).
- L2: `lambda * mean(cat(rw, rl)^2)`, same members, lambda 1e-3.
- BCE: `bce_w * mean_i BCE(sigmoid(r_i), [i == y])`, hard-label questions only, types `choice,noul,score` (`--bce_types`).
- ScalarHead bias cancels in BT; it matters only with BSR / L2 / BCE.
- BT is shift-invariant, so softmax over rewards is the eval readout for all arms (the sigmoid readout is reported for D/E/F).

## Metrics

On decision-v7 development and transfer-v4 development (out-of-source):

- top1 accuracy, pairwise accuracy;
- NLL and ECE of the softmax readout; sigmoid-reward ECE (D/E/F);
- reward mean / std / |r| p99;
- permutation flip rate (`kev.benchmark` report);
- training reward stats from logs: `r_mean`, `r_std`, `r_absmax`, `pair_acc`.

Decision Index only for the final winner (cost: ~15-30 h per full-suite pass with the HF backend on one 3090 [INFERENCE],
see studies/README.md open items). Locked test split only for the final verdict.

## Decision rule

Set before any run.

1. Seed-0 screen of A-E. A top1 difference < 2 points on a single seed is noise (studies show seed spread +-0.4-1.5).
2. Run seeds 1-2 for A and for the best of B-E (by seed-0 mean of the two dev top1s).
3. **Supported**: best scalar-pref arm >= A - 0.5 top1 (3-seed mean) on both dev sets AND better calibration (lower ECE) or lower flip rate than A.
4. **Refuted**: best scalar-pref arm >= 2 points worse than A top1 on both dev sets.
5. Otherwise **inconclusive**.

F and G are judged against C only (sub-claims above) and do not enter the verdict unless they become the best arm; if so, repeat 3 seeds.

## Compute

~2-3 h per run on one 3090 [estimate]. Wave 1 = 5 arms (A-E) x seed 0. Full = 5 x 3 seeds + F + C0.

## Commands

Training machine only, from `kev/`:

```
uv run pytest tests/test_pref.py              # CPU; run on the training machine first
bash scripts/pref/setup_3090.sh               # env check; prints the chat template and an example prompt
bash scripts/pref/run_arm.sh ARM SEED [GPU]   # e.g. run_arm.sh C 0 0
bash scripts/pref/wave1.sh                    # A-E seed 0, 1 GPU
bash scripts/pref/wave1.sh 0,1                # 2 GPUs
bash scripts/pref/eval_arm.sh runs/pref/C/seed0
python scripts/pref/reward_report.py ...      # see scripts/pref/README.md
```

OOM fallback: `BATCH_FALLBACK=1` (batch 2 × accum 4, same effective batch 8; defined in `scripts/pref/arms.sh`).

## Risks / open

- Chat template availability on the Base tokenizer (`Qwen3.5-4B-Base`): if it has none, chat arms refuse to run (checked by `setup_3090.sh`).
- Separate tokenization at the prefix/branch boundary (`Grade this option: ` | `o_k`): token boundaries there differ from tokenizing the
  joined string; identical in training and evaluation, but not what a one-pass render would give.
- The chat format repeats the state per question (each question has its own prefix) and costs K extra short rows per question.
- A chat template written for instruct models on a Base model: the tail (assistant header) embeddings may be untrained; LoRA adapts them.
- bf16 weights vs released fp32: A may differ from the published Kev number.
- F's LR schedule restarts from the C checkpoint.
- BT is shift-invariant: only softmax probabilities are comparable across arms.
- DI validation top1 is a weak DI proxy in earlier work ([studies/README.md](../studies/README.md) conclusion 1).

## Results

| Arm | Seed | Run dir | Dev top1 | Transfer top1 | ECE | Flip rate | Notes |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

Verdict: pending
