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
(training: `kev.shared_prefix.branch_hidden`, exact with gradients; inference without gradients: `DecisionModel.chat_hidden`,
which also runs the head + state part once per record and continues it per question, the same tokens at the same positions,
so it agrees with training's computation up to kernel rounding); option k's branch is `tokens(o_k + template tail)`, positions
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
- Environment (deviates from upstream Kev): Python 3.12, torch 2.10.0 (cu128 default; cu130 optional), causal-conv1d 1.7.0 and
  flash-attn 2.8.3 prebuilt wheels, flash-linear-attention 0.5.2, triton 3.6 from torch; `kev/uv.lock` regenerated on the training
  machine. Same for every arm. See `kev/UPSTREAM.md`.
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
.venv/bin/python -m pytest tests/test_pref.py  # CPU; run on the training machine first
TORCH_EXTRA=cu128 bash scripts/pref/setup_3090.sh  # (cu130 for CUDA 13) uv lock + uv sync --extra; commit the generated kev/uv.lock; env check; prints the chat template and an example prompt
bash scripts/pref/run_arm.sh ARM SEED [GPU]   # e.g. run_arm.sh C 0 0
bash scripts/pref/wave1.sh                    # A-E seed 0, 1 GPU
bash scripts/pref/wave1.sh 0,1                # 2 GPUs
bash scripts/pref/eval_arm.sh runs/pref/C/seed0
.venv/bin/python scripts/pref/reward_report.py ...  # see scripts/pref/README.md
```

OOM fallback: `BATCH_FALLBACK=1` (batch 2 × accum 4 Kev, 1 × 8 chat; same effective batch 8; defined in `scripts/pref/arms.sh`).
If a chat arm still runs out at batch 1, add `ROW_BUDGET=2048`: `--row_budget` runs a record's questions in several
forward/backward passes (no record, question or token dropped; same gradient sum). D/E's `--reg` is computed per pass, so
use the same `ROW_BUDGET` for every D/E run and seed.

## Risks / open

- Chat template availability on the Base tokenizer (`Qwen3.5-4B-Base`): if it has none, chat arms refuse to run (checked by `setup_3090.sh`).
- Separate tokenization at the prefix/branch boundary (`Grade this option: ` | `o_k`): token boundaries there differ from tokenizing the
  joined string; identical in training and evaluation, but not what a one-pass render would give.
- The chat format repeats the state per question (each question has its own prefix) and costs K extra short rows per question.
- A chat template written for instruct models on a Base model: the tail (assistant header) embeddings may be untrained; LoRA adapts them.
- torch 2.10 / triton 3.6 vs Kev's released torch 2.8 numerics: A may differ slightly from the published number; identical for all arms.
- bf16 weights vs released fp32: A may differ from the published Kev number.
- F's LR schedule restarts from the C checkpoint.
- BT is shift-invariant: only softmax probabilities are comparable across arms.
- DI validation top1 is a weak DI proxy in earlier work ([studies/README.md](../studies/README.md) conclusion 1).

## Results

### Wave 1, seed 0 (A-F)

Code at `0d697fd`; archive `pref-AE.tgz` (logs, metrics, eval outputs; no weights). Final checkpoints, no selection. One
RTX 3090 per run. A/B trained 4 x accum 2; C-F 1 x accum 8 (`BATCH_FALLBACK=1`, no `ROW_BUDGET`); peak allocated 14.7 GB
(A/B), 19.7 GB (C-F). Wall time per run: A/B 2.4 h, C/D/E 9.1 h, F 4.6 h (1 epoch).

Clean rows: decision-v7 development n = 1264, transfer-v4 development n = 656. Raw = softmax over the run's logits; G =
temperature T fit on that run's decision-v7 calibration split (n = 1148), applied to both dev sets. Flip rate:
`report.json` permutation (60 questions on v7, 36 on t4: one flip = 1.7 / 2.8 points).

| Arm | Seed | Run dir | v7 top1 | t4 top1 | v7 ECE raw / G | t4 ECE raw / G | v7 NLL raw / G | T | Flip v7 / t4 | v7 \|r\| p99 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | runs/pref/A/seed0 | 87.74 | 78.51 | .064 / .025 | .135 / .070 | .459 / .348 | 1.85 | 0 / .083 | 15.8 | baseline |
| B | 0 | runs/pref/B/seed0 | 87.03 | 80.03 | .063 / .028 | .118 / .068 | .450 / .365 | 1.72 | .017 / .083 | 19.7 | |
| C | 0 | runs/pref/C/seed0 | 86.55 | 78.66 | .060 / .029 | .143 / .095 | .447 / .380 | 1.53 | .017 / .056 | 10.6 | |
| D | 0 | runs/pref/D/seed0 | 87.42 | 79.73 | .053 / .036 | .124 / .088 | .430 / .374 | 1.47 | 0 / .028 | 7.1 | best of B-E |
| E | 0 | runs/pref/E/seed0 | 86.55 | 77.74 | .046 / .038 | .133 / .114 | .415 / .403 | 1.16 | 0 / .028 | 4.0 | |
| F | 0 | runs/pref/F/seed0 | 86.16 | 77.74 | .092 / .045 | .172 / .096 | .697 / .403 | 2.75 | .017 / .028 | 18.2 | from C |

Paired top1 differences (same questions; 95% CI from 5,000 bootstrap resamples of records; test-set sampling only, seed
variance not included):

| Diff | v7 | t4 |
|---|---|---|
| B - A | -0.71 [-1.72, 0.24] | +1.52 [-1.08, 4.19] |
| C - A | -1.19 [-2.20, -0.08] | +0.15 [-2.19, 2.40] |
| D - A | -0.32 [-1.31, 0.70] | +1.22 [-1.21, 3.74] |
| E - A | -1.19 [-2.20, -0.23] | -0.76 [-2.69, 1.23] |
| F - A | -1.58 [-2.77, -0.38] | -0.76 [-3.30, 1.76] |
| D - C | +0.87 [0.00, 1.76] | +1.07 [-0.76, 3.07] |
| E - C | 0.00 [-0.87, 0.88] | -0.91 [-2.87, 1.07] |
| F - C | -0.40 [-1.44, 0.64] | -0.91 [-2.90, 1.03] |

Sigmoid-reward ECE (vs one-hot labels; raw / after Platt fit on v7 calibration): v7 C .063 / .010, D .061 / .010,
E .070 / .013, F .015 / .007; t4 C .159 / .054, D .091 / .063, E .077 / .066, F .107 / .067.

Readings (seed 0 only):

- Screen (rule 1): every B-E arm is within 2 top1 points of A on both dev sets: no arm is separated from A on one seed.
  Rule 2 picks D (mean of the two dev top1s: D 83.57, B 83.53, A 83.12, C 82.61, E 82.15). D is also the best scalar arm,
  the one rule 3 judges. B and D are tied for "best of B-E".
- A vs B (loss): BT on the pointer head is within noise of CE (v7 -0.7, t4 +1.5). B's t4 lead comes from
  `composition_held_conditional` (96.9 vs 68.8, n = 32) and `composition_held_and_or` (100 vs 90.6, n = 32).
- B vs C (head + format): C is 0.5 below B on v7 and 1.4 below on t4; C - A on v7 is -1.19 with a CI excluding 0 (test
  sampling only). C0 was not run, so head and format are not separated.
- C vs D / E (reward scale): BSR and L2 shrink the reward scale (v7 |r| p99 10.6 -> 7.1 / 4.0; std 4.7 -> 3.4 / 2.0)
  without costing top1 (D +0.9 / +1.1, E 0.0 / -0.9). Sub-claim holds on seed 0.
- Calibration: raw softmax ECE is lowest for E / D on v7 (E .046, D .053 vs A .064), but after G (temperature) A is best on
  both sets (v7 .025 vs D .036; t4 .070 vs D .088). Regularized arms start closer to calibrated (E's T = 1.16) and gain
  less from a temperature; the temperature fit on v7 transfers worse to t4 for C-F than for A/B.
- C vs F (BCE stage 2): F's sigmoid(r) is calibrated on v7 (ECE .063 -> .015; t4 .159 -> .107), top1 within noise of C
  (-0.4 / -0.9), but its softmax gets sharper (label margin 7.4 -> 13.4, T = 2.75, raw NLL .447 -> .697). Platt scaling on C
  (no retraining) reaches the same sigmoid ECE (v7 .010 vs F .007; t4 .054 vs F .067): G matches F on this metric.
- G: a temperature fit on v7 calibration lowers softmax ECE for every arm on both sets: about half for A, B, C and F on v7,
  less for the regularized arms (D .053 -> .036, E .046 -> .038).
- Flip rates are 0-1 flips of 60 (v7) and 1-3 of 36 (t4): too few questions to separate arms.
- Rule 3 preview for D (seed 0): top1 passes (v7 -0.32 >= -0.5; t4 +1.22). Calibration passes on raw ECE, fails on
  G ECE; flip rate ties on v7 (0 vs 0) and is lower on t4 (1 vs 3 of 36). Which ECE rule 3 uses is not fixed above;
  `scripts/pref/README.md` says to compare G numbers. This has to be decided before seeds 1-2 are read.

### Seeds 1-2 (A and D, rule 2)

Archive `pref-seeds12.tgz` (all wave-1 runs plus A/D seeds 1-2; its `COMMIT` file, written at pack time, reads `9bce974`,
which is not on `origin`). Same microbatch per arm as seed 0 (A 4 x 2, D 1 x 8 via `BATCH_FALLBACK=1`); each
`training_config.json` differs from its seed-0 run only in `seed` and `out`. Seed-0 numbers in the archive reproduce the
wave-1 table. Commands (from `kev/`, one GPU per line):

```
nohup bash -c 'BATCH_FALLBACK=1 scripts/pref/run_arm.sh D 1 0; scripts/pref/run_arm.sh A 1 0' > seeds-gpu0.log 2>&1 &
nohup bash -c 'BATCH_FALLBACK=1 scripts/pref/run_arm.sh D 2 1; scripts/pref/run_arm.sh A 2 1' > seeds-gpu1.log 2>&1 &
scripts/pref/pack_results.sh pref-seeds12.tgz
```

| Arm | Seed | v7 top1 | t4 top1 | v7 ECE raw / G | t4 ECE raw / G | v7 NLL raw / G | T | Flip v7 / t4 (count) | v7 \|r\| p99 |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 87.74 | 78.51 | .064 / .025 | .135 / .070 | .459 / .348 | 1.85 | 0 / 3 | 15.8 |
| A | 1 | 87.26 | 77.90 | .068 / .030 | .136 / .086 | .460 / .356 | 1.77 | 1 / 3 | 13.2 |
| A | 2 | 87.66 | 79.57 | .060 / .029 | .128 / .069 | .445 / .351 | 1.72 | 0 / 3 | 16.2 |
| D | 0 | 87.42 | 79.73 | .053 / .036 | .124 / .088 | .430 / .374 | 1.47 | 0 / 1 | 7.1 |
| D | 1 | 86.31 | 77.90 | .058 / .038 | .135 / .098 | .418 / .373 | 1.43 | 0 / 2 | 6.9 |
| D | 2 | 86.55 | 78.81 | .059 / .030 | .130 / .102 | .430 / .375 | 1.43 | 3 / 1 | 7.4 |
| **A mean** | | **87.55** (sd .25) | **78.66** (sd .85) | .064 / .028 | .133 / .075 | .455 / .352 | 1.78 | 1 / 9 of 180 / 108 | 15.0 |
| **D mean** | | **86.76** (sd .58) | **78.81** (sd .91) | .057 / .035 | .130 / .096 | .426 / .374 | 1.44 | 3 / 4 of 180 / 108 | 7.1 |

Paired D - A top1 (same questions; 95% CI from 5,000 bootstrap resamples of record groups, 880 on v7, 544 on t4;
test-set sampling only):

| Diff | v7 | t4 |
|---|---|---|
| 3-seed mean (D_s - A_s averaged over s) | **-0.79** [-1.54, -0.05] | +0.15 [-1.41, 1.75] |
| seed 0 | -0.32 [-1.34, 0.73] | +1.22 [-1.25, 3.75] |
| seed 1 | -0.95 [-2.15, 0.23] | 0.00 [-2.51, 2.58] |
| seed 2 | -1.11 [-2.22, 0.00] | -0.76 [-2.99, 1.28] |
| A seed 1 / 2 - A seed 0 | -0.47 / -0.08 | -0.61 / +1.07 |
| D seed 1 / 2 - D seed 0 | -1.11 / -0.87 | -1.83 / -0.91 |

Readings:

- Rule 3 fails on top1: D's 3-seed v7 mean is 0.79 below A, past the -0.5 margin (t4 +0.15 passes). Rule 4 does not
  apply (not >= 2 points worse on either set). Verdict by rule 5: **inconclusive**. The open raw-vs-G ECE question does
  not change this: rule 3 needs both the top1 condition and a calibration/flip condition.
- Seed 0 was D's best seed on both sets (v7 87.42, then 86.31 / 86.55): the seed-0 screen (-0.32) understated the gap.
  The paired 3-seed v7 CI excludes 0 (test sampling only; seed variance is not in it).
- Where v7 is lost (3-seed means): `choice` -1.33 (n 576) and `score` -1.11 (n 240); `noul` +0.07. By task, banking77
  (77 options) gives -5.8 (89.6 -> 83.8, n 80), -0.37 of the -0.79 overall; amazon -2.5, dbpedia14 -1.7, the rest within 1.5.
  [INFERENCE] the chat format's per-option rows with a 77-item option list in every prompt may be what costs banking77;
  arm H (no option list) would test it.
- Calibration, 3-seed means: D has lower raw softmax ECE (v7 .057 vs .064; t4 .130 vs .133) and raw NLL, but after G
  A is better on both (v7 .028 vs .035; t4 .075 vs .096) and on NLL (v7 .352 vs .374). Same pattern as seed 0.
- Reward scale: D's |r| p99 is 7.1 vs 15.0 and its fitted T 1.44 vs 1.78, stable across seeds (sub-claim C vs D holds).
- Flips: v7 3 vs 1 of 180, t4 4 vs 9 of 108 (D vs A): too few to separate the arms.

Open: whether rule 3's "lower ECE" means raw or G (temperature) ECE (moot for this verdict).
Deferred: arm H (no option list in the chat prompt) on branch `idea/pointwise-chat-input`
([pointwise-chat-input.md](pointwise-chat-input.md) there).

### Decision Index 0.3 (arm D, secondary)

External check, not part of the decision rule: D seed 0 vs the published base recipe `jaredpalmer/kev-4b@v7-base`, both
through `kev/scripts/pref/dindex_engine.py` (LocalPredictor, 64k serving context, bf16, checkpoint temperature;
unsupported = wrong, no truncation), on the same 15,000-row stratified subset of Decision Index 0.3's index benchmarks
(kit `apolinario/decision-index@9eb2dbe`, ~400 cases per benchmark), scored by the kit with the suite restricted to the
subset (`dindex_subset.py`). Paired estimate of the public index, not a board number; the board's Kev-4B (r10) is not
a comparator either (dates, documents-v1 and hard/devtools skill stages after the base recipe, T = 2.41), but it is the
reproduction check: its board checkpoint (`jaredpalmer/kev-4b@139fdd94`) runs on the same subset through our engine and
each benchmark is compared with its board score (`dindex_subset.py compare`, z on accuracy metrics).
Commands: `kev/scripts/pref/README.md` "Decision Index 0.3" (`dindex.sh check runs/pref/D/seed0` runs all three).

Archive `dindex-subset.tgz` (index/score files per run, retemper grid, pairings; no per-row results). Rows: 15,000 per
model; unsupported (scored wrong): r10 12, v7-base 12 (Kev format), D 0 (chat format).

| Model | DI 0.3 public index (subset) | Raw index | Notes |
|---|---|---|---|
| kev-4b r10 (board 37.95) | 36.69 | 52.61 | reproduction check passes: 24 accuracy benchmarks z mean -0.03, sd 1.14, none beyond 3 (2 beyond 2: ChessBench -2.17, CLadder +2.29); 12 F1/nDCG benchmarks (no z) mean diff -0.010, largest CLINC150 -0.064 |
| kev-4b@v7-base | 33.60 | 50.10 | comparator; served at its own T = 1.0 |
| D seed 0 | **36.91** | 53.18 | served at T = 1.0 |

Readings:

- D - v7-base = **+3.31** index points. Per benchmark (37): 27 up, 9 down, 1 tied; mean +0.034, median +0.032; sign
  test p = 0.004. No paired row-level CI (per-row results not shipped). Largest gains: RAGTruth (59) +0.226, BPoMP (20)
  +0.174, Humicroedit (21) +0.136, When2Call (62) +0.096. Losses: BANKING77 (4) -0.067 (the same task D loses on v7
  dev, -5.8), home appliance (9) -0.057, GSM8K (30) -0.040, BFCL (1) -0.035. Areas: Arts & Human Taste +0.071,
  Language Understanding +0.066, Knowledge & Reasoning +0.015, Tools +0.012, Retrieval & Classification +0.003.
- D seed 0 on the subset (36.91) matches the board's r10 through the same engine (36.69), which has later training
  stages than the base recipe.
- Temperature is irrelevant here: every T in the grid (0.5-4) gives the same index for both models (the 0.3 metrics are
  argmax / ranking based; T rescales scores without reordering them and noul stays on the same side of 0.5), so the
  "shipped", "raw" and "best T" pairings are identical.
- Confound: v7-base is the published base-recipe model, not our arm A (different training run; our A is the Kev recipe
  on bf16 weights). The +3.31 is D-vs-published-base, not pref-vs-CE. One seed of D (its best on dev).
- Dev and index disagree in sign: D trails A on v7 dev (-0.79, 3 seeds) yet beats v7-base on the index. Consistent with
  `docs/studies/README.md` finding that in-distribution validation is a weak proxy for the index.

Pending to attribute the gain: A seed 0 on the same subset (`dindex.sh subset runs/pref/A/seed0`), paired with D.

Verdict: **inconclusive** (rule 5) on the pre-registered dev rule. Best scalar-pref arm D: 3-seed top1 v7 -0.79 vs A
(fails the -0.5 margin), t4 +0.15; not refuted. Externally D seed 0 scores 36.91 on the Decision Index 0.3 subset,
+3.31 over the published base recipe (v7-base) and level with the board's r10; whether that comes from the pref
loss / scalar head or from our training run awaits A on the subset.
