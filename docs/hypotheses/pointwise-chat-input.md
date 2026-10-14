# Hypothesis: pointwise chat input (arms H, HP, HL, HB, HBL)

Branch `idea/pointwise-chat-input`, cut from `hyp/preference-reward-model` after its wave-1 and seeds-1-2 results
(`docs/hypotheses/preference-reward-model.md`, HEAD `c47005e`, whose "Deferred" note points here). Code:
`kev/kev/model.py` (the `chat_pointwise` input format), `kev/scripts/pref/` (`arms.sh`, `convert_pref_data.py`,
`rewardbench2.py`), `kev/tests/test_pref.py`. Everything below (arms, thresholds, rule) is fixed before any run.

The branch touches only the format plumbing of `kev/kev/model.py`: `INPUT_FORMATS` / `CHAT_FORMATS` / `CHAT_LAYOUTS`,
`chat_lists_options`, `chat_splits(..., fmt=)`, `chat_prompt(..., fmt=)`, `encode(..., chat=<format>)` and
`DecisionModel.encode`. A concurrent line of work optimizes chat serving speed in the other worktree (`chat_hidden`,
`forward_chat_batch`, `dindex_engine`); nothing here restructures those, so that rebase is mechanical. Every check of
the form `input_format == "chat"` goes through `CHAT_FORMATS` instead of string equality, at the checkpoint meta and in
`dindex_engine`'s OOM fallback.

## Claim

Arm D's chat prompt lists every option in each option's prefix (`Options:\n- o_1\n- ...\n\nGrade this option: o_k`,
`kev.model.chat_splits`). That list is what the format buys (relative cues: a 77-way intent list, a score scale read side
by side, multiple-choice items written to be compared) and what it costs (D's out-of-source top1 lead over the parent
study's baseline A came with a banking77 loss of 5.8 points, the parent doc's [INFERENCE]: the long listing).

Dropping the listing (`chat_pointwise`, arm H) makes the reward a function of (state, instruction, o_k) only: the
pointwise score Bradley-Terry is trained on, and the score a preference reward model is asked for when it grades one
response. Two claims, one per arm:

| Arm | Claim |
|---|---|
| **H** (format) | Removing the listing keeps dev top1 within 1 point of D on both dev sets, and makes the reward of a response independent of the set it is scored in (exactly, by construction — including the option order, which `kev.benchmark`'s permutation probe measures for D). |
| **HP** (data) | H plus open human preference pairs (RewardBench 2's own task: ranking a prompt's candidates) is strictly better than H and D at ranking (RewardBench 2 overall), at dev top1 within 1 point of H. The pairs are the only difference from H. |
| **HL** (loss) | H's grader trained as a reranker (RankLLaMA's objective: softmax CE over a question's option rewards) instead of Bradley-Terry + BSR, with probabilities from the softmax at G's fitted temperature, generalizes at least as well as H (transfer-v4 top1) and is better calibrated (G NLL). The loss is the only difference from H. |
| **HB** (absolute grader) | A grader asked "Is this option a correct answer? Answer Yes or No." about one option at a time (`chat_yesno`), its scalar head started at the base LM head's `W_yes - W_no` and trained with class-balanced per-option BCE, gives a reward on one scale across questions: higher pooled AUROC (every option of every question in one pool) than H and than H + Platt on transfer-v4 development and on RewardBench 2, at dev top1 within 1 point of H. Its sigmoid(r - log(k - 1)) is calibrated without a fitted temperature. |
| **HBL** (absolute + listwise) | HB + 0.25 × Kev's listwise CE keeps HB's pooled AUROC gain while recovering any top1 HB loses to H. |

The released Kev recipe (arm A: pointer head, CE, Kev format) has no reward-model prompt at all: it is the reference arm
for RewardBench 2, not a comparator for the format.

## Arms

| ID | What changes | Flags |
|---|---|---|
| D | parent study, scalar head + pref + BSR 1e-3, chat format (`Options:` list) | `--head scalar --input_format chat --loss pref --reg bsr --reg_w 1e-3` |
| **H** | D with the listing removed: scalar head + pref + BSR 1e-3, `chat_pointwise` | `--head scalar --input_format chat_pointwise --loss pref --reg bsr --reg_w 1e-3` |
| **HP** | H plus the converted preference pairs | H + `--data PREF_DATA --replay 12576` |
| **HL** | H with Kev's CE loss over the option rewards (added after the A/D RewardBench 2 run, before any H result) | `--head scalar --input_format chat_pointwise --loss ce` |
| **HB** | each option asked a Yes/No question on its own, head = LM head's `W_yes - W_no`, class-balanced BCE (added after the HL result, before any HB run) | `--head scalar --input_format chat_yesno --head_init yesno --loss bce` |
| **HBL** | HB + listwise CE at weight 0.25 | HB + `--ce_w 0.25` |
| A | parent study baseline, Kev pointer head + CE (reference for RewardBench 2 only) | `--head pointer --input_format kev --loss ce` |

H and D differ in exactly one flag. HP and H differ in exactly one pair of flags (the data file and the replay count).
HL and H differ in the loss (`--loss ce` and no `--reg`, which only regularizes preference pairs).
HB differs from H in three coupled things (prompt, head init, loss), deliberately: the Yes/No question is what makes the
`W_yes - W_no` init meaningful, and the init is what puts BCE's absolute target on a scale the base already reads. HB is
a recipe test against H, not an ablation; HBL and HB differ in exactly one flag.
Exact strings live in `kev/scripts/pref/arms.sh` (source of truth if it differs from this table; update here).
D's seed-0 number is arithmetically unchanged by this branch: `chat`'s layout is byte-identical (the branch keeps
`chat_lists_options("chat") == True`, and `kev/tests/test_pref.py` pins the chat prefix).

## The absolute-grader prompt (arms HB / HBL, `chat_yesno`)

Pointwise like `chat_pointwise`; the row ends on a question whose next-token answer the LM head already scores:

```
State:
{state}

Instruction: {instruction}

Option: {o_k}

Is this option a correct answer? Answer Yes or No.
```

`DecisionModel(head_init="yesno")` copies the LM head rows of `Yes` and `No` (`kev.model.YESNO_TOKENS`) into the scalar
head as `w = W_yes - W_no`, bias 0, so at step 0 the reward is the base model's own `logit(Yes) - logit(No)` at the last
prompt token. `--loss bce` (`kev.pref.balanced_bce_loss`) weighs a question's positives and negatives half each (the
1/k prior is no shortcut: all-zero rewards cost log 2 at any k), takes soft targets as given, and leaves the none
wordings (`kev.data.NONE_TEXTS`) out: "none of these" is a statement about the set, not a gradable option. Under that
balance the reward estimates a log likelihood ratio, so the k-option probability is sigmoid(r - log(k - 1)).

## The pointwise prompt (arm H, `chat_pointwise`)

Same rendering as `chat` (one user message per (question, option) through the tokenizer's own chat template, reward read
at the hidden state of the prompt's last token), with the listing block gone:

```
State:
{state}

Instruction: {instruction}

Grade this option: {o_k}
```

(Its predecessor has `Options:\n- {o_1}\n- ...\n\n` between the instruction and `Grade this option: `; that listing is
the only difference.) Prefix and branch are tokenized separately, exactly as in `chat`, so option k's prefix ids are
identical for all K options and its branch is `tokens(o_k + template tail)` at positions continuing the prefix. The
checkpoint records `input_format="chat_pointwise"` and the same `chat_template_sha256`; the digest check at load applies
unchanged.

Consequences, all of them intended:

- The reward of option k cannot see the other options: neither their text, their number, nor their order.
- The `none` option (the parent doc's `p_none_pair` 0.25 shortcut, `kev.data.NONE_OPTIONS`, e.g. `none: None of these`
  or a bare `none`) is graded as an option text with no list to refer to. `report.json`'s `variants.none_present` /
  `none_absent` are reported separately, as the idea note asks; a large none_present drop is the expected failure mode.
- Score-type questions keep their legends (`sst5`, `amazon` levels are rendered in the option text itself), so the scale
  survives; a 77-way choice question (banking77) no longer shows the other 76 intents.
- Training rows get shorter (the listing is inside the shared prefix, once per question, and counts against the chat row
  limit), so H never admits fewer records than D: the training population is the Kev-format `fits` filter, unchanged.
  The saving is the listing's tokens per question, not K extra rows (both formats read K option rows).

## The preference data (arm HP)

**Dataset: `nvidia/HelpSteer3`, split `preference`** (38,459 rows; CC-BY-4.0; revision resolved at conversion time and
recorded in the manifest). The alternative this study started from, `Skywork/Skywork-Reward-Preference-80K-v0.2`, is not
used: its card declares no license (`license: None`). HelpSteer3 is human-annotated (`overall_preference`, an integer
-3..3 per pair, plus the individual annotators' votes) and its `context` is already a chat message list, single- and
multi-turn, which is the state shape `kev.api.render` takes. `individual_preference` and `language` are not used:
one label per pair, the aggregate.

`kev/scripts/pref/convert_pref_data.py` (CPU, no GPU) converts a pair into one Kev labelled request:

```json
{"state": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "questions": {"better": {"type": "choice", "instructions": "Which response is better?",
                          "criteria": {"a": "...", "b": "..."}, "label": "a", "src": "helpsteer3_pref_general"}}}
```

- The better response (by `overall_preference`) goes to a seeded-random key (`--seed` 0), so neither the option key nor
  its position says which is better. `overall_preference == 0` ("about the same", 2,160 rows) is not a preference and is
  dropped, as are empty responses.
- Records are admitted exactly as every training record is (`kev.train.training_requests`):
  `kev.model.fits(materialize(record), tokenizer, **training_context(max_state))`, at the arm's default `max_state` 384,
  with `kev.suite.ADMISSION_BRANCH_HEADROOM` (64 branch tokens) reserved, the rule every frozen suite was admitted
  under: `kev.data.augment` can add an option to a Choice question at training time (a none option, p 0.12, or a
  distractor, p 0.15) that the converted pair does not carry, and the augmented record must still encode. The row check
  the arm's own format adds comes on top (`kev.model.encode` on the chat layout: the row must fit
  `max_branch + CHAT_ROW_SLACK`, with the same headroom), so nothing in the file can make `kev.train`'s encode raise.
  `kev.train`'s own re-filter of `--data` records reserves the same headroom, so the rule holds for any data file, not
  only this converter's output.
  From the character lengths (context p50 1,022 chars, responses p50 1,510 chars; the Kev branch holds the instruction
  and BOTH responses) ~13,000 of the 36,299 non-tie pairs fit [INFERENCE]; the manifest reports the real number and the
  per-limit drop counts.
- `--cap 12000`: a seeded sample of the admitted pool (12,000 = decision-v7's training partition, so one HP epoch is
  the suite once plus ~one pass over the pairs). `--replay 12576` puts the whole suite training partition back in, so
  HP's optimizer sees 24,576 records per epoch against H's 12,576.
- Decontamination: a pair whose prompt text — the last user turn, or its whole rendered state — equals a RewardBench 2
  prompt after normalization (casefold, whitespace collapse; the same key `kev.data.load_records` hashes states with) is
  dropped. Exact match is a weak filter (the Skywork v0.2 release used n-gram overlap against RewardBench 1) and all
  this converter promises; the count is in the manifest.
- The instruction `Which response is better?` is one constant, written in `convert_pref_data.py` and in
  `rewardbench2.py`: the preference data trains on the prompt that RewardBench 2 scores (the same string, one place to
  change).
- The manifest (`<out>.manifest.json`) records dataset/config/split/revision/rows, the tokenizer and its revision, the
  admission context and rule, admitted/written/dropped counts, the token-length percentiles of the admitted population,
  the decontamination rule and reference revision, and the output's sha256. The JSONL itself lives in `runs/` (ignored
  by git); `pack_results.sh` packs manifests, so the provenance travels with the results.

## Fixed across arms (unchanged from the parent study)

- Base `Qwen/Qwen3.5-4B-Base` rev `1001bb4d826a52d1f399e183466143f4da7b741b`; LoRA r16, alpha 32, dropout 0.05.
- Trained on decision-v7 train (`--suite evals/v7/decision-v7`), 2 epochs, lr 5e-5, effective batch 8, OneCycleLR,
  AdamW wd 0.01, grad clip, bf16 autocast, gradient checkpointing, `p_none_pair` 0.25, option permutation re-drawn each
  epoch, `--weights_dtype bf16`; final checkpoint only, no selection.
- HP additionally reads the converted JSONL (`--data`) with `--replay 12576`.
- Microbatch: `BATCH_FALLBACK=1` (chat 1 x accum 8) as for D/E in the parent study. `--reg` is computed per pass, so
  keep the split (and `ROW_BUDGET`, if ever used) identical across H, HP and D runs.
- Eval: `scripts/pref/eval_arm.sh` (kev.benchmark + reward_report.py) on `decision-v7-development`,
  `decision-v7-calibration`, `transfer-v4-development`; the locked test split only for the final verdict.

## Metrics

- Dev top1 on decision-v7 development and transfer-v4 development (clean rows; same questions for every arm), with the
  same per-type / per-task breakdown as the parent doc, plus `variants.none_present` / `none_absent` and the
  `permutation` flip rate from `report.json`; reward stats and calibration from `reward_report.json` (A's ECE story is
  unchanged: only softmax probabilities are comparable across arms).
- RewardBench 2 (`scripts/pref/rewardbench2.py`): per-subset (Factuality, Precise IF, Math, Safety, Focus, Ties) and the
  six-subset mean, unsupported counts, wall time and token totals; the Ties subset exactly as `allenai/reward-bench`
  scores it (weighted 0.30/0.30/0.20/0.20/0.01; the reference prompt of a tied prompt is its `ref:N` row), the other
  subsets as the chosen response winning against three rejected ones (credit shared at a tied top). The prompt layout is
  the checkpoint's own: A/D put the prompt's candidates in one question (set-relative, D's listing includes them), H/HP
  score each candidate on its own. The dataset lists the chosen response first, so A/D are shown each prompt's
  candidates in a seeded per-prompt shuffle (`--order_seed 0`; `-1` = the dataset's order) and their scores are mapped
  back before scoring: under the dataset's order a first-position preference is credited as a correct ranking.
  An unsupported candidate (over the serving context, or OOM) is never truncated and
  never scored around: a non-Ties prompt with any unsupported candidate earns 0 (the ranking rule is defined on the four
  candidates the benchmark posed), and a Ties row with one keeps its place in every denominator while contributing
  nothing to any metric, so no arm can gain from failing to score. Per-prompt and per-subset counts are in the JSON.
- Set-independence probe (`--probe 64`, part of `rewardbench2.py`): for 64 best-of-4 prompts, the chosen response's
  reward scored with all four candidates versus with itself and one rejected one: mean/median/max |Δ| and the pairwise
  flip rate of the chosen-vs-that-rejected comparison. H's Δ must be exactly 0 (the same one-option request both times);
  A's and D's are what those formats pay. A prompt that either of the probe's two calls could not score entirely is
  skipped (its scores are placeholders), and the skip count is reported.
- Serving path (not a score): H/HP are chat-format checkpoints, so `dindex_engine` and `rewardbench2.py` replay their
  passes as CUDA graphs (`kev.cuda_graphs.chat`, on for `input_format in kev.model.CHAT_FORMATS` on CUDA; off with
  `GRAPHS=0` / `--graphs 0`). The graphed passes are eager `chat_hidden`'s tokens and positions, so rewards agree up to
  floating-point reassociation; `chat_speed_check.py` prints both clocks and max |Δ reward| per record on the training
  machine before any number is trusted.
- Decision Index 0.3 subset for H only (secondary, external: the parent doc's `dindex.sh subset` + `dindex_subset.py
  pair`), against D 36.91 and v7-base 33.60 on the same 15,000 rows.
- Pooled metrics (arms HB / HBL vs H / HL / D / A): `reward_report.py` adds `auroc_pooled` (every option of every
  question pooled: P(a right option of one question outscores a wrong option of another), rank-based, so Platt cannot
  move it) and `ece_sigmoid_prior` (ECE of sigmoid(r - log(k - 1))). `rb2_pooled.py` does the same on a RewardBench 2
  JSON (candidates pooled per subset and overall; `ece_prior` uses the prompt's known num_correct odds;
  `--report` adds `ece_platt` from G's Decision Index (a, b)). Unsupported prompts count at their worst case (correct
  candidates lowest, wrong ones highest).

## Decision rule

Set before any run.

1. **Screen (H vs D, seed 0, dev).** H's top1 must be within 1 point of D's on BOTH development sets. D's numbers are the
   parent study's (seed 0: v7 87.42, t4 79.73; 3-seed means 86.76 / 78.81). If one set is more than 1 point below, H's
   format claim is refuted at seed 0 and seeds 1-2 are not run (the RB2 and probe numbers already collected are still
   reported).
2. **Seeds.** If the screen passes, run H seeds 1-2 and apply rule 1 to the 3-seed means (the parent study's seeds show
   a spread of about 1 point per arm, so a single seed cannot settle a 1-point bound).
3. **The gain H is for.** Exactly-zero RB2 probe deltas for H (a non-zero delta is an implementation bug: the request
   for a candidate does not mention its siblings), and at least one of: t4 top1 >= D's (3-seed means), or a lower
   permutation flip rate than D on either dev set. The flip counts are small (1-4 of 180 / 108 in the parent study:
   1.7 / 2.8 points per flip), so the flip clause is read as a sign, never as a test.
4. **Ranking (RewardBench 2, seed 0).** H vs D: H's six-subset mean >= D's - 0.5 points (a format that also ranks
   better is a free win; one that ranks worse must not be more than half a point worse). HP vs H: HP must beat H by
   >= 1 point overall AND on at least 4 of the 6 subsets. Between +1 and -1 point overall is reported as "no measurable
   gain from mixing" (not support for the data claim); worse than -1 is a loss. A's overall is reported as the reference
   (the released recipe's prompt is not a reward-model prompt, so it is not a comparator). All arms are scored under the
   same unsupported policy (see the metrics), fixed here before any run: no arm is credited for a prompt or row it could
   not score, so the comparison cannot be moved by a cheaper way to fail.
5. **Verdict.**
   - Format claim **supported** if rules 1-3 hold; **refuted** if rule 1 fails on either set at seed 0 (or on the 3-seed
     means if reached); **inconclusive** otherwise (e.g. the flip-rate clause alone missing).
   - Data claim (HP) is **supported** if rule 4's HP clause holds and HP is within 1 point of H on both dev sets
     (a ranking gain that costs decision accuracy is not the arm's claim); **refuted** if HP loses >= 1 point overall;
     **inconclusive** in between.
   - The idea note's H-vs-D rule (within 1 point on both dev sets, plus a visible gain) is rule 1 + rule 3 and is not
     re-scored on RB2; RB2 is the ranking test the note did not have.
   - Loss claim (HL vs H, seed 0) is **supported** if HL's top1 is >= H's + 1 on transfer-v4 development and not more
     than 1 point below on decision-v7 development, or within 1 point on both with a lower G NLL (softmax at fitted T)
     on both; **refuted** if HL is more than 1 point below H on either dev set; **inconclusive** otherwise. Seeds 1-2 as
     in rule 2 if it passes at seed 0.
   - Absolute-grader claim (HB vs H, seed 0) is **supported** if HB's pooled AUROC beats H's (H + Platt has the same
     AUROC, Platt being monotone) by >= 0.01 on transfer-v4 development AND on RewardBench 2 (overall, `rb2_pooled.py`),
     with HB's top1 not more than 1 point below H's on either dev set; **refuted** if HB's pooled AUROC is not above H's on
     either, or its top1 is more than 1 point below H on either dev set; **inconclusive** otherwise. Calibration
     (secondary): HB's `ece_sigmoid_prior` vs H + Platt's `ece_sigmoid` on decision-v7 development. HBL vs HB: same
     AUROC test (HBL within 0.01 of HB) plus HBL's top1 >= HB's. Per-subset prediction, recorded before the run:
     the gain shows on the verifiable subsets (Factuality, Math, Precise IF) and not on Focus or Ties.
   - Next stages, only if HB is supported: HBX (HB on long-form candidate data, the pointwise grader at answer length)
     and preference fine-tuning from HB's checkpoint with its BCE kept as an anchor. Not pre-registered here.
6. **Secondary, not in the verdict:** H's DI 0.3 subset index vs D's 36.91 and v7-base's 33.60 (already measured), and
   the per-subset RB2 pattern (Safety / Precise IF where the pairs should help most).

## Compute

One RTX 3090 [estimate, H's chat-format cost is D's 9.1 h minus the listing's tokens per question; the listing is
inside the shared prefix, so the saving is small]:

- H: ~9 h per run. Seed 0 first (~9 h); seeds 1-2 only if the screen passes (~18 h more).
- HP: 12,000 pairs on top of the 12,576 suite records, so ~1.95x H's records per epoch: ~18 h.
- RewardBench 2: 1,865 prompts. A/D: one request per prompt (D's prompt lists all 4 candidates: ~1.5-3 s each on a
  3090) [INFERENCE: candidates p50 1,150 chars], ~1-1.5 h per arm. H/HP: 8,977 one-candidate requests (~2-4 s each
  with a ~300-900-token prompt), ~5-8 h per arm, plus the 64-prompt probe (6 requests each) ~10 min. All four arms
  ~15-20 h. D and H/HP run the graphed chat passes by the serving default, so these are the fast-path estimates;
  `chat_speed_check.py runs/pref/H/seed0` measures H's before the rest.
- DI 0.3 subset (H only): D's 7.6 h (27,246 s) was the old eager chat path. Graphed (on by default since `4668509`),
  D seed 0 on a 3090: the first run's graphs' memory pool grew until eager passes ran out of memory, 586 of 15,000
  requests refused (0 in the first 2,000, ~90 per 1,000 at the end; API-Bank 237, POP909-CL 221, CLINC150 80; index
  34.42), 15,154 s. dindex_engine now releases the graphs on out of memory and reruns the request (CudaGraphs.release,
  `2efe65f`); those 586 rows rerun with it took 3,450 s (5.9 s each: the suite's most-option requests). Result: index
  37.01, 15,000 ok, against eager's 36.91 (kernel rounding: `dindex_subset.py flips`, 372 of 123,375 answers changed,
  0.30 %, most in near-tied many-option benchmarks -- ChessBench 4.9 %, POP909-CL 3.8 %, GSM8K 3.2 % at max |dp| <= 0.05;
  max |dp| anywhere 0.15 (ANLI); per-benchmark deltas 14 up, 13 down). Graphed wall time ~15-18.6k s (the refused rows' failed
  attempts are in the 15,154 s), so ~4.2-5.2 h: 1.5-1.8x eager, still 2-3x the Kev format's 6,048-7,367 s. `GRAPHS=0`
  runs the old eager path.
- Total for the seed-0 path (H, HP, RB2 on A/D/H/HP, DI on H): ~35-40 h of one 3090; the seed screen adds ~18 h.

## Commands

Training machine only, from `kev/`. Setup, env and the D/A numbers are the parent doc's (`setup_3090.sh` now prints the
pointwise prompt too).

```bash
cd kev
.venv/bin/python -m pytest tests/test_pref.py            # CPU; includes the chat_pointwise layout, set-independence and train smoke tests
TORCH_EXTRA=cu128 bash scripts/pref/setup_3090.sh        # prints both rendered prompts (chat and chat_pointwise) and their token counts

# HP's data (once, CPU; downloads HelpSteer3 and RewardBench 2)
.venv/bin/python scripts/pref/convert_pref_data.py --out runs/pref/data/pref-helpsteer3.jsonl
# check the manifest: admitted/written counts, drop reasons, dataset revision, sha256

# arms (one GPU each; BATCH_FALLBACK=1 = D's microbatch, 1 x accum 8)
BATCH_FALLBACK=1 scripts/pref/run_arm.sh H 0 0           # arm H seed 0: train + eval
BATCH_FALLBACK=1 PREF_DATA=runs/pref/data/pref-helpsteer3.jsonl scripts/pref/run_arm.sh HP 0 0
BATCH_FALLBACK=1 scripts/pref/run_arm.sh H 1 0           # only if rule 1 passes
BATCH_FALLBACK=1 scripts/pref/run_arm.sh H 2 0

# ranking (RewardBench 2), one line per arm: A (reference), D (format comparator), H, HP
# (the chat arms replay their passes as CUDA graphs, as dindex_engine runs them; --graphs 0 for the eager path)
.venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/A/seed0 --out runs/pref/A/seed0/rewardbench2.json
.venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/D/seed0 --out runs/pref/D/seed0/rewardbench2.json
.venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/H/seed0 --out runs/pref/H/seed0/rewardbench2.json
.venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/HP/seed0 --out runs/pref/HP/seed0/rewardbench2.json

# H's serving path: the graphed passes against the eager ones on the same DI rows (must agree to kernel rounding,
# and only the graphed clock is what a subset run costs). Run before H's DI subset, as D's was.
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/H/seed0

# Decision Index 0.3 subset (secondary, H only; same rows as the parent study's D / v7-base runs)
scripts/pref/dindex.sh subset runs/pref/H/seed0
.venv/bin/python scripts/pref/dindex_subset.py pair runs/dindex/kev4b-v7base-sub15000 runs/pref/H/seed0/dindex-0.3-sub15000

scripts/pref/pack_results.sh pref-H-HP.tgz                # logs, metrics, eval outputs, rewardbench2.json, data manifests
```

## Risks / open

- The listing is the format's strength: if banking77-style many-way choice questions are where most of v7 lives, H loses
  there and nowhere else (the parent doc measured D's loss there too). The per-task breakdown decides whether the
  within-1-point screen passed for the right reason.
- The `none` option without a referent (rule 3's flip clause and the variants breakdown are what would show it) and
  multiple-choice items whose options are meant to be compared (mmlu) are the other two expected losses.
- RewardBench 2 is a hard external benchmark for a 4B base recipe: all six arms may land far from its frontier, and the
  Ties subset's 102 prompts (51 ref + 51 tied pairs) make its score coarse. The rule compares arms at the same
  checkpoint budget, not against the leaderboard.
- Unsupported requests: H scores every candidate on its own, so one long candidate no longer takes its prompt down with
  it (the set-relative formats overflow per prompt). No arm is credited for what it could not score: the prompt/row is
  zeroed as described under the metrics. The counts are reported per arm; they are not comparable across formats by
  construction.
- HP's wall time and memory are H's plus ~12,000 records (no new memory, more steps); the same microbatch applies.
- Decontamination is exact-match only; a paraphrase of a RewardBench 2 prompt stays in the training data.
- The converted pairs carry one question type (`choice`) and one instruction: HP tests the data, not instruction
  diversity. `overall_preference` -3..3 is used as a binary label, ignoring its strength.
- This branch was cut from `c47005e` and is rebased onto the chat serving speedup (`28a6047`, `kev.cuda_graphs.chat`): the
  format plumbing is isolated from those functions on purpose, `CHAT_FORMATS` (`kev.model`) holds `chat` and
  `chat_pointwise` and is the only coupling point, and the graphed passes build their row tokens and positions from
  `chat_splits`' output, so the pointwise prefix flows through them unchanged (the CUDA parity test runs both layouts).

## Results

Archive `pref-results.tgz` (`pack_results.sh`, code at `fc08457`). Clean rows as in the parent doc (v7 n = 1264, t4
n = 656); G = temperature / Platt fit on the run's decision-v7 calibration split (n = 1148). HB / HBL not run.

### Verdict (seed 0)

| Claim | Rule | Numbers | Verdict |
|---|---|---|---|
| Format (H vs D) | 1: within 1 point of D seed 0 on both dev sets | v7 85.84 vs 87.42 (-1.58), t4 78.51 vs 79.73 (-1.22) | **refuted**; seeds 1-2 not run |
| Format, gain clauses | 3: probe Δ exactly 0; t4 >= D or fewer flips | probe 0 / 0, 0 of 64; flips v7 0/60 vs 0/60, t4 0/36 vs 1/36 | probe passes; flip clause a sign only |
| Ranking (H vs D) | 4: H's RB2 >= D's - 0.5 | 0.6653 vs 0.6641 (+0.12) | passes |
| Data (HP vs H) | 4 + 5: HP >= H + 1 on RB2 and on 4 of 6 subsets; within 1 point on dev | RB2 0.6517 vs 0.6653 (-1.36), ahead on 3 of 6; t4 -1.98 | **refuted** (loses >= 1 point overall) |
| Loss (HL vs H) | 5: refuted if HL > 1 point below H on either dev set | v7 +0.79, t4 -1.22 | **refuted** |

The format refutation rests on D's seed 0, its best seed on both sets. Paired by question, H vs D seed 0 is
-1.58 ± 1.07 on v7 (McNemar p .006, 14 H-only vs 34 D-only right) and -1.22 ± 1.93 on t4 (p .28). Against D seeds
1 / 2, H is -0.47 / -0.71 on v7 and +0.61 / -0.30 on t4, and against D's 3-seed means -0.92 / -0.30, inside the bound.
The rule fixed seed 0 before the runs, so the verdict stands; rule 2's H seeds 1-2 are what would test the 3-seed
means, and the rule does not allow them after a seed-0 failure.

### Development sets

| Arm | v7 top1 | t4 top1 | v7 ECE raw / G | t4 ECE raw / G | v7 NLL raw / G | t4 NLL raw / G | T | Flip v7 / t4 | v7 / t4 pooled AUROC | v7 \|r\| p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| D (parent, seed 0) | 87.42 | 79.73 | .054 / .036 | .124 / .088 | .430 / .374 | .708 / .561 | 1.47 | 0 / 1 (n 60 / 36) | | 7.1 |
| H | 85.84 | 78.51 | .062 / .035 | .125 / .092 | .442 / .375 | .771 / .592 | 1.56 | 0 / 0 | .981 / .898 | 7.5 |
| HP | 86.55 | 76.52 | .059 / .032 | .150 / .107 | .464 / .385 | .852 / .630 | 1.60 | 0 / 0 | .979 / .903 | 8.0 |
| HL | 86.63 | 77.29 | .075 / .030 | .146 / .087 | .476 / .356 | .838 / .569 | 1.90 | 0 / 0 | .982 / .882 | 11.0 |

D seeds 1 / 2: v7 86.31 / 86.55, t4 77.90 / 78.81. Sigmoid ECE after Platt, v7 / t4: H .009 / .062, HP .009 / .077,
HL .014 / .066, D .010 / .063.

Per type, top1 (H / D seed 0 / D 3-seed mean):

| Set | choice | noul | score |
|---|---|---|---|
| v7 (n 576 / 448 / 240) | 90.97 / 91.67 / 91.61 | 91.74 / 92.63 / 92.41 | 62.50 / 67.50 / 64.58 |
| t4 (n 336 / 280 / 40) | 76.49 / 78.27 / 76.69 | 85.00 / 86.43 / 85.60 | 50.00 / 45.00 / 49.17 |

Where H's gap to D is (tasks >= 2 points off D's 3-seed mean): v7 amazon 57.5 vs 62.9 (n 80), yelp 67.5 vs 70.0,
banking77 81.25 vs 83.75, composition_conditional 75.0 vs 81.25 (n 16); t4 mmlu 62.5 vs 71.7 (n 80, -1.1 points of the
overall), composition_held_or_not 78.1 vs 83.3; H ahead on t4 composition_held_conditional 90.6 vs 74.0 (n 32) and
emotion 58.75 vs 55.0. `none_present` on t4: H .722 vs D .528 (all three D seeds; n 36).

### RewardBench 2

Candidates shuffled per prompt for A / D (`--order_seed 0`); H / HP / HL score each candidate alone. 0 unsupported
prompts for every arm; chat arms graphed.

| Arm | Factuality | Precise IF | Math | Safety | Focus | Ties | Overall | no-Ties | Pooled AUROC (all / no-Ties) | Probe mean / max \|Δ\|, flips | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (pointer, kev) | 0.3747 | 0.3063 | 0.4481 | 0.5556 | 0.8545 | 0.5289 | 0.5113 | 0.5078 | .780 / .734 | 1.04 / 4.02, 13 of 64 | 622 s |
| D (scalar, chat) | 0.6232 | 0.3500 | 0.5792 | 0.7733 | 0.8970 | 0.7618 | 0.6641 | 0.6445 | .858 / .817 | 0.577 / 1.64, 6 of 64 | 1,527 s |
| H (pointwise) | 0.6063 | 0.3125 | 0.6995 | 0.7622 | 0.9111 | 0.7004 | **0.6653** | 0.6583 | .855 / .803 | 0 / 0, 0 of 64 | 1,997 s |
| HP (pointwise + pairs) | 0.7137 | 0.3187 | 0.7268 | 0.6556 | 0.8646 | 0.6309 | 0.6517 | 0.6559 | .836 / .810 | 0 / 0, 0 of 64 | 1,988 s |
| HL (pointwise, listwise loss) | 0.5916 | 0.2875 | 0.6393 | 0.6956 | 0.9556 | 0.6297 | 0.6332 | 0.6339 | .828 / .780 | 0 / 0, 0 of 64 | 1,993 s |

Differences from H, points: D +1.7 / +3.7 / -12.0 / +1.1 / -1.4 / +6.1; HP +10.7 / +0.6 / +2.7 / -10.7 / -4.6 / -6.9;
HL -1.5 / -2.5 / -6.0 / -6.7 / +4.4 / -7.1 (subset order as in the table).

### Decision Index 0.3 subset (secondary)

Same 15,000 rows: H **35.78** (raw 51.91, breadth 34.89; 15,000 ok) vs D 36.91 eager / 37.01 graphed and v7-base
33.60: H - D = -1.1 to -1.2, H - v7-base = +2.2. The `dindex_subset.py pair` log is not in the archive.

### Position effect on RewardBench 2 (A / D, shuffled runs)

The shuffled runs store scores in the dataset's order, and the shuffle is `random.Random(f"order:0:{id}")`, so each
prompt's displayed order can be rebuilt offline. Non-Ties prompts with one chosen and three rejected (n 1763), strict
win (chosen above all three):

| Arm | Win, chosen shown 1st / 2nd / 3rd / 4th (n 404 / 415 / 465 / 479) | Top-scored candidate shown 1st / 2nd / 3rd / 4th | Dataset-order run (chosen always 1st) |
|---|---|---|---|
| A | .822 / .547 / .462 / .434 | .458 / .217 / .176 / .149 | .837 |
| D | .750 / .747 / .697 / .653 | .272 / .263 / .247 / .218 | .767 |

The rebuild checks out: each arm's win with the chosen shown first matches its dataset-order run (.822 vs .837, .750 vs
.767). A prefers the first position strongly (top-scored shown first 46% vs 25% chance; win falls 39 points from 1st to
4th). D's effect is a mild primacy: top-scored share 27 -> 22%, win -10 points from 1st to 4th (z about 3.2;
chi-square on the top-scored positions 12.1 on 3 df, p .007). Pointwise H has none by
construction (one request per candidate). On the dev sets' short options the same effect is small (D 0-3 of 60 / 36
permutation flips across seeds).

### Readings

- Dropping the listing costs decision accuracy where the pre-registered risks put it: the score type on v7 (amazon /
  yelp ratings, -5.0 vs D seed 0 on n 240, about 0.95 of the 1.58-point gap), where the listing shows the scale
  side by side, and mmlu on t4 (-9 to -10 points, n 80), whose options are meant to be compared. banking77, the
  many-way listing case, costs only 1-2.5 points.
- What it buys is set independence: probe Δ exactly 0 for H / HP / HL (D: 6 of 64 flips), 0 permutation flips on both
  sets, and on t4 `none_present` 72 vs 53. That's the property the format was for; on these sets it doesn't pay
  for the accuracy.
- Ranking is a wash: H ties D on RewardBench 2 (+0.12 overall, pooled AUROC .855 vs .858), trading Math (+12.0)
  against Ties (-6.1) and Precise IF (-3.7). Rule 4's H clause passes, so the ranking side doesn't reject the format.
- Mixing HelpSteer3 pairs (HP) moves RB2 by subset, not overall: Factuality +10.7 and Math +2.7 against Safety -10.7,
  Ties -6.9, Focus -4.6, and costs 2.0 t4 points (paired p .047). [INFERENCE] HelpSteer3's helpfulness preferences
  reward compliance, which Safety's refusal-preferred prompts penalize.
- The listwise loss (HL) gains nothing over H: +0.79 v7, -1.22 t4, -3.2 RB2 overall; its one gain is the lowest
  softmax NLL after G (v7 .356 vs H .375, t4 .569 vs .592), at a larger reward scale (|r| p99 11.0) and T 1.90.
- Absolute-grader arms (HB / HBL) are compared with H, now measured: t4 pooled AUROC .898, RB2 pooled AUROC .855
  (`rb2_pooled.py`); HB is supported only at >= .908 and >= .865 with top1 >= 84.84 v7 / 77.51 t4.

### RewardBench 2, dataset order (superseded)

The first A/D runs (seed 0) showed every prompt's chosen response as option `a`, before the shuffle above existed.
Reported for the record, not for the rule: a first-position preference counts as a correct ranking here, so they are
rerun shuffled.

| Arm | Factuality | Precise IF | Math | Safety | Focus | Ties | Overall | no-Ties | Probe mean / max abs Δ, flips | Wall |
|---|---|---|---|---|---|---|---|---|---|---|
| A (pointer, kev, graphs off) | 0.8126 | 0.4688 | 0.9071 | 0.8378 | 0.9535 | 0.5500 | 0.7550 | 0.7960 | 0.0758 / 0.304, 0 of 64 | 622 s |
| D (scalar, chat, graphs on) | 0.7326 | 0.3688 | 0.6612 | 0.8200 | 0.9192 | 0.6765 | 0.6964 | 0.7004 | 0.409 / 1.65, 5 of 64 | 1,526 s |

0 unsupported prompts for either arm. Probe |Δ| is in raw logits (pointer vs scalar head scales), so only the flip
counts compare across arms.
