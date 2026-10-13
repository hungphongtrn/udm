# Hypothesis: pointwise chat input (arms H and HP)

Branch `idea/pointwise-chat-input`, cut from `hyp/preference-reward-model` after its wave-1 and seeds-1-2 results
(`docs/hypotheses/preference-reward-model.md`, HEAD `c47005e`, whose "Deferred" note points here). Code:
`kev/kev/model.py` (the `chat_pointwise` input format), `kev/scripts/pref/` (`arms.sh`, `convert_pref_data.py`,
`rewardbench2.py`), `kev/tests/test_pref.py`. Everything below (arms, thresholds, rule) is fixed before any run.

The branch touches only the format plumbing of `kev/kev/model.py`: `INPUT_FORMATS` / `CHAT_FORMATS`,
`chat_lists_options`, `chat_splits(..., options=)`, `chat_prompt(..., options=)`, `encode(..., chat_listing=)` and
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

The released Kev recipe (arm A: pointer head, CE, Kev format) has no reward-model prompt at all: it is the reference arm
for RewardBench 2, not a comparator for the format.

## Arms

| ID | What changes | Flags |
|---|---|---|
| D | parent study, scalar head + pref + BSR 1e-3, chat format (`Options:` list) | `--head scalar --input_format chat --loss pref --reg bsr --reg_w 1e-3` |
| **H** | D with the listing removed: scalar head + pref + BSR 1e-3, `chat_pointwise` | `--head scalar --input_format chat_pointwise --loss pref --reg bsr --reg_w 1e-3` |
| **HP** | H plus the converted preference pairs | H + `--data PREF_DATA --replay 12576` |
| A | parent study baseline, Kev pointer head + CE (reference for RewardBench 2 only) | `--head pointer --input_format kev --loss ce` |

H and D differ in exactly one flag. HP and H differ in exactly one pair of flags (the data file and the replay count).
Exact strings live in `kev/scripts/pref/arms.sh` (source of truth if it differs from this table; update here).
D's seed-0 number is arithmetically unchanged by this branch: `chat`'s layout is byte-identical (the branch keeps
`chat_lists_options("chat") == True`, and `kev/tests/test_pref.py` pins the chat prefix).

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
  score each candidate on its own. An unsupported candidate (over the serving context, or OOM) is never truncated and
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
- DI 0.3 subset (H only): D's 7.6 h was the old eager chat path; the graphed path (on by default since `4668509`) targets
  the Kev format's ~2 h and is unmeasured, so ~2-7 h [INFERENCE]. Re-measure D first (`chat_speed_check.py`, then a fresh
  subset run) before relying on it; `GRAPHS=0` runs the old eager path, only for the comparison.
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

To be filled after the runs, in the parent doc's format (per arm and seed: dev top1 / ECE / flip rate, then the RB2
table, the probe, and H's DI subset row). Nothing has been run on this branch; the numbers above that are not marked
[INFERENCE] are the parent study's measurements.
