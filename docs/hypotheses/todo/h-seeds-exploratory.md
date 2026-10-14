# Question: is H's true gap to D 0.9 or 1.6 points on v7? (H seeds 1-2, exploratory)

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

**Exploratory: the format claim was refuted at seed 0, and the pre-registered rule does not allow these seeds to
re-open it.** They would only annotate the refuted verdict with the 3-seed gap.

## Question

Rule 1 of `done/pointwise-chat-input.md` fixed seed 0 before any H run and refuted the format claim there
(v7 −1.58, t4 −1.22 vs D); rule 2 ("run H seeds 1-2 only if the screen passes") was therefore not triggered. Against D's
other seeds the gap is roughly half: v7 −0.47 / −0.71 vs D seeds 1 / 2 and −0.92 against D's 3-seed mean. So: is H's
true (3-seed) v7 gap ≈ **0.9 points** — inside rule 1's 1-point bound — or ≈ **1.6** (D seed 0's gap, D's best seed on
both sets)?

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| v7 H − D seed 0 | −1.58 ± 1.07 (McNemar p .006; 14 H-only vs 34 D-only right) | `done/pointwise-chat-input.md` §Verdict |
| t4 H − D seed 0 | −1.22 ± 1.93 (p .28) | same |
| v7 H − D vs D seeds 1 / 2 | −0.47 / −0.71 | same |
| t4 H − D vs D seeds 1 / 2 | +0.61 / −0.30 | same |
| H − D 3-seed means | v7 −0.92, t4 −0.30 | same |
| D seeds 1 / 2 | v7 86.31 / 86.55, t4 77.90 / 78.81 | same (§Development sets) |
| rule 2's status | "seeds 1-2 are not run (the RB2 and probe numbers already collected are still reported)"; "the rule does not allow them after a seed-0 failure" | §Verdict |
| seed spread context | arm seeds spread ~1 point; a single seed cannot settle a 1-point bound | §Decision rule 2 |

The number matters beyond the verdict: H is the pointwise reference for the in-progress PF-RCR wave (`Hh`) and the HB /
HL / HP comparisons, so how much of H's deficit is seed noise bounds how much of D's dev top1 is really the listing.

## Arms

| Arm | Flags | Status |
|---|---|---|
| H seed 1 | `arms.sh` `H` (`--head scalar --input_format chat_pointwise --loss pref --reg bsr --reg_w 1e-3`) | built, not run |
| H seed 2 | same, `--seed 2` | built, not run |

D seeds 1-2 already exist (`done/preference-reward-model.md` §Seeds 1-2, archive `pref-seeds12.tgz`); no D run is needed.
Keep H's microbatch identical across seeds (D/E/H's `--reg bsr` is computed per pass, `arms.sh`) — `BATCH_FALLBACK=1`,
`ROW_BUDGET` unset, as H seed 0.

## Decision rule (pre-registered, exploratory)

- Report the paired-by-question 3-seed mean Δ(v7, H − D) with a 95% CI (bootstrap over record groups, as the parent's
  §Seeds 1-2 table does), the same for t4, and H's seed 1 / 2 / 3-seed-mean top1 beside D's — against the two candidate
  values (≈0.9 from D seeds 1-2, ≈1.6 from D seed 0).
- Read the outcome, fixed here before the runs: CI upper end ≤ −1.0 → the gap exceeds rule 1's bound even against D's
  weaker seeds (the refutation is robust); CI covering 0 → the seed-0 screen overstated the gap (the refutation stands
  but is a seed-0 result, and the doc must say so).
- Verdict classes: the format claim stays **refuted** either way (rule 1 fixed seed 0). No re-scoring of rules 3-5, no
  new arms, no RB2 re-run — H's RB2 and probe numbers are already collected.
- Report the per-type / per-task breakdown of the 3-seed mean (the parent's §Development sets and §Readings tables are
  the format), so "where H loses" is not read off one seed.

## Cost [estimate]

~9 h per run on one 3090 (`done/pointwise-chat-input.md` §Compute: "H: ~9 h per run"); H seed 0's `wall_seconds` is
30,961 (≈ 8.6 h) in the training machine's `runs/pref/H/seed0/training_metrics.json` (`kev/kev/train.py` writes it;
`runs/` is gitignored). Two seeds ≈ **17 h** of one 3090, one lane.

## Commands

```bash
BATCH_FALLBACK=1 ./kev.sh scripts/pref/run_arm.sh H 1 0     # train + eval (eval_arm.sh)
BATCH_FALLBACK=1 ./kev.sh scripts/pref/run_arm.sh H 2 0
```

## Results

Not run.
