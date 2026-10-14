# Question: does averaging D's reward over k shuffled option orders remove RewardBench 2's mild primacy?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

No training. Eval-side only, on an existing checkpoint.

## Question

D reads the option listing, so its reward for a candidate can depend on the position it is shown in — unlike a pointwise
grader (H's probe Δ is exactly 0 by construction). On RewardBench 2's shuffled D run the effect is a mild primacy of
about 10 points. Question: if a prompt's candidates are scored in k seeded shuffles and D's reward for a candidate is the
**mean** of its k scores, does the position table flatten to chance (each position ~25% top-scored) without costing
RB2's overall score?

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| D top-scored, chosen shown 1st / 2nd / 3rd / 4th | .272 / .263 / .247 / .218 (chance 25%) | `done/pointwise-chat-input.md` §Position effect on RewardBench 2 |
| D strict win, same positions (n 404 / 415 / 465 / 479) | .750 / .747 / .697 / .653 | same |
| significance | win −10 points 1st → 4th, z ≈ 3.2; χ² 12.1 on 3 df, p .007 | same |
| A for contrast (the listing prompt the pointwise arm dropped) | win .822 / .547 / .462 / .434; top-scored .458 / .217 / .176 / .149 | same |
| on the dev sets the effect is small | D 0-3 of 60 / 36 permutation flips across seeds | same |
| D's shuffled RB2 overall / wall | 0.6641 / 1,527 s | §RewardBench 2 |
| the shuffle is seeded and rebuildable | `--order_seed 0`; the displayed order is `random.Random(f"order:0:{id}")`, and the JSON stores scores in the dataset's order | §RewardBench 2, §Position effect |
| why it matters | D is the best scalar arm and the format comparator (H's RB2 clause is D's − 0.5), so how much of its score is position is live | §Verdict, §Decision rule 4 |

## Arms

No arms and no training: `runs/pref/D/seed0` scored k times, once per `--order_seed`, then averaged per candidate
offline. `rewardbench2.py` has `--order_seed` only — a built-in `--order_avg k` **is not built**; the average is
post-hoc over the k JSONs (each stores per-prompt candidate scores in the dataset's final order, and the displayed order
is rebuildable from the seed, so the mapping is offline arithmetic, not a rerun).

## Decision rule (pre-registered)

- Report, for the k-averaged score, the same two tables as `done/pointwise-chat-input.md` §Position effect: top-scored
  share per shown position and the strict win per shown position, with the same n, plus the per-position spread.
- **Removed** if both hold: (a) the four positions' top-scored shares are inside the binomial noise of 25% — at
  n 404-479 a 95% interval on a 0.75 win is about ±4 points, and the observed spread is 10 [INFERENCE from n] — and
  (b) the k-averaged RB2 overall is not below D seed 0's 0.6641 by more than 0.5 points (rule 4's own RB2 margin, the
  same one the format claim uses).
- **Priced in, not removed** if either fails: the number then replaces §Position effect's single-order reading, and D's
  RB2 overall is quoted with the position spread beside it.
- Secondary, reported: the k-averaged pooled AUROC (rank-based, so it should be insensitive to a monotone per-position
  change of the same candidate's score — D seed 0 is .858) and the per-subset table.
- k is chosen before the run; report k. One order already exists (seed 0), so k = 4 means three new runs.

## Cost [estimate]

k × D seed 0's graphed RB2 wall = k × 1,527 s = **~1.7 h for k = 4** (~3.4 h for k = 8) on one 3090, plus minutes of
CPU for the offline average (§RewardBench 2 wall, `done/pointwise-chat-input.md`).

## Commands

```bash
for s in 0 1 2 3; do
  ./kev.sh scripts/pref/rewardbench2.py --run runs/pref/D/seed0 \
    --out runs/pref/D/seed0/rewardbench2-order$s.json --order_seed $s
done
# then average each candidate's k scores per prompt and rebuild the two position tables offline (not built)
```

## Results

Not run. Only the k = 1 reading exists (§Position effect table, `--order_seed 0`).
