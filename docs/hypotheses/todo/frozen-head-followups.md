# Question: do the unrun frozen-head levers beat L24 + sigmoid (DI sample 36.33 ± 0.97)?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

Mined from `docs/studies/README.md` §Open items and `frozen_lossgrid.md`. **Not built in this worktree** — those scripts
and outputs are on the studies branch.

## Question

The frozen-backbone study stopped with **L24 + sigmoid BCE, 3000 steps, 3 seeds: 36.33 ± 0.97** on the fixed
1000-request DI sample. Which of its recorded-but-unrun levers changes that, and does any of them clear the bar by more
than the seed spread?

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| headline | frozen Qwen3.5-4B L24 + set head, sigmoid loss, 3000 steps, 3 seeds: 36.33 ± 0.97 (raw 50.67) | `docs/studies/README.md` §Headline results |
| loss family | every arm containing sigmoid beats its non-sigmoid counterpart; CE-only / CE+Brier are worst (~32) | §What we concluded 2 |
| training longer does not help | 15-arm mean 32.56 (5000 steps) / 31.84 (10000); validation top1 flat 0.72-0.74 from 3000 to 10000 | §Headline, conclusions 1 and 3 |
| layer | L16 ~7 points worse (25.60); last layer 32.92 (one seed), in L24's 5000/10000-step range; last vs L24 unresolved because the L24 3000-step seed-0 mean was never pulled; sigmoid on last 33.66 (one seed) is below L24's 36.33 | §Headline; `frozen_lossgrid.md` §5 |
| selection caveat | checkpoints/lrs chosen on UDM validation loss only; DI held out and scored afterwards; in-distribution validation is a weak proxy for DI | §What we concluded 1, §Shared caveats |
| the unrun list | "Not run: L24 vs last at 3 seeds, concatenated L24 + last features, `augment_variants=true`, larger heads, full-suite DI." | `frozen_lossgrid.md` (final section) |
| the open-item list | layer-24 seed-0 mean at 3000 steps; full-suite DI; contamination audit; next levers (concatenated L24 + last features; LoRA with a CE/Brier/sigmoid objective instead of BT; a larger head, deprioritised) | `docs/studies/README.md` §Open items |
| why it still matters | the Kev-side arms that took over are LoRA + BT/pref/RCR; this is the repo's only frozen-feature comparison, and its prior is what the preference study cites (BT-only LoRA 27.61 vs frozen sigmoid 36.33) | `done/preference-reward-model.md` §Claim |

Two of the open items are already their own docs here: full-suite DI → `todo/di-full-suite.md`, the contamination audit
→ `todo/di-contamination-audit.md`.

## Arms

| Lever | Status |
|---|---|
| L24 seed-0 mean at 3000 steps (the 15-arm grid re-read, not retrained) | features cached; per-run values in `outputs/dindex_frozen/qwen3_5_4b_pack_sample-1000/runs.csv` on the 3090 box (not in git) |
| L24 vs last at 3 seeds | last-layer seed 0 exists; the rest not built |
| concatenated L24 + last features | not built |
| LoRA with CE / Brier / sigmoid instead of BT | not built here (the Kev-side recipe change is the same idea, `done/preference-reward-model.md`) |
| larger head | deprioritised by the study itself (every loss saturates at the same validation top1) |
| `augment_variants=true` | not built |

## Decision rule (pre-registered)

- Basis and comparison set: the fixed 1000-request DI sample (1006 requests, `complete=False` in every result); arm
  spreads are ±0.4-1.5 at 3 seeds and single-seed differences ≤ 2 points are not meaningful (`docs/studies/README.md`
  §Shared caveats).
- Each lever is **supported** if its 3-seed sample-index mean beats 36.33 ± 0.97 by more than the seed spread, with
  validation top1 reported beside it and **never** used for selection (the study's own rule).
- A lever that raises validation top1 and not the index is reported as such: that is the study's conclusion 1 recurring,
  and it is the study's stated failure mode, not a neutral outcome.
- The L24 seed-0 mean is the first cell: it decides whether "last ≈ L24" holds at one seed before any new arm is run.
- These levers are the old frozen stack; they do not enter the Kev-side rules (PF-RCR D8-D11, HB's rule 5) and must not
  be mixed into them.

## Cost [estimate]

**Not recorded in the repo** for these arms. The grid's unit is 15 arms × 3 seeds at 3000 steps on cached features
(`frozen_lossgrid.md` §3, seed 0 alone in §5). Any cost quoted here would be invented, so this doc leaves it open.

## Commands

None in this worktree: the study's scripts, features and `outputs/` live on the studies branch —
`docs/studies/README.md` names `outputs/lossgrid/<grid>/` and `outputs/dindex_frozen/<grid>_sample-1000/{runs.csv,summary.md}`,
and says the how-to material is `docs/TRAINING.md` on `claude/set-conditioned-reward-model-hvsb9s`. Not built here.

## Results

Not run (the study's stop line: "Stopped here (2026-10-09). Best measured configuration: L24, sigmoid loss, 3000 steps
(DI sample 36.33 ± 0.97)", `frozen_lossgrid.md`).
