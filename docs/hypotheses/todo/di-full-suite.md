# Question: a board-comparable full-suite Decision Index for the final head

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

No training: one scoring run on the winning checkpoint.

## Question

What is the final head's index on the **whole** DI 0.3 suite (140,620 requests), so the number is board-comparable
rather than the 15,000-row subset estimate?

## Why now / evidence

| Statement | Source |
|---|---|
| "DI only for the final winner (cost: ~15-30 h per full-suite pass with the HF backend on one 3090 [INFERENCE], see studies/README.md open items)" | `done/preference-reward-model.md` §Metrics |
| "Full-suite DI (~150k requests; estimated 15-30 h with the HF backend on one 3090 [INFERENCE]) for the final head" | `docs/studies/README.md` §Open items |
| "`dindex.sh full runs/pref/D/seed0` — whole suite (140,620 requests): board-comparable only" | `kev/scripts/pref/README.md` §Decision Index 0.3 |
| the subset is a paired estimate, not a board number | same ("the same file for every checkpoint", kit scorers with the suite restricted to those rows) |
| board context to compare against (public, not reproduced by us): Kev-4B r10 37.95; Jobe 32.35; Decider 4B 40.70; JPT-4B 43.04; Nox 42.55 (Vela reproduction) / 43.8 (card) | `done/preference-reward-model.md` §Decision Index, `docs/studies/README.md` §Headline results |
| why r10 is not a like-for-like comparator: later stages (documents-v1, hard/devtools skills) and T = 2.41 after the base recipe | `done/preference-reward-model.md` §Decision Index |
| the result is currently unknown: D's subset 36.91 is +3.31 over v7-base, but the full-suite number has never been run | same |

## Arms

No arms. The checkpoint is "the final winner": the allwrong doc's D9-D11 winner if the queue lands one (P2h is the
priority, `todo/pointwise-rcr-mlp.md`), otherwise D seed 0. **Not yet chosen** — record the choice (checkpoint path +
seed) with the result.

## Decision rule (pre-registered)

- Report `index.json` with `complete=True`, per-benchmark scores, and the unsupported-row counts; quote the board's
  published numbers beside it and say explicitly which of them are not reproduced here.
- A full-suite number is reported as **the final head's number** only if the checkpoint is the pre-registered winner and
  its v7 / t4 dev top1 is reported in the same place — dev and the index disagreed in sign once already (D vs A: v7
  −0.79, index +3.31 over v7-base).
- No threshold and no claim: this is a reporting run. It does not enter any hypothesis's decision rule.
- Order: run `todo/di-attribution-a-subset.md` first, so the 15-30 h is spent on a checkpoint whose gain is already
  attributed.
- A partial or interrupted run is reported as such (the script resumes; rows already answered are copied, not rerun).

## Cost [estimate]

**15-30 h on one 3090** (the doc's [INFERENCE]), occupying one lane for that time. The DI runner is serial per
checkpoint, so the run cannot be split across the three GPUs.

## Commands

```bash
./kev.sh scripts/pref/dindex.sh full runs/pref/D/seed0     # whole suite; board-comparable only
```

## Results

Not run.
