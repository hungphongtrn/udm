# Question: is the training data contaminated against Decision Index items?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

No training: a data audit. **Not built in this worktree** — see Commands.

## Question

Do any training shards contain Decision Index items (or near-copies)? If they do, every external index number quoted
here is optimistic, including the ones the preference study leans on.

## Why now / evidence

`docs/studies/external_comparison.md` §3 ("Contamination audit") records:

| Statement | Detail |
|---|---|
| status | "investigated, not measured" |
| our build removes only rows matching our own held-out sets | `src/scrm_data/build.py` uses `_heldout.npy`; nothing checks DI items |
| the DI suite's own exclusions do not look at external training data | 442 rows plus scoring subsets |
| overlap surface | twelve DI panel benchmarks share upstream datasets with our sources (`docs/studies/data.md` §6) |
| named risk | `tasksource-jev` lists 670 sources including HellaSwag and Banking77; DI scores HellaSwag validation and BANKING77 test; risk highest for tasksource |
| what is needed before quoting any external comparison | exact canonical-content-key matching plus normalised-state matching of all train shards against the DI suite, reported per benchmark / source / split |

Why it matters here: the hypothesis docs quote DI numbers as their external evidence (D seed 0 36.91 on the DI 0.3
subset, +3.31 over v7-base and level with the board's r10, `done/preference-reward-model.md` §Decision Index 0.3; and
D's BANKING77 losses on both dev and the index — the same task the overlap risk names). `todo/di-full-suite.md` would
add a board-comparable number on the same basis, so this should land before that number is quoted externally.

## Arms

No arms. The audit surface, as recorded: the training shards (`src/scrm_data/build.py` and its `_heldout.npy` on the
studies branch) against the DI suite (`$DI_DIR`, the pinned kit `apolinario/decision-index@9eb2dbe`,
`kev/scripts/pref/dindex.sh`). Tier (c) below is the loose filter the Skywork v0.2 release used (n-gram overlap against
RewardBench 1) — the converter already does exact-match decontamination against RewardBench 2 prompts
(`convert_pref_data.py`), which is the same idea on a different suite and shows the intended shape.

## Decision rule (pre-registered)

- Report per benchmark / source / split the matched row counts under (a) exact canonical-content-key equality (the key
  `kev.data.load_records` hashes states with), (b) normalised-state equality, and (c) n-gram overlap, if run.
- **Clean enough to quote** if no DI scored split matches any training shard under (a) or (b).
- **Contaminated** if any does: every external number for that benchmark is then reported with the overlap count, and the
  arm's rank on that benchmark is discounted (reported, not hidden).
- Tier (c) is reported, never judged — it is a signal about paraphrase risk, not a match.
- Consequence, fixed here: if (a) or (b) hits, either drop the overlapping training rows and retrain, or stop quoting the
  affected benchmarks. No third option (re-labelling the DI items is out of scope).
- A "not measured" status stays that way until (a) and (b) have been run; this doc does not assert cleanliness.

## Cost [estimate]

**Not recorded in the repo.** CPU-only, dominated by hashing and normalising the training shards; [INFERENCE] hours at
the shard counts already recorded (`docs/studies/data.md`). Treat any number here as invented until measured.

## Commands

None in this worktree: `src/scrm_data/` and the study's `outputs/` live on the studies branch
(`docs/studies/README.md`: "How-to material ... lives in `docs/TRAINING.md` (on `claude/set-conditioned-reward-model-hvsb9s`)").
The DI suite side is reachable (`./kev.sh scripts/pref/dindex.sh setup`); the train-shard side is the blocker. Not built.

## Results

Not run ("investigated, not measured", `docs/studies/external_comparison.md` §3).
