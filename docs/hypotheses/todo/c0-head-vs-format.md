# Question: did the scalar head or the chat format cost C its 0.5 / 1.4 points vs B?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

One arm (`C0`) was defined for this and never run.

## Question

Arm B (pointer head, Kev format, `--loss pref`) and arm C (scalar head, chat format, `--loss pref`) differ in **two**
things at once — head and input format. C0 (scalar head, Kev format, pref) is the missing cell that separates them.
Question: which of the two explains C's deficit against B?

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| B vs C top1 | C −0.5 v7 / −1.4 t4 below B | `done/preference-reward-model.md` §Readings ("B vs C (head + format)") |
| C vs A, v7 | −1.19, CI [−2.20, −0.08] excludes 0 (test sampling only) | §Wave 1 paired top1 differences |
| B vs A | −0.71 v7 / +1.52 t4 (within noise) | same |
| the statement | "C0 was not run, so head and format are not separated" | §Readings |
| arm definition | C0 = `--head scalar --input_format kev --loss pref` | §Arms; `arms.sh` `C0` |
| sub-claim at stake | "B vs C: head + input format" (the study's table lists C0 as the splitter) | §Claim |

Why now: the head question is live again as the MLP-head ablation (P2h vs PRh, `todo/pointwise-rcr-mlp.md`) and the
format question is what `done/pointwise-chat-input.md` refuted at seed 0 — C0 is the one cheap cell that says whether
"the scalar head costs something on Kev's format" is even true, on an arm that already exists.

## Arms

| Arm | Flags | Status |
|---|---|---|
| C0 | `--head scalar --input_format kev --loss pref` (Kev-format microbatch: `_KEV` = 4 × accum 2) | built (`arms.sh`), not run |

## Decision rule (pre-registered, seed 0)

Noise scale is the parent study's rule 1: "a top1 difference < 2 points on a single seed is noise" (its seed spreads are
±0.4-1.5, `docs/studies/README.md` §Shared caveats).

- **C0 vs B** isolates the head; **C vs C0** isolates the format. Whichever difference is ≥ 2 points on v7 or t4 names
  the cause.
- Both < 2 points → **inconclusive at one seed**; seeds 1-2 would be needed for a < 2-point call, exactly as the parent's
  rule 2 requires, and are not run here.
- Report C0's dev top1 on both sets beside B / C / A seed 0, the paired bootstrap differences where the parent's method
  applies (5,000 resamples of record groups, test sampling only), C0's reward scale (v7 |r| p99 — the head is also the
  scale story: A 15.8, C 10.6, D 7.1) and its ECE.
- This doc fills one cell of the parent's arm table; it does not re-score the parent verdict (rule 5, **inconclusive**).
- If C0 lands within noise of both B and C, the honest reading is that on one seed the head/format split is unresolved,
  and the AGENTS-level conclusion ("head and format are not separated") stands with a measured null, not an unfilled
  cell.

## Cost [estimate]

**~2-3 h** on one 3090 (`done/preference-reward-model.md` §Compute: "~2-3 h per run"); C0 is a Kev-format arm at
4 × accum 2 like A / B, whose seed-0 wall was 2.4 h.

## Commands

```bash
./kev.sh scripts/pref/run_arm.sh C0 0 0          # train + eval (eval_arm.sh)
RUN_C0=1 ./kev.sh scripts/pref/wave1.sh 0,1      # or inside a wave; C0 is skipped unless RUN_C0=1
```

## Results

Not run.
