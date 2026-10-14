# Question: which recorded chat-serving idea pays off, and what does a graphed D subset actually cost?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

No training: serving measurements on existing checkpoints.

## Question

The chat format's one clear cost was serving time: on the DI 0.3 subset D ran **3.7x** the Kev format's wall time with
the eager chat path. A graphed path (`kev.cuda_graphs.chat`) was built and measured once; the doc's own re-measure is
"not run yet". Question: what does the graphed D subset actually cost, and which of the recorded ideas (2-8 below) pays
off next? Idea 1 (the pointwise prompt) is closed: it was run as arm H and refuted on dev accuracy.

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| the cost | D 27,246 s for the 15,000 rows vs 6,048 s (r10) and 7,367 s (v7-base), one 3090, eager | `done/preference-reward-model.md` §Inference speed |
| the first graphed reading | index 37.01 vs eager 36.91; 586 of 15,000 requests refused then rerun with graph release (3,450 s); graphed wall ~15-18.6k s ≈ 4.2-5.2 h = 1.5-1.8x eager, still 2-3x the Kev format | same |
| kernel rounding | 372 of 123,375 answers changed (0.30%), max \|Δp\| 0.15 (ANLI); ChessBench 4.9%, POP909-CL 3.8%, GSM8K 3.2% | same |
| the Kev-format target | ~0.40 s/request on the same rows [INFERENCE] | same |
| idea 1 is closed | H (pointwise prompt) bought set-independence but cost dev top1 (v7 −1.58, t4 −1.22) and was refuted | `done/pointwise-chat-input.md` §Verdict |
| the deferred ideas, ranked with gains and risks | items 2-8 | `done/preference-reward-model.md` §Inference speed |

## Arms

No arms, no training. The unrun queue as recorded (each with its [INFERENCE] gain and risk in
`done/preference-reward-model.md` §Inference speed):

| # | Idea | Recorded [INFERENCE] gain / risk |
|---|---|---|
| 0 | the graphed-path re-measure itself (the pending measurement) | expected at or under the Kev format's ~0.40 s/request; D's subset well under 27,246 s |
| 2 | graph the questions' prefixes at exact length instead of a bucket | a few tens of ms per many-question record; risk: graph keys multiply on distinct prefix lengths (~0.4 s capture each) |
| 3 | prefix-cache the state across requests (`kev.serve`'s cache) | ~0 on the DI subset (one state per row); large for many questions about one document |
| 4 | batch several requests into one pass (`GRAPH_STATES=16` states or `GRAPH_ROWS=32` rows per replay) | sizable when many small records arrive together; serving-side only (the DI runner is serial) |
| 5 | one row pass for the option rows of several records | ~10-30% if pass launches dominate; risk: per-record bookkeeping, records must share a row bucket |
| 6 | varlen / flash kernels instead of padded rows | a few percent; dynamic shapes cannot be captured as graphs |
| 7 | int8 / fp8 weights or KV cache | 1.5-2x on memory-bound passes; risk: every reported number and the Kev-format comparisons would be re-derived |
| 8 | speculative early exit | the largest in principle; risk: changes the readouts, so it is an accuracy study first |

## Decision rule (pre-registered)

- **First, the pending measurement's own rule** (the doc's, unchanged): `chat_speed_check.py runs/pref/D/seed0 --n 100`
  prints both clocks and the max |Δ reward| per record — the graphed pass must agree with the eager one up to
  floating-point reassociation, and the subset must land at or under the Kev format's ~0.40 s/request. If it does not,
  that is a regression to report before any further idea; run with a fresh `NAME` so nothing is seeded from a past
  `full` run.
- **Then, per idea:** adopted only if it beats the current graphed clock on the same rows by more than the run-to-run
  spread of two runs of the same command, **and** the rewards it serves are unchanged up to kernel rounding (the check
  above; `kev/tests/test_pref.py::test_chat_graphs_match_chat_hidden` is the CUDA-only invariant).
- Every idea that can change a readout (item 8; item 7's accuracy risk) is an accuracy study first, never a speed fix:
  report dev top1 / RB2 before and after, and keep it out of the speed number.
- Report wall time in the same units everywhere: seconds for the 15,000 rows and per-record ms from the check.
- Serving work never enters a hypothesis's decision rule; it is a cost line.

## Cost [estimate]

- The re-measure: **~4.2-5.2 h** per full subset pass (D's graphed wall, §Inference speed) plus ~10 min for the check.
- Each further idea is an engine change with no training; its cost is not recorded in the repo, so this doc does not
  quote one.

## Commands

```bash
./kev.sh scripts/pref/chat_speed_check.py runs/pref/D/seed0 --n 100   # both paths, per-record ms, max |Δ reward|
./kev.sh scripts/pref/dindex.sh subset runs/pref/D/seed0 graphed      # default; run.log's `complete` event has elapsed_seconds
GRAPHS=0 ./kev.sh scripts/pref/dindex.sh subset runs/pref/D/seed0 eager
```

## Results

Not run ("Re-measure (not run yet ...)", §Inference speed). The earlier graphed run and the eager baseline exist; the
table above quotes them.
