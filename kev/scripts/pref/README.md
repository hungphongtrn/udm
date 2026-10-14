# Preference-first reward study: run scripts

Run on the training machine (1-2 x RTX 3090 24GB), from `kev/`. Arms and flags live only in `arms.sh`.
Every arm: Kev-4B recipe on decision-v7 (LoRA r16, 2 epochs, lr 5e-5, eff. batch 8, bf16, p_none_pair 0.25) + `--weights_dtype bf16` + `--seed`.
HP also reads the preference pairs that `convert_pref_data.py` writes (see "Pointwise chat arms" below).

| Arm | Change vs. previous |
|---|---|
| A | Kev baseline: pointer head + CE |
| B | A + `--loss pref` |
| C0 | scalar head, Kev input format, pref (optional) |
| C | scalar head, `--input_format chat`, pref |
| D / E | C + `--reg bsr` / `--reg l2`, `--reg_w 1e-3` |
| F | Stage 2: warm start from C (`--init_from`), + `--bce_w 0.5`, 1 epoch, same lr (fresh OneCycleLR) |
| G | no training: temperature + Platt fitted on the calibration split (done inside `reward_report.py`) |
| H | D + `--input_format chat_pointwise`: each option is graded on its own, the `Options:` listing is gone |
| HP | H + open human preference pairs (`--data $PREF_DATA --replay 12576`) |
| HL | H with `--loss ce`: softmax CE over a question's option rewards (reranker loss) instead of BT + BSR; probabilities via G's fitted T |
| HB | H's pointwise grader as a Yes/No question (`--input_format chat_yesno`), head started at the LM head's `W_yes - W_no` (`--head_init yesno`), class-balanced per-option BCE (`--loss bce`) |
| HBL | HB + `--ce_w 0.25` (listwise CE term) |
| Ah / Ch / Fh / HBh | A / C / F / HB on the seeded half of the training records (`--train_frac ${HALF_FRAC:-0.5}`); Fh warm-starts from Ch |
| Rh | chat, linear head, `--loss rcr --rcr_alpha 0.5` (per-option sigmoid CE + ListCE over sigmoid rewards), half data |
| R2h / Sh | Rh with `--head mlp`; Sh with `--rcr_alpha 0` (sigmoid CE only) |
| STh | Stage 2 on Sh: `--loss pref --bce_w 0.5`, 1 epoch (F's stage 2) |
| RYh | RCR on `chat_yesno` with `--head_init yesno` (linear head) |
| Dh / Hh | D / H (BT + BSR on `chat` / `chat_pointwise`) on half data: the listing-removal pair |
| PBh | `chat_pointwise`, linear head, `--loss bce` (class-balanced per-option BCE) |
| PRh / PUh | `chat_pointwise`, linear head, `--loss rcr --rcr_alpha 0.5`, `--rcr_point balanced` / `mean` |
| P2h | PRh with `--head mlp` |
| PLh | PRh's loss on `chat_yesno` with `--head_init yesno` |

The h arms are the all-wrong abstention screen ([`docs/hypotheses/allwrong-abstention.md`](../../docs/hypotheses/allwrong-abstention.md));
`scripts/abstain/queue.sh` runs them on three GPUs. `run_arm.sh` also scores every arm on the all-wrong minimal pairs
(`scripts/abstain/eval_allwrong.sh`; `ALLWRONG=0` skips it).

H and HP belong to branch `idea/pointwise-chat-input` (pre-registered in
[`docs/hypotheses/pointwise-chat-input.md`](../../docs/hypotheses/pointwise-chat-input.md), which also holds the rule and
the RewardBench 2 comparison); `wave1.sh` does not run them. H's only difference from D is the input format, HP's only
difference from H is the preference data.

## Commands, in order
```bash
cd kev
TORCH_EXTRA=cu128 scripts/pref/setup_3090.sh   # cu128 (default) or cu130; generates kev/uv.lock if missing (commit it), uv sync --extra; never `uv sync`/`uv run` without the extra afterwards (removes torch). Check the printed chat template and rendered example prompt (and that CUDA devices are listed) before anything else
.venv/bin/python -m pytest tests/test_pref.py   # CPU tests, after setup
scripts/pref/run_arm.sh C 0 0           # optional smoke test: arm, seed, GPU (train + eval); check loss goes down, then FORCE=1 to rerun or keep it

scripts/pref/wave1.sh 0                 # 1 GPU: C F A B D E, seed 0, sequentially
scripts/pref/wave1.sh 0,1,2             # N GPUs: one worker per GPU takes the next free arm (C A B D E); F runs after C on C's GPU
SEEDS="0 1 2" scripts/pref/wave1.sh 0,1 # more seeds (seeds run one after another)
RUN_C0=1 scripts/pref/wave1.sh 0,1      # also C0
RUN_F=0 scripts/pref/wave1.sh 0         # skip F (run later: scripts/pref/run_arm.sh F 0 0 after C is done)
# Resume a stopped wave: rerun the same command. Arms with head.pt are evaluated only (scored splits skipped); an arm
# stopped mid-training has no head.pt and cannot resume (LoRA): delete its runs/pref/ARM/seedN first, then it retrains.

# G: eval_arm.sh already runs the calibration split and passes it to reward_report.py; to redo by hand for any arm:
.venv/bin/python scripts/pref/reward_report.py runs/pref/C/seed0/eval/decision-v7-development --calib runs/pref/C/seed0/eval/decision-v7-calibration

# Download results: logs, training metrics, eval outputs (no weights) into one tgz; prints each run's splits and G's n_calib
scripts/pref/pack_results.sh pref-AE.tgz

# FINAL VERDICT ONLY (locked test split, read once):
TEST=1 scripts/pref/eval_arm.sh runs/pref/C/seed0     # each finished run; skips splits already done
```

## Pointwise chat arms (H, HP; branch `idea/pointwise-chat-input`)
`--input_format chat_pointwise` renders the same chat prompt as `chat` with the `Options:` listing removed, so a response
is graded on its own (state + instruction + that option): the pointwise score Bradley-Terry trains on, and the score a
reward model is asked for. It is recorded in the checkpoint like `chat` (`input_format`, `chat_template_sha256`) and every
format check accepts it (`kev.model.CHAT_FORMATS`). Recipe, rule and thresholds:
[`docs/hypotheses/pointwise-chat-input.md`](../../docs/hypotheses/pointwise-chat-input.md).
```bash
# HP's data (once, CPU): HelpSteer3 preference pairs -> Kev --data records + a manifest (dataset revision, admission
# counts and drop reasons, sha256). Admission is the trainer's own rule with kev.suite.ADMISSION_BRANCH_HEADROOM reserved
# on the branch (a training augment can add an option), plus the chat row limit of the format the arm trains. kev.train's
# own --data re-filter reserves the headroom too, so the rule holds for any data file.
# Exact-match decontamination against RewardBench 2 prompts.
.venv/bin/python scripts/pref/convert_pref_data.py --out runs/pref/data/pref-helpsteer3.jsonl

# H vs D is one flag; HP vs H is the data file (PREF_REPLAY defaults to the whole 12,576-record v7 train partition)
BATCH_FALLBACK=1 scripts/pref/run_arm.sh H 0 0
BATCH_FALLBACK=1 PREF_DATA=runs/pref/data/pref-helpsteer3.jsonl scripts/pref/run_arm.sh HP 0 0
BATCH_FALLBACK=1 scripts/pref/run_arm.sh HB 0 0
BATCH_FALLBACK=1 scripts/pref/run_arm.sh HBL 0 0
scripts/pref/eval_arm.sh runs/pref/H/seed0                    # run_arm.sh already evals; this redoes/reports one run

# RewardBench 2 (best-of-4 + Ties) for any arm's checkpoint: A (reference), D (format comparator), H, HP. The layout is
# the checkpoint's own: chat_pointwise scores each candidate on its own, chat/kev pass a prompt's candidates as one
# question's options in a seeded per-prompt shuffle (--order_seed 0; -1 = the dataset's chosen-first order).
# Same scoring as allenai/reward-bench (utils.py: reroll_and_score_dataset, process_single_model),
# six-subset mean, plus the set-independence probe (--probe 64). An unsupported candidate (over context / OOM) is never
# truncated and never scored around: a non-Ties prompt with any unsupported candidate earns 0, a Ties row with one keeps
# its place in every denominator but contributes nothing, and the probe skips such a prompt (counts in the JSON).
.venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/H/seed0 --out runs/pref/H/seed0/rewardbench2.json
# Pooled (absolute-scale) metrics on a RewardBench 2 JSON: per subset + overall pooled AUROC, sigmoid ECE, prior-adjusted
# ECE; --report adds Platt ECE with G's (a, b). Writes <json stem>.pooled.json. Dev-set equivalents are in reward_report.py.
.venv/bin/python scripts/pref/rb2_pooled.py runs/pref/H/seed0/rewardbench2.json --report runs/pref/H/seed0/eval/decision-v7-development/reward_report.json
```

## Decision Index 0.3 (arms D / H; secondary, external)
`dindex.sh` runs the public Decision Index kit (github.com/apolinario/decision-index, pinned commit in the script) on a
checkpoint through `dindex_engine.py:KevEngine` (kev.predictors.LocalPredictor, serving context, bf16; over-context / OOM /
>255-option requests are refused as unsupported = counted wrong, never truncated).
```bash
hf auth login                                   # and accept https://huggingface.co/datasets/cais/hle terms first
scripts/pref/dindex.sh setup                    # clone kit to ~/decision-index (DI_DIR), install into .venv, build suite (~7 GB dl, 17 GB disk)
scripts/pref/dindex.sh sample runs/pref/D/seed0 # 100 requests: smoke + latency -> runs/pref/D/seed0/dindex-0.3-sample/
scripts/pref/dindex.sh check runs/pref/D/seed0  # everything below in one go: r10 vs board, v7-base, D, retemper, pairs; rerun to resume
scripts/pref/dindex.sh subset runs/pref/D/seed0                                # shared subset -> runs/pref/D/seed0/dindex-0.3-sub15000/
scripts/pref/dindex.sh subset jaredpalmer/kev-4b@v7-base kev4b-v7base           # comparator on the same rows -> runs/dindex/kev4b-v7base-sub15000/
.venv/bin/python scripts/pref/dindex_subset.py pair runs/dindex/kev4b-v7base-sub15000 runs/pref/D/seed0/dindex-0.3-sub15000  # D - v7-base
scripts/pref/dindex.sh full runs/pref/D/seed0                                  # whole suite (140,620 requests): board-comparable only
```
D is compared with `v7-base`, not with the board, so `subset` is the default: `N` (15000) rows of the index's 37
benchmarks (excluded rows dropped, complete case groups, round-robin over benchmarks, so ~400 cases each; the kit's
`suite.sample.stratified_sample`), the same file (`$DI_DIR/subset-0.3-$N.jsonl.gz`) for every checkpoint.
`dindex_subset.py score` runs the kit's scorers, chance levels and area weights with the suite restricted to those rows,
so `index.json` is the index estimated on the subset: a paired D vs v7-base comparison, not a board number. Rows a
checkpoint already answered (ok / abstained) in `RUN/dindex-0.3` (a `full` run, finished or stopped) are copied into the
subset run on every `subset` call instead of being rerun, if the full run used the same engine options (else it refuses:
e.g. a different `TEMP`). The subset holds ~15,000 of the suite's 140,620 rows (all of the small benchmarks, ~400 cases
of each large one), so only the full run's rows that fall in it are reused.
Reproduction check: the board's Kev-4B r10 is Hub `jaredpalmer/kev-4b` at `139fdd94` (weights unchanged in later
commits), so run it on the same subset and compare it with its own board row (per-benchmark native scores; the board
publishes no per-request results):
```bash
scripts/pref/dindex.sh subset jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 kev4b-r10
.venv/bin/python scripts/pref/dindex_subset.py compare runs/dindex/kev4b-r10-sub15000 kev-4b-r10   # -> table + board-compare-kev-4b-r10.json
```
Temperature is a scoring choice, not a run choice: `dindex_subset.py retemper ROWS OUT T..` (from `$DI_DIR`) rescores a
subset run at other temperatures from its saved probabilities (exact: T_run * log p = reward + const per question; the
top choice never moves, so only probability-sensitive metrics change). D's fitted T comes from decision-v7, and nothing
says it fits Decision Index, so `check` pairs D with v7-base three ways: shipped (v7-base's own T vs D at `D_T`, default
1.47 = seed 0's G), raw (both T = 1), and the best T of `TGRID` for each, chosen on the subset itself (optimistic). If
the winner changes across the three, temperature decides it, not the training loss. Compare D with `v7-base`, not with
the board's Kev-4B (r10 adds three training stages after the base recipe and T = 2.41).
Chat checkpoints on the hybrid backbone score through `DecisionModel.chat_hidden`'s passes: the state runs once per
request, each question continues it with its instruction (and its option listing, for the `chat` format; a `chat_pointwise`
checkpoint grades each option on its own), and its option rows continue that (the old path
reran the state per question). On CUDA those passes are replayed as CUDA graphs (`kev.cuda_graphs.chat`, on by default;
`GRAPHS=0` opts out, see "Chat checkpoints: inference speed" below), and eagerly where a record does not fit the graphed
buffers. The questions' prefixes stay single unpadded passes there, as in `chat_hidden`: their states are what the option
rows run from, and a pass padded to a bucket would leave a DeltaNet one advanced past the prefix's last token (the pad inputs
are zeroed, but the decay still advances). Either way the rows are the same tokens at the same positions, so rewards differ
by bf16 kernel rounding only and rows scored before the change stay valid for a resume. Requests that still run out of memory
(many options: CLINC150, API-Bank, ToolRet) are rerun with the option rows in chunks (exact per option up to kernel rounding;
the sample run's
`raw_output.fallback_chunk` shows it, `full` runs `--compact` and keeps no raw output). A resume keeps `unsupported` rows; to retry them, drop them from
`results.jsonl` first (backup kept):
`.venv/bin/python -c "import json,shutil,sys; f=sys.argv[1]; shutil.copy(f,f+'.bak'); open(f,'w').writelines(l for l in open(f+'.bak') if l.endswith('\n') and json.loads(l)['status']!='unsupported')" runs/pref/D/seed0/dindex-0.3/results.jsonl`

## Chat checkpoints: inference speed
The chat format's serving cost was its one drawback: on the DI subset a chat checkpoint was 3.7x slower than a Kev-format
one (arm D 27,246 s vs r10 6,048 s / v7-base 7,367 s, 15,000 requests, one 3090, bf16). The eager chat path
(`DecisionModel.chat_hidden`) ran the record's state once and then, per question, a pass over that question's own prefix
plus one padded pass per `kev.model.rows_per_pass` option rows, each pass issuing its ~2,000 kernels from Python, with the
question's cached prefix replicated per row chunk. The same tokens now run as graphed passes over the Kev format's own
buffers (`kev.cuda_graphs.chat`: the state into a bank entry, the questions' prefixes as eager unpadded row passes, one per
distinct prefix length -- their states are kept, so they cannot be batched under one bucket, each a few hundred tokens
against the option rows -- then one graphed row pass for all their option rows; a single-question record's prefix *is* its
state pass). No prefix replication, and the option rows -- the bulk -- cost one launch instead of one padded pass per row
chunk per question. `dindex_engine` turns it on for chat checkpoints on CUDA (as `kev.serve` does for the requests it replays);
`GRAPHS=0` opts out, Kev-format checkpoints keep their row-form path (which never touches the graphs, so this engine leaves
them without the ~3 GB of buffers), and a record the graphed buffers do not admit (state, a question's own tokens or an
option row past the buckets) falls back to the eager passes inside the same run.

Verify on the training machine before trusting a subset number:
```bash
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0                  # 20 DI subset rows, both paths, per-record ms + max |d reward|
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --n 100          # more rows; --offset for another slice
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --suite evals/decision-v7 --split calibration
# the fp32 reference (smaller graph buffers to fit 24 GB), then bf16 against it: is graphed further from it than eager?
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --n 100 --dtype fp32 --bank-width 1024 --graph-states 8 --graph-rows 8 --graph-tokens 8192 --dump fp32.json
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --n 100 --against fp32.json
.venv/bin/python scripts/pref/chat_speed_check.py runs/pref/H/seed0                  # the pointwise layout: same passes, no option listing in the prefix
scripts/pref/dindex.sh subset runs/pref/D/seed0 graphed     # the default (graphed passes); NAME = a fresh out dir, so nothing is seeded
GRAPHS=0 scripts/pref/dindex.sh subset runs/pref/D/seed0 eager   # the old eager path on the same rows, for the comparison
```
`chat_pointwise` (arms H / HP) is the same layout minus the `Options:` listing: the graphed passes read `chat_splits`'
prefix and branch ids, so the pointwise prefix flows through them unchanged and `dindex_engine` turns the graphs on for it
by the same rule (`input_format in kev.model.CHAT_FORMATS`, `GRAPHS=0` opts out). `rewardbench2.py` follows the same rule.
`chat_speed_check.py` scores each record twice through `dindex_engine.KevEngine` with one loaded model (eager first, then
graphed, capturing buckets as the engine does), so a record that runs out of memory takes the engine's chunked path in its
timing, as in a subset run, and prints per-record and per-second rates, the projected 15,000-row subset time for both paths,
how many records each path chunked or refused, GPU memory after loading and at peak, max |d
reward| (the head's logits) / max |d prob| and answer flips, split by path (graphed / chunked / eager both ways) with the
worst records and an eager rerun as the run-to-run floor. Measured on D seed 0, 100 subset rows, a 3090 (bf16): eager
repeats to 0, graphed differs by up to 0.39 in reward / 0.077 in probability with 2 flips in 790 questions, the chunked
path by 0.09 -- more than the ~1e-3 first expected, and not yet attributed to rounding or a bug: the fp32 pair above
decides it (fp32 eager-vs-graphed should be ~1e-4; bf16 graphed should be no further from fp32 eager than bf16 eager is).
Speed there: graphed 1.61x eager (5.7 h vs 9.1 h projected for that slice; mean 6x the median, long eager states
dominate), not the ~2 h first targeted from the Kev format's subset average (~0.42 s against r10's 6,048 s).
The subset re-measure: the kit's `complete` event in
`RUN/dindex-0.3-sub$N/run.log` carries `elapsed_seconds` (per-row `total_wall_ms` is in `results.jsonl`); a subset run
seeded from a `full` run of the same checkpoint copies that run's already-answered rows, so time a fresh `NAME` to see the
new path's own speed.

## Env vars
- `PYTORCH_ALLOC_CONF`: run_arm.sh defaults it to `expandable_segments:True` (allocator fragmentation on the chat-format arms; no effect on results).
- Batch: every arm takes 8 records per optimizer step. A B C0 run 4 per forward x `--accum 2`; the chat-format arms C D E F H HP run 2 x `--accum 4`, because at 4 they run out of memory on a 24 GB 3090. The gradient does not depend on the split (train.py divides each pass's loss by the step's record count), so A/B trained at 4 x 2 stay comparable: with LoRA dropout off, a CPU check gave the same step-1 gradient norm for 4 x 2, 2 x 4 and 1 x 8 on A B C and C+BCE. Two caveats: LoRA dropout (0.05) draws different masks for a different split, which is noise like a different seed, not a bias; and the `--reg` arms' term is computed per forward pass, so their gradient changes with the split (+0.2% at step 1). Keep the `--reg` arms' (D E H HP) split the same across seeds.
- `BATCH_FALLBACK=1`: halves every arm's microbatch (Kev 2 x 4, chat 1 x 8) if the defaults still run out of memory.
- `FORCE=1`: delete an existing `runs/pref/ARM/seedN` and retrain it (applies to every arm a wave1.sh run touches). Without it run_arm.sh evaluates a trained run (head.pt) only and refuses an unfinished one.
- `BCE_W` (default 0.5), `F_EPOCHS` (default 1): arm F.
- `GRAPHS=0`: dindex.sh, the eager chat path instead of `kev.cuda_graphs` (unset: on for a chat checkpoint on CUDA). A set value joins the engine options, so a subset run will not seed from a `full` run made without it.
- `PREF_DATA` (arm HP, required): the JSONL `convert_pref_data.py` wrote. `PREF_REPLAY` (default 12576): how many of decision-v7's training records are mixed back in with `--replay`.
- `SEEDS`, `RUN_C0`, `RUN_F`: wave1.sh. `TEST=1`: eval on the locked test split.

## Outputs
```
runs/pref/<ARM>/seed<N>/              checkpoint (head.pt, adapter, tokenizer), train.log (also runs/pref/<ARM>/seed<N>.train.log)
  eval/<suite>-<split>/                rows.json report.json predictions.jsonl (kev.benchmark), reward_report.json (our table)
  rewardbench2.json                    rewardbench2.py: per-subset + overall accuracy, unsupported counts, wall/token totals, set-independence probe
runs/pref/data/<name>.jsonl            convert_pref_data.py's --data file (gitignored) and <name>.jsonl.manifest.json (its provenance)
```
Splits: `decision-v7-{calibration,development}`, `transfer-v4-development` (+ `-test` with TEST=1). transfer-v4 is eval-only: its calibration split is empty by design.
`reward_report.json`: `raw` (by type and overall: top1, pairwise_acc, reward stats, nll, ece_softmax, ece_sigmoid), `permutation`, and `G` (fitted T and Platt a,b; metrics at fitted T / Platt).
Platt and T are fit per run on that run's decision-v7 calibration rows and applied to its development, transfer-v4 and test rows; decision rule should compare `G` numbers across arms for calibration, raw numbers for ranking quality.

## Compute
~2-3 h per arm on one 3090 [estimate, unmeasured; the released recipe is ~1 h on an H100], plus ~15-30 min eval [estimate]. Wave 1 seed 0 on one GPU: ~12-18 h; on two GPUs ~7-10 h; on three ~5-7 h [estimate: two runs on the busiest GPU]. F is ~half a run.
H is one chat-format run (D measured 9.1 h) [estimate]; HP adds ~12,000 records to H's 12,576, so ~18 h [estimate].
RewardBench 2 is 1,865 prompts: ~1-1.5 h for a set-relative arm (A, D) and ~5-8 h for a pointwise one (H, HP, one request per candidate), plus ~10 min of probe.
First step pays Triton compilation. The log is also at `runs/pref/<ARM>/seed<N>.train.log` if the run dir was removed.
