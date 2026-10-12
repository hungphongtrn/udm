# Preference-first reward study: run scripts

Run on the training machine (1-2 x RTX 3090 24GB), from `kev/`. Arms and flags live only in `arms.sh`.
Every arm: Kev-4B recipe on decision-v7 (LoRA r16, 2 epochs, lr 5e-5, eff. batch 8, bf16, p_none_pair 0.25) + `--weights_dtype bf16` + `--seed`.

| Arm | Change vs. previous |
|---|---|
| A | Kev baseline: pointer head + CE |
| B | A + `--loss pref` |
| C0 | scalar head, Kev input format, pref (optional) |
| C | scalar head, `--input_format chat`, pref |
| D / E | C + `--reg bsr` / `--reg l2`, `--reg_w 1e-3` |
| F | Stage 2: warm start from C (`--init_from`), + `--bce_w 0.5`, 1 epoch, same lr (fresh OneCycleLR) |
| G | no training: temperature + Platt fitted on the calibration split (done inside `reward_report.py`) |

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

## Decision Index 0.3 (arm D; secondary, external)
`dindex.sh` runs the public Decision Index kit (github.com/apolinario/decision-index, pinned commit in the script) on a
checkpoint through `dindex_engine.py:KevEngine` (kev.predictors.LocalPredictor, serving context, bf16; over-context / OOM /
>255-option requests are refused as unsupported = counted wrong, never truncated).
```bash
hf auth login                                   # and accept https://huggingface.co/datasets/cais/hle terms first
scripts/pref/dindex.sh setup                    # clone kit to ~/decision-index (DI_DIR), install into .venv, build suite (~7 GB dl, 17 GB disk)
scripts/pref/dindex.sh sample runs/pref/D/seed0 # 100 requests: smoke + latency -> runs/pref/D/seed0/dindex-0.3-sample/
scripts/pref/dindex.sh check runs/pref/D/seed0  # everything below in one go (r10 reproduction check, D, v7-base, pair); rerun to resume
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
`TEMP=1.47` scores D seed 0 at its G temperature instead of the checkpoint's own. Compare D with `v7-base`, not with the
board's Kev-4B (r10 adds three training stages after the base recipe and T = 2.41).
Chat checkpoints that run out of memory (many-option requests: CLINC150, API-Bank, ToolRet) are rerun per question with
option branches in chunks (exact per option up to kernel rounding; the sample run's `raw_output.fallback_chunk` shows it,
`full` runs `--compact` and keeps no raw output). A resume keeps `unsupported` rows; to retry them, drop them from
`results.jsonl` first (backup kept):
`.venv/bin/python -c "import json,shutil,sys; f=sys.argv[1]; shutil.copy(f,f+'.bak'); open(f,'w').writelines(l for l in open(f+'.bak') if l.endswith('\n') and json.loads(l)['status']!='unsupported')" runs/pref/D/seed0/dindex-0.3/results.jsonl`

## Env vars
- `PYTORCH_ALLOC_CONF`: run_arm.sh defaults it to `expandable_segments:True` (allocator fragmentation on the chat-format arms; no effect on results).
- Batch: every arm takes 8 records per optimizer step. A B C0 run 4 per forward x `--accum 2`; the chat-format arms C D E F run 2 x `--accum 4`, because at 4 they run out of memory on a 24 GB 3090. The gradient does not depend on the split (train.py divides each pass's loss by the step's record count), so A/B trained at 4 x 2 stay comparable: with LoRA dropout off, a CPU check gave the same step-1 gradient norm for 4 x 2, 2 x 4 and 1 x 8 on A B C and C+BCE. Two caveats: LoRA dropout (0.05) draws different masks for a different split, which is noise like a different seed, not a bias; and D/E's `--reg` is computed per forward pass, so their gradient changes with the split (+0.2% at step 1). Keep D/E's split the same across seeds.
- `BATCH_FALLBACK=1`: halves every arm's microbatch (Kev 2 x 4, chat 1 x 8) if the defaults still run out of memory.
- `FORCE=1`: delete an existing `runs/pref/ARM/seedN` and retrain it (applies to every arm a wave1.sh run touches). Without it run_arm.sh evaluates a trained run (head.pt) only and refuses an unfinished one.
- `BCE_W` (default 0.5), `F_EPOCHS` (default 1): arm F.
- `SEEDS`, `RUN_C0`, `RUN_F`: wave1.sh. `TEST=1`: eval on the locked test split.

## Outputs
```
runs/pref/<ARM>/seed<N>/              checkpoint (head.pt, adapter, tokenizer), train.log (also runs/pref/<ARM>/seed<N>.train.log)
  eval/<suite>-<split>/                rows.json report.json predictions.jsonl (kev.benchmark), reward_report.json (our table)
```
Splits: `decision-v7-{calibration,development}`, `transfer-v4-development` (+ `-test` with TEST=1). transfer-v4 is eval-only: its calibration split is empty by design.
`reward_report.json`: `raw` (by type and overall: top1, pairwise_acc, reward stats, nll, ece_softmax, ece_sigmoid), `permutation`, and `G` (fitted T and Platt a,b; metrics at fitted T / Platt).
Platt and T are fit per run on that run's decision-v7 calibration rows and applied to its development, transfer-v4 and test rows; decision rule should compare `G` numbers across arms for calibration, raw numbers for ranking quality.

## Compute
~2-3 h per arm on one 3090 [estimate, unmeasured; the released recipe is ~1 h on an H100], plus ~15-30 min eval [estimate]. Wave 1 seed 0 on one GPU: ~12-18 h; on two GPUs ~7-10 h; on three ~5-7 h [estimate: two runs on the busiest GPU]. F is ~half a run.
First step pays Triton compilation. The log is also at `runs/pref/<ARM>/seed<N>.train.log` if the run dir was removed.
