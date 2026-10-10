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

scripts/pref/wave1.sh 0                 # 1 GPU: A B C D E then F, seed 0, sequentially
scripts/pref/wave1.sh 0,1               # 2 GPUs: queue GPU0 = A C F, queue GPU1 = B D E
SEEDS="0 1 2" scripts/pref/wave1.sh 0,1 # more seeds (seeds run one after another)
RUN_C0=1 scripts/pref/wave1.sh 0,1      # also C0
RUN_F=0 scripts/pref/wave1.sh 0         # skip F (run later: scripts/pref/run_arm.sh F 0 0 after C is done)

# G: eval_arm.sh already runs the calibration split and passes it to reward_report.py; to redo by hand for any arm:
.venv/bin/python scripts/pref/reward_report.py runs/pref/C/seed0/eval/decision-v7-development --calib runs/pref/C/seed0/eval/decision-v7-calibration

# FINAL VERDICT ONLY (locked test split, read once):
TEST=1 scripts/pref/eval_arm.sh runs/pref/C/seed0     # each finished run; skips splits already done
```

## Env vars
- `BATCH_FALLBACK=1`: `--batch 2 --accum 4` (same effective batch 8) if batch 4 OOMs on 24GB. Use it for every arm of a comparison, not just the one that OOMed.
- `FORCE=1`: delete an existing `runs/pref/ARM/seedN` and rerun (without it run_arm.sh refuses if the dir exists, finished or not).
- `BCE_W` (default 0.5), `F_EPOCHS` (default 1): arm F.
- `SEEDS`, `RUN_C0`, `RUN_F`: wave1.sh. `TEST=1`: eval on the locked test split.

## Outputs
```
runs/pref/<ARM>/seed<N>/              checkpoint (head.pt, adapter, tokenizer), train.log (also runs/pref/<ARM>/seed<N>.train.log)
  eval/<suite>-<split>/                rows.json report.json predictions.jsonl (kev.benchmark), reward_report.json (our table)
```
Splits: `decision-v7-{calibration,development}`, `transfer-v4-{calibration,development}` (+ `-test` with TEST=1).
`reward_report.json`: `raw` (by type and overall: top1, pairwise_acc, reward stats, nll, ece_softmax, ece_sigmoid), `permutation`, and `G` (fitted T and Platt a,b; metrics at fitted T / Platt).
Platt and T are fit per run on that run's calibration rows and applied to its development rows; decision rule should compare `G` numbers across arms for calibration, raw numbers for ranking quality.

## Compute
~2-3 h per arm on one 3090 [estimate, unmeasured; the released recipe is ~1 h on an H100], plus ~15-30 min eval [estimate]. Wave 1 seed 0 on one GPU: ~12-18 h; on two GPUs ~7-10 h. F is ~half a run.
First step pays Triton compilation. The log is also at `runs/pref/<ARM>/seed<N>.train.log` if the run dir was removed.
