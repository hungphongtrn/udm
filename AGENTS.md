# AGENTS.md

Read the relevant doc before changing code or proposing experiments.

## Machines

This dev machine is for editing only: NEVER install packages (`uv sync`, `pip`, ...), create a venv, or run the repo's
code or tests here, not even CPU smoke runs. Write the change and hand the exact command to the user, who runs it on the
training machine and reports the output.

The training machine has 3x RTX 3090 (24 GB): plan work as up to three experiments at once, one per GPU.

## Working directory

Work from the repository root. The Kev fork's code, `.venv` and `runs/` stay in `kev/`; run any Kev command from the
root through `./kev.sh <command>` (paths after it are kev-relative, e.g. `./kev.sh scripts/pref/run_arm.sh H 0 0`).
Write new commands in docs in that form.

## Specs

| Doc | What it covers |
|---|---|
| [`README.md`](README.md) | What this branch holds and how to run it |
| [`docs/hypotheses/README.md`](docs/hypotheses/README.md) | Index of every question, by status: `todo/`, `in-progress/`, `done/`; one doc each with claim, arms, decision rule, exact commands, results. Read it first; the GPU plan for the 3 GPUs is there |
| [`docs/data-eval.md`](docs/data-eval.md) | Training-data budget and the controlled evaluation set: splits, what each eval measures, which split may be used for what |

A new question starts as `docs/hypotheses/todo/<slug>.md`; `git mv` it to `in-progress/` when its first run launches
and to `done/` once its decision rule is applied and the results are recorded. Never copy a doc between folders.

This file and CLAUDE.md are shared by every branch (CI copies changes to them onto `main`), so keep them
branch-neutral: anything specific to one branch's code goes in its README or hypothesis doc.

## Study results

Start at [`docs/studies/README.md`](docs/studies/README.md) (headline Decision Index table, conclusions, open items).

| Doc | What it covers |
|---|---|
| [`docs/studies/data.md`](docs/studies/data.md) | Data build, prompt rendering, eval-sampling bug, training-signal shape |
| [`docs/studies/lora_end_to_end.md`](docs/studies/lora_end_to_end.md) | LoRA SCRM runs (4090, A4000, 3090, 2×H200), gradient cache, loss diagnosis, checkpoint selection, Peek branching |
| [`docs/studies/frozen_features.md`](docs/studies/frozen_features.md) | Frozen feature extraction: HF vs vLLM timings, caches, vLLM sweep, Decision Index featurization |
| [`docs/studies/frozen_lossgrid.md`](docs/studies/frozen_lossgrid.md) | Frozen-head loss grid: pilot, 15 arms × 3 seeds, step budget, layer ablation |
| [`docs/studies/external_comparison.md`](docs/studies/external_comparison.md) | Public board comparison, Vela/Nox, contamination audit |

When a study finishes, record its results in `docs/studies/` and update the headline table in
`docs/studies/README.md`.
