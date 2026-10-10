# AGENTS.md

Read the relevant doc before changing code or proposing experiments.

## Specs

| Doc | What it covers |
|---|---|
| [`README.md`](README.md) | What this branch holds and how to run it |
| [`docs/hypotheses/`](docs/hypotheses/) | One doc per `hyp/` branch: claim, arms, decision rule, exact commands, results, and pointers to that branch's code |

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
