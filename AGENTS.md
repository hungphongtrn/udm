# AGENTS.md

Read the relevant doc before changing code or proposing experiments.

## Specs

| Doc | What it covers |
|---|---|
| [`README.md`](README.md) | Repo overview: hypothesis branches on a vendored Kev fork |
| [`kev/UPSTREAM.md`](kev/UPSTREAM.md) | The Kev fork's change log relative to upstream |
| [`kev/scripts/pref/README.md`](kev/scripts/pref/README.md) | Run commands for the preference-model scripts (training machine only) |
| [`docs/hypotheses/`](docs/hypotheses/) | One doc per `hyp/` branch: claim, arms, decision rule, commands, results (e.g. [`preference-reward-model.md`](docs/hypotheses/preference-reward-model.md)) |

## Study results

These studies' code (SCRM) was removed from this branch; it lives on `main`.

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
