# AGENTS.md

Read the relevant doc before changing code or proposing experiments.

## Specs

| Doc | What it covers |
|---|---|
| [`README.md`](README.md) | Project overview, two-machine workflow (CPU data build → HF repo → GPU training) |
| [`docs/CONTRACT.md`](docs/CONTRACT.md) | Data ↔ trainer contract: row semantics, label kinds → tiers |
| [`docs/DATA_SPEC.md`](docs/DATA_SPEC.md) | Schema, hash recipes, per-source mapping, metadata recovery, drop reasons |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Model design, memory budget, configs, metrics, checkpoints, frozen feature pipeline, inference |

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
