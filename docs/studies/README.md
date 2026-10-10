# SCRM studies log (2026-10-02 → 2026-10-09)

Record of every experiment run on this branch (`claude/set-conditioned-reward-model-hvsb9s`): what was asked, how it
was set up, the measured numbers and the decision taken. How-to material (configs, scripts, flags) lives in
`docs/TRAINING.md` (on main); this directory holds results only.

| Doc | Covers |
|---|---|
| [`data.md`](data.md) | Dataset build and push, row counts, prompt rendering, context-length measurements, the eval-sampling bug |
| [`lora_end_to_end.md`](lora_end_to_end.md) | LoRA + set-head runs: 4090, A4000 OOM, 3090, 2×H200; gradient caching, DDP balancing, memory; loss diagnosis; Peek/branching redesign |
| [`frozen_features.md`](frozen_features.md) | Frozen-backbone feature extraction: HF prefix cache vs vLLM, throughput, resumable caches, Decision Index featurization cost |
| [`frozen_lossgrid.md`](frozen_lossgrid.md) | Frozen-head loss grid (CE / Brier / BT / sigmoid): pilot, 3000-step 3-seed Decision Index, step budget, layer ablation |
| [`external_comparison.md`](external_comparison.md) | Decision Index board peers, Vela / Nox, contamination audit status |

## Headline results

All Decision Index (DI, edition 0.2.1) numbers below are on the fixed stratified **1000-request sample**
(`sample-1000.jsonl.gz`, 1006 requests). They estimate the index for ranking our own runs; they are not
board-comparable full-suite scores (`complete=False` in every result). Sample noise is ~±1–2 index points [INFERENCE].

| System | DI index (raw) | Where |
|---|---|---|
| LoRA r64 + set head, BT only, 4090 (`vqn31gtl`), step 2000 | 22.17 (39.45) | [lora](lora_end_to_end.md#4-first-packed-run-4090-vqn31gtl) |
| LoRA, 3090 (`02x2vl3b`), max_len 4096, 8 graded, step 4000 | 24.73 | [lora](lora_end_to_end.md#7-3090-continuation-02x2vl3b) |
| LoRA, 2×H200 (`0d3l3ivx`), max_len 32768, all options, step 2000 / 4000 | 27.20 (42.19) / 27.61 (42.01) | [lora](lora_end_to_end.md#8-2h200-runs-9ax53199--0d3l3ivx) |
| **Frozen Qwen3.5-4B L24 + set head, sigmoid loss, 3000 steps, 3 seeds** | **36.33 ± 0.97 (50.67)** | [lossgrid](frozen_lossgrid.md#3-full-grid-3000-steps-3-seeds-layer-24) |
| Frozen L24, mean of 15 loss arms, 3000 steps, 3 seeds | 31.91–36.33 per arm | [lossgrid](frozen_lossgrid.md#3-full-grid-3000-steps-3-seeds-layer-24) |
| Frozen L24, mean of 15 arms, seed 0, 5000 / 10000 steps | 32.56 / 31.84 | [lossgrid](frozen_lossgrid.md#4-step-budget-5000-vs-10000-steps) |
| Frozen L16 / last layer, mean of 15 arms, seed 0, 3000 steps | 25.60 / 32.92 | [lossgrid](frozen_lossgrid.md#5-layer-ablation-l16-vs-last) |

Board context (public, not reproduced by us): Jobe (frozen Qwen3.5-4B-Base) 32.35, open-jev 29.91, Decider 4B 40.70,
JPT-4B 43.04, Decision-2.0-Nox-4B 42.55 (Vela authors' reproduction) / 43.8 (model card).

## What we concluded

1. **End-to-end LoRA with BT-only loss underperformed** a frozen backbone with a small head trained on CE/Brier/sigmoid:
   27.61 after ~29.6 h on 2×H200 (step 4000) vs 36.33 for a frozen-feature head. In-distribution validation kept improving
   while DI barely moved, so validation top1 is a weak proxy for DI.
2. **Loss choice matters on the frozen backbone:** sigmoid (per-option absolute BCE) alone is best (36.33 ± 0.97); every
   arm containing sigmoid beats its non-sigmoid counterpart on average; CE-only / CE+Brier are worst (~32).
3. **Training longer does not help** the frozen head: validation top1 is flat at ~0.72–0.74 from 3000 to 10000 steps,
   best checkpoints land at 4750–6750 of 10000 steps, and the 15-arm mean DI drops 32.56 → 31.84.
4. **Feature layer:** L16 is ~7 points worse; the last layer is in the same range as L24 (single seed, so no ranking
   between them yet).
5. **Throughput:** for many-option questions vLLM's per-option requests reuse little of the prompt (block-aligned
   hybrid-state prefix cache); HF prefix caching (prompt once, options branch from a copied cache) is the default.

## Open items (not run)

- Layer-24 seed-0 mean at 3000 steps, to compare directly with the L16 / last seed-0 grids (the per-run values are in
  `outputs/dindex_frozen/qwen3_5_4b_pack_sample-1000/runs.csv` on the 3090 box).
- Full-suite DI (~150k requests; estimated 15–30 h with the HF backend on one 3090 [INFERENCE]) for the final head.
- Contamination audit of training data against DI items ([external_comparison.md](external_comparison.md#3-contamination-audit)).
- Next levers proposed but not run: concatenated L24 + last features; LoRA with a CE/Brier/sigmoid objective instead of
  BT; a larger head (deprioritised, since every loss saturates at the same validation top1).

## Shared caveats

- **HF vs vLLM drift:** the frozen study trains on HF-extracted train features and validates on vLLM-extracted
  validation features. Accepted; identical for all arms, so arm comparisons stay fair.
- **Selection:** checkpoints and learning rates are chosen on UDM validation loss only; DI is held out and scored
  afterwards. The LoRA runs selected on validation loss (later `val_index`); DI was logged, never used for selection.
- **Seeds:** 3-seed arm spreads are ±0.4–1.5; single-seed differences of ≤2 points are not meaningful.
