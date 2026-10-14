# Frozen-head loss-grid studies

Question: on a frozen Qwen3.5-4B, which readout loss — softmax CE, Brier, Bradley–Terry (BT), per-option sigmoid
(SigLIP-style BCE with soft labels) or any combination — gives the best held-out decisions?

## 1. Protocol

`configs/lossgrid_qwen3_5_4b.yaml`, `python -m scrm.lossgrid` / `scripts/train/lossgrid.sh`:

- Features: `outputs/features_qwen3_5_4b_pack` (HF train 43,757 sets + 4096 vLLM validation sets,
  [frozen_features.md §3](frozen_features.md#3-mixed-cache-for-the-loss-grid)), `data.layer: 24` unless stated,
  `augment_variants: false`.
- Head: set transformer, d_set 256, 2 layers, 4 heads, dropout 0.1.
- Train: batch 64 sets, AdamW, weight decay 0.01, warmup 100, cosine, readout scalars (T, tau, alpha, bias) at 10× lr,
  `w_center` 0.01, `w_perm` 0.
- Arms: the 15 non-empty subsets of {ce, brier, bt, sigmoid}; every term weight 1.0 (matched budget, no per-term tuning).
- Tuning: lr ∈ {3e-4, 1e-3}, per arm chosen by mean validation `loss/total` over seeds; checkpoint = lowest validation
  loss (eval every 250 steps). DI never used for selection.
- Test: Decision Index 0.2.1 on the 1000-request sample, live frozen features, HF backend
  (`scripts/train/dindex_frozen.sh`, run automatically at grid end). `answered_frac` = 1.0 for every head.
  Outputs: `outputs/lossgrid/<grid>/`, `outputs/dindex_frozen/<grid>_sample-1000/{runs.csv,summary.md}`.

## 2. Pilot (2026-10-07)

2,000 train sets, ~500 validation sets, 200 steps, seed 0. Validation, best lr per arm:

| arm | lr | top1 | pair_acc | mrr | ndcg | ece_top1 | nll_softmax | flip_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | 3e-4 | .634 | .814 | .779 | .839 | .081 | .966 | .243 |
| brier | 1e-3 | .638 | .806 | .784 | .843 | .111 | 1.043 | .222 |
| bt | 3e-4 | .611 | .804 | .767 | .831 | .100 | 1.075 | .256 |
| sigmoid | 3e-4 | .634 | .810 | .776 | .836 | .147 | 1.210 | .208 |
| ce+brier | 3e-4 | .638 | .819 | .782 | .842 | .093 | .966 | .238 |
| ce+bt | 3e-4 | .625 | .809 | .773 | .836 | .083 | .983 | .236 |
| ce+sigmoid | 3e-4 | .643 | .817 | .783 | .841 | .100 | .997 | .231 |
| brier+bt | 3e-4 | .634 | .815 | .780 | .841 | .079 | .998 | .224 |
| brier+sigmoid | 3e-4 | .641 | .813 | .781 | .840 | .114 | 1.073 | .220 |
| bt+sigmoid | 3e-4 | .627 | .810 | .775 | .836 | .096 | 1.094 | .215 |
| ce+brier+bt | 3e-4 | .627 | .815 | .776 | .838 | .088 | .976 | .236 |
| ce+brier+sigmoid | 3e-4 | .638 | .815 | .782 | .841 | .103 | .988 | .220 |
| ce+bt+sigmoid | 3e-4 | .632 | .814 | .778 | .838 | .089 | .999 | .229 |
| brier+bt+sigmoid | 3e-4 | .634 | .813 | .780 | .839 | .090 | 1.040 | .231 |
| ce+brier+bt+sigmoid | 3e-4 | .627 | .811 | .776 | .837 | .083 | .988 | .222 |

Main effects on top1: CE +0.0018, Brier +0.0054, BT −0.0111, sigmoid +0.0048 — all under 0.02 on one seed; no winner.
`flip_rate` (~0.23, top choice changes under option reordering) exceeded every loss effect.

## 3. Full grid, 3000 steps, 3 seeds, layer 24

DI sample, mean ± sd over seeds 0/1/2 (best_step always 2750 or 3000):

| arm | index | raw index |
|---|---|---:|
| **sigmoid** | **36.33 ± 0.97** | **50.67** |
| brier+sigmoid | 34.64 ± 0.66 | 49.36 |
| ce+sigmoid | 33.89 ± 0.90 | 48.68 |
| bt+sigmoid | 33.55 ± 1.00 | 48.39 |
| ce+brier+sigmoid | 33.51 ± 1.46 | 48.43 |
| ce+brier+bt+sigmoid | 33.50 ± 0.79 | 48.31 |
| brier | 33.40 ± 0.43 | 48.03 |
| ce+bt+sigmoid | 33.29 ± 0.76 | 48.14 |
| brier+bt+sigmoid | 33.18 ± 0.74 | 48.03 |
| bt | 33.08 ± 0.50 | 48.07 |
| brier+bt | 32.60 ± 0.96 | 47.56 |
| ce+brier+bt | 32.58 ± 0.51 | 47.80 |
| ce+bt | 32.19 ± 1.18 | 47.42 |
| ce | 32.00 ± 0.44 | 47.14 |
| ce+brier | 31.91 ± 0.51 | 47.16 |

Sigmoid seeds: 37.42, 35.57, 35.99. Adding sigmoid raises the mean of all 7 non-sigmoid arms
(e.g. ce 32.00 → 33.89, ce+brier 31.91 → 33.51). Sigmoid alone leads the next arm by 1.69.

Area means over seeds (skill × 100):

| arm | knowledge | language | retrieval | tools | arts |
|---|---:|---:|---:|---:|---:|
| sigmoid | 22.27 | 38.38 | 35.85 | 61.37 | 22.51 |
| brier | 20.70 | 37.06 | 28.58 | 58.86 | 19.89 |
| bt | 21.11 | 37.14 | 29.56 | 56.53 | 17.66 |
| ce | 17.42 | 36.38 | 31.26 | 54.70 | 18.41 |

Sigmoid gains everywhere, most in retrieval and tools. Why sigmoid helps (absolute per-option targets keep rewards
comparable across questions, which the ranking and Brier benchmarks use) is [INFERENCE], not tested.

## 4. Step budget: 5000 vs 10000 steps

Seed 0 only, same 15 arms (`output_dir=outputs/lossgrid/steps5000|steps10000`). Index / raw / best_step:

| arm | 5000 steps | 10000 steps |
|---|---|---|
| ce | 30.33 / 46.13 / 4750 | 28.93 / 45.15 / 6750 |
| brier | 33.39 / 48.24 / 5000 | 32.76 / 48.18 / 6500 |
| bt | 33.60 / 48.06 / 4500 | 32.95 / 47.89 / 4750 |
| sigmoid | 33.70 / 48.78 / 4500 | 34.18 / 48.91 / 6750 |
| ce+brier | 33.73 / 48.59 / 4750 | 31.90 / 46.72 / 6750 |
| ce+bt | 32.70 / 47.73 / 4500 | 33.08 / 47.99 / 5000 |
| ce+sigmoid | 30.78 / 46.52 / 4750 | 32.22 / 47.58 / 4750 |
| brier+bt | 32.98 / 47.42 / 4500 | 31.23 / 46.57 / 4750 |
| brier+sigmoid | 32.87 / 47.58 / 4750 | 30.26 / 46.19 / 4750 |
| bt+sigmoid | 32.22 / 47.23 / 4500 | 32.24 / 46.67 / 4750 |
| ce+brier+bt | 34.43 / 48.79 / 4500 | 31.76 / 46.64 / 4750 |
| ce+brier+sigmoid | 32.12 / 47.16 / 4750 | 32.39 / 48.13 / 6750 |
| ce+bt+sigmoid | 32.58 / 47.35 / 4500 | 30.82 / 46.31 / 6750 |
| brier+bt+sigmoid | 30.93 / 46.26 / 4500 | 32.07 / 47.12 / 4750 |
| ce+brier+bt+sigmoid | 32.05 / 46.81 / 4500 | 30.83 / 46.62 / 4750 |
| **mean** | **32.56** | **31.84** |

- 10000 vs 5000: 9 arms down, 5 up, 1 flat.
- Validation top1 / MRR / NDCG identical within ~0.005 across budgets (e.g. ce top1 0.736 at both; val top1 0.72–0.74
  for every arm).
- At 10000 steps the best checkpoints sit at 4750–6750 although lr decays to 0 only at the end: the head overfits
  past ~5–7k steps.
- Chosen lrs flip between 3e-4 and 1e-3 across budgets: validation loss barely separates them and is a weak proxy for DI.

Decision: keep `train.steps: 3000`; step count is not a lever.

## 5. Layer ablation: L16 vs last

3000 steps, seed 0, `data.layer=16|last` (`output_dir=outputs/lossgrid/layer16|layerlast`), features already cached.

| arm | L16 index / raw | L16 tools | last index / raw | last tools |
|---|---|---:|---|---:|
| ce | 25.74 / 42.19 | 42.33 | 33.60 / 48.24 | 58.94 |
| brier | 27.12 / 43.11 | 50.34 | 33.95 / 48.55 | 58.35 |
| bt | 25.48 / 41.80 | 45.84 | 32.81 / 48.39 | 58.92 |
| sigmoid | 25.75 / 41.75 | 40.52 | 33.66 / 48.68 | 57.53 |
| ce+brier | 26.12 / 42.51 | 45.45 | 33.83 / 47.90 | 58.10 |
| ce+bt | 25.44 / 41.76 | 43.83 | 33.41 / 48.30 | 60.22 |
| ce+sigmoid | 25.98 / 42.29 | 49.01 | 33.00 / 48.09 | 55.97 |
| brier+bt | 25.49 / 42.14 | 46.47 | 34.28 / 48.76 | 59.08 |
| brier+sigmoid | 23.30 / 40.27 | 42.80 | 30.65 / 46.57 | 56.54 |
| bt+sigmoid | 24.18 / 40.70 | 45.83 | 33.18 / 48.18 | 59.87 |
| ce+brier+bt | 25.21 / 41.79 | 43.72 | 34.37 / 48.68 | 60.12 |
| ce+brier+sigmoid | 26.90 / 43.35 | 52.75 | 32.25 / 47.38 | 56.56 |
| ce+bt+sigmoid | 25.85 / 41.96 | 46.01 | 32.18 / 47.59 | 58.31 |
| brier+bt+sigmoid | 24.17 / 40.61 | 45.87 | 31.49 / 46.96 | 56.17 |
| ce+brier+bt+sigmoid | 27.30 / 43.52 | 54.08 | 31.10 / 46.93 | 56.42 |
| **mean** | **25.60** | | **32.92** | |

- L16 is ~7 points below the last layer on every arm (its best arm, 27.30, is below the last layer's worst, 30.65).
- The last layer's spread is tight (30.65–34.37); its mean sits in the L24 range seen at 5000/10000 steps (32.56 / 31.84).
  The L24 3000-step seed-0 mean was not pulled, so last vs L24 is unresolved; sigmoid on last (33.66, one seed) is below
  the L24 3-seed sigmoid mean (36.33).
- Knowledge skill stays low at every layer (L16 6.8–14.8, last 10.8–15.8) while tools and language rise with depth.

## 6. Status

Stopped here (2026-10-09). Best measured configuration: L24, sigmoid loss, 3000 steps (DI sample 36.33 ± 0.97).
Not run: L24 vs last at 3 seeds, concatenated L24 + last features, `augment_variants=true`, larger heads, full-suite DI.
