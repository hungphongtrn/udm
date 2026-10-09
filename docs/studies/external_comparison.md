# External comparison studies

Public Decision Index 0.2.1 numbers below come from the board, model cards and entrants' write-ups; we did not reproduce
them. Ours are 1000-request sample estimates, so the comparison is only indicative.

## 1. Board peers vs LoRA SCRM (2026-10-06)

| entrant | index | adaptation | published objective | spend |
|---|---:|---|---|---|
| JPT-4B | 43.04 | LoRA r16, merged | multi-class Brier over option labels; 49,221 questions, 1 epoch | n/a |
| Jet v6.2 | 42.60 | LoRA, merged | CE on soft targets + ordinal RPS; 4,000 examples | 250 steps |
| Hopper (G) 1.2 | 40.77 | LoRA adapter | undisclosed | n/a |
| Decider 4B | 40.70 | full FT, then LoRA r64 | CE on a slot readout (+ optional Brier); 742M tokens | 577 min on 2×B300 |
| lev | 38.54 | LoRA r32 + head | ordinal / classification; 200k examples | 7.8 h on 1×H100 |
| Jobe | 32.35 | none (Qwen3.5-4B-Base, frozen) | — | — |
| open-jev (pngwn) | 29.91 | LoRA r16 | CE over N option logits; 45,932 questions | 3.8 h |
| openvons / SemIf / mini-jev | 28.42 / 25.94 / 20.98 | frozen / inference | — | — |
| **SCRM LoRA, H200 step 4000** | **27.61** | LoRA r64 + set head | BT pairwise only | ~29.6 h on 2×H200 |
| **SCRM frozen L24 + sigmoid head** | **36.33 ± 0.97** | none (frozen Qwen3.5-4B) | per-option sigmoid BCE | head only |

Pattern [INFERENCE]: the strong same-size entrants read label-token probabilities or a softmax over options and train with
CE / Brier-type targets (survey of 76 entrants: 35 read label tokens, 18 use a candidate head). Our LoRA run was the
only BT-only recipe and sat ~15 points below them and 4.74 below the frozen Jobe. That motivated the frozen loss grid,
where the frozen head with sigmoid loss reached 36.33 (sample) — above Jobe, below the LoRA peers.

## 2. Vela-2.0-4B vs Decision-2.0-Nox-4B (2026-10-09)

Vela-2.0-4B (`vllm-sr`): 31.63 (31.91 with `noul_calibration=True`). It fine-tunes Decision-2.0-Nox-4B, which scores
42.55 (Vela authors' reproduction) / 43.8 (card). Not a like-for-like baseline for us:

- Vela/Nox start from `Qwen/Qwen3.5-4B-Base` and are fine-tuned; ours is the frozen instruct checkpoint.
- Vela has typed outputs (Choice, Noul, Score, Set, Span), router heads and a bilinear question–option CandidateHead
  with calibration; ours is one scalar per option from a 2-layer set head.
- Vela's data is curated (blind relabel agreement, audit against benchmark test items); ours only deduplicates against
  our own validation/test.
- Vela keeps native Noul/Score distributions; we convert labels to tiers, possibly losing ordinal/probability
  information [INFERENCE].

Decision: Nox is the relevant reference; any comparison needs our full-suite score.

## 3. Contamination audit

Status: investigated, not measured.

- `src/scrm_data/build.py` removes only rows matching our own held-out sets (`_heldout.npy`); nothing checks Decision
  Index items. The DI suite's own exclusions (442 rows plus scoring subsets) do not look at external training data.
- The `tasksource-jev` card lists 670 sources including HellaSwag and Banking77; DI scores HellaSwag validation and
  BANKING77 test. Twelve DI panel benchmarks share upstream datasets with our sources ([data.md §6](data.md#6-training-signal-shape-2026-10-06)).
- Open-Jev is probably not official Jev training data; risk is highest for tasksource.

Needed before quoting any external comparison: exact canonical-content-key matching plus normalised-state matching of
all train shards against the DI suite, reported per benchmark / source / split.
