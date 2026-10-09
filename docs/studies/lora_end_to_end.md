# LoRA end-to-end SCRM studies

Model: Qwen3.5-4B text decoder (vision tower dropped), LoRA r64 / alpha 16 rsLoRA all-linear, 2-layer set transformer
head (d_set 768), Bradley–Terry (BT) loss over tier pairs, bf16 backbone. Configs: `configs/scrm_qwen3_5_4b_24gb.yaml`,
`_40gb`, `_h200`, `peek_qwen3_5_4b_*`. W&B projects `yuuart/scrm` and `aivforever/scrm`.

## 1. Readiness and stack (2026-10-02)

- Nothing had run before 2026-10-02; the GPU memory estimate of 14–17 GB was unmeasured.
- Stack: Python 3.12, torch 2.10.0 + cu128, causal-conv1d 1.7.0 prebuilt, flash-linear-attention. A cu130 plan
  (`23a585b`) was reverted to cu128 for the rented cards (`c1192bd`); cu130 is used only on H200 (driver 580).
- CUDA-only causal-conv1d cannot run a CPU forward (`Expected x.is_cuda()`); default device made auto (`71ffc60`).

## 2. First smoke on a 4090 (2026-10-02)

30 steps, max_len 4096, state budget 3072, 4096 tokens / micro-batch, grad_accum 8, `expandable_segments`:
loss 0.69 → ~0.39; eval step 20 on 1500 items: pair_acc 0.907, top1 0.586, permutation agreement 0.895.

- Shared-prefix cache leaked memory through a reference cycle: 17.5 GB then OOM; a weak reference in `_LayerView`
  brought it back to 9.0 GB after each backward.
- Speed 1.5–2k tok/s; the 1500-item eval took 504 s. Optimizer steps covered 15–33 sets / 60–137 pairs.

## 3. Packing and no-truncation redesign (2026-10-02)

The shared-prefix cache was removed in favour of full sequences (prompt + one graded option each), padding-free BFD
packing with `cu_seq_lens` (fla delta-rule + causal-conv1d kernels already active, so no Unsloth patch was needed),
and whole-set drops instead of truncation (`deb6dc6`). Cost: a set costs ~n_graded × (prompt + option) tokens.

## 4. First packed run (4090, `vqn31gtl`)

Vast RTX 4090, 24 GB config, 20k-step plan, DI every 2000 steps.

- Step 1: 15.6 GB, 100 % GPU. 30-step peak ~17.5 GB at ~16 s/step. At step 450: 16.7 s/step, ~2.9–3.0k tok/s
  between evals, loss 0.23–0.34, peak 15.7 GB, 11–18 sets/step; most steps clipped (grad norm 5–22 vs clip 1.0).
- First real run died at step 2000 between test eval and save because the DI suite was missing; fixed by saving
  before benchmarks and validating DI setup at startup (`464a3ac`).
- Final state: stopped at step 2280 when Vast credit ran out; instance destroyed, checkpoints lost.

| metric | step 1000 | step 2000 |
|---|---:|---:|
| val loss | 0.098 | 0.212 |
| val pair_acc / top1 | 0.963 / 0.754 | 0.906 / 0.682 |
| val permutation agreement | 0.950 | 0.923 |
| test loss | 0.195 | 0.252 |
| test pair_acc / top1 | 0.912 / 0.770 | 0.899 / 0.733 |
| DI index / raw (sample) | — | 22.17 / 39.45 (4738 s) |

The apparent drop was mostly the broken eval sample ([data.md §4](data.md#4-eval-sampling-bug-2026-10-03)); training
itself was stable (400-step loss means 0.364 → 0.228, no NaN, max grad spikes 97.2).

## 5. A4000 OOM (`zfb1k9tv`, 2026-10-03)

On the host with 4× 3090 + 1× A4000 the process was pinned to the A4000 (15.61 GiB): allocation 11.2 GB at step 1 →
14.3 GB at step 10, then OOM in backward. Step-0 eval took 2104 s. Fix: log GPU name/memory at startup (`5ef1e92`),
default `expandable_segments` (`d371da4`). The 24 GB config needs a 24 GB card.

## 6. First 3090 run (`crimson-star-4` / `tgr3z1kz`, 2026-10-03)

1,540–1,545 tok/s, 31.8–34 s/step (20k steps ≈ 177 h). After the eval-sampling fix:

| metric | step 0 | step 1000 | step 2000 |
|---|---:|---:|---:|
| val loss | 0.698 | 0.368 | 0.361 |
| val pair_acc | 0.383 | 0.854 | 0.846 |
| val top1 | 0.2375 | 0.675 | 0.658 |
| val ndcg | 0.543 | 0.851 | — |
| val kendall_tau | 0.03 | 0.295 | 0.390 |
| val ece_top1 | 0.04 | 0.148 | 0.097 |
| test top1 / pair_acc / loss | — | 0.692 / 0.835 / 0.497 | 0.716 / 0.851 / 0.355 |

Per source at step 1000 (val top1, step 0 → 1000): massive 0.00 → 0.58, samatv 0.12 → 0.75, Open-Jev 0.49 → 0.76,
tasksource 0.34 → 0.61. Open-Jev's step-0 top1 0.49 hints at a position/length shortcut. Reward mean drifted
−1 → −5.7 with within-set spread 2–3. Ended `failed` at step 2000 (DI suite missing).

## 7. 3090 continuation (`02x2vl3b`)

Resumed from `outputs/scrm_qwen3_5_4b_24gb/best`; max_len 4096, max_graded 8, ~14 sets / ~50 pairs per step,
~35 s/step, 98–100 % GPU at the 350 W cap.

| step | val top1 | pair_acc | kendall | val loss | ECE | test top1 |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | .675 | .854 | .295 | .368 | .148 | .692 |
| 2000 | .658 | .847 | .390 | .361 | .097 | .716 |
| 3000 | .653 | .868 | .417 | .316 | .087 | .710 |
| 4000 | .670 | .850 | — | .403 | .146 | .710 |
| 5000 | .6525 | — | — | .379 | .115 | .694 |

DI (sample): 24.73 at step 4000 (that eval took ~188 min). Top1 plateaued while loss / ECE / overconfidence (+0.133)
worsened; reward std rose ~2.0 → 3.2 and mean confidence 0.72 → 0.80. Judged stalled; superseded by the frozen study.

## 8. 2×H200 runs (`9ax53199` → `0d3l3ivx`)

H200 config: max_len 32768, all options graded, per-rank budget 262,144 tokens, gradient caching, manual DDP.

**Gradient caching** (3 passes: no-grad embed → head loss and dL/de → per-chunk re-encode with grad): CPU tests show
the same loss and grads as the one-pass path, including dropout replay and an uneven 2-rank split
(`tests/test_scrm_gradcache.py`, 7 passed). Not checked on GPU.

**Kernel setup (cu130):** FLA/Triton Hopper backward kernel `chunk_bwd_dqkwg` is wrong on Triton 3.4–3.7.0 → TileLang
with pinned nvcc 13.0.88 / CCCL 13.0.85; without `flash_attn` SDPA tried a 64 GiB dense mask for a 262k-token batch
→ prebuilt `flash-attn==2.8.3`. A resumed run hit HTTP 429 because `data.local_dir` was not passed (streamed from the Hub).

**`9ax53199`** (chunk 32768, checkpointing on): step 1 loss 0.7025, 12 sets / 393 pairs; memory 45.5 / 29.7 GiB;
both GPUs computing only 47 % of the time, one idle at the all-reduce 35 %. ~32.5–33.0 s/step to step 1180, 42.3 GB peak.
Early `tok/s` was mis-aggregated (fixed `9ed3f41`).

**Memory tests with checkpointing off** (`mmpdkqkz`, `2ifkfwq6`, `gotepo1q`): all OOM at the first step in pass 3
(>136 GB), because a ~32k sequence is always its own chunk. Checkpointing re-enabled (`3dd03b5`); selective
checkpointing `grad_cache_act_tokens` added (`2d30ca8`).

**`0d3l3ivx`** (resumed at step 1000; chunk = act tokens = 8192; sequence-level rank balancing): 25.1–25.9 s/step
(~20 % faster than 33.0), 78.2 GB flat, ~99 % GPU, no alerts over ~7.9 h.

| step | val top1 | val loss | val ECE | test top1 | test loss | test ECE | DI index (raw) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | .685 | .340 | .061 | .712 | .370 | .086 | 27.20 (42.19) |
| 3000 | .700 | .306 | .080 | .716 | .313 | .047 | — |
| 4000 | .715 | .286 | .059 | .730 | .351 | .068 | 27.61 (42.01) |

DI areas, step 2000 → 4000: language 19.67 → 29.09, retrieval 30.31 → 31.79, knowledge 12.08 → 13.92,
arts 22.66 → 18.94, tools 58.32 → 45.05 (BFCL 93.55 → 61.27, API-Bank 84.71 → 64.33). Test pair_acc dropped
.8478 → .8340. Areas traded off instead of improving together.

## 9. Loss diagnosis (2026-10-05/06)

- BT averages many easy pairs and lets reward gaps inflate once the order is right; low BT loss does not imply a correct
  top choice. Interpretation from logged metrics, not a controlled experiment.
- An initial claim that softmax mismatch and pair weighting were the main defects was later corrected: the DI headline
  mostly uses the argmax answer (probabilities matter for ForecastBench Brier and the two nDCG benchmarks), per-question
  mean-over-pairs does not over-weight big sets, and the like-for-like gap vs peers is confounded by prompts, harness
  and the 1006-request sample.
- Proposed but never run: 3090 restart with `loss.w_plackett_luce=0.5 loss.w_center=0.05`. The question moved to the
  controlled frozen-feature loss grid ([frozen_lossgrid.md](frozen_lossgrid.md)).

## 10. Checkpoint selection (2026-10-06)

Selection moved to validation `val_index` (100 × equal-source mean of chance-corrected source top1) and DI became
test-only (`32fafd4`); eval/save/DI every 1000 steps, top-3 validation checkpoints kept (`a1c6ab5`).

## 11. Peek: linear head + prefix branching (2026-10-06)

Design: one packed prefill per set, each option a branch off the shared prompt (Gated DeltaNet state / conv state copied
per branch), shared linear head `r_i = wᵀh_i + b`, BT + 0.01 center (`46633e1`). Option-reward independence difference
0 with the linear head vs 3.3e-2 with the set head; a `_branch_chunk` offset bug fixed (error 3.47 → 4.8e-7).

Equivalence on a tiny CPU model (4 sets with 4/3/2/5 options):

| layout | loss, dropout 0 | max grad diff | loss, dropout .05 | max grad diff |
|---|---:|---:|---:|---:|
| set-wise reference | .62212790 | — | .68034108 | — |
| grad cache, one set / chunk | .62212795 | 5.1e-6 | .68034106 | 1.0e-6 |
| grad cache, all sets packed | .62212789 | 5.2e-6 | .56898785 | 2.4 |
| packed, no grad cache | .62212789 | 5.2e-6 | .56898785 | 2.4 |
| per-option sequences | .62212794 | 5.6e-4 | .50649902 | 2.4 |

Equal without dropout; with LoRA dropout the masks differ by layout (same expectation, different samples). GPU branch
kernels passed against a bf16 oracle after fixes (`4722af6` … `ce81846`).

3090 Peek run (`configs/peek_qwen3_5_4b_3090.yaml`: LoRA lr 1e-4, head 5e-4, warmup 300, cosine to 6000, ≤32
sets/step, max_len 16384, 4–16 graded): OOM in `_branch_attn_torch` during gradient-cache recompute (202 MiB requested,
172 MiB free of 23.57 GiB). Prefix branching was disabled and end-to-end work paused in favour of the frozen study.
