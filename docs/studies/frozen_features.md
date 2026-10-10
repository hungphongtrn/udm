# Frozen-backbone feature extraction studies

Contract: Qwen3.5-4B, prefill only (no generation), one vector per option = hidden state at the last input token of
`[prompt with all options][Grade this choice: Option k: …]<assistant header>`, layers 16, 24 and last, final RMSNorm
applied, bf16. Two presentation variants per set (canonical + seeded shuffle). Setup and resume rules:
`docs/TRAINING.md` § Frozen feature extraction (on `claude/set-conditioned-reward-model-hvsb9s`).

## 1. HF prefix-cached extraction of train (2026-10-07)

The prompt is prefilled once; each option continues from a copied cache. Train capped at 50,000 sets, 512 sets/shard.

| shard | dropped | (set, variant) records | seconds |
|---|---:|---:|---:|
| 00019 | 53 | 918 | 420.4 |
| 00020 | 56 | 912 | 413.0 |
| 00021 | 56 | 912 | 409.2 |
| 00022 | 71 | 882 | 388.6 |
| 00029 | 65 | 894 | 340.9 (2.62 records/s) |
| 00075 | 79 | 866 | 383.4 |

GPU snapshots: 11.1 GiB / 24 GiB at 35–61 % utilisation. Batching changes (equal-length prompt batches + grouped
suffixes `bad11f6`; no full-cache deepcopy, `max_batch_size` caps `257a1d6`) did not visibly change shard times. Likely
limits: exact-length bucketing, `cache_tokens=32768` allowing ~2 option rows per batch for 16k prompts, sequential outer
batches, CPU render/write. Per-stage timings were never measured.

Final train cache `outputs/features_qwen3_5_4b`: 98 shards, 43,757 sets, 1,647,060 rows, 6,243 sets dropped
(over `max_len` 16384 or untrainable).

## 2. vLLM pooling backend (2026-10-07/08)

vLLM 0.19.1 (last build for torch 2.10/cu128) with a custom multi-layer pooling adapter (layers 16/24 captured,
RMSNorm only on pooled rows, no LM head). Startup fixes: in-process model registration (`8852c10`), GPU-UUID
`CUDA_VISIBLE_DEVICES` → NVML indices (`c22868d`), text-only M-RoPE positions (`87568db`). Two-wave submission (one
option per set first, siblings after) so siblings hit the cached prompt (`b74a069`).

- Validation shard 101 (vLLM): 512 rows, 73 dropped, 878 records in 477.8 s (vs HF train shard 75: 383.4 s; different
  splits, GPU possibly shared — not a clean comparison).
- Validation cache `outputs/features_qwen3_5_4b_vllm_eval`: 104 shards (capped in place at 53,248 = 104 × 512).

## 3. Mixed cache for the loss grid

`outputs/features_qwen3_5_4b_pack` = HF train (all 98 shards) + first 8 vLLM validation shards (4096 sets,
source-balanced prefix). HF/vLLM kernels are not bit-identical; drift accepted by the user since every arm sees the
same arrangement. No test split is used; the held-out test is the Decision Index.

## 4. vLLM engine-settings sweep on one 3090 (2026-10-08)

First 512 Decision Index requests, 16k-token batch budget:

| tag | req/s | rows/s | unique tok/s | submitted tok/s | cache hit | ideal hit | ETA h |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 12.9 | 26 | 4549 | 8710 | 11.2 % | 47.8 % | 3.23 |
| graphs (`enforce_eager=false`) | 12.9 | 26 | 4540 | 8693 | 11.2 % | 47.8 % | 3.23 |
| graphs_seqs64 | 13.1 | 26 | 4605 | 8818 | 7.1 % | 47.8 % | 3.19 |
| nocache | 12.3 | 25 | 4313 | 8257 | 0.0 % | 47.8 % | 3.40 |
| seqs128 | 13.4 | 27 | 4706 | 9011 | 7.1 % | 47.8 % | 3.12 |
| seqs64 | 13.0 | 26 | 4569 | 8748 | 7.1 % | 47.8 % | 3.21 |
| seqs64_win1024 | 12.5 | 25 | 4411 | 8445 | 0.0 % | 47.8 % | 3.33 |

Engine knobs moved the ETA only 3.40 → 3.12 h. The ETA itself was ~50× too low (next section).

## 5. Decision Index featurization: vLLM vs HF (2026-10-09)

Full-suite vLLM shards (1024 requests each):

| shard | questions | option rows | seconds | submitted tok/s | unique tok/s |
|---|---:|---:|---:|---:|---:|
| 00002 | 11,253 | 62,289 | 3307.7 | 65,959 | 2640 |
| 00003 | 1024 | 78,848 | 3608.0 | 19,633 | 576 |
| 00004 | 1024 | 78,848 | 3609.4 | 19,635 | 576 |
| 00005 | 1024 | 91,946 | 4387.3 | 22,395 | 533 |
| 00006 | 1024 | 154,624 | 8137.7 | 28,204 | 440 |
| 00007 | 1024 | 154,624 | 8127.6 | 28,237 | 441 |

Most suite questions have a ~900-token prompt and 77–151 options. vLLM sends one request per option and Qwen3.5's
block-aligned (`mamba_cache_mode=align`) prefix cache recomputes most of the prompt each time, so ~1–2.3 h per shard
(full suite ≈ 150–300 h [INFERENCE]). Decision: DI featurization uses the HF prefix-cached path by default
(`10a6225`), estimated 15–30 h for the full suite [INFERENCE], and the fixed 1000-request sample for arm ranking
(`a089efe`; 128-request shards for progress/resume, `800b382`). Sample scoring of a grid took ~25 min.

## 6. Loss-grid I/O (2026-10-09)

`WORKERS=4` looked stalled: each worker loaded the whole layer cache into RAM plus an fp32 copy for normalisation, with
logs suppressed and every worker using all cores. Fix `d2b0aaf`: memory-mapped shard reads, cached normalisation stats,
per-run logs every 250 steps, `torch.set_num_threads(cpu_count // workers)`. The GPU stays lightly used by design (small
head, CPU-built batches).
