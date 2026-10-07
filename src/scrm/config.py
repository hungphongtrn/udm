"""Config handling: YAML + defaults + dotted CLI overrides."""
from __future__ import annotations

import copy
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 1234,
    "output_dir": "outputs/scrm",
    "model": {
        # HF id / local path of the backbone, or "tiny" for a randomly initialised
        # offline model (tests / smoke runs).
        "name_or_path": "Qwen/Qwen3.5-4B",
        "tiny": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                 "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
                 "linear_value_head_dim": 16, "seed": 0},
        "dtype": "bfloat16",               # backbone dtype (bfloat16 | float32)
        "attn_implementation": "auto",     # auto -> flash_attention_2 if installed, else sdpa
        "gradient_checkpointing": True,
        "liger_kernel": True,              # Liger RMSNorm + SwiGLU via liger_kernel.transformers (CUDA only)
        "freeze_backbone": False,          # frozen-backbone ablation (no LoRA, no grads through backbone)
        "quantize_4bit": False,            # QLoRA (bitsandbytes nf4)
        # rsLoRA: scaling = alpha / sqrt(r) (r=64, alpha=16 -> 2.0). "all-linear" = every nn.Linear in the
        # text decoder: full-attention q/k/v/o, Gated DeltaNet in_proj_qkv/z/a/b + out_proj, MLP gate/up/down.
        "lora": {"enabled": True, "r": 64, "alpha": 16, "dropout": 0.05, "use_rslora": True,
                 "target_modules": "all-linear"},
        "input_norm": True,                # LayerNorm(d) before the d->d_set projection
        # reward head: "set" = SetEncoder over the set's candidate embeddings (v1/v2 checkpoints, cross-candidate
        # interaction); "linear" = one nn.Linear(hidden, 1) applied per candidate ("Peek": r_i = w^T h_i + b, no
        # cross-candidate interaction). scrm_head.pt keys are prefixed with the module name (`set_encoder.` /
        # `reward_head.`) so loading a checkpoint into the other head fails loudly.
        "head": "set",
        "d_set": 768,
        "set_layers": 2,                   # 0 = no-interaction ablation
        "set_heads": 8,
        "set_ffn_mult": 2,
        "set_dropout": 0.1,
        "head_hidden": None,               # default d_set // 2
        # Option encoding. true = shared-prefix branching: each set's prompt is encoded once and every graded option
        # continues it as a branch (custom attention / Gated DeltaNet layout, see scrm.model). false = every graded
        # option is its own full sequence `prompt + option` through the stock HF forward (n_graded x the prompt
        # tokens; the reference path). Mirrored into data.render.branching (batch token costs) by load_config.
        "branching": True,
    },
    "loss": {
        "tau": 1.0, "margin_alpha": 0.0,
        "w_bt": 1.0, "w_listwise": 0.0, "w_plackett_luce": 0.0,
        "w_center": 0.0, "w_perm": 0.0, "perm_detach": True,
    },
    "data": {
        "repo": "hungphongtrn/udm-massive-typed",
        "local_dir": None,                 # local snapshot dir (contains data/*.parquet) -> no network
        "streaming": False,
        "train_split": "train",
        "eval_split": "validation",
        "filters": {"include": {}, "exclude": {}},          # train filters (keys: source_id, family, label_kind, source_split)
        "eval_filters": {"include": {}, "exclude": {"source_split": ["ood"]}},
        # mixing groups; first match wins. Each: {name, match:{col:[..]}, files:[globs], max_rows, weight}
        "groups": [],
        "other_weight": 1.0,               # weight of rows matched by no group
        "drop_unmatched": False,
        "shuffle_buffer": 10000,           # streaming only
        "eval_max_rows": 2000,
        "eval_max_rows_per_source": 200,
        "perm_eval_rows": 64,
        # max_len: tokens per sequence (prompt + one graded option); longer sets are dropped, never truncated
        "render": {"max_len": 8192, "max_candidates": 64},
        "batch": {"max_tokens_per_batch": 8192, "max_batch_size": 16, "bucket_size": 128},
        "num_workers": 2,
        "prefetch_factor": 4,
    },
    "train": {
        "max_steps": 10000, "epochs": None, "grad_accum": 8,
        "lr_lora": 1e-4, "lr_head": 5e-4,
        "weight_decay": 0.01, "warmup_steps": 200, "warmup_ratio": None, "min_lr_ratio": 0.0,
        "early_stop_patience": 8, "early_stop_min_delta": 0.0,   # stop after N evals of best_metric without improvement (0/None = off)
        # `best` checkpoint + early stop, both on validation at every eval: val_index (max; chance-corrected top-1,
        # sources weighted equally, scored like the Decision Index) | val_loss (min). The Decision Index is test-only.
        "best_metric": "val_index",
        "optim": "adamw_8bit",              # adamw_8bit (bitsandbytes, CUDA) | adamw (torch fused)
        "grad_clip": 1.0, "amp": True, "device": "auto",
        # gradient caching: encode a whole loader micro-batch (group) without grad, then re-encode only the chunks
        # needed to push the set head's embedding grads into the backbone (see docs/TRAINING.md). The group budget
        # is data.batch.max_tokens_per_batch; grad_cache_chunk_tokens caps the backbone pass memory.
        "grad_cache": False, "grad_cache_chunk_tokens": 16384,
        # selective checkpointing in grad-cache pass 3 (needs model.gradient_checkpointing): a chunk of T tokens keeps
        # the activations of floor(L * act_tokens / T) of its L decoder layers, recomputing only the rest. null = all
        # layers checkpointed. Memory ~ act_tokens tokens of full (uncheckpointed) activations.
        "grad_cache_act_tokens": None,
        "ddp_timeout_min": 240,             # process-group timeout (min) under torchrun; rank 0 alone runs eval/dindex
        "log_every": 10, "eval_every": 500, "hist_every": 500, "save_every": 500, "keep_last": 3,
        "keep_best": 0,                     # also keep the top-k evals by best_metric as best_step_N (weights only; 0 = off)
        "eval_at_start": True,
    },
    # frozen-backbone feature cache (scrm.features): one backbone pass, per-option features for head experiments.
    # layers: 1..num_hidden_layers = residual stream after that decoder layer (then the final norm), -1 = last layer.
    # max_sets: rows drawn per split (sources interleaved; rows that do not render are dropped), null = every row.
    "features": {
        "out_dir": "outputs/features", "layers": [-1], "variants": 2, "seed": 0, "shard_size": 512,
        "splits": ["train", "validation", "test"],
        "max_sets": {"train": None, "validation": None, "test": None},
        # max_tokens: sum(prompt + all suffix lengths), prompt counted once per set; equal-length prompts batch together.
        # max_batch_size: sets per prompt batch; cache_tokens: (prompt + suffix length) * suffix batch size.
        "max_tokens": 16384, "max_batch_size": 32, "cache_tokens": 131072, "overwrite": False,
    },
    # test-only evaluations at every save step (and at the end); never used for checkpoint selection
    "benchmarks": {
        "test_split": "test",              # held-out split, same filters / row caps as validation (null = off)
        # Decision Index 0.2.1 on a fixed stratified sample (scrm.dindex.DecisionIndexEval); needs the rebuilt suite
        # and a sample made with `python -m decision_index suite sample` (scripts/train/dindex_setup.sh).
        # `every`: run at save steps divisible by it (and at the end); null = every save. 1000 rows ~ 40M tokens.
        "decision_index": {"enabled": False, "suite_dir": None, "rows": None, "edition": "0.2.1",
                           "max_len": 16384, "max_tokens": 16384, "every": 2000},
    },
    # Hugging Face Hub backup: every checkpoint save uploads the LoRA adapter + SCRM set block/head + config + tokenizer
    # (not the frozen base model) in the background to {repo_id}/{run}/best and {repo_id}/{run}/last; each upload is a
    # commit, so older checkpoints stay reachable by revision. run = basename(output_dir) unless set.
    # Load with load_scrm("hf://{repo_id}/{run}/best"). include_optimizer also uploads trainer_state.pt (resumable).
    "hub": {"repo_id": None, "run": None, "private": True, "include_optimizer": False},
    "wandb": {"enabled": True, "project": None, "entity": None, "run_name": None,
              "tags": [], "mode": None},
}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("include", "exclude", "match"):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def parse_overrides(items: list[str]) -> dict:
    """['train.max_steps=10', 'model.lora.r=8'] -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        cur = out
        parts = k.strip().lstrip("-").split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


def _load_yaml(path: str) -> dict:
    """YAML with optional `base: other.yaml` (relative to this file) inheritance."""
    import os
    with open(path) as f:
        y = yaml.safe_load(f) or {}
    base = y.pop("base", None)
    if base:
        b = _load_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), base))
        y = deep_update(b, y)
    return y


def load_config(path: str | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        deep_update(cfg, _load_yaml(path))
    if overrides:
        deep_update(cfg, parse_overrides(overrides))
    # one switch: the renderer's per-set token cost (batching / chunking budgets) must match how the model encodes
    cfg["data"]["render"]["branching"] = bool(cfg["model"]["branching"])
    return cfg
