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
        "attn_implementation": "auto",     # auto -> flash_attention_2 if installed else sdpa
        "gradient_checkpointing": True,
        "liger_kernel": True,              # Liger RMSNorm + SwiGLU via liger_kernel.transformers (CUDA only)
        "freeze_backbone": False,          # frozen-backbone ablation (no LoRA, no grads through backbone)
        "quantize_4bit": False,            # QLoRA (bitsandbytes nf4)
        # rsLoRA: scaling = alpha / sqrt(r) (r=64, alpha=16 -> 2.0). "all-linear" = every nn.Linear in the
        # text decoder: full-attention q/k/v/o, Gated DeltaNet in_proj_qkv/z/a/b + out_proj, MLP gate/up/down.
        "lora": {"enabled": True, "r": 64, "alpha": 16, "dropout": 0.05, "use_rslora": True,
                 "target_modules": "all-linear"},
        "input_norm": True,                # LayerNorm(d) before the d->d_set projection
        "d_set": 768,
        "set_layers": 2,                   # 0 = no-interaction ablation
        "set_heads": 8,
        "set_ffn_mult": 2,
        "set_dropout": 0.1,
        "head_hidden": None,               # default d_set // 2
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
        "eval_max_scan_rows": 200000,      # streaming eval: max rows scanned
        "perm_eval_rows": 64,
        "render": {"max_len": 2048, "cand_max_tokens": 128, "state_max_tokens": 1024,
                   "instr_max_tokens": 256, "max_candidates": 64, "min_state_tokens": 64,
                   "state_truncate": "middle"},
        "batch": {"max_tokens_per_batch": 4096, "max_batch_size": 16, "bucket_size": 128},
        "num_workers": 2,
        "prefetch_factor": 4,
    },
    "train": {
        "max_steps": 10000, "epochs": None, "grad_accum": 8,
        "lr_lora": 1e-4, "lr_head": 5e-4,
        "weight_decay": 0.01, "warmup_steps": 200, "warmup_ratio": None, "min_lr_ratio": 0.0,
        "grad_clip": 1.0, "amp": True, "device": "auto",
        "log_every": 10, "eval_every": 500, "hist_every": 500, "save_every": 500, "keep_last": 3,
        "eval_at_start": False,
    },
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
    return cfg
