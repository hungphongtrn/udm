"""Export a checkpoint for inference: optionally merge LoRA into the backbone.

python -m scrm.export --ckpt outputs/scrm/best|hf://org/repo/run/best --out exported/scrm [--merge]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from .hub import resolve_ckpt
from .model import load_scrm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into backbone weights (bf16) and save it")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if not a.merge:
        shutil.copytree(resolve_ckpt(a.ckpt), a.out, dirs_exist_ok=True)
        for f in ("trainer_state.pt",):
            p = os.path.join(a.out, f)
            if os.path.exists(p):
                os.remove(p)
        print(f"[export] copied adapter+head+tokenizer to {a.out}")
        return
    model = load_scrm(a.ckpt, a.device)
    bb = model.backbone
    if hasattr(bb, "merge_and_unload"):
        bb = bb.merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    bb.save_pretrained(os.path.join(a.out, "backbone_merged"))
    torch.save(model.head_state_dict(), os.path.join(a.out, "scrm_head.pt"))
    model.tokenizer.save_pretrained(os.path.join(a.out, "tokenizer"))
    cfg = {"model": model.cfg, "render": model.render_cfg}
    with open(os.path.join(a.out, "scrm_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[export] merged model saved to {a.out}")


if __name__ == "__main__":
    main()
