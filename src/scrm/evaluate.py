"""Evaluation: shared eval loop + CLI (python -m scrm.evaluate --ckpt DIR [--split test] [--filter k=v,v] ...)."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from .collator import collate, to_device, align_slots
from .data import EvalSet, make_batches
from .losses import compute_loss
from .metrics import MetricAccumulator, permutation_metrics
from .render import Renderer


def _amp(device, enabled=True):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def run_eval(model, batches, lcfg, device, amp=True, max_tokens=None) -> dict:
    was = model.training
    model.eval()
    acc = MetricAccumulator()
    for b in batches:
        bd = to_device(b, device)
        with _amp(device, amp):
            r = model.score(bd, max_tokens=max_tokens)
        lo = compute_loss(r.float(), bd["tiers"], lcfg, bd["pair_mask"])
        loss = lo.parts["bt"]
        acc.add(r.float().cpu(), b["tiers"], b["family"], b["source_id"], loss.cpu())
    model.train(was)
    return acc.compute()


@torch.no_grad()
def _rewards_for(model, items, renderer, bcfg, device, amp):
    out = [None] * len(items)
    idx = sorted(range(len(items)), key=lambda i: items[i].n_tokens)
    cur = []
    groups = []
    for i in idx:
        if cur and ((len(cur) + 1) * items[i].n_tokens > bcfg["max_tokens_per_batch"] or len(cur) >= bcfg["max_batch_size"]):
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    for g in groups:
        b = to_device(collate([items[i] for i in g], renderer.pad_id), device)
        with _amp(device, amp):
            r = model.score(b, max_tokens=bcfg["max_tokens_per_batch"]).float().cpu()
        for k, i in enumerate(g):
            out[i] = r[k, : len(items[i].order)].numpy()
    return out


@torch.no_grad()
def run_perm_eval(model, evalset: EvalSet, renderer: Renderer, n: int, bcfg, device, amp=True, seeds=(11, 22)) -> dict:
    """Score the same sets under two random candidate shuffles; report agreement."""
    was = model.training
    model.eval()
    base = evalset.items[:n]
    its1 = [renderer.reshuffle(i, np.random.default_rng(seeds[0] + k)) for k, i in enumerate(base)]
    its2 = [renderer.reshuffle(i, np.random.default_rng(seeds[1] + k)) for k, i in enumerate(base)]
    r1 = _rewards_for(model, its1, renderer, bcfg, device, amp)
    r2 = _rewards_for(model, its2, renderer, bcfg, device, amp)
    ms = []
    for a, b, i1, i2 in zip(r1, r2, its1, its2):
        ms.append(permutation_metrics(a, b[align_slots(i1.order, i2.order)]))
    model.train(was)
    if not ms:
        return {}
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def main(argv=None):
    from .config import load_config
    from .model import load_scrm
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="optional YAML for data settings (default: ckpt render cfg + defaults)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--filter", action="append", default=[], help="include filter col=v1,v2 (e.g. source_split=ood)")
    ap.add_argument("--exclude", action="append", default=[], help="exclude filter col=v1,v2")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--per_source", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    model = load_scrm(a.ckpt, device)
    model.eval()
    cfg["data"]["render"] = {**cfg["data"]["render"], **(model.render_cfg or {})}
    renderer = Renderer(model.tokenizer, cfg["data"]["render"])
    parse = lambda L: {k: v.split(",") for k, v in (x.split("=", 1) for x in L)}
    filters = {"include": parse(a.filter), "exclude": parse(a.exclude)}
    es = EvalSet.from_config(cfg, renderer, a.split, filters, a.max_rows, a.per_source)
    metrics = run_eval(model, es.batches(), cfg["loss"], device, max_tokens=cfg["data"]["batch"]["max_tokens_per_batch"])
    perm = run_perm_eval(model, es, renderer, cfg["data"]["perm_eval_rows"], cfg["data"]["batch"], device)
    res = {"ckpt": a.ckpt, "split": a.split, "filters": filters, "n_rows": es.n_rows, "n_dropped": es.n_dropped,
           "metrics": metrics, "permutation": perm}
    clean = json.loads(json.dumps(res, default=float).replace("NaN", "null"))
    s = json.dumps(clean, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    allm = metrics.get("all", {})
    print(f"[eval] split={a.split} n={allm.get('n')} pair_acc={allm.get('pair_acc', math.nan):.4f} "
          f"top1={allm.get('top1', math.nan):.4f} mrr={allm.get('mrr', math.nan):.4f} ndcg={allm.get('ndcg', math.nan):.4f}")
    if not a.out:
        print(s)
    return res


if __name__ == "__main__":
    main()
