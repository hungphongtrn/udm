"""Plain PyTorch single-GPU training loop for SCRM.

python -m scrm.train --config configs/scrm_qwen3_5_4b_24gb.yaml [--resume auto|DIR] [key.sub=value ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time

import numpy as np
import torch

from .collator import align_slots, collate, to_device
from .config import load_config
from .data import TrainStream, build_groups, make_train_loader, EvalSet, select_eval_rows
from .evaluate import run_eval, run_perm_eval, _amp
from .losses import compute_loss, reduce_loss
from .metrics import flatten
from .model import build_scrm
from .render import Renderer
from .wandb_utils import init_wandb


# ----------------------------------------------------------------------------- optimiser
def make_optimizer(model, tcfg: dict):
    lora, dec, nodec = [], [], []
    for n, p in model.backbone.named_parameters():
        if p.requires_grad:
            lora.append(p)
    for n, p in model.set_encoder.named_parameters():
        (nodec if (p.ndim < 2) else dec).append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": tcfg["lr_lora"], "weight_decay": 0.0, "name": "lora"})
    groups.append({"params": dec, "lr": tcfg["lr_head"], "weight_decay": tcfg["weight_decay"], "name": "head"})
    groups.append({"params": nodec, "lr": tcfg["lr_head"], "weight_decay": 0.0, "name": "head_nodecay"})
    fused = torch.cuda.is_available() and next(model.set_encoder.parameters()).is_cuda
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=fused or None)


def make_scheduler(opt, warmup: int, total: int, min_ratio: float):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ----------------------------------------------------------------------------- checkpoints
def list_ckpts(out_dir):
    if not os.path.isdir(out_dir):
        return []
    ds = [d for d in os.listdir(out_dir) if re.fullmatch(r"step_\d+", d)]
    return [os.path.join(out_dir, d) for d in sorted(ds)]


def save_ckpt(model, opt, sched, step, cfg, out_dir, state_extra: dict, name=None, keep_last=3, with_opt=True):
    path = os.path.join(out_dir, name or f"step_{step:07d}")
    tmp = path + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if with_opt:
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, **state_extra},
                   os.path.join(tmp, "trainer_state.pt"))
    else:
        torch.save({"step": step, **state_extra}, os.path.join(tmp, "trainer_state.pt"))
    with open(os.path.join(tmp, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.rename(tmp, path)
    if name is None:
        for old in list_ckpts(out_dir)[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)
    return path


def load_ckpt(model, opt, sched, path):
    from safetensors.torch import load_file
    ad = os.path.join(path, "adapter", "adapter_model.safetensors")
    if os.path.exists(ad):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.backbone, load_file(ad))
    model.load_head_state_dict(torch.load(os.path.join(path, "scrm_head.pt"), map_location="cpu"))
    st = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu", weights_only=False)
    if opt is not None and "optimizer" in st:
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
    return st


# ----------------------------------------------------------------------------- misc
def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def estimate_steps(cfg, renderer, seed):
    d = cfg["data"]
    groups = build_groups(d, d["train_split"], d["filters"], d["streaming"], seed)
    total = sum(g.size for g in groups) if all(g.size is not None for g in groups) else None
    if total is None:
        raise ValueError("train.epochs needs a non-streaming dataset (unknown size); set train.max_steps instead")
    ts = TrainStream(groups, renderer, d, seed, max_examples=min(total, 1500))
    nb = ne = 0
    for b in ts:
        nb += 1
        ne += b["input_ids"].size(0)
    avg = ne / max(nb, 1)
    return int(math.ceil(cfg["train"]["epochs"] * total / (avg * cfg["train"]["grad_accum"]))), total


@torch.no_grad()
def _train_pair_acc(r, tiers, M):
    d = r.unsqueeze(2) - r.unsqueeze(1)
    return (((d > 0) & M).sum().float() / M.sum().clamp(min=1)).item()


# ----------------------------------------------------------------------------- main
def train(cfg: dict, resume: str | None = None):
    tcfg, lcfg, d = cfg["train"], cfg["loss"], cfg["data"]
    seed_all(cfg["seed"])
    dev = tcfg["device"]
    device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    model, tok = build_scrm(cfg["model"], device, seed=cfg["seed"])
    model.render_cfg = dict(d["render"])
    renderer = Renderer(tok, d["render"])
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[scrm] device={device} trainable params={n_train/1e6:.2f}M / {n_all/1e6:.2f}M total", flush=True)

    max_steps = tcfg["max_steps"]
    if tcfg.get("epochs"):
        max_steps, nrows = estimate_steps(cfg, renderer, cfg["seed"])
        print(f"[scrm] epochs={tcfg['epochs']} over {nrows} rows -> ~{max_steps} optimizer steps", flush=True)
    warm = tcfg["warmup_steps"] if tcfg.get("warmup_ratio") is None else int(tcfg["warmup_ratio"] * max_steps)
    opt = make_optimizer(model, tcfg)
    sched = make_scheduler(opt, warm, max_steps, tcfg["min_lr_ratio"])

    step, best = 0, -1.0
    if resume:
        if resume == "auto":
            cks = list_ckpts(out_dir)
            resume = cks[-1] if cks else None
        if resume:
            st = load_ckpt(model, opt, sched, resume)
            step, best = st["step"], st.get("best", -1.0)
            print(f"[scrm] resumed from {resume} at step {step}", flush=True)

    use_perm = lcfg.get("w_perm", 0.0) > 0
    loader, stream = make_train_loader(cfg, renderer, cfg["seed"] + step, keep_items=use_perm)
    it = iter(loader)
    evalset = None
    wb = init_wandb(cfg["wandb"], cfg, out_dir)
    mlog = open(os.path.join(out_dir, "metrics.jsonl"), "a")

    def log(data, s):
        data = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in data.items()}
        mlog.write(json.dumps({"step": s, **data}) + "\n"); mlog.flush()
        wb.log({k: v for k, v in data.items() if v is not None}, s)

    def do_eval(s):
        nonlocal evalset, best
        if evalset is None:
            evalset = EvalSet.from_config(cfg, renderer)
            print(f"[eval] set: {len(evalset.items)} items ({evalset.n_dropped} dropped)", flush=True)
        t0 = time.time()
        m = run_eval(model, evalset.batches(), lcfg, device, amp)
        pm = run_perm_eval(model, evalset, renderer, d["perm_eval_rows"], d["batch"], device, amp)
        flat = flatten(m)
        flat.update({f"eval/perm/{k}": v for k, v in pm.items()})
        flat["eval/loss"] = m.get("all", {}).get("loss", float("nan"))
        log(flat, s)
        wb.log_group_table("eval/by_group", m, s)
        a = m.get("all", {})
        print(f"[eval] step {s}: pair_acc={a.get('pair_acc', 0):.4f} top1={a.get('top1', 0):.4f} "
              f"mrr={a.get('mrr', 0):.4f} ndcg={a.get('ndcg', 0):.4f} loss={a.get('loss', 0):.4f} "
              f"perm_agree={pm.get('rank_agree', float('nan')):.3f} ({time.time()-t0:.0f}s)", flush=True)
        acc = a.get("pair_acc", -1.0)
        if acc > best:
            best = acc
            save_ckpt(model, opt, sched, s, cfg, out_dir, {"best": best}, name="best", with_opt=False)
        model.train()

    if tcfg.get("eval_at_start") and step == 0:
        do_eval(0)

    t_last, tok_acc, real_tok_acc = time.time(), 0, 0
    while step < max_steps:
        try:
            batches = [next(it) for _ in range(tcfg["grad_accum"])]
        except StopIteration:
            print("[scrm] data exhausted", flush=True)
            break
        n_valid = max(1, sum(int(b["pair_mask"].flatten(1).any(1).sum()) for b in batches))
        tot, parts_acc, pairs, ex_n = 0.0, {}, 0, 0
        for b in batches:
            bd = to_device(b, device)
            with _amp(device, amp):
                r = model(bd["input_ids"], bd["attention_mask"], bd["candidate_positions"], bd["candidate_mask"])
            r = r.float()
            r2 = None
            if use_perm:
                its2 = [renderer.reshuffle(i, np.random.default_rng(cfg["seed"] * 7 + step * 131 + k)) for k, i in enumerate(b["items"])]
                b2 = to_device(collate(its2, renderer.pad_id), device)
                with torch.set_grad_enabled(not lcfg.get("perm_detach", True)):
                    with _amp(device, amp):
                        rr2 = model(b2["input_ids"], b2["attention_mask"], b2["candidate_positions"], b2["candidate_mask"]).float()
                r2 = torch.zeros_like(r)
                for k, (i1, i2) in enumerate(zip(b["items"], its2)):
                    n = len(i1.order)
                    r2[k, :n] = rr2[k, align_slots(i1.order, i2.order)]
            lo = compute_loss(r, bd["tiers"], lcfg, bd["pair_mask"], r2)
            loss = reduce_loss(lo, n_valid)
            loss.backward()
            tot += float(loss.detach())
            for k, v in lo.parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + float((v.detach() * lo.valid).sum()) / n_valid
            pairs += int(lo.n_pairs.sum()); ex_n += r.size(0)
            tok_acc += int(b["attention_mask"].numel()); real_tok_acc += int(b["attention_mask"].sum())
            last_r, last_b = r.detach(), bd
        params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]))
        if not math.isfinite(gnorm):
            print(f"[scrm] non-finite grad norm at step {step}; skipping update", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1

        if step % tcfg["log_every"] == 0 or step == 1:
            now = time.time(); dt = max(now - t_last, 1e-6)
            m = last_r[last_b["candidate_mask"]]
            rec = {"train/loss": tot, "train/grad_norm": gnorm, "train/pairs_per_step": pairs, "train/examples_per_step": ex_n,
                   "train/tokens_per_s": real_tok_acc / dt, "train/padded_tokens_per_s": tok_acc / dt,
                   "train/pair_acc_lastmb": _train_pair_acc(last_r, last_b["tiers"], last_b["pair_mask"]),
                   "train/reward_mean": float(m.mean()), "train/reward_std": float(m.std()) if m.numel() > 1 else 0.0}
            rec.update({f"train/loss_{k}": v for k, v in parts_acc.items()})
            for g in opt.param_groups:
                rec[f"lr/{g['name']}"] = g["lr"]
            if device.type == "cuda":
                rec["sys/gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log(rec, step)
            print(f"[train] step {step}/{max_steps} loss={tot:.4f} gnorm={gnorm:.2f} ex/step={ex_n} pairs={pairs} "
                  f"tok/s={real_tok_acc/dt:.0f}", flush=True)
            t_last, tok_acc, real_tok_acc = now, 0, 0
        if tcfg["hist_every"] and step % tcfg["hist_every"] == 0 and wb.enabled:
            wb.log_hist("train/reward_hist", last_r[last_b["candidate_mask"]].cpu().numpy(), step)
        if tcfg["eval_every"] and step % tcfg["eval_every"] == 0:
            do_eval(step)
        if tcfg["save_every"] and step % tcfg["save_every"] == 0:
            save_ckpt(model, opt, sched, step, cfg, out_dir, {"best": best}, keep_last=tcfg["keep_last"])

    if tcfg["eval_every"] and step % tcfg["eval_every"] != 0:
        do_eval(step)
    path = save_ckpt(model, opt, sched, step, cfg, out_dir, {"best": best}, keep_last=tcfg["keep_last"])
    print(f"[scrm] done. final checkpoint: {path}; best pair_acc={best:.4f}", flush=True)
    wb.finish(); mlog.close()
    return {"step": step, "best": best, "ckpt": path}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", default=None, help="'auto' or checkpoint dir")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.max_steps=100 model.lora.r=16")
    a = ap.parse_args(argv)
    train(load_config(a.config, a.overrides), a.resume)


if __name__ == "__main__":
    main()
