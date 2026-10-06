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
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from .collator import align_slots, collate, to_device
from .config import load_config
from .data import TrainStream, build_groups, make_train_loader, EvalSet
from .evaluate import run_eval, run_perm_eval, _amp
from .gradcache import grad_cache_step
from .losses import compute_loss, reduce_loss
from .metrics import flatten, val_index
from .hub import HubSync, resolve_ckpt
from .model import build_scrm
from .render import Renderer
from .wandb_utils import init_wandb, NullRun


# ----------------------------------------------------------------------------- distributed
def dist_info():
    """(world_size, rank, local_rank) from the torchrun env; defaults are the single-process values."""
    world = int(os.environ.get("WORLD_SIZE") or 1)
    rank = int(os.environ.get("RANK") or 0)
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    return world, rank, local_rank


def dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_grads(params):
    """One coalesced SUM all-reduce over all trainable params' grads (flattened per dtype). Params without a grad
    contribute zeros so every rank reduces the same layout; after this all ranks hold identical grads."""
    if not dist_on():
        return
    by_dtype: dict = {}
    for p in params:
        by_dtype.setdefault(p.dtype, []).append(p)
    for ps in by_dtype.values():
        gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
        flat = torch._utils._flatten_dense_tensors(gs)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for p, g in zip(ps, torch._utils._unflatten_dense_tensors(flat, gs)):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)


def broadcast_params(model):
    """Rank 0's trainable weights (LoRA + set block/head) win; called after build and after a resume load."""
    if not dist_on():
        return
    for p in model.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=0)


def reduce_ints(x, device, op=dist.ReduceOp.SUM):
    if not dist_on():
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=op)
    return int(t.item())


# ----------------------------------------------------------------------------- optimiser
def make_optimizer(model, tcfg: dict):
    lora, dec, nodec = [], [], []
    for n, p in model.backbone.named_parameters():
        if p.requires_grad:
            lora.append(p)
    for n, p in model.head_module().named_parameters():
        (nodec if (p.ndim < 2) else dec).append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": tcfg["lr_lora"], "weight_decay": 0.0, "name": "lora"})
    groups.append({"params": dec, "lr": tcfg["lr_head"], "weight_decay": tcfg["weight_decay"], "name": "head"})
    groups.append({"params": nodec, "lr": tcfg["lr_head"], "weight_decay": 0.0, "name": "head_nodecay"})
    cuda = torch.cuda.is_available() and model.head_device().is_cuda
    if tcfg.get("optim", "adamw_8bit") == "adamw_8bit":
        if not cuda:
            print("[scrm] adamw_8bit needs CUDA; using torch AdamW on this device", flush=True)
        else:
            import bitsandbytes as bnb      # tensors < 4096 elements automatically keep 32-bit state
            return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=cuda or None)


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


def save_ckpt(model, opt, sched, step, cfg, out_dir, state_extra: dict, name=None, keep_last=3, with_opt=True,
              hub: HubSync | None = None):
    if hub is not None:
        hub.wait()   # the previous upload may still be reading a directory replaced / pruned below
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
    if hub is not None:
        hub.push(path, name or "last", f"step {step}, best {state_extra.get('best_metric', 'val_loss')} "
                                       f"{state_extra.get('best', float('nan')):.4f}")
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


def estimate_steps(cfg, renderer, seed, world: int = 1):
    d = cfg["data"]
    groups = build_groups(d, d["train_split"], d["filters"], d["streaming"], seed)
    total = sum(g.size for g in groups) if all(g.size is not None for g in groups) else None
    if total is None:
        raise ValueError("train.epochs needs a non-streaming dataset (unknown size); set train.max_steps instead")
    ts = TrainStream(groups, renderer, d, seed, max_examples=min(total, 1500))
    nb = ne = 0
    for b in ts:
        nb += 1
        ne += b["candidate_mask"].size(0)
    avg = ne / max(nb, 1)
    return int(math.ceil(cfg["train"]["epochs"] * total / (avg * cfg["train"]["grad_accum"] * max(1, world)))), total


@torch.no_grad()
def _train_pair_acc(r, tiers, M):
    d = r.unsqueeze(2) - r.unsqueeze(1)
    return (((d > 0) & M).sum().float() / M.sum().clamp(min=1)).item()


def accumulate_step(model, batches, *, lcfg, d_batch, device, amp, n_valid, grad_cache=False,
                    chunk_tokens: int = 16384, act_tokens: int | None = None, renderer=None, seed: int = 0, step: int = 0,
                    use_perm=False):
    """Forward + backward over one optimizer step's micro-batches and accumulate gradients into `model`.

    `n_valid` is the number of valid sets over the WHOLE accumulation window (already globally reduced under DDP),
    so the per-example losses are divided consistently. Both the standard path and `train.grad_cache` live here:
    grad-cache groups each micro-batch into backbone chunks (see `scrm.gradcache`). Returns loss/parts/pair/example/
    token sums plus the last micro-batch's rewards/batch for logging."""
    tot, parts_acc, pairs, ex_n, tok_n = 0.0, {}, 0, 0, 0
    src_acc: dict[str, list] = {}
    last_r = last_b = None
    max_tokens = d_batch["max_tokens_per_batch"]
    perm_detach = lcfg.get("perm_detach", True)
    for b in batches:
        bd = to_device(b, device)
        its2, b2 = None, None
        if use_perm:
            its2 = [renderer.reshuffle(i, np.random.default_rng(seed * 7 + step * 131 + k))
                    for k, i in enumerate(b["items"])]
            b2 = to_device(collate(its2, renderer.pad_id), device)
        if grad_cache:
            packs = [(bd["pack"], bd["candidate_mask"])]
            if use_perm and not perm_detach:
                packs.append((b2["pack"], b2["candidate_mask"]))

            def head_loss(leaves):
                rewards = [model.head(leaves[i], packs[i][0], packs[i][1]) for i in range(len(packs))]
                rr = rewards[0].float()
                r2a = None
                if use_perm:
                    if perm_detach:
                        with torch.no_grad(), _amp(device, amp):
                            rr2 = model.score(b2, max_tokens=max_tokens).float()
                    else:
                        rr2 = rewards[1].float()
                    r2a = torch.zeros_like(rr)
                    for k, (i1, i2) in enumerate(zip(b["items"], its2)):
                        n = len(i1.order)
                        r2a[k, :n] = rr2[k, align_slots(i1.order, i2.order)]
                lo = compute_loss(rr, bd["tiers"], lcfg, bd["pair_mask"], r2a)
                return reduce_loss(lo, n_valid), (rr, lo)

            loss, (rr, lo), _ = grad_cache_step(model, packs, chunk_tokens, act_tokens=act_tokens,
                                                amp_ctx=lambda: _amp(device, amp), head_loss=head_loss)
            r = rr.detach().float()
        else:
            with _amp(device, amp):
                r = model.score(bd, max_tokens=max_tokens)
            r = r.float()
            r2 = None
            if use_perm:
                with torch.set_grad_enabled(not perm_detach):
                    with _amp(device, amp):
                        rr2 = model.score(b2, max_tokens=max_tokens).float()
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
        pairs += int(lo.n_pairs.sum())
        ex_n += r.size(0)
        for s, l, v in zip(b["source_id"], lo.per_example.detach().cpu().tolist(), lo.valid.cpu().tolist()):
            if v:
                a = src_acc.setdefault(s, [0.0, 0])
                a[0] += l
                a[1] += 1
        tok_n += int(b["n_tokens"])
        last_r, last_b = r.detach(), bd
    return {"loss": tot, "parts": parts_acc, "pairs": pairs, "examples": ex_n, "tokens": tok_n,
            "last_r": last_r, "last_b": last_b, "src": src_acc}


# ----------------------------------------------------------------------------- main
def train(cfg: dict, resume: str | None = None):
    tcfg, lcfg, d = cfg["train"], cfg["loss"], cfg["data"]
    world, rank, local_rank = dist_info()
    if world > 1:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=int(tcfg.get("ddp_timeout_min", 240))))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    seed_all(cfg["seed"] + 10007 * rank)
    dev = tcfg["device"]
    if world > 1 and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    if dist_on():
        dist.barrier()
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    model, tok = build_scrm(cfg["model"], device, seed=cfg["seed"])
    model.render_cfg = dict(d["render"])
    renderer = Renderer(tok, d["render"])
    model.train()
    broadcast_params(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    gpu = ""
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu = f" ({p.name}, {p.total_memory / 2**30:.1f} GiB; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    if rank == 0:
        print(f"[scrm] device={device}{gpu} trainable params={n_train/1e6:.2f}M / {n_all/1e6:.2f}M total "
              f"(world={world}, rank={rank})", flush=True)

    max_steps = tcfg["max_steps"]
    if tcfg.get("epochs"):
        max_steps, nrows = estimate_steps(cfg, renderer, cfg["seed"], world)
        if rank == 0:
            print(f"[scrm] epochs={tcfg['epochs']} over {nrows} rows -> ~{max_steps} optimizer steps", flush=True)
    warm = tcfg["warmup_steps"] if tcfg.get("warmup_ratio") is None else int(tcfg["warmup_ratio"] * max_steps)
    opt = make_optimizer(model, tcfg)
    sched = make_scheduler(opt, warm, max_steps, tcfg["min_lr_ratio"])

    # checkpoint selection + early stopping on validation, at every eval: train.best_metric = val_index (highest
    # Decision-Index-style chance-corrected top-1, sources weighted equally) or val_loss (lowest validation loss).
    # The Decision Index is a test benchmark only. bad_evals = evals since the metric improved.
    metric = tcfg.get("best_metric") or "val_index"
    if metric not in ("val_index", "val_loss"):
        raise ValueError(f"train.best_metric must be val_index or val_loss, got {metric!r}")
    sign = 1.0 if metric == "val_loss" else -1.0   # internal score = sign * metric, lower is better
    step, best, bad_evals = 0, float("inf") if metric == "val_loss" else float("-inf"), 0
    meta = lambda: {"best": best, "best_metric": metric, "optim": tcfg["optim"], "bad_evals": bad_evals}
    if resume:
        if resume == "auto":
            cks = list_ckpts(out_dir)
            resume = cks[-1] if cks else None
        if resume:
            resume = resolve_ckpt(resume)   # hf://... works when the run uploaded with hub.include_optimizer
            st = load_ckpt(model, opt, sched, resume)
            if st.get("best_metric", "val_loss") != metric or st.get("optim") != tcfg["optim"]:
                raise ValueError(f"cannot resume {resume}: checkpoint has best_metric={st.get('best_metric')!r}, "
                                 f"optim={st.get('optim')!r}; this run uses best_metric={metric!r}, optim={tcfg['optim']!r}")
            step, best, bad_evals = st["step"], st["best"], st.get("bad_evals", 0)
            broadcast_params(model)
            if rank == 0:
                print(f"[scrm] resumed from {resume} at step {step}", flush=True)

    use_perm = lcfg.get("w_perm", 0.0) > 0
    loader, stream = make_train_loader(cfg, renderer, cfg["seed"] + step, keep_items=use_perm, world=world, rank=rank)
    it = iter(loader)
    evalset = testset = dindex = None
    bcfg = cfg["benchmarks"]
    wb = init_wandb(cfg["wandb"], cfg, out_dir) if rank == 0 else NullRun()
    mlog = open(os.path.join(out_dir, "metrics.jsonl"), "a") if rank == 0 else None
    hub = HubSync(cfg["hub"], out_dir) if rank == 0 else None

    def log(data, s):
        if rank != 0:
            return
        data = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in data.items()}
        mlog.write(json.dumps({"step": s, **data}) + "\n"); mlog.flush()
        wb.log({k: v for k, v in data.items() if v is not None}, s)

    def consider_best(value, s):
        """Save `best` when `value` (of train.best_metric) improves on it; else count a bad eval."""
        nonlocal best, bad_evals
        if math.isfinite(value) and sign * value < sign * best - tcfg["early_stop_min_delta"]:
            best, bad_evals = value, 0
            save_ckpt(model, opt, sched, s, cfg, out_dir, meta(), name="best", hub=hub)   # with optimizer: `best` is resumable
        else:
            bad_evals += 1

    def stop_now() -> bool:
        pat = tcfg["early_stop_patience"]
        if pat and bad_evals >= pat:
            print(f"[scrm] early stop at step {step}: {metric} did not improve for {bad_evals} evals "
                  f"(best {best:.4f})", flush=True)
            return True
        return False

    def do_eval(s):
        nonlocal evalset
        if evalset is None:
            evalset = EvalSet.from_config(cfg, renderer)
            print(f"[eval] set: {evalset.describe()}", flush=True)
        t0 = time.time()
        m = run_eval(model, evalset.batches(), lcfg, device, amp, max_tokens=d["batch"]["max_tokens_per_batch"])
        pm = run_perm_eval(model, evalset, renderer, d["perm_eval_rows"], d["batch"], device, amp)
        flat = flatten(m)
        flat.update({f"eval/perm/{k}": v for k, v in pm.items()})
        flat["eval/loss"] = m.get("all", {}).get("loss", float("nan"))
        vi, vraw = val_index(m)
        flat["eval/index"], flat["eval/raw_index"] = vi, vraw
        log(flat, s)
        wb.log_group_table("eval/by_group", m, s)
        a = m.get("all", {})
        print(f"[eval] step {s}: index={vi:.2f} raw={vraw:.2f} pair_acc={a.get('pair_acc', 0):.4f} "
              f"top1={a.get('top1', 0):.4f} mrr={a.get('mrr', 0):.4f} ndcg={a.get('ndcg', 0):.4f} "
              f"loss={a.get('loss', 0):.4f} ece={a.get('ece_top1', float('nan')):.3f} "
              f"overconf={a.get('overconf', float('nan')):+.3f} "
              f"perm_agree={pm.get('rank_agree', float('nan')):.3f} ({time.time()-t0:.0f}s)", flush=True)
        consider_best(a.get("loss", float("inf")) if metric == "val_loss" else vi, s)
        model.train()

    done_bench = {}   # benchmark -> last step it ran at

    def do_benchmarks(s, final=False):
        """Test benchmarks (log only; `best` is selected on validation): test split + Decision Index sample.
        Decision Index runs at steps divisible by its `every` (null = every call) and always at the end."""
        nonlocal testset, dindex
        if bcfg.get("test_split") and done_bench.get("test") != s:
            done_bench["test"] = s
            if testset is None:
                testset = EvalSet.from_config(cfg, renderer, bcfg["test_split"])
                print(f"[test] set: {testset.describe()}", flush=True)
            t0 = time.time()
            m = run_eval(model, testset.batches(), lcfg, device, amp, max_tokens=d["batch"]["max_tokens_per_batch"])
            flat = flatten(m, "test")
            flat["test/loss"] = m.get("all", {}).get("loss", float("nan"))
            ti, traw = val_index(m)
            flat["test/index"], flat["test/raw_index"] = ti, traw
            log(flat, s)
            wb.log_group_table("test/by_group", m, s)
            a = m.get("all", {})
            print(f"[test] step {s}: index={ti:.2f} raw={traw:.2f} pair_acc={a.get('pair_acc', 0):.4f} "
                  f"top1={a.get('top1', 0):.4f} mrr={a.get('mrr', 0):.4f} loss={a.get('loss', 0):.4f} "
                  f"ece={a.get('ece_top1', float('nan')):.3f} overconf={a.get('overconf', float('nan')):+.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        di = bcfg.get("decision_index") or {}
        if di.get("enabled") and done_bench.get("dindex") != s and (final or not di.get("every") or s % di["every"] == 0):
            done_bench["dindex"] = s
            from .dindex import DecisionIndexEval
            if dindex is None:
                dindex = DecisionIndexEval(di)
                print(f"[dindex] sample: {len(dindex.rows)} requests from {di['rows']}", flush=True)
            m = dindex.run(model, device, amp, os.path.join(out_dir, "decision_index", f"step-{s:07d}"))
            log({f"dindex/{k}": v for k, v in m.items()}, s)
            print(f"[dindex] step {s}: index={m['index']:.2f} raw={m['raw_index']:.2f} "
                  f"answered={m['answered_frac']:.3f} ({m['seconds']:.0f}s)", flush=True)
        model.train()

    di0 = bcfg.get("decision_index") or {}
    if di0.get("enabled") and rank == 0:   # fail at step 0, not at the first benchmark hours in, if the suite / sample / kit is missing
        from .dindex import DecisionIndexEval
        dindex = DecisionIndexEval(di0)
        print(f"[dindex] sample: {len(dindex.rows)} requests from {di0['rows']}", flush=True)

    if tcfg.get("eval_at_start") and step == 0 and rank == 0:
        do_eval(0)

    def reduce_log(vals):
        if not dist_on():
            return list(vals)
        t = torch.tensor(list(vals), device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    saved_at = None
    win_tok, win_time, win_steps = 0, 0.0, 0   # tokens / wall time / steps since the last log (excludes eval / saves)
    while step < max_steps:
        t_step = time.time()
        batches, ok = [], 1
        try:
            batches = [next(it) for _ in range(tcfg["grad_accum"])]
        except StopIteration:
            ok = 0
        if world > 1:
            ok = reduce_ints(ok, device, dist.ReduceOp.MIN)
        if not ok:
            if rank == 0:
                print("[scrm] data exhausted", flush=True)
            break
        n_valid = max(1, sum(int(b["pair_mask"].flatten(1).any(1).sum()) for b in batches))
        n_valid = reduce_ints(n_valid, device, dist.ReduceOp.SUM)   # global over ranks: SUM-reduced grads == mean grad
        st = accumulate_step(model, batches, lcfg=lcfg, d_batch=d["batch"], device=device, amp=amp, n_valid=n_valid,
                             grad_cache=tcfg.get("grad_cache", False),
                             chunk_tokens=tcfg.get("grad_cache_chunk_tokens", 16384),
                             act_tokens=tcfg.get("grad_cache_act_tokens"),
                             renderer=renderer, seed=cfg["seed"], step=step, use_perm=use_perm)
        tot, pairs, ex_n = st["loss"], st["pairs"], st["examples"]
        win_tok += st["tokens"]
        last_r, last_b, parts_acc, src_acc = st["last_r"], st["last_b"], st["parts"], st["src"]
        all_reduce_grads([p for p in model.parameters() if p.requires_grad])   # one coalesced SUM before clipping
        params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]))
        if not math.isfinite(gnorm):
            if rank == 0:
                print(f"[scrm] non-finite grad norm at step {step}; skipping update", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        win_time += time.time() - t_step; win_steps += 1

        if step % tcfg["log_every"] == 0 or step == 1:
            tot, pairs, ex_n, tok_acc = reduce_log([tot, pairs, ex_n, win_tok])   # SUM over ranks (global loss)
            dt = max(win_time, 1e-6)
            m = last_r[last_b["candidate_mask"]]
            rec = {"train/loss": tot, "train/grad_norm": gnorm, "train/pairs_per_step": pairs, "train/examples_per_step": ex_n,
                   "train/tokens_per_s": tok_acc / dt, "train/s_per_step": dt / win_steps,
                   "train/pair_acc_lastmb": _train_pair_acc(last_r, last_b["tiers"], last_b["pair_mask"]),
                   "train/reward_mean": float(m.mean()), "train/reward_std": float(m.std()) if m.numel() > 1 else 0.0}
            rec.update({f"train/loss_{k}": v for k, v in parts_acc.items()})
            for s, (l, n) in src_acc.items():   # mean per-set loss per source over the log window (rank-local)
                rec[f"train/source/{s}/loss"] = l / n
                rec[f"train/source/{s}/n_sets"] = n
            for g in opt.param_groups:
                rec[f"lr/{g['name']}"] = g["lr"]
            if device.type == "cuda":
                rec["sys/gpu_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log(rec, step)
            if rank == 0:
                mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.0f}G" if device.type == "cuda" else ""
                print(f"[train] step {step}/{max_steps} loss={tot:.4f} gnorm={gnorm:.2f} ex/step={ex_n} pairs={pairs} "
                      f"tok/s={tok_acc/dt:.0f} s/step={dt/win_steps:.1f}{mem}", flush=True)
            win_tok, win_time, win_steps = 0, 0.0, 0
        if tcfg["hist_every"] and step % tcfg["hist_every"] == 0 and wb.enabled:
            wb.log_hist("train/reward_hist", last_r[last_b["candidate_mask"]].cpu().numpy(), step)
        stop = False
        if tcfg["eval_every"] and step % tcfg["eval_every"] == 0 and rank == 0:
            do_eval(step)
            stop = stop_now()
        saving = bool(tcfg["save_every"]) and step % tcfg["save_every"] == 0
        if saving and rank == 0:
            save_ckpt(model, opt, sched, step, cfg, out_dir, meta(), keep_last=tcfg["keep_last"], hub=hub)   # before the log-only benchmarks: a benchmark crash must not lose the interval
            saved_at = step
            do_benchmarks(step)
        if world > 1 and ((tcfg["eval_every"] and step % tcfg["eval_every"] == 0) or saving):   # same steps on every rank
            stop = bool(reduce_ints(int(stop), device, dist.ReduceOp.MAX))
        if stop:
            break

    path = None
    if rank == 0:
        final_eval = bool(tcfg["eval_every"]) and step % tcfg["eval_every"] != 0
        if final_eval:
            do_eval(step)
        do_benchmarks(step, final=True)
        if saved_at == step and not final_eval:   # this exact state was just saved (and uploaded) in the loop
            path = os.path.join(out_dir, f"step_{step:07d}")
        else:
            path = save_ckpt(model, opt, sched, step, cfg, out_dir, meta(), keep_last=tcfg["keep_last"], hub=hub)
        print(f"[scrm] done. final checkpoint: {path}; best {metric}={best:.4f}", flush=True)
        hub.wait()
        wb.finish()
        mlog.close()
    if dist_on():
        dist.barrier()
        dist.destroy_process_group()
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
