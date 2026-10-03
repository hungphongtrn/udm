"""Ranking metrics with per-family / per-source breakdown."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .losses import tier_pair_mask


@torch.no_grad()
def example_metrics(r: torch.Tensor, tiers: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-example metrics from rewards [B,N] and tiers [B,N] (-1 pad). Returns dict of [B] numpy arrays
    (kendall_tau is NaN for examples with < 3 distinct tiers)."""
    r = r.float().cpu()
    tiers = tiers.cpu()
    valid = tiers >= 0
    M = tier_pair_mask(tiers)
    npairs = M.sum((1, 2)).float()
    diff = r.unsqueeze(2) - r.unsqueeze(1)
    correct = ((diff > 0) & M).sum((1, 2)).float() + 0.5 * ((diff == 0) & M).sum((1, 2)).float()
    pair_acc = correct / npairs.clamp(min=1)
    # top-1 (ties share credit)
    rmax = r.masked_fill(~valid, float("-inf")).max(1, keepdim=True).values
    tops = (r == rmax) & valid
    tmin = tiers.masked_fill(~valid, 10**9).min(1, keepdim=True).values
    in0 = valid & (tiers == tmin)
    top1 = (tops & in0).sum(1).float() / tops.sum(1).clamp(min=1).float()
    # average-tie ranks
    gt = ((r.unsqueeze(1) > r.unsqueeze(2)) & valid.unsqueeze(1)).sum(2).float()     # #j with r_j > r_i
    eq = ((r.unsqueeze(1) == r.unsqueeze(2)) & valid.unsqueeze(1)).sum(2).float() - 1
    rank = 1 + gt + 0.5 * eq.clamp(min=0)
    first = rank.masked_fill(~in0, float("inf")).min(1).values
    mrr = 1.0 / first
    # NDCG with gains 2^rel-1, rel = tmax - tier
    tmax = tiers.max(1, keepdim=True).values
    rel = (tmax - tiers).clamp(min=0).float()
    gain = (2.0 ** rel - 1) * valid
    disc = 1.0 / torch.log2(1 + rank)
    dcg = (gain * disc).sum(1)
    sg = gain.sort(1, descending=True).values
    ideal_disc = 1.0 / torch.log2(2 + torch.arange(r.size(1), dtype=torch.float32))
    idcg = (sg * ideal_disc).sum(1)
    ndcg = dcg / idcg.clamp(min=1e-9)
    # Kendall tau-b vs tiers (only >=3 tiers)
    n = valid.sum(1).float()
    n0 = n * (n - 1) / 2
    iu = torch.triu(torch.ones(r.size(1), r.size(1), dtype=torch.bool), diagonal=1)
    pv = valid.unsqueeze(2) & valid.unsqueeze(1) & iu
    n1 = ((tiers.unsqueeze(2) == tiers.unsqueeze(1)) & pv).sum((1, 2)).float()
    n2 = ((diff == 0) & pv).sum((1, 2)).float()
    C = ((diff > 0) & M).sum((1, 2)).float()
    D = ((diff < 0) & M).sum((1, 2)).float()
    tau = (C - D) / torch.sqrt((n0 - n1).clamp(min=1e-9) * (n0 - n2).clamp(min=1e-9))
    ntier = torch.tensor([len(set(t[v].tolist())) for t, v in zip(tiers, valid)])
    tau = torch.where(ntier >= 3, tau, torch.full_like(tau, float("nan")))
    # calibration of the set softmax (the probabilities rank()/Decision Index use): confidence = top-1 probability vs
    # top-1 correctness; p_best = probability mass on the best tier, nll_best = -log p_best
    logp = torch.log_softmax(r.masked_fill(~valid, float("-inf")), 1)
    p = logp.exp()
    conf = p.max(1).values
    p_best = (p * in0).sum(1)
    return {"pair_acc": pair_acc.numpy(), "correct": correct.numpy(), "npairs": npairs.numpy(), "top1": top1.numpy(),
            "mrr": mrr.numpy(), "ndcg": ndcg.numpy(), "kendall_tau": tau.numpy(), "n_cands": n.numpy(),
            "conf": conf.numpy(), "brier_top1": ((conf - top1) ** 2).numpy(), "p_best": p_best.numpy(),
            "nll_best": (-torch.log(p_best.clamp(min=1e-12))).numpy()}


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: |mean hit - mean conf| per equal-width confidence bin, weighted by bin size."""
    if len(conf) == 0:
        return float("nan")
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).sum() for k in np.unique(b)) / len(conf))


class MetricAccumulator:
    def __init__(self):
        self.rows: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def add(self, r, tiers, families, sources, loss_per_example=None):
        m = example_metrics(r, tiers)
        if loss_per_example is not None:
            m["loss"] = loss_per_example.detach().float().cpu().numpy()
        for b in range(len(families)):
            for key in ("all", f"family/{families[b]}", f"source/{sources[b]}"):
                for k, v in m.items():
                    self.rows[key][k].append(float(v[b]))

    def compute(self) -> dict[str, dict[str, float]]:
        out = {}
        for key, cols in self.rows.items():
            d = {"n": len(cols["pair_acc"])}
            for k, v in cols.items():
                if k in ("correct", "npairs"):
                    continue
                a = np.asarray(v)
                d[k] = float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
            tot = float(np.sum(cols["npairs"]))
            d["pair_acc_micro"] = float(np.sum(cols["correct"]) / tot) if tot else float("nan")
            d["ece_top1"] = ece(np.asarray(cols["conf"]), np.asarray(cols["top1"]))
            d["overconf"] = d["conf"] - d["top1"]     # > 0: top-1 probability exceeds top-1 accuracy
            out[key] = d
        return out


def flatten(metrics: dict[str, dict[str, float]], prefix: str = "eval") -> dict[str, float]:
    flat = {}
    for g, d in metrics.items():
        for k, v in d.items():
            flat[f"{prefix}/{g}/{k}"] = v
    return flat


@torch.no_grad()
def permutation_metrics(r1: np.ndarray, r2: np.ndarray) -> dict[str, float]:
    """r1, r2: rewards of the same candidates under two shuffles (aligned)."""
    n = len(r1)
    s = np.std(np.stack([r1, r2]), axis=0)
    d1 = np.sign(r1[:, None] - r1[None, :])
    d2 = np.sign(r2[:, None] - r2[None, :])
    iu = np.triu_indices(n, 1)
    agree = float(np.mean(d1[iu] == d2[iu])) if n > 1 else 1.0
    return {"score_std": float(s.mean()), "abs_diff": float(np.abs(r1 - r2).mean()), "rank_agree": agree,
            "top1_agree": float(np.argmax(r1) == np.argmax(r2))}
