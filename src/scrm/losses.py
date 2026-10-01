"""Tier-based ranking losses. All terms are returned PER EXAMPLE so the trainer can normalise over an
accumulation window (examples with no pair are masked out)."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


def tier_pair_mask(tiers: torch.Tensor) -> torch.Tensor:
    """M[b,i,j] = tier_i < tier_j (i strictly better). Padding (-1) and same-tier pairs excluded."""
    valid = tiers >= 0
    return (tiers.unsqueeze(2) < tiers.unsqueeze(1)) & valid.unsqueeze(2) & valid.unsqueeze(1)


def bt_loss(rewards: torch.Tensor, tiers: torch.Tensor, tau: float = 1.0, margin_alpha: float = 0.0,
            pair_mask: torch.Tensor | None = None):
    """Bradley-Terry over tier pairs. Returns (per_example [B], n_pairs [B]). Per-example MEAN over pairs."""
    M = tier_pair_mask(tiers) if pair_mask is None else pair_mask
    diff = rewards.unsqueeze(2) - rewards.unsqueeze(1)
    if margin_alpha:
        diff = diff - margin_alpha * (tiers.unsqueeze(1) - tiers.unsqueeze(2)).abs().float()
    l = F.softplus(-diff / tau)
    n = M.sum((1, 2))
    return (l * M).sum((1, 2)) / n.clamp(min=1), n


def _lse(x, mask):
    return torch.logsumexp(x.masked_fill(~mask, float("-inf")), dim=-1)


def listwise_loss(rewards, tiers):
    """-log P(top-ranked item in tier 0) under softmax over the set."""
    valid = tiers >= 0
    tmin = tiers.masked_fill(~valid, 10**9).min(1, keepdim=True).values
    top = valid & (tiers == tmin)
    return -(_lse(rewards, top) - _lse(rewards, valid))


def plackett_luce_loss(rewards, tiers, max_levels: int = 64):
    """Tier-wise Plackett-Luce: for each tier k, -log P(pick a tier-k item among remaining tiers >= k).
    Averaged over the non-trivial steps."""
    valid = tiers >= 0
    B = rewards.size(0)
    tot = rewards.new_zeros(B)
    cnt = rewards.new_zeros(B)
    kmax = int(min(tiers.max().item(), max_levels)) if valid.any() else -1
    for k in range(kmax + 1):
        grp = valid & (tiers == k)
        rem = valid & (tiers >= k)
        nontrivial = grp.any(1) & (rem.sum(1) > grp.sum(1))
        step = -(_lse(rewards, grp.clone().masked_fill(~grp.any(1, keepdim=True), True)) - _lse(rewards, rem | ~rem.any(1, keepdim=True)))
        tot = tot + torch.where(nontrivial, step, torch.zeros_like(step))
        cnt = cnt + nontrivial.float()
    return tot / cnt.clamp(min=1)


def center_loss(rewards, tiers):
    valid = (tiers >= 0).float()
    mean = (rewards * valid).sum(1) / valid.sum(1).clamp(min=1)
    return mean ** 2


def perm_consistency_loss(r1, r2_aligned, mask):
    """MSE between rewards of two shuffles of the same set (r2 already aligned to r1's order)."""
    m = mask.float()
    return (((r1 - r2_aligned) ** 2) * m).sum(1) / m.sum(1).clamp(min=1)


class LossOut(NamedTuple):
    per_example: torch.Tensor     # [B] weighted total
    valid: torch.Tensor           # [B] bool (has >=1 trainable pair)
    n_pairs: torch.Tensor         # [B]
    parts: dict                   # name -> [B]


def compute_loss(rewards, tiers, lcfg: dict, pair_mask=None, rewards_perm=None) -> LossOut:
    bt, n = bt_loss(rewards, tiers, lcfg.get("tau", 1.0), lcfg.get("margin_alpha", 0.0), pair_mask)
    valid = n > 0
    parts = {"bt": bt}
    total = lcfg.get("w_bt", 1.0) * bt
    if lcfg.get("w_listwise", 0.0):
        parts["listwise"] = listwise_loss(rewards, tiers)
        total = total + lcfg["w_listwise"] * parts["listwise"]
    if lcfg.get("w_plackett_luce", 0.0):
        parts["plackett_luce"] = plackett_luce_loss(rewards, tiers)
        total = total + lcfg["w_plackett_luce"] * parts["plackett_luce"]
    if lcfg.get("w_center", 0.0):
        parts["center"] = center_loss(rewards, tiers)
        total = total + lcfg["w_center"] * parts["center"]
    if lcfg.get("w_perm", 0.0) and rewards_perm is not None:
        parts["perm"] = perm_consistency_loss(rewards, rewards_perm, tiers >= 0)
        total = total + lcfg["w_perm"] * parts["perm"]
    return LossOut(total, valid, n, parts)


def reduce_loss(lo: LossOut, denom: float | None = None) -> torch.Tensor:
    """Sum of per-example losses over valid examples / denom (default: number of valid examples)."""
    s = (lo.per_example * lo.valid).sum()
    return s / (denom if denom is not None else lo.valid.sum().clamp(min=1))
