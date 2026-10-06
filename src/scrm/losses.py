"""Tier-based ranking losses. All terms are returned PER EXAMPLE so the trainer can normalise over an
accumulation window (examples with no pair are masked out)."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
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


# --------------------------------------------------------------------------------------------------------------------
# Readout losses for frozen-feature experiments (scrm.feat_train / scrm.lossgrid). Each readout has its OWN learnable
# scale: softmax temperature T (ce, brier), BT temperature tau, sigmoid slope alpha + bias. All return PER EXAMPLE [B].
# --------------------------------------------------------------------------------------------------------------------

def _masked_softmax_inputs(rewards, valid, scale):
    """rewards / scale with padding at -inf (rows with no valid slot stay finite)."""
    return (rewards / scale).masked_fill(~(valid | ~valid.any(1, keepdim=True)), float("-inf"))


def target_distribution(tiers: torch.Tensor, probs: torch.Tensor | None = None,
                        has_probs: torch.Tensor | None = None) -> torch.Tensor:
    """Target over the set [B,N]: the given `probabilities` (renormalised over valid slots) where has_probs, else uniform
    over the best tier (tier-0 = lowest tier present). Padding (-1) gets 0."""
    valid = tiers >= 0
    tmin = tiers.masked_fill(~valid, 10**9).min(1, keepdim=True).values
    top = (valid & (tiers == tmin)).float()
    q = top / top.sum(1, keepdim=True).clamp(min=1)
    if probs is not None and has_probs is not None:
        p = probs.float().masked_fill(~valid, 0.0).clamp(min=0)
        p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
        q = torch.where(has_probs.unsqueeze(1) & (p.sum(1, keepdim=True) > 0), p, q)
    return q


def softmax_ce_loss(rewards, tiers, temperature=1.0, probs=None, has_probs=None):
    """Cross-entropy of softmax(s/T) against the target distribution (soft `probabilities` if present, else uniform
    over tier-0; with a single tier-0 item this equals listwise_loss at T=1)."""
    valid = tiers >= 0
    q = target_distribution(tiers, probs, has_probs)
    logp = torch.log_softmax(_masked_softmax_inputs(rewards, valid, temperature), dim=-1).masked_fill(~valid, 0.0)
    return -(q * logp).sum(1)


def brier_loss(rewards, tiers, temperature=1.0, probs=None, has_probs=None):
    """Brier score sum_i (softmax(s/T)_i - q_i)^2 against the same target distribution as softmax_ce_loss."""
    valid = tiers >= 0
    q = target_distribution(tiers, probs, has_probs)
    p = torch.softmax(_masked_softmax_inputs(rewards, valid, temperature), dim=-1).masked_fill(~valid, 0.0)
    return ((p - q) ** 2).sum(1)


def sigmoid_loss(rewards, abs_labels, abs_mask, alpha, bias):
    """SigLIP-style 'score each option': BCE(sigmoid(alpha * s + b), label) with soft labels in [0,1], averaged over the
    options where abs_mask is set. bias: scalar or [B] (per-family). Returns (per_example [B], n_labelled [B])."""
    bias = torch.as_tensor(bias, dtype=rewards.dtype, device=rewards.device)
    if bias.dim() == 1:
        bias = bias.unsqueeze(1)
    logits = alpha * rewards + bias
    l = F.binary_cross_entropy_with_logits(logits, abs_labels.float().clamp(0, 1).nan_to_num(0.0), reduction="none")
    m = abs_mask.float()
    n = m.sum(1)
    return (l * m).sum(1) / n.clamp(min=1), n


class Readouts(nn.Module):
    """Learnable scalars of the readouts, one set per loss so each is calibrated independently:
    softmax temperatures T_ce / T_brier, BT temperature tau_bt, sigmoid alpha (= exp(log_alpha), clamped to
    <= max_alpha like CLIP's logit_scale) and bias b. Bias init is SigLIP-like -log(n_neg / n_pos) with n_neg/n_pos
    ~ avg_set_size - 1 unless `bias_init` is given; `n_families > 1` gives one bias per family (index passed to
    `bias_for`)."""

    def __init__(self, avg_set_size: float = 4.0, n_families: int = 1, ce_temperature: float = 1.0,
                 brier_temperature: float = 1.0, bt_tau: float = 1.0, alpha_init: float = 1.0,
                 bias_init: float | None = None, max_alpha: float = 100.0, learnable: bool = True):
        super().__init__()
        self.max_log_alpha = math.log(max_alpha)
        self.log_T_ce = nn.Parameter(torch.tensor(math.log(ce_temperature)), requires_grad=learnable)
        self.log_T_brier = nn.Parameter(torch.tensor(math.log(brier_temperature)), requires_grad=learnable)
        self.log_tau_bt = nn.Parameter(torch.tensor(math.log(bt_tau)), requires_grad=learnable)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)), requires_grad=learnable)
        b0 = -math.log(max(avg_set_size - 1.0, 1.0)) if bias_init is None else float(bias_init)
        self.bias = nn.Parameter(torch.full((max(1, n_families),), b0), requires_grad=learnable)

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(max=self.max_log_alpha).exp()

    def T_ce(self): return self.log_T_ce.exp()
    def T_brier(self): return self.log_T_brier.exp()
    def tau_bt(self): return self.log_tau_bt.exp()

    def bias_for(self, fam_idx: torch.Tensor | None = None) -> torch.Tensor:
        """[B] bias per set (family index), or the scalar bias when there is a single one."""
        if self.bias.numel() == 1 or fam_idx is None:
            return self.bias[0]
        return self.bias[fam_idx]

    @torch.no_grad()
    def values(self) -> dict:
        return {"T_ce": float(self.T_ce()), "T_brier": float(self.T_brier()), "tau_bt": float(self.tau_bt()),
                "alpha": float(self.alpha()), "bias": [float(b) for b in self.bias]}
