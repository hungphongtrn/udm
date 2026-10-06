"""Metrics for frozen-feature readout experiments: ranking (reusing metrics.example_metrics), softmax and sigmoid
calibration, abstention test and presentation-order stability, overall and per source_id / label_kind."""
from __future__ import annotations

import numpy as np
import torch

from .losses import target_distribution
from .metrics import ece, example_metrics

ECE_BINS = 15
RANK_KEYS = ("top1", "pair_acc", "mrr", "ndcg", "kendall_tau")


def _softmax_np(s: np.ndarray, valid: np.ndarray, T: float) -> np.ndarray:
    x = np.where(valid, s / T, -np.inf)
    x = x - np.where(valid.any(1, keepdims=True), x.max(1, keepdims=True), 0.0)
    e = np.where(valid, np.exp(x), 0.0)
    return e / np.clip(e.sum(1, keepdims=True), 1e-12, None)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mean(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")


def compute_metrics(scores, tiers, probs, has_probs, abs_labels, abs_mask, sources, kinds, *, temperature: float = 1.0,
                    alpha: float | None = None, bias=None, after_scores=None, after_mask=None, alt_scores=None) -> dict:
    """scores/tiers/probs/abs_labels/abs_mask [S,N] (padding: tiers = -1), has_probs [S] bool, sources/kinds length S.
    temperature: softmax temperature of the reported softmax readout. alpha/bias ([S] or scalar): sigmoid readout
    (None = not trained -> sigmoid metrics NaN). after_scores/after_mask [S,N]: the same sets re-scored with their single
    tier-0 option removed (abstention; used for sets with exactly one tier-0 option and >= 2 options).
    alt_scores: list of (set_idx [k], scores [k,N]) from other presentation variants (order stability).
    Returns {"all": {...}, "source/<id>": {...}, "label_kind/<k>": {...}}."""
    scores = np.asarray(scores, dtype=np.float64)
    tiers_t = torch.as_tensor(tiers)
    tiers = tiers_t.numpy()
    valid = tiers >= 0
    S = len(scores)
    em = example_metrics(torch.as_tensor(scores, dtype=torch.float32), tiers_t)
    p = _softmax_np(scores, valid, temperature)
    q = target_distribution(tiers_t, torch.as_tensor(np.nan_to_num(np.asarray(probs, dtype=np.float32))),
                            torch.as_tensor(has_probs)).numpy()
    conf = p.max(1)
    hit = em["top1"]
    brier_sm = ((p - q) ** 2).sum(1)
    nll_sm = -(q * np.log(np.clip(p, 1e-12, None))).sum(1)
    abs_labels = np.nan_to_num(np.asarray(abs_labels, dtype=np.float64))
    abs_mask = np.asarray(abs_mask, dtype=bool) & valid
    has_sig = alpha is not None
    if has_sig:
        b = np.broadcast_to(np.asarray(bias, dtype=np.float64).reshape(-1, 1) if np.ndim(bias) else float(bias), scores.shape)
        sig = _sigmoid(alpha * scores + b)
        sig_max = np.where(valid, sig, -1).max(1)
    # abstention
    elig = np.zeros(S, bool)
    if after_scores is not None:
        tmin = np.where(valid, tiers, 10**9).min(1, keepdims=True)
        elig = ((valid & (tiers == tmin)).sum(1) == 1) & (valid.sum(1) >= 2)
        am = np.asarray(after_mask, dtype=bool)
        sm_after = _softmax_np(np.asarray(after_scores, dtype=np.float64), am, temperature).max(1)
        if has_sig:
            sig_after = np.where(am, _sigmoid(alpha * np.asarray(after_scores) + b), -1).max(1)
    # order stability
    flip = np.full(S, np.nan)
    dp = np.full(S, np.nan)
    if alt_scores:
        fl, dd, cnt = np.zeros(S), np.zeros(S), np.zeros(S)
        for idx, alt in alt_scores:
            idx = np.asarray(idx)
            if len(idx) == 0:
                continue
            pa = _softmax_np(np.asarray(alt, dtype=np.float64), valid[idx], temperature)
            fl[idx] += (np.where(valid[idx], pa, -1).argmax(1) != np.where(valid[idx], p[idx], -1).argmax(1))
            dd[idx] += (np.abs(pa - p[idx]) * valid[idx]).sum(1) / valid[idx].sum(1).clip(min=1)
            cnt[idx] += 1
        flip = np.where(cnt > 0, fl / np.maximum(cnt, 1), np.nan)
        dp = np.where(cnt > 0, dd / np.maximum(cnt, 1), np.nan)

    def group(ix: np.ndarray) -> dict:
        d = {"n": int(len(ix))}
        for k in RANK_KEYS:
            d[k] = _mean(em[k][ix])
        d["ece_top1"] = ece(conf[ix], hit[ix], ECE_BINS) if len(ix) else float("nan")
        d["conf"] = _mean(conf[ix])
        d["brier_softmax"] = _mean(brier_sm[ix])
        d["nll_softmax"] = _mean(nll_sm[ix])
        d["max_softmax"] = _mean(conf[ix])
        if has_sig:
            m = abs_mask[ix]
            pr, lb = sig[ix][m], abs_labels[ix][m]
            d["sig_n"] = int(m.sum())
            d["sig_ece"] = ece(pr, lb, ECE_BINS) if len(pr) else float("nan")
            d["sig_brier"] = _mean((pr - lb) ** 2)
            d["max_sigmoid"] = _mean(sig_max[ix])
        else:
            d.update(sig_n=0, sig_ece=float("nan"), sig_brier=float("nan"), max_sigmoid=float("nan"))
        e = ix[elig[ix]]
        d["abst_n"] = int(len(e))
        if len(e) and after_scores is not None:
            d["abst_softmax_before"], d["abst_softmax_after"] = _mean(conf[e]), _mean(sm_after[e])
            d["abst_softmax_drop"] = d["abst_softmax_before"] - d["abst_softmax_after"]
            if has_sig:
                d["abst_sigmoid_before"], d["abst_sigmoid_after"] = _mean(sig_max[e]), _mean(sig_after[e])
                d["abst_sigmoid_drop"] = d["abst_sigmoid_before"] - d["abst_sigmoid_after"]
        for k in ("abst_softmax_before", "abst_softmax_after", "abst_softmax_drop", "abst_sigmoid_before",
                  "abst_sigmoid_after", "abst_sigmoid_drop"):
            d.setdefault(k, float("nan"))
        d["flip_rate"] = _mean(flip[ix])
        d["mean_abs_dp"] = _mean(dp[ix])
        return d

    sources, kinds = np.asarray(sources, dtype=object), np.asarray(kinds, dtype=object)
    out = {"all": group(np.arange(S))}
    for s in sorted(set(sources.tolist())):
        out[f"source/{s}"] = group(np.where(sources == s)[0])
    for k in sorted(set(kinds.tolist())):
        out[f"label_kind/{k}"] = group(np.where(kinds == k)[0])
    return out
