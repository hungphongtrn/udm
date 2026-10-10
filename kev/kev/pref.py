"""Preference-first reward objectives (hypothesis branch hyp/preference-reward-model; docs/hypotheses/preference-reward-model.md).

Each option's head output is an unbounded scalar reward r_i (ScalarHead), or the pointer logit (PointerHead) for arm B.
kev.train --loss pref replaces question_loss's cross-entropy with a Bradley-Terry loss over pairs derived from the label:

  hard label y:     y > j for every j != y, P(y > j) = 1   ->  softplus(r_j - r_y)
  soft target t:    every pair i < j with t_i + t_j > 0,    ->  BCE(sigmoid(r_i - r_j), t_i / (t_i + t_j))
                    P(i > j) = t_i / (t_i + t_j)                (a uniform target pulls the rewards together)

averaged over the question's pairs, so a question counts once however many options it has (the CE path also averages
per question). On top of it, per micro-batch:
  --reg bsr   lambda * mean(cat(r_chosen, r_rejected))^2   (batch-wise sum-to-zero; the mean is over pair members,
                                                             so a chosen option counts once per pair it is in)
  --reg l2    lambda * mean(cat(r_chosen, r_rejected)^2)   (same pair members, squared)
  --bce_w w   w * mean_i BCE(sigmoid(r_i), [i == y])       (Stage 2 absolute supervision, hard-label questions only)
"""
import torch
import torch.nn.functional as F

REGS = ("none", "bsr", "l2")


def preference_pairs(q, K):
    """-> (i, j, p) lists: option i beats option j with probability p, from q's soft target when it has one, else its label."""
    if q.get("target") is not None:
        t = q["target"]
        pairs = [(i, j, t[i] / (t[i] + t[j])) for i in range(K) for j in range(i + 1, K) if t[i] + t[j] > 0]
    else:
        y = q["label"]
        pairs = [(y, j, 1.0) for j in range(K) if j != y]
    return [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]


def preference_loss(z, q, dev):
    """-> (mean pair loss of one question, rewards of the pairs' first members, of their second members). A question
    without pairs (one option) returns None."""
    i, j, p = preference_pairs(q, len(z))
    if not i: return None
    i, j = torch.tensor(i, device=dev), torch.tensor(j, device=dev)
    rw, rl = z[i], z[j]
    loss = F.binary_cross_entropy_with_logits(rw - rl, torch.tensor(p, device=dev, dtype=z.dtype))   # p = 1: softplus(rl - rw)
    return loss, rw, rl


def bce_loss(z, q, dev, types):
    """Per-option binary cross-entropy on sigmoid(r_i) against [i == label], mean over options; None for soft-target questions
    and question types not in `types` (no meaningful absolute label)."""
    if q.get("target") is not None or q["qtype"] not in types: return None
    y = F.one_hot(torch.tensor(q["label"], device=dev), len(z)).to(z.dtype)
    return F.binary_cross_entropy_with_logits(z, y)


def reward_reg(kind, rw, rl):
    """The --reg term over one micro-batch's pair members (cat of every question's rw and rl)."""
    r = torch.cat([rw, rl])
    if kind == "bsr": return r.mean().square()
    if kind == "l2": return r.square().mean()
    raise ValueError(f"unknown reg {kind!r}")
