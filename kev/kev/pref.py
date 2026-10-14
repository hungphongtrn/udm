"""Preference-first reward objectives (hypothesis branch hyp/preference-reward-model; docs/hypotheses/done/preference-reward-model.md).

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

kev.train --loss bce replaces both with balanced_bce_loss: a class-balanced per-option BCE (positives and negatives each
half of a question's weight, soft targets as partial credit, none-of-the-above rows left out), optionally plus
--ce_w x Kev's softmax cross-entropy over the options (the listwise term; the BCE stays the anchor of the scale).

kev.train --loss rcr replaces them with rcr_loss, Regression Compatible Ranking (Bai et al., CIKM 2023, arXiv 2211.01494,
Definition 2): (1 - alpha) x per-option sigmoid CE + alpha x ListCE(sigmoid), the listwise term a softmax CE over the
virtual logits log sigmoid(r_i). Both terms reach their minimum at sigmoid(r_i) = P(option i is right), so the listwise
term ranks without pulling the level away from the BCE's (the conflict SigmoidCE + SoftmaxCE has).
"""
import torch
import torch.nn.functional as F
from .data import NONE_TEXTS

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


def balanced_bce_loss(z, q, dev):
    """--loss bce, one question: -0.5 * sum_i t_i log s_i / sum_i t_i - 0.5 * sum_i (1 - t_i) log(1 - s_i) / sum_i (1 - t_i),
    s_i = sigmoid(r_i), t_i = [i == label] or the question's soft target (partial credit; a uniform target asks for s = 0.5).
    Positives and negatives each carry half the question's weight, so the 1/k prior of a k-option question (which the task
    reveals even when the row hides k) is no shortcut: r_i estimates the log likelihood ratio of "right" vs "wrong", and a
    deployment prior pi is the offset logit(pi). A half with no weight (every kept option wrong) contributes 0. Options
    whose text is a none-of-the-above wording (kev.data.NONE_TEXTS) are left out: graded alone, the same row is right or
    wrong depending on options it cannot see. None when every option is left out."""
    keep = [i for i, o in enumerate(q["options"]) if o not in NONE_TEXTS]
    if not keep: return None
    t = torch.tensor(q["target"], device=dev, dtype=z.dtype) if q.get("target") is not None else F.one_hot(torch.tensor(q["label"], device=dev), len(z)).to(z.dtype)
    idx = torch.tensor(keep, device=dev); z, t = z[idx], t[idx]
    loss = z.sum() * 0.0
    for w, y in ((t, 1.0), (1 - t, 0.0)):
        if w.sum() > 0:
            loss = loss + 0.5 * (w * F.binary_cross_entropy_with_logits(z, torch.full_like(z, y), reduction="none")).sum() / w.sum()
    return loss


RCR_POINTS = ("mean", "balanced")


def rcr_loss(z, q, dev, alpha, drop_none, point="mean", beta=0.0):
    """--loss rcr, one question (Bai et al. 2023, eq. 1, 12, 20) with y_i = [i == label] or the soft target:
        (1 - alpha) * Point  +  alpha * -(1 / sum_i y_i) * sum_i y_i * log(sigmoid(r_i) / sum_j sigmoid(r_j))
    Point (--rcr_point):
      mean      mean_i SigmoidCE(r_i, y_i). One deviation from the paper: the mean over the question's options, not the sum,
                so a question weighs once whatever its option count. Its minimum is sigmoid(r_i) = P(option i is right), so
                it learns the question's 1/K prior: a K = 77 question's rewards sit ~log(76) below a K = 2 question's.
      balanced  0.5 * sum_i y_i SigmoidCE(r_i, 1) / sum_i y_i  +  0.5 * sum_j w_j SigmoidCE(r_j, 0) / sum_j w_j, with
                w_j = (1 - y_j) exp(beta * r_j) (weights detached; beta = 0: every wrong option alike, beta > 0 weighs the
                hardest wrong options more). Positives and negatives carry half the question each whatever K is, so r = 0
                is one class-balanced boundary for every option count (alpha = 0, beta = 0 is balanced_bce_loss). The
                price: sigmoid(r_i) is then a likelihood ratio, not P(right), and the listwise term (whose minimum is
                sigmoid(r_i) proportional to P(right)) agrees with it only up to that reweighting.
    alpha = 0 is the pointwise term alone. drop_none (the pointwise formats) leaves none-of-the-above rows out, as
    balanced_bce_loss does: graded alone, such a row is right or wrong depending on options it cannot see. A question with
    no positive mass left (its answer was a none row) keeps only the pointwise term: every kept option is trained as
    wrong. -> (loss, pointwise term, listwise term or None); None when every option is left out."""
    if point not in RCR_POINTS: raise ValueError(f"unknown rcr point term {point!r}")
    keep = [i for i, o in enumerate(q["options"]) if o not in NONE_TEXTS] if drop_none else list(range(len(z)))
    if not keep: return None
    t = torch.tensor(q["target"], device=dev, dtype=z.dtype) if q.get("target") is not None else F.one_hot(torch.tensor(q["label"], device=dev), len(z)).to(z.dtype)
    idx = torch.tensor(keep, device=dev); z, t = z[idx], t[idx]
    if point == "mean":
        pt = F.binary_cross_entropy_with_logits(z, t)
    else:
        neg = (1 - t) * torch.exp(beta * (z.detach() - z.detach().max())) if beta else 1 - t   # max shift: same weights, no overflow
        pt = z.sum() * 0.0
        for w, y in ((t, 1.0), (neg, 0.0)):
            if w.sum() > 0:
                pt = pt + 0.5 * (w * F.binary_cross_entropy_with_logits(z, torch.full_like(z, y), reduction="none")).sum() / w.sum()
    if t.sum() <= 0: return (1 - alpha) * pt, pt, None
    ls = F.logsigmoid(z)
    listwise = -(t * (ls - torch.logsumexp(ls, 0))).sum() / t.sum()
    return (1 - alpha) * pt + alpha * listwise, pt, listwise


def reward_reg(kind, rw, rl):
    """The --reg term over one micro-batch's pair members (cat of every question's rw and rl)."""
    r = torch.cat([rw, rl])
    if kind == "bsr": return r.mean().square()
    if kind == "l2": return r.square().mean()
    raise ValueError(f"unknown reg {kind!r}")
