import math

import torch

from scrm.losses import (bt_loss, center_loss, compute_loss, listwise_loss, plackett_luce_loss, tier_pair_mask)


def test_tier_pair_mask():
    tiers = torch.tensor([[0, 1, 1, 2, -1]])
    M = tier_pair_mask(tiers)[0]
    exp = torch.zeros(5, 5, dtype=torch.bool)
    for i, ti in enumerate([0, 1, 1, 2]):
        for j, tj in enumerate([0, 1, 1, 2]):
            exp[i, j] = ti < tj
    assert torch.equal(M[:4, :4], exp[:4, :4])
    assert not M[:, 4].any() and not M[4, :].any()
    assert not M[1, 2] and not M[2, 1]           # same tier never a pair
    assert not M.diagonal().any()


def test_bt_matches_hand_computation_and_per_example_mean():
    r = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.0]])
    tiers = torch.tensor([[0, 1, 2], [0, 1, -1]])
    loss, n = bt_loss(r, tiers)
    sp = lambda x: math.log(1 + math.exp(x))
    e0 = (sp(-(2 - 0.5)) + sp(-(2 + 1)) + sp(-(0.5 + 1))) / 3
    e1 = sp(-(0 - 1))
    assert n.tolist() == [3, 1]
    assert abs(loss[0] - e0) < 1e-6 and abs(loss[1] - e1) < 1e-6
    # batch mean is mean over examples, not over pairs
    lo = compute_loss(r, tiers, {})
    assert abs(lo.per_example.mean() - (e0 + e1) / 2) < 1e-6


def test_margin_tau():
    r = torch.tensor([[1.0, 0.0, -2.0]])
    tiers = torch.tensor([[0, 1, 2]])
    l, _ = bt_loss(r, tiers, tau=2.0, margin_alpha=0.5)
    sp = lambda x: math.log(1 + math.exp(x))
    e = (sp(-(1 - 0.5) / 2) + sp(-(3 - 1.0) / 2) + sp(-(2 - 0.5) / 2)) / 3
    assert abs(l[0] - e) < 1e-6


def test_no_pair_example_invalid():
    r = torch.zeros(1, 3)
    lo = compute_loss(r, torch.tensor([[0, 0, 0]]), {})
    assert not lo.valid[0] and lo.per_example[0] == 0


def test_optional_losses_finite_and_grad():
    r = torch.randn(3, 5, requires_grad=True)
    tiers = torch.tensor([[0, 1, 1, 2, -1], [0, 0, 1, 1, 1], [1, 0, 2, -1, -1]])
    lcfg = {"w_listwise": 1, "w_plackett_luce": 1, "w_center": 0.1}
    lo = compute_loss(r, tiers, lcfg)
    assert set(lo.parts) == {"bt", "listwise", "plackett_luce", "center"}
    for v in lo.parts.values():
        assert torch.isfinite(v).all()
    lo.per_example.sum().backward()
    assert torch.isfinite(r.grad).all()
    # listwise hand check on example 1: tier0 = {0,1}
    x = r.detach()[1]
    exp = -(torch.logsumexp(x[:2], 0) - torch.logsumexp(x, 0))
    assert abs(listwise_loss(r.detach(), tiers)[1] - exp) < 1e-5
    assert abs(center_loss(r.detach(), tiers)[0] - r.detach()[0, :4].mean() ** 2) < 1e-6
