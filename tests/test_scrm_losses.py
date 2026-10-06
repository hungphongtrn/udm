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


# ---- readout losses (brier / softmax CE / sigmoid) ----

from scrm.losses import (Readouts, brier_loss, sigmoid_loss, softmax_ce_loss, target_distribution)  # noqa: E402


def test_brier_known_values_and_probs_target():
    r = torch.zeros(2, 3)                         # softmax = uniform 1/3
    tiers = torch.tensor([[0, 1, 2], [0, 0, -1]])
    b = brier_loss(r, tiers)
    assert abs(b[0] - ((2 / 3) ** 2 + 2 * (1 / 3) ** 2)) < 1e-6
    assert abs(b[1] - 0.0) < 1e-6 or abs(b[1] - 2 * 0.0) < 1e-6 + 1e-6   # uniform over 2 valid = target
    probs = torch.tensor([[0.2, 0.5, 0.3], [0.0, 0.0, 0.0]])
    has = torch.tensor([True, False])
    q = target_distribution(tiers, probs, has)
    assert torch.allclose(q[0], probs[0]) and torch.allclose(q[1], torch.tensor([0.5, 0.5, 0.0]))
    assert abs(brier_loss(r, tiers, probs=probs, has_probs=has)[0] - ((1 / 3 - 0.2) ** 2 + (1 / 3 - 0.5) ** 2 + (1 / 3 - 0.3) ** 2)) < 1e-6
    # temperature: higher T flattens softmax
    r2 = torch.tensor([[3.0, 0.0, 0.0]])
    t = torch.tensor([[0, 1, 1]])
    assert brier_loss(r2, t, 1.0)[0] < brier_loss(r2, t, 10.0)[0]


def test_softmax_ce_matches_listwise_and_soft_target():
    r = torch.randn(3, 5)
    tiers = torch.tensor([[0, 1, 1, 2, -1], [0, 0, 1, 1, 1], [1, 0, 2, -1, -1]])
    # single tier-0 item (examples 0, 2): identical to listwise; with ties CE targets uniform over tier-0 instead
    assert torch.allclose(softmax_ce_loss(r, tiers)[[0, 2]], listwise_loss(r, tiers)[[0, 2]], atol=1e-5)
    probs = torch.tensor([[0.1, 0.6, 0.3, 0.0, 0.0]] * 3)
    ce = softmax_ce_loss(r, tiers, 2.0, probs, torch.tensor([True, False, False]))
    exp = -(probs[0, :4] * torch.log_softmax(r[0, :4] / 2.0, 0)).sum()
    assert abs(ce[0] - exp) < 1e-5
    assert abs(ce[1] - softmax_ce_loss(r, tiers, 2.0)[1]) < 1e-6


def test_sigmoid_masking_soft_labels_and_grad():
    r = torch.tensor([[1.0, -1.0, 5.0]], requires_grad=True)
    lab = torch.tensor([[1.0, 0.25, 0.0]])
    mask = torch.tensor([[True, True, False]])
    l, n = sigmoid_loss(r, lab, mask, alpha=2.0, bias=-0.5)
    sp = lambda x: math.log(1 + math.exp(x))
    bce = lambda z, y: sp(z) - y * z
    exp = (bce(2 * 1 - 0.5, 1.0) + bce(2 * -1 - 0.5, 0.25)) / 2
    assert abs(l[0] - exp) < 1e-6 and n[0] == 2
    l.sum().backward()
    assert r.grad[0, 2] == 0 and r.grad[0, 0] != 0
    # fully unlabelled set -> zero loss, n = 0; per-set bias vector
    l2, n2 = sigmoid_loss(r.detach(), lab, torch.zeros(1, 3, dtype=torch.bool), 1.0, torch.tensor([0.3]))
    assert l2[0] == 0 and n2[0] == 0


def test_shift_invariance_ce_bt_but_not_sigmoid():
    r = torch.randn(2, 4)
    tiers = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 2]])
    lab = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    m = torch.ones(2, 4, dtype=torch.bool)
    for f in (lambda x: softmax_ce_loss(x, tiers, 1.5), lambda x: brier_loss(x, tiers, 1.5), lambda x: bt_loss(x, tiers)[0]):
        assert torch.allclose(f(r), f(r + 3.0), atol=1e-5)
    assert not torch.allclose(sigmoid_loss(r, lab, m, 1.0, 0.0)[0], sigmoid_loss(r + 3.0, lab, m, 1.0, 0.0)[0], atol=1e-3)


def test_readouts_alpha_clamp_bias_init_and_family_bias():
    ro = Readouts(avg_set_size=5.0, n_families=3, alpha_init=1.0, max_alpha=100.0)
    assert torch.allclose(ro.bias, torch.full((3,), -math.log(4.0)))
    with torch.no_grad():
        ro.log_alpha.fill_(100.0)                  # way past the cap
    assert abs(float(ro.alpha().detach()) - 100.0) < 1e-3
    ro.log_alpha.grad = None
    ro.alpha().backward()
    assert ro.log_alpha.grad is None or float(ro.log_alpha.grad) == 0.0   # clamped: no gradient past the cap
    with torch.no_grad():
        ro.bias[:] = torch.tensor([0.0, 1.0, 2.0])
    assert ro.bias_for(torch.tensor([2, 0])).tolist() == [2.0, 0.0]
    assert Readouts(avg_set_size=1.0).bias.item() == 0.0       # -log(max(n-1, 1))
    # separate scales
    assert {"T_ce", "T_brier", "tau_bt", "alpha", "bias"} <= set(Readouts().values())
    # BT with its own tau reuses bt_loss
    r, tiers = torch.tensor([[1.0, 0.0]]), torch.tensor([[0, 1]])
    assert torch.allclose(bt_loss(r, tiers, tau=Readouts(bt_tau=2.0).tau_bt())[0], bt_loss(r, tiers, tau=2.0)[0])
