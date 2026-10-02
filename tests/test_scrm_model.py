import pytest
import torch

from scrm.model import SetEncoder, build_scrm

TINY = {"name_or_path": "tiny", "dtype": "float32", "d_set": 32, "set_heads": 4,
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0}}


def test_set_encoder_permutation_equivariant_and_masking():
    torch.manual_seed(0)
    enc = SetEncoder(16, 32, 2, 4, 2, 0.1).eval()
    E = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    r = enc(E, mask)
    perm = torch.tensor([3, 1, 5, 0, 2, 4])
    r_p = enc(E[:, perm], mask[:, perm])
    assert torch.allclose(r[:, perm], r_p, atol=1e-5)
    assert (r[1, 3:] == 0).all()
    # padded content must not influence real candidates
    E2 = E.clone(); E2[1, 3:] = 100 * torch.randn(3, 16)
    assert torch.allclose(enc(E2, mask)[1, :3], r[1, :3], atol=1e-5)


def test_fully_padded_row_is_safe():
    enc = SetEncoder(8, 16, 2, 2, 2, 0.0)
    r = enc(torch.randn(2, 3, 8), torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool))
    assert torch.isfinite(r).all() and (r[1] == 0).all()


def test_no_set_layers_ablation_is_per_candidate():
    enc = SetEncoder(8, 16, 0, 2, 2, 0.0).eval()
    E = torch.randn(1, 4, 8)
    m = torch.ones(1, 4, dtype=torch.bool)
    assert torch.allclose(enc(E, m)[:, :2], enc(E[:, :2], m[:, :2]), atol=1e-6)


def _sets(n_list, P=11, seed=0):
    """Packed batch (collator layout): set b has n_list[b] option sequences of length P + 3..7."""
    g = torch.Generator().manual_seed(seed)
    lens, rs, sl = [], [], []
    for b, n in enumerate(n_list):
        for k in range(n):
            lens.append(P + 3 + (k * 2) % 5); rs.append(b); sl.append(k)
    L = torch.tensor(lens)
    pack = {"input_ids": torch.randint(0, 300, (int(L.sum()),), generator=g),
            "position_ids": torch.cat([torch.arange(n) for n in lens]), "seq_lens": lens,
            "read_idx": L.cumsum(0) - 1, "row_set": torch.tensor(rs), "row_slot": torch.tensor(sl)}
    cm = torch.zeros(len(n_list), max(n_list), dtype=torch.bool)
    for b, n in enumerate(n_list):
        cm[b, :n] = True
    return pack, cm


def test_full_model_forward_no_new_tokens():
    from scrm.tiny import make_tiny_tokenizer
    model, tok = build_scrm(TINY, "cpu")
    assert len(tok) == len(make_tiny_tokenizer())                        # no tokens added
    pack, cm = _sets([2, 1])
    r = model(pack, cm)
    assert r.shape == (2, 2) and r[1, 1] == 0 and r.dtype == torch.float32
    # trainable: lora + head, but not base weights
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in names) and any(n.startswith("set_encoder.") for n in names)
    assert not any("embed_tokens" in n for n in names)
    r.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)


def _fwd_bwd(model, pack, cm, max_tokens):
    model.zero_grad()
    r = model(pack, cm, max_tokens=max_tokens)
    r.square().sum().backward()
    return r.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same(a, b, rtol=2e-2):
    (r1, g1), (r2, g2) = a, b
    assert torch.allclose(r1, r2, atol=1e-4), (r1 - r2).abs().max()
    assert g1.keys() == g2.keys() and any("lora_" in n for n in g1)
    for n in g1:   # delta-rule gate params accumulate a little fp32 noise; everything else is ~exact
        assert (g1[n] - g2[n]).abs().max() <= rtol * g1[n].abs().max() + 1e-6, n


def test_chunked_encoding_matches_single_chunk_forward_and_grad():
    """Splitting the packed sequences into token-budgeted chunks (gradient checkpointing on) changes nothing."""
    model, _ = build_scrm(dict(TINY, set_dropout=0.0), "cpu")
    model.train()
    pack, cm = _sets([4, 2], P=37)
    _assert_same(_fwd_bwd(model, pack, cm, None), _fwd_bwd(model, pack, cm, 90))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen packing kernels are CUDA-only")
def test_varlen_packing_matches_padded_forward_and_grad(monkeypatch):
    """Padding-free packed row (cu_seqlens / seq_idx) == right-padded batch: no attention or Gated DeltaNet state
    leaks across sequence boundaries."""
    import scrm.model as M
    if not M._varlen_kernels(torch.device("cuda")):
        pytest.skip("fla / causal-conv1d not installed")
    model, _ = build_scrm(dict(TINY, set_dropout=0.0), "cuda")
    model.train()
    pack, cm = _sets([5, 3, 1], P=70)
    pack = {k: v.cuda() if torch.is_tensor(v) else v for k, v in pack.items()}
    cm = cm.cuda()
    ref = _fwd_bwd(model, pack, cm, None)
    monkeypatch.setattr(M, "_varlen_kernels", lambda d: False)
    pad = _fwd_bwd(model, pack, cm, None)
    _assert_same(pad, ref)


def test_frozen_backbone_has_no_backbone_grads():
    cfg = dict(TINY, freeze_backbone=True)
    model, tok = build_scrm(cfg, "cpu")
    assert not any(p.requires_grad for p in model.backbone.parameters())
    pack, cm = _sets([2])
    r = model(pack, cm)
    r.sum().backward()
    assert model.set_encoder.proj.weight.grad is not None
