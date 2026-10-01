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


def test_full_model_forward_no_new_tokens():
    from scrm.tiny import make_tiny_tokenizer
    model, tok = build_scrm(TINY, "cpu")
    assert len(tok) == len(make_tiny_tokenizer())                        # no tokens added
    B, L = 2, 12
    x = torch.randint(0, 300, (B, L))
    pos = torch.tensor([[3, 7], [3, 7]])
    cm = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    r = model(x, torch.ones(B, L, dtype=torch.long), pos, cm)
    assert r.shape == (2, 2) and r[1, 1] == 0 and r.dtype == torch.float32
    # trainable: lora + head, but not base weights
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in names) and any(n.startswith("set_encoder.") for n in names)
    assert not any("embed_tokens" in n for n in names)
    r.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)


def test_frozen_backbone_has_no_backbone_grads():
    cfg = dict(TINY, freeze_backbone=True)
    model, tok = build_scrm(cfg, "cpu")
    assert not any(p.requires_grad for p in model.backbone.parameters())
    x = torch.randint(0, 300, (1, 6))
    r = model(x, torch.ones(1, 6, dtype=torch.long), torch.tensor([[2, 5]]), torch.ones(1, 2, dtype=torch.bool))
    r.sum().backward()
    assert model.set_encoder.proj.weight.grad is not None
