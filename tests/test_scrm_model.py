import numpy as np
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


def _items(n_list, P=11, seed=0, one_token=False):
    """Decision sets (render.Item): set b has n_list[b] graded options whose suffixes are 3..7 tokens after a P-token
    shared prefix (`one_token`: set b's option 0 is a single-token suffix)."""
    from scrm.render import Item
    g = np.random.default_rng(seed)
    items = []
    for b, n in enumerate(n_list):
        pre = g.integers(0, 300, P).astype(np.int64)
        sfx = [g.integers(0, 300, 1 if (one_token and k == 0) else 3 + (k * 2) % 5).astype(np.int64) for k in range(n)]
        items.append(Item(prefix=pre, suffixes=sfx, tiers=np.arange(n, dtype=np.int64), order=np.arange(n),
                          choice_ids=[f"c{k}" for k in range(n)], family="f", source_id="s",
                          decision_set_id=f"d{b}"))
    return items


def _sets(n_list, P=11, seed=0, one_token=False):
    """Packed batch (collator branch layout): set b has n_list[b] graded options of length 3..7 after a P-token
    prefix; the prefix is encoded once and shared by the set's option branches."""
    from scrm.collator import collate
    b = collate(_items(n_list, P=P, seed=seed, one_token=one_token))
    return b["pack"], b["candidate_mask"]


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


def _oracle_embeds(model, items):
    """The old per-option computation: every full sequence `prefix + suffix_k` encoded on its own (right-padded batch
    through the backbone), read out at its last token. Ground truth for the branch encoder."""
    seqs = [x for it in items for x in it.seqs]
    S = max(len(x) for x in seqs)
    x = torch.zeros(len(seqs), S, dtype=torch.long, device=next(model.set_encoder.parameters()).device)
    am = torch.zeros(len(seqs), S, dtype=torch.long, device=x.device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.as_tensor(s, device=x.device)
        am[i, :len(s)] = 1
    h = model.backbone(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state
    return h[torch.arange(len(seqs), device=x.device), torch.tensor([len(s) for s in seqs], device=x.device) - 1]


def _oracle_fwd_bwd(model, items, pack, cm):
    """Rewards + grads from the full-sequence oracle embeddings (the set head is shared, so this isolates the
    backbone)."""
    model.zero_grad()
    r = model.head(_oracle_embeds(model, items), pack, cm)
    r.square().sum().backward()
    return r.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same(a, b, rtol=2e-2, atol=1e-4, label=""):
    (r1, g1), (r2, g2) = a, b
    assert torch.allclose(r1, r2, atol=atol, rtol=rtol), f"{label} rewards: max err {(r1 - r2).abs().max():.3e}"
    assert g1.keys() == g2.keys() and any("lora_" in n for n in g1)
    stats = []   # (max err / max ref, name, norm-relative err, cosine)
    for n in g1:
        x, y = g1[n].float().flatten(), g2[n].float().flatten()
        stats.append((((x - y).abs().max() / x.abs().max().clamp(min=1e-12)).item(), n,
                      ((x - y).norm() / x.norm().clamp(min=1e-12)).item(),
                      torch.nn.functional.cosine_similarity(x, y, dim=0).item()))
    bad = [n for n in g1 if (g1[n] - g2[n]).abs().max() > rtol * g1[n].abs().max() + 1e-6]
    worst = "\n".join(f"  {m:.3e} max-rel  {r:.3e} norm-rel  cos {c:.6f}  {n}" for m, n, r, c in sorted(stats)[::-1][:8])
    # delta-rule gate params accumulate a little fp32 noise; everything else is ~exact
    assert not bad, f"{label}: {len(bad)}/{len(g1)} grads differ beyond rtol={rtol}; worst:\n{worst}"


def test_chunked_encoding_matches_single_chunk_forward_and_grad():
    """Splitting the packed sequences into token-budgeted chunks (gradient checkpointing on) changes nothing."""
    model, _ = build_scrm(dict(TINY, set_dropout=0.0), "cpu")
    model.train()
    pack, cm = _sets([4, 2], P=37)
    _assert_same(_fwd_bwd(model, pack, cm, None), _fwd_bwd(model, pack, cm, 90))


def test_frozen_backbone_has_no_backbone_grads():
    cfg = dict(TINY, freeze_backbone=True)
    model, tok = build_scrm(cfg, "cpu")
    assert not any(p.requires_grad for p in model.backbone.parameters())
    pack, cm = _sets([2])
    r = model(pack, cm)
    r.sum().backward()
    assert model.set_encoder.proj.weight.grad is not None


def test_head_defaults_to_set_encoder():
    """`model.head` defaults to "set", so v1/v2 configs and their `scrm_head.pt` (set_encoder.* keys) keep loading."""
    model, _ = build_scrm(TINY, "cpu")
    assert model.head_kind == "set" and model.head_module() is model.set_encoder
    assert all(k.startswith("set_encoder.") for k in model.head_state_dict())


def test_linear_head_reward_is_per_option():
    """Peek (`head: linear`): an option's reward depends only on its own branch of the shared prompt, so grading it
    alone or together with the prompt's other options gives the same reward (the set encoder would mix them)."""
    from scrm.collator import collate
    from scrm.render import Item
    model, _ = build_scrm(dict(TINY, head="linear"), "cpu")
    model.eval()
    item = _items([4, 3])[0]                             # the set 0 that _sets([4, 3]) builds below
    solo = Item(prefix=item.prefix, suffixes=item.suffixes[:1], tiers=item.tiers[:1], order=item.order[:1],
                choice_ids=item.choice_ids[:1], family=item.family, source_id=item.source_id,
                decision_set_id=item.decision_set_id)
    b_one = collate([solo])
    pack, cm = _sets([4, 3])
    with torch.no_grad():
        r_all = model(pack, cm)                          # option 0 graded with its siblings and a second set
        r_one = model(b_one["pack"], b_one["candidate_mask"])   # the same option graded alone
    assert r_all.dtype == torch.float32 and (r_all[1, 3:] == 0).all()   # ungraded slots stay 0
    assert torch.allclose(r_one[0, 0], r_all[0, 0], atol=1e-5)


def test_linear_head_save_load_roundtrip(tmp_path):
    """The head type is recorded in scrm_config.json and the weights are keyed by it: a linear-head checkpoint
    rebuilds a linear head and reproduces its rewards."""
    from scrm.model import load_scrm
    model, _ = build_scrm(dict(TINY, head="linear"), "cpu")
    model.eval()
    pack, cm = _sets([3, 2])
    with torch.no_grad():
        r = model(pack, cm)
    model.save_pretrained(str(tmp_path))
    sd = torch.load(tmp_path / "scrm_head.pt", map_location="cpu")
    assert sorted(sd) == ["reward_head.bias", "reward_head.weight"]
    m2 = load_scrm(str(tmp_path), device="cpu")
    assert m2.head_kind == "linear"
    with torch.no_grad():
        assert torch.allclose(m2(pack, cm), r, atol=1e-6)

