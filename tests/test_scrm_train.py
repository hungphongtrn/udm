import numpy as np
import pytest
import torch

from scrm.collator import collate, to_device
from scrm.config import load_config
from scrm.data import EvalSet
from scrm.evaluate import run_eval, run_perm_eval
from scrm.losses import compute_loss, reduce_loss
from scrm.metrics import example_metrics
from scrm.model import build_scrm, load_scrm
from scrm.render import Renderer
from scrm.synth import write_synth
from scrm.train import list_ckpts, train

TINY = {"name_or_path": "tiny", "dtype": "float32", "d_set": 32, "set_heads": 4,
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0}}
RENDER = {"max_len": 256, "cand_max_tokens": 16, "state_max_tokens": 96, "max_candidates": 16}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return write_synth(str(tmp_path_factory.mktemp("synth")))


def _items(synth, model, tok, n=8):
    cfg = load_config(None, [f"data.local_dir={synth}", "data.eval_max_rows=%d" % n, "data.eval_max_rows_per_source=%d" % n])
    r = Renderer(tok, RENDER)
    es = EvalSet.from_config(cfg, r)
    return cfg, r, es


def test_overfit_fixed_batch(synth):
    torch.manual_seed(0)
    model, tok = build_scrm(TINY, "cpu")
    cfg, r, es = _items(synth, model, tok)
    b = collate(es.items[:8], r.pad_id)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    model.train()
    losses = []
    for _ in range(25):
        rew = model(b["input_ids"], b["attention_mask"], b["candidate_positions"], b["candidate_mask"])
        lo = compute_loss(rew, b["tiers"], {}, b["pair_mask"])
        loss = reduce_loss(lo)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.15, losses
    m = example_metrics(rew.detach(), b["tiers"])
    assert m["pair_acc"].mean() > 0.6


def test_checkpoint_roundtrip_and_inference(synth, tmp_path):
    out = tmp_path / "run"
    cfg = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={out}", "train.max_steps=4",
                                                  "train.eval_every=2", "train.save_every=2", "train.keep_last=1"])
    res = train(cfg)
    assert res["step"] == 4
    assert (out / "best" / "scrm_head.pt").exists() and (out / "metrics.jsonl").exists()
    assert len(list_ckpts(str(out))) == 1          # keep_last
    ck = res["ckpt"]
    m1 = load_scrm(ck)
    m2 = load_scrm(ck)
    rank = m1.rank("Pick the best", {"k": "v"}, ["alpha", "beta", "gamma"])
    assert sorted(d["index"] for d in rank) == [0, 1, 2]
    assert rank[0]["reward"] >= rank[-1]["reward"]
    rank2 = m2.rank("Pick the best", {"k": "v"}, ["alpha", "beta", "gamma"])
    assert np.allclose([d["reward"] for d in rank], [d["reward"] for d in rank2])
    p = m1.pairwise_probability(1.0, 0.0)
    assert 0.5 < float(p) < 1
    # eval on the loaded model matches eval on a second load; permutation metrics are produced
    r = Renderer(m1.tokenizer, m1.render_cfg)
    es = EvalSet.from_config(load_config(None, [f"data.local_dir={synth}", "data.eval_max_rows=30"]), r)
    a = run_eval(m1, es.batches(), cfg["loss"], torch.device("cpu"))["all"]
    b = run_eval(m2, es.batches(), cfg["loss"], torch.device("cpu"))["all"]
    assert abs(a["pair_acc"] - b["pair_acc"]) < 1e-9
    pm = run_perm_eval(m1, es, r, 6, cfg["data"]["batch"], torch.device("cpu"))
    assert 0 <= pm["rank_agree"] <= 1
    # resume continues from the saved step
    cfg2 = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={out}", "train.max_steps=6",
                                                   "train.eval_every=0", "train.save_every=0"])
    res2 = train(cfg2, resume="auto")
    assert res2["step"] == 6


def test_resume_restores_weights(synth, tmp_path):
    out = tmp_path / "run"
    cfg = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={out}", "train.max_steps=3",
                                                  "train.eval_every=0", "train.save_every=0"])
    res = train(cfg)
    m = load_scrm(res["ckpt"])
    sd = torch.load(f"{res['ckpt']}/scrm_head.pt")
    assert torch.equal(m.special_emb.detach(), sd["special_emb"])


def test_permutation_consistency_loss_runs(synth, tmp_path):
    cfg = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={tmp_path/'r'}", "train.max_steps=2",
                                                  "train.eval_every=0", "train.save_every=0", "loss.w_perm=0.5",
                                                  "loss.w_listwise=0.5", "loss.w_plackett_luce=0.5", "loss.w_center=0.01"])
    assert train(cfg)["step"] == 2
