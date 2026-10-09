import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from scrm.feat_metrics import compute_metrics
from scrm.feat_train import evaluate, get_split, load_split, run_experiment, FEAT_DEFAULTS
from scrm.lossgrid import arms, load_grid_config, main as grid_main, make_tasks

D = 16
SOURCES = ["srcA", "srcB"]


def write_cache(root, sizes=None, d=D, seed=0, strength=2.0, variants=3, shard=40):
    """Synthetic cache in the documented format: the best option carries +strength along a fixed direction."""
    sizes = sizes or {"train": 120, "validation": 30, "test": 30}
    rng = np.random.default_rng(seed)
    direction = np.random.default_rng(123).normal(size=d)
    direction /= np.linalg.norm(direction)
    manifest = {"format": 2, "layers": [16], "hidden_size": d, "splits": {}}
    for split, n_sets in sizes.items():
        os.makedirs(os.path.join(root, split), exist_ok=True)
        state = manifest["splits"][split] = {"complete": True, "shards": []}
        rows, feats, meta = [], [], []
        for s in range(n_sets):
            n = int(rng.integers(3, 6))
            best = int(rng.integers(n))
            tiers = [0 if i == best else int(rng.integers(1, 3)) for i in range(n)]
            f = rng.normal(size=(n, d)) + strength * np.outer([float(t == 0) for t in tiers], direction)
            absl = [1.0 if t == 0 else (None if i == 0 and s % 5 == 0 else 0.0) for i, t in enumerate(tiers)]
            for v in range(variants):
                perm = list(range(n)) if v == 0 else list(rng.permutation(n))
                meta.append({"decision_set_id": f"{split}{s}", "variant": v, "perm_json": json.dumps([int(p) for p in perm]),
                             "n_options": n, "tiers_json": json.dumps(tiers),
                             "probabilities_json": json.dumps([0.7 if t == 0 else 0.15 for t in tiers]) if s % 4 == 0 else None,
                             "abs_label_json": json.dumps(absl), "label_kind": "graded" if s % 2 else "binary",
                             "family": "fam", "source_id": SOURCES[s % 2], "split": split, "truncated": False})
                feats.append(f + 0.05 * rng.normal(size=f.shape))
        for k in range(0, len(meta), shard):
            m = meta[k:k + shard]
            off = np.cumsum([0] + [r["n_options"] for r in m])[:-1]
            x = np.concatenate(feats[k:k + shard])
            tag = os.path.join(root, split, f"shard_{k // shard:05d}")
            # a second layer in the same file: one of the two tensors starts at a non-zero data offset
            save_file({"feat_L16": torch.tensor(x, dtype=torch.bfloat16),
                       "feat_L24": torch.tensor(-x, dtype=torch.bfloat16)}, tag + ".safetensors")
            pd.DataFrame([{**r, "row_offset": int(o)} for r, o in zip(m, off)]).to_parquet(tag + ".parquet")
            state["shards"].append({"name": f"shard_{k // shard:05d}", "n_records": len(m)})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return root


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return write_cache(str(tmp_path_factory.mktemp("cache")))


def _cfg(cache, **kw):
    cfg = load_grid_config("configs/lossgrid_debug.yaml", [f"cache_dir={cache}"] + [f"{k}={v}" for k, v in kw.items()])
    return cfg


def test_load_split(cache):
    sp = load_split(cache, "validation", 16)
    assert len(sp) == 30 and sp.d == D
    r = sp.sets[0]
    assert sorted(r.rows) == [0, 1, 2] and len(r.tiers) == r.n
    assert sp.sets[0].probs is not None and sp.sets[1].probs is None
    with pytest.raises(KeyError):
        load_split(cache, "validation", 99)


@pytest.mark.parametrize("layer,sign", [(16, 1.0), (24, -1.0)])
def test_load_split_rows_match_written_features(cache, layer, sign):
    """Memory-mapped rows of every (set, variant) block, across shard boundaries, equal the written bf16 tensor."""
    from safetensors.torch import load_file
    root = os.path.join(cache, "validation")
    names = sorted(n for n in os.listdir(root) if n.endswith(".safetensors"))
    full = torch.cat([load_file(os.path.join(root, n))[f"feat_L{layer}"] for n in names])
    ref = torch.cat([load_file(os.path.join(root, n))["feat_L16"] for n in names])
    assert torch.equal(full, (sign * ref.float()).to(torch.bfloat16))
    sp = load_split(cache, "validation", layer)
    for r in sp.sets:
        for off in r.rows.values():
            got = sp.rows(off, r.n)
            assert got.dtype == torch.bfloat16 and torch.equal(got, full[off:off + r.n])
    mu, sd = sp.norm_stats()
    x = torch.cat([full[r.rows[0]:r.rows[0] + r.n] for r in sp.sets]).double()
    assert torch.allclose(mu.double(), x.mean(0), atol=1e-5) and torch.allclose(sd.double(), x.std(0), atol=1e-4)


def test_incomplete_cache_cannot_train(tmp_path):
    root = write_cache(str(tmp_path / "cache"), sizes={"train": 4})
    path = os.path.join(root, "manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    manifest["splits"]["train"]["complete"] = False
    with open(path, "w") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="incomplete"):
        load_split(root, "train", 16)


def test_uncommitted_cache_files_are_ignored(tmp_path):
    root = write_cache(str(tmp_path / "cache"), sizes={"train": 4})
    # A forced stop may leave files after their rename but before manifest commit.
    with open(os.path.join(root, "train", "shard_99999.parquet"), "wb") as f:
        f.write(b"unfinished")
    assert len(load_split(root, "train", 16)) == 4


@pytest.mark.parametrize("head,terms", [("linear", ["ce"]), ("mlp", ["brier", "bt"]), ("set", ["sigmoid", "ce"])])
def test_trains_better_than_chance(cache, head, terms):
    cfg = _cfg(cache, **{"train.steps": 150, "train.lr": 0.01, "train.batch_size": 16, "model.head": head,
                         "model.d_set": 16, "model.set_heads": 2, "model.set_layers": 1})
    cfg["loss"]["terms"] = terms
    res = run_experiment(cfg, ["validation"], log=lambda *_: None)
    m = res["metrics"]["validation"]["all"]
    chance = np.mean([1 / n for n in (3, 4, 5)])
    assert m["top1"] > chance + 0.25, m
    assert set(res["metrics"]["validation"]) >= {"all", "source/srcA", "label_kind/binary"}
    assert math.isfinite(m["flip_rate"]) and math.isfinite(m["ece_top1"])
    assert math.isfinite(m["sig_ece"]) == ("sigmoid" in terms)


def test_deterministic_and_arm_independent_data_order(cache):
    cfg = _cfg(cache, **{"train.steps": 5})
    cfg["loss"]["terms"] = ["bt"]
    a = run_experiment(cfg, ["validation"], log=lambda *_: None)
    b = run_experiment(cfg, ["validation"], log=lambda *_: None)
    assert a["metrics"]["validation"]["all"]["top1"] == b["metrics"]["validation"]["all"]["top1"]
    assert a["train_loss_ema"] == b["train_loss_ema"]


def test_perm_and_augment_flags_run(cache):
    cfg = _cfg(cache, **{"train.steps": 3, "loss.w_perm": 0.5, "data.augment_variants": True})
    cfg["loss"]["terms"] = ["ce", "sigmoid"]
    run_experiment(cfg, ["validation"], log=lambda *_: None)


def test_grid_enumerates_15_arms_and_summarises(cache, tmp_path):
    assert len(arms()) == 15 and arms()[0] == ("ce",) and arms()[-1] == ("ce", "brier", "bt", "sigmoid")
    cfg = load_grid_config("configs/lossgrid_debug.yaml", [f"cache_dir={cache}", f"output_dir={tmp_path}"])
    tasks = make_tasks(cfg)
    assert len(tasks) == 15 * 2 * 1 * 2          # arms x lr x weight x seeds: equal per arm
    per_arm = {}
    for t in tasks:
        per_arm[t["arm"]] = per_arm.get(t["arm"], 0) + 1
    assert set(per_arm.values()) == {4}
    grid_main(["--config", "configs/lossgrid_debug.yaml", f"cache_dir={cache}", f"output_dir={tmp_path}"])
    md = open(tmp_path / "summary.md").read()
    assert "ce+brier+bt+sigmoid" in md and "Main effects" in md
    sel = json.load(open(tmp_path / "selected.json"))
    assert len(sel) == 15 and all(len(v["heads"]) == 2 and all(os.path.exists(h) for h in v["heads"])
                                  for v in sel.values())
    eff = pd.read_csv(tmp_path / "effects_validation.csv")
    assert {"ce", "sigmoid", "cexsigmoid"} <= set(eff["effect"])
    summ = pd.read_csv(tmp_path / "summary_validation.csv")
    assert len(summ) == 15 and (summ["n_seeds"] == 2).all()
    assert len(open(tmp_path / "results.jsonl").read().splitlines()) == 60
    # resumable: a rerun does not retrain (mtime unchanged)
    p = tmp_path / "runs"
    before = {f.name: f.stat().st_mtime_ns for f in p.iterdir()}
    grid_main(["--config", "configs/lossgrid_debug.yaml", f"cache_dir={cache}", f"output_dir={tmp_path}"])
    assert before == {f.name: f.stat().st_mtime_ns for f in p.iterdir()}
    # a different per-run config (here the step budget) must not silently reuse those finished runs
    with pytest.raises(SystemExit, match="train"):
        grid_main(["--config", "configs/lossgrid_debug.yaml", f"cache_dir={cache}", f"output_dir={tmp_path}",
                   "train.steps=7"])
    assert before == {f.name: f.stat().st_mtime_ns for f in p.iterdir()}


def test_best_validation_checkpoint_is_restored_and_saved(cache, tmp_path):
    """The restored state scores the recorded best validation loss; the saved head reloads to the same rewards."""
    from scrm.feat_train import load_head, make_batch, train_run, val_loss
    cfg = _cfg(cache, **{"train.steps": 40, "train.lr": 0.5, "train.eval_every": 5, "train.warmup_steps": 1})
    cfg["loss"]["terms"] = ["ce"]
    model, readouts, info = train_run(cfg, log=lambda *_: None)
    val = get_split(cfg, "validation")
    terms, w = info["terms"], info["weights"]
    again = val_loss(model, readouts, val, terms, w, cfg["loss"], cfg["train"], torch.device("cpu"))
    assert again == pytest.approx(info["val_loss"])
    res = run_experiment(cfg, ["validation"], log=lambda *_: None, head_path=str(tmp_path / "h.pt"))
    assert res["best_step"] % 5 == 0 or res["best_step"] == 40
    head, blob = load_head(str(tmp_path / "h.pt"), torch.device("cpu"))
    assert blob["feature_key"] == "feat_L16" and blob["best_step"] == res["best_step"]
    b = make_batch(val, list(range(8)))
    with torch.no_grad():
        assert torch.allclose(head(b["x"], b["mask"]), model.eval()(b["x"], b["mask"]))


# ---- metrics ----

def _toy(S=4, N=3):
    tiers = np.tile([0, 1, 2], (S, 1))
    return dict(tiers=tiers, probs=np.zeros((S, N)), has_probs=np.zeros(S, bool), abs_labels=np.tile([1.0, 0, 0], (S, 1)),
                abs_mask=np.ones((S, N), bool), sources=["a", "a", "b", "b"], kinds=["k"] * S)


def test_ece_known_case_and_ranking():
    t = _toy()
    # perfect ranking, softmax confidence ~ p; hit = 1 -> ECE = 1 - mean conf
    s = np.tile([2.0, 0.0, 0.0], (4, 1))
    m = compute_metrics(s, **t)
    p = math.exp(2) / (math.exp(2) + 2)
    assert m["all"]["top1"] == 1 and m["all"]["pair_acc"] == pytest.approx(5 / 6)   # tiers 1 vs 2 tied
    assert abs(m["all"]["ece_top1"] - (1 - p)) < 1e-6
    assert abs(m["all"]["brier_softmax"] - ((1 - p) ** 2 + 2 * ((1 - p) / 2) ** 2)) < 1e-6
    assert abs(m["all"]["nll_softmax"] + math.log(p)) < 1e-6
    assert {"source/a", "source/b", "label_kind/k"} <= set(m)
    assert math.isnan(m["all"]["sig_ece"])                       # no sigmoid readout supplied
    ms = compute_metrics(s, **t, alpha=1.0, bias=0.0)
    sig = 1 / (1 + math.exp(-2)), 0.5, 0.5
    exp_brier = ((sig[0] - 1) ** 2 + 2 * 0.25) / 3
    assert abs(ms["all"]["sig_brier"] - exp_brier) < 1e-6
    assert ms["all"]["sig_ece"] > 0


def test_abstention_and_order_stability():
    t = _toy()
    s = np.tile([4.0, 0.0, 0.0], (4, 1))
    after = np.tile([0.0, 0.0, 0.0], (4, 1))
    am = np.tile([False, True, True], (4, 1))
    alt = [(np.array([0, 1]), np.tile([0.0, 4.0, 0.0], (2, 1))), (np.array([0, 1, 2, 3]), s)]
    m = compute_metrics(s, **t, alpha=1.0, bias=-2.0, after_scores=after, after_mask=am, alt_scores=alt)["all"]
    assert m["abst_n"] == 4
    assert m["abst_sigmoid_after"] < m["abst_sigmoid_before"] and m["abst_sigmoid_drop"] > 0.1
    assert abs(m["abst_sigmoid_after"] - 1 / (1 + math.exp(2))) < 1e-6
    assert m["abst_softmax_after"] == pytest.approx(0.5)         # softmax cannot abstain: still sums to 1
    assert m["flip_rate"] == pytest.approx(0.25)                 # sets 0,1 flip in 1 of 2 variants -> mean (0.5+0.5+0+0)/4
    assert m["mean_abs_dp"] > 0


def test_abstention_via_model_removal(cache):
    cfg = _cfg(cache, **{"train.steps": 120, "train.lr": 0.01, "train.batch_size": 16})
    cfg["loss"]["terms"] = ["sigmoid"]
    from scrm.feat_train import train_run, _device
    model, ro, info = train_run(cfg, log=lambda *_: None)
    m = evaluate(model, ro, get_split(cfg, "validation"), info["terms"], cfg["train"], _device(cfg["train"]))["all"]
    assert m["abst_n"] > 0 and m["abst_sigmoid_after"] < m["abst_sigmoid_before"]
