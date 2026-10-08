"""Frozen-backbone feature cache (scrm.features): format, row/permutation mapping, resume, absolute labels."""
import json
import os

import numpy as np
import pytest
import torch

from scrm import features as F
from scrm.collator import collate
from scrm.config import load_config
from scrm.model import build_scrm
from scrm.render import Renderer
from scrm.synth import write_synth


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return write_synth(str(tmp_path_factory.mktemp("synth")))


def _cfg(synth, out, *extra):
    return load_config(None, [f"data.local_dir={synth}", "model.name_or_path=tiny", "model.dtype=float32",
                              "data.render.max_len=2048", f"features.out_dir={out}", "features.layers=[2,-1]",
                              "features.shard_size=4", "features.splits=[train,validation]",
                              "features.max_sets={train: 6, validation: 3}", *extra])


def _model(cfg):
    mcfg = {**cfg["model"], "freeze_backbone": True, "gradient_checkpointing": False, "head": "linear",
            "branching": False, "lora": {**cfg["model"]["lora"], "enabled": False}}
    model, tok = build_scrm(mcfg, "cpu")
    return model.eval(), Renderer(tok, {**cfg["data"]["render"], "max_graded": None, "min_graded": None})


def test_extract_end_to_end_format(synth, tmp_path):
    out = str(tmp_path / "f")
    man = F.run(_cfg(synth, out))
    assert json.load(open(os.path.join(out, "manifest.json")))["fingerprint"] == man["fingerprint"]
    assert man["layers"] == [2, -1] and man["variants"] == 2 and man["hidden_size"] == 64 and man["dtype"] == "bfloat16"
    assert man["keys"] == ["feat_L2", "feat_Llast"]
    for split, n in (("train", 6), ("validation", 3)):
        sp = man["splits"][split]
        assert sp["complete"] and sp["n_selected"] == n and sp["n_sets"] == n - sp["n_dropped"]
        assert [s["name"] for s in sp["shards"]] == [f"shard_{i:05d}" for i in range(len(sp["shards"]))]
        tot = 0
        for i, s in enumerate(sp["shards"]):
            tens, tab = F.load_shard(os.path.join(out, split), i)
            assert set(tens) == {"feat_L2", "feat_Llast"}
            assert all(t.dtype == torch.bfloat16 and t.shape == (s["n_rows"], 64) for t in tens.values())
            assert tab.column_names == list(F.SHARD_COLS)
            rows = tab.to_pylist()
            assert len(rows) == 2 * s["n_sets"]
            off = 0
            for r in rows:
                assert r["row_offset"] == off and r["split"] == split
                off += r["n_options"]
                perm = json.loads(r["perm_json"])
                assert sorted(perm) == list(range(r["n_options"]))
                if r["variant"] == 0:
                    assert perm == list(range(r["n_options"]))
                assert len(json.loads(r["tiers_json"])) == len(json.loads(r["abs_label_json"])) == r["n_options"]
                assert min(json.loads(r["tiers_json"])) == 0 and r["truncated"] is False
            assert off == s["n_rows"]
            tot += s["n_rows"]
        assert tot == sp["n_rows"]


def test_rows_match_embed_through_perm(synth, tmp_path):
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.variants=3")
    man = F.run(cfg)
    model, renderer = _model(cfg)
    from scrm.data import _read_rows
    files, locs = F.select_locations(cfg["data"], "train", cfg["data"]["filters"], 6, 0)
    rows = {r["decision_set_id"]: r for r in _read_rows(files, [tuple(x) for x in locs[:4]], F.EXTRA_COLS)}
    tens, tab = F.load_shard(os.path.join(out, "train"), 0)
    checked = 0
    for rec in tab.to_pylist():
        meta, items, perms = F._set_jobs(renderer, rows[rec["decision_set_id"]], 3, cfg["features"]["seed"])
        it, perm = items[rec["variant"]], perms[rec["variant"]]
        assert json.loads(rec["perm_json"]) == perm
        assert rec["tiers_json"] == json.dumps(meta["tiers"])
        with torch.no_grad():
            e = model.embed(collate([it], renderer.pad_id)["pack"])    # presentation order
        o = rec["row_offset"]
        for p, c in enumerate(perm):                                   # position p shows canonical option c
            assert torch.allclose(tens["feat_Llast"][o + c].float(), e[p], rtol=2e-2, atol=2e-2)
        checked += 1
    assert checked > 0
    assert man["splits"]["train"]["n_sets"] >= 1
    # variants >= 1 really reorder at least one set with >= 3 options
    assert any(json.loads(r["perm_json"]) != sorted(json.loads(r["perm_json"])) for r in tab.to_pylist() if r["variant"] > 0)


def test_embed_layers_api(synth):
    model, renderer = _model(_cfg(synth, "unused"))
    from test_scrm_model import _sets
    pack, _ = _sets([3, 2], P=9, seed=2)
    with torch.no_grad():
        base = model.embed(pack)
        d = model.embed(pack, layers=[2, 4, -1])
        d2 = model.embed(pack, max_tokens=1, layers=[-1])      # one full sequence per chunk
    n_layers = model.backbone.config.num_hidden_layers
    assert n_layers == 4 and list(d) == [2, 4, -1]
    assert torch.equal(d[-1], base) and torch.equal(d[4], base)   # layer 4 + final norm == last layer
    assert torch.allclose(d2[-1], base, atol=1e-5)
    assert not torch.allclose(d[2], base) and d[2].shape == base.shape
    with pytest.raises(ValueError):
        model.embed(pack, layers=[0])
    with pytest.raises(ValueError):
        model.embed(pack, layers=[n_layers + 1])


@pytest.mark.parametrize("one_token", [False, True])
@pytest.mark.parametrize("cache_tokens", [10 ** 6, 1])
@pytest.mark.parametrize("prefill_tokens", [10 ** 6, 1])
@pytest.mark.parametrize("max_batch_size", [1, 2])
def test_prefix_cached_matches_full_sequences(synth, one_token, cache_tokens, prefill_tokens, max_batch_size):
    """Selected cache rows stay independent across suffix forwards and preserve pack order, including mixed
    prompt lengths, row/token chunk boundaries, single-token continuation and intermediate-layer readouts."""
    model, _ = _model(_cfg(synth, "unused"))
    from test_scrm_model import _items
    items = _items([3, 1, 4], P=11, seed=4, one_token=one_token)
    items[1].prefix = items[1].prefix[:-2]
    pack = collate(items)["pack"]
    with torch.no_grad():
        a = model.embed(pack, layers=[1, 2, 3, -1])
    b = model.embed_prefix_cached(pack, layers=[1, 2, 3, -1], cache_tokens=cache_tokens,
                                  prefill_tokens=prefill_tokens, max_batch_size=max_batch_size)
    assert list(b) == [1, 2, 3, -1]
    for l in a:
        assert torch.allclose(a[l], b[l], atol=2e-5), (l, (a[l] - b[l]).abs().max().item())
    assert torch.allclose(model.embed_prefix_cached(pack, cache_tokens=cache_tokens, prefill_tokens=prefill_tokens,
                                                   max_batch_size=max_batch_size), a[-1], atol=2e-5)


def test_resume_skips_completed_shards(synth, tmp_path, monkeypatch):
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.shard_size=2", "features.splits=[train]")
    F.run(cfg)
    d = os.path.join(out, "train")
    files = sorted(f for f in os.listdir(d))
    assert files == [f"shard_{i:05d}.{e}" for i in range(3) for e in ("parquet", "safetensors")]
    stamp = {f: os.stat(os.path.join(d, f)).st_mtime_ns for f in files}
    # forget the last shard (as after a crash before the manifest update) and re-run
    man = json.load(open(os.path.join(out, "manifest.json")))
    man["splits"]["train"]["shards"] = [s for s in man["splits"]["train"]["shards"] if s["name"] != "shard_00002"]
    man["splits"]["train"]["complete"] = False
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"))
    calls = []
    real = F.extract_rows
    monkeypatch.setattr(F, "extract_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    man2 = F.run(cfg)
    assert len(calls) == 1 and man2["splits"]["train"]["complete"]
    assert len(man2["splits"]["train"]["shards"]) == 3
    new = {f: os.stat(os.path.join(d, f)).st_mtime_ns for f in files}
    assert all(new[f] == stamp[f] for f in files if not f.startswith("shard_00002"))
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    calls.clear()
    cfg["features"].update(max_tokens=32768, max_batch_size=64, cache_tokens=65536)
    F.run(cfg)
    assert calls == []                                              # everything complete: nothing to do


def test_lowered_max_sets_keeps_prefix_shards(synth, tmp_path, monkeypatch):
    """Lowering a split's cap keeps the committed shards fully inside the new prefix (same rows, untouched files),
    recomputes only a straddling shard, and refuses to raise the cap."""
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]"))       # 6 rows -> 3 shards
    d = os.path.join(out, "train")
    stamp = {f: os.stat(os.path.join(d, f)).st_mtime_ns for f in os.listdir(d)}
    calls = []
    real = F.extract_rows
    monkeypatch.setattr(F, "extract_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    man = F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]", "features.max_sets.train=4"))
    sp = man["splits"]["train"]
    assert calls == [] and sp["complete"] and [s["name"] for s in sp["shards"]] == ["shard_00000", "shard_00001"]
    assert sorted(os.listdir(d)) == sorted(f for f in stamp if not f.startswith("shard_00002"))
    assert all(os.stat(os.path.join(d, f)).st_mtime_ns == stamp[f] for f in os.listdir(d))
    man = F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]", "features.max_sets.train=3"))
    sp = man["splits"]["train"]
    assert calls == [1] and sp["complete"] and sp["n_selected"] == 3
    with pytest.raises(SystemExit):
        F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]", "features.max_sets.train=6"))


def test_config_change_needs_new_out_dir(synth, tmp_path):
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.splits=[validation]"))
    with pytest.raises(SystemExit):
        F.run(_cfg(synth, out, "features.splits=[validation]", "features.variants=3"))


def test_candidate_cap_is_recorded(synth, tmp_path):
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "data.render.max_candidates=4", "features.splits=[train]", "features.max_sets={train: 3}")
    F.run(cfg)
    _, tab = F.load_shard(os.path.join(out, "train"), 0)
    rows = tab.to_pylist()
    assert all(r["n_options"] <= 4 for r in rows) and any(r["truncated"] for r in rows)       # massive: 8 options


def _abs(kind, tiers, ids=None, **kw):
    return F.absolute_labels({"label_kind": kind, "tiers": tiers, "choice_ids": ids or [f"c{i}" for i in range(len(tiers))], **kw})


def test_absolute_labels_rules():
    assert _abs("source_dataset_label", [1, 0, 1]) == [0.0, 1.0, 0.0]
    assert _abs("source_dataset_label", [0, 0, 1]) == [1.0, 1.0, 0.0]
    # noul: the option's own probability
    assert _abs("independent_labels_binary_tiers", [1, 0], ["no", "yes"], probabilities=[0.2, 0.8]) == [0.2, 0.8]
    assert _abs("independent_labels_binary_tiers", [0, 1], ["no", "yes"], probabilities={"yes": 0.3, "no": 0.7}) == [0.7, 0.3]
    assert _abs("independent_labels_binary_tiers", [1, 0], probabilities=None) == [None, None]
    # agent_choice: first target 1.0, co-targets excluded, rest 0.0
    sl = {"ordered_target_candidate_ids": ["c2", "c0"]}
    assert _abs("agent_choice_target", [1, 1, 0, 1], source_label=sl) == [None, 0.0, 1.0, 0.0]
    assert _abs("agent_choice_target", [1, 0, 1], source_label={"ordered_target_candidate_ids": []}) == [0.0, 1.0, 0.0]
    assert _abs("agent_choice_target", [1, 0, 1]) == [0.0, 1.0, 0.0]
    assert _abs("agent_choice_target", [1, 0, 1], source_label={"ordered_target_candidate_ids": ["zzz"]}) == [0.0, 1.0, 0.0]
    # no absolute meaning
    assert _abs("ordinal_score_distance_tiers", [0, 1, 2]) == [None] * 3
    assert _abs("soft_choice_distribution", [0, 1]) == [None] * 2
    assert _abs("something_new", [0, 1]) == [None] * 2


def test_stop_commits_shard_then_resumes(synth, tmp_path, monkeypatch):
    """The first stop request finishes and durably commits the shard in flight; a rerun resumes it, never recomputing
    a committed shard and never marking the untouched split complete."""
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.shard_size=2")
    ctrl = F._StopController()
    real, calls = F.extract_rows, []

    def stopping(*a, **k):
        res = real(*a, **k)
        calls.append(1)
        if len(calls) == 1:                        # first shard's features are ready: request the stop
            ctrl.request()
        return res

    monkeypatch.setattr(F, "extract_rows", stopping)
    man = F.run(cfg, stop=ctrl)
    assert man["stopped"] and calls == [1]
    assert not man["splits"]["train"]["complete"] and man["splits"]["train"]["n_shards"] == 3
    assert [s["name"] for s in man["splits"]["train"]["shards"]] == ["shard_00000"]
    assert "validation" not in man["splits"]            # never reached: not marked complete
    d = os.path.join(out, "train")
    assert sorted(os.listdir(d)) == ["shard_00000.parquet", "shard_00000.safetensors"]
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    stamp = os.stat(os.path.join(d, "shard_00000.safetensors")).st_mtime_ns
    monkeypatch.setattr(F, "extract_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    calls.clear()
    man2 = F.run(cfg)                                   # resume without a stop
    assert man2["splits"]["train"]["complete"] and man2["splits"]["validation"]["complete"]
    assert len(calls) == 4                              # 2 missing train + 2 validation shards recomputed
    assert os.stat(os.path.join(d, "shard_00000.safetensors")).st_mtime_ns == stamp
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_changed_data_selection_needs_new_out_dir(synth, tmp_path):
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.splits=[train]"))
    with pytest.raises(SystemExit):
        F.run(_cfg(synth, out, "features.splits=[train]", "data.repo=someone/other-dataset"))


def test_manifest_selection_count_mismatch_errors(synth, tmp_path):
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.splits=[train]"))
    p = os.path.join(out, "manifest.json")
    man = json.load(open(p))
    man["splits"]["train"]["n_selected"] += 1          # selection changed behind the (matching) fingerprint
    json.dump(man, open(p, "w"))
    with pytest.raises(SystemExit, match="input selection changed"):
        F.run(_cfg(synth, out, "features.splits=[train]"))


def test_legacy_hf_manifest_resumes_and_upgrades(synth, tmp_path, monkeypatch):
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.shard_size=2", "features.splits=[train]")
    F.run(cfg)
    p = os.path.join(out, "manifest.json")
    cur = json.load(open(p))
    key_keep = ("model", "layers", "variants", "seed", "render", "shard_size", "max_sets")
    splits = {k: {kk: vv for kk, vv in v.items() if kk != "selection"} for k, v in cur["splits"].items()}
    legacy = {"format": 1, "fingerprint": cur["legacy_fingerprint"], "hidden_size": cur["hidden_size"],
              "dtype": cur["dtype"], "keys": cur["keys"], "splits": splits, **{k: cur[k] for k in key_keep}}
    json.dump(legacy, open(p, "w"))                    # rewrite in the pre-selection-fingerprint layout
    with pytest.raises(SystemExit, match="legacy HF manifest"):   # opt-in: its selection was never recorded
        F.run(cfg)
    calls, real = [], F.extract_rows
    monkeypatch.setattr(F, "extract_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    man = F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]",
                     "features.resume_legacy_hf=true"))
    assert calls == [] and man["splits"]["train"]["complete"]
    assert json.load(open(p))["backend"] == "hf"       # upgraded in place, still resumable next time


def test_changed_source_files_rejected(synth, tmp_path):
    import glob
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.shard_size=2", "features.splits=[train]")
    F.run(cfg)
    src = sorted(glob.glob(os.path.join(synth, "data", "train-*.parquet")))[0]
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 10))     # same rows, new file revision: not the same selection
    with pytest.raises(SystemExit, match="selected files/rows changed"):
        F.run(cfg)


def test_vllm_backend_identity_and_hf_manifest_rejected(synth, tmp_path):
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.splits=[train]"))           # HF cache
    with pytest.raises(SystemExit):                              # an HF manifest cannot be resumed as vLLM
        F.run(_cfg(synth, out, "features.splits=[train]", "features.backend=vllm"))


def test_missing_committed_shard_is_a_hard_error(synth, tmp_path):
    out = str(tmp_path / "f")
    cfg = _cfg(synth, out, "features.shard_size=2", "features.splits=[train]")
    F.run(cfg)
    os.unlink(os.path.join(out, "train", "shard_00000.safetensors"))
    with pytest.raises(SystemExit, match="missing or unreadable"):
        F.run(cfg)


def test_lock_blocks_concurrent_writers(synth, tmp_path):
    import fcntl
    out = str(tmp_path / "f")
    os.makedirs(out)
    fh = open(os.path.join(out, F.LOCK_FILE), "a+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(SystemExit, match="locked by another"):
            F.run(_cfg(synth, out, "features.splits=[train]"))
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    assert not os.path.exists(os.path.join(out, "manifest.json"))   # the blocked writer never touched the cache
    man = F.run(_cfg(synth, out, "features.splits=[train]"))
    assert man["splits"]["train"]["complete"] and F.LOCK_FILE in os.listdir(out)


def test_overwrite_drops_stale_shards(synth, tmp_path):
    out = str(tmp_path / "f")
    F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]"))
    d = os.path.join(out, "train")
    assert sorted(os.listdir(d)) == [f"shard_{i:05d}.{e}" for i in range(3) for e in ("parquet", "safetensors")]
    man = F.run(_cfg(synth, out, "features.shard_size=2", "features.splits=[train]",
                     "features.max_sets={train: 2}", "features.overwrite=true"))
    assert man["splits"]["train"]["complete"] and man["splits"]["train"]["n_shards"] == 1
    assert sorted(os.listdir(d)) == [f"shard_00000.{e}" for e in ("parquet", "safetensors")]
