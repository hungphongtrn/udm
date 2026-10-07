"""Frozen-backbone feature cache: run the backbone once over the data (same "Grade this choice: Option k" prompt as
training; stock HF forward with prefix caching: each set's prompt is prefilled once, every option continues from that
cache, see `SCRM.embed_prefix_cached`) and store one feature vector per (set, variant, option), so heads / losses can
be trained on the cached features.

Cache layout, `<out_dir>/<split>/`:
  shard_00000.safetensors   `feat_L{layer}` per requested layer (`feat_Llast` for -1), [M, d] bfloat16, one row per
                            (set, variant, option); inside a (set, variant) block rows are in CANONICAL option order
  shard_00000.parquet       one row per (set, variant): decision_set_id, variant, perm_json, n_options, row_offset,
                            tiers_json, probabilities_json, abs_label_json, label_kind, family, source_id, split, truncated
  `<out_dir>/manifest.json` model / layers / render / variants / seed + per-split completed shards and counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zlib

import numpy as np
import torch

from .config import load_config
from .render import Renderer, parse_row

EXTRA_COLS = ("probabilities_json", "source_label_or_null")
SHARD_COLS = ("decision_set_id", "variant", "perm_json", "n_options", "row_offset", "tiers_json", "probabilities_json",
              "abs_label_json", "label_kind", "family", "source_id", "split", "truncated")


# ----------------------------------------------------------------------------- absolute labels
def _loads(v):
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v


def absolute_labels(row: dict) -> list:
    """Per-option absolute (sigmoid-loss) target in canonical option order: float in [0, 1], or None = excluded.

    `row` keys: label_kind, tiers (list[int], lower = better), choice_ids (list[str]), and optionally
    probabilities (list aligned with choice_ids, or {choice_id: p}, or None/{}) and source_label (the parsed
    `source_label_or_null` object, or None). Rules by label_kind:
      source_dataset_label             exclusive choice: tier 0 -> 1.0, others 0.0 (all tier-0 options if several)
      independent_labels_binary_tiers  noul: the option's own probability P(option is right) from probabilities_json
                                       ({no: 1-p_yes, yes: p_yes} for the yes/no form, the independent probs for the
                                       multi-label form); None for every option if probabilities are missing
      agent_choice_target              source_label.ordered_target_candidate_ids: first present id -> 1.0, other
                                       co-targets -> None (excluded), rest 0.0; no such ids -> the tier-0 rule
      soft_choice_distribution         None (a normalised distribution, not a per-option calibrated probability)
      ordinal_score_distance_tiers     None (graded levels have no absolute 0/1 meaning)
      anything else                    None
    """
    kind = row.get("label_kind")
    tiers = [int(t) for t in row["tiers"]]
    ids = list(row.get("choice_ids") or [])
    n = len(tiers)
    top = [1.0 if t == min(tiers) else 0.0 for t in tiers] if n else []
    if kind == "source_dataset_label":
        return top
    if kind == "independent_labels_binary_tiers":
        p = row.get("probabilities")
        if isinstance(p, dict):
            p = [p.get(c) for c in ids] if p else None
        if not p or len(p) != n or any(x is None for x in p):
            return [None] * n
        return [float(min(1.0, max(0.0, x))) for x in p]
    if kind == "agent_choice_target":
        sl = row.get("source_label") or {}
        ordered = [c for c in (sl.get("ordered_target_candidate_ids") or []) if c in ids]
        if not ordered:
            return top
        out = [0.0] * n
        for k, c in enumerate(dict.fromkeys(ordered)):
            out[ids.index(c)] = 1.0 if k == 0 else None
        return out
    return [None] * n


# ----------------------------------------------------------------------------- selection / reading
def select_locations(dcfg: dict, split: str, filters: dict, max_sets, seed: int) -> tuple[list[str], np.ndarray]:
    """(files, [n, 2] array of (file_idx, row_idx)) of the rows to featurize: every filtered row of the split, sources
    interleaved round-robin over seeded random per-source orders, truncated to `max_sets` (None = all rows)."""
    import pyarrow.parquet as pq
    from .data import FILTER_COLS, Pred, _concrete, _files, _open
    pred = Pred(filters.get("include"), filters.get("exclude"))
    files = _concrete(_files(dcfg, split))
    by_src: dict[str, list] = {}
    for fi, f in enumerate(files):
        t = pq.read_table(_open(f), columns=list(FILTER_COLS))
        keep = np.where(pred.mask(t))[0]
        src = np.asarray(t.column("source_id").to_numpy(zero_copy_only=False))[keep]
        for s in np.unique(src):
            by_src.setdefault(str(s), []).append(np.stack([np.full((src == s).sum(), fi), keep[src == s]], 1))
    rng = np.random.default_rng(seed)
    locs, keys = [], []
    for si, s in enumerate(sorted(by_src)):
        a = rng.permutation(np.concatenate(by_src[s]))
        if max_sets:
            a = a[:max_sets]
        locs.append(a)
        keys.append(np.arange(len(a)) * len(by_src) + si)      # rank within source, then source: round robin
    if not locs:
        return files, np.zeros((0, 2), dtype=np.int64)
    locs, keys = np.concatenate(locs), np.concatenate(keys)
    out = locs[np.argsort(keys, kind="stable")]
    return files, out[:max_sets] if max_sets else out


# ----------------------------------------------------------------------------- one shard
def _set_jobs(renderer: Renderer, row: dict, variants: int, seed: int):
    """Row -> (meta, items) with one Item per variant (variant 0 canonical), or None when it does not render.
    All variants grade exactly the same kept options (`kept` of the canonical assembly)."""
    ex = parse_row(row, use_candidate_rows=True)
    if ex is None:
        return None
    t = renderer.tokenize(ex)
    it0 = renderer.assemble(t, None, False, relax=True)       # canonical order, deterministic candidate cap
    if it0 is None:
        return None
    canon = list(it0.kept)
    pos = {c: i for i, c in enumerate(canon)}
    items, perms = [it0], [list(range(len(canon)))]
    crc = zlib.crc32(ex.decision_set_id.encode())
    for v in range(1, variants):
        rng = np.random.default_rng([seed, v, crc])
        it = renderer.assemble(t, rng, True, kept=canon, graded=canon, relax=True)
        items.append(it)
        perms.append([pos[int(o)] for o in it.order])
    ids = [ex.choice_ids[i] for i in canon]
    probs = _loads(row.get("probabilities_json")) or None
    probs = [probs.get(c) for c in ids] if isinstance(probs, dict) else None
    if probs is not None and all(p is None for p in probs):
        probs = None
    meta = {"decision_set_id": ex.decision_set_id, "n_options": len(canon), "tiers": [int(ex.tiers[i]) for i in canon],
            "probabilities": probs, "label_kind": ex.label_kind, "family": ex.family, "source_id": ex.source_id,
            "truncated": len(canon) < len(ex.choice_ids), "choice_ids": ids,
            "source_label": _loads(row.get("source_label_or_null")) if row.get("source_label_or_null") else None}
    meta["abs_label"] = absolute_labels({**meta, "tiers": meta["tiers"]})
    return meta, items, perms


@torch.no_grad()
def extract_rows(model, renderer: Renderer, rows: list[dict], fcfg: dict, split: str, device) -> tuple[dict, list, int]:
    """Featurize rows -> ({layer: [M, d] bf16 tensors}, parquet records, number of dropped rows)."""
    from .collator import collate
    from .packing import pack_bfd
    layers, K = [int(l) for l in fcfg["layers"]], int(fcfg["variants"])
    jobs, dropped = [], 0
    for row in rows:
        j = _set_jobs(renderer, row, K, int(fcfg["seed"]))
        if j is None:
            dropped += 1
        else:
            jobs.append(j)
    d = model.d_hidden
    total = sum(m["n_options"] * K for m, _, _ in jobs)
    feats = {l: torch.zeros(total, d, dtype=torch.bfloat16) for l in layers}
    records, where, off = [], {}, 0
    for meta, items, perms in jobs:
        for v, (it, perm) in enumerate(zip(items, perms)):
            where[id(it)] = (off, perm)
            records.append({"decision_set_id": meta["decision_set_id"], "variant": v, "perm_json": json.dumps(perm),
                            "n_options": meta["n_options"], "row_offset": off, "tiers_json": json.dumps(meta["tiers"]),
                            "probabilities_json": None if meta["probabilities"] is None else json.dumps(meta["probabilities"]),
                            "abs_label_json": json.dumps(meta["abs_label"]), "label_kind": meta["label_kind"],
                            "family": meta["family"], "source_id": meta["source_id"], "split": split,
                            "truncated": bool(meta["truncated"])})
            off += meta["n_options"]
    buckets = {}
    for _, its, _ in jobs:
        for it in its:
            buckets.setdefault(len(it.prefix), []).append(it)
    amp = device.type == "cuda"
    batches = (
        [items[i] for i in indices]
        for P, items in buckets.items()
        for indices in pack_bfd([P + sum(len(s) for s in it.suffixes) for it in items],
                                int(fcfg["max_tokens"]), int(fcfg["max_batch_size"]))
    )
    for batch_items in batches:
        b = collate(batch_items, pad_id=renderer.pad_id)
        # Scheduling metadata stays on CPU: embed_prefix_cached reads it as Python lists.
        # Moving it to CUDA only to call .tolist() forces an avoidable GPU synchronization.
        for key in ("input_ids", "position_ids"):
            b["pack"][key] = b["pack"][key].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            e = model.embed_prefix_cached(b["pack"], layers=layers, cache_tokens=int(fcfg["cache_tokens"]),
                                          prefill_tokens=int(fcfg["max_tokens"]),
                                          max_batch_size=int(fcfg["max_batch_size"]))
        e = {l: x.to(torch.bfloat16).cpu() for l, x in e.items()}
        s = 0
        for it in batch_items:
            o, perm = where[id(it)]
            n = len(perm)
            dest = o + torch.as_tensor(perm)       # slot p was presented at position p = canonical option perm[p]
            for l in layers:
                feats[l][dest] = e[l][s:s + n]
            s += n
    return feats, records, dropped


def layer_key(l: int) -> str:
    return "feat_Llast" if int(l) == -1 else f"feat_L{int(l)}"


def _atomic(path: str, write) -> None:
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def write_shard(split_dir: str, idx: int, feats: dict, records: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors.torch import save_file
    base = os.path.join(split_dir, f"shard_{idx:05d}")
    _atomic(base + ".safetensors", lambda p: save_file({layer_key(l): x.contiguous() for l, x in feats.items()}, p))
    schema = pa.schema([("decision_set_id", pa.string()), ("variant", pa.int32()), ("perm_json", pa.string()),
                        ("n_options", pa.int32()), ("row_offset", pa.int64()), ("tiers_json", pa.string()),
                        ("probabilities_json", pa.string()), ("abs_label_json", pa.string()),
                        ("label_kind", pa.string()), ("family", pa.string()), ("source_id", pa.string()),
                        ("split", pa.string()), ("truncated", pa.bool_())])
    table = pa.Table.from_pylist(records, schema=schema)
    _atomic(base + ".parquet", lambda p: pq.write_table(table, p))


def load_shard(split_dir: str, idx: int):
    """(tensors {feat key: [M, d] bf16}, pyarrow Table of the (set, variant) rows) of one shard."""
    import pyarrow.parquet as pq
    from safetensors.torch import load_file
    base = os.path.join(split_dir, f"shard_{idx:05d}")
    return load_file(base + ".safetensors"), pq.read_table(base + ".parquet")


# ----------------------------------------------------------------------------- driver
def _write_manifest(out_dir: str, man: dict) -> None:
    _atomic(os.path.join(out_dir, "manifest.json"), lambda p: open(p, "w").write(json.dumps(man, indent=2)))


def run(cfg: dict, device=None) -> dict:
    from .data import _read_rows
    fcfg = cfg["features"]
    out_dir = fcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # stock HF forward + prefix caching (`embed_prefix_cached`)
    mcfg = {**cfg["model"], "freeze_backbone": True, "gradient_checkpointing": False, "head": "linear", "branching": False,
            "lora": {**cfg["model"]["lora"], "enabled": False}}
    render_cfg = {**cfg["data"]["render"], "max_graded": None, "min_graded": None, "eval_max_graded": None,
                  "branching": False}
    ident = {"model": mcfg["name_or_path"], "layers": [int(l) for l in fcfg["layers"]], "variants": int(fcfg["variants"]),
             "seed": int(fcfg["seed"]), "render": render_cfg, "shard_size": int(fcfg["shard_size"]),
             "max_sets": fcfg["max_sets"]}
    fp = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = os.path.join(out_dir, "manifest.json")
    man = None
    if os.path.exists(path) and not fcfg.get("overwrite"):
        man = json.load(open(path))
        if man.get("fingerprint") != fp:
            raise SystemExit(f"{path} was made with a different features/render/model config; use a new "
                             "features.out_dir or features.overwrite=true")
    if man is None:
        man = {"format": 1, "fingerprint": fp, **ident, "hidden_size": None, "dtype": "bfloat16",
               "keys": [layer_key(l) for l in ident["layers"]], "splits": {}}
    model = renderer = None
    d = cfg["data"]
    for split in fcfg["splits"]:
        filters = d["filters"] if split == d["train_split"] else d["eval_filters"]
        sp = man["splits"].setdefault(split, {"shards": [], "complete": False})
        done = {s["name"] for s in sp["shards"]}
        files, locs = select_locations(d, split, filters, (fcfg["max_sets"] or {}).get(split), int(fcfg["seed"]))
        S = int(fcfg["shard_size"])
        n_shards = -(-len(locs) // S)
        sp.update(n_selected=int(len(locs)), n_shards=n_shards)
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(n_shards):
            name = f"shard_{i:05d}"
            if name in done:
                print(f"[features] {split} {name} done, skipping", flush=True)
                continue
            if model is None:
                from .model import build_scrm
                model, tok = build_scrm(mcfg, device, seed=cfg["seed"])
                model.eval()
                renderer = Renderer(tok, render_cfg)
                man["hidden_size"] = model.d_hidden
            t0 = time.time()
            rows = _read_rows(files, [tuple(x) for x in locs[i * S:(i + 1) * S]], EXTRA_COLS)
            feats, records, dropped = extract_rows(model, renderer, rows, fcfg, split, device)
            if records:
                write_shard(split_dir, i, feats, records)
            sp["shards"].append({"name": name, "n_rows": int(next(iter(feats.values())).shape[0]) if records else 0,
                                 "n_sets": len({r["decision_set_id"] for r in records}), "n_records": len(records),
                                 "n_dropped": dropped, "files": bool(records)})
            _write_manifest(out_dir, man)
            print(f"[features] {split} {name}: {len(rows)} rows read, {dropped} dropped, {len(records)} (set, variant) "
                  f"records in {time.time() - t0:.1f}s", flush=True)
        sp["complete"] = {s["name"] for s in sp["shards"]} >= {f"shard_{i:05d}" for i in range(n_shards)}
        sp["shards"].sort(key=lambda s: s["name"])
        sp["n_sets"] = sum(s["n_sets"] for s in sp["shards"])
        sp["n_rows"] = sum(s["n_rows"] for s in sp["shards"])
        sp["n_dropped"] = sum(s["n_dropped"] for s in sp["shards"])
        _write_manifest(out_dir, man)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache frozen-backbone per-option features (scrm.features)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. features.variants=3 features.max_sets.train=1000")
    a = ap.parse_args(argv)
    run(load_config(a.config, a.overrides), a.device)


if __name__ == "__main__":
    main()
