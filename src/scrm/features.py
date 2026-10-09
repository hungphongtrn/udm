"""Frozen-backbone feature cache: run the backbone once over the data (same "Grade this choice: Option k" prompt as
training; prefill-only vLLM pooling or the HF prefix-cache reference path) and store one feature vector per
(set, variant, option), so heads / losses can be trained on the cached features.

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

MANIFEST_FORMAT = 2
LOCK_FILE = ".lock"                                # per-out_dir flock file (created on first run, never deleted)
# identity of the vLLM backend: pinned engine + the exact feature contract; both go into the resume fingerprint.
VLLM_ENGINE_VERSION = "0.19.1"
VLLM_FEATURE_CONTRACT = "last-input-final-norm-v1"


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
def embed_items_hf(model, items, fcfg: dict, device, pad_id: int) -> dict[int, torch.Tensor]:
    """HF backend: {layer: [graded options of all items, d] bf16 CPU}, item order then suffix order (as the vLLM
    extractor's `embed_items`). Each set's shared prompt is encoded once and its options continue from copies of that
    prompt's cache (`embed_prefix_cached`), so cost ~ unique tokens, not prompt x options."""
    from .collator import collate
    from .packing import pack_bfd
    layers = [int(l) for l in fcfg["layers"]]
    start, n = [], 0
    for it in items:
        start.append(n)
        n += len(it.suffixes)
    out = {l: torch.empty(n, model.d_hidden, dtype=torch.bfloat16) for l in layers}
    buckets = {}
    for i, it in enumerate(items):
        buckets.setdefault(len(it.prefix), []).append(i)
    amp = device.type == "cuda"
    for P, idx in buckets.items():
        for sel in pack_bfd([P + sum(len(s) for s in items[i].suffixes) for i in idx],
                            int(fcfg["max_tokens"]), int(fcfg["max_batch_size"])):
            batch = [idx[j] for j in sel]
            b = collate([items[i] for i in batch], pad_id=pad_id)
            # Scheduling metadata stays on CPU: embed_prefix_cached reads it as Python lists.
            # Moving it to CUDA only to call .tolist() forces an avoidable GPU synchronization.
            for key in ("input_ids", "position_ids"):
                b["pack"][key] = b["pack"][key].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                e = model.embed_prefix_cached(b["pack"], layers=layers, cache_tokens=int(fcfg["cache_tokens"]),
                                              prefill_tokens=int(fcfg["max_tokens"]),
                                              max_batch_size=int(fcfg["max_batch_size"]))
            s = 0
            for i in batch:
                k = len(items[i].suffixes)
                for l in layers:
                    out[l][start[i]:start[i] + k] = e[l][s:s + k].to(torch.bfloat16).cpu()
                s += k
    return out


@torch.no_grad()
def extract_rows(model, renderer: Renderer, rows: list[dict], fcfg: dict, split: str, device) -> tuple[dict, list, int]:
    """Featurize rows -> ({layer: [M, d] bf16 tensors}, parquet records, number of dropped rows)."""
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
    items = [it for _, its, _ in jobs for it in its]
    if not items:
        return feats, records, dropped
    if fcfg.get("backend", "hf") == "vllm":
        # The engine schedules variable lengths; requests remain independent causal sequences.
        e = model.embed_items(items)
    else:
        e = embed_items_hf(model, items, fcfg, device, renderer.pad_id)
    s = 0
    for it in items:
        o, perm = where[id(it)]
        n = len(perm)
        dest = o + torch.as_tensor(perm)           # slot p was presented at position p = canonical option perm[p]
        for l in layers:
            feats[l][dest] = e[l][s:s + n]
        s += n
    return feats, records, dropped


def layer_key(l: int) -> str:
    return "feat_Llast" if int(l) == -1 else f"feat_L{int(l)}"


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    import errno
    try:
        fd = os.open(path or ".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return                                             # directory missing / not openable: nothing to flush
    try:
        os.fsync(fd)
    except OSError as e:                                   # EINVAL/ENOTSUP/EROFS/ENOSYS: dir fsync unsupported
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EROFS, errno.ENOSYS):
            raise
    finally:
        os.close(fd)


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _atomic(path: str, write) -> None:
    """Write `path` through a `.tmp` sibling and commit with rename: fsync the file, rename, fsync the directory.
    A crash (SIGKILL/OOM) leaves either the old file or the new one, never a torn write; a failed write drops the tmp."""
    tmp = path + ".tmp"
    try:
        write(tmp)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_dir(os.path.dirname(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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


_SAFETENSORS_DTYPE = {"bfloat16": "BF16", "float16": "F16", "float32": "F32", "float64": "F64"}


def _file_identities(files: list[str]) -> list:
    """Local file revisions and immutable Hub blob IDs, not mutable remote path names alone."""
    out, fs = [], None
    for f in files:
        if f.startswith("hf://"):
            if fs is None:
                import fsspec
                fs = fsspec.filesystem("hf")
            info = fs.info(f[len("hf://"):], refresh=True, expand_info=True)
            blob = info.get("blob_id")
            if not blob:
                raise RuntimeError(f"Cannot establish immutable source identity for {f}; use a local data snapshot")
            out.append([f, info["size"], blob])
        else:
            st = os.stat(f)
            out.append([f, st.st_size, st.st_mtime_ns])
    return out


def _selection_digest(files: list[str], locs: np.ndarray) -> str:
    """Digest of WHICH rows a split featurizes: source file identities + the exact (file_idx, row_idx) locations. Two
    runs share it only when they select the same rows of the same file revisions, so a changed selection (even one
    with the same row/shard counts) cannot silently resume the previous shards."""
    h = hashlib.sha256()
    h.update(json.dumps(_file_identities(files), sort_keys=True).encode())
    h.update(np.ascontiguousarray(locs, dtype=np.int64).tobytes())
    return h.hexdigest()[:32]


def _purge_shards(split_dir: str, kept: set[str] | None = None) -> list[str]:
    """Remove shard files whose stem is not in `kept` (all when `kept` is None) plus stray `.tmp` siblings; returns
    the removed names. Committed shards are protected by passing `kept`; uncommitted leftovers from a crash and a
    stale cache being overwritten are discarded here so they cannot be read as current."""
    if not os.path.isdir(split_dir):
        return []
    removed = []
    for name in sorted(os.listdir(split_dir)):
        stem, ext = os.path.splitext(name)
        stale = ext in (".safetensors", ".parquet") and stem.startswith("shard_") and (kept is None or stem not in kept)
        if name.endswith(".tmp") or stale:
            path = os.path.join(split_dir, name)
            if os.path.isfile(path):
                os.unlink(path)
                removed.append(name)
    return removed


def _verify_shard(split_dir: str, rec: dict, keys: list[str], hidden, dtype: str) -> None:
    """A shard listed as committed must still match the manifest: silently recomputing it would shift the row offsets
    of later shards, so a missing or corrupt file is a hard error rather than a re-read."""
    if not rec.get("files", rec.get("n_records", 0) > 0):
        return                                              # zero-record shard: nothing was ever written
    name = rec["name"]
    n_rows, n_records = int(rec["n_rows"]), int(rec["n_records"])

    def bad(reason: str):
        raise SystemExit(f"committed feature shard {name} {reason}; restore it, remove {name!r} from the manifest's "
                         "shards list to recompute it, or rerun with features.overwrite=true")

    st_path = os.path.join(split_dir, name + ".safetensors")
    pq_path = os.path.join(split_dir, name + ".parquet")
    for p in (st_path, pq_path):
        if not os.path.isfile(p) or os.path.getsize(p) <= 0:
            bad(f"{os.path.basename(p)} is missing or unreadable")
    expected = _SAFETENSORS_DTYPE.get(dtype)
    from safetensors import safe_open
    try:
        with safe_open(st_path, framework="pt") as f:
            got = list(f.keys())
            if sorted(got) != sorted(keys):
                bad(f"safetensors keys {got} != {keys}")
            for k in keys:
                sl = f.get_slice(k)
                if hidden is not None and tuple(sl.get_shape()) != (n_rows, int(hidden)):
                    bad(f"{k} has shape {tuple(sl.get_shape())}, expected {(n_rows, int(hidden))}")
                if expected is not None and sl.get_dtype() != expected:
                    bad(f"{k} has dtype {sl.get_dtype()}, expected {expected}")
    except SystemExit:
        raise
    except Exception as e:
        bad(f"has an unreadable safetensors file ({e!r})")
    import pyarrow.parquet as pq
    try:
        pf = pq.ParquetFile(pq_path)
        if pf.metadata.num_rows != n_records:
            bad(f"parquet has {pf.metadata.num_rows} rows, manifest says {n_records}")
        if n_records:
            t = pq.read_table(pq_path, columns=["row_offset", "n_options"])
            off = t.column("row_offset").to_numpy(zero_copy_only=False).astype(np.int64)
            nopt = t.column("n_options").to_numpy(zero_copy_only=False).astype(np.int64)
            if (nopt <= 0).any() or off[0] != 0 or not np.array_equal(off[1:], np.cumsum(nopt)[:-1]) \
                    or int(off[-1] + nopt[-1]) != n_rows:
                bad("row_offset/n_options do not contiguously cover the feature rows")
    except SystemExit:
        raise
    except Exception as e:
        bad(f"has an unreadable parquet file ({e!r})")


# ----------------------------------------------------------------------------- driver
def build_feature_extractor(cfg: dict, device):
    backend = cfg["features"].get("backend", "hf")
    if backend == "vllm":
        from .vllm_features import VLLMFeatureExtractor
        model = VLLMFeatureExtractor(cfg, device)
        return model, model.tokenizer
    if backend != "hf":
        raise ValueError(f"features.backend must be 'hf' or 'vllm', got {backend!r}")
    from .model import build_scrm
    mcfg = {**cfg["model"], "freeze_backbone": True, "gradient_checkpointing": False, "head": "linear",
            "branching": False, "lora": {**cfg["model"]["lora"], "enabled": False}}
    model, tok = build_scrm(mcfg, device, seed=cfg["seed"])
    return model.eval(), tok


def _write_manifest(out_dir: str, man: dict) -> None:
    _atomic(os.path.join(out_dir, "manifest.json"), lambda p: _write_bytes(p, json.dumps(man, indent=2).encode()))


def _fingerprint(ident: dict) -> str:
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _identities(cfg: dict, backend: str) -> tuple[dict, dict, dict]:
    """(base, full, render) identity. `base` is the pre-selection-fingerprint identity that legacy HF manifests were
    written with (compared on resume); `full` adds the backend, the vLLM engine/feature contract, the model numerics
    (dtype / quantization / attention / liger) and the data-selection fields that decide which rows a shard holds, so a
    changed input selection or backbone precision cannot silently resume wrong shards.
    Batch-only knobs (max_tokens / max_batch_size / cache_tokens) are in neither: they can be retuned without
    invalidating committed shards."""
    fcfg = cfg["features"]
    mcfg = {**cfg["model"], "freeze_backbone": True, "gradient_checkpointing": False, "head": "linear",
            "branching": False, "lora": {**cfg["model"]["lora"], "enabled": False}}
    render_cfg = {**cfg["data"]["render"], "max_graded": None, "min_graded": None, "eval_max_graded": None,
                  "branching": False}
    base = {"model": mcfg["name_or_path"], "layers": [int(l) for l in fcfg["layers"]], "variants": int(fcfg["variants"]),
            "seed": int(fcfg["seed"]), "render": render_cfg, "shard_size": int(fcfg["shard_size"]),
            "max_sets": fcfg["max_sets"]}
    d = cfg["data"]
    mo = cfg["model"]
    full = {**base, "backend": backend,
            "model_opts": {"dtype": mo.get("dtype"), "quantize_4bit": bool(mo.get("quantize_4bit")),
                           "attn_implementation": mo.get("attn_implementation"),
                           "liger_kernel": bool(mo.get("liger_kernel"))},
            "data": {k: d.get(k) for k in ("local_dir", "repo", "train_split", "eval_split", "filters", "eval_filters")}}
    if backend == "vllm":
        full["engine_version"] = VLLM_ENGINE_VERSION
        full["feature_contract"] = VLLM_FEATURE_CONTRACT
    return base, full, render_cfg


class _OutputLock:
    """Exclusive advisory lock on the output directory: one writer per out_dir, held for the whole run. A second
    process fails fast instead of racing the manifest and shard files."""

    def __init__(self, out_dir: str):
        self.path = os.path.join(out_dir, LOCK_FILE)
        self._fh = None

    def __enter__(self):
        import fcntl
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            raise SystemExit(f"{self.path} is locked by another features run; wait for it to finish or use a new "
                             "features.out_dir") from None
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False


class _StopController:
    """Cooperative stop: the first SIGINT/SIGTERM asks the driver to commit the current shard and stop cleanly; a
    repeated signal aborts immediately (only uncommitted shards are lost, they are recomputed on the next run)."""

    def __init__(self):
        self.count = 0
        self.requested = False
        self.forced = False

    def request(self):
        self.count += 1
        if self.count > 1:
            self.forced = True
            raise KeyboardInterrupt
        self.requested = True
        print("[features] stop requested: committing the current shard, then stopping (rerun the same command to "
              "resume; a second signal aborts immediately)", flush=True)

    def handle(self, signum, frame):
        self.request()


def _install_signal_handlers(ctrl: _StopController) -> list:
    import signal
    prev = []
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            old = signal.getsignal(sig)
            signal.signal(sig, ctrl.handle)
        except ValueError:                                  # not the main thread
            continue
        prev.append((sig, old))
    return prev


def _finish_split(sp: dict, n_shards: int) -> None:
    sp["complete"] = {s["name"] for s in sp["shards"]} >= {f"shard_{i:05d}" for i in range(n_shards)}
    sp["shards"].sort(key=lambda s: s["name"])
    sp["n_sets"] = sum(s["n_sets"] for s in sp["shards"])
    sp["n_rows"] = sum(s["n_rows"] for s in sp["shards"])
    sp["n_dropped"] = sum(s["n_dropped"] for s in sp["shards"])


def _caps_lowered(old: dict | None, new: dict | None) -> bool:
    """Every split's max_sets equal or lower (None = all rows)."""
    old, new = old or {}, new or {}
    return all(old.get(s) == new.get(s) or (new.get(s) is not None and (old.get(s) is None or new[s] <= old[s]))
               for s in set(old) | set(new))


def _lower_caps(cfg: dict, man: dict, old_caps: dict | None) -> None:
    """Resume with a lowered features.max_sets. `select_locations` truncates one fixed seeded round-robin order, so the
    new selection is a prefix of the old one and every full committed shard inside that prefix holds exactly the rows
    the new cap selects. Checks the old selection against its recorded digest and the prefix, then forgets shards
    beyond it (their files are purged when the split is next processed). Mutates `man` only; caller commits it."""
    fcfg, d = cfg["features"], cfg["data"]
    S, seed, new_caps = int(fcfg["shard_size"]), int(fcfg["seed"]), fcfg["max_sets"] or {}
    for split, sp in man["splits"].items():
        o, n = (old_caps or {}).get(split), new_caps.get(split)
        if o == n:
            continue
        filters = d["filters"] if split == d["train_split"] else d["eval_filters"]
        files, old_locs = select_locations(d, split, filters, o, seed)
        if sp.get("selection") != _selection_digest(files, old_locs):
            raise SystemExit(f"{split}: cannot lower max_sets: the recorded selection no longer matches the data; "
                             "use a new features.out_dir")
        _, locs = select_locations(d, split, filters, n, seed)
        if not np.array_equal(locs, old_locs[:len(locs)]):
            raise SystemExit(f"{split}: lowered max_sets did not select a prefix of the recorded rows; use a new "
                             "features.out_dir")
        same = len(locs) == len(old_locs)
        before = len(sp["shards"])
        sp["shards"] = [s for s in sp["shards"] if same or (int(s["name"][len("shard_"):]) + 1) * S <= len(locs)]
        n_shards = -(-len(locs) // S)
        sp.update(n_selected=int(len(locs)), n_shards=n_shards, selection=_selection_digest(files, locs))
        _finish_split(sp, n_shards)
        print(f"[features] {split}: max_sets {o} -> {n}: kept {len(sp['shards'])}/{before} committed shard(s) "
              f"({len(locs)} rows, {n_shards} shards, complete={sp['complete']})", flush=True)


def run(cfg: dict, device=None, *, stop=None) -> dict:
    """Drive the feature cache. `stop` is a test seam: pass a `_StopController` to exercise the cooperative stop
    without signals (a real run installs SIGINT/SIGTERM handlers and uses its own controller)."""
    fcfg = cfg["features"]
    out_dir = fcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ctrl = stop if stop is not None else _StopController()
    prev = [] if stop is not None else _install_signal_handlers(ctrl)
    try:
        with _OutputLock(out_dir):
            return _run_locked(cfg, device, out_dir, ctrl)
    finally:
        if prev:
            import signal
            for sig, handler in prev:
                signal.signal(sig, handler)


def _run_locked(cfg: dict, device, out_dir: str, ctrl: _StopController) -> dict:
    from .data import _read_rows
    fcfg = cfg["features"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backend = str(fcfg.get("backend") or "hf")
    base, full, render_cfg = _identities(cfg, backend)
    fp, fp_legacy = _fingerprint(full), _fingerprint(base)
    overwrite = bool(fcfg.get("overwrite"))
    path = os.path.join(out_dir, "manifest.json")
    old = json.load(open(path)) if os.path.exists(path) else None
    man, fresh = None, old is None or overwrite
    if fresh:
        man = {"format": MANIFEST_FORMAT, "fingerprint": fp, "legacy_fingerprint": fp_legacy, "backend": backend,
               "identity": full, **base, "hidden_size": None, "dtype": "bfloat16",
               "keys": [layer_key(l) for l in base["layers"]], "splits": {}}
        if old is not None:
            for split in sorted(set(old.get("splits") or {}) - set(fcfg["splits"])):
                gone = _purge_shards(os.path.join(out_dir, split))     # dropped split: stale cache cutover
                if gone:
                    print(f"[features] overwrite: removed {len(gone)} stale shard file(s) of dropped split "
                          f"{split!r}", flush=True)
    else:
        if old.get("fingerprint") == fp:
            print(f"[features] resuming {out_dir} (backend={backend})", flush=True)
        elif backend == "hf" and old.get("backend", "hf") == "hf" and old.get("fingerprint") == fp_legacy:
            if not fcfg.get("resume_legacy_hf"):
                raise SystemExit(f"{path} is a legacy HF manifest that did not record the input selection (no "
                                 "files/rows digest); set features.resume_legacy_hf=true to resume it as-is, or use a "
                                 "new features.out_dir")
            print(f"[features] resuming legacy HF cache {out_dir} (opt-in; its selection was not recorded)", flush=True)
        else:
            old_ident = old.get("identity") or {k: old.get(k) for k in base}
            changed = sorted(k for k in set(old_ident) | set(full)
                             if json.dumps(old_ident.get(k), sort_keys=True, default=str)
                             != json.dumps(full.get(k), sort_keys=True, default=str))
            if changed == ["max_sets"] and old.get("identity") and _caps_lowered(old_ident["max_sets"], full["max_sets"]):
                _lower_caps(cfg, old, old_ident["max_sets"])
            else:
                detail = "; ".join(f"{k}: {old_ident.get(k)!r} -> {full.get(k)!r}" for k in changed) or "fingerprint only"
                raise SystemExit(f"{path} was made with a different features/render/model/data-selection config "
                                 f"({detail}); use a new features.out_dir or features.overwrite=true")
        man = old
        man["format"] = max(int(man.get("format", 1)), MANIFEST_FORMAT)
        man["fingerprint"] = fp
        man["legacy_fingerprint"] = fp_legacy
        man["backend"] = backend
        man["identity"] = full
        man.update(base)
        man["keys"] = [layer_key(l) for l in base["layers"]]
        man["splits"] = man.get("splits") or {}
    # Commit the manifest before the first shard: an interruption right after startup still pins the config, so a
    # later run with a changed selection cannot silently resume this (empty) cache.
    _write_manifest(out_dir, man)
    renderer = model = None
    d = cfg["data"]
    for split in fcfg["splits"]:
        if ctrl.requested:
            break
        filters = d["filters"] if split == d["train_split"] else d["eval_filters"]
        files, locs = select_locations(d, split, filters, (fcfg["max_sets"] or {}).get(split), int(fcfg["seed"]))
        S = int(fcfg["shard_size"])
        n_shards = -(-len(locs) // S)
        sp = man["splits"].setdefault(split, {"shards": [], "complete": False})
        if sp.get("n_selected") is not None and int(sp["n_selected"]) != len(locs):
            raise SystemExit(f"{split}: manifest recorded {sp['n_selected']} selected rows but the current data "
                             f"config selects {len(locs)}; the input selection changed — use a new features.out_dir")
        if sp.get("n_shards") is not None and int(sp["n_shards"]) != n_shards:
            raise SystemExit(f"{split}: manifest recorded {sp['n_shards']} shards but the current config makes "
                             f"{n_shards}; the input selection changed — use a new features.out_dir")
        sp.update(n_selected=int(len(locs)), n_shards=n_shards)
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        sel = _selection_digest(files, locs)
        if sp.get("selection") is not None and sp["selection"] != sel:
            raise SystemExit(f"{split}: the selected files/rows changed since the manifest was written "
                             f"({sp['selection'][:12]} != {sel[:12]}); use a new features.out_dir")
        sp["selection"] = sel
        # Record the per-split selection identity and discard crash leftovers before producing any shard, so a stop
        # right after startup still pins what this split must contain.
        done = {s["name"] for s in sp["shards"]}
        _purge_shards(split_dir, kept=done)                     # uncommitted pairs are recomputed, never reused
        for rec in sp["shards"]:
            _verify_shard(split_dir, rec, man["keys"], man.get("hidden_size"), man.get("dtype") or "bfloat16")
        _write_manifest(out_dir, man)
        for i in range(n_shards):
            if ctrl.requested:
                break
            name = f"shard_{i:05d}"
            if name in done:
                print(f"[features] {split} {name} done, skipping", flush=True)
                continue
            if model is None:
                model, tok = build_feature_extractor(cfg, device)
                if hasattr(model, "eval"):
                    model.eval()
                renderer = Renderer(tok, render_cfg)
                man["hidden_size"] = getattr(model, "d_hidden", None)
                _write_manifest(out_dir, man)
            t0 = time.time()
            rows = _read_rows(files, [tuple(x) for x in locs[i * S:(i + 1) * S]], EXTRA_COLS)
            feats, records, dropped = extract_rows(model, renderer, rows, fcfg, split, device)
            if records:
                write_shard(split_dir, i, feats, records)       # fsynced before it is listed as committed
            sp["shards"].append({"name": name, "n_rows": int(next(iter(feats.values())).shape[0]) if records else 0,
                                 "n_sets": len({r["decision_set_id"] for r in records}), "n_records": len(records),
                                 "n_dropped": dropped, "files": bool(records)})
            done.add(name)
            _write_manifest(out_dir, man)
            print(f"[features] {split} {name}: {len(rows)} rows read, {dropped} dropped, {len(records)} (set, variant) "
                  f"records in {time.time() - t0:.1f}s", flush=True)
            if ctrl.requested:
                print(f"[features] stopped cleanly after committing {split}/{name}; rerun the same command to resume",
                      flush=True)
                break
        _finish_split(sp, n_shards)
        _write_manifest(out_dir, man)
        if ctrl.requested:
            break
    man["stopped"] = bool(ctrl.requested)
    if ctrl.requested:
        n_done = sum(len(s.get("shards", [])) for s in man["splits"].values())
        print(f"[features] stop: {n_done} shard(s) committed, unfinished splits have complete=false; rerun the same "
              "command to resume", flush=True)
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
