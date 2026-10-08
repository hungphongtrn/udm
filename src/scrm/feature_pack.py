"""Pack committed feature shards from separate extraction caches into ONE head-training cache (symlinks + manifest).

    python -m scrm.feature_pack --train outputs/features_qwen3_5_4b \
        --validation outputs/features_qwen3_5_4b_vllm_eval --val-shards 8 --out outputs/features_qwen3_5_4b_pack

* train: every committed shard of a COMPLETE train split.
* validation: the first `--val-shards` committed shards (shard_00000 ..). Extraction orders rows round-robin over
  sources, so any shard prefix is source-balanced: a small validation set that still covers every source. The source
  split need not be complete; the prefix must be contiguous.
* Sources may come from different backends (e.g. HF train, vLLM validation): that is recorded, not hidden.

The packed directory holds only symlinks and its own manifest; it is not an extraction output (scrm.features refuses
to resume it: no identity). Re-running rebuilds it; regular files inside it are never deleted."""
from __future__ import annotations

import argparse
import json
import os

PACKED_SPLITS = ("train", "validation")


def _manifest(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        return json.load(f)


def _committed(man: dict, split: str) -> list[dict]:
    sp = (man.get("splits") or {}).get(split)
    if not sp:
        raise SystemExit(f"split {split!r} missing from the source manifest")
    return sorted(sp["shards"], key=lambda s: s["name"])


def _clear_links(split_dir: str) -> None:
    for name in os.listdir(split_dir):
        p = os.path.join(split_dir, name)
        if not os.path.islink(p):
            raise SystemExit(f"{p} is not a symlink: refusing to rebuild a directory that is not a feature pack")
        os.unlink(p)


def pack(train_dir: str, val_dir: str, val_shards: int, out_dir: str) -> dict:
    if val_shards < 1:
        raise SystemExit("--val-shards must be >= 1")
    mt, mv = _manifest(train_dir), _manifest(val_dir)
    if mt.get("hidden_size") != mv.get("hidden_size"):
        raise SystemExit(f"hidden_size differs: train {mt.get('hidden_size')} vs validation {mv.get('hidden_size')}")
    keys = [k for k in mt["keys"] if k in set(mv["keys"])]
    if not keys:
        raise SystemExit(f"no common feature keys: train {mt['keys']} vs validation {mv['keys']}")
    if not mt["splits"].get("train", {}).get("complete"):
        raise SystemExit(f"{train_dir}: train split is incomplete")
    train = _committed(mt, "train")
    val = _committed(mv, "validation")
    want = [f"shard_{i:05d}" for i in range(val_shards)]
    have = {s["name"]: s for s in val}
    missing = [n for n in want if n not in have]
    if missing:
        raise SystemExit(f"{val_dir}: validation prefix needs {want[0]}..{want[-1]} but {missing[:3]} not committed "
                         f"({len(val)} committed); lower --val-shards")
    val = [have[n] for n in want]
    if os.path.exists(os.path.join(out_dir, "manifest.json")) and _manifest(out_dir).get("packed") is None:
        raise SystemExit(f"{out_dir} holds an extraction manifest; choose a new --out")
    plan = {"train": (train_dir, train), "validation": (val_dir, val)}
    splits = {}
    for split, (src, shards) in plan.items():
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        _clear_links(d)
        for s in shards:
            if not s.get("files", s.get("n_records", 0) > 0):
                continue
            for ext in (".parquet", ".safetensors"):
                target = os.path.abspath(os.path.join(src, split, s["name"] + ext))
                if not os.path.exists(target):
                    raise SystemExit(f"committed shard file missing: {target}")
                os.symlink(target, os.path.join(d, s["name"] + ext))
        splits[split] = {"complete": True, "shards": shards, "n_shards": len(shards),
                         "n_sets": sum(int(s.get("n_sets", 0)) for s in shards)}
    sources = {"train": {"dir": os.path.abspath(train_dir), "backend": mt.get("backend", "hf"), "shards": len(train)},
               "validation": {"dir": os.path.abspath(val_dir), "backend": mv.get("backend", "hf"), "shards": len(val)}}
    man = {"format": 2, "packed": sources, "model": mt.get("model"), "layers": mt.get("layers"),
           "hidden_size": mt["hidden_size"], "dtype": mt.get("dtype") or "bfloat16", "keys": keys, "splits": splits}
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="extraction cache holding the complete train split")
    ap.add_argument("--validation", required=True, help="extraction cache holding the validation shards")
    ap.add_argument("--val-shards", type=int, default=8, help="validation shard prefix length (512 sets each)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = pack(a.train, a.validation, a.val_shards, a.out)
    for split, src in man["packed"].items():
        sp = man["splits"][split]
        print(f"[pack] {split}: {sp['n_shards']} shards, {sp['n_sets']} sets <- {src['dir']} ({src['backend']})")
    print(f"[pack] keys {man['keys']}, hidden {man['hidden_size']} -> {a.out}")


if __name__ == "__main__":
    main()
