"""Full Decision Index (edition 0.2.1) for frozen-backbone heads (scrm.lossgrid): the backbone runs ONCE over the whole
suite, every head is then scored on those features.

    python -m scrm.dindex_frozen featurize --config configs/features_qwen3_5_4b.yaml --out DIR [overrides ...]
    python -m scrm.dindex_frozen score --feats DIR --heads <lossgrid output_dir>/selected.json --out OUT

featurize: each typed question becomes one decision set rendered exactly like the training data and like
scrm.dindex (question_example: `choice` = the option descriptions, `noul` = "no", "yes"; every option graded, nothing
truncated). A request with a question longer than data.render.max_len, or of an unsupported type, is recorded as
`unsupported` (counted as wrong by the index). Features of every option, in option-key order, go into resumable shards
of `--shard-requests` requests: shard_XXXXX.safetensors (`feat_L{layer}`, [rows, d] bf16) + shard_XXXXX.json (index).
The backend (vLLM by default, as in the features config) and its scheduling knobs come from the features config.

score: per head, rewards over each question's options -> softmax -> the kit's answer format (scrm.dindex._answer),
written as a kit results.jsonl and scored with the kit's own edition scorers (decision_index.pipeline.score_run) into
<OUT>/<run>/scores.json; then a per-arm table (mean ± std over seeds) and the board ranks in <OUT>/summary.md."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .dindex import _answer, compare_with_board, question_example

FEATS_FORMAT = "scrm-dindex-feats-1"
DEFAULT_DI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "decision-index")


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _suite(suite_dir: str, edition: str):
    from decision_index.suite.io import Suite
    return Suite(suite_dir, edition)


# ----------------------------------------------------------------------------- featurize
def _render_cfg(cfg: dict) -> dict:
    """The feature cache's rendering (scrm.features._identities), every option shown and graded (scrm.dindex)."""
    return {**cfg["data"]["render"], "max_graded": None, "min_graded": None, "eval_max_graded": None,
            "branching": False, "max_candidates": 10**6}


def render_request(renderer, row: dict):
    """Row -> ([(qkey, keys, qtype, item)], None) or (None, error) when the request cannot be answered."""
    out = []
    for qk, q in row["questions"].items():
        try:
            keys, ex = question_example(row["state"], q)
        except ValueError as e:
            return None, str(e)
        it = renderer.assemble(renderer.tokenize(ex), None, False, relax=True)
        if it is None:
            return None, f"prompt + option exceeds max_len={renderer.cfg['max_len']} tokens"
        if len(it.order) != len(keys):
            return None, f"{len(it.order)} of {len(keys)} options graded"
        out.append((qk, keys, q["type"], it))
    return out, None


def featurize(cfg: dict, out_dir: str, suite_dir: str, edition: str, shard_requests: int, limit: int | None = None):
    from safetensors.torch import save_file
    from .features import build_feature_extractor, embed_items_hf, layer_key
    from .render import Renderer
    fcfg = cfg["features"]
    # hf: each question's prompt is encoded once and its options branch off it (cost ~ unique tokens). vllm: one
    # request per option; its block-aligned prefix cache recomputes most of the shared prompt per option.
    backend = fcfg.get("backend", "hf")
    layers = [int(l) for l in fcfg["layers"]]
    suite = _suite(suite_dir, edition)
    render_cfg = _render_cfg(cfg)
    identity = {"model": cfg["model"]["name_or_path"], "backend": backend, "layers": layers,
                "render": render_cfg, "edition": edition, "rows_sha256": suite.edition.get("rows_sha256"),
                "added_sha256": suite.edition.get("added_sha256"), "shard_requests": int(shard_requests),
                "limit": limit}           # a --limit run is a separate (timing) cache: its last shard is short
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        old = man.get("identity") or {}
        if old != identity:
            changed = sorted(k for k in set(identity) | set(old) if old.get(k) != identity.get(k))
            raise SystemExit(f"{mpath} was made with a different config ({changed}); use a new --out")
    else:
        man = {"format": FEATS_FORMAT, "identity": identity, "keys": [layer_key(l) for l in layers],
               "hidden_size": None, "shards": {}, "complete": False}
        _atomic_json(mpath, man)
    model = renderer = prev = None
    rows = suite.rows(apply_exclusions=True)
    n_req, shard = 0, 0
    while True:
        chunk = []
        for r in rows:
            chunk.append(r)
            if len(chunk) == shard_requests:
                break
        if limit is not None:
            chunk = chunk[:max(0, limit - n_req)]
        if not chunk:
            break
        name = f"shard_{shard:05d}"
        n_req += len(chunk)
        shard += 1
        if name in man["shards"]:
            continue
        if model is None:
            dev = torch.device("cuda")
            model, tok = build_feature_extractor(cfg, dev)
            renderer = Renderer(tok, render_cfg)
            man["hidden_size"] = int(model.d_hidden)
        t0 = time.time()
        questions, unsupported, items, off = [], [], [], 0
        for r in chunk:
            rid = r["_evaluation"]["run_id"]
            qs, err = render_request(renderer, r)
            if qs is None:
                unsupported.append({"run_id": rid, "error": err})
                continue
            for qk, keys, qtype, it in qs:
                questions.append({"run_id": rid, "q": qk, "type": qtype, "keys": keys, "offset": off, "n": len(keys)})
                items.append(it)
                off += len(keys)
        t_render = time.time() - t0
        feats = {l: torch.empty(off, man["hidden_size"], dtype=torch.bfloat16) for l in layers}
        if items:
            e = (model.embed_items(items) if backend == "vllm"
                 else embed_items_hf(model, items, fcfg, dev, renderer.pad_id))
            s = 0
            for q, it in zip(questions, items):
                dest = q["offset"] + torch.as_tensor(it.order)      # suffix j is option order[j] (key order)
                for l in layers:
                    feats[l][dest] = e[l][s:s + q["n"]]
                s += q["n"]
        tmp = os.path.join(out_dir, name + ".safetensors.tmp")
        save_file({layer_key(l): x.contiguous() for l, x in feats.items()}, tmp)
        os.replace(tmp, os.path.join(out_dir, name + ".safetensors"))
        _atomic_json(os.path.join(out_dir, name + ".json"), {"questions": questions, "unsupported": unsupported,
                                                              "n_requests": len(chunk)})
        sec = time.time() - t0
        n_tok = sum(len(it.prefix) * len(it.suffixes) + sum(len(x) for x in it.suffixes) for it in items)
        uniq = sum(len(it.prefix) + sum(len(x) for x in it.suffixes) for it in items)   # perfect prefix reuse
        man["shards"][name] = {"n_requests": len(chunk), "n_rows": off, "n_unsupported": len(unsupported),
                               "seconds": sec, "render_seconds": t_render, "tokens": n_tok, "unique_tokens": uniq}
        cache = model.prefix_cache_stats() if backend == "vllm" else None
        hit = ""
        if cache:
            q = cache.get("prefix_cache_queries", 0) - (prev or {}).get("prefix_cache_queries", 0)
            h = cache.get("prefix_cache_hits", 0) - (prev or {}).get("prefix_cache_hits", 0)
            man["shards"][name]["prefix_cache"] = {"queries": q, "hits": h}
            hit = f", prefix-cache hit {h / max(q, 1):.1%} (ideal {1 - uniq / max(n_tok, 1):.1%})"
            prev = cache
        _atomic_json(mpath, man)              # the shard counts as done only once listed here
        print(f"[dindex-feats] {name}: {len(chunk)} requests, {len(questions)} questions, {off} option rows, "
              f"{len(unsupported)} unsupported in {sec:.1f}s (render {t_render:.1f}s, {off / max(sec, 1e-9):.0f} "
              f"rows/s, {n_tok / max(sec, 1e-9):.0f} submitted tok/s, {uniq / max(sec, 1e-9):.0f} unique tok/s{hit})",
              flush=True)
    if limit is None:
        man["complete"] = True
        man["n_requests"] = n_req
        _atomic_json(mpath, man)
    tot = sum(s["seconds"] for s in man["shards"].values())
    print(f"[dindex-feats] {len(man['shards'])} shards, {n_req} requests seen, featurize time {tot / 3600:.2f} h, "
          f"complete={man['complete']}", flush=True)


# ----------------------------------------------------------------------------- score
def _heads(spec: list[str]) -> list[str]:
    paths = []
    for p in spec:
        if p.endswith(".json"):
            sel = json.load(open(p))
            paths += [h for arm in sel.values() for h in arm["heads"]]
        else:
            paths.append(p)
    return paths


@torch.no_grad()
def _rewards(head, x: torch.Tensor, questions: list[dict], device, batch: int = 256) -> list[np.ndarray]:
    out: list = [None] * len(questions)
    idx = sorted(range(len(questions)), key=lambda i: questions[i]["n"])   # similar sizes -> little padding
    for b in range(0, len(idx), batch):
        part = idx[b:b + batch]
        N = max(questions[i]["n"] for i in part)
        xb = torch.zeros(len(part), N, x.shape[1], device=device)
        mask = torch.zeros(len(part), N, dtype=torch.bool, device=device)
        for j, i in enumerate(part):
            q = questions[i]
            xb[j, :q["n"]] = x[q["offset"]:q["offset"] + q["n"]].to(device).float()
            mask[j, :q["n"]] = True
        r = head(xb, mask).float().cpu().numpy()
        for j, i in enumerate(part):
            out[i] = r[j, :questions[i]["n"]]
    return out


def score(feats_dir: str, head_specs: list[str], out_dir: str, suite_dir: str, edition: str, board: str | None):
    from safetensors import safe_open
    from decision_index.pipeline import score_run
    from .dindex import summarize
    from .feat_train import load_head
    man = json.load(open(os.path.join(feats_dir, "manifest.json")))
    if not man.get("complete"):
        print(f"[dindex-frozen] WARNING: {feats_dir} is incomplete; scores will have complete=false", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {}
    for p in _heads(head_specs):
        model, blob = load_head(p, device)
        if blob["feature_key"] not in man["keys"] or blob["d"] != man["hidden_size"]:
            raise SystemExit(f"{p}: needs {blob['feature_key']} d={blob['d']}, features have {man['keys']} "
                             f"d={man['hidden_size']}")
        heads[os.path.basename(p)[:-len(".pt")]] = (model, blob)
    if not heads:
        raise SystemExit("no heads to score")
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name in heads:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        files[name] = open(os.path.join(out_dir, name, "results.jsonl"), "w")
    suite = _suite(suite_dir, edition)
    evals = {r["_evaluation"]["run_id"]: r for r in suite.rows(apply_exclusions=True)}
    t0, head_sec = time.time(), {n: 0.0 for n in heads}
    keys_needed = sorted({b["feature_key"] for _, b in heads.values()})
    for shard in sorted(man["shards"]):
        idx = json.load(open(os.path.join(feats_dir, shard + ".json")))
        ms = 1000 * man["shards"][shard]["seconds"] / max(idx["n_requests"], 1)     # amortised backbone time
        with safe_open(os.path.join(feats_dir, shard + ".safetensors"), "pt") as f:
            x = {k: f.get_tensor(k) for k in keys_needed}
        by_req: dict = {}
        for i, q in enumerate(idx["questions"]):
            by_req.setdefault(q["run_id"], []).append(i)
        for name, (model, blob) in heads.items():
            th = time.time()
            rew = _rewards(model, x[blob["feature_key"]], idx["questions"], device)
            hms = ms + 1000 * (time.time() - th) / max(idx["n_requests"], 1)
            head_sec[name] += time.time() - th
            lines = []
            for rid, qi in by_req.items():
                row = evals[rid]
                ans = {idx["questions"][i]["q"]: _answer(idx["questions"][i]["keys"],
                                                         row["questions"][idx["questions"][i]["q"]], rew[i])
                       for i in qi}
                lines.append({**row["_evaluation"], "engine": "scrm-frozen", "status": "ok",
                              "response": {"model": f"scrm-frozen:{name}", "answers": ans},
                              "total_wall_ms": hms, "model_request_wall_ms": hms})
            for u in idx["unsupported"]:
                lines.append({**evals[u["run_id"]]["_evaluation"], "engine": "scrm-frozen", "status": "unsupported",
                              "error": u["error"], "total_wall_ms": hms, "model_request_wall_ms": hms})
            files[name].write("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines))
    for fh in files.values():
        fh.close()
    rows = []
    for name, (_, blob) in heads.items():
        d = os.path.join(out_dir, name)
        s = score_run(suite, os.path.join(d, "results.jsonl"), f"scrm-frozen:{name}", d)
        m = summarize(s, len(evals), head_sec[name])
        rows.append({"run": name, "arm": "+".join(blob["terms"]), "seed": blob["seed"], "lr": blob["lr"],
                     "best_step": blob["best_step"], "index": m["index"], "raw_index": m["raw_index"],
                     "answered_frac": m["answered_frac"], "complete": s["complete"],
                     **{k: v for k, v in m.items() if k.startswith("area/")}})
        print(f"[dindex-frozen] {name}: index {m['index']:.2f} raw {m['raw_index']:.2f} "
              f"answered {m['answered_frac']:.3f}", flush=True)
    _report(rows, out_dir, board, time.time() - t0)


def _report(rows: list[dict], out_dir: str, board: str | None, seconds: float) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    g = df.groupby("arm").agg(index_mean=("index", "mean"), index_std=("index", "std"), raw_mean=("raw_index", "mean"),
                              n_seeds=("index", "size"), answered=("answered_frac", "mean")).reset_index()
    g = g.sort_values("index_mean", ascending=False)
    g.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    md = ["# Decision Index 0.2.1 (full suite), frozen backbone + head\n",
          f"Scoring time {seconds / 60:.1f} min; complete={bool(df['complete'].all())}.\n",
          "| arm | index | raw index | seeds | answered | board rank |", "|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        rank = "-"
        if board and os.path.exists(board):
            rank = next(k for k, n, _ in compare_with_board(r["index_mean"], board, top=10**6) if n == ">> SCRM <<")
        md.append(f"| {r['arm']} | {r['index_mean']:.2f}±{0 if np.isnan(r['index_std']) else r['index_std']:.2f} | "
                  f"{r['raw_mean']:.2f} | {r['n_seeds']} | {r['answered']:.3f} | {rank} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def main(argv=None):
    from .config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("featurize")
    f.add_argument("--config", required=True, help="features config (backbone, render, layers, vLLM knobs)")
    f.add_argument("--out", required=True)
    f.add_argument("--shard-requests", type=int, default=1024)
    f.add_argument("--limit", type=int, default=None, help="first N requests only (timing / smoke)")
    f.add_argument("overrides", nargs="*", help="dotted overrides, e.g. 'features.layers=[24]'")
    s = sub.add_parser("score")
    s.add_argument("--feats", required=True)
    s.add_argument("--heads", required=True, nargs="+", help="selected.json and/or head .pt files")
    s.add_argument("--out", required=True)
    s.add_argument("--board", default=None, help="default: <di-dir>/tests/fixtures/board-0.2.1.json")
    for p in (f, s):
        p.add_argument("--di-dir", default=os.environ.get("DI_DIR", DEFAULT_DI_DIR))
        p.add_argument("--edition", default="0.2.1")
    a = ap.parse_args(argv)
    suite_dir = os.path.join(a.di_dir, "suite-0.2")
    if a.cmd == "featurize":
        featurize(load_config(a.config, a.overrides), a.out, suite_dir, a.edition, a.shard_requests, a.limit)
    else:
        score(a.feats, a.heads, a.out, suite_dir, a.edition,
              a.board or os.path.join(a.di_dir, "tests", "fixtures", "board-0.2.1.json"))


if __name__ == "__main__":
    main()
