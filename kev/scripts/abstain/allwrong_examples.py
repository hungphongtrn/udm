"""Print matched all-wrong minimal pairs from one or more scored runs: per option, the reward r and softmax(r) on the
answerable row and on the same question with the right option removed (allwrong). Shows what maxprob (shift-invariant)
and maxr (the level) do on the same pair.

Usage: scripts/abstain/allwrong_examples.py NAME=RUN_DIR [NAME=RUN_DIR ...] [--suite decision-v7|transfer-v4] [--n 3]
Reads RUN_DIR/eval/allwrong-<suite>/{answerable,allwrong}/rows.json (scripts/abstain/eval_allwrong.sh).
Pairs are the ones every given run scored and the first run answered right; the same pairs are shown for every run.
Picks: the n pairs where the first run's allwrong maxprob is highest (softmax cannot flag them), n at the median maxr
drop, n where maxr did not drop at all; plus per-run medians.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(run, suite, cond):
    rows = json.load(open(Path(run) / "eval" / f"allwrong-{suite}" / cond / "rows.json"))
    return {(r["id"], r["question"]): r for r in rows}


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def stats(z):
    return {"maxr": float(z.max()), "maxprob": float(softmax(z).max()), "mass": float((1 / (1 + np.exp(-z))).sum())}


def show(key, runs, data):
    first = data[runs[0][0]]
    a, b = first["answerable"][key], first["allwrong"][key]
    removed = [k for k in a["keys"] if k not in b["keys"]]
    print(f"\n### {key[0]} / {key[1]}  task {a.get('task')}  K {len(a['keys'])}  right option (removed): {removed}")
    for name, _ in runs:
        ra, rb = data[name]["answerable"][key], data[name]["allwrong"][key]
        za, zb = np.array(ra["logits"], float), np.array(rb["logits"], float)
        pa, pb = softmax(za), softmax(zb)
        sa, sb = stats(za), stats(zb)
        print(f"  {name}: maxr {sa['maxr']:+.2f} -> {sb['maxr']:+.2f} (d {sb['maxr'] - sa['maxr']:+.2f})   "
              f"maxprob {sa['maxprob']:.3f} -> {sb['maxprob']:.3f}   mass {sa['mass']:.2f} -> {sb['mass']:.2f}")
        bi = {k: i for i, k in enumerate(rb["keys"])}
        order = np.argsort(-za)[:8]
        for i in order:
            k = ra["keys"][i]
            j = bi.get(k)
            tail = f"r {zb[j]:+7.2f}  p {pb[j]:.3f}" if j is not None else "(removed)"
            mark = "*" if i == ra["label"] else " "
            print(f"    {mark}{str(k)[:40]:<40} r {za[i]:+7.2f}  p {pa[i]:.3f}   |  {tail}")
        if len(za) > 8: print(f"    ... {len(za) - 8} more options")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    ap.add_argument("--suite", default="decision-v7")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    runs = [tuple(s.split("=", 1)) for s in args.runs]
    data = {n: {c: load(d, args.suite, c) for c in ("answerable", "allwrong")} for n, d in runs}
    keys = sorted(set.intersection(*(set(data[n][c]) for n, _ in runs for c in ("answerable", "allwrong"))))
    f = data[runs[0][0]]
    keys = [k for k in keys if int(np.argmax(f["answerable"][k]["logits"])) == f["answerable"][k]["label"]]
    print(f"{len(keys)} pairs every run scored, {runs[0][0]} right on the answerable row")
    for name, _ in runs:
        d = data[name]
        S = {c: [stats(np.array(d[c][k]["logits"], float)) for k in keys] for c in ("answerable", "allwrong")}
        med = lambda c, m: float(np.median([s[m] for s in S[c]]))
        print(f"  {name} medians answerable -> allwrong: maxr {med('answerable', 'maxr'):+.2f} -> {med('allwrong', 'maxr'):+.2f}  "
              f"maxprob {med('answerable', 'maxprob'):.3f} -> {med('allwrong', 'maxprob'):.3f}  "
              f"mass {med('answerable', 'mass'):.2f} -> {med('allwrong', 'mass'):.2f}")
    fa = {k: stats(np.array(f["answerable"][k]["logits"], float)) for k in keys}
    fb = {k: stats(np.array(f["allwrong"][k]["logits"], float)) for k in keys}
    drop = {k: fb[k]["maxr"] - fa[k]["maxr"] for k in keys}
    by_drop = sorted(keys, key=drop.get)
    mid = len(by_drop) // 2
    picks = {
        f"softmax still confident on allwrong (top {args.n} allwrong maxprob, {runs[0][0]})":
            sorted(keys, key=lambda k: -fb[k]["maxprob"])[:args.n],
        f"median maxr drop ({runs[0][0]})": by_drop[max(0, mid - args.n // 2):mid - args.n // 2 + args.n],
        f"maxr did not drop ({runs[0][0]}, largest rise)": [k for k in by_drop[::-1] if drop[k] >= 0][:args.n],
    }
    for title, ks in picks.items():
        print(f"\n## {title}")
        for k in ks: show(k, runs, data)


if __name__ == "__main__":
    main()
