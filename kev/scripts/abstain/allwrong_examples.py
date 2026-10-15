"""Print matched all-wrong minimal pairs from one or more scored runs: per option, the reward r and softmax(r) on the
answerable row and on the same question with the right option removed (allwrong). Shows what maxprob (shift-invariant)
and maxr (the level) do on the same pair.

Usage: scripts/abstain/allwrong_examples.py NAME=RUN_DIR [NAME=RUN_DIR ...] [--suite decision-v7|transfer-v4] [--n 3]
Reads RUN_DIR/eval/allwrong-<suite>/{answerable,allwrong}/rows.json (scripts/abstain/eval_allwrong.sh).
Pairs are the ones every given run scored and the first run answered right; the same pairs are shown for every run.
Picks (first run): pairs where its softmax stays confident on the allwrong row but maxr falls below the threshold that
keeps 95% of its answerable rows, pairs where both readouts are fooled, the median maxr drop, and pairs where maxr rose.
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
    """The answerable row drops one seeded wrong option w, the allwrong row drops the label (build_allwrong.py), so w is
    scored only on the allwrong side: it is listed last, marked +."""
    first = data[runs[0][0]]
    a, b = first["answerable"][key], first["allwrong"][key]
    removed = [k for k in a["keys"] if k not in b["keys"]]
    hidden = [k for k in b["keys"] if k not in a["keys"]]
    print(f"\n### {key[0]} / {key[1]}  task {a.get('task')}  K {len(a['keys']) + 1} (each row K-1)  "
          f"* right option (allwrong drops it): {removed}  + wrong option only in allwrong: {hidden}")
    for name, _ in runs:
        ra, rb = data[name]["answerable"][key], data[name]["allwrong"][key]
        za, zb = np.array(ra["logits"], float), np.array(rb["logits"], float)
        pa, pb = softmax(za), softmax(zb)
        sa, sb = stats(za), stats(zb)
        print(f"  {name}: maxr {sa['maxr']:+.2f} -> {sb['maxr']:+.2f} (d {sb['maxr'] - sa['maxr']:+.2f})   "
              f"maxprob {sa['maxprob']:.3f} -> {sb['maxprob']:.3f}   mass {sa['mass']:.2f} -> {sb['mass']:.2f}")
        bi = {k: i for i, k in enumerate(rb["keys"])}
        for i in np.argsort(-za)[:8]:
            k = ra["keys"][i]
            j = bi.get(k)
            tail = f"r {zb[j]:+7.2f}  p {pb[j]:.3f}" if j is not None else "(dropped)"
            mark = "*" if i == ra["label"] else " "
            print(f"    {mark}{str(k)[:40]:<40} r {za[i]:+7.2f}  p {pa[i]:.3f}   |  {tail}")
        if len(za) > 8: print(f"    ... {len(za) - 8} more options")
        for k in hidden:
            j = bi[k]
            print(f"    +{str(k)[:40]:<40} {'(dropped)':<20}   |  r {zb[j]:+7.2f}  p {pb[j]:.3f}")


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
    # the maxr threshold that keeps 95% of answerable rows: below it the reward flags the allwrong row
    t = float(np.quantile([fa[k]["maxr"] for k in keys], 0.05))
    top_hidden = sum(int(np.argmax(f["allwrong"][k]["logits"])) == f["allwrong"][k]["keys"].index(
        next(x for x in f["allwrong"][k]["keys"] if x not in f["answerable"][k]["keys"])) for k in keys)
    print(f"  {runs[0][0]}: maxr 95%-keep threshold {t:+.2f}; allwrong rows flagged {np.mean([fb[k]['maxr'] < t for k in keys]):.3f}; "
          f"allwrong top option is the + option on {top_hidden}/{len(keys)}")
    picks = {
        f"softmax confident, reward flags it (allwrong maxr < {t:+.2f}, top {args.n} allwrong maxprob, {runs[0][0]})":
            sorted([k for k in keys if fb[k]["maxr"] < t], key=lambda k: -fb[k]["maxprob"])[:args.n],
        f"both fooled (top {args.n} allwrong maxprob, {runs[0][0]})":
            sorted(keys, key=lambda k: -fb[k]["maxprob"])[:args.n],
        f"median maxr drop ({runs[0][0]})": by_drop[max(0, mid - args.n // 2):mid - args.n // 2 + args.n],
        f"maxr did not drop ({runs[0][0]}, largest rise)": [k for k in by_drop[::-1] if drop[k] >= 0][:args.n],
    }
    for title, ks in picks.items():
        print(f"\n## {title}")
        for k in ks: show(k, runs, data)


if __name__ == "__main__":
    main()
