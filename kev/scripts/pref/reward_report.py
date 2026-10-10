"""Reward-style report over a kev.benchmark output dir (rows.json + report.json). numpy only.

    python3 scripts/pref/reward_report.py runs/pref/C/seed0/eval/decision-v7-development [--calib <calibration dir of the same run>]

Per question type and overall (clean, knowable rows; the same population as report.json["clean"]): top-1, pairwise accuracy
(label vs each other option), reward stats (the row's logits are the rewards), softmax NLL and ECE, sigmoid(reward) ECE vs the
one-hot label (absolute calibration: meaningful mainly for D/E/F). With --calib, G: fit temperature T (softmax) and Platt (a, b)
on the calibration rows, apply both to the eval rows, report the calibrated metrics. Writes <dir>/reward_report.json.
"""
import argparse, json
from pathlib import Path
import numpy as np

BINS = 15

def load(d):
    rows = json.load(open(Path(d) / "rows.json"))
    rows = [r for r in rows if r["variant"] == "clean" and r["source"] != "unknowable"]
    if not rows or "logits" not in rows[0]: raise SystemExit(f"{d}: no clean rows with logits")
    return [(np.array(r["logits"], float), int(r["label"]), r["type"]) for r in rows]

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def sigmoid(x): return 0.5 * (1 + np.tanh(0.5 * x))

def ece(conf, hit):
    """Expected calibration error, equal-width bins."""
    conf, hit = np.asarray(conf), np.asarray(hit, float)
    idx = np.minimum((conf * BINS).astype(int), BINS - 1); e = 0.0
    for b in range(BINS):
        m = idx == b
        if m.any(): e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(e)

def metrics(rows, T=1.0, ab=(1.0, 0.0)):
    r_all = np.concatenate([r for r, _, _ in rows])
    top1, pair, nll, conf, hit, gap, s_conf, s_hit = [], [], [], [], [], [], [], []
    for r, y, _ in rows:
        p = softmax(r / T); k = int(np.argmax(r))
        top1.append(k == y); nll.append(-np.log(max(p[y], 1e-12))); conf.append(p.max()); hit.append(k == y)
        oth = np.delete(r, y)
        pair += [1.0 if r[y] > o else 0.5 if r[y] == o else 0.0 for o in oth]
        gap.append(r[y] - oth.mean())
        s_conf += list(sigmoid(ab[0] * r + ab[1])); s_hit += [i == y for i in range(len(r))]
    return {"n": len(rows), "top1": float(np.mean(top1)), "pairwise_acc": float(np.mean(pair)),
            "r_mean": float(r_all.mean()), "r_std": float(r_all.std()), "r_min": float(r_all.min()), "r_max": float(r_all.max()),
            "abs_r_p99": float(np.quantile(np.abs(r_all), .99)), "label_minus_other": float(np.mean(gap)),
            "nll": float(np.mean(nll)), "ece_softmax": ece(conf, hit), "ece_sigmoid": ece(s_conf, s_hit)}

def fit_T(rows):
    """Temperature minimizing softmax NLL: NLL is convex in beta=1/T, so golden-section search on log beta."""
    nll = lambda lb: np.mean([-np.log(max(softmax(np.exp(lb) * r)[y], 1e-12)) for r, y, _ in rows])
    lo, hi, g = -6.0, 6.0, (np.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b): hi = b
        else: lo = a
    return float(np.exp(-(lo + hi) / 2))

def fit_platt(rows, iters=50, ridge=1e-6):
    """Logistic regression of the one-hot label on [r, 1] over every option (Newton). -> (a, b)."""
    x = np.concatenate([r for r, _, _ in rows]); t = np.concatenate([np.eye(len(r))[y] for r, y, _ in rows])
    X = np.stack([x, np.ones_like(x)], 1); q = t.mean(); w = np.array([0.0, np.log(q / (1 - q))])
    loss = lambda w: np.sum(np.logaddexp(0, X @ w) - t * (X @ w)) + ridge * w @ w / 2
    for _ in range(iters):
        p = sigmoid(X @ w); g = X.T @ (p - t) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g); s = 1.0
        while loss(w - s * step) > loss(w) and s > 1e-8: s /= 2   # backtracking: plain Newton overshoots when sigmoid saturates
        w = w - s * step
        if np.abs(s * step).max() < 1e-9: break
    return float(w[0]), float(w[1])

def by_type(rows, **kw):
    out = {"overall": metrics(rows, **kw)}
    for t in sorted({t for _, _, t in rows}): out[t] = metrics([x for x in rows if x[2] == t], **kw)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--calib", help="calibration-split benchmark dir (arm G)")
    a = ap.parse_args()
    rows = load(a.dir); res = {"dir": a.dir, "raw": by_type(rows)}
    rep = Path(a.dir) / "report.json"
    if rep.exists(): res["permutation"] = json.load(open(rep)).get("permutation")   # flip_rate: Choice argmax flips under option permutation
    if a.calib:
        cal = load(a.calib); T = fit_T(cal); ab = fit_platt(cal)
        res["G"] = {"calib_dir": a.calib, "T": T, "platt_a": ab[0], "platt_b": ab[1], "n_calib": len(cal),
                    "temperature": by_type(rows, T=T), "platt": by_type(rows, ab=ab)}   # temperature: softmax metrics; platt: ece_sigmoid
    json.dump(res, open(Path(a.dir) / "reward_report.json", "w"), indent=1)
    cols = ["n", "top1", "pairwise_acc", "r_mean", "r_std", "r_min", "r_max", "abs_r_p99", "label_minus_other", "nll", "ece_softmax", "ece_sigmoid"]
    def show(title, tab):
        print(f"\n{title}\n{'type':9}" + "".join(f"{c[:11]:>12}" for c in cols))
        for t, m in tab.items(): print(f"{t:9}" + "".join(f"{m[c]:12.4g}" for c in cols))
    print(f"== {a.dir}"); show("raw (T=1, no Platt)", res["raw"])
    if res.get("permutation"): print(f"\npermutation flip_rate {res['permutation']['flip_rate']} (n={res['permutation']['n']})")
    if "G" in res:
        g = res["G"]; print(f"\nG: T={g['T']:.4g}  Platt a={g['platt_a']:.4g} b={g['platt_b']:.4g}  (fit on {g['n_calib']} calibration rows)")
        show("G softmax at fitted T (nll, ece_softmax)", g["temperature"]); show("G sigmoid(a*r+b) (ece_sigmoid)", g["platt"])

if __name__ == "__main__": main()
