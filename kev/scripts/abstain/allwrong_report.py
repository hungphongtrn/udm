"""All-wrong detection and risk-coverage across arms (numpy only). Reads what eval_allwrong.sh wrote.

    .venv/bin/python scripts/abstain/allwrong_report.py --arm A=runs/pref/A/seed0 --arm C=runs/pref/C/seed0 \
        --arm F=runs/pref/F/seed0 --out runs/allwrong/report-full
    (half-data screen: --arm Ah=runs/pref/Ah/seed0 --arm Rh=... ; scripts/abstain/queue.sh prints both commands)

For each suite in RUN/eval/allwrong-<suite>/<condition>/rows.json (build_allwrong.py's four conditions) and each arm, the
pairs every arm scored in every condition (the join is on (id, question); the counts of pairs lost to rejected records
are reported). Readouts of a row's rewards r (the logits rows.json records):

    maxprob     max softmax(r)                        shift-invariant: what a softmax-trained cascade thresholds
    maxprob_G   max softmax(r / T), T = arm's G       (T from RUN/eval/decision-v7-development/reward_report.json)
    maxr        max r                                 the absolute level; sigmoid(max r) and Platt sigmoid(a max r + b)
                                                      (a > 0) rank identically, so every number below is the same for them
    lse         logsumexp(r)                          the energy score: the level read through every option
    pnone       softmax(r)[none]  (none conditions)   the product's own mechanism: offer a none option, read its mass
    pnone_G     softmax(r / T)[none]

Per (suite, arm, readout): AUROC of separating allwrong (positive) from answerable by low confidence (pnone: high),
FPR@95TPR (the fraction of answerable rows flagged at the threshold that flags 95% of allwrong rows), the paired win
rate (answerable more confident than its own allwrong twin), and the risk-coverage curve on the pool answerable +
allwrong (one of each per question; an allwrong answer is always wrong): AURC, selective accuracy at fixed coverages,
and the oracle AURC that arm's answerable top1 allows. AURC mixes detection with top1; AUROC is detection only.
pnone has no curve (its pool would answer "none" correctly), only AUROC and the two top1s. AUROC is also reported per
option-count stratum (the question's original K = answerable's K - 1 options + 1: small K <= 4, mid 5-14, large >= 15,
i.e. banking77 on decision-v7), so a readout whose level drifts with K shows up as a stratum gap.
Pointwise consistency: per arm, the largest |reward difference| of an option shared by a pair's answerable and allwrong
rows (the same option text scored in two sets). A set-independent grader (chat_pointwise / chat_yesno) is 0 up to kernel
rounding; a listing prompt is not.
Paired differences (--compare X.r-Y.r, default: the hypothesis doc's decision list for the arms given) are bootstrapped
over pairs (2,000 resamples, seed 0; question sampling only). Each arm's development top1, softmax NLL and sigmoid ECE on
decision-v7 and transfer-v4 (reward_report.json raw.overall) is printed beside it. Threshold transfer (D6 / gate 3): per
arm, curve readout and target risk in --risk (0.05,0.10), the lowest threshold whose decision-v7 pool risk is <= the
target (whole tie blocks), applied unchanged to the transfer-v4 pool and to each K stratum of both pools: coverage, risk,
the fractions of answerable and of allwrong rows answered, and the coverage a threshold fit on transfer-v4 itself would
reach (the ceiling). Writes <out>/report.json and <out>/rc-<suite>.svg.
"""
import argparse, json
from pathlib import Path
import numpy as np

CONDITIONS = ("answerable", "allwrong", "answerable_none", "allwrong_none")
CURVE_READOUTS = ("maxprob", "maxprob_G", "maxr", "lse")
NONE_READOUTS = ("pnone", "pnone_G")
COVERAGES = (0.3, 0.5, 0.7, 0.9, 1.0)
K_STRATA = (("small", 0, 4), ("mid", 5, 14), ("large", 15, 10 ** 9))   # original option count, inclusive bounds
# The hypothesis doc's decisions (arm names cannot contain "-"; compares naming an arm not given are skipped).
DEFAULT_COMPARE = (
    # D1: readout alone on the full-data arms (no training)
    "A.maxr-A.maxprob", "A.lse-A.maxprob", "C.maxr-C.maxprob", "F.maxr-F.maxprob", "F.maxr-A.maxprob", "F.maxr-A.pnone",
    # half-data screen. D2: RCR (linear head) vs pairwise C / C -> F and vs Kev's own readout
    "Ah.maxr-Ah.maxprob", "Rh.maxr-Fh.maxr", "Rh.maxr-Ch.maxr", "Rh.maxr-Ah.maxprob",
    # D3: pointwise-then-pairwise stage vs RCR in one stage; D4: MLP vs linear head under RCR; listwise term (alpha 0.5 vs 0)
    "STh.maxr-R2h.maxr", "STh.maxr-Fh.maxr", "R2h.maxr-Rh.maxr", "R2h.maxr-Sh.maxr",
    # D5: RCR vs class-balanced BCE on the yes/no prompt, and yes/no vs the listing prompt
    "RYh.maxr-HBh.maxr", "RYh.maxr-Rh.maxr",
    # D7: every anchored arm's level vs offering a none option to Kev
    "Rh.maxr-Ah.pnone", "R2h.maxr-Ah.pnone", "Sh.maxr-Ah.pnone", "STh.maxr-Ah.pnone", "RYh.maxr-Ah.pnone", "Fh.maxr-Ah.pnone",
    # PF-RCR wave. D8 topology: the listing out of the prompt (BT + BSR both)
    "Hh.maxr-Dh.maxr", "Hh.maxprob-Dh.maxprob",
    # D9 objective on the pointwise prompt: + listwise term, K balance, RCR vs BT, vs RCR on the listing prompt
    "PRh.maxr-PBh.maxr", "PRh.maxr-PUh.maxr", "PRh.maxr-Hh.maxr", "PRh.maxr-Rh.maxr",
    # D10 head: MLP and LM-head init under the same loss
    "P2h.maxr-PRh.maxr", "PLh.maxr-PRh.maxr",
    # D11 against Kev's own readouts
    "PRh.maxr-Ah.maxprob", "PRh.maxr-Ah.pnone", "PBh.maxr-Ah.maxprob",
    # P2h leave-one-out (docs/hypotheses/todo/pointwise-rcr-mlp.md): listing back in, BT + BSR for RCR, linear head
    "P2h.maxr-R2Bh.maxr", "P2h.maxr-P2Bh.maxr", "P2h.maxr-Dh.maxr")
FIT_SUITE, APPLY_SUITE = "decision-v7", "transfer-v4"   # D6: threshold fit on the first, applied unchanged to the second
B, SEED = 2000, 0


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def lse(z):
    m = z.max(); return float(m + np.log(np.exp(z - m).sum()))


def avg_ranks(x):
    """1-based ranks, ties averaged."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt); return ((end - cnt + 1 + end) / 2)[inv]


def auroc(pos, neg):
    """P(a positive scores above a negative), ties half."""
    rk = avg_ranks(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(pos, neg, tpr=0.95):
    """Flag s >= t with the highest t that flags at least `tpr` of the positives (ties flagged together) -> the fraction
    of negatives flagged."""
    t = np.sort(pos)[int(np.floor((1 - tpr) * len(pos)))]
    return float(np.mean(neg >= t))


def risk_coverage(conf, hit):
    """Answer in order of decreasing confidence. Ties answer as a block: each member gets the block's mean hit (random
    tie-breaking in expectation). -> accuracy among the first i answered, i = 1..n."""
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    order = np.argsort(-conf, kind="stable"); c, h = conf[order], hit[order]
    _, start, cnt = np.unique(-c, return_index=True, return_counts=True)
    tie = cnt > 1
    for s, n in zip(start[tie], cnt[tie]): h[s:s + n] = h[s:s + n].mean()
    return np.cumsum(h) / np.arange(1, len(h) + 1)


def fit_threshold(conf, hit, risk):
    """Lowest threshold t (answer iff conf >= t) whose answered set has risk (1 - accuracy) <= `risk`. A threshold
    answers whole tie blocks, so only block ends are candidates. -> t, or inf (answer nothing) if no block end qualifies."""
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    order = np.argsort(-conf, kind="stable"); c, h = conf[order], hit[order]
    end = np.flatnonzero(np.append(c[1:] != c[:-1], True))   # last index of each tie block
    ok = end[1 - np.cumsum(h)[end] / (end + 1) <= risk + 1e-12]
    return float(c[ok[-1]]) if len(ok) else float("inf")


def apply_threshold(conf, hit, t, n_ans):
    """The first n_ans rows of the pool are answerable, the rest allwrong (pool()). -> coverage, risk and answer rates."""
    on = np.asarray(conf, float) >= t
    return {"threshold": t, "coverage": float(on.mean()), "risk": float(1 - hit[on].mean()) if on.any() else 0.0,
            "answered_answerable": float(on[:n_ans].mean()), "answered_allwrong": float(on[n_ans:].mean())}


def rc_metrics(acc):
    n = len(acc)
    return {"aurc": float(np.mean(1 - acc)), **{f"acc@{c:g}": float(acc[int(np.ceil(c * n)) - 1]) for c in COVERAGES}}


def load_rows(d):
    rows = json.load(open(Path(d) / "rows.json"))
    if rows and "logits" not in rows[0]: raise SystemExit(f"{d}: rows without logits")
    return {(r["id"], r["question"]): r for r in rows}


def load_T(run):
    p = Path(run) / "eval" / "decision-v7-development" / "reward_report.json"
    if not p.exists(): return None
    g = json.load(open(p)).get("G"); return g["T"] if g else None


def load_dev(run):
    """Development top1, softmax NLL and sigmoid ECE per suite (scripts/pref/reward_report.py raw.overall); a suite is
    None before eval_arm.sh scored it."""
    out = {}
    for suite in ("decision-v7", "transfer-v4"):
        p = Path(run) / "eval" / f"{suite}-development" / "reward_report.json"
        if not p.exists(): out[suite] = None; continue
        o = json.load(open(p))["raw"]["overall"]; out[suite] = {"top1": o["top1"], "nll": o["nll"], "ece_sigmoid": o.get("ece_sigmoid")}
    return out


def stratum(k):
    return next(name for name, lo, hi in K_STRATA if lo <= k <= hi)


def readouts(run, suite):
    """-> (keys present in every condition, {readout: {"ans": arr, "bad": arr}},
    {"ans_hit", "ansnone_hit", "badnone_hit", "K" (original option count), "shared_diff" (max |reward difference| of the
    options the pair's answerable and allwrong rows share)}, T)."""
    base = Path(run) / "eval" / f"allwrong-{suite}"
    rows = {c: load_rows(base / c) for c in CONDITIONS}
    keys = sorted(set.intersection(*(set(v) for v in rows.values())))
    T = load_T(run)
    r = {c: [np.array(rows[c][k]["logits"], float) for k in keys] for c in CONDITIONS}
    none_idx = [rows["allwrong_none"][k]["label"] for k in keys]
    if any(rows["answerable_none"][k]["keys"][i] != rows["allwrong_none"][k]["keys"][i] for k, i in zip(keys, none_idx)):
        raise SystemExit(f"{base}: the none option is not at the same position in the two none conditions")
    out = {}
    for name, f in (("maxprob", lambda z: softmax(z).max()), ("maxr", lambda z: z.max()), ("lse", lse)):
        out[name] = {"ans": np.array([f(z) for z in r["answerable"]]), "bad": np.array([f(z) for z in r["allwrong"]])}
    if T is not None:
        out["maxprob_G"] = {"ans": np.array([softmax(z / T).max() for z in r["answerable"]]),
                            "bad": np.array([softmax(z / T).max() for z in r["allwrong"]])}
    for name, t in (("pnone", 1.0), ("pnone_G", T)):
        if t is None: continue
        out[name] = {"ans": np.array([softmax(z / t)[i] for z, i in zip(r["answerable_none"], none_idx)]),
                     "bad": np.array([softmax(z / t)[i] for z, i in zip(r["allwrong_none"], none_idx)])}
    def shared_diff(k, za, zb):
        ka, kb = rows["answerable"][k]["keys"], rows["allwrong"][k]["keys"]
        d = [abs(za[ka.index(o)] - zb[kb.index(o)]) for o in set(ka) & set(kb)]
        return max(d) if d else np.nan
    hits = {"ans_hit": np.array([int(np.argmax(z)) == rows["answerable"][k]["label"] for z, k in zip(r["answerable"], keys)], float),
            "ansnone_hit": np.array([int(np.argmax(z)) == rows["answerable_none"][k]["label"] for z, k in zip(r["answerable_none"], keys)], float),
            "badnone_hit": np.array([int(np.argmax(z)) == i for z, i in zip(r["allwrong_none"], none_idx)], float),
            "K": np.array([len(rows["answerable"][k]["keys"]) + 1 for k in keys]),
            "shared_diff": np.array([shared_diff(k, za, zb) for k, za, zb in zip(keys, r["answerable"], r["allwrong"])], float)}
    return keys, out, hits, T


def detection(ro, name):
    """AUROC with allwrong positive: low confidence flags it, except pnone where high mass on none does."""
    s = 1 if name in NONE_READOUTS else -1
    return auroc(s * ro["bad"], s * ro["ans"])


def fpr95(ro, name):
    s = 1 if name in NONE_READOUTS else -1
    return fpr_at_tpr(s * ro["bad"], s * ro["ans"])


def win_rate(ro, name):
    a, b = (ro["bad"], ro["ans"]) if name in NONE_READOUTS else (ro["ans"], ro["bad"])
    return float(np.mean((a > b) + 0.5 * (a == b)))


def pool(ro, ans_hit):
    return np.concatenate([ro["ans"], ro["bad"]]), np.concatenate([ans_hit, np.zeros_like(ans_hit)])


def boot_compare(X, Y, rng):
    """X, Y: (readouts, readout name, hits) restricted to the same pairs. -> point and 95% CI of dAUROC and dAURC (X - Y)."""
    (rx, nx, hx), (ry, ny, hy) = X, Y
    n = len(hx); idx_all = np.arange(n)
    def stats(idx):
        sx = {k: v[idx] for k, v in rx[nx].items()}; sy = {k: v[idx] for k, v in ry[ny].items()}
        d_auc = detection(sx, nx) - detection(sy, ny)
        curves = nx not in NONE_READOUTS and ny not in NONE_READOUTS
        d_aurc = (np.mean(1 - risk_coverage(*pool(sx, hx[idx]))) - np.mean(1 - risk_coverage(*pool(sy, hy[idx])))) if curves else np.nan
        return d_auc, d_aurc
    point = stats(idx_all)
    bs = np.array([stats(rng.integers(0, n, n)) for _ in range(B)])
    ci = lambda col: [float(np.quantile(col, .025)), float(np.quantile(col, .975))] if not np.isnan(col).all() else None
    return {"d_auroc": float(point[0]), "d_auroc_ci": ci(bs[:, 0]),
            "d_aurc": None if np.isnan(point[1]) else float(point[1]), "d_aurc_ci": ci(bs[:, 1])}


COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f")


def svg(curves, title, path):
    """Selective accuracy vs coverage; one colour per arm, solid = maxr, dashed = maxprob."""
    W, H, L, R, T, Bm = 640, 420, 60, 160, 30, 50
    x = lambda c: L + c * (W - L - R); y = lambda a: T + (1 - a) * (H - T - Bm)
    arms = sorted({a for a, _ in curves}); parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>', f'<text x="{L}" y="18" font-size="14">{title}</text>']
    for v in np.linspace(0, 1, 6):
        parts += [f'<line x1="{x(v):.1f}" y1="{y(0):.1f}" x2="{x(v):.1f}" y2="{y(1):.1f}" stroke="#ddd"/>',
                  f'<line x1="{x(0):.1f}" y1="{y(v):.1f}" x2="{x(1):.1f}" y2="{y(v):.1f}" stroke="#ddd"/>',
                  f'<text x="{x(v):.1f}" y="{y(0) + 16:.1f}" text-anchor="middle">{v:.1f}</text>',
                  f'<text x="{x(0) - 6:.1f}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>']
    parts += [f'<text x="{(x(0) + x(1)) / 2:.1f}" y="{H - 12}" text-anchor="middle">coverage (fraction answered)</text>',
              f'<text transform="translate(16 {(y(0) + y(1)) / 2:.1f}) rotate(-90)" text-anchor="middle">accuracy of answered</text>']
    for i, ((arm, name), acc) in enumerate(sorted(curves.items())):
        col = COLORS[arms.index(arm) % len(COLORS)]; n = len(acc)
        pts = " ".join(f"{x((j + 1) / n):.1f},{y(acc[j]):.1f}" for j in range(0, n, max(1, n // 200)))
        dash = ' stroke-dasharray="6 4"' if name.startswith("maxprob") else ""
        parts += [f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"{dash}/>',
                  f'<line x1="{W - R + 10}" y1="{T + 14 * i + 10}" x2="{W - R + 34}" y2="{T + 14 * i + 10}" stroke="{col}" stroke-width="1.6"{dash}/>',
                  f'<text x="{W - R + 40}" y="{T + 14 * i + 14}">{arm} {name}</text>']
    Path(path).write_text("\n".join(parts + ["</svg>"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=RUN_DIR (repeat)")
    ap.add_argument("--compare", action="append", help="X.readout-Y.readout (repeat; default: the hypothesis doc's list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk", default="0.05,0.10", help=f"comma-separated target risks on the {FIT_SUITE} pool (answerable + allwrong) for threshold transfer")
    a = ap.parse_args()
    arms = dict(s.split("=", 1) for s in a.arm)
    suites = sorted({p.name[len("allwrong-"):] for run in arms.values() for p in (Path(run) / "eval").glob("allwrong-*")})
    if not suites: raise SystemExit("no RUN/eval/allwrong-* dirs: run eval_allwrong.sh first")
    compares = a.compare or [c for c in DEFAULT_COMPARE if {c.split("-")[0].split(".")[0], c.split("-")[1].split(".")[0]} <= set(arms)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    risks = [float(x) for x in a.risk.split(",") if x]
    rng = np.random.default_rng(SEED); res = {"arms": arms, "dev": {name: load_dev(run) for name, run in arms.items()}, "suites": {}}
    print("development (reward_report.json raw.overall): top1 / softmax NLL / sigmoid ECE")
    show = lambda d: "not evaluated" if d is None else f"{d['top1']:.4f} / {d['nll']:.4f} / " + ("n/a" if d["ece_sigmoid"] is None else f"{d['ece_sigmoid']:.4f}")
    for name, d in res["dev"].items(): print(f"  {name:5} decision-v7 {show(d['decision-v7'])}   transfer-v4 {show(d['transfer-v4'])}")
    pools = {}   # (suite, arm, readout) -> (conf, hit, n answerable, original K per pool row) on that suite's common pairs
    for suite in suites:
        per = {name: readouts(run, suite) for name, run in arms.items()}
        common = sorted(set.intersection(*(set(k) for k, _, _, _ in per.values())))
        sel = {}
        for name, (keys, ro, hits, T) in per.items():
            pos = {k: i for i, k in enumerate(keys)}; idx = np.array([pos[k] for k in common])
            sel[name] = ({r: {s: v[idx] for s, v in d.items()} for r, d in ro.items()}, {h: v[idx] for h, v in hits.items()}, T, len(keys))
        S = res["suites"][suite] = {"pairs": len(common), "arms": {}, "compare": {}}
        curves = {}
        print(f"\n== {suite}: {len(common)} pairs scored by every arm")
        Ks = next(iter(sel.values()))[1]["K"]
        strata = [s for s, _, _ in K_STRATA if any(stratum(k) == s for k in Ks)]
        print("strata (original K): " + ", ".join(f"{s} {sum(stratum(k) == s for k in Ks)}" for s in strata))
        print(f"{'arm':5}{'readout':11}{'auroc':>8}{'fpr95':>8}{'win':>8}{'aurc':>8}{'oracle':>8}" + "".join(f"{'acc@' + format(c, 'g'):>9}" for c in COVERAGES)
              + "".join(f"{'auc_' + s:>11}" for s in strata))
        for name, (ro, hits, T, n_own) in sel.items():
            ah = hits["ans_hit"]; m = len(ah)
            oracle = rc_metrics(np.cumsum(np.sort(np.concatenate([ah, np.zeros(m)]))[::-1]) / np.arange(1, 2 * m + 1))["aurc"]
            sd = hits["shared_diff"]; sd = sd[~np.isnan(sd)]
            A = S["arms"][name] = {"T": T, "pairs_own": n_own, "top1_answerable": float(ah.mean()),
                                   "top1_answerable_none": float(hits["ansnone_hit"].mean()),
                                   "top1_allwrong_none": float(hits["badnone_hit"].mean()), "aurc_oracle": oracle,
                                   "top1_answerable_strata": {s: float(ah[[stratum(k) == s for k in hits["K"]]].mean()) for s in strata},
                                   "shared_option_diff": {"max": float(sd.max()) if len(sd) else None, "mean": float(sd.mean()) if len(sd) else None},
                                   "readouts": {}}
            print(f"{name}: top1 answerable {ah.mean():.4f}  with none offered: answerable {hits['ansnone_hit'].mean():.4f}, "
                  f"allwrong picks none {hits['badnone_hit'].mean():.4f}  (T={T})  shared-option |dr| max "
                  + ("n/a" if not len(sd) else f"{sd.max():.2e} mean {sd.mean():.2e}"))
            for r, d in ro.items():
                m_ = {"auroc": detection(d, r), "fpr95": fpr95(d, r), "win": win_rate(d, r), "auroc_strata": {}}
                for s in strata:
                    mk = np.array([stratum(k) == s for k in hits["K"]])
                    m_["auroc_strata"][s] = detection({x: v[mk] for x, v in d.items()}, r)
                if r in CURVE_READOUTS:
                    conf, hit = pool(d, ah); acc = risk_coverage(conf, hit); m_.update(rc_metrics(acc)); curves[(name, r)] = acc
                    pools[(suite, name, r)] = (conf, hit, len(ah), np.concatenate([hits["K"], hits["K"]]))
                A["readouts"][r] = m_
                print(f"{'':5}{r:11}{m_['auroc']:8.4f}{m_['fpr95']:8.4f}{m_['win']:8.4f}"
                      + (f"{m_['aurc']:8.4f}{oracle:8.4f}" + "".join(f"{m_['acc@' + format(c, 'g')]:9.4f}" for c in COVERAGES) if "aurc" in m_ else " " * (16 + 9 * len(COVERAGES)))
                      + "".join(f"{m_['auroc_strata'][s]:11.4f}" for s in strata))
        for c in compares:
            (xa, xr), (ya, yr) = (s.split(".") for s in c.split("-"))
            if xr not in sel[xa][0] or yr not in sel[ya][0]:
                print(f"skip {c}: readout missing (no G temperature?)"); continue
            S["compare"][c] = d = boot_compare((sel[xa][0], xr, sel[xa][1]["ans_hit"]), (sel[ya][0], yr, sel[ya][1]["ans_hit"]), rng)
            fmt = lambda v, ci: "n/a" if v is None else f"{v:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            print(f"{c:22} dAUROC {fmt(d['d_auroc'], d['d_auroc_ci'])}   dAURC {fmt(d['d_aurc'], d['d_aurc_ci'])}")
        svg({k: v for k, v in curves.items() if k[1] in ("maxprob", "maxr")}, f"{suite}: answerable + all-wrong pool", out / f"rc-{suite}.svg")
    if {FIT_SUITE, APPLY_SUITE} <= set(suites):
        # D6 / gate 3: a threshold fit for each target risk on one suite's whole pool, applied unchanged to the other's pool
        # and to each K stratum of both; "oracle" = the coverage the same readout reaches on the apply suite with a threshold
        # fit there (the best that readout's ranking allows)
        res["threshold_transfer"] = T6 = {"fit": FIT_SUITE, "apply": APPLY_SUITE, "risks": {}}
        def stratum_pool(c, h, n, K, s):   # keeps the pool's order: answerable rows first
            mk = np.array([stratum(k) == s for k in K]); return c[mk], h[mk], int(mk[:n].sum())
        for risk in risks:
            R = T6["risks"][f"{risk:g}"] = {}
            print(f"\n== threshold transfer: fit on {FIT_SUITE} at risk <= {risk:g}, applied to {APPLY_SUITE} and to each K stratum")
            print(f"{'arm':5}{'readout':11}{'fit_cov':>9}{'cov':>8}{'risk':>8}{'ans_ok':>8}{'ans_bad':>8}{'oracle':>8}   per stratum (suite:stratum cov/risk)")
            for name in arms:
                for r in CURVE_READOUTS:
                    if (FIT_SUITE, name, r) not in pools: continue
                    fc, fh, fn, fK = pools[(FIT_SUITE, name, r)]; tc, th, tn, tK = pools[(APPLY_SUITE, name, r)]
                    t = fit_threshold(fc, fh, risk)
                    m_ = {"fit": apply_threshold(fc, fh, t, fn), "apply": apply_threshold(tc, th, t, tn),
                          "apply_oracle_coverage": apply_threshold(tc, th, fit_threshold(tc, th, risk), tn)["coverage"], "strata": {}}
                    for su, (c, h, n, K) in ((FIT_SUITE, (fc, fh, fn, fK)), (APPLY_SUITE, (tc, th, tn, tK))):
                        for s, _, _ in K_STRATA:
                            sc, sh, sn = stratum_pool(c, h, n, K, s)
                            if sn: m_["strata"][f"{su}:{s}"] = apply_threshold(sc, sh, t, sn)
                    R.setdefault(name, {})[r] = m_; p = m_["apply"]
                    print(f"{name:5}{r:11}{m_['fit']['coverage']:9.4f}{p['coverage']:8.4f}{p['risk']:8.4f}{p['answered_answerable']:8.4f}"
                          f"{p['answered_allwrong']:8.4f}{m_['apply_oracle_coverage']:8.4f}   "
                          + "  ".join(f"{k} {v['coverage']:.3f}/{v['risk']:.3f}" for k, v in m_["strata"].items()))
    json.dump(res, open(out / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
