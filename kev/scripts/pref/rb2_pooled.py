"""Pooled (absolute-scale) metrics over a RewardBench 2 output (scripts/pref/rewardbench2.py JSON). numpy only.

    .venv/bin/python scripts/pref/rb2_pooled.py runs/pref/H/seed0/rewardbench2.json [--report runs/pref/G/.../reward_report.json]

RewardBench 2's own score only compares a prompt's candidates with each other. This pools every candidate of every prompt
(per subset and overall; a candidate at position i is correct when i < num_correct) and asks whether the reward is an
absolute scale:
- auroc_pooled: P(a correct candidate of one prompt outscores a wrong candidate of any prompt). Rank-based, so any
  monotone recalibration (Platt with a > 0) leaves it unchanged.
- ece_sigmoid: ECE of sigmoid(score) vs correctness (the raw score read as a logit; the claim of the BCE arms).
- ece_prior: ECE of sigmoid(score + log(nc / (n - nc))), the score read as a log likelihood ratio and shifted by the
  prompt's known prior odds (the balanced-BCE arms' claim). A prompt with nc == n has no wrong candidate: probability 1.
- ece_platt (with --report): ECE of sigmoid(a * score + b), the (a, b) arm G fitted on Decision Index (reward_report.json
  ["G"]); the D/H + Platt baseline.

Unsupported prompts (any candidate unscoreable, rewardbench2.py `unsupported`) are kept in every denominator at their worst
case, as the benchmark's own score does: their correct candidates take the lowest score of the run (probability 0) and
their wrong candidates the highest (probability 1). Writes <json stem>.pooled.json next to the input.
"""
import argparse, json
from pathlib import Path

import numpy as np

from reward_report import auroc, ece, sigmoid


def pool(prompts, ab):
    """-> metrics over the given prompts' candidates, unsupported prompts at their worst case."""
    seen = [s for p in prompts if not any(p["unsupported"]) for s in p["scores"]] or [0.0]
    lo, hi = min(seen), max(seen)
    score, hit, prior_logit, bad = [], [], [], []
    for p in prompts:
        n, nc = len(p["scores"]), p["num_correct"]
        h = [i < nc for i in range(n)]
        worst = any(p["unsupported"])
        score += [(lo if c else hi) if worst else s for s, c in zip(p["scores"], h)]
        hit += h; bad += [worst] * n
        prior_logit += [np.inf if nc >= n else (-np.inf if nc <= 0 else np.log(nc / (n - nc)))] * n
    score, hit, bad, prior_logit = np.asarray(score, float), np.asarray(hit, bool), np.asarray(bad, bool), np.asarray(prior_logit)
    fix = lambda conf: np.where(bad, (~hit).astype(float), conf)       # worst case: correct -> 0, wrong -> 1
    with np.errstate(over="ignore", invalid="ignore"):
        prior = np.where(np.isposinf(prior_logit), 1.0, np.where(np.isneginf(prior_logit), 0.0, sigmoid(score + np.where(np.isfinite(prior_logit), prior_logit, 0.0))))
    out = {"prompts": len(prompts), "candidates": int(len(score)), "positives": int(hit.sum()), "unsupported_prompts": sum(any(p["unsupported"]) for p in prompts),
           "auroc_pooled": auroc(score, hit), "ece_sigmoid": ece(fix(sigmoid(score)), hit), "ece_prior": ece(fix(prior), hit)}
    if ab is not None: out["ece_platt"] = ece(fix(sigmoid(ab[0] * score + ab[1])), hit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="scripts/pref/rewardbench2.py output")
    ap.add_argument("--report", help="reward_report.json with a G section: its Platt (a, b) gives ece_platt")
    a = ap.parse_args()
    src = json.load(open(a.json))
    ab = None
    if a.report:
        g = json.load(open(a.report)).get("G")
        if not g: raise SystemExit(f"{a.report}: no G section (run reward_report.py with --calib)")
        ab = (g["platt_a"], g["platt_b"])
    prompts = src["prompts"]
    subsets = list(dict.fromkeys(p["subset"] for p in prompts))
    res = {"source": a.json, "run": src.get("run"), "platt": None if ab is None else {"report": a.report, "a": ab[0], "b": ab[1]},
           "overall": pool(prompts, ab),
           "no_ties": pool([p for p in prompts if str(p["subset"]).lower() != "ties"], ab),
           "subsets": {s: pool([p for p in prompts if p["subset"] == s], ab) for s in subsets}}
    out = Path(a.json).with_suffix(".pooled.json")
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    cols = ["prompts", "candidates", "unsupported_prompts", "auroc_pooled", "ece_sigmoid", "ece_prior"] + (["ece_platt"] if ab else [])
    print(f"== {a.json}\n{'subset':12}" + "".join(f"{c[:12]:>13}" for c in cols))
    for name, m in [("overall", res["overall"]), ("no_ties", res["no_ties"])] + list(res["subsets"].items()):
        print(f"{name[:12]:12}" + "".join(f"{m[c]:13.4g}" if m[c] is not None else f"{'-':>13}" for c in cols))
    print(f"wrote {out}")


if __name__ == "__main__": main()
