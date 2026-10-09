"""Loss-function grid on frozen cached features: every non-empty subset of {ce, brier, bt, sigmoid} x seeds x a
hyperparameter grid that has the SAME size for every arm (python -m scrm.lossgrid --config configs/lossgrid_*.yaml).

Protocol (matched tuning budget): all arms get the identical grid `grid.lr x grid.weight`, the same step count, data order
and init per seed. In a multi-term arm every term has weight 1.0 (x the grid's global `weight` multiplier), so there is no
per-term tuning that would favour bigger arms. Every (arm, lr, weight, seed) run keeps its best-validation-loss
checkpoint (train.eval_every) and saves that head under <output_dir>/heads/. Per arm the (lr, weight) with the best
mean-over-seeds validation selection metric is chosen; selected.json lists its heads (one per seed) for the test
benchmark, the full Decision Index on live backbone features (scrm.dindex_frozen). Resumable: finished runs are skipped."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from .config import deep_update, _load_yaml, parse_overrides
from .feat_train import FEAT_DEFAULTS, TERMS, run_experiment

GRID_DEFAULTS = {
    "output_dir": "outputs/lossgrid",
    "workers": 1,
    "grid": {"terms": list(TERMS), "lr": [3e-4, 1e-3], "weight": [1.0], "seeds": [0, 1, 2]},
    "selection": {"metric": "loss/total", "mode": "min"},       # on validation; "<group>/<metric>"
    "summary_metrics": ["all/top1", "all/pair_acc", "all/mrr", "all/ndcg", "all/kendall_tau", "all/ece_top1",
                        "all/brier_softmax", "all/nll_softmax", "all/sig_ece", "all/sig_brier",
                        "all/abst_sigmoid_drop", "all/abst_softmax_drop", "all/flip_rate", "all/mean_abs_dp"],
}


def load_grid_config(path: str | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(FEAT_DEFAULTS)
    deep_update(cfg, copy.deepcopy(GRID_DEFAULTS))
    if path:
        deep_update(cfg, _load_yaml(path))
    if overrides:
        deep_update(cfg, parse_overrides(overrides))
    return cfg


def arms(terms=TERMS) -> list[tuple]:
    """All non-empty subsets, ordered by size then lexicographically in `terms` order."""
    return [c for k in range(1, len(terms) + 1) for c in itertools.combinations(terms, k)]


def arm_name(terms) -> str:
    return "+".join(terms)


def run_id(terms, lr, weight, seed) -> str:
    return f"{arm_name(terms)}__lr{lr:g}__w{weight:g}__s{seed}"


def make_tasks(cfg: dict) -> list[dict]:
    """Task = full run config + result path + head path, for the whole grid."""
    g = cfg["grid"]
    tasks = []
    for terms in arms(g["terms"]):
        for lr, w, seed in itertools.product(g["lr"], g["weight"], g["seeds"]):
            c = copy.deepcopy(cfg)
            c["seed"], c["train"]["lr"] = int(seed), float(lr)
            c["loss"]["terms"], c["loss"]["weight"] = list(terms), float(w)
            rid = run_id(terms, lr, w, seed)
            tasks.append({"cfg": c, "evals": [cfg["data"]["val_split"]], "arm": arm_name(terms), "weight": float(w),
                          "path": os.path.join(cfg["output_dir"], "runs", rid + ".json"),
                          "head_path": os.path.join(cfg["output_dir"], "heads", rid + ".pt")})
    return tasks


def _done(task: dict) -> bool:
    return os.path.exists(task["path"]) and os.path.exists(task["head_path"])


def run_task(task: dict) -> str:
    if _done(task):
        return task["path"]
    rid = os.path.basename(task["path"])[:-len(".json")]
    print(f"[{rid}] start", flush=True)
    res = run_experiment(task["cfg"], task["evals"], log=lambda *a: print(f"[{rid}]", *a, flush=True),
                         head_path=task["head_path"])
    res.update(arm=task["arm"], weight=task["weight"])
    os.makedirs(os.path.dirname(task["path"]), exist_ok=True)
    tmp = task["path"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, task["path"])           # atomic: a half-written file never counts as done
    return task["path"]


def _init_worker(threads: int):
    import torch
    torch.set_num_threads(threads)          # N workers x all-core intra-op pools would oversubscribe the CPU


def run_tasks(tasks: list[dict], workers: int = 1):
    todo = [t for t in tasks if not _done(t)]
    print(f"[lossgrid] {len(tasks) - len(todo)}/{len(tasks)} runs already done, running {len(todo)}", flush=True)
    if workers > 1 and len(todo) > 1:
        threads = max(1, (os.cpu_count() or 1) // workers)
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(threads,)) as pool:
            for i, p in enumerate(pool.imap_unordered(run_task, todo), 1):
                print(f"[lossgrid] ({i}/{len(todo)}) {os.path.basename(p)}", flush=True)
    else:
        for i, t in enumerate(todo, 1):
            print(f"[lossgrid] ({i}/{len(todo)}) {os.path.basename(run_task(t))}", flush=True)


def load_results(tasks: list[dict]) -> list[dict]:
    return [json.load(open(t["path"])) for t in tasks if _done(t)]


def metric_of(res: dict, split: str, spec: str) -> float:
    group, m = spec.rsplit("/", 1)
    return float(res["metrics"][split].get(group, {}).get(m, float("nan")))


def select(cfg: dict, results: list[dict]) -> dict:
    """Per arm: (lr, weight) with the best mean-over-seeds validation selection metric (NaN loses)."""
    sel, sign = cfg["selection"], 1.0 if cfg["selection"]["mode"] == "max" else -1.0
    score: dict = {}
    for r in results:
        score.setdefault((r["arm"], r["lr"], r["weight"]), []).append(
            metric_of(r, cfg["data"]["val_split"], sel["metric"]))
    best: dict = {}
    for (arm, lr, w), v in score.items():
        m = sign * float(np.mean(v)) if np.isfinite(v).all() else -np.inf
        if arm not in best or m > best[arm][0]:
            best[arm] = (m, lr, w)
    return {arm: (lr, w) for arm, (_, lr, w) in best.items()}


def _fmt(m, s):
    return "nan" if not np.isfinite(m) else f"{m:.3f}±{s:.3f}"


def summarize(cfg: dict, results: list[dict], split: str, selected: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm mean/std over seeds + main effects / pairwise interactions. Returns (arm_table, effects_table)."""
    specs = cfg["summary_metrics"]
    rows = []
    for arm in [arm_name(a) for a in arms(cfg["grid"]["terms"])]:
        rs = [r for r in results if r["arm"] == arm]
        if not rs:
            continue
        row = {"arm": arm, "n_terms": len(rs[0]["terms"]), "lr": rs[0]["lr"], "weight": rs[0]["weight"], "n_seeds": len(rs)}
        for s in specs:
            v = np.array([metric_of(r, split, s) for r in rs])
            row[f"{s}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[f"{s}_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    eff = []
    terms = list(cfg["grid"]["terms"])
    for s in specs:
        col = table.set_index("arm")[f"{s}_mean"]
        has = {t: np.array([t in a.split("+") for a in col.index]) for t in terms}
        for t in terms:
            p, a = col[has[t]], col[~has[t]]
            eff.append({"metric": s, "effect": t, "present": p.mean(), "absent": a.mean() if len(a) else np.nan,
                        "diff": p.mean() - a.mean() if len(a) else np.nan})
        for t1, t2 in itertools.combinations(terms, 2):
            cell = {(x, y): col[(has[t1] == x) & (has[t2] == y)].mean() for x in (True, False) for y in (True, False)}
            eff.append({"metric": s, "effect": f"{t1}x{t2}", "present": cell[(True, True)],
                        "absent": cell[(False, False)],
                        "diff": cell[(True, True)] - cell[(True, False)] - cell[(False, True)] + cell[(False, False)]})
    return table, pd.DataFrame(eff)


def _md(table: pd.DataFrame, specs) -> str:
    head = ["arm", "lr", "weight", "seeds"] + [s.split("/", 1)[-1] if s.startswith("all/") else s for s in specs]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join([r["arm"], f"{r['lr']:g}", f"{r['weight']:g}", str(r["n_seeds"])] +
                                       [_fmt(r[f"{s}_mean"], r[f"{s}_std"]) for s in specs]) + " |")
    return "\n".join(lines)


def write_reports(cfg: dict, results: list[dict], selected: dict):
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    split = cfg["data"]["val_split"]
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    chosen = [r for r in results if selected.get(r["arm"]) == (r["lr"], r["weight"])]
    with open(os.path.join(out, "selected.json"), "w") as f:
        json.dump({a: {"lr": lr, "weight": w,
                       "heads": sorted(r["head_path"] for r in chosen if r["arm"] == a),
                       "best_steps": [r["best_step"] for r in sorted(chosen, key=lambda r: r["head_path"])
                                      if r["arm"] == a]}
                   for a, (lr, w) in selected.items()}, f, indent=1)
    specs, md = cfg["summary_metrics"], [f"# Loss grid ({cfg['model']['head']} head, layer {cfg['data']['layer']})\n",
                                         f"Checkpoint per run: best validation loss (every {cfg['train']['eval_every']} "
                                         f"steps). Config selection: validation {cfg['selection']['metric']} "
                                         f"({cfg['selection']['mode']}), mean over seeds {cfg['grid']['seeds']}. "
                                         "Test: full Decision Index (scrm.dindex_frozen).\n"]
    if chosen:
        table, eff = summarize(cfg, chosen, split)
        table.to_csv(os.path.join(out, f"summary_{split}.csv"), index=False)
        eff.to_csv(os.path.join(out, f"effects_{split}.csv"), index=False)
        md += [f"## {split} (selected configs)\n", _md(table, specs), "",
               f"### Main effects and pairwise interactions ({split})\n",
               "`diff` for a term = mean(arms with) - mean(arms without); for `AxB` = interaction "
               "m11 - m10 - m01 + m00 (cell means over arms).\n", eff.round(4).to_markdown(index=False)
               if _has_tabulate() else eff.round(4).to_string(index=False), ""]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true", help="only (re)write reports from existing run files")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.steps=50 grid.seeds=[0,1]")
    a = ap.parse_args(argv)
    cfg = load_grid_config(a.config, a.overrides)
    workers = a.workers or cfg["workers"]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    tasks = make_tasks(cfg)
    if not a.summarize_only:
        run_tasks(tasks, workers)
    results = load_results(tasks)
    write_reports(cfg, results, select(cfg, results))
    print(f"[lossgrid] wrote reports to {cfg['output_dir']}; heads for the Decision Index: "
          f"{os.path.join(cfg['output_dir'], 'selected.json')}")


if __name__ == "__main__":
    main()
