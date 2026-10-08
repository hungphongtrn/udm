"""Loss-function grid on frozen cached features: every non-empty subset of {ce, brier, bt, sigmoid} x seeds x a
hyperparameter grid that has the SAME size for every arm (python -m scrm.lossgrid --config configs/lossgrid_*.yaml).

Protocol (matched tuning budget): all arms get the identical grid `grid.lr x grid.weight`, the same step count, data order
and init per seed. In a multi-term arm every term has weight 1.0 (x the grid's global `weight` multiplier), so there is no
per-term tuning that would favour bigger arms. Every (arm, lr, weight, seed) run keeps its best-validation-loss
checkpoint (train.eval_every) and saves that head under <output_dir>/heads/. Per arm the (lr, weight) with the best
mean-over-seeds validation selection metric is chosen; selected.json lists its heads (one per seed) for the test
benchmark, the full Decision Index on live backbone features (scrm.dindex_frozen). Resumable: finished runs are skipped.

dindex.enabled: after training, every head (dindex.heads=all; or only the selected ones) is scored on the full Decision
Index from a backbone feature cache made once by `scrm.dindex_frozen featurize` (scripts/train/lossgrid.sh runs that
step first). The reports then add per-arm Decision Index (validation-selected config), main effects / interactions of
each loss term on it, every config's score, and how well the validation metrics predict it. Scored heads are reused."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from .config import deep_update, _load_yaml, parse_overrides
from .feat_train import FEAT_DEFAULTS, TERMS, run_experiment

GRID_DEFAULTS = {
    "output_dir": "outputs/lossgrid",
    "workers": 1,
    "grid": {"terms": list(TERMS), "lr": [3e-4, 1e-3], "weight": [1.0], "seeds": [0, 1, 2]},
    "selection": {"metric": "loss/total", "mode": "min"},       # on validation; "<group>/<metric>"
    "dindex": {"enabled": False,
               "feats_dir": "outputs/dindex_feats_qwen3_5_4b",  # scrm.dindex_frozen featurize output (backbone, once)
               "out_dir": None,                                 # default <output_dir>/dindex
               "heads": "all",                                  # all | selected (val-selected config per arm)
               "di_dir": None,                                  # default $DI_DIR or ../decision-index
               "edition": "0.2.1", "board": None},              # board default <di_dir>/tests/fixtures/board-0.2.1.json
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
    res = run_experiment(task["cfg"], task["evals"], log=lambda *_: None, head_path=task["head_path"])
    res.update(arm=task["arm"], weight=task["weight"])
    os.makedirs(os.path.dirname(task["path"]), exist_ok=True)
    tmp = task["path"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, task["path"])           # atomic: a half-written file never counts as done
    return task["path"]


def _worker_init(threads: int):
    import torch
    torch.set_num_threads(threads)          # N workers x all cores would oversubscribe the CPU


def run_tasks(tasks: list[dict], workers: int = 1):
    todo = [t for t in tasks if not _done(t)]
    print(f"[lossgrid] {len(tasks) - len(todo)}/{len(tasks)} runs already done, running {len(todo)}", flush=True)
    if workers > 1 and len(todo) > 1:
        threads = max(1, (os.cpu_count() or 1) // workers)
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(threads,)) as pool:
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


def summarize(cfg: dict, results: list[dict], split: str, specs: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm mean/std over seeds + main effects / pairwise interactions. Returns (arm_table, effects_table)."""
    specs = specs or cfg["summary_metrics"]
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


DI_SPLIT = "dindex"              # pseudo-split under res["metrics"] holding a run's Decision Index scores
DI_HEAD = ["di/index", "di/raw_index", "di/answered_frac"]


def di_dir(cfg: dict) -> str:
    from .dindex_frozen import DEFAULT_DI_DIR
    return cfg["dindex"]["di_dir"] or os.environ.get("DI_DIR") or DEFAULT_DI_DIR


def di_out(cfg: dict) -> str:
    return cfg["dindex"]["out_dir"] or os.path.join(cfg["output_dir"], "dindex")


def di_feats_pending(cfg: dict) -> str | None:
    """The feature cache dir when the Decision Index is on and that cache is not complete yet, else None."""
    d = cfg["dindex"]
    if not d["enabled"]:
        return None
    man = os.path.join(d["feats_dir"], "manifest.json")
    return None if os.path.exists(man) and json.load(open(man)).get("complete") else d["feats_dir"]


def score_dindex(cfg: dict, results: list[dict], selected: dict) -> list[dict]:
    """Score the grid's heads on the full Decision Index (scrm.dindex_frozen.score, resumable per head) and attach each
    run's scores as res["metrics"]["dindex"]. Returns the per-head rows."""
    from . import dindex_frozen, feat_train
    d = cfg["dindex"]
    if not os.path.exists(os.path.join(d["feats_dir"], "manifest.json")):
        raise SystemExit(f"[lossgrid] dindex.enabled but no feature cache at {d['feats_dir']}: run "
                         "scripts/train/lossgrid.sh (it featurizes first) or `python -m scrm.dindex_frozen featurize "
                         f"--config configs/features_qwen3_5_4b.yaml --out {d['feats_dir']}`")
    if d["heads"] not in ("all", "selected"):
        raise ValueError(f"dindex.heads must be all|selected, got {d['heads']!r}")
    runs = [r for r in results if d["heads"] == "all" or selected.get(r["arm"]) == (r["lr"], r["weight"])]
    if not runs:
        return []
    feat_train._SPLITS.clear()              # free the training features (workers=1 keeps them in this process)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    root = di_dir(cfg)
    t0 = time.time()
    rows = dindex_frozen.score(d["feats_dir"], [r["head_path"] for r in runs], di_out(cfg),
                               os.path.join(root, "suite-0.2"), d["edition"],
                               d["board"] or os.path.join(root, "tests", "fixtures", "board-0.2.1.json"), report=False)
    print(f"[lossgrid] Decision Index: {len(rows)} heads in {(time.time() - t0) / 60:.1f} min", flush=True)
    attach_dindex(results, rows)
    return rows


def _run_name(res: dict) -> str:
    return os.path.basename(res["head_path"])[:-len(".pt")]


def attach_dindex(results: list[dict], rows: list[dict]) -> None:
    """index / raw_index / answered_frac -> metrics["dindex"]["di"]; "area/<id>" -> metrics["dindex"]["area"]."""
    by_run = {r["run"]: r for r in rows}
    for res in results:
        row = by_run.get(_run_name(res))
        if row is None:
            continue
        m: dict = {}
        for k, v in row.items():
            if k in ("index", "raw_index", "answered_frac"):
                m.setdefault("di", {})[k] = float(v)
            elif k.startswith("area/"):
                m.setdefault("area", {})[k[len("area/"):]] = float(v) if v is not None else float("nan")
        res["metrics"][DI_SPLIT] = m


def di_specs(results: list[dict]) -> list[str]:
    areas = sorted({k for r in results for k in r["metrics"].get(DI_SPLIT, {}).get("area", {})})
    return DI_HEAD + [f"area/{a}" for a in areas]


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False)


def _di_section(cfg: dict, results: list[dict], chosen: list[dict]) -> list[str]:
    """Decision Index report (markdown lines; csv files alongside): validation-selected config per arm, the effect of
    each loss term, every config's score, and how well validation metrics rank the runs by index."""
    out, val = cfg["output_dir"], cfg["data"]["val_split"]
    scored = [r for r in results if DI_SPLIT in r["metrics"]]
    if not scored:
        return []
    specs = di_specs(scored)
    md = ["## Decision Index (full suite; config per arm selected on validation)\n",
          f"Heads scored: {len(scored)} (dindex.heads={cfg['dindex']['heads']}). index = chance-corrected balanced "
          "skill x 100, area/* = per-area skill x 100. Sorted by index.\n"]
    sel = [r for r in chosen if DI_SPLIT in r["metrics"]]
    if sel:
        table, eff = summarize(cfg, sel, DI_SPLIT, specs)
        table.to_csv(os.path.join(out, "summary_dindex.csv"), index=False)
        eff.to_csv(os.path.join(out, "effects_dindex.csv"), index=False)
        md += [_md(table.sort_values("di/index_mean", ascending=False), specs), "",
               "### Effect of each loss term on the Decision Index\n",
               "`diff` for a term = mean over arms with it - mean over arms without it; `AxB` = interaction "
               "m11 - m10 - m01 + m00 (cell means over arms). All areas: effects_dindex.csv.\n",
               _table(eff[eff["metric"].isin(DI_HEAD[:2])].round(3)), ""]
    rows = [{"run": _run_name(r), "arm": r["arm"], "lr": r["lr"], "weight": r["weight"], "seed": r["seed"],
             "best_step": r["best_step"], "val_loss": r["val_loss"],
             **{f"val/{s}": metric_of(r, val, s) for s in cfg["summary_metrics"]},
             **{s: metric_of(r, DI_SPLIT, s) for s in specs}} for r in scored]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "dindex_runs.csv"), index=False)
    cfgs = (df.groupby(["arm", "lr", "weight"])
            .agg(index=("di/index", "mean"), index_std=("di/index", "std"), val_loss=("val_loss", "mean"),
                 seeds=("seed", "size")).reset_index())
    picked = {r["arm"]: (r["lr"], r["weight"]) for r in chosen}
    cfgs["val_selected"] = [picked.get(a) == (lr, w) for a, lr, w in zip(cfgs["arm"], cfgs["lr"], cfgs["weight"])]
    cfgs["oracle_gap"] = cfgs.groupby("arm")["index"].transform("max") - cfgs["index"]
    cfgs.to_csv(os.path.join(out, "dindex_configs.csv"), index=False)
    if len(cfgs) > cfgs["arm"].nunique():
        md += ["### Every config (lr, weight)\n",
               "`val_selected` = the config the table above reports; `oracle_gap` = best index among this arm's "
               "configs (picked on the test benchmark itself, so optimistic) minus this config's index.\n",
               _table(cfgs.sort_values(["arm", "lr", "weight"]).round(3)), ""]
    cand = ["val_loss"] + [f"val/{s}" for s in cfg["summary_metrics"]]
    rho = [(c, df[c].rank().corr(df["di/index"].rank())) for c in cand if df[c].notna().sum() >= 3]   # Spearman
    rho = [(c, v) for c, v in rho if np.isfinite(v)]
    if rho:
        md += ["### Does validation predict the Decision Index?\n",
               f"Spearman rank correlation with the index over {len(df)} scored runs (val_loss is each run's own "
               "objective, so it is not comparable across arms).\n", "| validation metric | rho |", "|---|---|"]
        md += [f"| {c} | {v:+.3f} |" for c, v in sorted(rho, key=lambda t: -abs(t[1]))] + [""]
    return md


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
    md += _di_section(cfg, results, chosen)
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
    ap.add_argument("--print-dindex-feats", action="store_true",
                    help="print dindex.feats_dir if the Decision Index is enabled and its feature cache is incomplete "
                         "(scripts/train/lossgrid.sh featurizes it first), then exit")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.steps=50 grid.seeds=[0,1]")
    a = ap.parse_args(argv)
    cfg = load_grid_config(a.config, a.overrides)
    if a.print_dindex_feats:
        print(di_feats_pending(cfg) or "")
        return
    workers = a.workers or cfg["workers"]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    tasks = make_tasks(cfg)
    if not a.summarize_only:
        run_tasks(tasks, workers)
    results = load_results(tasks)
    selected = select(cfg, results)
    if cfg["dindex"]["enabled"]:
        score_dindex(cfg, results, selected)
    write_reports(cfg, results, selected)
    print(f"[lossgrid] wrote reports to {cfg['output_dir']}" +
          (f" (Decision Index per head: {di_out(cfg)})" if cfg["dindex"]["enabled"] else
           f"; heads for the Decision Index: {os.path.join(cfg['output_dir'], 'selected.json')}"))


if __name__ == "__main__":
    main()
