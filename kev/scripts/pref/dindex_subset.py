"""A shared stratified subset of Decision Index 0.3 for comparing two Kev checkpoints (D vs v7-base) on the same rows.

The index estimated on the subset is not the board's index (fewer cases per benchmark), but both checkpoints see the
same requests and the kit's own scorers, chance levels and area weights, so their difference is a paired comparison.
Run from the kit checkout (dindex.sh does), with the kit installed:
  make N ROWS                  N rows of the index's benchmarks (the five areas), excluded rows dropped, complete case
                               groups, round-robin over benchmarks in hash order (kit's suite.sample.stratified_sample)
  seed ROWS FROM OUT OPT..     copy rows already answered (ok / abstained) in FROM (dir of a `full` run of the same
                               checkpoint, possibly still running) into OUT/results.jsonl, skipping rows OUT already
                               has, so the subset run does not redo them; refuses if FROM ran with other engine options
                               than OPT (key=value, as given to --option)
  score ROWS RESULTS OUT       the kit's scoring (pipeline.score_run) with the suite restricted to ROWS
  compare OUT ENGINE           a scored subset run (OUT) vs a board entrant's published 0.3 per-benchmark results
                               (ENGINE: its board id, e.g. kev-4b-r10); writes OUT/board-compare-ENGINE.json
  pair OUT_A OUT_B             two scored subset runs side by side: index, areas, benchmarks (B - A)
  retemper ROWS OUT T..        rescore a subset run (OUT) at other temperatures, without rerunning the model: each
                               T -> OUT/T<T>/ (results.jsonl + the kit's scores), OUT/best -> the best T's dir; prints
                               the index per T

compare is the reproduction check: run a board entrant's own checkpoint on the subset and see whether each benchmark
lands within sampling error of its board score. Only per-benchmark results are published, not per-request ones, so the
board score covers the whole benchmark and ours the subset's cases: SE = sqrt(p(1-p)/n) on our n, for the accuracy
metrics only (F1, nDCG, Brier and quality have no simple SE; their difference is shown without z).

retemper: within a question p_i = softmax(r_i / T_run), so T_run * log p_i = r_i + const and the answers saved in
results.jsonl (full-precision probabilities, also under --compact) give softmax(T_run / T * log p_i) at any T exactly;
noul answers are a two-option (false, true) softmax, retempered through their log-odds. The top choice never moves, so
only probability-sensitive metrics change. T_run is the temperature the run served at (environment.json
model_source.temperature). The best T over a grid is chosen on the scored rows themselves: an optimistic bound.
"""
import json
import math
import pathlib
import sys

import decision_index
from decision_index.pipeline import score_run
from decision_index.suite.io import Suite, read_jsonl
from decision_index.suite.sample import stratified_sample

EDITION, SUITE_DIR = "0.3", "suite-0.3"
TINY = 1e-300   # a probability that underflowed to 0 stays (near) 0 at any T


class Restricted(Suite):
    """The suite as every kit scorer and the sampler see it (they all read Suite.rows), filtered by keep(_evaluation)."""

    def __init__(self, keep):
        super().__init__(SUITE_DIR, EDITION)
        self.keep = keep

    def rows(self, apply_exclusions=False):
        return (r for r in super().rows(apply_exclusions) if self.keep(r["_evaluation"]))


def run_ids(rows):
    return {r["_evaluation"]["run_id"] for r in read_jsonl(rows)}


BOARD = "https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json"


def compare(out, engine):
    import urllib.request

    board = json.loads(urllib.request.urlopen(BOARD).read())
    entrant = next((m for m in board["models"] if m["engine"] == engine), None)
    if entrant is None: raise SystemExit(f"{engine} is not on the board ({BOARD})")
    ours = json.loads((out / "benchmark-summary.json").read_text())["benchmarks"]
    index = json.loads((out / "index.json").read_text())
    rows = []
    for b in sorted(ours, key=lambda b: b["catalog_id"]):
        n, theirs = b.get("scored_requests") or 0, entrant["results"].get(str(b["catalog_id"]), {})
        if b.get("score") is None or theirs.get("score") is None: continue
        diff = b["score"] - theirs["score"]
        se = math.sqrt(b["score"] * (1 - b["score"]) / n) if n and "accuracy" in (b.get("metric") or "") else None
        rows.append(dict(id=b["catalog_id"], dataset=b["dataset"], metric=b.get("metric"), n=n, ours=round(b["score"], 4),
                         board=theirs["score"], diff=round(diff, 4), z=round(diff / se, 2) if se else None,
                         unsupported=b.get("unsupported", 0), errors=b.get("errors", 0)))
    zs = [r["z"] for r in rows if r["z"] is not None]
    report = {"engine": engine, "board_generated_utc": board.get("generated_utc"), "board_public_index": entrant["v03"]["pub"],
              "subset_index": index["index"], "benchmarks": rows, "mean_diff": round(sum(r["diff"] for r in rows) / len(rows), 4),
              "z_beyond_3": [r["dataset"] for r in rows if r["z"] is not None and abs(r["z"]) > 3], "with_z": len(zs)}
    (out / f"board-compare-{engine}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'id':>3} {'dataset':<28} {'n':>5} {'ours':>7} {'board':>7} {'diff':>7} {'z':>6}  metric")
    for r in rows:
        z = "" if r["z"] is None else f"{r['z']:6.2f}"
        print(f"{r['id']:>3} {r['dataset'][:28]:<28} {r['n']:>5} {r['ours']:7.4f} {r['board']:7.4f} {r['diff']:+7.4f} {z:>6}  {r['metric']}")
    print(json.dumps({k: report[k] for k in ("board_public_index", "subset_index", "mean_diff", "z_beyond_3", "with_z")}))


def pair(a, b):
    ia, ib = (json.loads((o / "index.json").read_text()) for o in (a, b))
    print(f"{'':<34} {a.parent.name + '/' + a.name:>24} {b.parent.name + '/' + b.name:>24} {'B - A':>8}")
    print(f"{'index':<34} {ia['index']:>24.2f} {ib['index']:>24.2f} {ib['index'] - ia['index']:>+8.2f}")
    for x, y in zip(ia["areas"], ib["areas"]):
        print(f"{'  ' + x['label']:<34} {x['raw']:>24.4f} {y['raw']:>24.4f} {y['raw'] - x['raw']:>+8.4f}")
    for n, x in ia["benchmarks"].items():
        y = ib["benchmarks"][n]
        if x["raw"] is not None and y["raw"] is not None:
            print(f"{'    benchmark ' + n:<34} {x['raw']:>24.4f} {y['raw']:>24.4f} {y['raw'] - x['raw']:>+8.4f}")


def retempered(answer, ratio):
    """An answer served at T_run, at T = T_run / ratio."""
    if answer["type"] == "noul":
        p = min(max(answer["noul"], TINY), 1 - 1e-16)
        return {**answer, "noul": 1 / (1 + math.exp(-ratio * math.log(p / (1 - p))))}
    z = {k: ratio * math.log(max(p, TINY)) for k, p in answer["probabilities"].items()}
    top = max(z.values())
    e = {k: math.exp(v - top) for k, v in z.items()}
    s = sum(e.values())
    return {**answer, "probabilities": {k: v / s for k, v in e.items()}}   # the choice (argmax) is unchanged


def retemper(rows, out, temps):
    keep = run_ids(rows)
    t_run = json.loads((out / "environment.json").read_text())["model_source"]["temperature"]
    results = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    table = []
    for t in temps:
        ratio, d = t_run / t, out / f"T{t:g}"
        d.mkdir(exist_ok=True)
        with open(d / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                if r["status"] == "ok":
                    r = {**r, "response": {**r["response"], "answers": {q: retempered(a, ratio) for q, a in r["response"]["answers"].items()}}}
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        scores = score_run(Restricted(lambda e: e["run_id"] in keep), d / "results.jsonl", "kev", d)
        table.append((t, scores["decision_index"]))
        print(json.dumps({"T": t, "T_run": t_run, "index": scores["decision_index"]}))
    best = max(table, key=lambda x: x[1])
    link = out / "best"
    link.unlink(missing_ok=True)
    link.symlink_to(f"T{best[0]:g}")
    print(json.dumps({"best_T": best[0], "best_index": best[1], "note": "best T of the grid, chosen on these same rows: optimistic"}))


def main(cmd, *args):
    if cmd == "make":
        n, rows = int(args[0]), pathlib.Path(args[1])
        spec = json.loads((pathlib.Path(decision_index.__file__).parent / f"data/index-{EDITION}.json").read_text())
        ids = {b for a in spec["areas"] for b in a["benchmarks"]}
        part = rows.with_name(rows.name + ".part.gz")
        out = stratified_sample(Restricted(lambda e: e["catalog_id"] in ids), n, part)
        part.rename(rows)
        print(json.dumps({**out, "path": str(rows)}))
    elif cmd == "seed":
        from decision_index.cli import parse_value

        keep, src, out = run_ids(args[0]), pathlib.Path(args[1]), pathlib.Path(args[2])
        want = {k: parse_value(v) for k, v in (o.partition("=")[::2] for o in args[3:])}
        ran = json.loads((src / "environment.json").read_text())["engine_options"]
        if ran != want: raise SystemExit(f"{src} ran with engine options {ran}, not {want}: its rows cannot seed this run")
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "results.jsonl"
        have = set()
        if dst.exists():   # an interrupted subset run: drop a torn last line (as the kit's resume does), keep the rest
            text = dst.read_text(encoding="utf-8")
            dst.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
            have = {r["run_id"] for r in map(json.loads, text.splitlines()[: text.count("\n")]) if r["status"] != "error"}
        with open(src / "results.jsonl", encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as g:
            lines = [l for l in f if l.endswith("\n") and (r := json.loads(l))["run_id"] in keep - have and r["status"] in ("ok", "abstained")]
            g.writelines(lines)
        print(json.dumps({"seeded": len(lines), "already_in_subset": len(have & keep), "subset_rows": len(keep)}))
    elif cmd == "score":
        keep = run_ids(args[0])
        scores = score_run(Restricted(lambda e: e["run_id"] in keep), args[1], "kev", args[2])
        print(json.dumps({k: scores[k] for k in ("decision_index", "scores", "completed", "counts")}, indent=2))
    elif cmd == "compare":
        compare(pathlib.Path(args[0]), args[1])
    elif cmd == "pair":
        pair(pathlib.Path(args[0]), pathlib.Path(args[1]))
    elif cmd == "retemper":
        retemper(args[0], pathlib.Path(args[1]), [float(t) for t in args[2:]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(*sys.argv[1:])
