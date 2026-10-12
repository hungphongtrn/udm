"""A shared stratified subset of Decision Index 0.3 for comparing two Kev checkpoints (D vs v7-base) on the same rows.

The index estimated on the subset is not the board's index (fewer cases per benchmark), but both checkpoints see the
same requests and the kit's own scorers, chance levels and area weights, so their difference is a paired comparison.
Run from the kit checkout (dindex.sh does), with the kit installed:
  make N ROWS                  N rows of the index's benchmarks (the five areas), excluded rows dropped, complete case
                               groups, round-robin over benchmarks in hash order (kit's suite.sample.stratified_sample)
  seed ROWS FROM_RESULTS OUT   copy rows already answered in FROM_RESULTS (a `full` / `index` run of the same checkpoint)
                               into OUT/results.jsonl, so the subset run does not redo them; `error` rows are left out
  score ROWS RESULTS OUT       the kit's scoring (pipeline.score_run) with the suite restricted to ROWS
  compare OUT ENGINE           a scored subset run (OUT) vs a board entrant's published 0.3 per-benchmark results
                               (ENGINE: its board id, e.g. kev-4b-r10); writes OUT/board-compare-ENGINE.json

compare is the reproduction check: run a board entrant's own checkpoint on the subset and see whether each benchmark
lands within sampling error of its board score. Only per-benchmark results are published, not per-request ones, so the
board score covers the whole benchmark and ours the subset's cases: SE = sqrt(p(1-p)/n) on our n, for the accuracy
metrics only (F1, nDCG, Brier and quality have no simple SE; their difference is shown without z).
"""
import json
import pathlib
import sys

import decision_index
from decision_index.pipeline import score_run
from decision_index.suite.io import Suite, read_jsonl
from decision_index.suite.sample import stratified_sample

EDITION, SUITE_DIR = "0.3", "suite-0.3"


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
    import math
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
        keep, out = run_ids(args[0]), pathlib.Path(args[2])
        out.mkdir(parents=True, exist_ok=True)
        with open(args[1], encoding="utf-8") as src, open(out / "results.jsonl", "x", encoding="utf-8") as dst:
            lines = [l for l in src if l.endswith("\n") and (r := json.loads(l))["run_id"] in keep and r["status"] != "error"]
            dst.writelines(lines)
        print(json.dumps({"seeded": len(lines), "subset_rows": len(keep)}))
    elif cmd == "score":
        keep = run_ids(args[0])
        scores = score_run(Restricted(lambda e: e["run_id"] in keep), args[1], "kev", args[2])
        print(json.dumps({k: scores[k] for k in ("decision_index", "raw_index", "scores", "completed", "counts")}, indent=2))
    elif cmd == "compare":
        compare(pathlib.Path(args[0]), args[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(*sys.argv[1:])
