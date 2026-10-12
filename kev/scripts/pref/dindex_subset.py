"""A shared stratified subset of Decision Index 0.3 for comparing two Kev checkpoints (D vs v7-base) on the same rows.

The index estimated on the subset is not the board's index (fewer cases per benchmark), but both checkpoints see the
same requests and the kit's own scorers, chance levels and area weights, so their difference is a paired comparison.
Run from the kit checkout (dindex.sh does), with the kit installed:
  make N ROWS                  N rows of the index's benchmarks (the five areas), excluded rows dropped, complete case
                               groups, round-robin over benchmarks in hash order (kit's suite.sample.stratified_sample)
  seed ROWS FROM_RESULTS OUT   copy rows already answered in FROM_RESULTS (a `full` / `index` run of the same checkpoint)
                               into OUT/results.jsonl, so the subset run does not redo them; `error` rows are left out
  score ROWS RESULTS OUT       the kit's scoring (pipeline.score_run) with the suite restricted to ROWS
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
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(*sys.argv[1:])
