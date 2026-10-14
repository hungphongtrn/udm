"""All-wrong minimal pairs from a frozen suite partition, for kev.benchmark --data (CPU, no model).

    .venv/bin/python scripts/abstain/build_allwrong.py --suite evals/v7/decision-v7 --out runs/allwrong/data/decision-v7-development

Every eligible Choice question (clean record, not `unknowable`, hard label, >= 3 options, no none-of-the-above key) is
written four times, one JSONL per condition, each a labelled request kev.data.load_records reads:

    answerable       one seeded wrong option removed        (K - 1 options, the label among them)
    allwrong         the labelled option removed            (K - 1 options, every one wrong)
    answerable_none  answerable + a none option             (K options, label = the true option)
    allwrong_none    allwrong + the same none option        (K options, label = the none option)

answerable and allwrong differ only in which option is missing: same state, instruction, option count and order of the
options they share, so a detector cannot use the count or the position. The none pair adds the same wording (one of
kev.data.NONE_OPTIONS, seeded per question) at the same position, as kev.data.none_pair does in training.
allwrong has no right option, but load_records needs a label: it carries the first remaining key and
`_meta.allwrong = true`; its top1 in report.json is meaningless (allwrong_report.py never reads it).
Non-eligible questions are dropped from the record; a record with none left is dropped. Ids, sources and group ids stay
the suite's, so the conditions join on (id, question). Writes manifest.json (suite digest, seed, counts).
"""
import argparse
import random
from pathlib import Path

from kev.data import NONE_OPTIONS
from kev.suite import digest, load_split, write_json, write_jsonl

CONDITIONS = ("answerable", "allwrong", "answerable_none", "allwrong_none")
NONE_KEYS = {k for k, _ in NONE_OPTIONS}


def eligible(q):
    return (q["type"] == "choice" and q.get("target") is None and len(q["criteria"]) >= 3
            and not set(q["criteria"]) & NONE_KEYS)


def variants(q, rng):
    """-> {condition: question} for one eligible question; rng draws the removed wrong option, the none wording and its
    position, in that order."""
    keys, y = list(q["criteria"]), q["label"]
    w = rng.choice([k for k in keys if k != y])
    nk, nd = rng.choice(NONE_OPTIONS)
    ans = [k for k in keys if k != w]; bad = [k for k in keys if k != y]
    pos = rng.randrange(len(ans) + 1)

    def question(order, label, none=False):
        crit = {k: q["criteria"][k] for k in order}
        if none:
            items = list(crit.items()); items.insert(pos, (nk, nd)); crit = dict(items)
        return {**q, "criteria": crit, "label": label}

    return {"answerable": question(ans, y), "allwrong": question(bad, bad[0]),
            "answerable_none": question(ans, y, True), "allwrong_none": question(bad, nk, True)}


def build(records, seed):
    """-> ({condition: records}, counts)."""
    out = {c: [] for c in CONDITIONS}
    counts = {"records": 0, "questions": 0, "dropped_questions": 0}
    for r in records:
        meta = r["_meta"]
        if meta["variant"] != "clean" or meta["source"] == "unknowable":
            continue
        qs = {c: {} for c in CONDITIONS}
        for qid, q in r["questions"].items():
            if not eligible(q):
                counts["dropped_questions"] += 1; continue
            for c, v in variants(q, random.Random(f"{seed}:{meta['id']}:{qid}")).items():
                qs[c][qid] = v
        if not qs["answerable"]:
            continue
        counts["records"] += 1; counts["questions"] += len(qs["answerable"])
        for c in CONDITIONS:
            out[c].append({"state": r["state"], "questions": qs[c], "_meta": {**meta, "allwrong": c == "allwrong"}})
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--split", default="development", choices=("development", "calibration"))
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f"{out} exists")
    data, counts = build(load_split(a.suite, a.split), a.seed)
    if not counts["questions"]: raise SystemExit("no eligible Choice questions")
    out.mkdir(parents=True)
    for c, recs in data.items(): write_jsonl(out / f"{c}.jsonl", recs)
    write_json(out / "manifest.json", {"suite": a.suite, "split": a.split, "seed": a.seed,
                                       "suite_manifest_sha256": digest(Path(a.suite) / "manifest.json"), **counts,
                                       "files": {c: digest(out / f"{c}.jsonl") for c in CONDITIONS}})
    print(f"{out}: {counts}")


if __name__ == "__main__":
    main()
