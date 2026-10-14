"""Contamination check: Lu & Wu's frozen benchmark texts against our own training records.

    .venv/bin/python scripts/external/overlap.py --cases runs/dmb/core-inputs \
        --suite evals/v7/decision-v7 --out runs/dmb/overlap-core.json

Their `decision-benchmark prepare --with-text` directory supplies one text per case; our side is a kev suite's training
partition (`--suite`, fetched from its Hub mirror when the partition is not local) or a plain `--records` JSONL. Every
training record contributes three kinds of unit text: the rendered state (`kev.api.render`, what a model actually reads),
each question's instruction, and each option text (criteria values, or score levels). Units are compared against every
case text in both directions plus exact equality after `kev.suite.normalise_text` (casefolded, whitespace collapsed, the
key kev.data.load_records hashes states with):

    mode      rule
    exact     normalised texts equal
    contains  our unit occurs in their case text (the unit is the contained side, >= --min-chars)
    contained their case text occurs in our unit (their case is the contained side, >= --min-chars)

A dataset is flagged when the fraction of ITS cases that matched any unit of ours exceeds `--gate` (default 0.01);
scripts/external/luwu_policies.py --overlap consumes this file, drops the flagged datasets (or only the flagged case ids
with --drop-overlapping-cases) and records the exclusion. Nothing is dropped here: this script only measures.

The scan is the ticket's naive double loop (every case against every unit), so a run over the core profile's ~1.5k cases
and a 12k-record training partition takes minutes and is meant to be run once per partition, not per arm.
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kev.api import render
from kev.data import load_records
from kev.suite import ENCODING, load_split, normalise_text

MIN_CHARS_DEFAULT, GATE_DEFAULT, EXAMPLES_DEFAULT = 32, 0.01, 5


def compact(name):
    """A dataset name with digit suffixes dropped, for the informational same-name note (ag_news -> agnews, dbpedia14
    and dbpedia -> dbpedia). Never used to skip comparisons, only to point at the breakdown."""
    return re.sub(r"[^a-z]", "", str(name).casefold()).rstrip("0123456789")


def state_texts(state):
    """The texts a model reads off a state: the rendered state itself, and every string leaf in it (a structured state
    holds the dataset text under a field name, and their case text is that text, so the leaf is what exact equality
    needs)."""
    out = [("state", render(state))]
    stack = [state]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            out.append(("state_value", value))
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return out


def their_cases(directory):
    """Their prepare --with-text directory -> {case_id: (task, split, text)}; refuses a text-less preparation."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding=ENCODING))
    if manifest.get("case_representation") == "text-index-v1":
        raise SystemExit(f"{directory}: no text; run `decision-benchmark prepare --with-text` first")
    cases = {}
    for line in (directory / "cases.jsonl").read_text(encoding=ENCODING).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SystemExit(f"{directory}: case {row.get('case_id')} has no text")
        if row["case_id"] in cases:
            raise SystemExit(f"{directory}: duplicate case_id {row['case_id']}")
        cases[row["case_id"]] = {"task": row["task"], "split": row["split"], "text": text}
    if not cases:
        raise SystemExit(f"{directory}/cases.jsonl: no cases")
    return directory, cases


def our_units(records):
    """One record -> its unit texts: the rendered state, each instruction, each option. Deduplicated on the normalised
    text (the first record that carried it wins, the rest are counted) so a repeated boilerplate unit does not multiply
    the scan."""
    units, seen = [], {}
    for record in records:
        source, record_id = record["_meta"]["source"], record["_meta"]["id"]
        texts = state_texts(record["state"])
        for question in record["questions"].values():
            texts.append(("instruction", question.get("instructions") or ""))
            criteria = question.get("criteria") or {}
            for value in (criteria.values() if isinstance(criteria, dict) else criteria):
                texts.append(("option", value or ""))
        for kind, text in texts:
            if not isinstance(text, str):
                continue
            key = normalise_text(text)
            if len(key) < 1:
                continue
            if key in seen:
                seen[key]["records"] += 1
                continue
            unit = {"source": source, "kind": kind, "record": record_id, "text": key, "records": 1}
            seen[key] = unit
            units.append(unit)
    return units


def matches_in(case_text, units, min_chars):
    """Every unit that is this case's text, contains it, or is contained in it: the naive double loop of the ticket,
    yielded as (mode, unit)."""
    case_len = len(case_text)
    for unit in units:
        text = unit["text"]
        if text == case_text:
            yield "exact", unit
        elif len(text) >= min_chars and len(text) <= case_len and text in case_text:
            yield "contains", unit
        elif len(text) > case_len and len(case_text) >= min_chars and case_text in text:
            yield "contained", unit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True, help="their prepare --with-text output dir")
    ap.add_argument("--suite", help="our suite directory (its train partition is the training side)")
    ap.add_argument("--split", default="train", help="partition of --suite to check (default train)")
    ap.add_argument("--records", help="a JSONL of labelled requests instead of --suite")
    ap.add_argument("--source", default="custom", help="source name load_records stamps on --records")
    ap.add_argument("--out", required=True, help="the JSON this script writes")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT,
                    help=f"shortest contained side a containment match needs (default {MIN_CHARS_DEFAULT})")
    ap.add_argument("--gate", type=float, default=GATE_DEFAULT,
                    help=f"a dataset over this share of matched cases is flagged (default {GATE_DEFAULT})")
    ap.add_argument("--examples", type=int, default=EXAMPLES_DEFAULT,
                    help="matches listed per dataset in the report")
    ap.add_argument("--limit-cases", type=int, help="check only the first N cases (debug)")
    a = ap.parse_args()
    if bool(a.suite) == bool(a.records):
        raise SystemExit("pass exactly one of --suite or --records")
    out = Path(a.out).resolve()

    directory, cases = their_cases(a.cases)
    records = load_split(Path(a.suite), a.split) if a.suite else load_records(a.records, source=a.source)
    units = our_units(records)
    order = sorted(cases)
    if a.limit_cases:
        order = order[:a.limit_cases]
    print(f"{len(order)} case texts vs {len(units)} unit texts from {len(records)} records "
          f"({len(units) * len(order)} pairs, min-chars {a.min_chars})", flush=True)

    per_dataset = defaultdict(lambda: {"n_cases": 0, "n_matched_cases": 0, "matched_case_ids": [],
                                       "by_split": defaultdict(lambda: {"n_cases": 0, "n_matched_cases": 0}),
                                       "by_source": defaultdict(int), "by_kind": defaultdict(int),
                                       "by_mode": defaultdict(int), "examples": []})
    started = time.monotonic()
    for index, case_id in enumerate(order, 1):
        case = cases[case_id]
        entry, text = per_dataset[case["task"]], normalise_text(case["text"])
        entry["n_cases"] += 1
        entry["by_split"][case["split"]]["n_cases"] += 1
        found = list(matches_in(text, units, a.min_chars))
        if found:
            entry["n_matched_cases"] += 1
            entry["by_split"][case["split"]]["n_matched_cases"] += 1
            entry["matched_case_ids"].append(case_id)
            for mode, unit in found:
                entry["by_source"][unit["source"]] += 1
                entry["by_kind"][unit["kind"]] += 1
                entry["by_mode"][mode] += 1
                if len(entry["examples"]) < a.examples:
                    entry["examples"].append({"case_id": case_id, "split": case["split"], "mode": mode,
                                              "our_source": unit["source"], "our_kind": unit["kind"],
                                              "our_record": unit["record"], "chars": len(unit["text"]),
                                              "unit": unit["text"][:200]})
        if index % 200 == 0 or index == len(order):
            print(f"{index}/{len(order)} cases, {time.monotonic() - started:.0f}s, "
                  f"{sum(e['n_matched_cases'] for e in per_dataset.values())} matched", flush=True)

    datasets = {}
    for task, entry in per_dataset.items():
        entry["by_split"] = {k: {**v, "rate": v["n_matched_cases"] / v["n_cases"] if v["n_cases"] else 0.0}
                             for k, v in entry["by_split"].items()}
        entry["rate"] = entry["n_matched_cases"] / entry["n_cases"] if entry["n_cases"] else 0.0
        entry["flagged"] = entry["rate"] > a.gate
        entry["by_source"] = dict(sorted(entry["by_source"].items(), key=lambda kv: -kv[1]))
        entry["by_kind"] = dict(sorted(entry["by_kind"].items(), key=lambda kv: -kv[1]))
        entry["by_mode"] = dict(sorted(entry["by_mode"].items(), key=lambda kv: -kv[1]))
        entry["same_name_sources"] = sorted(s for s in entry["by_source"] if compact(s) == compact(task))
        datasets[task] = entry
    flagged = sorted(task for task, entry in datasets.items() if entry["flagged"])
    result = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "their": {"directory": str(directory), "n_cases": len(order)},
              "ours": {"suite": a.suite, "split": a.split if a.suite else None, "records": a.records,
                       "source": a.source if a.records else None, "n_records": len(records), "n_units": len(units),
                       "by_kind": dict(Counter(unit["kind"] for unit in units)),
                       "by_source": dict(sorted(Counter(unit["source"] for unit in units).items(),
                                                key=lambda kv: -kv[1]))},
              "rule": {"normaliser": "kev.suite.normalise_text (casefold, whitespace collapsed)",
                       "modes": {"exact": "normalised texts equal",
                                 "contains": "our unit occurs in their case text",
                                 "contained": "their case text occurs in our unit"},
                       "min_chars": a.min_chars, "scan": "naive double loop, both directions"},
              "gate": a.gate, "flagged_datasets": flagged, "per_dataset": datasets,
              "notes": [f"a dataset is flagged when more than {a.gate:.0%} of its cases matched; "
                        "luwu_policies --overlap excludes flagged datasets unless --drop-overlapping-cases is given",
                        "by_source counts matches, not cases: one case can match units of several sources",
                        f"units are deduplicated on the normalised text; {len(units) * len(order)} pairs were compared, "
                        "so a run over a large partition takes minutes"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)
    print(json.dumps({"out": str(out), "seconds": round(time.monotonic() - started, 1),
                      "flagged_datasets": flagged,
                      "rates": {task: round(entry["rate"], 5) for task, entry in sorted(datasets.items())}}, indent=2))


if __name__ == "__main__":
    main()
