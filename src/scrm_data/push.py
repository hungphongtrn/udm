"""Append built shards to the Hub dataset repo (additive only) and regenerate the README metadata.

    python -m scrm_data.push --out <out> [--dry-run] [--repo hungphongtrn/udm-massive-typed]

* Never deletes or overwrites data: every target path is checked against the repo listing and the push
  aborts if any already exists. Only README.md is rewritten (YAML `dataset_info` counts + one added section);
  the previous README is saved to receipts/README.before-<timestamp>.md in the repo as well.
* Commits are chunked (--batch-files / --batch-bytes) so a failure leaves a consistent, resumable state:
  re-running skips nothing silently - existing paths abort, so use --skip-existing to resume after a failure.
* Token: HF_TOKEN environment variable or `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
from typing import Optional

import pyarrow.parquet as pq

from . import TARGET_REPO
from .validate import NAME_RE

MARK_BEGIN = "<!-- scrm-data:added-sources:begin -->"
MARK_END = "<!-- scrm-data:added-sources:end -->"
BASE_RE = re.compile(r"<!-- scrm-base: (\{.*?\}) -->")


def plan_files(out: str) -> list:
    """[(local_path, path_in_repo, size)] for data shards + receipts."""
    items = []
    for p in sorted(glob.glob(os.path.join(out, "data", "*.parquet"))):
        if not NAME_RE.match(os.path.basename(p)):
            raise SystemExit(f"unexpected file name in data/: {p}")
        items.append((p, "data/" + os.path.basename(p), os.path.getsize(p)))
    for p in sorted(glob.glob(os.path.join(out, "receipts", "*.json"))):
        items.append((p, "receipts/" + os.path.basename(p), os.path.getsize(p)))
    return items


def shard_stats(out: str) -> dict:
    """Per split: num_examples, num_bytes (sum of uncompressed parquet row-group bytes, an approximation of the
    arrow size HF reports) and download bytes (file sizes) of the NEW shards."""
    st = {sp: {"n": 0, "b": 0, "dl": 0} for sp in ("train", "validation", "test")}
    for p in glob.glob(os.path.join(out, "data", "*.parquet")):
        sp = NAME_RE.match(os.path.basename(p)).group(1)
        md = pq.ParquetFile(p).metadata
        st[sp]["n"] += md.num_rows
        st[sp]["b"] += sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
        st[sp]["dl"] += os.path.getsize(p)
    return st


def _get(text: str, split: str, field: str) -> int:
    m = re.search(rf"- name: {split}\n\s+num_bytes: (\d+)\n\s+num_examples: (\d+)", text)
    if not m:
        raise ValueError(f"cannot find split {split} in README dataset_info")
    return int(m.group(1)) if field == "b" else int(m.group(2))


def update_readme_text(old: str, new_stats: dict, receipts: dict) -> str:
    """Return the README with dataset_info counts = (original counts) + (new shards) and the added section."""
    m = BASE_RE.search(old)
    if m:  # already updated once: use the recorded base counts so re-runs do not double count
        base = json.loads(m.group(1))
        body = old[: old.index(MARK_BEGIN)].rstrip("\n") + "\n" if MARK_BEGIN in old else old
    else:
        base = {sp: {"n": _get(old, sp, "n"), "b": _get(old, sp, "b")} for sp in ("train", "validation", "test")}
        base["download_size"] = int(re.search(r"download_size: (\d+)", old).group(1))
        body = old
    text = body
    total_b = 0
    for sp in ("train", "validation", "test"):
        nb = base[sp]["b"] + new_stats[sp]["b"]
        nn = base[sp]["n"] + new_stats[sp]["n"]
        total_b += nb
        text = re.sub(rf"(- name: {sp}\n\s+num_bytes: )\d+(\n\s+num_examples: )\d+", rf"\g<1>{nb}\g<2>{nn}", text)
    dl = base["download_size"] + sum(v["dl"] for v in new_stats.values())
    text = re.sub(r"download_size: \d+", f"download_size: {dl}", text)
    text = re.sub(r"dataset_size: \d+", f"dataset_size: {total_b}", text)
    section = render_section(new_stats, receipts, base)
    return text.rstrip("\n") + "\n\n" + section


def render_section(new_stats: dict, receipts: dict, base: dict) -> str:
    lines = [MARK_BEGIN, f"<!-- scrm-base: {json.dumps(base, sort_keys=True)} -->", "",
             "## Added sources (SCRM data pipeline)", "",
             "Additional decision sets were appended as new shards `data/{split}-{source_slug}-NNNNN-of-MMMMM.parquet` "
             "(the `configs` globs are unchanged and pick them up). Every row uses the schema above; all sources are "
             "framed as classification / ranking decision sets: `state_json` + `instruction_json` + candidate set "
             "(`options_json`) + best-first `tier_json`. Only pairs `tier_i < tier_j` carry preference signal.", "",
             "| source slug | upstream | rows (train / validation / test) | label kinds |", "|---|---|---|---|"]
    for slug, r in sorted(receipts.items()):
        rw = r["rows_written"]
        kinds = sorted({k for sp in r["label_kinds"].values() for k in sp})
        lines.append(f"| `{slug}` | `{r['source_repo']}` @ `{r['source_revision'][:8]}` | "
                     f"{rw.get('train', 0):,} / {rw.get('validation', 0):,} / {rw.get('test', 0):,} | {', '.join(kinds)} |")
    lines += ["", "### Label kinds", "",
              "| `label_kind` | tiers | `probabilities_json` | `source_label_or_null` |", "|---|---|---|---|",
              "| `source_dataset_label` | `[[winner],[rest]]` | `{}` | `{\"kind\":\"choice\",\"target\":[...],\"winner\":[id]}` |",
              "| `soft_choice_distribution` | distinct probability groups, descending | the distribution | `{\"kind\":\"choice_soft\",\"target\":[...]}` |",
              "| `ordinal_score_distance_tiers` | tier k = levels at ordinal distance k from the true level | raw distribution over levels | `{\"kind\":\"score\",\"levels\":[...],\"level_choice_ids\":[...],\"target\":[...],\"true_level_index\":int,\"expected_level\":float}` |",
              "| `independent_labels_binary_tiers` | `[[p>=0.5],[p<0.5]]` (yes/no: candidates `no`,`yes`) | independent probabilities | `{\"kind\":\"noul\",\"target\":[...],\"threshold\":0.5}` |",
              "| `agent_choice_target` | `[[target],[rest]]` | `{}` | `{\"kind\":\"agent_choice\",\"target\":{...},\"labels\":{...},\"decision_type\":...}` |",
              "", "### Metadata conventions", "",
              "* `source_label_or_null` always holds canonical JSON with the ORIGINAL supervision, so `score` and `noul` "
              "information is recoverable (original target vector, level texts, expected level, threshold).",
              "* `source_split` keeps the raw upstream split (`calibration`, `ood`, ... for Open-Jev); filter on it to "
              "exclude e.g. OOD rows. `partition_role` is `train` / `dev` / `test`.",
              "* `family` is `tasksource:<task>`, `openjev:<task>`, or `jev_agent:<dataset>[/<config>]`.",
              "* New hash recipes (`decision_set_id`, `record_id`, `raw_record_sha256`) are documented in `docs/DATA_SPEC.md` "
              "of the data pipeline repository; `choice_id == sha256(option text)` as before.",
              "* Per-source build receipts (counts, drop reasons, leakage audit) are in `receipts/`.",
              MARK_END, ""]
    return "\n".join(lines)


def _retry(fn, tries=6, what=""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            last = e
            print(f"  retry {i + 1}/{tries} ({what}): {e}", file=sys.stderr)
            time.sleep(min(10 * 2 ** i, 300))
    raise RuntimeError(f"giving up: {what}: {last}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=TARGET_REPO)
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; upload nothing")
    ap.add_argument("--batch-files", type=int, default=25)
    ap.add_argument("--batch-bytes", type=int, default=12 * 1024 ** 3)
    ap.add_argument("--skip-existing", action="store_true", help="resume: skip paths that already exist (same name) instead of aborting")
    ap.add_argument("--skip-readme", action="store_true")
    ap.add_argument("--token", default=None)
    a = ap.parse_args(argv)

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=a.token or os.environ.get("HF_TOKEN"))
    items = plan_files(a.out)
    if not items:
        raise SystemExit("nothing to push")
    existing = set(_retry(lambda: api.list_repo_files(a.repo, repo_type="dataset"), what="list repo"))
    conflicts = [r for _, r, _ in items if r in existing]
    if conflicts and not a.skip_existing:
        print("ABORT: target paths already exist in the repo (never overwriting):", file=sys.stderr)
        for c in conflicts[:30]:
            print("  ", c, file=sys.stderr)
        return 2
    todo = [(p, r, s) for p, r, s in items if r not in existing]
    total = sum(s for _, _, s in todo)
    print(f"[push] repo={a.repo} files_to_add={len(todo)} (skipped existing={len(items) - len(todo)}) bytes={total / 1e9:.2f} GB")
    batches, cur, cb = [], [], 0
    for it in todo:
        if cur and (len(cur) >= a.batch_files or cb + it[2] > a.batch_bytes):
            batches.append(cur)
            cur, cb = [], 0
        cur.append(it)
        cb += it[2]
    if cur:
        batches.append(cur)
    for i, b in enumerate(batches, 1):
        print(f"  commit {i}/{len(batches)}: {len(b)} files, {sum(x[2] for x in b) / 1e9:.2f} GB; first={b[0][1]} last={b[-1][1]}")

    receipts = {}
    for p in glob.glob(os.path.join(a.out, "receipts", "*.json")):
        if not p.endswith("summary.json"):
            r = json.load(open(p))
            receipts[r["source"]] = r
    stats = shard_stats(a.out)
    readme_path = hf_hub_download(a.repo, "README.md", repo_type="dataset", token=api.token, force_download=True)
    old = open(readme_path, encoding="utf-8").read()
    new = update_readme_text(old, stats, receipts)
    print("[push] README dataset_info after update:")
    for sp in ("train", "validation", "test"):
        print(f"   {sp}: +{stats[sp]['n']:,} rows, +{stats[sp]['b'] / 1e9:.2f} GB (uncompressed parquet bytes)")
    if a.dry_run:
        print("[push] dry-run: nothing uploaded. README section preview:\n")
        print(new[new.index(MARK_BEGIN):][:2500])
        return 0
    for i, b in enumerate(batches, 1):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=p) for p, r, _ in b]
        _retry(lambda: api.create_commit(a.repo, ops, repo_type="dataset",
                                         commit_message=f"Add SCRM data shards ({i}/{len(batches)})"), what=f"commit {i}")
        print(f"[push] committed {i}/{len(batches)}", flush=True)
    if not a.skip_readme:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        ops = [CommitOperationAdd(path_in_repo=f"receipts/README.before-{stamp}.md", path_or_fileobj=io.BytesIO(old.encode())),
               CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(new.encode()))]
        _retry(lambda: api.create_commit(a.repo, ops, repo_type="dataset",
                                         commit_message="Update README dataset_info and describe added SCRM sources"), what="readme")
        print("[push] README updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
