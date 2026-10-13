"""Convert an open pairwise preference dataset into `kev --data` records for arm HP.

    .venv/bin/python scripts/pref/convert_pref_data.py --out runs/pref/data/pref-helpsteer3.jsonl

Arm HP (docs/hypotheses/pointwise-chat-input.md) is arm H plus general preference pairs: `--data <jsonl> --replay 12576`
mixes them with the decision-v7 training partition, so the preference loss also sees the chat pointwise format on real
human preference judgements (and `--replay` keeps the released recipe in the run).

Source: `nvidia/HelpSteer3` (CC-BY-4.0), split `preference`; the revision is resolved at run time and recorded in the
manifest. The dataset this study started from, `Skywork/Skywork-Reward-Preference-80K-v0.2`, is not used: its card
declares no license (``license: None``), while HelpSteer3 is CC-BY-4.0, human-annotated (one `overall_preference` per
pair, -3..3, plus the individual annotators'), and its `context` is already a chat message list (single- and
multi-turn), which is the state shape Kev renders.

One record per pair: the conversation is the state, and there is one choice question whose options are the two
responses, the better one on a seeded-random key:

    {"state": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...],
     "questions": {"better": {"type": "choice", "instructions": "Which response is better?", "criteria": {"a": "...", "b": "..."},
                              "label": "a", "src": "helpsteer3_pref_general"}}}

`overall_preference` 0 ("about the same") is not a preference and is dropped. Admission is the training rule every
record passes (`kev.train.training_requests`): `kev.model.fits(materialize(record), tokenizer, **training_context(max_state))`
with `kev.suite.ADMISSION_BRANCH_HEADROOM` reserved on the branch — the headroom every frozen suite was admitted under,
because a training-time `kev.data.augment` can add an option (a none option, p 0.12, or a distractor, p 0.15) that the
converted pair does not carry yet, and such a pair must still encode when the trainer augments it. The format arm HP
trains is checked too (`kev.model.encode` on the chat layout: the row must fit `max_branch + CHAT_ROW_SLACK`).
The manifest records how many rows each limit dropped. At the default 384-token state and 1024-token branch, ~13,000 of
the 36,299 non-tie pairs fit (estimate from the character lengths: the Kev branch holds the instruction and BOTH
responses), so the default `--cap 12000` (the size of decision-v7's training partition) samples rather than truncates.

Decontamination: RewardBench 2 is this study's ranking benchmark (scripts/pref/rewardbench2.py), so a pair is dropped
when its prompt text matches a RewardBench 2 prompt exactly after normalization (casefold + whitespace collapse, the
normalization `kev.data.load_records` hashes states with). The prompt text of a pair is its last user turn, and its
whole rendered state is checked too. Exact match is a weak filter (the Skywork v0.2 release used n-gram overlap against
RewardBench 1) and is all this converter promises.

CPU-only; the manifest (default `<out>.manifest.json`) is the provenance: dataset and tokenizer revisions, counts, drop
reasons, the context the records were admitted against, the instruction, and the output file's sha256.
"""
import argparse, hashlib, json, random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from kev.api import render
from kev.data import load_records, materialize
from kev.model import MAX_STATE, ContextOverflow, encode, fits, load_tokenizer, training_context
from kev.suite import ADMISSION_BRANCH_HEADROOM

DATASET, CONFIG, SPLIT = "nvidia/HelpSteer3", "preference", "train"
BENCHMARK, BENCHMARK_SPLIT = "allenai/reward-bench-2", "test"
INSTRUCTION = "Which response is better?"      # also scripts/pref/rewardbench2.py: training and scoring read the same prompt
QID, KEYS = "better", ("a", "b")
BASE, BASE_REVISION = "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"   # as arms.sh COMMON pins it
ROLES = ("system", "user", "assistant")


def normalize(text):
    """The decontamination key: casefolded, whitespace collapsed (as kev.data.load_records normalizes a state)."""
    return " ".join(text.casefold().split())


def resolve(repo, revision):
    """The dataset revision to pin: the caller's, or the repo's current main resolved here (recorded in the manifest)."""
    return revision or HfApi().dataset_info(repo).sha


def benchmark_prompts(repo, split, revision):
    """Normalized RewardBench 2 prompt texts ({} when --benchmark is empty: no decontamination)."""
    if not repo: return set()
    return {normalize(r["prompt"]) for r in load_dataset(repo, split=split, revision=revision)}


def state_of(row):
    """The pair's conversation as Kev's state: the message list, with every role checked (a role Kev does not render
    would be flattened silently)."""
    out = []
    for m in row["context"]:
        role, content = m.get("role"), m.get("content")
        if role not in ROLES or not isinstance(content, str):
            raise ValueError(f"context message {m!r}: expected one of {ROLES} and a string content")
        out.append({"role": role, "content": content})
    return out


def record_of(row, key_rng):
    """-> (labelled request, None) for one pair, or (None, reason) when the pair carries no usable preference. The better
    response goes to a seeded-random key, so neither the option key nor its position in the listing says which is better."""
    pref = row.get("overall_preference")
    if isinstance(pref, bool) or not isinstance(pref, (int, float)): return None, "no_preference"
    if pref == 0: return None, "tie"                    # "Response 1 is about the same as Response 2"
    better, worse = (row["response1"], row["response2"]) if pref < 0 else (row["response2"], row["response1"])
    if not (better or "").strip() or not (worse or "").strip(): return None, "empty_response"
    pair = [(better, True), (worse, False)]
    key_rng.shuffle(pair)
    rec = {"state": state_of(row), "questions": {QID: {"type": "choice", "instructions": INSTRUCTION,
           "criteria": {k: text for k, (text, _) in zip(KEYS, pair)},
           "label": KEYS[next(i for i, (_, is_better) in enumerate(pair) if is_better)],
           "src": f"helpsteer3_pref_{row.get('domain') or 'unknown'}"}}}
    return rec, None


def overflow_reason(tok, rec, ctx):
    """Which training-context limit a record breaks: `fits` is the rule, this only names the reason ('state', 'branch',
    'packed') so the manifest says what the cap costs."""
    try:
        enc = encode(tok, rec, max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
    except ContextOverflow as e:
        return "state" if e.max_state is not None else "branch"
    return "packed" if len(enc["ids"]) > ctx["max_packed"] else None


def admission_context(ctx):
    """The training context with kev.suite.ADMISSION_BRANCH_HEADROOM reserved on the branch, the rule every frozen suite
    was admitted under: a training-time augment (kev.data.augment) can add one option to a Choice question (a none option,
    p 0.12, or a distractor, p 0.15), which the converted pair does not carry yet. For the `chat` layout the added option
    also lengthens the prefix's listing, so the headroom covers the chat row too; under `chat_pointwise` (arm HP) the row
    holds the graded option alone and the reserve is only conservative."""
    return {**ctx, "max_branch": ctx["max_branch"] - ADMISSION_BRANCH_HEADROOM}


def chat_row_fits(tok, rec, ctx):
    """True when the record's chat row fits the row limit encode enforces on the format arm HP trains (max_branch +
    CHAT_ROW_SLACK). `fits` measures only the Kev encoding, which holds the question's whole option list, so it is the
    stricter of the two: this check exists so nothing the converter writes can make kev.train's own encode raise."""
    try:
        encode(tok, rec, max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True, chat=True, chat_listing=False)
        return True
    except ContextOverflow:
        return False


def pcts(values):
    """{'n', 'mean', 'p50', 'p90', 'p99', 'max'} of a token-count list (the admitted population's shape)."""
    if not values: return {"n": 0}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1)))]
    return {"n": len(v), "mean": round(sum(v) / len(v), 1), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": v[-1]}


def main():
    ap = argparse.ArgumentParser(description="Convert an open pairwise preference dataset into kev --data records (arm HP)")
    ap.add_argument("--out", required=True, help="output JSONL, one kev labelled request per line")
    ap.add_argument("--manifest", default="", help="provenance JSON (default <out>.manifest.json)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--revision", default="", help="dataset revision to pin (default: resolve the repo's main and record it)")
    ap.add_argument("--cap", type=int, default=12000, help="write at most this many pairs, sampled by --seed from the admitted ones (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_state", type=int, default=MAX_STATE, help=f"state limit of the training context the records are admitted against (the arm's --max_state; default {MAX_STATE})")
    ap.add_argument("--base", default=BASE, help="tokenizer base (the arm's --base)")
    ap.add_argument("--base_revision", default=BASE_REVISION, help="tokenizer revision to pin (the arm's --base_revision)")
    ap.add_argument("--benchmark", default=BENCHMARK, help="decontamination reference (RewardBench 2); empty disables it")
    ap.add_argument("--benchmark_split", default=BENCHMARK_SPLIT)
    ap.add_argument("--benchmark_revision", default="", help="revision to pin for the benchmark (default: resolve main)")
    ap.add_argument("--limit", type=int, default=0, help="read at most this many rows (debug)")
    args = ap.parse_args()

    ctx = training_context(args.max_state)
    actx = admission_context(ctx)   # kev.suite.ADMISSION_BRANCH_HEADROOM reserved: a training augment can add an option
    tok = load_tokenizer(args.base, args.base_revision or None)
    rev = resolve(args.dataset, args.revision)
    rows = load_dataset(args.dataset, args.config, split=args.split, revision=rev)
    if args.limit: rows = rows.select(range(min(args.limit, len(rows))))
    bench_rev = resolve(args.benchmark, args.benchmark_revision) if args.benchmark else ""
    bench = benchmark_prompts(args.benchmark, args.benchmark_split, bench_rev)
    print(f"{args.dataset}@{rev[:8]} {args.config}/{args.split}: {len(rows)} rows; decontaminating against "
          f"{len(bench)} {args.benchmark or '(disabled)'} prompts; context {ctx}", flush=True)

    key_rng, cap_rng = random.Random(f"keys:{args.seed}"), random.Random(f"cap:{args.seed}")
    drops, admitted, state_tokens, record_tokens = {}, [], [], []
    for row in rows:
        rec, reason = record_of(row, key_rng)
        # the pair's own prompt, and the rendered state (a multi-turn context could match one inside its history)
        if rec is not None and bench & {normalize(rec["state"][-1]["content"]), normalize(render(rec["state"]))}:
            reason = "decontaminated"; rec = None
        if rec is None:
            drops[reason] = drops.get(reason, 0) + 1; continue
        internal = materialize(rec)
        if not fits(internal, tok, **actx):                      # kev.train.training_requests' rule, same call, headroom reserved
            reason = overflow_reason(tok, internal, actx) or "packed"
            drops[reason] = drops.get(reason, 0) + 1; continue
        if not chat_row_fits(tok, internal, actx):               # the row arm HP's own encode reads
            drops["chat_row"] = drops.get("chat_row", 0) + 1; continue
        enc = encode(tok, internal, max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
        admitted.append(rec); state_tokens.append(enc["state_tokens"]); record_tokens.append(len(enc["ids"]))

    kept = admitted
    if args.cap and len(admitted) > args.cap:
        kept = [admitted[i] for i in sorted(cap_rng.sample(range(len(admitted)), args.cap))]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in kept: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # read the file back the way kev.train does (--data -> load_records -> materialize): the schema is checked here, not
    # on the training machine, and the options must come back exactly as the criteria were written
    back = load_records(str(out), source="helpsteer3")
    if len(back) != len(kept): raise SystemExit(f"{out}: {len(back)} records read back, {len(kept)} written")
    first = materialize(back[0])
    if first["questions"][0]["options"] != [f"{k}: {v}" for k, v in kept[0]["questions"][QID]["criteria"].items()]:
        raise SystemExit(f"{out}: the first record's options do not read back as its criteria")
    if not fits(first, tok, **actx): raise SystemExit(f"{out}: the first record does not fit the admission context it was admitted against")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "out": str(out), "records": len(kept), "bytes": out.stat().st_size, "sha256": digest,
        "dataset": {"repo": args.dataset, "config": args.config, "split": args.split, "revision": rev, "rows": len(rows), "license": "cc-by-4.0"},
        "tokenizer": {"base": args.base, "revision": args.base_revision},
        "admission": {"rule": "kev.model.fits(materialize(record), tokenizer, **admission_context(training_context(max_state))): "
                              "kev.train.training_requests' rule with kev.suite.ADMISSION_BRANCH_HEADROOM reserved on the branch, "
                              "the headroom every frozen suite was admitted under (a training augment can add one option to a "
                              "Choice question), plus kev.model.encode's chat-row limit on the format the arm trains",
                      "context": ctx, "branch_headroom": ADMISSION_BRANCH_HEADROOM, "admitted": len(admitted), "written": len(kept),
                      "cap": args.cap, "seed": args.seed, "dropped": drops},
        "record": {"question": QID, "type": "choice", "instructions": INSTRUCTION, "keys": list(KEYS),
                   "label": "the better response's key (which response is better is assigned to a random key)",
                   "state": "the pair's conversation, as Kev's message-list state (kev.api.render flattens it)"},
        "tokens": {"state": pcts(state_tokens), "packed": pcts(record_tokens)},
        "decontamination": {"benchmark": args.benchmark, "split": args.benchmark_split, "revision": bench_rev, "prompts": len(bench),
                            "rule": "normalized (casefolded, whitespace-collapsed) exact match of the last user turn or of the rendered state"},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    man = Path(args.manifest) if args.manifest else out.with_suffix(out.suffix + ".manifest.json")
    man.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"admitted {len(admitted)} of {len(rows)} rows, wrote {len(kept)} to {out} ({out.stat().st_size / 1e6:.1f} MB): "
          + ", ".join(f"{k} {v}" for k, v in sorted(drops.items())), flush=True)
    if args.cap and len(admitted) < args.cap:
        print(f"note: only {len(admitted)} pairs fit a {ctx['max_state']}-token state / {ctx['max_branch']}-token branch context "
              f"(cap {args.cap} not reached); raise --max_state and the arm's --max_state together to admit more", flush=True)
    print(f"manifest: {man} (sha256 {digest[:16]})", flush=True)


if __name__ == "__main__":
    main()
