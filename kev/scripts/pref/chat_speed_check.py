"""Score a chat-format checkpoint twice on the same rows -- the eager chat path (DecisionModel.chat_hidden) and the
graphed one (kev.cuda_graphs.chat, what scripts/pref/dindex.sh runs by default) -- and report the reward / probability
difference, each path's wall time and the projected Decision Index subset time. That subset is 15,000 requests (hours of
GPU time at a few ms each), so this is the same question on N rows of the same file.

    .venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0                    # 20 rows of the DI subset
    .venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --n 100            # more rows, both paths
    .venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --offset 100       # a different slice of the rows
    .venv/bin/python scripts/pref/chat_speed_check.py runs/pref/D/seed0 --suite evals/decision-v7 --split calibration

--rows defaults to $DI_DIR/subset-0.3-15000.jsonl.gz (DI_DIR: the kit checkout dindex.sh uses, ~/decision-index by
default) and reads only `state` and `questions` from it, with the placeholder labels dindex_engine.KevEngine passes
(labels are never scored). --suite scores records of a kev suite split instead. Both go through
kev.predictors.LocalPredictor at the serving context, bf16 and the checkpoint's own temperature, exactly as
dindex_engine builds it (device cuda, graphs on).

Numerical equivalence: both paths run the same tokens at the same positions -- the record's state once, each question's
prefix once, its option rows continuing that (kev.cuda_graphs: the module docstring's passes; the prefixes run eagerly and
unpadded there, one pass per distinct length). What differs is floating-point reassociation: the graphed passes pad rows to
buckets and the DeltaNet scan is chunked over a different batch, so logits move by ~1e-3 in bf16 (the same order as the
training-to-serving differences the docstrings quote) and probabilities by less; fp32 is orders of magnitude tighter.
Answer flips should be none: report them if not. Records the graphed passes do not admit (kev.cuda_graphs.admits_chat: a
state, a question prefix or an option row past the buffers) fall back to the eager passes in the graphed phase too -- the
count is printed.
"""
import argparse
import gzip
import json
import os
import statistics
import time
from pathlib import Path

import torch

from kev.checkpoint import LoadOptions
from kev.data import materialize
from kev.device import sync
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT, load_split

SUBSET = "subset-0.3-15000.jsonl.gz"   # dindex.sh's shared comparison subset (N=15000 rows), in the kit checkout
SUBSET_ROWS = 15000                    # what the projection below is for


def di_records(path, n, offset):
    """-> (rows of the kit's subset file as the records LocalPredictor scores, how many were skipped). `state` and the
    `questions` object are passed through as given (that is what dindex_engine does), each question with the label and src
    kev.data.materialize reads; the kit's rows carry no labels, so choice gets its first criterion, noul False. Questions
    of other types are skipped: the engine refuses them, so they are not part of a speed comparison."""
    rows, skipped = [], 0
    with (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")) as f:
        for i, line in enumerate(f):
            if i < offset or not line.strip(): continue
            row = json.loads(line)
            if any(q["type"] not in ("choice", "noul") for q in row["questions"].values()): skipped += 1; continue
            rows.append({"state": row["state"], "questions": {
                qid: {**q, "label": next(iter(q["criteria"])) if q["type"] == "choice" else False, "src": "decision-index"}
                for qid, q in row["questions"].items()}})
            if len(rows) == n: break
    return rows, skipped


def score(predictor, records, graphs, device):
    """Score every record, one at a time, as the kit's runner does (single-request batches; the graphs batch within a
    record) -> (the predictor's outputs, per-record seconds). `graphs` None is the eager path (model.graphs is what
    DecisionModel._chat_logits dispatches on); otherwise capture runs as dindex_engine.KevEngine runs it, one bucket after
    each request. A capture's time is in the phase's clock, not in the record's own time below."""
    predictor.model.graphs = graphs
    out, seconds = [], []
    for rec in records:
        sync(device); start = time.perf_counter()
        out.append(predictor(rec))
        sync(device); seconds.append(time.perf_counter() - start)
        if graphs is not None and graphs.capture_due(idle=False): graphs.capture_pending(limit=1)
    return out, seconds


def differences(ref, got, records):
    """-> (max |d probability|, max |d reward| (the head's logits, kev.predictors' `logits`), answer flips, questions
    compared). A flip is a choice question whose argmax moved (the engine answers choice with an argmax); noul answers are
    a probability, so their argmax is not an answer."""
    dp = dl = 0.0
    flips = questions = 0
    for a, b, rec in zip(ref, got, records):
        for qid, q in rec["questions"].items():
            pa, pb, za, zb = a["probabilities"][qid], b["probabilities"][qid], a["logits"][qid], b["logits"][qid]
            dp = max(dp, max(abs(pa[k] - pb[k]) for k in pa))
            dl = max(dl, max(abs(za[k] - zb[k]) for k in za))
            questions += 1
            if q["type"] == "choice" and max(pa, key=pa.get) != max(pb, key=pb.get): flips += 1
    return dp, dl, flips, questions


def line(label, seconds, total, rows=SUBSET_ROWS):
    per = statistics.mean(seconds)
    return (f"{label:<10} {1000 * per:>9.1f} {1000 * statistics.median(seconds):>10.1f} {1 / per:>11.2f} "
            f"{total / len(seconds) * rows / 3600:>15.1f}")


def main():
    ap = argparse.ArgumentParser(description="The eager chat path against the graphed one, on the same rows.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("run", help="checkpoint dir or Hub id[@rev]: a chat-format checkpoint (arms C D E F H HP)")
    ap.add_argument("--rows", help=f"the kit's rows file (default $DI_DIR/{SUBSET})")
    ap.add_argument("--suite", help="a kev suite directory to score instead (evals/decision-v7)")
    ap.add_argument("--split", default="development", help="the suite split to score (default development)")
    ap.add_argument("--n", type=int, default=20, help="records to score (default 20)")
    ap.add_argument("--offset", type=int, default=0, help="skip this many rows first (default 0)")
    ap.add_argument("--device", default="cuda", help="cuda (default), cpu or mps")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"), help="bf16 as dindex.sh runs (default) or fp32")
    args = ap.parse_args()
    if args.suite and args.rows: raise SystemExit("--rows and --suite are mutually exclusive: score one of them")

    if args.suite:
        records, skipped = load_split(Path(args.suite), args.split)[args.offset:args.offset + args.n], 0
    else:
        kit = Path(os.environ.get("DI_DIR") or Path.home() / "decision-index")
        records, skipped = di_records(Path(args.rows).expanduser() if args.rows else kit / SUBSET, args.n, args.offset)
    if not records: raise SystemExit("no records selected")
    print(f"{len(records)} records" + (f" ({skipped} skipped: a question type the engine refuses)" if skipped else ""), flush=True)

    print(f"loading {args.run} on {args.device} in {args.dtype} ...", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    predictor = LocalPredictor(args.run, args.device, LoadOptions(dtype=dtype, cuda_graphs=args.device == "cuda"), context=SERVING_CONTEXT)
    model, graphs = predictor.model, predictor.model.graphs
    if graphs is None:
        raise SystemExit(f"{args.run} got no graphed chat passes: they need the chat input format on a hybrid backbone on CUDA")
    encs = [model.encode(predictor.tok, materialize(r), max_state=SERVING_CONTEXT["max_state"], max_branch=SERVING_CONTEXT["max_branch"], strict=True) for r in records]
    fallbacks = sum(not graphs.admits_chat(e["chat"]) for e in encs)
    tokens = [len(e["ids"]) for e in encs]
    print(f"model: {model.dtype} {type(model.lm).__name__}, head {model.head_kind}, temperature {predictor.temperature}, "
          f"tokens/record {statistics.mean(tokens):.0f} avg {max(tokens)} max, "
          f"records the graphs do not admit (eager both ways): {fallbacks}", flush=True)

    t = time.perf_counter(); ref, ref_s = score(predictor, records, None, args.device); ref_total = time.perf_counter() - t
    t = time.perf_counter(); fast, fast_s = score(predictor, records, graphs, args.device); fast_total = time.perf_counter() - t
    dp, dl, flips, questions = differences(ref, fast, records)

    columns = f"{'path':<10} {'mean ms':>9} {'median ms':>10} {'records/s':>11} {'projected subset h':>19}"
    print(f"\n{columns}   (the subset: {SUBSET_ROWS} rows; per-record times exclude captures)")
    print(line("eager", ref_s, ref_total))
    print(line("graphed", fast_s, fast_total))
    print(f"\nspeedup {ref_total / fast_total:.2f}x ({ref_total:.1f} s vs {fast_total:.1f} s over {len(records)} records); "
          f"max |d reward| {dl:.2e} (the head's logits), max |d prob| {dp:.2e}, answer flips {flips}/{questions} questions")
    print(f"{graphs.captures} graphs captured ({fast_total - sum(fast_s):.1f} s of the graphed clock), stats {graphs.stats()}")


if __name__ == "__main__":
    main()
