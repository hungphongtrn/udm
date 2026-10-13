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
(labels are never scored). --suite scores records of a kev suite split instead. Every record goes through
dindex_engine.KevEngine itself -- kev.predictors.LocalPredictor at the serving context, bf16 and the checkpoint's own
temperature -- so a record that runs out of memory takes the engine's chunked path (option rows in chunks of 64, 16, 4, 1;
on a 24 GB card the graphed buffers leave the eager passes less room) and is timed with it, and one the engine refuses
(Unsupported: past the serving context, out of memory at one row, another question type) is left out of both phases.
Each phase prints how many records took the chunked path and how many were refused.

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

from decision_index.engines import Unsupported
from dindex_engine import KevEngine   # scripts/pref, this script's own directory (sys.path[0])
from kev.data import materialize
from kev.device import sync
from kev.model import ContextOverflow
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


def score(engine, records, graphs, device):
    """Score every record, one at a time, as the kit's runner does (single-request batches; the graphs batch within a
    record) -> (per record {"answers", "logits"} or None when refused, per-record seconds of the scored ones, how many
    took the chunked out-of-memory path, the refusals). `graphs` None is the eager path (model.graphs is what
    DecisionModel._chat_logits dispatches on); otherwise capture runs as KevEngine._capture runs it, one bucket after
    each request that scored without the chunked path. A capture's time is in the phase's clock, not in the record's own
    time below (engine.graphs is None, so the engine itself never captures)."""
    engine.predictor.model.graphs = graphs
    out, seconds, chunked, refused = [], [], 0, []
    for i, rec in enumerate(records):
        sync(device); start = time.perf_counter()
        try:
            response, raw = engine(rec["state"], rec["questions"])
        except Unsupported as e:
            out.append(None); refused.append((i, str(e))); continue
        sync(device); seconds.append(time.perf_counter() - start)
        out.append({"answers": response["answers"], "logits": raw["logits"], "chunk": raw.get("fallback_chunk")})
        if "fallback_chunk" in raw: chunked += 1
        elif graphs is not None and graphs.capture_due(idle=False): graphs.capture_pending(limit=1)
    return out, seconds, chunked, refused


def probabilities(answer):
    return answer["probabilities"] if answer["type"] == "choice" else {"true": answer["noul"]}


def record_diff(a, b, rec):
    """One record both phases scored -> (max |d reward| (the head's logits, the engine's raw `logits`), max |d
    probability|, answer flips, questions). A flip is a choice question whose argmax moved (the engine answers choice with
    an argmax); noul answers are P(true), so their argmax is not an answer."""
    dl = dp = 0.0
    flips = 0
    for qid, q in rec["questions"].items():
        pa, pb = probabilities(a["answers"][qid]), probabilities(b["answers"][qid])
        za, zb = a["logits"][qid], b["logits"][qid]
        dp = max(dp, max(abs(pa[k] - pb[k]) for k in pa))
        dl = max(dl, max(abs(za[k] - zb[k]) for k in za))
        if q["type"] == "choice" and max(pa, key=pa.get) != max(pb, key=pb.get): flips += 1
    return dl, dp, flips, len(rec["questions"])


def path(a, b, admitted):
    """Which passes produced the graphed phase's answer for one record, against the eager phase's."""
    if a["chunk"] or b["chunk"]: return f"chunked (eager {a['chunk'] or '-'}, graphed {b['chunk'] or '-'})"
    return "graphed" if admitted else "eager both (not admitted)"


def line(label, seconds, total, rows=SUBSET_ROWS):
    per = statistics.mean(seconds)
    return (f"{label:<10} {1000 * per:>9.1f} {1000 * statistics.median(seconds):>10.1f} {1 / per:>11.2f} "
            f"{total / len(seconds) * rows / 3600:>15.1f}")


def attribution(ref, fast, records, admitted, tokens, offset):
    """Per path, the records' max |d reward| / |d prob| and flips, then the worst records: where the differences come from."""
    rows = []
    for i, (a, b, rec) in enumerate(zip(ref, fast, records)):
        if a is None or b is None: continue
        dl, dp, flips, nq = record_diff(a, b, rec)
        k = max(len(q.get("criteria") or ()) for q in rec["questions"].values())
        rows.append((dl, dp, flips, nq, i, path(a, b, admitted[i]), tokens[i], k))
    print(f"\n{'path':<34} {'records':>8} {'max |d reward|':>15} {'max |d prob|':>13} {'flips':>6}")
    for name in sorted({r[5] for r in rows}):
        group = [r for r in rows if r[5] == name]
        print(f"{name:<34} {len(group):>8} {max(r[0] for r in group):>15.2e} {max(r[1] for r in group):>13.2e} {sum(r[2] for r in group):>6}")
    print(f"\nworst records: {'record':>7} {'tokens':>7} {'questions':>9} {'max options':>11} {'|d reward|':>11} {'|d prob|':>9} {'flips':>6}  path")
    worst = sorted(rows, reverse=True)[:10]
    for dl, dp, flips, nq, i, name, tok, k in worst:
        print(f"{'':<14} {offset + i:>7} {tok if tok is not None else '-':>7} {nq:>9} {k:>11} {dl:>11.2e} {dp:>9.2e} {flips:>6}  {name}")
    return [r[4] for r in worst[:5]]


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
    engine = KevEngine(args.run, dtype=args.dtype, device=args.device, graphs=args.device == "cuda")
    predictor = engine.predictor
    model, graphs = predictor.model, predictor.model.graphs
    if graphs is None:
        raise SystemExit(f"{args.run} got no graphed chat passes: they need the chat input format on a hybrid backbone on CUDA")
    engine.graphs = None   # score() captures, outside the records' own times
    admitted, tokens = [], []
    for r in records:
        try: enc = model.encode(predictor.tok, materialize(r), max_state=SERVING_CONTEXT["max_state"], max_branch=SERVING_CONTEXT["max_branch"], strict=True)
        except ContextOverflow: admitted.append(False); tokens.append(None); continue   # the engine refuses it (counted below)
        admitted.append(graphs.admits_chat(enc["chat"])); tokens.append(len(enc["ids"]))
    fallbacks = sum(t is not None and not a for a, t in zip(admitted, tokens))
    known = [t for t in tokens if t is not None]
    if torch.cuda.is_available() and args.device == "cuda":
        print(f"GPU memory after loading (weights + graph buffers): {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated "
              f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)
    print(f"model: {model.dtype} {type(model.lm).__name__}, head {model.head_kind}, temperature {predictor.temperature}, "
          f"tokens/record {statistics.mean(known):.0f} avg {max(known)} max, "
          f"records the graphs do not admit (eager both ways): {fallbacks}", flush=True)

    t = time.perf_counter(); ref, ref_s, ref_chunked, ref_refused = score(engine, records, None, args.device); ref_total = time.perf_counter() - t
    t = time.perf_counter(); fast, fast_s, fast_chunked, fast_refused = score(engine, records, graphs, args.device); fast_total = time.perf_counter() - t
    scored = [(a, b, r) for a, b, r in zip(ref, fast, records) if a is not None and b is not None]
    per = [record_diff(*s) for s in scored]
    dl, dp = max((p[0] for p in per), default=0.0), max((p[1] for p in per), default=0.0)
    flips, questions = sum(p[2] for p in per), sum(p[3] for p in per)

    columns = f"{'path':<10} {'mean ms':>9} {'median ms':>10} {'records/s':>11} {'projected subset h':>19} {'chunked':>8} {'refused':>8}"
    print(f"\n{columns}   (the subset: {SUBSET_ROWS} rows; per-record times exclude captures)")
    print(line("eager", ref_s, ref_total) + f" {ref_chunked:>8} {len(ref_refused):>8}")
    print(line("graphed", fast_s, fast_total) + f" {fast_chunked:>8} {len(fast_refused):>8}")
    for label, refused in (("eager", ref_refused), ("graphed", fast_refused)):
        for i, why in refused: print(f"  {label} refused record {args.offset + i}: {why}")
    print(f"\nspeedup {ref_total / fast_total:.2f}x ({ref_total:.1f} s vs {fast_total:.1f} s over {len(records)} records); "
          f"max |d reward| {dl:.2e} (the head's logits), max |d prob| {dp:.2e}, answer flips {flips}/{questions} questions")
    print(f"{graphs.captures} graphs captured ({fast_total - sum(fast_s):.1f} s of the graphed clock), stats {graphs.stats()}")
    if args.device == "cuda": print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated")
    worst = attribution(ref, fast, records, admitted, tokens, args.offset)
    # the floor: the worst records scored eagerly once more, against the eager phase (is the eager path itself repeatable?)
    again, _, _, _ = score(engine, [records[i] for i in worst], None, args.device)
    print("\neager rerun of the worst records, against the eager phase (the run-to-run floor): "
          + ", ".join(f"record {args.offset + i} |d reward| {record_diff(ref[i], b, records[i])[0]:.2e}" + (" (chunked)" if b["chunk"] else "")
                      for i, b in zip(worst, again) if b is not None))


if __name__ == "__main__":
    main()
