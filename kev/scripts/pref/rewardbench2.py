"""RewardBench 2 for a Kev checkpoint (any arm: A Kev-format pointer head, D chat, H / HP chat_pointwise).

    .venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/H/seed0 --out runs/pref/H/seed0/rewardbench2.json

Scores allenai/reward-bench-2 (1865 prompts, 8977 candidates) with the checkpoint's own head, in-process through
kev.predictors.LocalPredictor under kev.suite.SERVING_CONTEXT, the path kev.serve answers the API with: a state past the
serving context is refused there and counted wrong here (`unsupported`), never truncated. One row per prompt, the
dataset's candidate order (the chosen responses first, then the rejected ones), the Ties subset included.

Scoring is the benchmark's own, transcribed from allenai/reward-bench (main, sha 5b958c9d9dc5533cd6bca9e212244a5f5ff07512):
`reroll_and_score_dataset` (rewardbench/utils.py L548: the chosen response wins when it is (one of) the highest-scored, and
the credit is shared among the candidates tied at the top), `process_single_model` (L1036) with `_compute_prompt_stats`
(L1013) for the Ties subset (0.30 tied accuracy + 0.30 reference accuracy + 0.20 correctness_preferred + 0.20
correctness_preferred_hard + 0.01 correctness_margin_score; the reference prompt of a tied prompt is the "ref:N" row with
the same N), and the overall score is the mean of the six subsets (scripts/run_v2.py L345-378).

The prompt layout is the checkpoint's input format, which is what the arms differ in:
- chat_pointwise (arms H / HP): every candidate is scored on its own, one request per candidate, one choice question with
  one option (key "a") and the instruction the preference data trains on (scripts/pref/convert_pref_data.py INSTRUCTION).
  No other candidate enters the prompt, so a response's reward cannot depend on the set it is scored in: that is what a
  pointwise reward model is, and `--probe` reports the invariance it must have.
- chat / kev (arms A / D and the released checkpoints): the prompt's candidates are the options of one question, keys
  "a", "b", ... in the dataset's order, and a candidate's reward is the head's readout for its option. The prompt then
  lists the other candidates, so the reward of a response does shift with the set (`--probe` measures the shift).

Temperature: the predictor runs at 1.0 (raw logits). Every metric here is a ranking or a margin between the same prompt's
candidates, so the checkpoint's fitted temperature cannot change any of them.

CUDA graphs: as dindex_engine runs them (on for a chat-format checkpoint on CUDA, `--graphs 0` opts out; `kev.cuda_graphs.chat`,
the same passes as the eager chat path, batched and padded to buckets, equal up to floating-point reassociation). The Kev
format's graphs path is the batched `probs_batch`, which this one-request-at-a-time loop never enters, so arm A keeps them off
(their buffers are ~3 GB). A record past the graphed buckets runs the eager chat path anyway.

Wall time is per request, device-synchronized (kev.predictors.LocalPredictor). The JSON holds the per-prompt scores, the
per-subset scores, the overall and no-Ties means, the unsupported counts, the input-token totals, whether the graphs ran
and the probe.
"""
import argparse, json, math, random, time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi

from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import CHAT_FORMATS, ContextOverflow
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT

BENCHMARK, SPLIT = "allenai/reward-bench-2", "test"
SOURCE = "allenai/reward-bench@5b958c9d9dc5533cd6bca9e212244a5f5ff07512"
INSTRUCTION = "Which response is better?"    # scripts/pref/convert_pref_data.py: the preference data trains on the same prompt
QID = "better"
SUBSETS = ("Factuality", "Precise IF", "Math", "Safety", "Focus", "Ties")
KEYS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"    # option keys, in the dataset's candidate order (chosen first)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def request(state, options, keys):
    """One kev request: the prompt as the state, one choice question whose options are `options` under `keys`."""
    return {"state": state, "questions": {QID: {"type": "choice", "instructions": INSTRUCTION,
            "criteria": dict(zip(keys, options)), "label": keys[0], "src": "rewardbench2"}}}


class Scorer:
    """Scores a prompt's candidates with one checkpoint. `pointwise` (chat_pointwise) scores every candidate on its own
    request; the set-relative formats score all of a prompt's candidates as the options of one question."""

    def __init__(self, predictor, pointwise):
        self.p, self.pointwise = predictor, pointwise

    def scores(self, state, candidates):
        """-> ([score per candidate], [unsupported per candidate], input tokens). A candidate (pointwise) or a prompt's
        question (the set-relative formats) past the serving context is scored 0.0 and flagged: the benchmark counts it
        wrong. The token count is the predictor's `input_tokens`: the kev-format encoding of each request (the state plus
        that request's options), so it counts the rendered chat rows' own tokens only through their options."""
        if self.pointwise:
            out, bad, ntok = [], [], 0
            for c in candidates:
                try:
                    got = self.p(request(state, [c], ["a"]))
                    out.append(got["logits"][QID]["a"]); bad.append(False); ntok += got["input_tokens"]
                except (ContextOverflow, torch.cuda.OutOfMemoryError):
                    out.append(0.0); bad.append(True)
            return out, bad, ntok
        keys = list(KEYS[: len(candidates)])
        if len(keys) < len(candidates): raise ValueError(f"{len(candidates)} candidates, at most {len(KEYS)} options per question")
        try:
            got = self.p(request(state, candidates, keys))
            z = got["logits"][QID]
            return [z[k] for k in keys], [False] * len(candidates), got["input_tokens"]
        except (ContextOverflow, torch.cuda.OutOfMemoryError):
            if self.p.device == "cuda": torch.cuda.empty_cache()
            return [0.0] * len(candidates), [True] * len(candidates), 0


def non_ties_result(scores):
    """reroll_and_score_dataset (rewardbench/utils.py L548): the chosen response is scores[0]; when it is (one of) the
    highest-scored candidates it earns 1 / (how many are tied at the top), else 0."""
    top = max(scores)
    return (1 / sum(1 for s in scores if s == top)) if scores[0] == top else 0.0


def margin_term(ref_margin, tied_margin, diff_correct):
    """np.tanh(min(ref, tied) / diff_correct - 1) of process_single_model, with numpy's division semantics (x/0 -> ±inf,
    0/0 -> nan) and np.nan_to_num(nan=0.0): tanh(±inf) = ±1, tanh(nan) -> 0."""
    num = min(ref_margin, tied_margin)
    if diff_correct == 0: return 0.0 if num == 0 else (1.0 if num > 0 else -1.0)
    return math.tanh(num / diff_correct - 1)


def ties_score(rows):
    """process_single_model (rewardbench/utils.py L1036) with _compute_prompt_stats (L1013): the Ties subset's weighted
    score. `rows` are the subset's prompts ({"id", "num_correct", "scores"}); a row's id is "tied:N" or "ref:N" and its
    candidate at position i is correct when i < num_correct. -> (score, details)"""
    def stats(samples):   # (accurate, different_correct_margin, correct_incorrect_margin)
        correct = [s for ok, s in samples if ok]; wrong = [s for ok, s in samples if not ok]
        best_correct, worst_correct, best_wrong = max(correct), min(correct), max(wrong)
        return (worst_correct - best_wrong) > 0, (best_correct - worst_correct) if len(correct) > 1 else None, worst_correct - best_wrong
    ref, tied = {}, {}
    for r in rows:
        kind, pid = r["id"].split(":")
        (ref if kind == "ref" else tied)[pid] = stats([(i < r["num_correct"], s) for i, s in enumerate(r["scores"])])
    accuracy = lambda d: (sum(s[0] for s in d.values()) / len(d)) if d else 0.0
    paired = sorted(set(ref) & set(tied))
    if not paired: raise ValueError("no ref/tied prompt pairs in the Ties subset")
    diff = [tied[p][1] for p in paired]; ci_tied = [tied[p][2] for p in paired]; ci_ref = [ref[p][2] for p in paired]
    preferred = sum(1 for t, d in zip(ci_tied, diff) if d is not None and t > d) / len(paired)
    hard = sum(1 for t, r, d in zip(ci_tied, ci_ref, diff) if d is not None and min(t, r) > d) / len(paired)
    margins = [margin_term(r, t, d) for r, t, d in zip(ci_ref, ci_tied, diff)]
    details = {"prompts": len(rows), "pairs": len(paired), "tied_accuracy": accuracy(tied), "ref_accuracy": accuracy(ref),
               "correctness_preferred": preferred, "correctness_preferred_hard": hard, "correctness_margin_score": sum(margins) / len(margins)}
    return (0.30 * details["tied_accuracy"] + 0.30 * details["ref_accuracy"] + 0.20 * preferred + 0.20 * hard
            + 0.01 * details["correctness_margin_score"]), details


def probe(scorer, rows, count, seed):
    """Set independence (--probe): for `count` best-of-4 prompts, the chosen response's reward when it is scored with all
    four candidates, against the same response scored with itself and one rejected candidate. A pointwise format builds the
    same request both times (the candidate's own), so delta is exactly 0 -- also under CUDA graphs, where both runs build
    the same request, hence the same bucket, and a bucket's first pass runs the same code as its replays; a set-relative
    format's prompt carries the listing, so the delta is the shift the set causes. flip: the pairwise preference between
    the chosen response and that rejected candidate differs between the two scoring sets. -> the stats, and the widest
    examples."""
    pick = sorted(random.Random(f"probe:{seed}").sample(range(len(rows)), min(count, len(rows)))) if rows else []
    deltas, flips, examples = [], 0, []
    for i in pick:
        row = rows[i]
        cands = list(row["chosen"]) + list(row["rejected"])
        all4, _, _ = scorer.scores(row["prompt"], cands)
        two, _, _ = scorer.scores(row["prompt"], [cands[0], cands[1]])
        deltas.append(abs(all4[0] - two[0]))
        flipped = (all4[0] > all4[1]) != (two[0] > two[1])
        flips += flipped
        examples.append({"id": row["id"], "subset": row["subset"], "all": all4[0], "pair": two[0], "delta": abs(all4[0] - two[0]), "flip": flipped})
    if not deltas: return {"prompts": 0}
    examples.sort(key=lambda e: -e["delta"])
    ordered = sorted(deltas)
    return {"prompts": len(deltas), "mean_delta": sum(deltas) / len(deltas), "median_delta": ordered[len(ordered) // 2],
            "max_delta": ordered[-1], "flips": flips, "flip_rate": flips / len(deltas), "examples": examples[:4]}


def main():
    ap = argparse.ArgumentParser(description="RewardBench 2 (best-of-4 and Ties) for a Kev checkpoint")
    ap.add_argument("--run", required=True, help="checkpoint dir or Hub id[@rev], e.g. runs/pref/H/seed0")
    ap.add_argument("--out", default="", help="results JSON (default <run>/rewardbench2.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16", help="bf16 (the arms' backbone dtype) | fp32 (the exact path)")
    ap.add_argument("--dataset", default=BENCHMARK)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--revision", default="", help="dataset revision to pin (default: resolve the repo's main and record it)")
    ap.add_argument("--graphs", type=int, choices=(0, 1), default=None,
                    help="CUDA graphs: unset = the serving default (on for a chat-format checkpoint on CUDA, as dindex_engine runs it); 0 = eager chat_hidden, 1 = force on")
    ap.add_argument("--probe", type=int, default=64, help="set-independence probe over this many best-of-4 prompts (0 = skip)")
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="score at most this many prompts (debug; the subsets are then partial, so a Ties subset without both of every pair stops the score)")
    args = ap.parse_args()

    rev = args.revision or HfApi().dataset_info(args.dataset).sha
    rows = load_dataset(args.dataset, split=args.split, revision=rev)
    if args.limit: rows = rows.select(range(min(args.limit, len(rows))))
    fmt = Checkpoint(args.run).meta.input_format   # before the load, as dindex_engine reads it: graphs are worth their buffers only where serving replays
    graphs = args.graphs if args.graphs is not None else (args.device == "cuda" and fmt in CHAT_FORMATS)
    predictor = LocalPredictor(args.run, args.device, LoadOptions(dtype=DTYPES[args.dtype], temperature=1.0, cuda_graphs=graphs), context=SERVING_CONTEXT)
    pointwise = predictor.model.input_format == "chat_pointwise"
    scorer = Scorer(predictor, pointwise)
    print(f"{args.dataset}@{rev[:8]} {args.split}: {len(rows)} prompts; {predictor.run}: head {predictor.model.head_kind}, "
          f"input format {predictor.model.input_format} -> {'one request per candidate (pointwise)' if pointwise else 'one question per prompt (set-relative)'}, "
          f"CUDA graphs {'on' if predictor.model.graphs is not None else 'off'}", flush=True)

    results, wall, tokens = [], defaultdict(float), defaultdict(int)
    unsupported_prompts, unsupported_candidates = [], 0
    start = time.perf_counter()
    for i, row in enumerate(rows):
        candidates = list(row["chosen"]) + list(row["rejected"])
        t = time.perf_counter()
        scores, bad, ntok = scorer.scores(row["prompt"], candidates)
        wall[row["subset"]] += time.perf_counter() - t
        tokens[row["subset"]] += ntok
        if any(bad): unsupported_prompts.append(row["id"])
        unsupported_candidates += sum(bad)
        results.append({"id": row["id"], "subset": row["subset"], "num_correct": row["num_correct"],
                        "total_completions": row.get("total_completions"), "scores": scores, "unsupported": bad})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} prompts, {time.perf_counter() - start:.0f}s", flush=True)

    subsets, details = {}, {}
    seen = {r["subset"] for r in results}
    present = [s for s in SUBSETS if s in seen] + sorted(seen - set(SUBSETS))
    for s in present:
        rows_s = [r for r in results if r["subset"] == s]
        if s.lower() == "ties":
            subsets[s], details[s] = ties_score(rows_s)
        else:
            subsets[s] = sum(non_ties_result(r["scores"]) for r in rows_s) / len(rows_s)
            details[s] = {"prompts": len(rows_s)}
    scored_s = time.perf_counter() - start
    overall = sum(subsets.values()) / len(subsets)
    non_ties = [k for k in subsets if k.lower() != "ties"]
    no_ties = sum(subsets[k] for k in non_ties) / max(1, len(non_ties))
    print("  " + " | ".join(f"{s} {subsets[s]:.4f} ({details[s]['prompts']})" for s in present), flush=True)
    print(f"  overall {overall:.4f}  no-Ties {no_ties:.4f}  unsupported {len(unsupported_prompts)} prompts / "
          f"{unsupported_candidates} candidates  {scored_s:.0f}s", flush=True)

    probe_start = time.perf_counter()
    probe_stats = probe(scorer, [r for r in rows if str(r["subset"]).lower() != "ties"], args.probe, args.probe_seed) if args.probe else None
    probe_s = time.perf_counter() - probe_start
    if probe_stats:
        print(f"  probe: {probe_stats['prompts']} prompts, |delta| mean {probe_stats['mean_delta']:.3g} median "
              f"{probe_stats['median_delta']:.3g} max {probe_stats['max_delta']:.3g}, {probe_stats['flips']} pairwise flips "
              f"({probe_s:.0f}s)", flush=True)

    out = Path(args.out) if args.out else Path(str(predictor.run)) / "rewardbench2.json"
    payload = {
        "run": str(predictor.run), "head_kind": predictor.model.head_kind, "input_format": predictor.model.input_format,
        "scoring": "pointwise (one request per candidate)" if pointwise else "set-relative (one question per prompt)",
        "device": args.device, "dtype": args.dtype, "temperature": predictor.temperature, "context": SERVING_CONTEXT,
        "graphs": {"ran": predictor.model.graphs is not None, "flag": args.graphs,
                   "rule": "on for a chat-format checkpoint on CUDA unless --graphs 0 (kev.cuda_graphs.chat); the Kev format's graphs run in probs_batch, which this loop never enters"},
        "dataset": {"repo": args.dataset, "split": args.split, "revision": rev, "prompts": len(results), "source": SOURCE},
        "subsets": subsets, "subset_details": details, "overall": overall, "no_ties": no_ties,
        "unsupported": {"prompts": len(unsupported_prompts), "candidates": unsupported_candidates, "ids": unsupported_prompts,
                        "context": SERVING_CONTEXT},
        "wall_s": {"scored": scored_s, "probe": probe_s, "by_subset": dict(wall)},
        "tokens": {"total": sum(tokens.values()), "by_subset": dict(tokens),
                   "note": "kev.model.encode of each request (the state plus every option of every question)"},
        "probe": probe_stats, "prompts": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
