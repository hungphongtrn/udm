"""Run one of our checkpoints on Lu & Wu's frozen Decision Model Benchmark (luckykevvv/Decision_Model_Benchmark).

    .venv/bin/python scripts/external/luwu_run.py --inputs runs/dmb/core-inputs --ckpt runs/pref/Ah/seed0 \
        --out runs/dmb/ah-core

Reads a `decision-benchmark prepare --profile {core,controlled,intent} --with-text --output-dir DIR` directory and writes
a run directory their `decision-benchmark score --profile P --run-dir DIR --output-dir NEW` accepts: cases.jsonl and
requests.jsonl byte-identical to the prepared ones, manifest.json (`paper1-confirmation-v2`, per
systematic/confirmation_runtime.py:prepare_run and decision_benchmark/cli.py:run_model), predictions.jsonl with exactly
one row per frozen request, and the raw_rewards.jsonl sidecar (request_id -> per-key raw reward r) that
scripts/external/luwu_policies.py scores maxr / maxr_k / noisy_or from.

Mapping, one request -> one of our records (their lines, release 0.2.0):

* state: the case's reconstructed text (case['text']; decision_benchmark/datasets.py:reconstruct, required non-empty by
  systematic/confirmation_analysis.py:validate_run);
* question: one `choice` question with `instructions = request['instruction']` and `criteria = request['criteria']`,
  keyed by request_id. Those are exactly the three fields their own adapters send (systematic/run_confirmation_laya.py
  builds {'topic': {'type': 'choice', 'instructions': r['instruction'], 'criteria': r['criteria']}}), and what
  kev.api.SystemOneRequest accepts; kev.data.materialize reads a `label` per question, set to `gold_option` when the
  request offers the reference option and to the first offered key for an absent request (their validate_run requires
  gold_option null when presence is absent; scripts/abstain/build_allwrong.py labels its allwrong condition the same
  way). The label never leaves the process;
* NONE wording and option order are carried by `criteria` verbatim (ordered, `none` included where with_none is true), so
  the prompt a model sees is the frozen one; an over-context request is a failure row, never truncated.

Probability map handed to their scorer: softmax(r / T), T = the arm's fitted G temperature (G.T from
RUN/eval/decision-v7-development/reward_report.json, the same fit scripts/abstain/allwrong_report.py:load_T and
scripts/pref/eval_arm.sh use). Their scorer then reads negative_max = -max(p), normalized_entropy = entropy(p) and
none_probability = p['none'] off that map (systematic/confirmation_analysis.py:plain_scores / none_probability), which is
our maxprob_G / entropy / pnone_G. `--temperature` overrides T and is recorded. `choice` is the map's argmax with their
tie rule (first offered key wins, confirmation_analysis.py:argmax_ordinary).

Requests of one case share its state, so they are scored in chunks: the rows of one record are independent branches of
the same state prefix (kev.model), so a request's reward does not depend on which siblings share its forward pass, except
through the kernel selection kev.predictors documents for a record that holds a long row (CUDA only, a 1e-4-class
difference). Chunk boundaries are deterministic, so a fixed --budget-tokens reproduces a run.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kev.checkpoint import LoadOptions
from kev.device import DEVICE_HELP, DEVICES, select
from kev.model import ContextOverflow
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT, ENCODING, SERVING_CONTEXT, write_json


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def opts_fields(opts):
    """LoadOptions as JSON: the load knobs in force (KEV_DTYPE, KEV_TEMPERATURE, KEV_MERGE, KEV_BACKEND, ...), so a run
    says which path scored it. KEV_TEMPERATURE replaces the head temperature and therefore moves r itself."""
    return {name: (str(value) if name == "dtype" else value) for name, value in vars(opts).items()}


def jsonl_bytes(values):
    return "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values).encode(ENCODING)


def load_inputs(directory):
    """Their prepare --with-text directory -> (manifest, {case_id: case}, [request]). Verifies the prepared bytes against
    the digests the prepare manifest records, as confirmation_runtime.prepare_run does before scoring."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding=ENCODING))
    if manifest.get("case_representation") == "text-index-v1":
        raise SystemExit(f"{directory}: no text; run `decision-benchmark prepare --with-text` first")
    raw = {name: (directory / name).read_bytes() for name in ("cases.jsonl", "requests.jsonl")}
    for name, expected in (manifest.get("file_sha256") or {}).items():
        if name in raw and sha256_bytes(raw[name]) != expected:
            raise SystemExit(f"{directory / name}: differs from the digest its manifest records")
    cases = [json.loads(line) for line in raw["cases.jsonl"].decode(ENCODING).splitlines() if line.strip()]
    requests = [json.loads(line) for line in raw["requests.jsonl"].decode(ENCODING).splitlines() if line.strip()]
    by_id = {c["case_id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise SystemExit(f"{directory}/cases.jsonl: duplicate case_id")
    if len({r["request_id"] for r in requests}) != len(requests):
        raise SystemExit(f"{directory}/requests.jsonl: duplicate request_id")
    for case_id, case in by_id.items():
        if not isinstance(case.get("text"), str) or not case["text"].strip():
            raise SystemExit(f"{directory}: case {case_id} has no text")
    for request in requests:
        if request["case_id"] not in by_id:
            raise SystemExit(f"{directory}: request {request['request_id']} names an unknown case")
        if not request.get("criteria") or not isinstance(request["criteria"], dict):
            raise SystemExit(f"{directory}: request {request['request_id']} has no criteria")
        if request.get("gold_option") is not None and request["gold_option"] not in request["criteria"]:
            raise SystemExit(f"{directory}: request {request['request_id']} offers no gold_option")
    return manifest, by_id, requests


def question(request):
    """One request as a labelled `choice` question (kev.data.load_records shape)."""
    keys = list(request["criteria"])
    label = request["gold_option"] or keys[0]   # absent requests carry no reference option: any offered key labels them
    return {"type": "choice", "instructions": request["instruction"], "criteria": request["criteria"],
            "label": label, "src": request["task"]}


def record(state, items):
    """items: [(request_id, request)] -> our record shape for LocalPredictor."""
    return {"state": state, "questions": {rid: question(r) for rid, r in items}}


def estimate_tokens(state, items):
    """Packed-token estimate for one chunk (chars / 4, the rule the scripts use without a tokenizer): the state once per
    row plus each question's instruction and option texts. Only decides how chunks are cut; an over-estimate costs
    forward passes, never a number."""
    chars = len(state)
    for _, request in items:
        chars += len(request["instruction"]) + sum(len(k) + len(v or "") for k, v in request["criteria"].items())
    return chars // 4


def chunks(state, items, budget):
    """-> [[(request_id, request)]]; consecutive requests of one case up to the budget, then one chunk each."""
    out, current, size = [], [], 0
    for item in items:
        cost = estimate_tokens(state, [item])
        if current and size + cost > budget:
            out.append(current); current, size = [], 0
        current.append(item); size += cost
    if current:
        out.append(current)
    return out


def softmax(values):
    z = np.asarray(values, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def choice_of(keys, probabilities):
    """Their argmax_ordinary tie rule: the first offered key among the highest probabilities."""
    best = keys[0]
    for key in keys[1:]:
        if probabilities[key] > probabilities[best]:
            best = key
    return best


def failure_row(request, error):
    """confirmation_runtime.failure: null choice/probabilities plus error metadata (their validate_run requires extra
    keys on a failure row)."""
    return {"request_id": request["request_id"], "status": "interface_failure", "choice": None, "probabilities": None,
            "error_type": type(error).__name__, "error": str(error)[:300]}


def score_chunk(predictor, state, items, emit):
    """Score one chunk, splitting it on context overflow so a single over-context request becomes a failure row."""
    try:
        prediction = predictor(record(state, items))
    except ContextOverflow as error:
        if len(items) == 1:
            emit(items[0], None, error, 1); return
        half = len(items) // 2
        for part in (items[:half], items[half:]):
            score_chunk(predictor, state, part, emit)
        return
    for item in items:
        emit(item, prediction, None, len(items))


def temperature(ckpt, override):
    """-> (T, the whole G fit or None, source). T: --temperature, else the arm's G fit (reward_report.json), else
    refused: an unfitted T would silently change the probability map their scorer thresholds. The fit is recorded whole
    (its Platt pair belongs to the arm's provenance); these readouts use only its temperature, as allwrong_report does."""
    if override is not None:
        return float(override), None, "--temperature"
    path = Path(ckpt) / "eval" / "decision-v7-development" / "reward_report.json"
    if path.exists():
        g = json.loads(path.read_text(encoding=ENCODING)).get("G")
        if g:
            return float(g["T"]), dict(g), str(path)
    raise SystemExit(f"{path} has no fitted G temperature; run scripts/pref/eval_arm.sh on the checkpoint or pass "
                     "--temperature T")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="decision-benchmark prepare --with-text output dir")
    ap.add_argument("--ckpt", required=True, help="our run dir (checkpoint)")
    ap.add_argument("--out", required=True, help="new run dir for their `decision-benchmark score`")
    ap.add_argument("--temperature", type=float, help="override the checkpoint's fitted G temperature")
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    ap.add_argument("--context", choices=("training", "serving"), default="training",
                    help="which kev context to admit a request under: the training one kev.benchmark --data scores "
                         "external records under (default), or the serving limits (kev.suite.SERVING_CONTEXT)")
    ap.add_argument("--budget-tokens", type=int, default=1024, help="packed-token estimate one chunk may reach")
    ap.add_argument("--limit", type=int, help="SMOKE ONLY: score only the first N requests; the run dir is not scorable "
                                              "(their validator requires complete pairs and full-set baselines)")
    a = ap.parse_args()
    if a.budget_tokens < 1:
        raise SystemExit("--budget-tokens must be >= 1")

    inputs, out, ckpt = Path(a.inputs).resolve(), Path(a.out).resolve(), Path(a.ckpt).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; their new_output refuses to overwrite, so does this")
    opts = LoadOptions.from_env()
    if opts.temperature is not None:
        print(f"warning: KEV_TEMPERATURE={opts.temperature} replaces the checkpoint's head temperature, so r and the "
              "softmax(r / T) map both move; unset it for a scored run", file=sys.stderr)
    manifest, cases, requests = load_inputs(inputs)
    if a.limit:
        requests = requests[:a.limit]
    T, g_fit, t_source = temperature(ckpt, a.temperature)
    context = CONTEXT if a.context == "training" else SERVING_CONTEXT

    by_case, order = {}, []
    for request in requests:
        if request["case_id"] not in by_case:
            by_case[request["case_id"]] = []; order.append(request["case_id"])
        by_case[request["case_id"]].append((request["request_id"], request))

    out.mkdir(parents=True)
    (out / "cases.jsonl").write_bytes((inputs / "cases.jsonl").read_bytes() if not a.limit else
                                      jsonl_bytes([cases[c] for c in order]))
    (out / "requests.jsonl").write_bytes(jsonl_bytes(requests))
    # The run manifest is the prepared one plus systematic/confirmation_runtime.py:prepare_run's two edits (source_model
    # keeps the prepared `model`, `model` names the adapter that answered) and this adapter's own block, so a key their
    # scorer reads (protocol, profile, n_requests, case_representation, file_sha256) survives verbatim.
    run_manifest = dict(manifest)
    run_manifest.update(source_model=manifest.get("model"), model={"provider": "kev-local", "run": str(ckpt)},
                        n_cases=len(order), n_requests=len(requests),
                        file_sha256={"cases.jsonl": sha256_bytes((out / "cases.jsonl").read_bytes()),
                                     "requests.jsonl": sha256_bytes((out / "requests.jsonl").read_bytes())},
                        created_at_utc=datetime.now(timezone.utc).isoformat(), partial=bool(a.limit),
                        kev={"checkpoint": str(ckpt), "temperature": T, "g_fit": g_fit, "temperature_source": t_source,
                             "context": a.context, "context_limits": context, "opts": opts_fields(opts),
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "mapping": "state = the case's reconstructed text; one choice question per frozen request "
                                        "(instruction + ordered criteria, gold_option as label); requests of one case are "
                                        "chunked at --budget-tokens and the rows are independent branches of one state "
                                        "prefix, so chunk boundaries are deterministic; probabilities = softmax(reward "
                                        "/ temperature); over-context requests are failure rows",
                             "readouts": {"probability_map": "softmax(r / T)", "raw_reward": "the checkpoint's logit r"}})
    write_json(out / "manifest.json", run_manifest)

    device = select(a.device) if opts.dtype is None else select(a.device, dtype=opts.dtype)
    predictor = LocalPredictor(str(ckpt), device, opts, context=context)
    run_manifest["kev"]["environment"] = predictor.environment
    run_manifest["kev"]["inference_temperature"] = predictor.temperature

    counts = {"ok": 0, "interface_failure": 0, "chunks": 0}
    with (out / "predictions.jsonl").open("w", encoding=ENCODING) as predictions, \
            (out / "raw_rewards.jsonl").open("w", encoding=ENCODING) as rewards:
        def emit(item, prediction, error, chunk_size):
            rid, request = item
            if prediction is None:
                counts["interface_failure"] += 1
                predictions.write(json.dumps(failure_row(request, error), ensure_ascii=False) + "\n")
                return
            keys = list(request["criteria"])
            reward = prediction["logits"][rid]
            if set(reward) != set(keys) or not all(math.isfinite(reward[key]) for key in keys):
                raise SystemExit(f"{rid}: the checkpoint answered keys {sorted(reward)} with non-finite values, "
                                 f"the request offers {keys}")
            probabilities = softmax([reward[key] / T for key in keys])
            map_ = {key: float(p) for key, p in zip(keys, probabilities)}
            predictions.write(json.dumps({"request_id": rid, "status": "ok", "choice": choice_of(keys, map_),
                                          "probabilities": map_, "temperature": T,
                                          "chunk_input_tokens": prediction["input_tokens"],
                                          "chunk_size": chunk_size}, ensure_ascii=False) + "\n")
            rewards.write(json.dumps({"request_id": rid, "task": request["task"], "keys": keys,
                                      "reward": {key: float(reward[key]) for key in keys}, "temperature": T},
                                     ensure_ascii=False) + "\n")
            counts["ok"] += 1

        for index, case_id in enumerate(order, 1):
            state = cases[case_id]["text"]
            for chunk in chunks(state, by_case[case_id], a.budget_tokens):
                counts["chunks"] += 1
                score_chunk(predictor, state, chunk, emit)
            if index % 50 == 0 or index == len(order):
                print(f"scored {index}/{len(order)} cases, {counts['ok']} predictions, "
                      f"{counts['interface_failure']} failures", flush=True)

    run_manifest["kev"]["counts"] = dict(counts)
    write_json(out / "manifest.json", run_manifest)
    print(json.dumps({"run": str(out), "n_requests": len(requests), **counts, "temperature": T,
                      "environment": predictor.environment}, indent=2))


if __name__ == "__main__":
    main()
