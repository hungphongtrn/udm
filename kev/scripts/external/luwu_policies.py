"""Lu & Wu's rejection-policy protocol on our checkpoints' frozen-benchmark run (scripts/external/luwu_run.py).

    ./kev.sh scripts/external/luwu_policies.py --run runs/dmb/ah-s0-core --overlap runs/dmb/overlap-core.json \
        --subsets ~/dmb/Decision_Model_Benchmark/decision_benchmark/resources/subsets.json \
        --reference ~/dmb/Decision_Model_Benchmark/decision_benchmark/resources --out runs/dmb/ah-s0-core-policies

Thresholds are fit the way their scorer fits them (systematic/confirmation_analysis.py: select_threshold_weighted,
release 0.2.0): 80 fixed calibration texts per task from decision_benchmark/resources/subsets.json
(`equal_budget_calibration`, the plan their README calls "80 fixed calibration pairs per task"), text-equal weights,
maximise detection subject to empirical false rejection <= 0.05, rejection is the strict `score > threshold`, and a
cross-task transfer applies the source's threshold frozen to the target's test texts. Intervals are text-cluster
bootstraps (2000 draws, their `bootstrap_ids`), thresholds held fixed, as their primary intervals do; the supplementary
source-refit draw (`joint_bootstrap`) is not reproduced here.

A readout is a confidence the protocol consumes, entered as a score that rejection puts high:

    readout     value                                     score        their score
    maxprob     max softmax(r)                            -value       (none; raw-logit readouts only)
    maxprob_T   max softmax(r / T)                        -value       negative_max
    entropy_T   normalised entropy of softmax(r / T)      +value       normalized_entropy
    maxr        max r                                     -value       (none)
    maxr_k      max r - log(n - 1), n = options in the row -value       (none)
    noisy_or    1 - prod(1 - sigmoid(r))                  -value       (none)
    pnone       softmax(r)[none]                          +value       (none; T = 1)
    pnone_T     softmax(r / T)[none]                      +value       none_probability

r is the checkpoint's raw reward (the logits its own head emits, its fitted head temperature included, exactly the z
scripts/abstain/allwrong_report.py reads from rows.json) and T the arm's fitted G temperature from the run manifest
(`kev.temperature`: RUN/eval/decision-v7-development/reward_report.json's G fit, which the manifest records whole,
Platt pair included; these readouts use only its temperature). Every formula is allwrong_report.py's, so a readout here
is the same number that report prints. `negative_max`, `normalized_entropy` and `none_probability` are computed on the
same calls their scorer uses (confirmation_analysis.py:plain_scores / none_probability): the three confidence readouts on
the PLAIN (no-NONE) request of a pair, pnone on the with-NONE request. maxprob/maxr/maxr_k/noisy_or have no counterpart
in their release and carry no reference column; they are the readouts the all-wrong hypothesis is stated in
(docs/hypotheses/in-progress/allwrong-abstention.md).

The released reference metrics (resources/reference/core-{jev,laya,qwen}/metrics.json) are read for the three mapped
scores and printed beside ours; nothing is selected by target test outcomes.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

READOUTS = ("maxprob", "maxprob_T", "entropy_T", "maxr", "maxr_k", "noisy_or", "pnone", "pnone_T")
SIGN = {"maxprob": -1.0, "maxprob_T": -1.0, "entropy_T": 1.0, "maxr": -1.0, "maxr_k": -1.0, "noisy_or": -1.0,
        "pnone": 1.0, "pnone_T": 1.0}
# readout -> their score name (their released metrics are read for these)
REFERENCE_SCORE = {"maxprob_T": "negative_max", "entropy_T": "normalized_entropy", "pnone_T": "none_probability"}
NONE_READOUTS = ("pnone", "pnone_T")
DEFAULT_LIMIT, DEFAULT_SEED, DEFAULT_GATE = 0.05, 20261005, 0.01
WILSON_Z = 1.959963984540054


def stable_seed(base, *parts):
    """Their per-key bootstrap seed (confirmation_analysis.py:stable_seed): one declared base seed, derived per key."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "big")) % (2 ** 31 - 1)


def wilson(successes, n):
    if not n:
        return None
    p = successes / n
    d = 1 + WILSON_Z * WILSON_Z / n
    c = (p + WILSON_Z * WILSON_Z / (2 * n)) / d
    r = WILSON_Z * math.sqrt(p * (1 - p) / n + WILSON_Z * WILSON_Z / (4 * n * n)) / d
    return [c - r, c + r]


def softmax(values):
    z = np.asarray(values, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def entropy(probabilities):
    """Their normalised Shannon entropy: renormalised first, 0 for a single option."""
    p = np.asarray(probabilities, dtype=float)
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    positive = p[p > 0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(p)))


def readout_scores(reward, keys, temperature):
    """One request's raw per-key rewards -> {readout: value}. pnone only where the request offers a none option.

    Every formula is scripts/abstain/allwrong_report.py's, on the same z (the checkpoint's logits as its own head emits
    them, temperature included), so our columns are that report's readouts: maxprob = max softmax(z),
    maxprob_T = max softmax(z / T) (their maxprob_G), entropy_T = normalised entropy of softmax(z / T), maxr = max z,
    maxr_k = max r - log(n - 1) (max r when the row scores one option), noisy_or = 1 - prod(1 - sigmoid(r)) in log space,
    pnone = softmax(z)[none] and pnone_T the same divided by T."""
    r = np.array([reward[key] for key in keys], dtype=float)
    p_raw = softmax(r)
    p_T = softmax(r / temperature)
    values = {"maxprob": float(p_raw.max()), "maxprob_T": float(p_T.max()), "entropy_T": entropy(p_T),
              "maxr": float(r.max()),
              "maxr_k": float(r.max()) if len(r) < 2 else float(r.max() - math.log(len(r) - 1)),
              "noisy_or": float(-np.expm1(-np.logaddexp(0.0, r).sum()))}
    if "none" in keys:
        values["pnone"] = float(p_raw[keys.index("none")])
        values["pnone_T"] = float(p_T[keys.index("none")])
    return values


def argmax_ordinary(reward, keys, temperature, gold):
    """Their argmax_ordinary on the map we hand them: highest ordinary probability, ties to the first offered option."""
    ordinary = [key for key in keys if key != "none"]
    p = softmax([reward[key] / temperature for key in keys])
    index = {key: i for i, key in enumerate(keys)}
    best = ordinary[0]
    for key in ordinary[1:]:
        if p[index[key]] > p[index[best]]:
            best = key
    return best == gold


def rate_family(values_by_text, replicates, seed):
    """Their rate_family: text-equal mean over per-text rates, pooled fraction, Wilson CI only without variants."""
    values = {text: np.asarray(v, dtype=bool) for text, v in values_by_text.items() if len(v)}
    if not values:
        return {"n_text": 0, "n_variants": 0, "successes": 0, "text_mean": None, "pooled": None,
                "wilson95": None, "bootstrap95": None}
    means = {text: float(v.mean()) for text, v in values.items()}
    successes = int(sum(int(v.sum()) for v in values.values()))
    variants = int(sum(len(v) for v in values.values()))
    texts = sorted(means)
    per_text = np.array([means[text] for text in texts])
    rng = np.random.default_rng(seed)
    draws = per_text[rng.integers(0, len(texts), size=(replicates, len(texts)))].mean(axis=1)
    return {"n_text": len(values), "n_variants": variants, "successes": successes,
            "text_mean": float(np.mean(list(means.values()))), "pooled": successes / variants,
            "wilson95": wilson(successes, variants) if all(len(v) == 1 for v in values.values()) else None,
            "bootstrap95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def fit_threshold(present_by_text, absent_by_text, limit):
    """Their select_threshold_weighted: the threshold maximising text-equal detection subject to text-equal
    false rejection <= limit, rejection being the strict `score > threshold`; ties prefer lower F, then higher t."""
    present = {text: np.asarray(v, dtype=float) for text, v in present_by_text.items() if len(v)}
    absent = {text: np.asarray(v, dtype=float) for text, v in absent_by_text.items() if len(v)}
    if not present or not absent:
        return None
    candidates = np.unique(np.concatenate([np.concatenate(list(present.values())),
                                           np.concatenate(list(absent.values()))]))

    def family_rates(mapping):
        width = max(len(v) for v in mapping.values())
        matrix = np.full((len(mapping), width), np.nan)
        lengths = np.array([len(v) for v in mapping.values()], dtype=float)
        for i, values in enumerate(mapping.values()):
            matrix[i, :len(values)] = values
        hits = (matrix[None, :, :] > candidates[:, None, None]).sum(axis=2)
        return (hits / lengths[None, :]).mean(axis=1)

    false_rates, detections = family_rates(present), family_rates(absent)
    best = None
    for threshold, false_rate, detection in zip(candidates, false_rates, detections):
        if float(false_rate) <= limit:
            objective = (float(detection), -float(false_rate), float(threshold))
            if best is None or objective > best[0]:
                best = (objective, float(threshold))
    return None if best is None else best[1]


def evaluate(present_by_text, absent_by_text, correct_by_text, threshold, replicates, seed):
    """Their evaluate_policy: detection = absent rejected, false rejection = present rejected, plus the decomposition."""
    texts = sorted(set(present_by_text) & set(absent_by_text))
    detection, false_rejection, correct_accepted = {}, {}, {}
    wrong_accepted, acceptance, absent_false_acceptance = {}, {}, {}
    for text in texts:
        present = np.asarray(present_by_text[text], dtype=float)
        absent = np.asarray(absent_by_text[text], dtype=float)
        correct = np.asarray(correct_by_text[text], dtype=bool)
        rejected = present > threshold
        detection[text] = absent > threshold
        false_rejection[text] = rejected
        correct_accepted[text] = correct & ~rejected
        wrong_accepted[text] = ~correct & ~rejected
        acceptance[text] = ~rejected
        absent_false_acceptance[text] = absent <= threshold
    return {"threshold": threshold, "n_text": len(texts),
            "n_variants": int(sum(len(present_by_text[t]) for t in texts)),
            "detection": rate_family(detection, replicates, stable_seed(seed, "detection", threshold)),
            "false_rejection": rate_family(false_rejection, replicates, stable_seed(seed, "frr", threshold)),
            "present_correct_accepted": rate_family(correct_accepted, replicates, stable_seed(seed, "pca", threshold)),
            "present_wrong_accepted": rate_family(wrong_accepted, replicates, stable_seed(seed, "pwa", threshold)),
            "present_acceptance": rate_family(acceptance, replicates, stable_seed(seed, "pa", threshold)),
            "absent_false_acceptance": rate_family(absent_false_acceptance, replicates,
                                                   stable_seed(seed, "afa", threshold))}


def load_run(directory):
    """Our luwu_run.py run dir -> (temperature, [request], scores per readout, argmax correctness per request)."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    kev = manifest.get("kev") or {}
    temperature = float(kev.get("temperature", 1.0))
    requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    status = {}
    for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            status[row["request_id"]] = row["status"]
    rewards = {}
    for line in (directory / "raw_rewards.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rewards[row["request_id"]] = row
    # luwu_run.py writes an interface_failure prediction (no reward row) for a request that cannot fit the context;
    # those stay unscored and families() counts them as missing, as their scorer does. Every ok request needs its row.
    ok = {r["request_id"] for r in requests if status.get(r["request_id"]) == "ok"}
    if len(status) != len(requests) or not ok <= set(rewards):
        raise SystemExit(f"{directory}: {len(status)} predictions and {len(ok & set(rewards))} of {len(ok)} ok rewards for "
                         f"{len(requests)} requests; the run was interrupted or predates the adapter's sidecar")
    scores, values, correct = {}, {}, {}
    for request in requests:
        if request["request_id"] not in ok:
            continue
        row = rewards[request["request_id"]]
        keys = list(request["criteria"])
        if list(row["keys"]) != keys:
            raise SystemExit(f"{request['request_id']}: the sidecar's keys differ from the request's criteria")
        readouts = readout_scores(row["reward"], keys, temperature)
        for name, value in readouts.items():
            scores.setdefault(name, {})[request["request_id"]] = SIGN[name] * value
            values.setdefault(name, {})[request["request_id"]] = value
        correct[request["request_id"]] = argmax_ordinary(row["reward"], keys, temperature, request["gold_option"])
    return {"directory": str(directory), "manifest": manifest, "temperature": temperature,
            "requests": requests, "scores": scores, "values": values, "correct": correct}


def cell_index(requests):
    """Their cell_index_key, plus with_none: the lookup their scorer uses to pair a with-NONE variant with its plain call."""
    index = {}
    for request in requests:
        key = (request["case_id"], request["name_mode"], request["presence"], request["candidate_size"],
               request["with_none"], request["member_seed"], request["order_seed"], request["none_mode"])
        if key in index:
            raise SystemExit(f"duplicated experimental cell in requests.jsonl: {key}")
        index[key] = request
    return index


def variants(requests, seeds=None):
    """Their group_variants: case_id -> [(member_seed, order_seed, present, absent)] over with-NONE aligned calls.
    `seeds` keeps only the given (member_seed, order_seed) pairs (their transfer path passes {(0, 0)}) and includes
    every seed otherwise."""
    record = defaultdict(lambda: defaultdict(dict))
    for request in requests:
        if request["presence"] not in ("present", "absent") or not request["with_none"]:
            continue
        if request["none_mode"] != "aligned":
            continue
        if seeds is not None and (request["member_seed"], request["order_seed"]) not in seeds:
            continue
        record[request["case_id"]][(request["member_seed"], request["order_seed"])][request["presence"]] = request
    out = {}
    for case_id, by_variant in record.items():
        rows = []
        for member_seed, order_seed in sorted(by_variant):
            slot = by_variant[(member_seed, order_seed)]
            if "present" in slot and "absent" in slot:
                rows.append((member_seed, order_seed, slot["present"], slot["absent"]))
        out[case_id] = rows
    return out


def families(run, group_requests, split_of, readout, seeds=None):
    """Their extract_families for one readout: {split: {present: {case_id: [scores]}, absent, correct}}.

    A variant contributes to a confidence readout only through the plain (no-NONE) calls of the same cell, and to pnone
    through its with-NONE calls (their pairing lookup is none_mode 'aligned' on both sides, so the plain call of a pair
    lives in the same group); an incomplete variant is counted in `missing`, never imputed. `seeds` is their transfer
    path's {(0, 0)} filter; the local path passes None (every variant the run carries)."""
    index = cell_index(group_requests)
    scores, correct = run["scores"].get(readout, {}), run["correct"]
    out = {}
    for split in ("calibration", "test"):
        data = {"present": defaultdict(list), "absent": defaultdict(list), "correct": defaultdict(list), "missing": 0}
        for case_id, rows in sorted(variants(group_requests, seeds).items()):
            if split_of[case_id] != split:
                continue
            for member_seed, order_seed, present, absent in rows:
                if readout in NONE_READOUTS:
                    use_present, use_absent = present, absent
                else:
                    use_present = index.get((case_id, present["name_mode"], "present", present["candidate_size"], False,
                                             member_seed, order_seed, "aligned"))
                    use_absent = index.get((case_id, absent["name_mode"], "absent", absent["candidate_size"], False,
                                            member_seed, order_seed, "aligned"))
                if use_present is None or use_absent is None:
                    data["missing"] += 1; continue
                pid, aid = use_present["request_id"], use_absent["request_id"]
                if pid not in scores or aid not in scores:
                    data["missing"] += 1; continue
                data["present"][case_id].append(scores[pid])
                data["absent"][case_id].append(scores[aid])
                data["correct"][case_id].append(correct[pid])
        out[split] = data
    return out


def restrict(by_text, case_ids):
    return {text: values for text, values in by_text.items() if text in case_ids and len(values)}


def load_subsets(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan.get("equal_budget_calibration"), dict) or not plan["equal_budget_calibration"]:
        raise SystemExit(f"{path}: no equal_budget_calibration plan")
    return plan


def subset_ids(plan, task):
    entry = plan["equal_budget_calibration"].get(task)
    if entry is None:
        return None
    return set(entry["case_ids"])


def reference_metrics(directory, models=("jev", "laya", "qwen")):
    """Their released core metrics -> {model: {"groups": ..., "transfer": {(source, target, score): row}}}.

    `directory` is either their decision_benchmark/resources/reference directory or decision_benchmark/resources (the
    `reference` child is used when present)."""
    directory = Path(directory)
    if (directory / "reference").is_dir():
        directory = directory / "reference"
    out = {}
    for model in models:
        path = Path(directory) / f"core-{model}" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        transfer_block = metrics.get("transfer") or {}
        transfer = {}
        for row in transfer_block.get("rows") or []:
            transfer[(row["source"], row["target"], row["score"])] = row
        out[model] = {"groups": metrics.get("groups", {}), "transfer": transfer, "path": str(path),
                      "notes": {name: transfer_block.get(name) for name in
                                ("available", "note", "count_name_note", "no_target_selection_note",
                                 "trEC_supplemental_note")},
                      "bootstrap": {name: metrics.get(name) for name in
                                    ("bootstrap_replicates", "bootstrap_seed", "calibration_false_rejection_limit")}}
    return out


def summarize_reference(reference, scores=("negative_max", "normalized_entropy", "none_probability")):
    """Their released metrics trimmed to the scores we can reproduce: per group threshold/calibration/test, and the
    transfer rows (_joint_ source-uncertainty and supplemental blocks dropped; the joint variant lives in a separate
    core-{model}-joint directory their scorer writes)."""
    out = {}
    for model, entry in reference.items():
        groups = {}
        for group, value in (entry.get("groups") or {}).items():
            kept = {name: {"threshold": score.get("threshold"), "eligible": score.get("eligible"),
                           "missing_families": score.get("missing_families"),
                           "calibration": score.get("calibration"), "test": score.get("test")}
                    for name, score in (value.get("scores") or {}).items() if name in scores}
            if kept:
                groups[group] = {"n_text": value.get("n_text"), "n_variants": value.get("n_variants"),
                                 "scores": kept}
        rows = {}
        for key, row in (entry.get("transfer") or {}).items():
            if key[2] not in scores:
                continue
            rows["|".join(key)] = {name: row.get(name) for name in
                                   ("transfer_type", "source", "target", "source_task", "target_task", "score",
                                    "threshold", "source_calibration_subset", "target_test", "local_reference",
                                    "supplemental", "joint_source_uncertainty")}
        out[model] = {"path": entry.get("path"), "notes": entry.get("notes"), "bootstrap": entry.get("bootstrap"),
                      "groups": groups, "transfer": rows}
    return out


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_rate(rate):
    if not rate or rate["text_mean"] is None:
        return "n/a"
    if rate["bootstrap95"]:
        return f"{rate['text_mean']:.3f} [{rate['bootstrap95'][0]:.3f},{rate['bootstrap95'][1]:.3f}]"
    return f"{rate['text_mean']:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run dir written by scripts/external/luwu_run.py")
    ap.add_argument("--out", required=True, help="new output dir (policies.json, policies.md, scores.jsonl)")
    ap.add_argument("--subsets", help="their decision_benchmark/resources/subsets.json (the 80-text calibration plan)")
    ap.add_argument("--reference", help="their decision_benchmark/resources directory (released reference metrics)")
    ap.add_argument("--overlap", help="scripts/external/overlap.py output; tasks it flags are excluded")
    ap.add_argument("--drop-overlapping-cases", action="store_true",
                    help="drop only the flagged case ids instead of the whole flagged task")
    ap.add_argument("--gate", type=float, default=DEFAULT_GATE, help="contamination gate on a dataset's overlap rate")
    ap.add_argument("--limit", type=float, default=DEFAULT_LIMIT, help="calibration false-rejection limit")
    ap.add_argument("--replicates", type=int, default=2000, help="text-cluster bootstrap draws")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="their bootstrap base seed (20261005)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    if not a.subsets:
        raise SystemExit("--subsets is required for the threshold protocol (their resources/subsets.json)")

    run = load_run(a.run)
    requests = run["requests"]
    plan = load_subsets(a.subsets)
    split_of = {request["case_id"]: request["split"] for request in requests}
    cases = {request["case_id"]: request for request in requests}
    notes = []

    # Contamination gate: a flagged dataset leaves the comparisons, or (--drop-overlapping-cases) only its flagged
    # cases do; a task is either excluded whole or sampled down, never both.
    excluded_tasks, dropped_cases = set(), set()
    contamination = None
    if a.overlap:
        contamination = json.loads(Path(a.overlap).read_text(encoding="utf-8"))
        for task, entry in (contamination.get("per_dataset") or {}).items():
            if entry.get("rate", 0.0) > a.gate:
                if a.drop_overlapping_cases:
                    dropped_cases.update(entry.get("matched_case_ids") or [])
                else:
                    excluded_tasks.add(task)
        notes.append(f"contamination gate at {a.gate:g}: excluded tasks {sorted(excluded_tasks)}"
                     + (f", dropped {len(dropped_cases)} flagged cases" if dropped_cases else ""))

    def kept(case_id):
        if case_id in dropped_cases:
            return False
        return cases[case_id]["task"] not in excluded_tasks

    groups = defaultdict(list)
    for request in requests:
        groups[(request["task"], request["candidate_size"], request["name_mode"], request["none_mode"])].append(request)
    aligned = {key: value for key, value in groups.items() if key[3] == "aligned"}
    other = sorted(key for key in groups if key[3] != "aligned")
    if other:
        notes.append(f"groups with none_mode != aligned are not scored (their plain-call pairing is aligned-only): {other}")

    local, transfer = {}, []
    unusable_subsets, family_cache = {}, {}

    def fam_of(key, readout, requests_, seeds=None):
        slot = (key, readout, seeds)
        if slot not in family_cache:
            family_cache[slot] = families(run, requests_, split_of, readout, seeds)
        return family_cache[slot]

    for key in sorted(aligned):
        requests_ = [r for r in aligned[key] if kept(r["case_id"])]
        entry = {"group": "|".join([key[0], f"k{key[1]}", key[2], key[3]]), "n_requests": len(requests_),
                 "dropped_requests": len(aligned[key]) - len(requests_), "scores": {}}
        ids = subset_ids(plan, key[0])
        if ids is None:
            unusable_subsets[key[0]] = "no equal-budget calibration subset for this task"
        elif ids - set(cases):
            unusable_subsets[key[0]] = f"{len(ids - set(cases))} of its calibration texts are absent from the run"
        for readout in READOUTS:
            if readout in NONE_READOUTS and not any("none" in r["criteria"] for r in requests_):
                continue
            fam = fam_of(key, readout, requests_)
            found = {}
            for split in ("calibration", "test"):
                present = {t: v for t, v in fam[split]["present"].items() if v and kept(t)}
                absent = {t: v for t, v in fam[split]["absent"].items() if v and kept(t)}
                correct = {t: v for t, v in fam[split]["correct"].items() if v and kept(t)}
                found[split] = (present, absent, correct, fam[split]["missing"])
            cal, test = found["calibration"], found["test"]
            score_entry = {"eligible": {"calibration": {"n_text": len(set(cal[0]) & set(cal[1]))},
                                        "test": {"n_text": len(set(test[0]) & set(test[1]))}},
                           "missing_families": {"calibration": cal[3], "test": test[3]}}
            if cal[0] and cal[1]:
                threshold = fit_threshold(cal[0], cal[1], a.limit)
                score_entry["calibration_split"] = {
                    "threshold": threshold,
                    "calibration": evaluate(cal[0], cal[1], cal[2], threshold, a.replicates,
                                            stable_seed(a.seed, "cal", key, readout))}
                if test[0] and test[1]:
                    score_entry["calibration_split"]["test"] = evaluate(test[0], test[1], test[2], threshold,
                                                                       a.replicates,
                                                                       stable_seed(a.seed, "test", key, readout))
            else:
                score_entry["undefined_reason"] = "no complete score families in calibration"
            if ids is not None and key[0] not in unusable_subsets:
                subset_cal = (restrict(cal[0], ids), restrict(cal[1], ids), restrict(cal[2], ids))
                if subset_cal[0] and subset_cal[1]:
                    threshold = fit_threshold(subset_cal[0], subset_cal[1], a.limit)
                    subset = {"threshold": threshold, "source": key[0],
                              "n_text": len(set(subset_cal[0]) & set(subset_cal[1])),
                              "calibration": evaluate(subset_cal[0], subset_cal[1], subset_cal[2], threshold,
                                                      a.replicates,
                                                      stable_seed(a.seed, "subcal", key, readout))}
                    if test[0] and test[1]:
                        subset["test"] = evaluate(test[0], test[1], test[2], threshold, a.replicates,
                                                  stable_seed(a.seed, "subtest", key, readout))
                    score_entry["equal_budget"] = subset
                else:
                    score_entry["equal_budget"] = {"threshold": None, "source": key[0],
                                                   "undefined_reason": "no complete families in the 80-text subset"}
            entry["scores"][readout] = score_entry
        local[entry["group"]] = entry

    if unusable_subsets:
        notes.append(f"equal-budget calibration unusable for {unusable_subsets}; those groups keep their "
                     "calibration-split numbers and are excluded from transfer, as their build_transfer refuses it")

    core_keys = [key for key in sorted(aligned) if subset_ids(plan, key[0])]
    primary = [key for key in core_keys if key[1] == 2 and key[2] == "natural"]
    complete = not unusable_subsets
    pairs = []
    for source in primary:
        for target in primary:
            if target[0] != source[0]:
                pairs.append(("cross_task", source, target))
    for task in sorted({key[0] for key in core_keys}):
        natural_two = (task, 2, "natural", "aligned")
        if natural_two in aligned:
            for target in core_keys:
                if target[0] == task and target[2] == "natural" and target[1] != 2:
                    pairs.append(("count", natural_two, target))
        for source in core_keys:
            if source[0] == task and source[2] == "natural":
                target = (task, source[1], "neutral", "aligned")
                if target in aligned:
                    pairs.append(("name", source, target))

    for kind, source, target in pairs:
        if not complete:
            break
        if source[0] in excluded_tasks or target[0] in excluded_tasks:
            continue  # contamination gate: a flagged dataset is out of the comparisons entirely
        source_ids, target_ids = subset_ids(plan, source[0]), subset_ids(plan, target[0])
        for readout in READOUTS:
            # their build_transfer builds its families from member/order seed (0, 0) only, whatever the run carries
            source_fam = fam_of(source, readout, [r for r in aligned[source] if kept(r["case_id"])],
                                frozenset({(0, 0)}))
            target_fam = fam_of(target, readout, [r for r in aligned[target] if kept(r["case_id"])],
                                frozenset({(0, 0)}))
            sc, tc = source_fam["calibration"], target_fam["test"]
            present = restrict(sc["present"], source_ids)
            absent = restrict(sc["absent"], source_ids)
            correct = restrict(sc["correct"], source_ids)
            if readout in NONE_READOUTS and not present:
                continue
            threshold = fit_threshold(present, absent, a.limit)
            if threshold is None:
                continue
            t_present = {t: v for t, v in tc["present"].items() if v and kept(t)}
            t_absent = {t: v for t, v in tc["absent"].items() if v and kept(t)}
            t_correct = {t: v for t, v in tc["correct"].items() if v and kept(t)}
            if not t_present or not t_absent:
                continue
            row = {"transfer_type": kind, "source": "|".join([source[0], f"k{source[1]}", source[2], source[3]]),
                   "target": "|".join([target[0], f"k{target[1]}", target[2], target[3]]),
                   "source_task": source[0], "target_task": target[0], "readout": readout,
                   "their_score": REFERENCE_SCORE.get(readout), "threshold": threshold,
                   "source_calibration_subset": evaluate(present, absent, correct, threshold, a.replicates,
                                                         stable_seed(a.seed, "src", source, readout)),
                   "target_test": evaluate(t_present, t_absent, t_correct, threshold, a.replicates,
                                           stable_seed(a.seed, "tgt", target, readout))}
            local_threshold = fit_threshold(restrict(target_fam["calibration"]["present"], target_ids),
                                           restrict(target_fam["calibration"]["absent"], target_ids), a.limit)
            if local_threshold is not None:
                row["local_reference"] = {"threshold": local_threshold,
                                          "target_test": evaluate(t_present, t_absent, t_correct, local_threshold,
                                                                  a.replicates,
                                                                  stable_seed(a.seed, "local", target, readout))}
            if target[0] in excluded_tasks or source[0] in excluded_tasks:
                row["excluded"] = "contamination gate"
            transfer.append(row)

    reference = summarize_reference(reference_metrics(a.reference)) if a.reference else {}

    result = {"protocol": {"calibration_false_rejection_limit": a.limit, "rejection": "score > threshold",
                           "calibration_subset": "80 fixed texts per task (their subsets.json equal_budget_calibration)",
                           "bootstrap_replicates": a.replicates, "bootstrap_seed": a.seed,
                           "weights": "text-equal (variants averaged within a text before aggregation)",
                           "transfer": "source threshold frozen on the target test split; local = target's own 80 texts",
                           "transfer_seeds": "member/order seed (0, 0) only, as their build_transfer does; the local "
                                             "summary uses every variant the run carries",
                           "signs": SIGN, "reference_score": REFERENCE_SCORE},
              "run": {"directory": run["directory"], "temperature": run["temperature"], "g_fit": (run["manifest"].get("kev") or {}).get("g_fit"),
                      "n_requests": len(requests), "kev": run["manifest"].get("kev")},
              "local": local, "transfer": transfer, "reference": reference,
              "contamination": contamination,
              "excluded_tasks": sorted(excluded_tasks), "dropped_cases": sorted(dropped_cases), "notes": notes}
    out.mkdir(parents=True)
    (out / "policies.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    scores_path = out / "scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as stream:
        for request in requests:
            row = {"request_id": request["request_id"], "task": request["task"], "case_id": request["case_id"],
                   "presence": request["presence"], "with_none": request["with_none"], "name_mode": request["name_mode"],
                   "candidate_size": request["candidate_size"], "split": request["split"],
                   "scores": {name: run["scores"][name][request["request_id"]] for name in READOUTS
                              if name in run["scores"] and request["request_id"] in run["scores"][name]},
                   "values": {name: run["values"][name][request["request_id"]] for name in READOUTS
                              if name in run["values"] and request["request_id"] in run["values"][name]}}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "policies.md").write_text(markdown(result, reference, a), encoding="utf-8")
    print(json.dumps({"out": str(out), "groups": len(local), "transfer_rows": len(transfer),
                      "excluded_tasks": sorted(excluded_tasks), "notes": notes}, indent=2))


def markdown(result, reference, args):
    """The numbers beside their released reference metrics, for the mapped readouts."""
    models = sorted(reference)
    lines = ["# Lu & Wu rejection policies on our checkpoints", "",
             f"thresholds: 80 fixed calibration texts per task, text-equal, detection maximal subject to empirical "
             f"false rejection <= {args.limit:g}, rejection `score > threshold`; "
             f"{args.replicates} text-cluster bootstrap draws; thresholds frozen (their primary intervals).", ""]
    if result["notes"]:
        lines += ["## Notes"] + [f"- {note}" for note in result["notes"]] + [""]
    lines += ["## Local: fit and apply on one task (test split)", "",
              "| group | readout | threshold | detection (our test) | false rejection | " + " | ".join(models) + " |",
              "|---|---|---:|---|---|" + "---|" * len(models)]
    for name, entry in sorted(result["local"].items()):
        for readout, score in sorted(entry["scores"].items()):
            subset = score.get("equal_budget") or {}
            test = subset.get("test") or {}
            cells = []
            for model in models:
                metrics = reference[model]["groups"].get(name, {}).get("scores", {}).get(
                    REFERENCE_SCORE.get(readout, ""), {})
                test_ref = metrics.get("test")
                cells.append(fmt(test_ref["detection"]["text_mean"]) if test_ref else "n/a")
            lines.append(f"| {name} | {readout} | {fmt(subset.get('threshold'))} | {fmt_rate(test.get('detection'))} | "
                         f"{fmt_rate(test.get('false_rejection'))} | " + " | ".join(cells) + " |")
    lines += ["", "their columns: detection at their threshold on the same task's test split, readouts mapped as "
                  "maxprob_T/entropy_T/pnone_T -> negative_max/normalized_entropy/none_probability.", "",
              "## Cross-task and within-task transfer (source threshold frozen)", "",
              "| source -> target | readout | threshold | detection (target test) | false rejection | local detection | "
              + " | ".join(models) + " |",
              "|---|---|---:|---|---|---|" + "---|" * len(models)]
    for row in result["transfer"]:
        cells = []
        for model in models:
            reference_row = reference[model]["transfer"].get("|".join([row["source"], row["target"],
                                                                       row["their_score"] or ""])) or {}
            reference_test = (reference_row.get("target_test") or {}).get("detection") or {}
            cells.append(fmt(reference_test.get("text_mean")))
        target_test = row["target_test"]
        local = (row.get("local_reference") or {}).get("target_test") or {}
        lines.append(f"| {row['source']} -> {row['target']} | {row['readout']} | {fmt(row['threshold'])} | "
                     f"{fmt_rate(target_test.get('detection'))} | {fmt_rate(target_test.get('false_rejection'))} | "
                     f"{fmt_rate(local.get('detection'))} | " + " | ".join(cells) + " |")
    if result["contamination"]:
        lines += ["", "## Contamination gate", "",
                  "| dataset | cases | overlap (overlap.py) | flagged by overlap.py | excluded from this run |",
                  "|---|---:|---:|---|---|"]
        for task, entry in sorted(result["contamination"]["per_dataset"].items()):
            lines.append(f"| {task} | {entry['n_cases']} | {entry['rate']:.4f} | "
                         f"{'yes' if entry['flagged'] else 'no'} | "
                         f"{'yes' if task in result['excluded_tasks'] else 'no'} |")
        lines += ["", f"excluded tasks: {result['excluded_tasks'] or 'none'}; "
                      f"dropped cases: {len(result['dropped_cases'])}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
