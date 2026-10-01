"""Synthetic contract-format parquet generator (smoke tests / unit tests; no network)."""
from __future__ import annotations

import hashlib
import json
import os
import random

LABELS = ["alarm_set", "play_music", "weather_query", "lists_remove", "iot_hue_lightchange", "general_quirky",
          "calendar_set", "news_query"]


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _row(i, kind, rnd):
    if kind == "massive":
        utt = f"{rnd.choice(['set', 'play', 'what is', 'remove'])} thing {i}"
        opts = {_h(l): l for l in LABELS}
        win = rnd.choice(LABELS)
        tiers = [[_h(win)], [_h(l) for l in LABELS if l != win]]
        return dict(source_id="AmazonScience/massive", family="massive", label_kind="source_dataset_label",
                    source_split="test" if i % 7 else "ood", state_json=json.dumps(utt),
                    instruction_json=json.dumps("Select the intent that best matches the utterance."),
                    options_json=opts, tier_json=tiers)
    if kind == "typed":
        state = {"title": f"item {i}", "body": "lorem ipsum " * rnd.randint(2, 40), "tags": ["a", "b"]}
        levels = ["bad", "poor", "ok", "good", "great"]
        t = rnd.randrange(5)
        tiers = [[_h(levels[t])]]
        for k in range(1, 5):
            g = [_h(levels[j]) for j in (t - k, t + k) if 0 <= j < 5]
            if g:
                tiers.append(g)
        instr = {"type": "score", "instructions": "Rate the item quality.", "criteria": {"clarity": "is it clear", "length": "short"}}
        return dict(source_id="LocalLLaMA/typed-decisions", family="typed_decisions", label_kind="ordinal_score_distance_tiers",
                    source_split="train", state_json=json.dumps(state), instruction_json=json.dumps(instr),
                    options_json={_h(l): l for l in levels}, tier_json=tiers)
    n = rnd.randint(3, 6)
    tools = [f"tool_{j}({'x' * rnd.randint(1, 20)})" for j in range(n)]
    w = rnd.randrange(n)
    return dict(source_id="samatv256/agent", family="agent_decisions", label_kind="agent_choice_target", source_split="train",
                state_json=json.dumps({"history": ["user: hi"] * rnd.randint(1, 6), "goal": f"goal {i}"}),
                instruction_json=json.dumps("Choose the next tool."), options_json={_h(t): t for t in tools},
                tier_json=[[_h(tools[w])], [_h(t) for j, t in enumerate(tools) if j != w]])


def write_synth(out_dir: str, n_train=200, n_val=60, n_test=40, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    rnd = random.Random(seed)
    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        rows = []
        for i in range(n):
            r = _row(i, ["massive", "typed", "agent"][i % 3], rnd)
            opts = r["options_json"]
            ids = list(opts.keys())
            rnd.shuffle(ids)   # canonical order differs from key order
            r["options_json"] = json.dumps(dict(sorted(opts.items())))
            r["tier_json"] = json.dumps(r["tier_json"])
            dsid = _h(f"{split}{i}")
            r.update(decision_set_id=dsid, candidate_count=len(ids), probabilities_json="{}", partition_role=split,
                     candidate_rows=[{"record_id": _h(dsid + c), "decision_set_id": dsid, "choice_id": c, "candidate_order": k,
                                      "normalized_text_sha256": c} for k, c in enumerate(ids)])
            rows.append(r)
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(out_dir, "data", f"{split}-00000-of-00001.parquet"))
    return out_dir


if __name__ == "__main__":
    import sys
    print(write_synth(sys.argv[1]))
