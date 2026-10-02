"""Decision Index (github.com/apolinario/decision-index, edition 0.2.1) for SCRM.

Each typed question becomes one SCRM decision set rendered exactly like the training data (Open-Jev / tasksource rows):
instruction {"type", "instructions"}, the request `state`, and one candidate per option (`choice`: the option
descriptions, falling back to the key; `noul`: "no", "yes"). Every option is graded; probabilities are the softmax of
the rewards. A question whose longest sequence exceeds `max_len` makes its request `unsupported` (counted as wrong by
the index): nothing is truncated, no option is dropped.

Two entry points:
  * `SCRMEngine`: a decision-index engine, for the official full-suite run (`--engine scrm.dindex:SCRMEngine`, see
    scripts/train/dindex_eval.sh); its index is directly comparable with the public board.
  * `DecisionIndexEval`: in-process, batched evaluation on a fixed stratified sample of the suite (`decision_index
    suite sample`), scored with the kit's own 0.2.1 scorers restricted to the sampled rows. Used during training at
    save steps divisible by `benchmarks.decision_index.every`, and at the end.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from .evaluate import _rewards_for
from .render import Example, Renderer, _dump

NOUL_OPTIONS = ["no", "yes"]   # candidate order of the training data's noul rows


def question_example(state, q: dict) -> tuple[list[str], Example]:
    """Question -> (option keys, Example) with candidates in key order."""
    if q["type"] == "noul":
        keys, texts = list(NOUL_OPTIONS), list(NOUL_OPTIONS)
    elif q["type"] == "choice":
        keys = list(q["criteria"])
        texts = [(_dump(v).strip() if v is not None else "") or k for k, v in q["criteria"].items()]
    else:
        raise ValueError(f"unsupported question type {q['type']!r}")
    instruction = {"type": q["type"], "instructions": q.get("instructions") or ""}
    return keys, Example.from_raw(instruction, state if state is not None else "", texts)


def make_renderer(model, max_len: int | None) -> Renderer:
    rc = dict(model.render_cfg or {})
    rc.update(max_graded=None, eval_max_graded=None, max_candidates=10**6)
    if max_len:
        rc["max_len"] = int(max_len)
    return Renderer(model.tokenizer, rc)


def _answer(keys, q, rewards) -> dict:
    r = np.asarray(rewards, dtype=np.float64)
    p = np.exp(r - r.max())
    p /= p.sum()
    probs = {k: float(v) for k, v in zip(keys, p)}
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["yes"]}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}


@torch.no_grad()
def answer_requests(model, renderer: Renderer, requests: list[dict], bcfg: dict, device, amp=True) -> list[dict]:
    """requests: [{state, questions}] -> [{"status": "ok", "response": ...} | {"status": "unsupported", "error": ...}].
    All questions of all requests are packed together (padding-free micro-batches of whole sets)."""
    out: list[dict] = [{} for _ in requests]
    items, owners = [], []
    for ri, req in enumerate(requests):
        for qk, q in req["questions"].items():
            try:
                keys, ex = question_example(req["state"], q)
            except ValueError as e:
                out[ri] = {"status": "unsupported", "error": str(e)}
                break
            it = renderer.assemble(renderer.tokenize(ex), None, False, relax=True)
            if it is None:
                out[ri] = {"status": "unsupported", "error": f"prompt + option exceeds max_len={renderer.cfg['max_len']} tokens"}
                break
            items.append(it); owners.append((ri, qk, keys))
    keep = [k for k, (ri, _, _) in enumerate(owners) if not out[ri]]
    rewards = _rewards_for(model, [items[k] for k in keep], renderer, bcfg, device, amp) if keep else []
    answers: dict[int, dict] = {}
    for k, r in zip(keep, rewards):
        ri, qk, keys = owners[k]
        full = np.empty(len(keys))
        full[items[k].order] = r                       # item.order: option index of each graded row
        answers.setdefault(ri, {})[qk] = _answer(keys, requests[ri]["questions"][qk], full)
    for ri, a in answers.items():
        if not out[ri]:
            out[ri] = {"status": "ok", "response": {"model": "scrm", "answers": a}}
    return out


try:   # the engine base class only exists when the kit is installed (requirements-train.txt)
    from decision_index.engines import Engine as _Engine, Unsupported as _Unsupported
except ImportError:   # pragma: no cover
    _Engine, _Unsupported = object, ValueError


class SCRMEngine(_Engine):
    """`python -m decision_index run|pipeline --engine scrm.dindex:SCRMEngine --option ckpt=DIR
    [--option max_len=16384] [--option max_tokens=16384] [--option device=cuda]`."""
    name = "scrm"
    latency = "In-process request wall time including rendering; one request at a time; excludes model loading."

    def __init__(self, ckpt, max_len=16384, max_tokens=16384, device=None, **options):
        super().__init__(ckpt=ckpt, max_len=max_len, max_tokens=max_tokens, device=device, **options)
        from .model import load_scrm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_scrm(ckpt, self.device).eval()
        self.renderer = make_renderer(self.model, max_len)
        self.bcfg = {"max_tokens_per_batch": int(max_tokens), "max_batch_size": 10**6}
        self.provenance = {"kind": "scrm", "ckpt": os.path.abspath(ckpt), "max_len": int(max_len)}

    def __call__(self, state, questions):
        res = answer_requests(self.model, self.renderer, [{"state": state, "questions": questions}], self.bcfg,
                              self.device, self.device.type == "cuda")[0]
        if res["status"] != "ok":
            raise _Unsupported(res["error"])
        return res["response"], None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def runtime(self):
        return {"torch": torch.__version__, "device": str(self.device),
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}


class DecisionIndexEval:
    """Fixed stratified sample of the suite, scored with the kit's edition scorers on exactly the sampled rows.
    cfg: {suite_dir, rows (sample .jsonl.gz from `decision_index suite sample`), edition, max_len, max_tokens}."""

    def __init__(self, cfg: dict):
        from decision_index.suite.io import Suite, read_jsonl
        self.cfg = cfg
        self.rows = list(read_jsonl(cfg["rows"]))
        ids = {r["_evaluation"]["run_id"] for r in self.rows}
        base = Suite(cfg["suite_dir"], cfg.get("edition") or "0.2.1")

        class SampleSuite(Suite):      # the edition's suite restricted to the sampled requests
            def rows(self, apply_exclusions=False):
                return (r for r in super().rows(apply_exclusions) if r["_evaluation"]["run_id"] in ids)

        self.suite = SampleSuite(base.directory, base.edition["id"])
        excluded = base.excluded()
        self.rows = [r for r in self.rows if r["_evaluation"]["run_id"] not in excluded
                     and base.in_edition(r["_evaluation"])]

    @torch.no_grad()
    def run(self, model, device, amp, out_dir: str) -> dict:
        from decision_index.pipeline import score_run
        was = model.training
        model.eval()
        renderer = make_renderer(model, self.cfg.get("max_len"))
        bcfg = {"max_tokens_per_batch": int(self.cfg.get("max_tokens") or 16384), "max_batch_size": 64}
        t0 = time.perf_counter()
        res = answer_requests(model, renderer, [{"state": r["state"], "questions": r["questions"]} for r in self.rows],
                              bcfg, device, amp)
        ms = 1000 * (time.perf_counter() - t0) / max(len(self.rows), 1)   # amortised: requests run batched
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "results.jsonl")
        with open(path, "w") as f:
            for row, x in zip(self.rows, res):
                f.write(json.dumps({**row["_evaluation"], "engine": "scrm", **x, "total_wall_ms": ms,
                                    "model_request_wall_ms": ms}, ensure_ascii=False) + "\n")
        scores = score_run(self.suite, path, "scrm", out_dir)
        model.train(was)
        return summarize(scores, len(self.rows), time.perf_counter() - t0)


def summarize(scores: dict, n: int, seconds: float) -> dict:
    """Flat metrics for logging: index, raw index, coverage, per-area and per-benchmark chance-corrected skill."""
    m = {"index": scores["decision_index"], "raw_index": scores["raw_index"], "requests": n, "seconds": seconds}
    counts = scores.get("counts", {})
    m["answered_frac"] = counts.get("ok", 0) / n if n else math.nan
    for k, v in scores.get("scores", {}).items():
        m[k] = v
    for a in scores.get("areas", []):
        m[f"area/{a['id']}"] = 100 * a["skill"]
    for bid, b in scores.get("benchmarks", {}).items():
        if b.get("index_skill") is not None:
            name = (b.get("dataset") or bid).replace(" ", "_").replace("/", "_")
            m[f"bench/{name}"] = 100 * b["index_skill"]
    return m


def compare_with_board(index: float, board_path: str, top: int = 100) -> list[tuple[int, str, float]]:
    """Rank `index` among the entrants of a board file ({"entrants": {id: {"name", "scores": {"balanced_skill"}}}})."""
    with open(board_path) as f:
        ent = json.load(f)["entrants"]
    rows = sorted(((e["scores"]["balanced_skill"], e.get("name", k)) for k, e in ent.items()), reverse=True)
    rows = sorted(rows + [(index, ">> SCRM <<")], reverse=True)
    return [(i + 1, name, s) for i, (s, name) in enumerate(rows)][:top]


def main(argv=None):
    """Score an existing full-suite run and place it on a board: python -m scrm.dindex --scores RUN/scores.json
    --board decision-index/tests/fixtures/board-0.2.1.json"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--board", required=True)
    a = ap.parse_args(argv)
    with open(a.scores) as f:
        s = json.load(f)
    print(f"SCRM Decision Index {s['edition']}: {s['decision_index']:.2f} (raw {s['raw_index']:.2f}), "
          f"complete={s['complete']}, completed={s['completed']}")
    for rank, name, v in compare_with_board(s["decision_index"], a.board):
        print(f"{rank:4d}  {v:6.2f}  {name}")


if __name__ == "__main__":
    main()
