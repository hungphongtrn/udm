"""Decision Index baseline for an untrained causal LM (e.g. Qwen/Qwen3.5-4B), scored like the kit's `transformers`
engine (fixed prompt template, option score = summed log-probability of the option key tokens, probabilities = softmax
over options) but safe for hybrid models.

The kit engine scores all options in one forward with a block-diagonal attention mask and crops its KV cache to reuse
the state prefix across questions. Qwen3.5's Gated DeltaNet layers ignore the attention mask (options would read each
other) and their recurrent state cannot be cropped, so `BaseLMEngine` instead:
  1. encodes the prompt once (no cross-question prefix reuse) and takes every option's first key token from the last
     position's logits;
  2. for multi-token keys, copies the prompt cache into a batch of `option_batch` options and runs the key tokens
     after it (right-padded; causal, so padding never reaches a scored position), halving the batch on OOM.
Thinking is disabled in the chat template (`enable_thinking=False`), so the option key is the first answer token.

  python -m scrm.base_eval --model Qwen/Qwen3.5-4B --suite-dir ../decision-index/suite-0.2 \
      --rows ../decision-index/sample-1000.jsonl.gz --out outputs/base_eval/qwen3.5-4b
scores the fixed stratified sample used during training (comparable with the train-time `dindex/index`). The full suite
runs through the kit CLI: `python -m decision_index pipeline --engine scrm.base_eval:BaseLMEngine --model ...`
(scripts/train/base_eval.sh FULL=1).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from decision_index.engines.base import Unsupported
from decision_index.engines.transformers_engine import SYSTEM_PROMPT, TransformersEngine, render_prompt

from .model import _cpu_safe_gdn


class BaseLMEngine(TransformersEngine):
    name = "base-lm"
    latency = "Device-synchronized in-process request wall time; the prompt is encoded once per question (no " \
              "cross-question prefix reuse); excludes model loading."

    def __init__(self, model="Qwen/Qwen3.5-4B", max_tokens=32768, option_batch=16, thinking=False, **options):
        _cpu_safe_gdn()
        options.pop("cache_prefix", None)
        super().__init__(model=model, max_tokens=max_tokens, cache_prefix=False, **options)
        self.option_batch = max(1, int(option_batch))
        self.thinking = bool(thinking)
        self.provenance.update(
            kind="scrm.base_eval", prefix_cache=False, thinking=self.thinking, option_batch=self.option_batch,
            policy="Prompt encoded once per question; option score is the summed log-probability of the option key "
                   "tokens (first token from the prompt's last position, further tokens from a per-option copy of the "
                   "prompt cache); probabilities are the softmax over options. Prompts over the context limit are "
                   "unsupported, never truncated. No option is filtered and the prompt template is fixed.")

    def _prompt_ids(self, state, question):
        if not self.chat_template:
            return super()._prompt_ids(state, question)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_prompt(state, question)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=self.thinking)
        return self.tok(prompt, add_special_tokens=False)["input_ids"]

    def _score_options(self, p, q_ids, options):
        torch = self.torch
        assert p == 0, "prefix reuse is disabled for hybrid models"
        try:
            with torch.inference_mode():
                out = self.model(input_ids=torch.tensor([q_ids], device=self.device), use_cache=True, logits_to_keep=1)
                first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                scores = [first[o[0]].item() for o in options]
                cache = out.past_key_values
                del out
        except torch.OutOfMemoryError:
            self._empty_cache()
            raise Unsupported(f"out of memory encoding a {len(q_ids)}-token prompt")
        multi = [i for i, o in enumerate(options) if len(o) > 1]
        chunk, i = min(self.option_batch, max(len(multi), 1)), 0
        while i < len(multi):
            idx = multi[i:i + chunk]
            try:
                tails = self._tails(cache, [options[j] for j in idx])
            except torch.OutOfMemoryError:
                self._empty_cache()
                if chunk == 1:
                    raise Unsupported(f"out of memory scoring one option after a {len(q_ids)}-token prompt")
                chunk = max(1, chunk // 2)
                continue
            for j, t in zip(idx, tails):
                scores[j] += t
            i += len(idx)
        return scores

    def _tails(self, cache, options):
        """Summed log-probability of key tokens 2..m of each option, after a private batch copy of the prompt cache."""
        torch = self.torch
        n, width = len(options), max(len(o) for o in options) - 1
        rows = [o[:-1] + [o[0]] * (width - len(o) + 1) for o in options]   # right padding is never scored
        with torch.inference_mode():
            c = copy.deepcopy(cache)
            c.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
            logits = self.model(input_ids=torch.tensor(rows, device=self.device), past_key_values=c,
                                use_cache=True).logits
            del c
            lp = torch.log_softmax(logits.float(), dim=-1)
            return [sum(lp[k, t, o[t + 1]].item() for t in range(len(o) - 1)) for k, o in enumerate(options)]

    def _empty_cache(self):
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def main(argv=None):
    import argparse

    from decision_index.pipeline import score_run
    from decision_index.runner import run

    from .dindex import DecisionIndexEval, summarize

    ap = argparse.ArgumentParser(description="Decision Index of a base LM on the fixed stratified sample.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--rows", required=True, help="sample .jsonl.gz from `decision_index suite sample`")
    ap.add_argument("--edition", default="0.2.1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--option", action="append", default=[], help="engine option k=v (max_tokens, option_batch, ...)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results.jsonl instead of resuming")
    a = ap.parse_args(argv)
    opts = {"model": a.model}
    for kv in a.option:
        k, _, v = kv.partition("=")
        try:
            opts[k] = json.loads(v)
        except ValueError:
            opts[k] = v
    ev = DecisionIndexEval({"suite_dir": a.suite_dir, "rows": a.rows, "edition": a.edition})
    ids = {r["_evaluation"]["run_id"] for r in ev.rows}
    out = Path(a.out)
    t0 = time.perf_counter()
    run("scrm.base_eval:BaseLMEngine", opts, Path(a.rows), out, resume=not a.fresh,
        keep=lambda e: e["run_id"] in ids)
    scores = score_run(ev.suite, out / "results.jsonl", a.model, out)
    m = summarize(scores, len(ev.rows), time.perf_counter() - t0)
    (out / "summary.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
