"""Decision Index engine for a Kev checkpoint (kit: github.com/apolinario/decision-index, edition 0.3).

    python -m decision_index run --engine dindex_engine:KevEngine --option run=runs/pref/D/seed0 --out OUT
    (PYTHONPATH=scripts/pref; scripts/pref/dindex.sh wraps it)

Options (--option key=value):
    run          checkpoint dir or Hub id[@rev] (required), e.g. runs/pref/D/seed0 or jaredpalmer/kev-4b@v7-base
    temperature  None = the checkpoint's own temperature; a number overrides it (1.0 = raw logits)
    dtype        bf16 (default) | fp32. fp32 is kev's exact path but leaves no room for long states on a 24 GB card.
    device       cuda (default) | cpu | mps

Each request is scored by kev.predictors.LocalPredictor under kev.suite.SERVING_CONTEXT (64k state, no truncation), the
path kev.serve answers the /v1/systemone API with. The kit's (state, questions) is already that API's shape, so nothing is
reworded, filtered or truncated.

Out of memory on a chat-format checkpoint on a hybrid backbone (arms C-F): LocalPredictor runs every question's prefix in
one batch and all its K option branches at once, each branch holding its own gathered copy of the prefix's keys / values
and DeltaNet state, so a many-option request (CLINC150's 151 intents, API-Bank's 53 tools, ToolRet) does not fit a 24 GB
card. The fallback reruns such a request one question at a time with the branches in chunks (64, 16, 4, 1) under the
long-row kernels (kev.predictors.long_row_kernels: flash / memory-efficient SDPA). Branches never see each other and the
scalar head scores each option on its own, so chunking changes no option's reward; the kernels round differently from the
math kernel (measured on Kev-4B: max |dp| 6e-4, kev.predictors docstring). Rows that took it carry `fallback_chunk` in raw.
Requests the model still cannot hold (state past the serving context, more than 255 options, out of memory at one
branch, a question type other than choice / noul) are refused as Unsupported, which the index counts as wrong.
"""
import torch
from decision_index.engines import Engine, Unsupported
from pydantic import ValidationError

from kev.api import question_keys
from kev.checkpoint import LoadOptions
from kev.data import materialize
from kev.model import ContextOverflow
from kev.predictors import LocalPredictor, long_row_kernels
from kev.shared_prefix import branch_hidden
from kev.suite import SERVING_CONTEXT

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
CHUNKS = (64, 16, 4, 1)


class KevEngine(Engine):
    name = "kev"
    latency = "Device-synchronized in-process request wall time (kev.predictors.LocalPredictor), excluding model loading."

    def __init__(self, run, temperature=None, dtype="bf16", device="cuda", **options):
        super().__init__(**options)
        if dtype not in DTYPES: raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        temperature = None if temperature in (None, "", "none", "None") else float(temperature)
        self.device = device
        self.predictor = LocalPredictor(run, device, LoadOptions(dtype=DTYPES[dtype], temperature=temperature), context=SERVING_CONTEXT)
        self.provenance = {
            "kind": "kev", "run": str(run), "checkpoint_path": str(self.predictor.run), "device": device, "dtype": dtype,
            "temperature": self.predictor.temperature, "context": SERVING_CONTEXT,
            "policy": "kev.predictors.LocalPredictor on the kit's state and questions as given (the /v1/systemone request shape): "
                      "one reward per option from the checkpoint's head, softmax over options at the stated temperature; noul "
                      "is P(true). Chat checkpoints that run out of memory are rerun per question with option branches in "
                      "chunks (exact per option up to kernel rounding). Requests past the 64k serving context or 255 options, "
                      "still out of memory at one branch, or of other question types are refused as unsupported, never "
                      "truncated. No option filtering, no per-benchmark prompts.",
        }

    def runtime(self):
        info = {"torch": torch.__version__, "device": self.device, "kernels": self.predictor.environment}
        if self.device == "cuda": info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        return info

    def synchronize(self):
        if self.device == "cuda": torch.cuda.synchronize()

    def __call__(self, state, questions):
        record = {"state": state, "questions": {}}
        for qid, q in questions.items():
            if q["type"] not in ("choice", "noul"): raise Unsupported(f"unsupported question type {q['type']!r}")
            # LocalPredictor reads a label and src per question (kev.data.materialize); placeholders, never used for scoring
            label = next(iter(q["criteria"])) if q["type"] == "choice" else False
            record["questions"][qid] = {**q, "label": label, "src": "decision-index"}
        out = None
        try:
            out = self.predictor(record)
        except ContextOverflow as exc:
            raise Unsupported(f"context: {exc}") from None
        except torch.cuda.OutOfMemoryError:
            pass   # leave the except block first: its traceback holds the failed pass's tensors
        except ValidationError as exc:   # request outside the API's capacity (kev.api: e.g. more than 255 options); other errors surface as `error`
            raise Unsupported(f"request: {exc}") from None
        if out is None:
            torch.cuda.empty_cache()
            out = self._chunked(record, questions, state)
        answers = {}
        for qid, q in questions.items():
            p = out["probabilities"][qid]
            if q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
            else:
                answers[qid] = {"type": "noul", "noul": p["true"]}
        raw = {"logits": out["logits"], "input_tokens": out["input_tokens"], "temperature": out["inference_temperature"],
               **({"fallback_chunk": out["fallback_chunk"]} if "fallback_chunk" in out else {})}
        return {"model": self.provenance["run"], "answers": answers}, raw

    @torch.no_grad()
    def _chunked(self, record, questions, state):
        """The out-of-memory path (module docstring): per question, option branches in chunks of CHUNKS until one fits."""
        p, m = self.predictor, self.predictor.model
        size = f"{len(questions)} questions, state {len(str(state))} chars"
        if m.input_format != "chat" or not m.hybrid or self.device != "cuda":
            raise Unsupported(f"CUDA out of memory ({size})")
        rec = materialize(record)
        enc = m.encode(p.tok, rec, max_state=p.context["max_state"], max_branch=p.context["max_branch"], strict=True)
        for chunk in CHUNKS:
            try:
                logits = []
                with long_row_kernels():
                    for S, Sp, gs in enc["chat"]:
                        zs = [m.head(None, torch.stack([h[-1] for h in branch_hidden(m.lm, [(S, Sp, gs[i:i + chunk])], m.pad_id, m.device)[0]]))
                              for i in range(0, len(gs), chunk)]
                        logits.append(torch.cat(zs).float().cpu())
                break
            except torch.cuda.OutOfMemoryError:
                logits = None
                torch.cuda.empty_cache()
        if logits is None: raise Unsupported(f"CUDA out of memory at one branch per pass ({size})")
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in questions.items()}
        return {"probabilities": {qid: dict(zip(keys[qid], torch.softmax(z, -1).tolist())) for qid, z in zip(keys, logits)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, logits)},
                "inference_temperature": p.temperature, "input_tokens": len(enc["ids"]), "fallback_chunk": chunk}
