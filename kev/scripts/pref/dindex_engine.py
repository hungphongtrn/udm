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
reworded, filtered or truncated. Requests the model cannot hold (state past the serving context, more than 255 options,
GPU out of memory, a question type other than choice / noul) are refused as Unsupported, which the index counts as wrong.
"""
import torch
from decision_index.engines import Engine, Unsupported

from kev.checkpoint import LoadOptions
from kev.model import ContextOverflow
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


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
                      "one reward per option from the pointer head, softmax over options at the stated temperature; noul is "
                      "P(true). Requests past the 64k serving context or 255 options, out-of-memory requests and other "
                      "question types are refused as unsupported, never truncated. No option filtering, no per-benchmark prompts.",
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
        try:
            out = self.predictor(record)
        except ContextOverflow as exc:
            raise Unsupported(f"context: {exc}") from None
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise Unsupported("CUDA out of memory") from None
        except ValueError as exc:   # request validation (kev.api: e.g. more than 255 options)
            raise Unsupported(f"request: {exc}") from None
        answers = {}
        for qid, q in questions.items():
            p = out["probabilities"][qid]
            if q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
            else:
                answers[qid] = {"type": "noul", "noul": p["true"]}
        raw = {"logits": out["logits"], "input_tokens": out["input_tokens"], "temperature": out["inference_temperature"]}
        return {"model": self.provenance["run"], "answers": answers}, raw
