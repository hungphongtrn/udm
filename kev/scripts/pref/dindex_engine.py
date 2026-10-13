"""Decision Index engine for a Kev checkpoint (kit: github.com/apolinario/decision-index, edition 0.3).

    python -m decision_index run --engine dindex_engine:KevEngine --option run=runs/pref/D/seed0 --out OUT
    (PYTHONPATH=scripts/pref; scripts/pref/dindex.sh wraps it)

Options (--option key=value):
    run          checkpoint dir or Hub id[@rev] (required), e.g. runs/pref/D/seed0 or jaredpalmer/kev-4b@v7-base
    temperature  None = the checkpoint's own temperature; a number overrides it (1.0 = raw logits)
    dtype        bf16 (default) | fp32. fp32 is kev's exact path but leaves no room for long states on a 24 GB card.
    device       cuda (default) | cpu | mps
    graphs       None = CUDA graphs on a CUDA device for a chat-format checkpoint (chat or chat_pointwise; the serving
                 default: kev.cuda_graphs replays the chat format's passes per record instead of one eager pass per
                 question plus one per row chunk); 0 = the eager path (DecisionModel.chat_hidden), every chat number
                 reported before the graphs. The
                 Kev format's path in this engine runs the row form, which never touches the graphs, so it is left without
                 them (their buffers are ~3 GB on a 4B hybrid).

Each request is scored by kev.predictors.LocalPredictor under kev.suite.SERVING_CONTEXT (64k state, no truncation), the
path kev.serve answers the /v1/systemone API with. The kit's (state, questions) is already that API's shape, so nothing is
reworded, filtered or truncated.

Chat-format checkpoints on a hybrid backbone (arms C-F) score through DecisionModel.chat_hidden's passes: the request's
state runs once, each question continues it with its own instruction and option listing, and its K option rows continue
that, rows per pass bounded by kev.model.rows_per_pass. On CUDA those same passes run as CUDA graphs by default
(kev.cuda_graphs.chat, `graphs=0` opts out): the record's state into a bank entry, each question's prefix continuing it as
one eager, unpadded pass per distinct prefix length (the state it leaves is what the option rows run from, so it cannot be
padded), then its option rows as a graphed pass per question group -- a handful of passes per record instead of one per
question plus one per row chunk. Where a record does not fit the graphed buffers (which hold ~3 GB on a 4B hybrid), the
original eager code runs. Either way the rows
are branch_hidden's (training's) to kernel rounding: a graph replay and an eager pass differ by floating-point
reassociation only. Out of memory anyway (CLINC150's 151 intents, API-Bank's 53 tools, ToolRet on a long state): the
request is rerun with the option rows in chunks (64, 16, 4, 1) under the long-row kernels (kev.predictors.long_row_kernels:
flash / memory-efficient SDPA). Rows never see each other and the scalar head scores each option on its own, so chunking
changes no option's reward; the kernels round differently from the math kernel (measured on Kev-4B: max |dp| 6e-4,
kev.predictors docstring). Rows that took it carry `fallback_chunk` in raw.
Requests the model still cannot hold (state past the serving context, more than 255 options, out of memory at one
branch, a question type other than choice / noul) are refused as Unsupported, which the index counts as wrong.
"""
import torch
from decision_index.engines import Engine, Unsupported
from pydantic import ValidationError

from kev.api import question_keys
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.model import CHAT_FORMATS, ContextOverflow
from kev.predictors import LocalPredictor, long_row_kernels
from kev.suite import SERVING_CONTEXT

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
CHUNKS = (64, 16, 4, 1)


class KevEngine(Engine):
    name = "kev"
    latency = "Device-synchronized in-process request wall time (kev.predictors.LocalPredictor), excluding model loading."

    def __init__(self, run, temperature=None, dtype="bf16", device="cuda", graphs=None, **options):
        super().__init__(**options)
        if dtype not in DTYPES: raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
        temperature = None if temperature in (None, "", "none", "None") else float(temperature)
        if graphs in (None, "", "none", "None"): graphs = None          # the serving default below
        elif graphs in (False, 0, "0", "false"): graphs = False          # the eager chat path (chat_hidden)
        elif graphs in (True, 1, "1", "true"): graphs = True
        else: raise ValueError(f"graphs must be 0 or 1; got {graphs!r}")
        self.device = device
        fmt = Checkpoint(run).meta.input_format   # before the load: the graphs are worth their buffers only where serving replays
        cuda_graphs = graphs if graphs is not None else device == "cuda" and fmt in CHAT_FORMATS
        self.predictor = LocalPredictor(run, device, LoadOptions(dtype=DTYPES[dtype], temperature=temperature, cuda_graphs=cuda_graphs), context=SERVING_CONTEXT)
        self.graphs = self.predictor.model.graphs   # None unless the checkpoint is a chat format on a hybrid backbone on CUDA
        self.provenance = {
            "kind": "kev", "run": str(run), "checkpoint_path": str(self.predictor.run), "device": device, "dtype": dtype,
            "input_format": fmt, "temperature": self.predictor.temperature, "context": SERVING_CONTEXT,
            "graphs": self.predictor.model.graphs is not None,
            "policy": "kev.predictors.LocalPredictor on the kit's state and questions as given (the /v1/systemone request shape): "
                      "one reward per option from the checkpoint's head, softmax over options at the stated temperature; noul "
                      "is P(true). Chat checkpoints run the state once per request and each question's option rows from "
                      "it; on CUDA those passes are replayed as CUDA graphs (kev.cuda_graphs.chat: the same tokens at the "
                      "same positions, up to floating-point reassociation) unless graphs=0; those that run out of memory are "
                      "rerun with option rows in chunks (exact per option up to kernel "
                      "rounding). Requests past the 64k serving context or 255 options, "
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
        else:
            self._capture()
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

    def _capture(self):
        """After a request that scored normally, capture one pending CUDA graph: a bucket that keeps running eagerly
        (kev.cuda_graphs.HOT_BUCKET) is worth its ~0.4 s capture, and with one request at a time nothing else is on the
        device (kev.serve captures off its request path the same way). Never on the chunked path: that one just ran out of
        memory, and a failed capture is remembered as permanent."""
        if self.graphs is not None and self.graphs.capture_due(idle=False): self.graphs.capture_pending(limit=1)

    @torch.no_grad()
    def _chunked(self, record, questions, state):
        """The out-of-memory path (module docstring): option rows in chunks of CHUNKS until one fits."""
        p, m = self.predictor, self.predictor.model
        size = f"{len(questions)} questions, state {len(str(state))} chars"
        if m.input_format not in CHAT_FORMATS or not m.hybrid or self.device != "cuda":
            raise Unsupported(f"CUDA out of memory ({size})")
        rec = materialize(record)
        enc = m.encode(p.tok, rec, max_state=p.context["max_state"], max_branch=p.context["max_branch"], strict=True)
        for chunk in CHUNKS:
            try:
                with long_row_kernels():
                    logits = [m.head(None, torch.stack([h[-1] for h in hs])).float().cpu() for hs in m.chat_hidden(enc["chat"], chunk=chunk)]
                break
            except torch.cuda.OutOfMemoryError:
                logits = None
                torch.cuda.empty_cache()
        if logits is None: raise Unsupported(f"CUDA out of memory at one branch per pass ({size})")
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in questions.items()}
        return {"probabilities": {qid: dict(zip(keys[qid], torch.softmax(z, -1).tolist())) for qid, z in zip(keys, logits)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, logits)},
                "inference_temperature": p.temperature, "input_tokens": len(enc["ids"]), "fallback_chunk": chunk}
