"""Two GPU checks of the prefix-forked chat path (docs/hypotheses/done/allwrong-abstention.md, gates G0 and G4).

    # G0, before training the pointwise wave: the forked training forward (kev.shared_prefix: each question's prefix once,
    # its K options as branches from it) against one naive causal row per option, rewards and every trainable gradient
    .venv/bin/python scripts/abstain/fork_check.py parity --base Qwen/Qwen3.5-0.8B-Base --format chat_pointwise --n 10
    # G0 with transformers' PyTorch reference kernels (no fla / causal-conv1d, no TF32): the thresholds apply as stated
    .venv/bin/python scripts/abstain/fork_check.py parity --kernels reference
    # G4, on a trained pointwise checkpoint: each option scored alone (a batch of one branch) against all K together,
    # with the reference kernels (exact fp32, eager; the gate's 1e-3 applies as stated only there)
    .venv/bin/python scripts/abstain/fork_check.py k1 --run runs/pref/PRh/seed0 --kernels reference

parity: LoRA (rank --lora, every target) on the base, LoRA B and the head drawn at random (zero-initialised B would
leave A without gradient), dropout off, gradient checkpointing as in training (--checkpointing). The loss is a fixed
random linear form sum_i c_i r_i over every option of the --n development records (seeded sample, --min_k forces in
records with a question of at least that many options, e.g. banking77's 77): its gradient is the reward VJP every
training loss is built from, and being linear it can be accumulated question by question, so the naive side never holds
more than one question's rows. Three runs: fork (forward_chat_batch with gradients), naive (one causal row per question
batch) and naive1 (one row at a time). naive vs naive1 is the noise floor of the same exact computation batched
differently; the fork is exact when its distance to naive is of that order. Prints max |d reward| and the relative
gradient error ||g - g_naive|| / ||g_naive|| (all trainable parameters, and the worst single tensor) for fork and
naive1, against the thresholds 1e-4 (rewards) and 1e-3 (gradients). With --kernels bound (default) the Gated DeltaNet
layers run whatever transformers bound at import: on this install fla's and causal-conv1d's CUDA kernels, which round
fp32 dots like TF32 (kev/shared_prefix.py), so naive1 itself misses the thresholds and the fork can only be read against
that floor. causal-conv1d is bound on every device and is CUDA-only, so --device cpu fails rather than falling back.
--kernels reference rebinds the layers' kernel hooks to transformers' own PyTorch functions and turns TF32 and the fused
SDPA kernels off, on any device: exact fp32, where the thresholds are meaningful (slower and more memory than bound).

k1: the checkpoint in eval mode, inference path (graphed passes where they fit, else chat_hidden). For each question,
the K rewards scored together against each option's branch scored alone from the same prefix: max |d reward|,
max |d sigmoid(reward)| and how many questions' argmax would change. The same tokens at the same positions, so only
kernel rounding may differ; a set-dependent number here is a bug, not a property of the model. (Set independence across
different option sets is allwrong_report.py's shared-option |dr|.) With --kernels bound (default) the together and
alone passes differ in batch shape (the K rows batched and padded vs one row) under fla's TF32-class kernels, so the
gate reads that rounding as well (the bound run of 2026-10-15 on the seven pointwise h arms: max |d sigmoid| 1.3e-3 to
1.6e-2, |d reward| 0.05-0.11, no argmax flip, against G0's bound-kernel floor of 3e-3 on an untrained 0.8B);
--kernels reference rebinds the kernels as parity does and drops any CUDA graphs (KEV_CUDA_GRAPHS=1), so both sides
run chat_hidden's eager passes in exact fp32 and the 1e-3 threshold applies as stated.
"""
import argparse
import functools
import inspect
import random
import sys

import torch

from kev.checkpoint import Checkpoint
from kev.data import materialize
from kev.model import CHAT_FORMATS, DecisionModel, load_tokenizer
from kev.predictors import DELTANET_KERNELS, bound_implementation
from kev.suite import load_split

SUITE = "evals/v7/decision-v7"
DELTANET_HOOKS = (*DELTANET_KERNELS, "causal_conv1d_update", "torch_recurrent_gated_delta_rule")


def _deltanet_module(model):
    """The module whose globals a hybrid backbone's Gated DeltaNet forward reads its kernels from."""
    layer = next(m for m in model.lm.modules() if type(m).__name__.endswith("GatedDeltaNet"))
    return sys.modules[type(layer).__module__]


def _keyword_filter(fn):
    """fn, dropping keyword arguments outside its signature (as transformers' kernel wrapper does before calling)."""
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()): return fn

    @functools.wraps(fn)
    def call(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in params})
    return call


def reference_kernels(model):
    """Rebind the Gated DeltaNet forward's kernel hooks (module globals, read at call time) to the PyTorch functions
    transformers wraps (`use_kernel_func_from_hub_with_fallback` keeps them as __wrapped__); TF32 and SDPA's flash /
    memory-efficient kernels off."""
    module = _deltanet_module(model)
    for name in DELTANET_HOOKS:
        if hasattr(module, name): setattr(module, name, _keyword_filter(inspect.unwrap(getattr(module, name))))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)


def kernel_report(model):
    module = _deltanet_module(model)
    return ", ".join(f"{n}={bound_implementation(getattr(module, n))}" for n in DELTANET_HOOKS if hasattr(module, n))


def records(suite, n, min_k, seed=0):
    """n development records (seeded), the first one with a question of >= min_k options swapped in if none has one."""
    pool = load_split(suite, "development")
    pick = random.Random(seed).sample(pool, min(n, len(pool)))
    big = lambda r: max(len(q["options"]) for q in materialize(r)["questions"]) >= min_k
    if min_k and not any(big(r) for r in pick):
        extra = next((r for r in pool if big(r)), None)
        if extra is None: raise SystemExit(f"no development record has a question with >= {min_k} options")
        pick[-1] = extra
    return [materialize(r) for r in pick]


def parity(a):
    dev = a.device
    tok = load_tokenizer(a.base, revision=a.revision)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=a.revision, head_kind=a.head, input_format=a.format,
                          dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
    if not model.hybrid: raise SystemExit("parity checks the hybrid (Qwen3.5) fork; this backbone runs rows already")
    if a.checkpointing: model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    if a.kernels == "reference": reference_kernels(model)
    print(f"kernels ({a.kernels}): {kernel_report(model)}")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name: p.copy_(torch.randn(p.shape, generator=g) * 1e-3)
        for p in model.head.parameters(): p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.0
    model.train()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    recs = records(a.suite, a.n, a.min_k)
    encs = [model.encode(tok, r) for r in recs]
    ks = [len(gs) for e in encs for _, _, gs in e["chat"]]
    print(f"{len(recs)} records, {len(ks)} questions, K min/median/max {min(ks)}/{sorted(ks)[len(ks) // 2]}/{max(ks)}; "
          f"{len(params)} trainable tensors, {a.dtype}, checkpointing {a.checkpointing}, {dev}")
    coef = [[torch.randn(len(gs), generator=g).to(dev) for _, _, gs in e["chat"]] for e in encs]

    def run(kind):
        model.zero_grad(set_to_none=True); rewards = []
        for e, cs in zip(encs, coef):
            if kind == "fork":
                zs = model.forward_chat_batch([e])[0]
                sum((c * z).sum() for c, z in zip(cs, zs)).backward(); rewards += [z.detach().float().cpu() for z in zs]
                continue
            for (S, Sp, gs), c in zip(e["chat"], cs):
                rows = [(S + b["ids"], Sp + b["pos"]) for b in gs]
                hs = model._rows_hidden(rows, chunk=1 if kind == "naive1" else len(rows))
                z = model.head(None, torch.stack([h[-1] for h in hs]))
                (c * z).sum().backward(); rewards.append(z.detach().float().cpu())
        grads = {n: (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)) for n, p in params}
        return torch.cat(rewards), grads

    ref_r, ref_g = run("naive")
    ref_norm = torch.sqrt(sum(v.square().sum() for v in ref_g.values()))
    print(f"naive: |g| {ref_norm:.3e}, tensors with zero gradient {sum(int(not v.any()) for v in ref_g.values())}")
    ok = True
    for kind in ("naive1", "fork"):
        r, gr = run(kind)
        dr = (r - ref_r).abs().max().item()
        rel = torch.sqrt(sum((gr[n] - ref_g[n]).square().sum() for n in ref_g)) / ref_norm
        worst = max(ref_g, key=lambda n: ((gr[n] - ref_g[n]).norm() / ref_g[n].norm().clamp_min(1e-30)).item())
        wrel = ((gr[worst] - ref_g[worst]).norm() / ref_g[worst].norm().clamp_min(1e-30)).item()
        print(f"{kind:7} vs naive: max |d reward| {dr:.2e}   grad rel err {rel:.2e}   worst tensor {wrel:.2e} ({worst})")
        if kind == "fork": ok = dr < 1e-4 and rel < 1e-3
    print("G0 PASS" if ok else "G0 FAIL" + ("" if a.kernels == "reference" else
          " (bound kernels: compare with naive1's floor above, or rerun with --kernels reference)"))


@torch.no_grad()
def k1(a):
    tok, model = Checkpoint(a.run).load(a.device)
    if model.input_format not in CHAT_FORMATS: raise SystemExit(f"{a.run}: {model.input_format} is not a chat format")
    if a.kernels == "reference":
        if not model.hybrid: raise SystemExit("--kernels reference rebinds the hybrid (Qwen3.5) backbone's kernels")
        # the arms train on a bf16 backbone (--weights_dtype bf16), which Checkpoint.load keeps whatever is asked: upcast
        # it (the same weights computed in fp32; the adapter is already fp32), as chat_speed_check.py does
        reference_kernels(model); model.graphs = None; model.lm.float()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"kernels ({a.kernels}): {kernel_report(model) if model.hybrid else 'attention only'}; graphs "
          f"{'off' if model.graphs is None else 'on'}; {next(model.lm.parameters()).dtype}")
    recs = records(a.suite, a.n, a.min_k)
    dr = dp = 0.0; flips = nq = nopt = 0
    for rec in recs:
        splits = model.encode(tok, rec)["chat"]
        together = model._chat_logits(splits)
        for (S, Sp, gs), z in zip(splits, together):
            alone = torch.stack([model._chat_logits([(S, Sp, [b])])[0][0] for b in gs]).float()
            z = z.float()
            dr = max(dr, (alone - z).abs().max().item()); dp = max(dp, (alone.sigmoid() - z.sigmoid()).abs().max().item())
            flips += int(alone.argmax() != z.argmax()); nq += 1; nopt += len(gs)
    print(f"{a.run} ({model.input_format}): {len(recs)} records, {nq} questions, {nopt} options")
    print(f"K=1 vs K together: max |d reward| {dr:.2e}   max |d sigmoid| {dp:.2e}   argmax flips {flips}/{nq}")
    print("G4 PASS" if dp < 1e-3 and not flips else "G4 FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity"); k = sub.add_parser("k1")
    for s in (p, k):
        s.add_argument("--suite", default=SUITE); s.add_argument("--n", type=int, default=10)
        s.add_argument("--min_k", type=int, default=15, help="force in a record with a question of at least this many options (0: off)")
        s.add_argument("--device", default="cuda")
    p.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base"); p.add_argument("--revision", default=None)
    p.add_argument("--format", choices=CHAT_FORMATS, default="chat_pointwise"); p.add_argument("--head", choices=("scalar", "mlp"), default="scalar")
    p.add_argument("--lora", type=int, default=16); p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--checkpointing", type=int, choices=(0, 1), default=1)
    for s in (p, k):
        s.add_argument("--kernels", choices=("bound", "reference"), default="bound",
                       help="bound: what transformers bound at import (fla / causal-conv1d); reference: its PyTorch functions, no TF32"
                            " (k1: CUDA graphs off as well)")
    k.add_argument("--run", required=True)
    a = ap.parse_args()
    parity(a) if a.cmd == "parity" else k1(a)


if __name__ == "__main__":
    main()
