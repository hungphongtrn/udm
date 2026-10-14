"""Two GPU checks of the prefix-forked chat path (docs/hypotheses/in-progress/allwrong-abstention.md, gates G0 and G4).

    # G0, before training the pointwise wave: the forked training forward (kev.shared_prefix: each question's prefix once,
    # its K options as branches from it) against one naive causal row per option, rewards and every trainable gradient
    .venv/bin/python scripts/abstain/fork_check.py parity --base Qwen/Qwen3.5-0.8B-Base --format chat_pointwise --n 10
    # G4, on a trained pointwise checkpoint: each option scored alone (a batch of one branch) against all K together
    .venv/bin/python scripts/abstain/fork_check.py k1 --run runs/pref/PRh/seed0 --n 200

parity: LoRA (rank --lora, every target) on the base, LoRA B and the head drawn at random (zero-initialised B would
leave A without gradient), dropout off, gradient checkpointing as in training (--checkpointing). The loss is a fixed
random linear form sum_i c_i r_i over every option of the --n development records (seeded sample, --min_k forces in
records with a question of at least that many options, e.g. banking77's 77): its gradient is the reward VJP every
training loss is built from, and being linear it can be accumulated question by question, so the naive side never holds
more than one question's rows. Three runs: fork (forward_chat_batch with gradients), naive (one causal row per question
batch) and naive1 (one row at a time). naive vs naive1 is the noise floor of the same exact computation batched
differently; the fork is exact when its distance to naive is of that order. Prints max |d reward| and the relative
gradient error ||g - g_naive|| / ||g_naive|| (all trainable parameters, and the worst single tensor) for fork and
naive1, against the thresholds 1e-4 (rewards) and 1e-3 (gradients). On CUDA fla's kernels round fp32 dots like TF32
(kev/shared_prefix.py), so read the fork against the floor there; --device cpu runs transformers' reference kernels.

k1: the checkpoint in eval mode, inference path (graphed passes where they fit, else chat_hidden). For each question,
the K rewards scored together against each option's branch scored alone from the same prefix: max |d reward|,
max |d sigmoid(reward)| and how many questions' argmax would change. The same tokens at the same positions, so only
kernel rounding may differ; a set-dependent number here is a bug, not a property of the model. (Set independence across
different option sets is allwrong_report.py's shared-option |dr|.)
"""
import argparse
import random

import torch

from kev.checkpoint import Checkpoint
from kev.data import materialize
from kev.model import CHAT_FORMATS, DecisionModel, load_tokenizer
from kev.suite import load_split

SUITE = "evals/v7/decision-v7"


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
    print("G0 PASS" if ok else "G0 FAIL (compare with naive1's floor above before concluding: on CUDA both carry TF32-like rounding)")


@torch.no_grad()
def k1(a):
    tok, model = Checkpoint(a.run).load(a.device)
    if model.input_format not in CHAT_FORMATS: raise SystemExit(f"{a.run}: {model.input_format} is not a chat format")
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
    k.add_argument("--run", required=True)
    a = ap.parse_args()
    parity(a) if a.cmd == "parity" else k1(a)


if __name__ == "__main__":
    main()
