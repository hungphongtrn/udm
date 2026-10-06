"""SCRM model: Qwen3.5 text backbone (+LoRA) -> hidden state at the end of each option line -> set transformer -> scalar reward."""
from __future__ import annotations

import contextlib
import functools
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collator import row_chunks
from .config import DEFAULTS, deep_update
from .tokens import prepare_tokenizer


class SetEncoder(nn.Module):
    """Bidirectional pre-LN transformer over candidates; no positional embeddings (permutation equivariant)."""

    def __init__(self, d_in, d_set=768, layers=2, heads=8, ffn_mult=2, dropout=0.1, head_hidden=None, input_norm=True):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in) if input_norm else nn.Identity()
        self.proj = nn.Linear(d_in, d_set)
        self.layers = layers
        if layers > 0:
            layer = nn.TransformerEncoderLayer(d_set, heads, dim_feedforward=ffn_mult * d_set, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, norm=None, enable_nested_tensor=False)
        else:
            self.encoder = None
        hh = head_hidden or d_set // 2
        self.head = nn.Sequential(nn.LayerNorm(d_set), nn.Linear(d_set, hh), nn.SiLU(), nn.Linear(hh, 1))

    def forward(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """E [B,N,d], mask [B,N] bool (True = real) -> rewards [B,N] fp32, 0 at masked slots."""
        dev_type = E.device.type
        with torch.autocast(device_type=dev_type, enabled=False):
            x = self.proj(self.in_norm(E.float()))
            if self.encoder is not None:
                kpm = ~mask
                kpm = kpm.clone()
                kpm[:, 0] = kpm[:, 0] & ~kpm.all(dim=1)   # fully padded rows: keep slot 0 visible (avoid NaN)
                x = self.encoder(x, src_key_padding_mask=kpm)
            r = self.head(x).squeeze(-1)
            r = torch.nan_to_num(r).masked_fill(~mask, 0.0)
        return r


class SCRM(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, cfg: dict, backbone_trainable: bool = True):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.backbone_trainable = backbone_trainable
        self.d_hidden = hidden
        self.branching = bool(cfg.get("branching", True))
        self.head_kind = cfg.get("head", "set")
        if self.head_kind == "linear":
            # Per-candidate reward from its own embedding only (no cross-candidate interaction in the head).
            self.set_encoder = None
            self.reward_head = nn.Linear(hidden, 1)
        elif self.head_kind == "set":
            self.set_encoder = SetEncoder(hidden, cfg["d_set"], cfg["set_layers"], cfg["set_heads"], cfg["set_ffn_mult"],
                                          cfg["set_dropout"], cfg.get("head_hidden"), cfg.get("input_norm", True))
        else:
            raise ValueError(f"model.head must be 'set' or 'linear', got {self.head_kind!r}")
        self.tokenizer = None
        self.render_cfg: dict | None = None
        self._ckpt_layers: list[nn.Module] | None = None

    def head_module(self) -> nn.Module:
        """The reward head: the set encoder (`head: set`) or the shared linear layer (`head: linear`)."""
        return self.reward_head if self.head_kind == "linear" else self.set_encoder

    def head_device(self) -> torch.device:
        """Device of the head's parameters (the head is the only always-present module besides the backbone)."""
        return next(self.head_module().parameters()).device

    def ckpt_layers(self) -> list[nn.Module]:
        """Decoder layers HF gradient checkpointing was enabled on at build time, in depth order ([] if it is off)."""
        if self._ckpt_layers is None:
            from transformers.modeling_layers import GradientCheckpointingLayer
            self._ckpt_layers = [m for m in self.backbone.modules()
                                 if isinstance(m, GradientCheckpointingLayer) and m.gradient_checkpointing]
        return self._ckpt_layers

    def keep_activations(self, k: int):
        """Selective checkpointing: `k` evenly spaced layers of `ckpt_layers()` keep their activations (no recompute in
        backward), the rest stay checkpointed. `k >= len(layers)` turns checkpointing off, `k = 0` restores it fully.
        Evenly spaced so the stored set keeps the hybrid model's linear/full attention layer mix."""
        layers = self.ckpt_layers()
        n = len(layers)
        k = max(0, min(int(k), n))
        keep = {(2 * i + 1) * n // (2 * k) for i in range(k)} if k else set()
        for j, m in enumerate(layers):
            m.gradient_checkpointing = j not in keep

    def embed(self, pack: dict, max_tokens=None) -> torch.Tensor:
        """Encode the pack -> [M, d]: last-layer hidden state at the last token of every graded option, in pack row
        order. Runs in contiguous chunks of <= max_tokens encoded tokens (see `row_chunks`: whole sets when
        branching, single full sequences otherwise; a unit bigger than the budget gets its own chunk)."""
        ctx = contextlib.nullcontext() if self.backbone_trainable else torch.no_grad()
        with ctx:
            return torch.cat([self._embed_rows(pack, rows)
                              for rows in row_chunks(pack, max_tokens, self.branching)], 0)

    def embed_indices(self, pack: dict, idx) -> torch.Tensor:
        """Hidden states at the last token of exactly the graded rows listed in `idx` (indices into pack row order).
        Same layout as `embed`'s output rows: `[len(idx), d]` in the order of `idx` (idx is used as given). Branching:
        every row of a set is encoded together with that set's prefix (one branch layout per set)."""
        return self._embed_rows(pack, [int(i) for i in idx])

    def _embed_rows(self, pack: dict, rows) -> torch.Tensor:
        if not self.branching:
            return self._encode_full(*_full_chunk(pack, rows))
        ids, pos, plan, read = _branch_chunk(pack, rows)
        return self._branch_encode(ids, pos, plan, read)

    def _encode_full(self, ids: torch.Tensor, pos: torch.Tensor, lens: list) -> torch.Tensor:
        """Stock HF forward over full sequences (concatenated in `ids`, positions restarting at 0) -> hidden state at
        the last token of each. CUDA varlen kernels + flash_attention_2: one padding-free row (block-diagonal
        attention + per-sequence Gated DeltaNet conv / scan via cu_seqlens / seq_idx; sdpa/eager would ignore the
        boundaries). Otherwise: a right-padded batch."""
        dev = ids.device
        if _varlen_kernels(dev) and getattr(self.backbone.config, "_attn_implementation", None) == "flash_attention_2":
            L = torch.tensor(lens, device=dev)
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev)
            cu[1:] = L.cumsum(0)
            seq_idx = torch.repeat_interleave(torch.arange(len(lens), device=dev, dtype=torch.int32), L)[None]
            h = self.backbone(input_ids=ids[None], position_ids=pos[None], cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                              max_length_q=max(lens), max_length_k=max(lens), seq_idx=seq_idx,
                              use_cache=False).last_hidden_state[0]
            return h[cu[1:].long() - 1]
        S = max(lens)
        x = ids.new_zeros(len(lens), S)
        am = ids.new_zeros(len(lens), S)
        t = 0
        for k, n in enumerate(lens):
            x[k, :n] = ids[t:t + n]
            am[k, :n] = 1
            t += n
        h = self.backbone(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state
        return h[torch.arange(len(lens), device=dev), torch.tensor(lens, device=dev) - 1]

    def _branch_encode(self, ids: torch.Tensor, pos: torch.Tensor, plan: BranchPlan, read: torch.Tensor):
        """Run the decoder over one branch layout (prefix segments + their suffix branches) and gather the read-out
        hidden states. Equivalent to encoding each full sequence `prefix + suffix_k` on its own, at a fraction of the
        prompt tokens."""
        tm = _inner_text(self.backbone)
        h = tm.embed_tokens(ids)[None]
        pid = pos[None]
        pe = tm.rotary_emb(h, pid[None].expand(3, 1, -1))
        ctx = BranchCtx(plan, _branch_backend(ids.device))
        for layer in tm.layers[: tm.config.num_hidden_layers]:
            h = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pid, past_key_values=None,
                      branch_ctx=ctx)
        return tm.norm(h)[0].index_select(0, read)

    def head(self, e: torch.Tensor, pack: dict, candidate_mask: torch.Tensor) -> torch.Tensor:
        """Per-row embeddings [M, d] (graded options, in pack row order) -> rewards [B, N] (0 at padded slots).
        The embedding of a row is its hidden state at the assistant header (see `embed`). `head: set` scores the
        options of a set jointly; `head: linear` applies the shared linear layer to each option on its own."""
        B, N = candidate_mask.shape
        E = e.new_zeros(B, N, e.size(-1)).index_put((pack["row_set"], pack["row_slot"]), e)
        if self.head_kind == "linear":
            with torch.autocast(device_type=E.device.type, enabled=False):   # raw rewards in fp32, as the BT loss expects
                r = self.reward_head(E.float()).squeeze(-1)
            return r.masked_fill(~candidate_mask, 0.0)
        return self.set_encoder(E, candidate_mask)

    def forward(self, pack: dict, candidate_mask: torch.Tensor, max_tokens=None) -> torch.Tensor:
        """pack: every graded option as a suffix branch off its set's shared prefix (see collator). Each option's
        embedding is the hidden state at the end of its suffix branch (assistant header); the head (set encoder or
        shared linear layer) then turns the embeddings into rewards. Returns rewards [B, N] (0 at padded slots)."""
        e = self.embed(pack, max_tokens)
        return self.head(e, pack, candidate_mask)

    def score(self, b: dict, max_tokens=None):
        """Forward on a collated batch dict."""
        return self(b["pack"], b["candidate_mask"], max_tokens=max_tokens)

    # ----- parameter groups / persistence -----
    def head_state_dict(self) -> dict:
        """Head weights keyed by the head module name, so loading into a differently-configured head fails loudly."""
        return {f"{self.head_kind_name()}.{k}": v.detach().cpu()
                for k, v in self.head_module().state_dict().items()}

    def load_head_state_dict(self, sd: dict):
        prefix = f"{self.head_kind_name()}."
        own = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        self.head_module().load_state_dict(own)

    def head_kind_name(self) -> str:
        """Module name of the head in state dicts / the optimizer (`set_encoder` or `reward_head`)."""
        return "reward_head" if self.head_kind == "linear" else "set_encoder"

    def lora_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if p.requires_grad]

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained") and self.cfg["lora"]["enabled"] and not self.cfg["freeze_backbone"]:
            self.backbone.save_pretrained(os.path.join(path, "adapter"))
        torch.save(self.head_state_dict(), os.path.join(path, "scrm_head.pt"))
        with open(os.path.join(path, "scrm_config.json"), "w") as f:
            json.dump({"model": self.cfg, "render": self.render_cfg}, f, indent=2)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

    # ----- inference helpers -----
    @torch.no_grad()
    def rank(self, instruction, state, candidates: list[str], device=None, max_tokens: int | None = 16384) -> list[dict]:
        """Score candidates (given order; no shuffling). Returns [{index, reward}] sorted by reward desc;
        `index` is the position in the input list."""
        from .render import Renderer, Example
        was_training = self.training
        self.eval()
        dev = device or self.head_device()
        rr = Renderer(self.tokenizer, {**(self.render_cfg or {}), "max_graded": None,       # grade every candidate
                                       "max_candidates": max(len(candidates), 1)})
        ex = Example.from_raw(instruction, state, candidates)
        item = rr.assemble(rr.tokenize(ex), rng=None, shuffle=False, relax=True)
        if item is None:
            raise ValueError(f"prompt + candidate exceeds render.max_len={rr.cfg['max_len']} tokens")
        from .collator import collate
        from .collator import to_device
        b = to_device(collate([item], pad_id=rr.pad_id), dev)
        amp = dev.type == "cuda"
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            r = self.score(b, max_tokens=max_tokens)[0]
        out = [{"index": int(item.order[k]), "reward": float(r[k])} for k in range(len(item.order))]
        out.sort(key=lambda d: -d["reward"])
        self.train(was_training)
        return out

    @staticmethod
    def pairwise_probability(r_i, r_j, tau: float = 1.0):
        """P(i preferred over j) = sigmoid((r_i - r_j)/tau)."""
        return torch.sigmoid((torch.as_tensor(r_i, dtype=torch.float32) - torch.as_tensor(r_j, dtype=torch.float32)) / tau)


def pairwise_probability(r_i, r_j, tau: float = 1.0):
    return SCRM.pairwise_probability(r_i, r_j, tau)


def _varlen_kernels(device: torch.device) -> bool:
    """Padding-free packing needs the CUDA Gated DeltaNet kernels that take sequence boundaries (fla
    chunk_gated_delta_rule: cu_seqlens; causal-conv1d: seq_idx). HF's torch fallbacks would leak state across them."""
    if device.type != "cuda":
        return False
    try:
        import causal_conv1d  # noqa: F401
        import fla.ops.gated_delta_rule  # noqa: F401
    except Exception:
        return False
    return True


def _cpu_safe_gdn() -> None:
    """HF binds causal-conv1d / fla Gated DeltaNet kernels at import time whenever the packages are installed, with no
    device check, so CPU tensors hit CUDA-only kernels (`Expected x.is_cuda()`). Rebind the module-level functions to
    dispatch on device: CUDA -> installed kernel, otherwise -> HF's torch reference. Idempotent."""
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    def dispatch(kernel):
        ref = inspect.unwrap(kernel)
        if ref is kernel or getattr(kernel, "_scrm_dispatch", False):
            return kernel
        sig = inspect.signature(ref).parameters
        ref_kw = None if any(p.kind is p.VAR_KEYWORD for p in sig.values()) else set(sig)

        def fn(x, *args, **kwargs):
            if x.is_cuda:
                return kernel(x, *args, **kwargs)
            return ref(x, *args, **(kwargs if ref_kw is None else {k: v for k, v in kwargs.items() if k in ref_kw}))
        fn._scrm_dispatch = True
        return fn

    for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule"):
        setattr(mq, name, dispatch(getattr(mq, name)))


# --------------------------------------------------------------------------- shared-prefix branching
#
# A pack encodes each decision set as ONE prefix segment plus one suffix segment per graded option (collator.collate).
# The branch encoder runs the decoder once over that layout; suffix k sees the prefix tokens (with the same position
# ids as the standalone `prefix + suffix_k` sequence) plus its own suffix, never another suffix, so the read-out
# hidden states AND their gradients equal the per-option full-sequence computation.
#   * full-attention layers: prefix queries attend the prefix causally, suffix queries attend the prefix plus their own
#     suffix. A suffix query's two softmaxes are merged exactly with a log-sum-exp (no approximation).
#   * Gated DeltaNet layers: the prefix scan runs once and yields the final recurrent state S_P; every suffix of the
#     set starts from S_P (gradients sum into it), and its short conv is seeded with the prefix's last kernel-1 conv
#     inputs, exactly as in a continuous sequence.


@dataclass
class BranchPlan:
    """Segments of one branch-encoded chunk: segment s covers tokens [seg_start[s], seg_start[s] + seg_len[s]).
    A suffix segment (owner[s] != s) continues prefix segment owner[s]; prefix segments own themselves."""
    seg_len: list
    seg_start: list
    owner: list

    def spans(self) -> list:
        return [(a, a + n) for a, n in zip(self.seg_start, self.seg_len)]

    def prefixes(self) -> list:
        return [s for s, o in enumerate(self.owner) if o == s]

    def suffixes(self) -> list:
        return [s for s, o in enumerate(self.owner) if o != s]

    def children(self, s: int) -> list:
        """Suffix segments that branch off prefix segment s."""
        return [t for t, o in enumerate(self.owner) if o == s and t != s]


@dataclass
class BranchCtx:
    """Per-forward branching context, handed to the patched decoder layers as the `branch_ctx` kwarg."""
    plan: BranchPlan
    varlen: bool


def _inner_text(backbone: nn.Module) -> nn.Module:
    """The Qwen3.5 text decoder inside `backbone` (PeftModel / vision-language wrappers included)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    if isinstance(backbone, Qwen3_5TextModel):
        return backbone
    for m in backbone.modules():
        if isinstance(m, Qwen3_5TextModel):
            return m
    raise TypeError(f"no Qwen3.5 text model inside {type(backbone).__name__}")


def _install_branching(backbone: nn.Module) -> None:
    """Wrap every full-attention / Gated DeltaNet module so a `branch_ctx` kwarg switches it to the branch layout
    (`SCRM._branch_encode`); without that kwarg the original forward runs untouched. Idempotent."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5GatedDeltaNet

    def wrap(mod, fn):
        if getattr(mod, "_scrm_branch", False):
            return
        orig = mod.forward

        def branch_forward(hidden_states, *args, **kwargs):
            ctx = kwargs.pop("branch_ctx", None)
            if ctx is None:
                return orig(hidden_states, *args, **kwargs)
            return fn(mod, hidden_states, *args, ctx=ctx, **kwargs)

        mod.forward = branch_forward
        mod._scrm_branch = True

    for m in backbone.modules():
        if isinstance(m, Qwen3_5Attention):
            wrap(m, _branch_full_attention)
        elif isinstance(m, Qwen3_5GatedDeltaNet):
            wrap(m, _branch_gated_deltanet)


def _branch_chunk(pack: dict, rows) -> tuple[torch.Tensor, torch.Tensor, BranchPlan, torch.Tensor]:
    """(input_ids, position_ids, plan, read-out token index) of the sets owning `rows` (ascending pack row order).
    Every set of the chunk contributes its whole prefix segment followed by its suffix segments, in pack order."""
    seg_prefix, row_seg = pack["seg_prefix"].tolist(), pack["row_seg"].tolist()
    seg_lens = [int(x) for x in pack["seg_lens"]]
    segs, seen = [], set()
    for r in rows:
        sg = row_seg[r]
        for s in (seg_prefix[sg], sg):     # the set's prefix first, then this suffix
            if s not in seen:
                seen.add(s)
                segs.append(s)
    segs.sort()                            # ascending segment index == each set's prefix before its suffixes
    starts, t = [], 0
    for s in segs:
        starts.append(t)
        t += seg_lens[s]
    plan = BranchPlan([seg_lens[s] for s in segs], starts, [segs.index(seg_prefix[s]) for s in segs])
    pack_start, off = [], 0                # the pack is the segments laid out in order: spans of the KEPT segments
    for n in seg_lens:
        pack_start.append(off)
        off += n
    spans = [(pack_start[s], pack_start[s] + seg_lens[s]) for s in segs]
    ids = torch.cat([pack["input_ids"][a:b] for a, b in spans])
    pos = torch.cat([pack["position_ids"][a:b] for a, b in spans])
    read = torch.as_tensor([plan.seg_start[segs.index(row_seg[r])] + seg_lens[row_seg[r]] - 1 for r in rows],
                           device=ids.device)
    return ids, pos, plan, read


def _full_chunk(pack: dict, rows) -> tuple[torch.Tensor, torch.Tensor, list]:
    """(input_ids, position_ids, lengths) of the full sequences `prefix + suffix` of `rows` (in the given order),
    concatenated; positions restart at 0 per sequence (the pack's own position ids for both segments)."""
    seg_prefix, row_seg = pack["seg_prefix"].tolist(), pack["row_seg"].tolist()
    seg_lens = [int(x) for x in pack["seg_lens"]]
    start, off = [], 0
    for n in seg_lens:
        start.append(off)
        off += n
    spans = [(start[s], start[s] + seg_lens[s]) for r in rows for s in (seg_prefix[row_seg[r]], row_seg[r])]
    ids = torch.cat([pack["input_ids"][a:b] for a, b in spans])
    pos = torch.cat([pack["position_ids"][a:b] for a, b in spans])
    lens = [seg_lens[seg_prefix[row_seg[r]]] + seg_lens[row_seg[r]] for r in rows]
    return ids, pos, lens


def _branch_backend(device: torch.device) -> bool:
    """Whether to branch with the CUDA varlen kernels (flash-attn + fla/causal-conv1d). `SCRM_BRANCH_TORCH=1`
    forces the torch path (debugging / equivalence checks)."""
    if os.environ.get("SCRM_BRANCH_TORCH"):
        return False
    return _varlen_kernels(device)


def _cum(lens, device) -> torch.Tensor:
    """int32 cu_seqlens (flash-attn / fla layout) from a list of segment lengths."""
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.as_tensor(list(lens), dtype=torch.int32, device=device).cumsum(0)
    return cu


def _tok_index(spans, device) -> torch.Tensor:
    """Concatenated token ranges of `spans` [(start, end)] as one index tensor (a plain slice for a single span)."""
    return torch.cat([torch.arange(a, b, device=device) for a, b in spans])


def _flat_lse(lse, cu_q, nheads) -> torch.Tensor:
    """flash-attn returns softmax_lse as [nheads, total_q] (varlen) or [nseq, nheads, max_q] (older builds):
    normalise both to [nheads, total_q]."""
    total = int(cu_q[-1])
    if lse.dim() == 2:
        return lse if lse.shape[-1] <= total else lse[:, :total]
    if lse.dim() == 3 and lse.shape[0] == cu_q.numel() - 1:
        out = lse.new_empty((nheads, total))
        for i in range(lse.shape[0]):
            a, b = int(cu_q[i]), int(cu_q[i + 1])
            out[:, a:b] = lse[i, :, : b - a]
        return out
    raise RuntimeError(f"unsupported flash-attn softmax_lse shape {tuple(lse.shape)}; set SCRM_BRANCH_TORCH=1 to "
                       "use the torch branch path")


@functools.lru_cache(maxsize=1)
def _flash_ok() -> bool:
    """Whether the installed flash-attn can run varlen attention with `return_softmax_lse` (needed to merge a suffix
    query's prefix and own-suffix softmaxes). Probed once per process; on failure the torch path is used."""
    if not torch.cuda.is_available():
        return False
    try:
        from flash_attn import flash_attn_varlen_func
        q = torch.zeros(2, 2, 8, dtype=torch.bfloat16, device="cuda")
        cu = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        _, lse = flash_attn_varlen_func(q, q, q, cu, cu, 2, 2, causal=True, return_softmax_lse=True)
        return _flat_lse(lse, cu, 2).shape == (2, 2)
    except Exception:
        return False


def _flash_attn(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale, dropout):
    """flash-attn varlen attention on [T, H, D] q/k/v -> (out [Tq, Hq, D], lse [Hq, Tq])."""
    from flash_attn import flash_attn_varlen_func
    o, lse = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=dropout, softmax_scale=scale,
                                    causal=causal, return_softmax_lse=True)
    return o, _flat_lse(lse, cu_q, q.shape[1])


def _branch_attn_flash(q, k, v, plan, scale, dropout, out) -> None:
    """Branch attention with flash-attn varlen kernels. Prefix segments are causal self-attentions; a set's suffix
    queries are ONE sequence attending (a) the whole prefix and (b) their own suffix causally, the two softmaxes
    merged by log-sum-exp — the exact result of the standalone `prefix + suffix_k` softmax."""
    spans = plan.spans()
    pref = plan.prefixes()
    p_lens = [plan.seg_len[s] for s in pref]
    idx = _tok_index([spans[s] for s in pref], q.device)
    o, _ = _flash_attn(q[idx], k[idx], v[idx], _cum(p_lens, q.device), _cum(p_lens, q.device),
                       max(p_lens), max(p_lens), True, scale, dropout)
    out[idx] = o
    for s in pref:
        a, b = spans[s]
        kids = plan.children(s)
        if not kids:
            continue
        kspans = [spans[t] for t in kids]
        idx = _tok_index(kspans, q.device)
        s_lens = [plan.seg_len[t] for t in kids]
        nq = sum(s_lens)
        o_x, lse_x = _flash_attn(q[idx], k[a:b], v[a:b], _cum([nq], q.device), _cum([b - a], q.device),
                                 nq, b - a, False, scale, dropout)
        o_s, lse_s = _flash_attn(q[idx], k[idx], v[idx], _cum(s_lens, q.device), _cum(s_lens, q.device),
                                 max(s_lens), max(s_lens), True, scale, dropout)
        lse = torch.logaddexp(lse_x, lse_s)
        out[idx] = (o_x * (lse_x - lse).T.unsqueeze(-1) + o_s * (lse_s - lse).T.unsqueeze(-1)).to(out.dtype)


def _gqa_scores(q, k, scale):
    """[b, Hq, D] x [Tk, Hkv, D] -> [b, Hq, Tk] (query head h reads kv head h // (Hq/Hkv), as in HF's repeat_kv)."""
    b, Hq, D = q.shape
    Hkv = k.shape[1]
    if Hq != Hkv:
        return torch.einsum("bhrd,ghd->bhrg", q.reshape(b, Hkv, Hq // Hkv, D), k).reshape(b, Hq, k.shape[0]) * scale
    return torch.einsum("bhd,ghd->bhg", q, k) * scale


def _gqa_apply(p, v):
    """[b, Hq, Tk] x [Tk, Hkv, D] -> [b, Hq, D]."""
    b, Hq, g = p.shape
    Hkv = v.shape[1]
    if Hq != Hkv:
        return torch.einsum("bhrg,ghd->bhrd", p.reshape(b, Hkv, Hq // Hkv, g), v).reshape(b, Hq, v.shape[-1])
    return torch.einsum("bhg,ghd->bhd", p, v)


def _torch_union_attn(q, parts, scale, dropout, qblk=512):
    """Exact softmax attention of `q` [Tq, Hq, D] over the UNION of `parts` [(k [Tk, Hkv, D], v, causal)], in fp32 and
    in query blocks (so the score matrices stay bounded). `causal` masks key j for query i when j > i (a part is a
    single sequence whose token order matches the query order)."""
    Tq, Hq, D = q.shape
    o = torch.empty(Tq, Hq, D, dtype=torch.float32, device=q.device)
    for a in range(0, Tq, qblk):
        b = min(a + qblk, Tq)
        qb = q[a:b].float()
        scores = []
        for k, v, causal in parts:
            s = _gqa_scores(qb, k.float(), scale)
            if causal:
                j = torch.arange(s.shape[-1], device=s.device)
                future = (j[None, :] > torch.arange(a, b, device=s.device)[:, None])[:, None]
                s = s.masked_fill(future, float("-inf"))   # [Tq, 1, Tk] against [Tq, Hq, Tk]
            scores.append(s)
        m = torch.stack([s.amax(-1) for s in scores]).amax(0)
        acc, z = None, None
        for (k, v, _), s in zip(parts, scores):
            p = (s - m.unsqueeze(-1)).exp()
            if dropout:
                p = torch.nn.functional.dropout(p, dropout, True)
            z = p.sum(-1) if z is None else z + p.sum(-1)
            t = _gqa_apply(p, v.float())
            acc = t if acc is None else acc + t
        o[a:b] = acc / z.unsqueeze(-1)
    return o.to(q.dtype)


def _branch_attn_torch(q, k, v, plan, scale, dropout, out) -> None:
    """Branch attention with plain torch ops (CPU / no flash-attn): the same key sets as `_branch_attn_flash`."""
    spans = plan.spans()
    for s in plan.prefixes():
        a, b = spans[s]
        out[a:b] = _torch_union_attn(q[a:b], [(k[a:b], v[a:b], True)], scale, dropout)
    for t in plan.suffixes():
        a, b = spans[t]
        pa, pb = spans[plan.owner[t]]
        out[a:b] = _torch_union_attn(q[a:b], [(k[pa:pb], v[pa:pb], False), (k[a:b], v[a:b], True)], scale, dropout)


def _branch_full_attention(attn, hidden_states, ctx, position_embeddings=None, **kwargs):
    """Full-attention layer over a branch layout. Mirrors Qwen3_5Attention.forward, but reads the keys/values of the
    layout instead of a causal mask. Returns (output, None) like HF."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    shape = hidden_states.shape[:-1]
    hd = attn.head_dim
    query, gate = torch.chunk(attn.q_proj(hidden_states).view(*shape, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    query = attn.q_norm(query)
    key = attn.k_norm(attn.k_proj(hidden_states).view(*shape, -1, hd))
    value = attn.v_proj(hidden_states).view(*shape, -1, hd)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
    q, k, v = query[0], key[0], value[0]
    flash = ctx.varlen and _flash_ok()
    if flash:
        dt = next(attn.parameters()).dtype
        if q.dtype not in (torch.float16, torch.bfloat16) and dt in (torch.float16, torch.bfloat16):
            q, k, v = q.to(dt), k.to(dt), v.to(dt)   # flash-attn needs half precision (HF casts the same way)
    out = torch.empty_like(q)
    dropout = attn.attention_dropout if attn.training else 0.0
    if flash:
        _branch_attn_flash(q, k, v, ctx.plan, attn.scaling, dropout, out)
    else:
        _branch_attn_torch(q, k, v, ctx.plan, attn.scaling, dropout, out)
    return attn.o_proj(out.reshape(*shape, -1) * torch.sigmoid(gate)), None


def _fix_conv_edges(gdn, conv, x, plan) -> torch.Tensor:
    """Replace the first kernel-1 outputs of every segment of `conv` ([1, C, T] causal depthwise conv over the whole
    layout) with the exact values computed from that segment's left context: the preceding prefix's tail for a suffix,
    the zero pad for a prefix. `x` is the pre-conv layout [1, C, T]."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    spans, win, C = plan.spans(), gdn.conv_kernel_size - 1, x.shape[1]
    rows, take = [], []
    for s in range(len(plan.seg_len)):
        a, b = spans[s]
        if plan.owner[s] == s:
            seed = x.new_zeros(1, C, win)
        else:
            pa, pb = spans[plan.owner[s]]
            tail = x[:, :, max(pa, pb - win):pb]
            seed = torch.cat([x.new_zeros(1, C, win - tail.shape[2]), tail], dim=2)
        n = min(win, b - a)
        rows.append(torch.cat([seed, x[:, :, a:a + n]], dim=2))
        take.append(n)
    L = max(r.shape[2] for r in rows)
    batch = x.new_zeros(len(rows), C, L)
    for i, r in enumerate(rows):
        batch[i, :, :r.shape[2]] = r[0]
    fix = mq.causal_conv1d_fn(batch, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, activation=gdn.activation)
    pieces = []
    for s in range(len(plan.seg_len)):
        a, b = spans[s]
        pieces.append(torch.cat([fix[s:s + 1, :, win:win + take[s]], conv[:, :, a + take[s]:b]], dim=2))
    return torch.cat(pieces, dim=2)


def _channel_last(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T] with stride 1 over C (memory laid out as [B, T, C])."""
    return x if x.stride(1) == 1 else x.transpose(1, 2).contiguous().transpose(1, 2)


class _ChannelLastGrad(torch.autograd.Function):
    """Identity whose backward hands a channel-last gradient to the op before it (the seq_idx conv kernel rejects the
    contiguous [B, C, T] gradients that slicing/concatenation produce)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return _channel_last(g)


def _branch_conv(gdn, mixed, plan, varlen) -> torch.Tensor:
    """Short causal conv over the branch layout ([1, C, T] in, same out). Each segment is convolved as its own
    sequence (CUDA: the `seq_idx` kernel; CPU: the torch reference over the whole layout) and the first kernel-1
    outputs of every segment are then recomputed from its true left context, so a suffix's conv window contains the
    prefix's last kernel-1 conv inputs."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    w, bias = gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias
    if varlen:
        # causal-conv1d's seq_idx kernel needs int32 segment ids and channel-last x / dout (stride 1 over C)
        seg = torch.repeat_interleave(torch.arange(len(plan.seg_len), device=mixed.device, dtype=torch.int32),
                                      torch.as_tensor(plan.seg_len, device=mixed.device))
        conv = _ChannelLastGrad.apply(mq.causal_conv1d_fn(_channel_last(mixed), w, bias, activation=gdn.activation,
                                                          seq_idx=seg[None]))
    else:
        conv = mq.causal_conv1d_fn(mixed, w, bias, activation=gdn.activation)
    if gdn.conv_kernel_size <= 1:
        return conv
    return _fix_conv_edges(gdn, conv, mixed, plan)


def _gdn_scan(q, k, v, g, beta, initial_state=None, output_final_state=False, cu_seqlens=None):
    """HF's chunked gated delta rule (fla kernel on CUDA, torch reference on CPU — see `_cpu_safe_gdn`)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    return mq.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                           output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                                           cu_seqlens=cu_seqlens)


def _branch_gated_deltanet(gdn, hidden_states, ctx, cache_params=None, attention_mask=None, **kwargs):
    """Gated DeltaNet layer over a branch layout. Mirrors Qwen3_5GatedDeltaNet.forward, but the prefix scan runs once
    per set (keeping its final recurrent state) and every suffix branch starts from that state; the conv is seeded from
    the prefix's tail (see `_branch_conv`)."""
    plan = ctx.plan
    shape = hidden_states.shape[:-1]
    mixed = gdn.in_proj_qkv(hidden_states).transpose(1, 2)
    z = gdn.in_proj_z(hidden_states).reshape(*shape, -1, gdn.head_v_dim)
    b = gdn.in_proj_b(hidden_states)
    a = gdn.in_proj_a(hidden_states)
    mixed = _branch_conv(gdn, mixed, plan, ctx.varlen).transpose(1, 2)
    T = mixed.shape[1]
    query, key, value = torch.split(mixed, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    query = query.reshape(1, T, -1, gdn.head_k_dim)
    key = key.reshape(1, T, -1, gdn.head_k_dim)
    value = value.reshape(1, T, -1, gdn.head_v_dim)
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    if gdn.num_v_heads // gdn.num_k_heads > 1:
        rep = gdn.num_v_heads // gdn.num_k_heads
        query, key = query.repeat_interleave(rep, dim=2), key.repeat_interleave(rep, dim=2)
    out = torch.empty(1, T, gdn.num_v_heads, gdn.head_v_dim, dtype=value.dtype, device=value.device)
    spans = plan.spans()
    for s in plan.prefixes():
        a0, b0 = spans[s]
        o, state = _gdn_scan(query[:, a0:b0], key[:, a0:b0], value[:, a0:b0], g[:, a0:b0], beta[:, a0:b0],
                             output_final_state=True)
        out[:, a0:b0] = o
        kids = plan.children(s)
        if not kids:
            continue
        state = state if state.dim() == 4 else state[None]
        if ctx.varlen:   # all of the set's suffixes in one varlen call, each starting from the shared prefix state
            idx = _tok_index([spans[t] for t in kids], mixed.device)
            o, _ = _gdn_scan(query[:, idx], key[:, idx], value[:, idx], g[:, idx], beta[:, idx],
                             initial_state=state.expand(len(kids), -1, -1, -1).contiguous(),
                             cu_seqlens=_cum([plan.seg_len[t] for t in kids], mixed.device))
            out[0, idx] = o[0]
        else:
            for t in kids:
                a1, b1 = spans[t]
                o, _ = _gdn_scan(query[:, a1:b1], key[:, a1:b1], value[:, a1:b1], g[:, a1:b1], beta[:, a1:b1],
                                 initial_state=state)
                out[:, a1:b1] = o
    core = gdn.norm(out.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
    return gdn.out_proj(core.reshape(1, T, -1))


def _resolve_attn(impl: str, device: torch.device) -> str:
    if impl != "auto":
        return impl
    if device.type == "cuda":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


def _apply_liger(mcfg: dict, device: torch.device) -> bool:
    """Liger kernels through the HF integration (`liger_kernel.transformers`, the same entry point HF Trainer's
    `use_liger_kernel` uses). Patches the HF modeling classes BEFORE the model is built (RMSNorm incl. q/k norms,
    SwiGLU MLP). No LM head is used, so fused-linear-cross-entropy is off. Triton kernels are CUDA-only."""
    if not mcfg.get("liger_kernel") or device.type != "cuda":
        return False
    from transformers import AutoConfig
    from liger_kernel.transformers import monkey_patch as lk
    model_type = AutoConfig.from_pretrained(mcfg["name_or_path"]).model_type
    fn = lk.MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if fn is None:
        print(f"[scrm] liger: no kernels for model_type={model_type}; skipped")
        return False
    import inspect
    kw = dict(rope=False, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=True, swiglu=True)
    fn(**{k: v for k, v in kw.items() if k in inspect.signature(fn).parameters})
    print(f"[scrm] liger kernels applied ({fn.__name__}: rms_norm, swiglu)")
    return True


def _text_only(m: nn.Module) -> nn.Module:
    """Qwen3.5 checkpoints are vision-language (Qwen3_5Model = visual + language_model). Keep the text decoder only."""
    lm = getattr(m, "language_model", None)
    if lm is None:
        return m
    if hasattr(m, "visual"):
        del m.visual
    return lm


def _load_backbone(mcfg: dict, device: torch.device, tokenizer_len: int):
    import transformers
    from transformers import AutoModel

    _cpu_safe_gdn()

    dtype = torch.bfloat16 if (mcfg["dtype"] == "bfloat16" and device.type == "cuda") or \
        (mcfg["dtype"] == "bfloat16" and device.type != "cuda" and mcfg["name_or_path"] != "tiny") else torch.float32
    if mcfg["name_or_path"] == "tiny":
        from .tiny import make_tiny_backbone
        return make_tiny_backbone(mcfg["tiny"], 512, dtype=torch.float32).to(device)
    _apply_liger(mcfg, device)
    kw: dict[str, Any] = {"attn_implementation": _resolve_attn(mcfg["attn_implementation"], device)}
    kw["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = dtype
    if mcfg.get("quantize_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": device.index or 0}
        return _text_only(AutoModel.from_pretrained(mcfg["name_or_path"], **kw))
    m = _text_only(AutoModel.from_pretrained(mcfg["name_or_path"], **kw))
    return m.to(device)


def build_scrm(mcfg: dict, device: torch.device | str = "cpu", tokenizer=None, adapter_dir: str | None = None,
               seed: int = 0) -> tuple["SCRM", Any]:
    """Build tokenizer + backbone (+LoRA) + set block + head. `adapter_dir` loads a trained LoRA adapter."""
    from transformers import AutoTokenizer
    full = deep_update(json.loads(json.dumps(DEFAULTS["model"])), mcfg)
    mcfg = full
    device = torch.device(device)
    if tokenizer is None:
        if mcfg["name_or_path"] == "tiny":
            from .tiny import make_tiny_tokenizer
            tokenizer = make_tiny_tokenizer()
        else:
            tokenizer = AutoTokenizer.from_pretrained(mcfg["name_or_path"])
    tokenizer = prepare_tokenizer(tokenizer)
    backbone = _load_backbone(mcfg, device, len(tokenizer))
    hidden = backbone.config.hidden_size
    freeze = mcfg["freeze_backbone"]
    lora = mcfg["lora"]

    for p in backbone.parameters():
        p.requires_grad_(False)
    if mcfg["gradient_checkpointing"] and not freeze:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    backbone_trainable = False
    if lora["enabled"] and not freeze:
        from peft import LoraConfig, get_peft_model, PeftModel
        if adapter_dir:
            backbone = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=True)
        else:
            lc = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                            target_modules=lora["target_modules"], bias="none",
                            use_rslora=lora.get("use_rslora", True))
            backbone = get_peft_model(backbone, lc)
        backbone_trainable = True
    model = SCRM(backbone, hidden, mcfg, backbone_trainable=backbone_trainable)
    model.head_module().to(device)
    model.tokenizer = tokenizer
    _install_branching(backbone)
    return model, tokenizer


def load_scrm(ckpt_dir: str, device: str | torch.device = "auto", merged_ok: bool = True) -> "SCRM":
    """Load a checkpoint written by SCRM.save_pretrained (LoRA adapter + head + tokenizer + config).
    `ckpt_dir` may be `hf://org/repo[@revision]/run/best` (a hub backup); the base model comes from model.name_or_path.
    device="auto" -> cuda when available."""
    from transformers import AutoTokenizer
    from .hub import resolve_ckpt
    ckpt_dir = resolve_ckpt(ckpt_dir)
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ckpt_dir, "scrm_config.json")) as f:
        saved = json.load(f)
    mcfg = saved["model"]
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    adapter = os.path.join(ckpt_dir, "adapter")
    mcfg = dict(mcfg)
    if os.path.isdir(os.path.join(ckpt_dir, "backbone_merged")):
        mcfg["name_or_path"] = os.path.join(ckpt_dir, "backbone_merged")
        mcfg["lora"] = dict(mcfg["lora"], enabled=False)
        adapter = None
    model, tok = build_scrm(mcfg, device, tokenizer=tok, adapter_dir=adapter if adapter and os.path.isdir(adapter) else None)
    sd = torch.load(os.path.join(ckpt_dir, "scrm_head.pt"), map_location="cpu")
    model.load_head_state_dict(sd)
    model.render_cfg = saved["render"]
    model.eval()
    return model
