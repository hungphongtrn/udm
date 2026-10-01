"""Shared-prefix encoding for Qwen3.5 (hybrid Gated DeltaNet + full attention), autograd-safe.

Every graded option of a decision set shares the same prompt prefix (state, instruction, all options,
"Grade this choice: "). The prefix is encoded ONCE; its per-layer states are then reused by all graded-option
suffixes, which run as one batch:

* full-attention layers: the prefix K/V (after RoPE) are concatenated in front of each suffix's K/V;
* Gated DeltaNet layers: the last conv-kernel inputs and the final recurrent state of the prefix seed each suffix.

The HF cache classes update their buffers in place (which breaks autograd through the prefix) and HF drops caches
under gradient checkpointing. The two tiny cache objects below are purely functional instead: `RecordCache` only
records what the prefix pass computes, `ExtendCache` only reads it. Re-running a layer (checkpoint recompute) gives
the same result, so `_can_checkpoint_with_cache` can be enabled on the decoder layers. Gradients flow from every
suffix back into the shared prefix. The suffix pass needs explicit 4D masks, so full attention runs through SDPA.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _LayerView:
    def __init__(self, cache, idx):
        self.cache, self.idx = cache, idx
        self.record_past = True          # never take HF's in-place single-token decode path

    @property
    def recurrent_states(self):
        return [self.cache.recurrent(self.idx)]


class RecordCache:
    """Prefix pass (batch 1): behaves like no cache, but records each layer's states."""

    def __init__(self, n_layers: int):
        self.kv, self.conv, self.rec = {}, {}, {}
        self.layers = [_LayerView(self, i) for i in range(n_layers)]

    def update(self, k, v, layer_idx, *args, **kwargs):
        self.kv[layer_idx] = (k, v)
        return k, v

    def has_previous_state(self, layer_idx, state_idx=0):
        return False

    def update_conv_state(self, x, layer_idx, conv_kernel_size=None, **kwargs):
        k = conv_kernel_size if isinstance(conv_kernel_size, int) else conv_kernel_size[0]
        tail = x[..., -k:]
        if tail.shape[-1] < k:
            tail = F.pad(tail, (k - tail.shape[-1], 0))   # == the causal conv's implicit zero padding
        self.conv[layer_idx] = tail
        return x

    def update_recurrent_state(self, s, layer_idx, **kwargs):
        self.rec[layer_idx] = s
        return s

    def recurrent(self, idx):
        return None

    def get_seq_length(self, layer_idx=0):
        return 0


class ExtendCache:
    """Suffix pass (batch n): prepends the recorded prefix states (shared across the n suffixes)."""

    def __init__(self, rec: RecordCache, n: int):
        self.kv, self.conv, self.rec, self.n = dict(rec.kv), dict(rec.conv), dict(rec.rec), n
        self.layers = [_LayerView(self, i) for i in range(len(rec.layers))]

    def update(self, k, v, layer_idx, *args, **kwargs):
        pk, pv = self.kv[layer_idx]
        n = k.shape[0]
        return (torch.cat([pk.expand(n, -1, -1, -1), k], dim=2), torch.cat([pv.expand(n, -1, -1, -1), v], dim=2))

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, x, layer_idx, conv_kernel_size=None, **kwargs):
        tail = self.conv[layer_idx]
        return torch.cat([tail.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=-1)

    def update_recurrent_state(self, s, layer_idx, **kwargs):
        return s

    def recurrent(self, idx):
        s = self.rec[idx]
        return s.expand(self.n, *s.shape[1:]).contiguous()

    def get_seq_length(self, layer_idx=0):
        return 0


def enable_cache_under_checkpointing(model) -> None:
    """Our caches are side-effect free, so keep them when HF gradient checkpointing is on."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    for m in model.modules():
        if isinstance(m, GradientCheckpointingLayer):
            m._can_checkpoint_with_cache = True


def _n_layers(backbone) -> int:
    cfg = backbone.config
    return int(getattr(cfg, "num_hidden_layers", None) or cfg.get_text_config().num_hidden_layers)


def encode_shared_prefix(backbone, prefix: torch.Tensor, suffix: torch.Tensor, suffix_mask: torch.Tensor,
                         read_pos: torch.Tensor, max_tokens: int | None = None) -> torch.Tensor:
    """prefix [P] ids; suffix [n, S] ids (right padded) with suffix_mask [n, S]; read_pos [n] (index in suffix).
    Returns the last-layer hidden state at each read position: [n, d]. Equivalent to encoding the n full sequences
    prefix + suffix_i independently (up to kernel numerics), at ~1/n of the prefix cost."""
    dev = prefix.device
    P = prefix.numel()
    rec = RecordCache(_n_layers(backbone))
    backbone(input_ids=prefix[None], attention_mask={"full_attention": None, "linear_attention": None},
             position_ids=torch.arange(P, device=dev)[None], past_key_values=rec, use_cache=False)
    n, S = suffix.shape
    step = n if not max_tokens else max(1, int(max_tokens) // max(S, 1))
    outs = []
    for a in range(0, n, step):
        sx, sm, rp = suffix[a:a + step], suffix_mask[a:a + step].bool(), read_pos[a:a + step]
        m = sx.shape[0]
        causal = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
        allowed = torch.cat([torch.ones(m, S, P, dtype=torch.bool, device=dev), causal.expand(m, S, S) & sm[:, None, :]],
                            dim=-1)
        masks = {"full_attention": allowed[:, None], "linear_attention": sm.long() if not sm.all() else None}
        pos = (P + torch.arange(S, device=dev))[None].expand(m, S)
        h = backbone(input_ids=sx, attention_mask=masks, position_ids=pos, past_key_values=ExtendCache(rec, m),
                     use_cache=False).last_hidden_state
        outs.append(h[torch.arange(m, device=dev), rp])
    return torch.cat(outs, 0)
