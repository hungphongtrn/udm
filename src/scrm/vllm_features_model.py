"""vLLM 0.22.1 model exposing the frozen multi-layer feature contract for Qwen3.5 text.

`scrm.vllm_features` registers this class under the architecture name ``Qwen3_5ForSCRMFeatures`` and
drives it with ``runner="pooling"`` + ``convert="none"``. The workload is prefill-only: vLLM's own
scheduler packs the prompt+suffix sequences, the native hybrid (full attention + gated delta net)
layers run once, and ``LastPool`` reads the hidden state of the last *input* token of each sequence.
This model has no logits, no sampling and no decode step.

Feature contract, identical to the Hugging Face path in `scrm.model.embed_prefix_cached`:

* layer ``l`` in ``1..num_hidden_layers`` -> the residual stream leaving decoder layer ``l`` (the
  ``hidden_states + residual`` sum that EAGLE-3 captures there), passed **once** through the
  checkpoint's final RMSNorm;
* ``-1`` (or ``num_hidden_layers``) -> the model's own final hidden state, i.e. that same residual
  stream after the last layer, passed once through that same final RMSNorm;
* the requested layers are concatenated on the hidden axis **in the order given by
  ``scrm_feature_layers``** (a vLLM ``hf_overrides`` entry), so one sequence yields one
  ``[len(layers) * hidden_size]`` vector.

The final RMSNorm is applied per layer *before* concatenation (it belongs to each single layer's
residual stream, it is not a norm over the concatenated vector), and only to the one last-token row
``LastPool`` keeps. The pooler head is built without an activation, so these are raw
checkpoint-normalized hidden states: never unit-length, whatever ``use_activation`` says.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple
import weakref

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead,
    LastPool,
    SequencePooler,
)
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

# Qwen3.5 checkpoints are published as the multimodal Qwen3_5ForConditionalGeneration, so their text
# parameters live under `model.language_model.`, the unused vision tower under `model.visual.`, and a
# multi-token-prediction head under `mtp.` (skipped by AutoWeightsLoader below). The text prefix maps
# onto this text-only backbone; the tower keys are dropped because the tower is never built here.
WEIGHT_PREFIX_MAP: dict[str, str | None] = {
    "model.language_model.": "model.",
    "model.visual.": None,
}


class FeaturePlan(NamedTuple):
    """How the requested feature layers map onto a single backbone forward pass."""

    layers: tuple[int, ...]
    """State index per output piece, in request order: the decoder layer whose residual stream it is."""

    capture: tuple[int, ...]
    """EAGLE-3 capture indices to enable, ascending (index ``k`` = residual stream after layer ``k``).

    Empty when every requested piece is the final hidden state, in which case the backbone returns a
    single tensor instead of a ``(hidden_states, aux_hidden_states)`` pair."""

    sources: tuple[int, ...]
    """Per piece: ``-1`` -> the backbone's own final hidden state, ``k >= 0`` -> ``capture[k]``."""


def plan_feature_layers(layers: Sequence[int], num_hidden_layers: int) -> FeaturePlan:
    """Validate ``scrm_feature_layers`` and map it onto backbone capture indices.

    Mirrors `scrm.model._check_layers`: every entry is ``-1`` or in ``1..num_hidden_layers``. The final
    layer is the backbone's own hidden state, which the backbone returns already normed, so it is never
    captured as an auxiliary state.
    """
    if not layers:
        raise ValueError("scrm_feature_layers must request at least one layer")
    want = [int(l) for l in layers]
    bad = [l for l in want if l != -1 and not 1 <= l <= num_hidden_layers]
    if bad:
        raise ValueError(f"scrm_feature_layers must be -1 or in 1..{num_hidden_layers}, got {bad}")
    states = tuple(num_hidden_layers if l == -1 else l for l in want)
    capture = tuple(sorted({s for s in states if s < num_hidden_layers}))
    capture_index = {s: k for k, s in enumerate(capture)}
    sources = tuple(
        -1 if s == num_hidden_layers else capture_index[s] for s in states
    )
    return FeaturePlan(states, capture, sources)


class _NormalizeFeatures(torch.nn.Module):
    """Apply the checkpoint norm only after LAST pooling, never over unused prefill tokens."""

    def __init__(self, model):
        super().__init__()
        # Do not register the parent or its norm again under the pooler: that creates cyclic modules/weight keys.
        self._model_ref = weakref.ref(model)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        model = self._model_ref()
        sources = model.feature_plan.sources
        pieces = pooled.chunk(len(sources), dim=-1)
        normed = [piece if source < 0 else model.model.norm(piece) for source, piece in zip(sources, pieces)]
        return normed[0] if len(normed) == 1 else torch.cat(normed, dim=-1)


_EmbeddingBackbone = as_embedding_model(Qwen3_5ForCausalLM)


class Qwen3_5ForSCRMFeatures(_EmbeddingBackbone, IsHybrid):
    """Text-only Qwen3.5 backbone with a last-token, multi-layer pooling head (see module docstring).

    Only the prefill path is used, but the class keeps the native hybrid declarations (`IsHybrid` plus
    the gated-delta-net state calculators) so that vLLM sizes and aligns the mamba state cache exactly
    as it does for the untampered ``Qwen3_5ForConditionalGeneration``.
    """

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix=WEIGHT_PREFIX_MAP)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "Feature extraction keeps the whole backbone in one pipeline stage; "
                "run with pipeline_parallel_size=1"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        hf_config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        # hf_overrides sets this on the top-level config; a text-only checkpoint has no separate one.
        requested = getattr(hf_config, "scrm_feature_layers", None)
        if requested is None:
            requested = getattr(text_config, "scrm_feature_layers", None)
        if not requested:
            raise ValueError(
                "scrm_feature_layers must be set through hf_overrides to the requested feature layers"
            )
        self.feature_plan = plan_feature_layers(requested, int(text_config.num_hidden_layers))
        self.set_aux_hidden_state_layers(self.feature_plan.capture)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        out = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(out, tuple):
            hidden_states, aux_hidden_states = out
        else:
            # No capture was requested: the backbone skips the auxiliary-state return entirely.
            hidden_states, aux_hidden_states = out, ()
        pieces = [hidden_states if source < 0 else aux_hidden_states[source] for source in self.feature_plan.sources]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)

    def _init_pooler(self, vllm_config: VllmConfig, prefix: str = "") -> DispatchPooler:
        """Last-token pooling, no activation, no sentence-transformers projector.

        Built explicitly instead of via ``DispatchPooler.for_embedding``: a head without an activation
        keeps the vectors raw even if a caller forgets ``use_activation=False``, which would otherwise
        silently store unit-length features that no longer match the HF path. Only the ``embed`` task is
        registered because ``token_embed`` would mean a different, per-token feature layout.
        """
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        seq_pooling_type = pooler_config.get_seq_pooling_type()
        if seq_pooling_type != "LAST":
            raise ValueError(
                "Feature extraction reads the hidden state of the last input token; "
                f"PoolerConfig(seq_pooling_type=...) must stay 'LAST', got {seq_pooling_type!r}"
            )
        head = EmbeddingPoolerHead(
            projector=_NormalizeFeatures(self), head_dtype=None, activation=None
        )
        return DispatchPooler({"embed": SequencePooler(pooling=LastPool(), head=head)})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        # Per-architecture verifier, keyed by the checkpoint's own architecture
        # (`Qwen3_5ForConditionalGeneration`), so our custom architecture name skips it: it resolves
        # `cache_config.mamba_ssm_cache_dtype` from the checkpoint's `mamba_ssm_dtype` (float32 for
        # Qwen3.5-4B, while the conv state follows `mamba_cache_dtype`). Replaying it here also fixes
        # the value for the mamba layers, which read the same shared cache config when they build
        # their KVCacheSpec: the platform calls this method while aligning the mamba page size, i.e.
        # before those specs exist, so alignment and allocation cannot disagree.
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
