"""Prefill-only vLLM engine for the frozen, multi-layer feature contract."""
from __future__ import annotations

from importlib.metadata import version
import os

import torch

from .tokens import prepare_tokenizer

VLLM_VERSION = "0.19.1"
ARCHITECTURE = "Qwen3_5ForSCRMFeatures"


def _numeric_visible_devices() -> None:
    """vLLM's NVML helpers `int()` each CUDA_VISIBLE_DEVICES entry; schedulers often export GPU UUIDs instead.

    Rewrite UUIDs (or unique UUID prefixes) to NVML indices, which follow PCI bus order, and pin CUDA to the
    same order. The selected physical GPUs are unchanged.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if all(e.isdigit() for e in entries):
        return
    from vllm.utils.import_utils import import_pynvml
    nvml = import_pynvml()
    nvml.nvmlInit()
    try:
        uuids = [nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i)) for i in range(nvml.nvmlDeviceGetCount())]
    finally:
        nvml.nvmlShutdown()
    uuids = [u.decode() if isinstance(u, bytes) else u for u in uuids]
    indices = []
    for e in entries:
        if e.isdigit():
            indices.append(e)
            continue
        match = [i for i, u in enumerate(uuids) if u.startswith(e if e.startswith("GPU-") else f"GPU-{e}")]
        if len(match) != 1:
            raise ValueError(f"CUDA_VISIBLE_DEVICES entry {e!r} does not identify exactly one GPU (MIG unsupported)")
        indices.append(str(match[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class VLLMFeatureExtractor:
    """Submit independent prompt+option sequences; let vLLM pack and schedule them."""

    def __init__(self, cfg: dict, device):
        if device.type != "cuda":
            raise ValueError("features.backend=vllm requires a CUDA device")
        if version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"Feature pooling requires vllm=={VLLM_VERSION}; run `uv sync` (cu128 group)")
        mcfg, fcfg = cfg["model"], cfg["features"]
        if mcfg["dtype"] != "bfloat16" or mcfg.get("quantize_4bit", False):
            raise ValueError("The vLLM feature contract requires an unquantized bfloat16 checkpoint")
        self.layers = list(fcfg["layers"])
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("features.layers must be a nonempty list of distinct layer indices")
        ecfg = fcfg["vllm"]
        self.request_batch_size = int(ecfg["request_batch_size"])
        if self.request_batch_size < 1:
            raise ValueError("features.vllm.request_batch_size must be positive")
        max_len = int(cfg["data"]["render"]["max_len"])
        if ecfg["enable_prefix_caching"] and not ecfg["enable_chunked_prefill"]:
            raise ValueError("Qwen3.5 aligned prefix caching requires features.vllm.enable_chunked_prefill=true")
        token_budget = int(ecfg["max_num_batched_tokens"])
        if not ecfg["enable_chunked_prefill"] and token_budget < max_len:
            raise ValueError("Unchunked prefill requires features.vllm.max_num_batched_tokens >= data.render.max_len")

        # A foreground Ctrl-C must not kill a separate EngineCore before the shard can commit.
        # Single GPU / TP=1 uses the same native scheduler and kernels in the calling process.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        _numeric_visible_devices()
        from vllm import LLM, ModelRegistry, PoolingParams
        from vllm.config import PoolerConfig

        # Register the class object, not a lazy "module:Class" string: vLLM inspects lazy entries in a child Python
        # process that hides the real import error. The engine runs in this process anyway (no CUDA-fork concern).
        from .vllm_features_model import Qwen3_5ForSCRMFeatures
        ModelRegistry.register_model(ARCHITECTURE, Qwen3_5ForSCRMFeatures)
        self.engine = LLM(
            model=mcfg["name_or_path"], runner="pooling", convert="none", model_impl="vllm",
            hf_overrides={"architectures": [ARCHITECTURE], "scrm_feature_layers": self.layers},
            pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=False),
            dtype="bfloat16", max_model_len=max_len,
            gpu_memory_utilization=float(ecfg["gpu_memory_utilization"]),
            max_num_batched_tokens=token_budget, max_num_seqs=int(ecfg["max_num_seqs"]),
            enable_prefix_caching=bool(ecfg["enable_prefix_caching"]),
            enable_chunked_prefill=bool(ecfg["enable_chunked_prefill"]),
            mamba_cache_mode="align" if ecfg["enable_prefix_caching"] else "none",
            enforce_eager=True, seed=int(cfg["seed"]),
        )
        self.d_hidden = int(self.engine.llm_engine.model_config.hf_text_config.hidden_size)
        self.tokenizer = prepare_tokenizer(self.engine.get_tokenizer())
        # These are raw checkpoint-normalized hidden states, not unit-length embeddings.
        self.pooling_params = PoolingParams(use_activation=False)

    @torch.no_grad()
    def embed_items(self, items) -> dict[int, torch.Tensor]:
        count = sum(len(it.suffixes) for it in items)
        storage = torch.empty((count, len(self.layers), self.d_hidden), dtype=torch.bfloat16)
        features = {layer: storage[:, i, :] for i, layer in enumerate(self.layers)}
        width = len(self.layers) * self.d_hidden

        def encode(batch: list[tuple[int, list[int]]]):
            outputs = self.engine.encode([{"prompt_token_ids": ids} for _, ids in batch],
                                         pooling_params=self.pooling_params, pooling_task="embed", use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned an invalid multi-layer feature batch")
            for (row, _), out in zip(batch, outputs):
                values = out.outputs.data
                if values.ndim != 1 or values.numel() != width:
                    raise RuntimeError("vLLM returned an invalid multi-layer feature vector")
                # Direct cast into final CPU storage; no float-list conversion or temporary stacked matrix.
                storage[row].copy_(values.reshape(len(self.layers), self.d_hidden))

        def submit(window):
            # All options of a set share its prompt. Submitted together, siblings are admitted while the prompt is
            # still being prefilled, so (especially for the hybrid Mamba state, cached only at aligned, computed
            # boundaries) they recompute it. Finish one option per set first, then its siblings reuse the cache.
            firsts = [reqs[0] for reqs in window]
            rest = [r for reqs in window for r in reqs[1:]]
            encode(firsts)
            if rest:
                encode(rest)

        window, pending, row = [], 0, 0
        for it in items:
            prefix = it.prefix.tolist()
            reqs = []
            for suffix in it.suffixes:
                reqs.append((row, prefix + suffix.tolist()))
                row += 1
            if reqs:
                window.append(reqs)
                pending += len(reqs)
            if pending >= self.request_batch_size:
                submit(window)
                window, pending = [], 0
        if window:
            submit(window)
        return features
