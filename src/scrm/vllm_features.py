"""Prefill-only vLLM engine for the frozen, multi-layer feature contract."""
from __future__ import annotations

from importlib.metadata import version
import os

import torch

from .tokens import prepare_tokenizer

VLLM_VERSION = "0.19.1"
ARCHITECTURE = "Qwen3_5ForSCRMFeatures"


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
        from vllm import LLM, ModelRegistry, PoolingParams
        from vllm.config import PoolerConfig

        ModelRegistry.register_model(ARCHITECTURE, f"scrm.vllm_features_model:{ARCHITECTURE}")
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
        requests = []
        offset = 0

        def submit():
            nonlocal offset
            outputs = self.engine.encode(requests, pooling_params=self.pooling_params, pooling_task="embed",
                                         use_tqdm=False)
            if len(outputs) != len(requests):
                raise RuntimeError("vLLM returned an invalid multi-layer feature batch")
            for j, out in enumerate(outputs):
                values = out.outputs.data
                if values.ndim != 1 or values.numel() != width:
                    raise RuntimeError("vLLM returned an invalid multi-layer feature vector")
                # Direct cast into final CPU storage; no float-list conversion or temporary stacked matrix.
                storage[offset + j].copy_(values.reshape(len(self.layers), self.d_hidden))
            offset += len(outputs)
            requests.clear()

        for it in items:
            prefix = it.prefix.tolist()
            for suffix in it.suffixes:
                requests.append({"prompt_token_ids": prefix + suffix.tolist()})
                if len(requests) == self.request_batch_size:
                    submit()
        if requests:
            submit()
        return features
