"""GPU-free feature math and cache-precision boundaries; run in the pinned vLLM environment."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="The multi-layer pooler requires the isolated vLLM environment")

from scrm.vllm_features_model import (  # noqa: E402
    Qwen3_5ForSCRMFeatures,
    _NormalizeFeatures,
    plan_feature_layers,
)


@pytest.mark.parametrize("layers", [[], [0], [33], [-2], [1, 33]])
def test_invalid_feature_layer_rejected(layers):
    with pytest.raises(ValueError):
        plan_feature_layers(layers, 32)


class _FeatureHost(torch.nn.Module):
    """Only the pooling math needs weights; no checkpoint, decoder, or GPU is constructed."""

    def __init__(self, layers):
        super().__init__()
        self.feature_plan = plan_feature_layers(layers, 32)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.RMSNorm(2, eps=1e-6)
        with torch.no_grad():
            self.model.norm.weight.copy_(torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("last_layer", [-1, 32])
def test_checkpoint_norm_is_per_layer_and_final_state_is_not_renormed(last_layer):
    host = _FeatureHost([16, last_layer, 24])
    pooled = torch.tensor([[3.0, 4.0, 11.0, 12.0, 5.0, 12.0]])
    result = _NormalizeFeatures(host)(pooled)
    expected = torch.cat([
        torch.tensor([[3.0, 8.0]]) / (12.5 + 1e-6) ** 0.5,
        torch.tensor([[11.0, 12.0]]),
        torch.tensor([[5.0, 24.0]]) / (84.5 + 1e-6) ** 0.5,
    ], dim=-1)
    torch.testing.assert_close(result, expected)
    # One RMSNorm over all six dimensions or a final L2 activation would change these values.


def test_final_only_feature_preserves_checkpoint_output():
    host = _FeatureHost([-1])
    pooled = torch.tensor([[11.0, 12.0], [7.0, 19.0]])
    torch.testing.assert_close(_NormalizeFeatures(host)(pooled), pooled)


def _cache_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_text_config=SimpleNamespace(mamba_ssm_dtype="float32")
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto", mamba_ssm_cache_dtype="auto"),
    )


def test_checkpoint_ssm_precision_and_explicit_override():
    config = _cache_config()
    assert Qwen3_5ForSCRMFeatures.get_mamba_state_dtype_from_config(config) == (
        torch.bfloat16, torch.float32
    )
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"
    config.cache_config.mamba_ssm_cache_dtype = "float16"
    assert Qwen3_5ForSCRMFeatures.get_mamba_state_dtype_from_config(config) == (
        torch.bfloat16, torch.float16
    )
