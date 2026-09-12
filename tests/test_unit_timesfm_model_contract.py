from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models import resolve_model
from hipengine.models.timesfm import (
    PINNED_TIMESFM_MODEL_ID,
    TIMESFM,
    TIMESFM_ARCHITECTURE,
    TimesFMModelSpec,
    expected_timesfm_weight_shapes,
    parse_timesfm_model_spec,
    validate_timesfm_weight_index,
)


def config() -> dict:
    """The pinned config.json of google/timesfm-2.5-200m-pytorch."""

    return {
        "architectures": [TIMESFM_ARCHITECTURE],
        "context_length": 16_384,
        "head_dim": 80,
        "hidden_size": 1_280,
        "horizon_length": 128,
        "intermediate_size": 1_280,
        "model_type": "timesfm",
        "num_attention_heads": 16,
        "num_hidden_layers": 20,
        "patch_length": 32,
        "quantile_horizon_length": 1_024,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "rms_norm_eps": 1.0e-6,
        "torch_compile": False,
    }


def weight_index(spec: TimesFMModelSpec, *, mutate=None) -> WeightIndex:
    tensors = {
        name: TensorInfo(name, Path("fake.safetensors"), "F32", shape)
        for name, shape in expected_timesfm_weight_shapes(spec).items()
    }
    if mutate is not None:
        mutate(tensors)
    return WeightIndex(Path("/fake/timesfm"), config(), tensors, (Path("fake.safetensors"),))


def test_parse_timesfm_model_spec_pins_actual_geometry() -> None:
    spec = parse_timesfm_model_spec(config())
    assert spec.model_id == PINNED_TIMESFM_MODEL_ID
    assert spec.architecture == TIMESFM_ARCHITECTURE
    assert spec.model_type == "timesfm"
    assert spec.stored_dtype == spec.runtime_dtype == "float32"
    assert spec.hidden_size == 1_280
    assert spec.num_hidden_layers == 20
    assert spec.num_attention_heads == 16
    assert spec.head_dim == 80
    assert spec.context_length == 16_384
    assert spec.patch_length == 32
    assert spec.horizon_length == 128
    assert spec.quantile_horizon_length == 1_024
    assert spec.quantiles == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert spec.rms_norm_eps == 1.0e-6
    assert spec.tokenizer_input_dims == 64
    assert spec.tokenizer_input_channels == 2
    assert spec.quantile_heads == 10
    assert spec.quantile_output_dims == 10_240
    assert spec.decode_index == 5
    assert spec.qkv_size == 3 * 1_280


def test_parse_timesfm_model_spec_rejects_drift() -> None:
    generation = config()
    for name, value in (
        ("hidden_size", 512),
        ("num_hidden_layers", 21),
        ("patch_length", 16),
        ("quantile_horizon_length", 512),
        ("horizon_length", 127),
        ("torch_compile", True),
    ):
        drifted = config()
        drifted[name] = value
        with pytest.raises(ValueError, match=name):
            parse_timesfm_model_spec(drifted)

    drifted = config()
    drifted["quantiles"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95]
    with pytest.raises(ValueError, match="quantiles"):
        parse_timesfm_model_spec(drifted)

    drifted = config()
    drifted["architectures"] = ["SomethingElse"]
    with pytest.raises(ValueError, match="architectures"):
        parse_timesfm_model_spec(drifted)


def test_complete_timesfm_weight_manifest_validates() -> None:
    spec = parse_timesfm_model_spec(config())
    shapes = expected_timesfm_weight_shapes(spec)
    assert len(shapes) == 232
    assert spec.parameter_count == 231_289_280
    assert sum(np.prod(shape, dtype=np.int64) for shape in shapes.values()) == 231_289_280
    assert shapes["tokenizer.hidden_layer.weight"] == (1_280, 64)
    assert shapes["tokenizer.hidden_layer.bias"] == (1_280,)
    assert shapes["stacked_xf.0.attn.qkv_proj.weight"] == (3_840, 1_280)
    assert shapes["stacked_xf.19.attn.per_dim_scale.per_dim_scale"] == (80,)
    assert shapes["stacked_xf.19.ff1.weight"] == (1_280, 1_280)
    assert shapes["output_projection_point.residual_layer.weight"] == (1_280, 1_280)
    assert shapes["output_projection_quantiles.output_layer.weight"] == (10_240, 1_280)
    validate_timesfm_weight_index(spec, weight_index(spec))


def test_weight_manifest_rejects_shape_dtype_and_missing() -> None:
    spec = parse_timesfm_model_spec(config())

    def wrong_shape(tensors):
        old = tensors["stacked_xf.0.attn.qkv_proj.weight"]
        tensors[old.name] = TensorInfo(old.name, old.shard_path, old.dtype, (3_839, 1_280))

    with pytest.raises(ValueError, match="shape"):
        validate_timesfm_weight_index(spec, weight_index(spec, mutate=wrong_shape))

    def wrong_dtype(tensors):
        old = tensors["stacked_xf.0.ff0.weight"]
        tensors[old.name] = TensorInfo(old.name, old.shard_path, "F16", old.shape)

    with pytest.raises(ValueError, match="dtype"):
        validate_timesfm_weight_index(spec, weight_index(spec, mutate=wrong_dtype))

    def missing(tensors):
        tensors.pop("tokenizer.output_layer.weight")

    with pytest.raises(ValueError, match="missing"):
        validate_timesfm_weight_index(spec, weight_index(spec, mutate=missing))

    def extra(tensors):
        tensors["stacked_xf.20.ff0.weight"] = TensorInfo(
            "stacked_xf.20.ff0.weight", Path("fake.safetensors"), "F32", (1_280, 1_280)
        )

    with pytest.raises(ValueError, match="extra"):
        validate_timesfm_weight_index(spec, weight_index(spec, mutate=extra))


def test_timesfm_model_plugin_resolves_and_exposes_unfused_decode_sequence() -> None:
    assert resolve_model(TIMESFM_ARCHITECTURE) is TIMESFM
    sequence = tuple(TIMESFM.layer_sequence())
    assert sequence[0] == "timesfm_tokenizer"
    assert "timesfm_qkv_proj" in sequence
    assert "timesfm_rope" in sequence
    assert "timesfm_qk_rmsnorm" in sequence
    assert "timesfm_per_dim_scale" in sequence
    assert "timesfm_unscaled_attention" in sequence
    assert "timesfm_ff_swish" in sequence
    assert sequence[-2:] == (
        "timesfm_output_projection_point",
        "timesfm_output_projection_quantiles",
    )
    layer_sequence = tuple(TIMESFM.transformer_layer_sequence())
    assert len(layer_sequence) == 11
