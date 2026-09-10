from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models import resolve_model
from hipengine.models.timesfm3 import (
    PINNED_TIMESFM3_MODEL_ID,
    TIMESFM3,
    TIMESFM3_ARCHITECTURE,
    TimesFM3ModelSpec,
    expected_timesfm3_weight_shapes,
    parse_timesfm3_model_spec,
    validate_timesfm3_weight_index,
)


def config() -> dict:
    """The pinned config.json of google/timesfm-3.0-pytorch."""

    return {
        "input_patch_len": 32,
        "output_patch_len": 64,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "residual_block_config": {
            "activation": "relu",
            "dropout": 0.0,
            "hidden_dims": 1280,
            "identity_skip": False,
            "output_dims": 1280,
            "prenorm": "none",
            "use_bias": False,
        },
        "transformer_config": {
            "num_layers": 20,
            "transformer": {
                "attention_norm": "rms",
                "causal_attention": True,
                "debug_no_masking": False,
                "deterministic": True,
                "feedforward_norm": "rms",
                "ff_activation": "relu",
                "hidden_dims": 1280,
                "max_variates": 32,
                "model_dims": 1280,
                "num_heads": 16,
                "paired_token_skip_second": False,
                "qk_norm": "rms",
                "training": True,
                "use_bias": False,
                "use_memory_efficient_attention": True,
                "use_rope_seq": True,
                "use_rope_var": False,
                "use_sdpa": True,
                "v_norm": "none",
            },
            "use_remat": True,
        },
        "use_frozen_running_stats": False,
        "use_iterative_cpm_revin": True,
        "use_linear_detrending": True,
        "linear_detrending_threshold": 0.5,
        "use_stitching": True,
        "use_variate_attention": True,
        "value_clip": 1.0e20,
        "input_transform": "identity",
    }


def weight_index(spec: TimesFM3ModelSpec, *, mutate=None) -> WeightIndex:
    tensors = {
        name: TensorInfo(name, Path("fake.safetensors"), "F32", shape)
        for name, shape in expected_timesfm3_weight_shapes(spec).items()
    }
    if mutate is not None:
        mutate(tensors)
    return WeightIndex(Path("/fake/timesfm3"), config(), tensors, (Path("fake.safetensors"),))


def test_parse_timesfm3_model_spec_pins_actual_geometry() -> None:
    spec = parse_timesfm3_model_spec(config())
    assert spec.model_id == PINNED_TIMESFM3_MODEL_ID
    assert spec.architecture == TIMESFM3_ARCHITECTURE
    assert spec.model_type == "timesfm3"
    assert spec.stored_dtype == spec.runtime_dtype == "float32"
    assert spec.model_dims == 1_280
    assert spec.hidden_dims == 1_280
    assert spec.num_layers == 20
    assert spec.num_heads == 16
    assert spec.head_dim == 80
    assert spec.input_patch_len == 32
    assert spec.output_patch_len == 64
    assert spec.rolls == 2
    assert spec.tokenizer_input_dims == 192
    assert spec.output_head_dims == 576
    assert spec.quantiles == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert spec.median_index == 4
    assert spec.use_variate_attention
    assert spec.use_stitching
    assert spec.use_linear_detrending
    assert spec.linear_detrending_threshold == 0.5
    assert spec.use_iterative_cpm_revin
    assert not spec.use_frozen_running_stats
    assert spec.value_clip == 1.0e20
    assert spec.max_variates == 32
    assert spec.parameter_count == 330_710_976


def test_parse_timesfm3_model_spec_rejects_drift() -> None:
    def mutate(name: str, value) -> dict:
        cfg = config()
        if name == "num_layers":
            cfg["transformer_config"]["num_layers"] = value
        elif name == "use_variate_attention":
            cfg["use_variate_attention"] = value
        elif name == "output_patch_len":
            cfg["output_patch_len"] = value
        elif name == "ff_activation":
            cfg["transformer_config"]["transformer"]["ff_activation"] = value
        else:
            cfg[name] = value
        return cfg

    with pytest.raises(ValueError):
        parse_timesfm3_model_spec(mutate("num_layers", 24))
    with pytest.raises(ValueError):
        parse_timesfm3_model_spec(mutate("use_variate_attention", False))
    with pytest.raises(ValueError):
        parse_timesfm3_model_spec(mutate("output_patch_len", 33))  # not a multiple
    with pytest.raises(ValueError):
        parse_timesfm3_model_spec(mutate("ff_activation", "swish"))
    with pytest.raises(TypeError):
        parse_timesfm3_model_spec(mutate("quantiles", [0.1, 0.5, 0.9]))


def test_expected_weight_manifest() -> None:
    spec = parse_timesfm3_model_spec(config())
    shapes = expected_timesfm3_weight_shapes(spec)
    assert len(shapes) == 445  # 5 top-level + 20 layers x 22
    assert shapes["pre_transformer_resblock.hidden_layer.weight"] == (1280, 192)
    assert shapes["output_head.weight"] == (576, 1280)
    assert shapes["output_head.bias"] == (576,)
    assert shapes["transformer_stack.layers.19.ff1.weight"] == (1280, 1280)
    assert shapes["transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale"] == (80,)


def test_validate_timesfm3_weight_index() -> None:
    spec = parse_timesfm3_model_spec(config())
    validate_timesfm3_weight_index(spec, weight_index(spec))

    def drop_one(tensors):
        del tensors["output_head.bias"]

    with pytest.raises(ValueError, match="missing"):
        validate_timesfm3_weight_index(spec, weight_index(spec, mutate=drop_one))

    def wrong_shape(tensors):
        tensors["output_head.weight"] = TensorInfo(
            "output_head.weight", Path("fake.safetensors"), "F32", (512, 1280)
        )

    with pytest.raises(ValueError, match="shape"):
        validate_timesfm3_weight_index(spec, weight_index(spec, mutate=wrong_shape))


def test_plugin_registered() -> None:
    assert resolve_model(TIMESFM3_ARCHITECTURE) is TIMESFM3
    assert TIMESFM3.name == "timesfm_3p0"
    layer = TIMESFM3.transformer_layer_sequence()
    assert "timesfm3_var_attn" in layer
    assert "timesfm3_stitching" in TIMESFM3.layer_sequence()
