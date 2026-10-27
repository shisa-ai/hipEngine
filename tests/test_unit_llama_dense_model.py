"""Dense Llama admission: config validation, tensor planning, and FP8 dequant.

Admission is decided by what the checkpoint declares, so the rejection cases here
are the named capability misses -- an unsupported bias flag, activation, rotary
scaling, or head geometry -- rather than an identity allowlist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.llama_safetensors import (
    fp8_e4m3fn_to_float32,
    plan_llama_tensor_sources,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models.llama import LlamaCapabilityMiss, parse_llama_spec
from tests._llama_tiny import TINY_LLAMA_CONFIG


def _index(config: dict[str, object], *, dtypes: dict[str, str] | None = None) -> WeightIndex:
    spec = parse_llama_spec(config)
    shapes = spec.expected_weight_shapes(lm_head=not spec.tie_word_embeddings)
    tensors: dict[str, TensorInfo] = {}
    overrides = dtypes or {}
    for name, shape in shapes.items():
        dtype = overrides.get(name, "BF16")
        tensors[name] = TensorInfo(
            name=name, shard_path=Path("/dev/null"), dtype=dtype, shape=tuple(shape)
        )
        if dtype == "F8_E4M3":
            scale_name = f"{name.rsplit('.', 1)[0]}.weight_scale"
            tensors[scale_name] = TensorInfo(
                name=scale_name,
                shard_path=Path("/dev/null"),
                dtype="F32",
                shape=(int(shape[0]), 1),
            )
    return WeightIndex(
        model_path=Path("/tmp/tiny-llama"),
        config=dict(config),
        tensors=tensors,
        shards=(Path("/dev/null"),),
    )


def test_tiny_config_parses_into_the_expected_geometry() -> None:
    spec = parse_llama_spec(TINY_LLAMA_CONFIG)
    assert spec.hidden_size == 256
    assert spec.q_width == 8 * 32
    assert spec.kv_width == 4 * 32
    assert spec.query_group_size == 2
    assert spec.attention_scale == pytest.approx(32**-0.5)
    assert spec.full_attention_layer_ids() == (0, 1)
    assert spec.compact_layer_index(1) == 1
    with pytest.raises(ValueError):
        spec.compact_layer_index(2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attention_bias": True}, "attention_bias"),
        ({"mlp_bias": True}, "mlp_bias"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"rope_scaling": {"type": "linear", "factor": 2.0}}, "rope_scaling"),
        ({"num_attention_heads": 6}, "divisible"),
        ({"head_dim": 33}, "even"),
        ({"num_query_groups": 3}, "num_query_groups"),
        ({"ffn_hidden_size": 1024}, "ffn_hidden_size"),
        ({"rotary_base": 500000.0}, "rotary_base"),
        ({"model_type": "qwen3"}, "model_type"),
    ],
)
def test_unsupported_declarations_raise_named_capability_misses(override, message) -> None:
    config = dict(TINY_LLAMA_CONFIG)
    config.update(override)
    with pytest.raises(LlamaCapabilityMiss, match=message):
        parse_llama_spec(config)


def test_missing_required_field_names_the_field() -> None:
    config = dict(TINY_LLAMA_CONFIG)
    del config["hidden_size"]
    with pytest.raises(LlamaCapabilityMiss, match="hidden_size"):
        parse_llama_spec(config)


def test_bf16_checkpoint_plans_norm_weights_as_fp32() -> None:
    index = _index(TINY_LLAMA_CONFIG)
    sources = plan_llama_tensor_sources(index, parse_llama_spec(TINY_LLAMA_CONFIG))
    by_name = {source.runtime_name: source for source in sources}
    assert by_name["model.norm.weight"].is_norm_weight
    assert by_name["model.layers.0.input_layernorm.weight"].is_norm_weight
    assert not by_name["model.layers.0.self_attn.q_proj.weight"].is_norm_weight
    assert by_name["model.embed_tokens.weight"].storage == "bf16"
    assert len(sources) == len(parse_llama_spec(TINY_LLAMA_CONFIG).expected_weight_shapes())


def test_float8_weights_require_a_per_channel_scale() -> None:
    index = _index(
        TINY_LLAMA_CONFIG,
        dtypes={"model.layers.0.mlp.gate_proj.weight": "F8_E4M3"},
    )
    del index.tensors["model.layers.0.mlp.gate_proj.weight_scale"]
    with pytest.raises(LlamaCapabilityMiss, match="weight_scale"):
        plan_llama_tensor_sources(index, parse_llama_spec(TINY_LLAMA_CONFIG))


def test_float8_scale_shape_must_match_the_output_channels() -> None:
    index = _index(
        TINY_LLAMA_CONFIG,
        dtypes={"model.layers.0.mlp.gate_proj.weight": "F8_E4M3"},
    )
    name = "model.layers.0.mlp.gate_proj.weight_scale"
    index.tensors[name] = TensorInfo(
        name=name, shard_path=Path("/dev/null"), dtype="F32", shape=(3, 1)
    )
    with pytest.raises(LlamaCapabilityMiss, match="one scale per output channel"):
        plan_llama_tensor_sources(index, parse_llama_spec(TINY_LLAMA_CONFIG))


def test_missing_tensor_is_a_named_capability_miss() -> None:
    index = _index(TINY_LLAMA_CONFIG)
    del index.tensors["model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(LlamaCapabilityMiss, match="down_proj"):
        plan_llama_tensor_sources(index, parse_llama_spec(TINY_LLAMA_CONFIG))


def test_shape_disagreement_with_config_is_refused() -> None:
    index = _index(TINY_LLAMA_CONFIG)
    name = "model.layers.0.self_attn.q_proj.weight"
    index.tensors[name] = TensorInfo(
        name=name, shard_path=Path("/dev/null"), dtype="BF16", shape=(128, 256)
    )
    with pytest.raises(LlamaCapabilityMiss, match="config declares"):
        plan_llama_tensor_sources(index, parse_llama_spec(TINY_LLAMA_CONFIG))


def test_tied_embeddings_do_not_require_an_lm_head_tensor() -> None:
    config = dict(TINY_LLAMA_CONFIG)
    config["tie_word_embeddings"] = True
    index = _index(config)
    assert "lm_head.weight" not in index.tensors
    sources = plan_llama_tensor_sources(index, parse_llama_spec(config))
    assert "lm_head.weight" not in {source.runtime_name for source in sources}


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ((0x00, 0x01, 0x08, 0x38, 0x3C, 0x7E, 0xB8, 0xFE), (0.0, 2.0**-9, 2.0**-6, 1.0, 1.5, 448.0, -1.0, -448.0)),
    ],
)
def test_fp8_e4m3fn_decodes_to_the_reference_values(codes, expected) -> None:
    decoded = fp8_e4m3fn_to_float32(np.asarray(codes, dtype=np.uint8).tobytes(), (len(codes),))
    assert np.array_equal(decoded, np.asarray(expected, dtype=np.float32))


def test_fp8_nan_codes_decode_to_nan() -> None:
    decoded = fp8_e4m3fn_to_float32(bytes((0x7F, 0xFF)), (2,))
    assert np.all(np.isnan(decoded))
