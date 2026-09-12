from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.moonshine import convert_moonshine_weight_to_fp16
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models import resolve_model
from hipengine.models.moonshine import (
    MOONSHINE,
    MoonshineModelSpec,
    expected_moonshine_weight_shapes,
    normalize_moonshine_token_ids,
    parse_moonshine_model_spec,
    validate_moonshine_weight_index,
)


def config() -> dict:
    return {
        "architectures": ["MoonshineForConditionalGeneration"],
        "attention_bias": False,
        "bos_token_id": 1,
        "decoder_hidden_act": "silu",
        "decoder_num_attention_heads": 8,
        "decoder_num_hidden_layers": 8,
        "decoder_num_key_value_heads": 8,
        "decoder_start_token_id": 1,
        "dtype": "float32",
        "encoder_hidden_act": "gelu",
        "encoder_num_attention_heads": 8,
        "encoder_num_hidden_layers": 8,
        "encoder_num_key_value_heads": 8,
        "eos_token_id": 2,
        "hidden_size": 416,
        "intermediate_size": 1664,
        "is_encoder_decoder": True,
        "max_position_embeddings": 194,
        "model_type": "moonshine",
        "pad_head_dim_to_multiple_of": 8,
        "pad_token_id": 2,
        "partial_rotary_factor": 0.62,
        "rope_parameters": {
            "partial_rotary_factor": 0.62,
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
        "tie_word_embeddings": True,
        "vocab_size": 36_864,
    }


def generation_config() -> dict:
    return {
        "bos_token_id": 1,
        "decoder_start_token_id": 1,
        "do_sample": False,
        "eos_token_id": [2],
        "max_length": 195,
        "num_beams": 5,
        "pad_token_id": 2,
        "use_cache": True,
    }


def weight_index(spec: MoonshineModelSpec, *, mutate=None) -> WeightIndex:
    tensors = {
        name: TensorInfo(name, Path("fake.safetensors"), "F32", shape)
        for name, shape in expected_moonshine_weight_shapes(spec).items()
    }
    if mutate is not None:
        mutate(tensors)
    return WeightIndex(Path("/fake/moonshine"), config(), tensors, (Path("fake.safetensors"),))


@pytest.mark.parametrize(("value", "expected"), [(2, (2,)), ([2], (2,)), ([2, 7], (2, 7))])
def test_normalize_moonshine_token_ids_accepts_scalar_and_list(value, expected) -> None:
    assert normalize_moonshine_token_ids(value, "eos_token_id", vocab_size=10) == expected


@pytest.mark.parametrize("value", [True, [], [2, 2], [-1], [10], None, "2"])
def test_normalize_moonshine_token_ids_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        normalize_moonshine_token_ids(value, "eos_token_id", vocab_size=10)


def test_parse_moonshine_model_spec_pins_actual_geometry_and_tokens() -> None:
    spec = parse_moonshine_model_spec(config(), generation_config())
    assert spec.architecture == "MoonshineForConditionalGeneration"
    assert spec.hidden_size == 416
    assert spec.encoder_layers == spec.decoder_layers == 8
    assert spec.head_dim == 52
    assert spec.padded_head_dim == 56
    assert spec.rotary_dim == 32
    assert spec.max_positions == spec.self_cache_capacity == 194
    assert spec.bos_token_ids == (1,)
    assert spec.decoder_start_token_id == 1
    assert spec.eos_token_ids == (2,)
    assert spec.pad_token_id == 2
    assert spec.generation_max_length == 195
    assert spec.generation_num_beams == 5
    assert spec.encoder_buckets == ((16_000, 40), (80_000, 207), (480_000, 1248))


def test_parse_moonshine_model_spec_accepts_singleton_lists_but_rejects_drift() -> None:
    generation = generation_config()
    generation["bos_token_id"] = [1]
    generation["decoder_start_token_id"] = [1]
    assert parse_moonshine_model_spec(config(), generation).decoder_start_token_id == 1

    generation["eos_token_id"] = [3]
    with pytest.raises(ValueError, match="differs"):
        parse_moonshine_model_spec(config(), generation)

    changed = config()
    changed["hidden_size"] = 512
    with pytest.raises(ValueError, match="hidden_size"):
        parse_moonshine_model_spec(changed, generation_config())


def test_complete_moonshine_weight_manifest_and_tied_owner_validate() -> None:
    spec = parse_moonshine_model_spec(config(), generation_config())
    shapes = expected_moonshine_weight_shapes(spec)
    assert len(shapes) == 210
    assert sum(np.prod(shape, dtype=np.int64) for shape in shapes.values()) == 63_217_856
    assert shapes[spec.embedding_weight_name] == (36_864, 416)
    assert spec.lm_head_alias_name == "proj_out.weight"
    assert spec.lm_head_alias_name not in shapes
    assert shapes["model.decoder.layers.7.mlp.fc1.weight"] == (3328, 416)
    assert shapes["model.encoder.conv2.weight"] == (832, 416, 7)
    validate_moonshine_weight_index(spec, weight_index(spec))


def test_weight_manifest_rejects_shape_dtype_missing_and_separate_lm_head() -> None:
    spec = parse_moonshine_model_spec(config(), generation_config())

    def wrong_shape(tensors):
        old = tensors["model.decoder.layers.0.mlp.fc1.weight"]
        tensors[old.name] = TensorInfo(old.name, old.shard_path, old.dtype, (3327, 416))

    with pytest.raises(ValueError, match="shape"):
        validate_moonshine_weight_index(spec, weight_index(spec, mutate=wrong_shape))

    def wrong_dtype(tensors):
        old = tensors[spec.embedding_weight_name]
        tensors[old.name] = TensorInfo(old.name, old.shard_path, "F16", old.shape)

    with pytest.raises(ValueError, match="dtype"):
        validate_moonshine_weight_index(spec, weight_index(spec, mutate=wrong_dtype))

    def missing(tensors):
        tensors.pop("model.decoder.norm.weight")

    with pytest.raises(ValueError, match="missing"):
        validate_moonshine_weight_index(spec, weight_index(spec, mutate=missing))

    def untied(tensors):
        tensors[spec.lm_head_alias_name] = TensorInfo(
            spec.lm_head_alias_name,
            Path("fake.safetensors"),
            "F32",
            (spec.vocab_size, spec.hidden_size),
        )

    with pytest.raises(ValueError, match="extra|tied"):
        validate_moonshine_weight_index(spec, weight_index(spec, mutate=untied))


def test_convert_moonshine_weight_to_fp16_is_contiguous_finite_and_deterministic() -> None:
    source = np.asarray([[1.0001, -2.0003], [0.0, 65_504.0]], dtype=np.float32)[:, ::-1]
    converted = convert_moonshine_weight_to_fp16("weight", source)
    assert converted.dtype == np.float16
    assert converted.flags.c_contiguous
    np.testing.assert_array_equal(converted, np.ascontiguousarray(source, dtype=np.float16))
    with pytest.raises(ValueError, match="finite"):
        convert_moonshine_weight_to_fp16("bad", np.asarray([np.inf], dtype=np.float32))


def test_moonshine_model_plugin_resolves_and_exposes_unfused_decode_sequence() -> None:
    assert resolve_model("MoonshineForConditionalGeneration") is MOONSHINE
    sequence = tuple(MOONSHINE.layer_sequence())
    assert sequence[0] == "moonshine_embedding"
    assert "moonshine_self_attention" in sequence
    assert "moonshine_cross_attention" in sequence
    assert "moonshine_decoder_mlp" in sequence
    assert sequence[-2:] == ("moonshine_lm_head", "moonshine_argmax")
