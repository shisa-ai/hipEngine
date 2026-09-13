"""Model-contract tests for deepgrove/maple-preview-2bit-mlx."""

from __future__ import annotations

import copy

import pytest

from hipengine.models import MAPLE, MAPLE_LAYER_PATTERN, resolve_model
from hipengine.models.maple import parse_maple_model_spec


def maple_config() -> dict:
    quantization = {
        "bits": 2,
        "group_size": 128,
        "mode": "affine",
        "lm_head": {"bits": 4, "group_size": 64},
        "model.word_embeddings": {"bits": 4, "group_size": 64},
    }
    return {
        "architectures": ["MapleForCausalLM"],
        "model_type": "maple",
        "dtype": "bfloat16",
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "first_k_dense_replace": 0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "layer_types": list(MAPLE_LAYER_PATTERN),
        "max_position_embeddings": 128000,
        "moe_intermediate_size": 512,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 4,
        "num_shared_experts": 0,
        "partial_rotary_factor": 0.5,
        "quantization": quantization,
        "quantization_config": copy.deepcopy(quantization),
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000,
        "router_dtype": "fp32",
        "sliding_window": 512,
        "tie_word_embeddings": False,
        "use_bias": False,
        "use_qk_norm": True,
        "vocab_size": 151936,
    }


def test_maple_model_plugin_resolves_and_declares_both_attention_paths() -> None:
    assert resolve_model("MapleForCausalLM") is MAPLE
    assert resolve_model("maple") is MAPLE
    assert MAPLE.default_quant == "maple_ternary2"

    sliding = MAPLE.decode_layer_sequence(attention_kind="sliding_attention")
    full = MAPLE.decode_layer_sequence(attention_kind="full_attention")
    assert "qknorm_partial_rope" in sliding
    assert "swa_attention_decode" in sliding
    assert "qknorm" in full
    assert "full_attention_decode" in full
    assert "clamped_swiglu" in sliding
    assert "clamped_swiglu" in full
    with pytest.raises(ValueError, match="attention_kind"):
        MAPLE.decode_layer_sequence(attention_kind="linear_attention")


def test_parse_maple_model_spec_pins_official_geometry_and_storage() -> None:
    spec = parse_maple_model_spec(maple_config())
    assert spec.architecture == "MapleForCausalLM"
    assert spec.hidden_size == spec.q_size == 2048
    assert spec.kv_size == 512
    assert spec.rotary_dim == 64
    assert spec.num_hidden_layers == 24
    assert spec.num_experts == 256
    assert spec.num_experts_per_tok == 8
    assert spec.ternary_bits == 2
    assert spec.ternary_group_size == 128
    assert (spec.embedding_bits, spec.embedding_group_size) == (4, 64)
    assert (spec.lm_head_bits, spec.lm_head_group_size) == (4, 64)
    assert sum(spec.uses_rope(layer) for layer in range(spec.num_hidden_layers)) == 18
    assert spec.attention_kind(3) == "full_attention"
    with pytest.raises(IndexError, match="outside"):
        spec.attention_kind(24)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("hidden_size", 4096, "hidden_size"),
        ("num_experts_per_tok", 4, "num_experts_per_tok"),
        ("partial_rotary_factor", 1.0, "partial_rotary_factor"),
        ("router_dtype", "bf16", "router_dtype"),
        ("eos_token_id", 151936, "eos_token_id"),
    ),
)
def test_parse_maple_model_spec_rejects_architecture_drift(
    field: str, value: object, match: str
) -> None:
    changed = maple_config()
    changed[field] = value
    with pytest.raises(ValueError, match=match):
        parse_maple_model_spec(changed)


def test_parse_maple_model_spec_rejects_attention_and_quant_drift() -> None:
    changed = maple_config()
    changed["layer_types"][3] = "sliding_attention"
    with pytest.raises(ValueError, match="3:1"):
        parse_maple_model_spec(changed)

    changed = maple_config()
    changed["quantization"]["lm_head"]["bits"] = 3
    changed["quantization_config"] = copy.deepcopy(changed["quantization"])
    with pytest.raises(ValueError, match="lm_head"):
        parse_maple_model_spec(changed)

    changed = maple_config()
    changed["quantization_config"]["bits"] = 4
    with pytest.raises(ValueError, match="must equal"):
        parse_maple_model_spec(changed)
