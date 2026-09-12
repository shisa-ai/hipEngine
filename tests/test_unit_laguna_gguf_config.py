from __future__ import annotations

import pytest

from hipengine.loading.laguna_gguf import (
    DENSE_MLP,
    FULL_ATTENTION,
    SPARSE_MOE,
    SLIDING_ATTENTION,
    laguna_gguf_config_from_metadata,
)
from hipengine.models import resolve_model
from tests._laguna_synthetic import laguna_metadata, make_laguna_info


def test_laguna_model_plugin_resolves_gguf_and_hf_architectures() -> None:
    gguf = resolve_model("laguna")
    hf = resolve_model("LagunaForCausalLM")

    assert gguf is hf
    assert gguf.name == "laguna_gguf"
    assert gguf.default_quant == "gguf_q4_k_m"
    assert gguf.default_backend == "auto"

    full_dense = gguf.decode_layer_sequence(
        attention_kind=FULL_ATTENTION,
        mlp_kind=DENSE_MLP,
    )
    sliding_moe = gguf.decode_layer_sequence(
        attention_kind=SLIDING_ATTENTION,
        mlp_kind=SPARSE_MOE,
    )
    assert full_dense[:3] == ("rmsnorm", "full_attention_qkv_proj", "yarn_rope")
    assert full_dense.index("softplus_head_gate") < full_dense.index("attention_o_proj")
    assert full_dense[-2:] == ("dense_mlp", "residual_add")
    assert "sliding_attention_decode" in sliding_moe
    assert "laguna_sigmoid_router_topk" in sliding_moe
    assert "laguna_shared_expert" in sliding_moe
    assert "blk.{layer}.ffn_norm.weight" in gguf.weight_name_templates
    assert "blk.{layer}.exp_probs_b.bias" in gguf.weight_name_templates


def test_laguna_gguf_config_decodes_production_metadata() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())

    assert config.architecture == "laguna"
    assert config.block_count == 48
    assert config.hidden_size == 3_072
    assert config.vocab_size == 100_352
    assert config.feed_forward_length == 12_288
    assert config.context_length == 262_144
    assert config.head_counts == tuple([48, 72, 72, 72] * 12)
    assert config.head_count_kv == 8
    assert config.key_length == 128
    assert config.value_length == 128
    assert config.layer_types == tuple([FULL_ATTENTION, SLIDING_ATTENTION, SLIDING_ATTENTION, SLIDING_ATTENTION] * 12)
    assert config.mlp_layer_types == (DENSE_MLP,) + (SPARSE_MOE,) * 47
    assert config.sliding_window == 512
    assert config.sliding_window_pattern == 4
    assert config.expert_count == 256
    assert config.expert_used_count == 10
    assert config.expert_weights_norm is True
    assert config.expert_weights_scale == 2.5
    assert config.expert_gating_func == "sigmoid"
    assert config.full_rope.rope_type == "yarn"
    assert config.full_rope.dimension_count == 64
    assert config.full_rope.freq_base == 500_000.0
    assert config.full_rope.scaling_factor == 32.0
    assert config.full_rope.original_context_length == 8_192
    assert config.full_rope.yarn_beta_fast == 32.0
    assert config.full_rope.yarn_beta_slow == 1.0
    assert config.swa_rope.rope_type == "default"
    assert config.swa_rope.dimension_count == 128
    assert config.swa_rope.freq_base == 10_000.0
    assert config.rope_for_layer(0) is config.full_rope
    assert config.rope_for_layer(1) is config.swa_rope
    assert config.head_count(0) == 48
    assert config.head_count(1) == 72
    assert config.mlp_type(0) == DENSE_MLP
    assert config.mlp_type(1) == SPARSE_MOE


def test_laguna_gguf_config_accepts_scalar_heads_for_all_full_family_member() -> None:
    metadata = laguna_metadata()
    metadata["laguna.block_count"] = 3
    metadata["laguna.attention.head_count"] = 64
    metadata["laguna.attention.sliding_window"] = 0
    metadata["laguna.leading_dense_block_count"] = 2

    config = laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))

    assert config.head_counts == (64, 64, 64)
    assert config.layer_types == (FULL_ATTENTION,) * 3
    assert config.mlp_layer_types == (DENSE_MLP, DENSE_MLP, SPARSE_MOE)
    assert config.swa_rope is None


def test_laguna_gguf_config_rejects_wrong_head_array_length() -> None:
    metadata = laguna_metadata()
    metadata["laguna.attention.head_count"] = [48, 72]

    with pytest.raises(ValueError, match="head_count.*48"):
        laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))


def test_laguna_gguf_config_rejects_heads_not_divisible_by_kv_heads() -> None:
    metadata = laguna_metadata()
    metadata["laguna.attention.head_count"] = [48, 70, 70, 70] * 12

    with pytest.raises(ValueError, match="divisible by head_count_kv"):
        laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))


def test_laguna_gguf_config_requires_swa_rope_metadata() -> None:
    metadata = laguna_metadata()
    del metadata["laguna.rope.dimension_count_swa"]

    with pytest.raises(KeyError, match="laguna.rope.dimension_count_swa"):
        laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))


def test_laguna_gguf_config_rejects_non_sigmoid_router() -> None:
    metadata = laguna_metadata()
    metadata["laguna.expert_gating_func"] = 1

    with pytest.raises(ValueError, match="sigmoid"):
        laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))


def test_laguna_gguf_config_rejects_wrong_architecture() -> None:
    metadata = laguna_metadata()
    metadata["general.architecture"] = "qwen35moe"

    with pytest.raises(ValueError, match="expected GGUF architecture 'laguna'"):
        laguna_gguf_config_from_metadata(make_laguna_info(metadata=metadata))
