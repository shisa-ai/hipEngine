"""Gemma 4 GGUF metadata and tensor-contract loading."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.gemma4_gguf import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    build_gemma4_gguf_tensor_map,
    gemma4_gguf_config_from_metadata,
    gemma4_rotated_pair_count,
    required_gemma4_gguf_tensor_names,
    validate_gemma4_gguf_tensor_map,
)
from hipengine.loading.gguf import scan_gguf
from hipengine.models import resolve_model
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gemma4_gguf_fixture import (
    FIXTURE_EXPERTS,
    FIXTURE_EXPERT_FF,
    FIXTURE_EXPERT_USED,
    FIXTURE_FFN,
    FIXTURE_GLOBAL_ROTATED_PAIRS,
    FIXTURE_HEAD_DIM_GLOBAL,
    FIXTURE_HEAD_DIM_SWA,
    FIXTURE_HEADS,
    FIXTURE_HIDDEN,
    FIXTURE_KV_HEADS_GLOBAL,
    FIXTURE_KV_HEADS_SWA,
    FIXTURE_LAYER_COUNT,
    FIXTURE_SLIDING_WINDOW,
    FIXTURE_VOCAB,
    default_fixture_tensors,
    fixture_metadata,
    rope_freqs_values,
    write_default_gemma4_gguf,
    write_fixture_gguf,
)


@pytest.fixture()
def fixture_info(tmp_path):
    return scan_gguf(write_default_gemma4_gguf(tmp_path / "gemma4.gguf"))


def test_config_decodes_dual_attention_geometry(fixture_info) -> None:
    config = gemma4_gguf_config_from_metadata(fixture_info)

    assert config.block_count == FIXTURE_LAYER_COUNT
    assert config.hidden_size == FIXTURE_HIDDEN
    assert config.vocab_size == FIXTURE_VOCAB
    assert config.layer_types == (SLIDING_ATTENTION, FULL_ATTENTION)

    # Sliding geometry.
    assert config.head_count(0) == FIXTURE_HEADS
    assert config.head_count_kv_for(0) == FIXTURE_KV_HEADS_SWA
    assert config.head_dim(0) == FIXTURE_HEAD_DIM_SWA
    assert config.kv_width(0) == FIXTURE_KV_HEADS_SWA * FIXTURE_HEAD_DIM_SWA
    assert config.attention_k_eq_v(0) is False

    # Global geometry in the same model.
    assert config.head_count(1) == FIXTURE_HEADS
    assert config.head_count_kv_for(1) == FIXTURE_KV_HEADS_GLOBAL
    assert config.head_dim(1) == FIXTURE_HEAD_DIM_GLOBAL
    assert config.kv_width(1) == FIXTURE_KV_HEADS_GLOBAL * FIXTURE_HEAD_DIM_GLOBAL
    assert config.attention_k_eq_v(1) is True

    assert config.sliding_window == FIXTURE_SLIDING_WINDOW
    assert config.feed_forward_length(0) == FIXTURE_FFN
    assert config.expert_count == FIXTURE_EXPERTS
    assert config.expert_used_count == FIXTURE_EXPERT_USED
    assert config.expert_feed_forward_length == FIXTURE_EXPERT_FF
    assert config.final_logit_softcapping == pytest.approx(30.0)
    assert config.embed_scale == pytest.approx(FIXTURE_HIDDEN**0.5)
    assert config.tied_embeddings is True


def test_rotated_span_comes_from_rope_freqs_not_metadata(fixture_info) -> None:
    """``rope.dimension_count`` is the full head width, not the rotated width."""

    config = gemma4_gguf_config_from_metadata(fixture_info)
    metadata = fixture_info.metadata
    assert metadata["gemma4.rope.dimension_count"] == FIXTURE_HEAD_DIM_GLOBAL

    assert config.full_rope.rotated_pairs == FIXTURE_GLOBAL_ROTATED_PAIRS
    assert config.full_rope.rotated_pairs < FIXTURE_HEAD_DIM_GLOBAL // 2
    assert config.full_rope.is_partial is True
    assert config.full_rope.rope_type == "proportional"

    # Sliding layers rotate the whole head.
    assert config.swa_rope.rotated_pairs == FIXTURE_HEAD_DIM_SWA // 2
    assert config.swa_rope.is_partial is False
    assert config.swa_rope.freq_base == pytest.approx(10_000.0)
    assert config.full_rope.freq_base == pytest.approx(1_000_000.0)


def test_missing_rope_freqs_falls_back_to_full_rotation(tmp_path) -> None:
    tensors = [
        item for item in default_fixture_tensors() if item[0] != "rope_freqs.weight"
    ]
    info = scan_gguf(
        write_fixture_gguf(tmp_path / "no_freqs.gguf", tensors, fixture_metadata())
    )
    config = gemma4_gguf_config_from_metadata(info)
    assert config.full_rope.rotated_pairs == FIXTURE_HEAD_DIM_GLOBAL // 2
    assert config.has_global_rope_freqs is False
    assert "rope_freqs.weight" not in required_gemma4_gguf_tensor_names(config)


def test_rotated_pair_count_rejects_a_non_prefix_layout() -> None:
    half = 8
    interleaved = np.full(half, 1.0, dtype=np.float32)
    interleaved[::2] = np.float32(1.0e30)
    with pytest.raises(ValueError, match="rotated prefix"):
        gemma4_rotated_pair_count(interleaved, head_dim=half * 2)


def test_rotated_pair_count_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="head_dim // 2"):
        gemma4_rotated_pair_count(np.ones(4, dtype=np.float32), head_dim=32)


def test_rotated_pair_count_accepts_a_fully_rotated_head() -> None:
    values = np.ones(16, dtype=np.float32)
    assert gemma4_rotated_pair_count(values, head_dim=32) == 16


def test_fixture_tensor_contract_passes(fixture_info) -> None:
    validation = validate_gemma4_gguf_tensor_map(fixture_info)
    assert validation.passed, (
        f"missing={validation.missing} unexpected={validation.unexpected} "
        f"shape={validation.shape_errors} type={validation.type_errors}"
    )


def test_global_layers_have_no_v_proj(fixture_info) -> None:
    model_map = build_gemma4_gguf_tensor_map(fixture_info)
    assert model_map.layer(0).has("attn_v") is True
    assert model_map.layer(1).has("attn_v") is False
    assert model_map.layer(1).k_eq_v is True
    # The missing v_proj is not reported as a missing tensor.
    assert not any("attn_v" in name for name in model_map.validation.missing)


def test_missing_v_proj_on_a_sliding_layer_is_reported(tmp_path) -> None:
    tensors = [
        item for item in default_fixture_tensors() if item[0] != "blk.0.attn_v.weight"
    ]
    info = scan_gguf(write_fixture_gguf(tmp_path / "no_v.gguf", tensors, fixture_metadata()))
    validation = validate_gemma4_gguf_tensor_map(info)
    assert validation.passed is False
    assert "blk.0.attn_v.weight" in validation.missing


def test_v_proj_on_a_global_layer_is_unexpected(tmp_path) -> None:
    """A global layer that ships ``v_proj`` contradicts ``attention_k_eq_v``."""

    tensors = list(default_fixture_tensors())
    tensors.append(
        (
            "blk.1.attn_v.weight",
            (FIXTURE_KV_HEADS_GLOBAL * FIXTURE_HEAD_DIM_GLOBAL, FIXTURE_HIDDEN),
            GGMLQuantizationType.Q8_0,
        )
    )
    info = scan_gguf(write_fixture_gguf(tmp_path / "extra_v.gguf", tensors, fixture_metadata()))
    validation = validate_gemma4_gguf_tensor_map(info)
    assert validation.passed is False
    assert "blk.1.attn_v.weight" in validation.unexpected


def test_shape_error_is_reported(tmp_path) -> None:
    tensors = [
        (
            name,
            (FIXTURE_EXPERTS, FIXTURE_EXPERT_FF, FIXTURE_HIDDEN)
            if name == "blk.0.ffn_gate_up_exps.weight"
            else shape,
            qtype,
        )
        for name, shape, qtype in default_fixture_tensors()
    ]
    info = scan_gguf(write_fixture_gguf(tmp_path / "bad_shape.gguf", tensors, fixture_metadata()))
    validation = validate_gemma4_gguf_tensor_map(info)
    assert validation.passed is False
    assert any("ffn_gate_up_exps" in error for error in validation.shape_errors)


def test_quantized_scale_tensor_is_reported(tmp_path) -> None:
    """The router scale is read as fp32; quantized storage must be rejected."""

    tensors = [
        (
            name,
            shape,
            GGMLQuantizationType.Q8_0 if name == "blk.0.ffn_gate_inp.scale" else qtype,
        )
        for name, shape, qtype in default_fixture_tensors()
    ]
    info = scan_gguf(write_fixture_gguf(tmp_path / "quant_scale.gguf", tensors, fixture_metadata()))
    validation = validate_gemma4_gguf_tensor_map(info)
    assert validation.passed is False
    assert any("ffn_gate_inp.scale" in error for error in validation.type_errors)


def test_shared_kv_layers_are_refused(tmp_path) -> None:
    metadata = [
        (key, value_type, 1 if key == "gemma4.attention.shared_kv_layers" else value)
        for key, value_type, value in fixture_metadata()
    ]
    info = scan_gguf(
        write_fixture_gguf(tmp_path / "shared.gguf", default_fixture_tensors(), metadata)
    )
    with pytest.raises(ValueError, match="KV sharing is not implemented"):
        gemma4_gguf_config_from_metadata(info)


def test_sliding_pattern_accepts_a_stride_encoding(tmp_path) -> None:
    """Older converters wrote the pattern length rather than a boolean array."""

    layer_types = (SLIDING_ATTENTION,) * 3
    metadata = [
        (key, value_type, value)
        for key, value_type, value in fixture_metadata(layer_types=layer_types)
        if key != "gemma4.attention.sliding_window_pattern"
    ]
    metadata.append(("gemma4.attention.sliding_window_pattern", 4, 3))
    info = scan_gguf(
        write_fixture_gguf(
            tmp_path / "stride.gguf",
            default_fixture_tensors(layer_types=layer_types),
            metadata,
        )
    )
    config = gemma4_gguf_config_from_metadata(info)
    # Every third layer is global, and the last layer is already global.
    assert config.layer_types == (SLIDING_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION)


def test_forced_global_last_layer_overrides_the_pattern(tmp_path) -> None:
    """The reference implementation forces the last layer to full attention."""

    layer_types = (SLIDING_ATTENTION,) * 2
    metadata = [
        (key, value_type, value)
        for key, value_type, value in fixture_metadata(layer_types=layer_types)
        if key != "gemma4.attention.sliding_window_pattern"
    ]
    metadata.append(("gemma4.attention.sliding_window_pattern", 4, 7))
    info = scan_gguf(
        write_fixture_gguf(
            tmp_path / "forced.gguf",
            default_fixture_tensors(layer_types=layer_types),
            metadata,
        )
    )
    config = gemma4_gguf_config_from_metadata(info)
    assert config.layer_types == (SLIDING_ATTENTION, FULL_ATTENTION)


def test_wrong_architecture_is_refused(tmp_path) -> None:
    metadata = [
        (key, value_type, "laguna" if key == "general.architecture" else value)
        for key, value_type, value in fixture_metadata()
    ]
    info = scan_gguf(
        write_fixture_gguf(tmp_path / "wrong_arch.gguf", default_fixture_tensors(), metadata)
    )
    with pytest.raises(ValueError, match="expected GGUF architecture 'gemma4'"):
        gemma4_gguf_config_from_metadata(info)


def test_rope_freqs_payload_round_trips_through_the_scanner(fixture_info) -> None:
    from hipengine.loading.gemma4_gguf import read_gemma4_rope_freqs

    values = read_gemma4_rope_freqs(fixture_info)
    expected = rope_freqs_values()
    assert values is not None
    assert np.array_equal(values, expected)


def test_model_plugin_registers_the_gemma4_architectures() -> None:
    for architecture in ("gemma4", "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
        plugin = resolve_model(architecture)
        assert plugin.name == "gemma4_gguf"
        assert plugin.default_quant == "gguf_q4_k_m"


def test_model_plugin_layer_sequence_includes_both_branches() -> None:
    plugin = resolve_model("gemma4")
    sliding = plugin.decode_layer_sequence(attention_kind=SLIDING_ATTENTION)
    global_layer = plugin.decode_layer_sequence(attention_kind=FULL_ATTENTION)

    # The global plan has no separate v projection.
    assert "full_attention_qk_proj" in global_layer
    assert "sliding_attention_qkv_proj" in sliding

    # Both feed-forward branches appear in every layer plan.
    for plan in (sliding, global_layer):
        assert "dense_mlp" in plan
        assert "selected_expert_mlp" in plan
        assert "gemma4_parallel_ffn_combine" in plan
        assert plan.index("dense_mlp") < plan.index("gemma4_parallel_ffn_combine")
        assert plan.index("selected_expert_mlp") < plan.index("gemma4_parallel_ffn_combine")
