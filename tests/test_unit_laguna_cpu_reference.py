from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.laguna import (
    LagunaAttentionConfig,
    LagunaAttentionWeights,
    LagunaDenseFFNWeights,
    LagunaLayerWeights,
    LagunaReferenceLayer,
    LagunaRopeConfig,
    LagunaSparseFFNWeights,
    laguna_apply_rope,
    laguna_block_streamed_gqa_attention,
    laguna_causal_mask,
    laguna_head_rmsnorm,
    laguna_model_forward,
    laguna_rope_tables,
    laguna_sigmoid_correction_topk,
    laguna_softplus_head_gate,
    register_laguna_cpu_reference_kernels,
)
from hipengine.kernels.registry import resolve

FIXTURE = Path(__file__).parent / "fixtures" / "laguna_cpu_reference.json"


def _dense_gqa_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    query_positions: np.ndarray,
    key_positions: np.ndarray,
    *,
    sliding_window: int | None = None,
) -> np.ndarray:
    rows, heads, head_dim = query.shape
    kv_heads = key.shape[1]
    group_size = heads // kv_heads
    visible = laguna_causal_mask(
        query_positions,
        key_positions,
        sliding_window=sliding_window,
    )
    output = np.empty_like(query, dtype=np.float32)
    scale = np.float32(head_dim**-0.5)
    for row in range(rows):
        selected = np.flatnonzero(visible[row])
        for head in range(heads):
            logits = (
                key[selected, head // group_size] @ query[row, head] * scale
            ).astype(np.float32)
            logits -= np.max(logits)
            probabilities = np.exp(logits).astype(np.float32)
            probabilities /= probabilities.sum(dtype=np.float32)
            output[row, head] = probabilities @ value[
                selected,
                head // group_size,
            ]
    return output


def test_laguna_block_streaming_oracle_matches_dense_at_boundaries_and_tails() -> None:
    rng = np.random.default_rng(3803)
    query_positions = np.arange(509, 514, dtype=np.int64)
    key_positions = np.arange(514, dtype=np.int64)
    query = rng.normal(0.0, 0.2, size=(5, 6, 8)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(514, 1, 8)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(514, 1, 8)).astype(np.float32)

    for sliding_window in (None, 512):
        expected = _dense_gqa_attention(
            query,
            key,
            value,
            query_positions,
            key_positions,
            sliding_window=sliding_window,
        )
        actual = laguna_block_streamed_gqa_attention(
            query,
            key,
            value,
            query_positions=query_positions,
            key_positions=key_positions,
            query_block_size=3,
            key_block_size=127,
            sliding_window=sliding_window,
        )
        np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_laguna_block_streaming_oracle_handles_final_128k_position() -> None:
    rng = np.random.default_rng(131_072)
    query_positions = np.asarray([131_071], dtype=np.int64)
    key_positions = np.arange(131_072, dtype=np.int64)
    query = rng.normal(0.0, 0.2, size=(1, 6, 4)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(131_072, 1, 4)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(131_072, 1, 4)).astype(np.float32)

    expected = _dense_gqa_attention(
        query,
        key,
        value,
        query_positions,
        key_positions,
    )
    actual = laguna_block_streamed_gqa_attention(
        query,
        key,
        value,
        query_positions=query_positions,
        key_positions=key_positions,
        query_block_size=16,
        key_block_size=127,
    )

    np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-6)
    assert np.isfinite(actual).all()


def test_laguna_softplus_head_gate_broadcasts_one_gate_per_head() -> None:
    attention = np.arange(12, dtype=np.float32).reshape(2, 2, 3) - 3.0
    gate_logits = np.array([[0.0, np.log(3.0)], [-100.0, 100.0]], dtype=np.float32)

    actual = laguna_softplus_head_gate(attention, gate_logits)
    expected_scale = np.logaddexp(np.float32(0.0), gate_logits).astype(np.float32)
    expected = attention * expected_scale[..., None]

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert expected_scale[0, 0] == pytest.approx(np.log(2.0), rel=1.0e-6)
    assert expected_scale[0, 1] == pytest.approx(np.log(4.0), rel=1.0e-6)


def test_laguna_softplus_head_gate_rejects_wrong_gate_shape() -> None:
    attention = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="gate_logits.*leading dimensions"):
        laguna_softplus_head_gate(attention, np.ones((2, 4), dtype=np.float32))


def test_laguna_router_bias_changes_selection_not_route_probability() -> None:
    hidden = np.array([[1.0, 0.0]], dtype=np.float32)
    router = np.array(
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    correction = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    result = laguna_sigmoid_correction_topk(
        hidden,
        router,
        correction,
        experts_used=2,
        routed_scaling_factor=2.5,
    )

    np.testing.assert_array_equal(result.selected_experts, np.array([[2, 0]]))
    unbiased = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0], dtype=np.float32)))
    normalized = unbiased / unbiased.sum()
    np.testing.assert_allclose(result.routing_weights[0], normalized, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        result.scaled_routing_weights[0],
        normalized * np.float32(2.5),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.routing_scores[0, 2] == pytest.approx(0.5)
    assert result.selection_scores[0, 2] == pytest.approx(1.5)


def test_laguna_router_uses_stable_lower_expert_id_ties() -> None:
    result = laguna_sigmoid_correction_topk(
        np.ones((1, 2), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        experts_used=3,
        routed_scaling_factor=1.0,
    )

    np.testing.assert_array_equal(result.selected_experts, np.array([[0, 1, 2]]))
    np.testing.assert_allclose(result.routing_weights, np.full((1, 3), 1.0 / 3.0))


def test_laguna_router_softcap_and_extreme_logits_remain_finite() -> None:
    result = laguna_sigmoid_correction_topk(
        np.array([[1000.0, -1000.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        experts_used=2,
        routed_scaling_factor=2.5,
        router_logit_softcapping=5.0,
    )

    assert np.isfinite(result.router_logits).all()
    assert np.isfinite(result.routing_scores).all()
    assert np.isfinite(result.routing_weights).all()
    assert np.max(np.abs(result.router_logits)) <= 5.0
    np.testing.assert_allclose(result.routing_weights.sum(axis=-1), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(result.scaled_routing_weights.sum(axis=-1), 2.5, atol=1.0e-6)


def test_laguna_router_rejects_bad_shapes_and_topk() -> None:
    hidden = np.ones((2, 3), dtype=np.float32)
    router = np.ones((4, 3), dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="correction_bias"):
        laguna_sigmoid_correction_topk(hidden, router, np.zeros(3), experts_used=2)
    with pytest.raises(ValueError, match="experts_used"):
        laguna_sigmoid_correction_topk(hidden, router, bias, experts_used=5)


def test_laguna_cpu_reference_primitives_register_under_architecture_keys() -> None:
    register_laguna_cpu_reference_kernels(replace=True)

    assert (
        resolve(
            backend="cpu_reference",
            layer="softplus_head_gate",
            quant="fp32",
            variant="laguna_per_head",
        )
        is laguna_softplus_head_gate
    )
    assert (
        resolve(
            backend="cpu_reference",
            layer="laguna_sigmoid_router_topk",
            quant="gguf_f32",
            variant="correction_bias",
        )
        is laguna_sigmoid_correction_topk
    )
    assert (
        resolve(
            backend="cpu_reference",
            layer="laguna_rope_tables",
            quant="fp32",
            variant="yarn_or_default",
        )
        is laguna_rope_tables
    )
    assert (
        resolve(
            backend="cpu_reference",
            layer="laguna_model",
            quant="fp32",
            variant="two_layer_reference",
        )
        is laguna_model_forward
    )


def test_laguna_production_yarn_and_swa_rope_match_transformers_fixture() -> None:
    fixture = _load_fixture()
    positions = np.asarray(fixture["production_rope"]["positions"], dtype=np.int64)
    full = fixture["production_rope"]["full_attention"]
    swa = fixture["production_rope"]["sliding_attention"]

    full_cos, full_sin = laguna_rope_tables(
        positions,
        LagunaRopeConfig(
            rope_type="yarn",
            rotary_dim=64,
            freq_base=500_000.0,
            scaling_factor=32.0,
            original_context_length=8_192,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
        ),
    )
    swa_cos, swa_sin = laguna_rope_tables(
        positions,
        LagunaRopeConfig(
            rope_type="default",
            rotary_dim=128,
            freq_base=10_000.0,
        ),
    )

    tolerance = fixture["tolerances"]
    np.testing.assert_allclose(
        full_cos,
        _array(full["cos"]),
        atol=tolerance["rope_atol"],
        rtol=tolerance["rope_rtol"],
    )
    np.testing.assert_allclose(
        full_sin,
        _array(full["sin"]),
        atol=tolerance["rope_atol"],
        rtol=tolerance["rope_rtol"],
    )
    np.testing.assert_allclose(
        swa_cos,
        _array(swa["cos"]),
        atol=tolerance["rope_atol"],
        rtol=tolerance["rope_rtol"],
    )
    np.testing.assert_allclose(
        swa_sin,
        _array(swa["sin"]),
        atol=tolerance["rope_atol"],
        rtol=tolerance["rope_rtol"],
    )
    assert full_cos.shape == (6, 64)
    assert swa_cos.shape == (6, 128)


def test_laguna_partial_rope_rotates_only_the_first_64_channels() -> None:
    positions = np.asarray([8_192], dtype=np.int64)
    cos, sin = laguna_rope_tables(
        positions,
        LagunaRopeConfig(
            rope_type="yarn",
            rotary_dim=64,
            freq_base=500_000.0,
            scaling_factor=32.0,
            original_context_length=8_192,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
        ),
    )
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(1, 1, 128)

    actual = laguna_apply_rope(values, cos[:, None, :], sin[:, None, :], rotary_dim=64)

    np.testing.assert_array_equal(actual[..., 64:], values[..., 64:])
    assert not np.array_equal(actual[..., :64], values[..., :64])


def test_laguna_global_and_swa_masks_match_transformers_at_511_512_513() -> None:
    fixture = _load_fixture()
    for length in (511, 512, 513):
        positions = np.arange(length, dtype=np.int64)
        expected = fixture["mask_boundaries"][str(length)]
        for layer_type, window in (("full_attention", None), ("sliding_attention", 512)):
            actual = laguna_causal_mask(
                np.asarray([length - 1], dtype=np.int64),
                positions,
                sliding_window=window,
            )[0]
            visible = np.flatnonzero(actual).tolist()
            assert visible == expected[layer_type]["visible_key_positions"]


def test_laguna_swa_mask_uses_absolute_positions_after_physical_wrap() -> None:
    fixture = _load_fixture()["absolute_position_ring"]
    physical_positions = np.asarray(fixture["physical_slot_token_positions"], dtype=np.int64)

    mask = laguna_causal_mask(
        np.asarray([fixture["query_position"]], dtype=np.int64),
        physical_positions,
        sliding_window=fixture["sliding_window"],
    )

    assert np.flatnonzero(mask[0]).tolist() == fixture["expected_visible_physical_slots"]
    assert physical_positions[0] == 512
    assert physical_positions[1] == 513
    assert physical_positions[-1] == 511


def test_laguna_head_rmsnorm_matches_transformers_layer0_fixture() -> None:
    fixture = _load_fixture()
    parameters = _tiny_parameters(fixture)
    hidden = parameters["model.embed_tokens.weight"][[1, 3, 5]]
    input_norm = laguna_head_rmsnorm(
        hidden,
        parameters["model.layers.0.input_layernorm.weight"],
        eps=1.0e-6,
    )
    query = np.matmul(
        input_norm,
        parameters["model.layers.0.self_attn.q_proj.weight"].T,
    ).reshape(3, 48, 4)

    actual = laguna_head_rmsnorm(
        query,
        parameters["model.layers.0.self_attn.q_norm.weight"],
        eps=1.0e-6,
    )
    expected = _capture(fixture, "layer0.q_head_norm")[0]

    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_laguna_two_layer_reference_matches_transformers_intermediates_and_logits() -> None:
    fixture = _load_fixture()
    parameters = _tiny_parameters(fixture)
    layers = _tiny_layers(parameters)
    input_ids = np.asarray(fixture["tiny_model"]["input_ids"], dtype=np.int64)[0]
    positions = np.asarray(fixture["tiny_model"]["position_ids"], dtype=np.int64)[0]

    result = laguna_model_forward(
        input_ids,
        parameters["model.embed_tokens.weight"],
        layers,
        parameters["model.norm.weight"],
        parameters["lm_head.weight"],
        positions=positions,
        eps=1.0e-6,
    )

    atol = fixture["tolerances"]["atol"]
    rtol = fixture["tolerances"]["rtol"]
    layer0 = result.layers[0]
    layer1 = result.layers[1]
    assert layer1.sparse_moe is not None

    comparisons = (
        (layer0.attention.normalized, "layer0.attention_norm"),
        (layer0.attention.query_normalized, "layer0.q_head_norm"),
        (layer0.attention.key_normalized, "layer0.k_head_norm"),
        (layer0.attention.gate_logits, "layer0.gate_logits"),
        (layer0.attention.output, "layer0.attention_output"),
        (layer0.ffn_normalized, "layer0.ffn_norm"),
        (layer0.ffn_output, "layer0.ffn_output"),
        (layer0.hidden, "layer0.output"),
        (layer1.attention.normalized, "layer1.attention_norm"),
        (layer1.attention.query_normalized, "layer1.q_head_norm"),
        (layer1.attention.key_normalized, "layer1.k_head_norm"),
        (layer1.attention.gate_logits, "layer1.gate_logits"),
        (layer1.attention.output, "layer1.attention_output"),
        (layer1.ffn_normalized, "layer1.ffn_norm"),
        (layer1.sparse_moe.routed_output_unscaled, "layer1.routed_output_unscaled"),
        (layer1.sparse_moe.shared_output, "layer1.shared_output"),
        (layer1.ffn_output, "layer1.ffn_output"),
        (layer1.hidden, "layer1.output"),
        (result.final_hidden, "model.final_norm"),
        (result.logits, "model.logits"),
    )
    for actual, capture_name in comparisons:
        expected = _capture(fixture, capture_name)
        if expected.ndim == actual.ndim + 1 and expected.shape[0] == 1:
            expected = expected[0]
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            err_msg=capture_name,
        )

    router = _capture_list(fixture, "layer1.router")
    np.testing.assert_allclose(
        layer1.sparse_moe.routing.router_logits,
        router[0],
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        layer1.sparse_moe.routing.routing_weights,
        router[1],
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_array_equal(
        layer1.sparse_moe.routing.selected_experts,
        router[2].astype(np.int64),
    )
    np.testing.assert_allclose(
        layer1.sparse_moe.routing.scaled_routing_weights.sum(axis=-1),
        np.full(3, 2.5, dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        layer1.sparse_moe.output,
        layer1.sparse_moe.routed_output + layer1.sparse_moe.shared_output,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        result.hidden_states[0],
        parameters["model.embed_tokens.weight"][input_ids],
        atol=0.0,
        rtol=0.0,
    )


def test_laguna_model_reference_requires_explicit_untied_output_weight() -> None:
    fixture = _load_fixture()
    parameters = _tiny_parameters(fixture)

    with pytest.raises(ValueError, match="untied output_weight"):
        laguna_model_forward(
            np.asarray([1], dtype=np.int64),
            parameters["model.embed_tokens.weight"],
            _tiny_layers(parameters),
            parameters["model.norm.weight"],
            None,
            positions=np.asarray([0], dtype=np.int64),
        )


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _array(payload: dict) -> np.ndarray:
    return np.asarray(payload["data"], dtype=np.dtype(payload["dtype"]))


def _capture(fixture: dict, name: str) -> np.ndarray:
    return _array(fixture["tiny_model"]["captures"][name])


def _capture_list(fixture: dict, name: str) -> list[np.ndarray]:
    return [_array(item) for item in fixture["tiny_model"]["captures"][name]]


def _fixture_parameter(name: str, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    index = np.arange(1, count + 1, dtype=np.float64)
    phase = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") % 10_007
    if name.endswith("norm.weight") or name == "model.norm.weight":
        values = 0.9 + 0.08 * np.sin(index * 0.071 + phase * 0.003)
    elif name.endswith("e_score_correction_bias"):
        values = 0.01 * np.sin(index * 0.13 + phase * 0.001)
        for rank, expert_id in enumerate((3, 17, 29, 47, 71, 101, 139, 173, 211, 251)):
            values[expert_id] = 0.9 - rank * 0.035
    else:
        values = 0.055 * np.sin(index * ((phase % 19) + 1) * 0.017) + 0.025 * np.cos(
            (index + phase) * 0.013
        )
    return np.ascontiguousarray(values.reshape(shape).astype(np.float32))


def _tiny_parameters(fixture: dict) -> dict[str, np.ndarray]:
    shapes = {
        "model.embed_tokens.weight": (7, 4),
        "model.layers.0.self_attn.q_proj.weight": (192, 4),
        "model.layers.0.self_attn.k_proj.weight": (32, 4),
        "model.layers.0.self_attn.v_proj.weight": (32, 4),
        "model.layers.0.self_attn.o_proj.weight": (4, 192),
        "model.layers.0.self_attn.q_norm.weight": (4,),
        "model.layers.0.self_attn.k_norm.weight": (4,),
        "model.layers.0.self_attn.g_proj.weight": (48, 4),
        "model.layers.0.mlp.gate_proj.weight": (6, 4),
        "model.layers.0.mlp.up_proj.weight": (6, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 6),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.post_attention_layernorm.weight": (4,),
        "model.layers.1.self_attn.q_proj.weight": (288, 4),
        "model.layers.1.self_attn.k_proj.weight": (32, 4),
        "model.layers.1.self_attn.v_proj.weight": (32, 4),
        "model.layers.1.self_attn.o_proj.weight": (4, 288),
        "model.layers.1.self_attn.q_norm.weight": (4,),
        "model.layers.1.self_attn.k_norm.weight": (4,),
        "model.layers.1.self_attn.g_proj.weight": (72, 4),
        "model.layers.1.mlp.experts.gate_up_proj": (256, 2, 4),
        "model.layers.1.mlp.experts.down_proj": (256, 4, 1),
        "model.layers.1.mlp.gate.weight": (256, 4),
        "model.layers.1.mlp.gate.e_score_correction_bias": (256,),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": (1, 4),
        "model.layers.1.mlp.shared_experts.up_proj.weight": (1, 4),
        "model.layers.1.mlp.shared_experts.down_proj.weight": (4, 1),
        "model.layers.1.input_layernorm.weight": (4,),
        "model.layers.1.post_attention_layernorm.weight": (4,),
        "model.norm.weight": (4,),
        "lm_head.weight": (7, 4),
    }
    expected_hashes = fixture["weight_generator"]["parameter_sha256"]
    assert set(shapes) == set(expected_hashes)
    parameters = {name: _fixture_parameter(name, shape) for name, shape in shapes.items()}
    for name, value in parameters.items():
        assert hashlib.sha256(value.tobytes()).hexdigest() == expected_hashes[name]
    return parameters


def _attention_weights(parameters: dict[str, np.ndarray], layer_id: int) -> LagunaAttentionWeights:
    prefix = f"model.layers.{layer_id}"
    return LagunaAttentionWeights(
        input_norm=parameters[f"{prefix}.input_layernorm.weight"],
        q_proj=parameters[f"{prefix}.self_attn.q_proj.weight"],
        k_proj=parameters[f"{prefix}.self_attn.k_proj.weight"],
        v_proj=parameters[f"{prefix}.self_attn.v_proj.weight"],
        gate_proj=parameters[f"{prefix}.self_attn.g_proj.weight"],
        q_norm=parameters[f"{prefix}.self_attn.q_norm.weight"],
        k_norm=parameters[f"{prefix}.self_attn.k_norm.weight"],
        o_proj=parameters[f"{prefix}.self_attn.o_proj.weight"],
    )


def _tiny_layers(parameters: dict[str, np.ndarray]) -> tuple[LagunaReferenceLayer, ...]:
    layer0_mlp = LagunaDenseFFNWeights(
        gate_proj=parameters["model.layers.0.mlp.gate_proj.weight"],
        up_proj=parameters["model.layers.0.mlp.up_proj.weight"],
        down_proj=parameters["model.layers.0.mlp.down_proj.weight"],
    )
    gate_up = parameters["model.layers.1.mlp.experts.gate_up_proj"]
    layer1_mlp = LagunaSparseFFNWeights(
        router=parameters["model.layers.1.mlp.gate.weight"],
        correction_bias=parameters["model.layers.1.mlp.gate.e_score_correction_bias"],
        expert_gate=gate_up[:, :1, :],
        expert_up=gate_up[:, 1:, :],
        expert_down=parameters["model.layers.1.mlp.experts.down_proj"],
        shared_gate=parameters["model.layers.1.mlp.shared_experts.gate_proj.weight"],
        shared_up=parameters["model.layers.1.mlp.shared_experts.up_proj.weight"],
        shared_down=parameters["model.layers.1.mlp.shared_experts.down_proj.weight"],
        experts_used=10,
        routed_scaling_factor=2.5,
    )
    return (
        LagunaReferenceLayer(
            config=LagunaAttentionConfig(
                num_heads=48,
                num_kv_heads=8,
                head_dim=4,
                rope=LagunaRopeConfig(
                    rope_type="yarn",
                    rotary_dim=2,
                    freq_base=500_000.0,
                    scaling_factor=32.0,
                    original_context_length=8_192,
                    yarn_beta_fast=32.0,
                    yarn_beta_slow=1.0,
                ),
            ),
            weights=LagunaLayerWeights(
                attention=_attention_weights(parameters, 0),
                ffn_norm=parameters["model.layers.0.post_attention_layernorm.weight"],
                mlp=layer0_mlp,
            ),
        ),
        LagunaReferenceLayer(
            config=LagunaAttentionConfig(
                num_heads=72,
                num_kv_heads=8,
                head_dim=4,
                rope=LagunaRopeConfig(
                    rope_type="default",
                    rotary_dim=4,
                    freq_base=10_000.0,
                ),
                sliding_window=512,
            ),
            weights=LagunaLayerWeights(
                attention=_attention_weights(parameters, 1),
                ffn_norm=parameters["model.layers.1.post_attention_layernorm.weight"],
                mlp=layer1_mlp,
            ),
        ),
    )
