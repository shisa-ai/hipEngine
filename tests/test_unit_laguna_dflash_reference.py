from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.laguna import (
    LagunaAttentionConfig,
    LagunaAttentionWeights,
    LagunaDenseFFNWeights,
    LagunaLayerWeights,
    LagunaReferenceLayer,
    LagunaRopeConfig,
    laguna_attention_forward,
    laguna_dflash_attention_forward,
    laguna_dflash_layer_forward,
    laguna_dflash_model_forward,
    laguna_dflash_project_target_hidden,
    laguna_head_rmsnorm,
)
from hipengine.kernels.cpu_reference.ops import linear


def test_laguna_dflash_target_projection_norms_each_tap_before_concat() -> None:
    taps = (
        np.array([[1.0, 2.0], [2.0, -1.0]], dtype=np.float32),
        np.array([[3.0, -4.0], [0.5, 1.5]], dtype=np.float32),
    )
    aux_norms = (
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([0.5, 1.5], dtype=np.float32),
    )
    fc = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, -0.25],
        ],
        dtype=np.float32,
    )
    hidden_norm = np.array([1.25, 0.75], dtype=np.float32)

    result = laguna_dflash_project_target_hidden(
        taps,
        aux_norms,
        fc,
        hidden_norm,
        eps=1.0e-6,
    )

    expected_taps = tuple(
        laguna_head_rmsnorm(tap, weight, eps=1.0e-6)
        for tap, weight in zip(taps, aux_norms, strict=True)
    )
    expected_concat = np.concatenate(expected_taps, axis=1)
    expected_projected = linear(expected_concat, fc)
    expected = laguna_head_rmsnorm(expected_projected, hidden_norm, eps=1.0e-6)
    for actual, expected_tap in zip(result.normalized_taps, expected_taps, strict=True):
        np.testing.assert_allclose(actual, expected_tap, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(result.concatenated, expected_concat, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(result.projected, expected_projected, atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(result.normalized, expected, atol=1.0e-6, rtol=1.0e-6)


def test_laguna_dflash_target_projection_validates_aux_count() -> None:
    with pytest.raises(ValueError, match="auxiliary norm"):
        laguna_dflash_project_target_hidden(
            (np.ones((1, 2), dtype=np.float32),) * 2,
            (np.ones(2, dtype=np.float32),),
            np.ones((2, 4), dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )


def test_laguna_dflash_attention_matches_query_slice_of_full_causal_attention() -> None:
    layer = _layer(seed=9)
    context = np.array(
        [[0.5, -0.25, 0.75, 0.125], [-0.4, 0.3, 0.2, 0.1]],
        dtype=np.float32,
    )
    query = np.array(
        [[0.2, 0.1, -0.3, 0.4], [0.6, -0.5, 0.25, -0.1]],
        dtype=np.float32,
    )
    context_positions = np.array([3, 4], dtype=np.int64)
    query_positions = np.array([5, 6], dtype=np.int64)

    full = laguna_attention_forward(
        np.concatenate((context, query), axis=0),
        layer.weights.attention,
        layer.config,
        positions=np.concatenate((context_positions, query_positions)),
    )
    draft = laguna_dflash_attention_forward(
        query,
        context,
        layer.weights.attention,
        layer.config,
        context_positions=context_positions,
        query_positions=query_positions,
    )

    np.testing.assert_allclose(draft.normalized, full.normalized[2:], atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(draft.context, full.context[2:], atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(draft.gated_context, full.gated_context[2:], atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(draft.output, full.output[2:], atol=1.0e-6, rtol=1.0e-6)


def test_laguna_dflash_attention_is_causal_within_noise_block() -> None:
    layer = _layer(seed=10)
    context = np.array([[0.5, -0.25, 0.75, 0.125]], dtype=np.float32)
    query = np.array(
        [[0.2, 0.1, -0.3, 0.4], [0.6, -0.5, 0.25, -0.1]],
        dtype=np.float32,
    )
    changed = query.copy()
    changed[1] = np.array([50.0, -40.0, 30.0, -20.0], dtype=np.float32)
    kwargs = {
        "context_positions": np.array([10], dtype=np.int64),
        "query_positions": np.array([11, 12], dtype=np.int64),
    }

    baseline = laguna_dflash_attention_forward(
        query, context, layer.weights.attention, layer.config, **kwargs
    )
    perturbed = laguna_dflash_attention_forward(
        changed, context, layer.weights.attention, layer.config, **kwargs
    )

    np.testing.assert_array_equal(baseline.output[0], perturbed.output[0])
    assert not np.array_equal(baseline.output[1], perturbed.output[1])


def test_laguna_dflash_layer_matches_full_layer_query_slice() -> None:
    layer = _layer(seed=11)
    context = np.array([[0.5, -0.25, 0.75, 0.125]], dtype=np.float32)
    query = np.array(
        [[0.2, 0.1, -0.3, 0.4], [0.6, -0.5, 0.25, -0.1]],
        dtype=np.float32,
    )
    context_positions = np.array([1], dtype=np.int64)
    query_positions = np.array([2, 3], dtype=np.int64)

    draft = laguna_dflash_layer_forward(
        query,
        context,
        layer,
        context_positions=context_positions,
        query_positions=query_positions,
    )
    from hipengine.kernels.cpu_reference.laguna import laguna_layer_forward

    full = laguna_layer_forward(
        np.concatenate((context, query), axis=0),
        layer,
        positions=np.concatenate((context_positions, query_positions)),
    )

    np.testing.assert_allclose(draft.hidden, full.hidden[1:], atol=2.0e-6, rtol=2.0e-6)


def test_laguna_dflash_model_forward_returns_candidate_rows_only() -> None:
    layers = (_layer(seed=12), _layer(seed=13))
    context = np.array([[0.5, -0.25, 0.75, 0.125]], dtype=np.float32)
    query = np.array(
        [[0.2, 0.1, -0.3, 0.4], [0.6, -0.5, 0.25, -0.1], [0.1, 0.2, 0.3, 0.4]],
        dtype=np.float32,
    )
    final_norm = np.ones(4, dtype=np.float32)
    output = np.arange(28, dtype=np.float32).reshape(7, 4) / np.float32(20.0)

    result = laguna_dflash_model_forward(
        query,
        context,
        layers,
        final_norm,
        output,
        context_positions=np.array([20], dtype=np.int64),
        query_positions=np.array([21, 22, 23], dtype=np.int64),
    )

    assert len(result.hidden_states) == 3
    assert result.final_hidden.shape == (3, 4)
    assert result.logits.shape == (3, 7)
    assert np.isfinite(result.logits).all()


def _layer(*, seed: int) -> LagunaReferenceLayer:
    rng = np.random.default_rng(seed)
    hidden = 4
    heads = 2
    kv_heads = 1
    head_dim = 2
    intermediate = 6

    def weight(shape: tuple[int, ...], scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)).astype(np.float32)

    attention = LagunaAttentionWeights(
        input_norm=np.ones(hidden, dtype=np.float32),
        q_proj=weight((heads * head_dim, hidden)),
        k_proj=weight((kv_heads * head_dim, hidden)),
        v_proj=weight((kv_heads * head_dim, hidden)),
        gate_proj=weight((heads, hidden)),
        q_norm=np.ones(head_dim, dtype=np.float32),
        k_norm=np.ones(head_dim, dtype=np.float32),
        o_proj=weight((hidden, heads * head_dim)),
    )
    mlp = LagunaDenseFFNWeights(
        gate_proj=weight((intermediate, hidden)),
        up_proj=weight((intermediate, hidden)),
        down_proj=weight((hidden, intermediate)),
    )
    return LagunaReferenceLayer(
        config=LagunaAttentionConfig(
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            rope=LagunaRopeConfig(
                rope_type="default",
                rotary_dim=head_dim,
                freq_base=500_000.0,
            ),
            sliding_window=512,
        ),
        weights=LagunaLayerWeights(
            attention=attention,
            ffn_norm=np.ones(hidden, dtype=np.float32),
            mlp=mlp,
        ),
    )
