from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.laguna import (
    laguna_sigmoid_correction_topk,
    laguna_softplus_head_gate,
    register_laguna_cpu_reference_kernels,
)
from hipengine.kernels.registry import resolve


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

    assert resolve(
        backend="cpu_reference",
        layer="softplus_head_gate",
        quant="fp32",
        variant="laguna_per_head",
    ) is laguna_softplus_head_gate
    assert resolve(
        backend="cpu_reference",
        layer="laguna_sigmoid_router_topk",
        quant="gguf_f32",
        variant="correction_bias",
    ) is laguna_sigmoid_correction_topk
