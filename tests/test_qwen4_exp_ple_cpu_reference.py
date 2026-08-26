from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEConvState,
    dilated_depthwise_conv,
    ple_injection,
    ple_signed_sqrt_gate,
)


def test_ple_signed_sqrt_gate_pins_positive_negative_and_zero_scores() -> None:
    scores = np.array([[-4.0, -1e-8, 0.0, 1e-8, 9.0]], dtype=np.float32)

    actual = ple_signed_sqrt_gate(scores)

    transformed = np.sign(scores) * np.sqrt(np.maximum(np.abs(scores), 1e-6))
    expected = 1.0 / (1.0 + np.exp(-transformed))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual[0, 2] == np.float32(0.5)


def test_dilated_depthwise_conv_pins_causal_taps_and_history() -> None:
    values = np.arange(1, 6, dtype=np.float32)[:, None]
    kernel = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    output, state = dilated_depthwise_conv(
        values,
        kernel,
        dilation=2,
        positions=np.arange(5),
        state=None,
    )

    np.testing.assert_allclose(output[:, 0], [3.0, 6.0, 11.0, 16.0, 22.0])
    np.testing.assert_array_equal(state.history[:, 0], [2.0, 3.0, 4.0, 5.0])
    assert state.next_position == 5


def test_dilated_depthwise_conv_is_chunk_invariant_and_resets_on_discontinuity() -> None:
    values = np.arange(1, 9, dtype=np.float32)[:, None]
    kernel = np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32)
    whole, whole_state = dilated_depthwise_conv(
        values,
        kernel,
        dilation=3,
        positions=np.arange(8),
        state=None,
    )
    first, state = dilated_depthwise_conv(
        values[:3],
        kernel,
        dilation=3,
        positions=np.arange(3),
        state=None,
    )
    second, state = dilated_depthwise_conv(
        values[3:],
        kernel,
        dilation=3,
        positions=np.arange(3, 8),
        state=state,
    )

    np.testing.assert_allclose(np.concatenate((first, second)), whole, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(state.history, whole_state.history)
    reset, _ = dilated_depthwise_conv(
        values[:1],
        kernel,
        dilation=3,
        positions=[20],
        state=state,
    )
    fresh, _ = dilated_depthwise_conv(
        values[:1],
        kernel,
        dilation=3,
        positions=[20],
        state=None,
    )
    np.testing.assert_array_equal(reset, fresh)


def test_ple_injection_broadcasts_gated_value_across_residual_branches() -> None:
    residual = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    embedding = np.array([[10.0, -2.0]], dtype=np.float32)
    key_weight = np.zeros((4, 2), dtype=np.float32)
    value_weight = np.eye(2, dtype=np.float32)
    grouped_weight = np.ones((2, 2), dtype=np.float32)
    conv_kernel = np.zeros((4, 4), dtype=np.float32)

    result = ple_injection(
        residual,
        embedding,
        key_weight,
        value_weight,
        grouped_weight,
        grouped_weight,
        grouped_weight,
        conv_kernel,
        positions=[0],
        state=None,
        dilation=3,
        eps=1e-6,
    )

    np.testing.assert_allclose(result.gate, 0.5, rtol=0.0, atol=0.0)
    expected_delta = np.array([[[5.0, -1.0], [5.0, -1.0]]], dtype=np.float32)
    np.testing.assert_allclose(result.gated_value, expected_delta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.conv_output, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.residual, residual + expected_delta, rtol=0.0, atol=0.0)
    assert result.state.history.shape == (9, 4)
    assert result.state.next_position == 1


def test_ple_injection_uses_per_branch_signed_dot_gate() -> None:
    residual = np.array([[[1.0, 0.0], [-1.0, 0.0]]], dtype=np.float32)
    embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    key_weight = np.array(
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    value_weight = np.eye(2, dtype=np.float32)
    grouped_weight = np.ones((2, 2), dtype=np.float32)
    conv_kernel = np.zeros((4, 1), dtype=np.float32)

    result = ple_injection(
        residual,
        embedding,
        key_weight,
        value_weight,
        grouped_weight,
        grouped_weight,
        grouped_weight,
        conv_kernel,
        positions=[0],
        state=PLEConvState(history=np.zeros((0, 4), dtype=np.float32), next_position=0),
        dilation=3,
        eps=1e-6,
    )

    assert result.gate[0, 0] > 0.5
    assert result.gate[0, 1] < 0.5
    assert result.gate[0, 0] == pytest.approx(1.0 - result.gate[0, 1], abs=1e-6)
