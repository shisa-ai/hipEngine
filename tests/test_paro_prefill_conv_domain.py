from __future__ import annotations

import numpy as np
import pytest

from scripts.paro_prefill_conv_domain import accumulator_summary, conv_accumulators, precise_silu


def _scalar_accumulators(qkv: np.ndarray, state: np.ndarray, weight: np.ndarray) -> np.ndarray:
    tokens, channels = qkv.shape
    kernel_size = weight.shape[1]
    out = np.zeros((tokens, channels), dtype=np.float32)
    for token in range(tokens):
        for channel in range(channels):
            acc = np.float32(0.0)
            for tap in range(kernel_size):
                padded = token + tap
                if padded < kernel_size - 1:
                    value = np.float32(state[channel, padded + 1])
                else:
                    value = np.float32(qkv[padded - (kernel_size - 1), channel])
                product = np.float32(value * np.float32(weight[channel, tap]))
                acc = np.float32(acc + product)
            out[token, channel] = acc
    return out


def test_conv_accumulators_matches_ordered_scalar_four_tap_reference() -> None:
    qkv = np.asarray(
        [
            [0.25, -0.5],
            [1.5, 2.0],
            [-3.0, 0.125],
        ],
        dtype=np.float16,
    )
    state = np.asarray(
        [
            [10.0, 11.0, 12.0, 13.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )
    weight = np.asarray(
        [
            [0.5, -0.25, 0.125, 2.0],
            [-1.5, 0.75, 0.5, -0.125],
        ],
        dtype=np.float32,
    )

    actual = conv_accumulators(qkv, state, weight)

    np.testing.assert_array_equal(actual.view(np.uint32), _scalar_accumulators(qkv, state, weight).view(np.uint32))


def test_conv_accumulators_rejects_mismatched_state_and_weight_shape() -> None:
    with pytest.raises(ValueError, match="matching"):
        conv_accumulators(
            np.zeros((2, 3), dtype=np.float16),
            np.zeros((3, 4), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        )


def test_accumulator_summary_counts_extreme_precise_expf_domains() -> None:
    values = np.asarray([-105.0, -90.0, -88.0, -1.0, 0.0, 16.0, 87.0, 90.0, 105.0], dtype=np.float32)

    summary = accumulator_summary(values)

    assert summary["finite_count"] == values.size
    assert summary["threshold_counts"]["le_88"] == 3
    assert summary["threshold_counts"]["ge_87"] == 3
    assert summary["threshold_counts"]["ge_104"] == 1


def test_precise_silu_keeps_float32_shape_and_saturation_behavior() -> None:
    values = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)

    actual = precise_silu(values)

    assert actual.dtype == np.float32
    assert actual.shape == values.shape
    assert actual[0] == np.float32(-0.0)
    assert actual[2] == np.float32(0.0)
    assert actual[-1] == np.float32(100.0)
