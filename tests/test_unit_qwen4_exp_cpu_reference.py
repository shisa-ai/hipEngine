from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEHashState,
    gr_read,
    gr_write,
    grouped_zero_centered_rmsnorm,
    ple_hash_rows,
    sigmoid_gated_rmsnorm,
)

PLE_EOS = 248_044
PLE_MULTIPLIERS = (23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071)
PLE_HEAD_SIZES = (
    20_000_003,
    20_000_023,
    20_000_033,
    20_000_047,
    20_000_059,
    20_000_063,
    20_000_069,
    20_000_077,
    20_000_081,
    20_000_093,
    20_000_107,
    20_000_147,
    20_000_153,
    20_000_159,
    20_000_161,
    20_000_171,
)
PLE_HEAD_OFFSETS = tuple(
    sum(PLE_HEAD_SIZES[:index]) for index in range(len(PLE_HEAD_SIZES))
)


def test_grouped_zero_centered_rmsnorm_normalizes_each_residual_branch() -> None:
    residual = np.array([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
    gamma = np.array([[2.0, 0.5], [1.5, 3.0]], dtype=np.float32)

    actual = grouped_zero_centered_rmsnorm(residual, gamma, eps=0.0)

    branch0 = residual[0, 0] / np.sqrt(np.mean(residual[0, 0] ** 2)) * gamma[0]
    branch1 = residual[0, 1] / np.sqrt(np.mean(residual[0, 1] ** 2)) * gamma[1]
    np.testing.assert_allclose(actual, [[branch0, branch1]], rtol=0.0, atol=1e-7)


def test_gr_read_and_write_pin_low_rank_gate_mean_and_branch_injection() -> None:
    residual = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    norm_weight = np.ones((2, 2), dtype=np.float32)
    down = np.zeros((1, 4), dtype=np.float32)
    up = np.zeros((4, 1), dtype=np.float32)
    inject = np.zeros((2, 4), dtype=np.float32)

    read = gr_read(residual, norm_weight, down, up, inject, eps=0.0)

    expected_norm = grouped_zero_centered_rmsnorm(residual, norm_weight, eps=0.0)
    np.testing.assert_allclose(read.normalized, expected_norm, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(read.gate, 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        read.mixed,
        np.mean(expected_norm * 0.5, axis=1),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(read.inject_logits, 0.0, rtol=0.0, atol=0.0)

    block = np.array([[10.0, -2.0]], dtype=np.float32)
    written = gr_write(residual, block, read.inject_logits)
    np.testing.assert_allclose(written, residual + block[:, None, :], rtol=0.0, atol=0.0)


def test_gr_read_uses_silu_after_branch_scaled_down_projection() -> None:
    residual = np.array([[[1.0, -2.0], [0.5, 3.0]]], dtype=np.float32)
    norm_weight = np.ones((2, 2), dtype=np.float32)
    down = np.array([[1.0, -1.0, 0.5, 2.0]], dtype=np.float32)
    up = np.array([[0.2], [-0.4], [0.6], [0.8]], dtype=np.float32)
    inject = np.array([[0.1, 0.2, 0.3, 0.4], [-0.5, 0.6, -0.7, 0.8]], dtype=np.float32)

    result = gr_read(residual, norm_weight, down, up, inject, eps=1e-6)

    normed = grouped_zero_centered_rmsnorm(residual, norm_weight, eps=1e-6)
    flat = normed.reshape(1, 4)
    low = flat @ down.T / np.float32(2.0)
    low = low / (np.float32(1.0) + np.exp(-low))
    gate = np.float32(1.0) / (np.float32(1.0) + np.exp(-(low @ up.T)))
    np.testing.assert_allclose(result.gate.reshape(1, 4), gate, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.inject_logits, flat @ inject.T, rtol=1e-6, atol=1e-7)


def test_ple_hash_rows_preserve_uint64_overflow_eos_cut_and_all_heads() -> None:
    rows, states = ple_hash_rows(
        [10, 20, PLE_EOS, 30],
        positions=[0, 1, 2, 3],
        sequence_ids=[7, 7, 7, 7],
        states={},
        eos_token_id=PLE_EOS,
        layer_multipliers=PLE_MULTIPLIERS,
        head_offsets=PLE_HEAD_OFFSETS,
        head_vocab_sizes=PLE_HEAD_SIZES,
        heads_per_ngram=8,
        ngram_size=3,
    )

    assert rows.shape == (4, 16)
    assert hashlib.sha256(rows.astype("<i8", copy=False).tobytes()).hexdigest() == (
        "8525e0aaa17266747dd144eb3963ba1e5db140c7b1de3a25d304945678077929"
    )
    assert rows[0].tolist() == [
        6_826_666,
        27_775_725,
        51_991_156,
        74_082_527,
        82_622_748,
        119_600_976,
        135_816_374,
        152_166_807,
        174_244_281,
        190_221_032,
        211_723_794,
        232_787_707,
        243_645_790,
        275_729_718,
        280_030_017,
        303_574_322,
    ]
    assert states[7] == PLEHashState(tokens=(PLE_EOS, 30), next_position=4)


def test_ple_hash_rows_are_chunk_invariant_and_request_local() -> None:
    kwargs = {
        "eos_token_id": PLE_EOS,
        "layer_multipliers": PLE_MULTIPLIERS,
        "head_offsets": PLE_HEAD_OFFSETS,
        "head_vocab_sizes": PLE_HEAD_SIZES,
        "heads_per_ngram": 8,
        "ngram_size": 3,
    }
    whole, whole_states = ple_hash_rows(
        [11, 12, 21, 22],
        positions=[0, 1, 0, 1],
        sequence_ids=[1, 1, 2, 2],
        states={},
        **kwargs,
    )
    first, state = ple_hash_rows(
        [11, 21],
        positions=[0, 0],
        sequence_ids=[1, 2],
        states={},
        **kwargs,
    )
    second, state = ple_hash_rows(
        [12, 22],
        positions=[1, 1],
        sequence_ids=[1, 2],
        states=state,
        **kwargs,
    )

    np.testing.assert_array_equal(np.stack((first[0], second[0], first[1], second[1])), whole)
    assert state == whole_states
    reset, reset_state = ple_hash_rows(
        [99],
        positions=[9],
        sequence_ids=[1],
        states=state,
        **kwargs,
    )
    fresh, _ = ple_hash_rows(
        [99],
        positions=[9],
        sequence_ids=[1],
        states={},
        **kwargs,
    )
    np.testing.assert_array_equal(reset, fresh)
    assert reset_state[2] == state[2]


def test_sigmoid_gated_rmsnorm_uses_sigmoid_not_silu() -> None:
    value = np.array([[[3.0, 4.0]]], dtype=np.float32)
    weight = np.array([2.0, 0.5], dtype=np.float32)
    gate = np.array([[[0.0, 2.0]]], dtype=np.float32)

    actual = sigmoid_gated_rmsnorm(value, weight, gate, eps=0.0)

    normed = value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True)) * weight
    expected = normed / (1.0 + np.exp(-gate))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_qwen4_exp_cpu_reference_rejects_invalid_state_shapes() -> None:
    with pytest.raises(ValueError, match="residual"):
        gr_read(
            np.zeros((2, 4), dtype=np.float32),
            np.ones((4,), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            np.ones((4, 1), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="positions"):
        ple_hash_rows(
            [1, 2],
            positions=[0],
            sequence_ids=[1, 1],
            states={},
            eos_token_id=PLE_EOS,
            layer_multipliers=PLE_MULTIPLIERS,
            head_offsets=PLE_HEAD_OFFSETS,
            head_vocab_sizes=PLE_HEAD_SIZES,
            heads_per_ngram=8,
            ngram_size=3,
        )
