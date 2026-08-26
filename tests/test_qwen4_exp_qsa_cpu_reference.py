from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.qwen4_exp import (
    qsa_index_scores,
    qsa_interleaved_rope,
    qsa_pool_complete_blocks,
    qsa_prepare_index_keys,
    qsa_select_positions,
    qsa_sparse_gqa_attention,
)


def test_qsa_pool_complete_blocks_uses_fp32_mean_and_preserves_physical_members() -> None:
    keys = np.array([[4.0], [0.0], [3.0], [1.0], [2.0]], dtype=np.float16)
    positions = np.array([4, 0, 3, 1, 2], dtype=np.int64)

    pooled = qsa_pool_complete_blocks(keys, positions, compression_ratio=4)

    np.testing.assert_array_equal(pooled.block_starts, [0])
    np.testing.assert_array_equal(pooled.member_indices, [[1, 3, 4, 2]])
    np.testing.assert_allclose(pooled.keys, [[1.5]], rtol=0.0, atol=0.0)
    assert pooled.keys.dtype == np.float32
    np.testing.assert_array_equal(pooled.tail_indices, [0])


def test_qsa_pool_complete_blocks_rejects_non_tail_logical_holes() -> None:
    with pytest.raises(ValueError, match="incomplete non-tail"):
        qsa_pool_complete_blocks(
            np.ones((4, 2), dtype=np.float32),
            [0, 1, 3, 4],
            compression_ratio=2,
        )


def test_qsa_interleaved_rope_rotates_pairs_and_preserves_tail_dimensions() -> None:
    values = np.array([[[1.0, 0.0, 2.0, 0.0, 9.0]]], dtype=np.float32)

    actual = qsa_interleaved_rope(
        values,
        positions=[1],
        rotary_dim=4,
        theta=100.0,
    )

    expected = np.array(
        [[[np.cos(1.0), np.sin(1.0), 2.0 * np.cos(0.1), 2.0 * np.sin(0.1), 9.0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_qsa_prepare_index_keys_pools_before_norm_and_rope() -> None:
    raw = np.array(
        [[3.0, 4.0], [0.0, 2.0], [1.0, 0.0], [2.0, 2.0], [9.0, 9.0]],
        dtype=np.float32,
    )
    prepared = qsa_prepare_index_keys(
        raw,
        np.arange(5),
        np.ones(2, dtype=np.float32),
        compression_ratio=4,
        rotary_dim=2,
        theta=10_000_000.0,
        eps=0.0,
    )

    mean = np.mean(raw[:4], axis=0, dtype=np.float32)
    normalized = mean / np.sqrt(np.mean(mean * mean))
    np.testing.assert_allclose(prepared.keys[0], normalized, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(prepared.block_starts, [0])
    np.testing.assert_array_equal(prepared.tail_indices, [4])


def test_qsa_index_scores_rectify_each_head_before_sum() -> None:
    queries = np.array([[[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]]], dtype=np.float32)
    keys = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)

    actual = qsa_index_scores(queries, keys)

    scale = np.sqrt(2.0)
    np.testing.assert_allclose(actual, [[3.0 / scale, 2.0 / scale]], rtol=1e-6, atol=1e-7)


def test_qsa_selection_keeps_best_complete_blocks_plus_incomplete_tail() -> None:
    block_starts = np.array([0, 2, 4], dtype=np.int64)
    available = np.arange(6, dtype=np.int64)

    at_tail = qsa_select_positions(
        np.array([[0.1, 0.9, 100.0]], dtype=np.float32),
        block_starts,
        query_positions=[4],
        available_positions=available,
        compression_ratio=2,
        block_budget=2,
    )
    np.testing.assert_array_equal(at_tail.selected_block_starts[0], [0, 2])
    np.testing.assert_array_equal(at_tail.selected_positions[0], [0, 1, 2, 3, 4])

    over_budget = qsa_select_positions(
        np.array([[0.1, 0.9, 0.8]], dtype=np.float32),
        block_starts,
        query_positions=[5],
        available_positions=available,
        compression_ratio=2,
        block_budget=2,
    )
    np.testing.assert_array_equal(over_budget.selected_block_starts[0], [2, 4])
    np.testing.assert_array_equal(over_budget.selected_positions[0], [2, 3, 4, 5])


def test_qsa_selection_is_deterministic_on_ties_and_dense_below_budget() -> None:
    selection = qsa_select_positions(
        np.ones((1, 3), dtype=np.float32),
        [0, 2, 4],
        query_positions=[5],
        available_positions=np.arange(6),
        compression_ratio=2,
        block_budget=3,
    )
    np.testing.assert_array_equal(selection.selected_block_starts[0], [0, 2, 4])
    np.testing.assert_array_equal(selection.selected_positions[0], np.arange(6))

    tied_cut = qsa_select_positions(
        np.ones((1, 3), dtype=np.float32),
        [0, 2, 4],
        query_positions=[5],
        available_positions=np.arange(6),
        compression_ratio=2,
        block_budget=2,
    )
    np.testing.assert_array_equal(tied_cut.selected_block_starts[0], [0, 2])


def test_qsa_sparse_gqa_attention_matches_dense_with_noncontiguous_physical_slots() -> None:
    rng = np.random.default_rng(7)
    query = rng.normal(size=(1, 4, 3)).astype(np.float32)
    key_dense = rng.normal(size=(5, 2, 3)).astype(np.float32)
    value_dense = rng.normal(size=(5, 2, 3)).astype(np.float32)
    permutation = np.array([3, 0, 4, 1, 2])
    key = key_dense[permutation]
    value = value_dense[permutation]
    key_positions = np.arange(5)[permutation]

    sparse = qsa_sparse_gqa_attention(
        query,
        key,
        value,
        query_positions=[4],
        key_positions=key_positions,
        selected_positions=(np.arange(5),),
    )
    dense = qsa_sparse_gqa_attention(
        query,
        key_dense,
        value_dense,
        query_positions=[4],
        key_positions=np.arange(5),
        selected_positions=(np.arange(5),),
    )

    np.testing.assert_allclose(sparse, dense, rtol=1e-6, atol=1e-6)


def test_qsa_sparse_gqa_attention_uses_original_uncompressed_values() -> None:
    query = np.ones((1, 2, 1), dtype=np.float32)
    key = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    value = np.array([[[10.0]], [[20.0]], [[100.0]]], dtype=np.float32)

    actual = qsa_sparse_gqa_attention(
        query,
        key,
        value,
        query_positions=[2],
        key_positions=[0, 1, 2],
        selected_positions=(np.array([0, 2]),),
    )

    weights = np.exp([1.0, 3.0] - np.float32(3.0))
    expected = np.sum(weights * np.array([10.0, 100.0])) / np.sum(weights)
    np.testing.assert_allclose(actual[0, :, 0], expected, rtol=1e-6, atol=1e-6)
