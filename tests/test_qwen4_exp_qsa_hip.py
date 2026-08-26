from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.qwen4_exp import (
    qsa_index_scores,
    qsa_prepare_index_keys,
    qsa_select_positions,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_qsa_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        plan_qwen4_exp_qsa_build,
        qwen4_exp_qsa_score_f32,
        register_qwen4_exp_qsa_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_qsa_build()
    assert artifact.output_path.name == "qwen4_exp_qsa.so"
    register_qwen4_exp_qsa_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_index_score",
            quant="f32",
            variant="strict",
        )
        is qwen4_exp_qsa_score_f32
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_native_query_norm_rope_matches_split_half_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.qwen4_exp import qsa_interleaved_rope
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_norm_rope_f32,
    )

    runtime = get_hip_runtime()
    query = np.array(
        [[1.0, 2.0, -3.0, 4.0], [-2.0, 0.5, 1.0, 3.0]],
        dtype=np.float32,
    )
    weight = np.array([1.0, 0.8, 1.2, 0.9], dtype=np.float32)
    position = np.array([7], dtype=np.int64)
    normalized = query / np.sqrt(
        np.mean(query * query, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1e-6)
    ) * weight
    expected = qsa_interleaved_rope(
        normalized[None], positions=position, rotary_dim=4, theta=100.0
    )[0]

    allocations = []
    try:
        d_query = _upload(query, runtime, allocations)
        d_weight = _upload(weight, runtime, allocations)
        d_position = _upload(position, runtime, allocations)
        d_output = _alloc(expected.shape, np.float32, runtime, allocations)
        qwen4_exp_qsa_norm_rope_f32(
            d_query.ptr,
            d_weight.ptr,
            d_position.ptr,
            d_output.ptr,
            heads=2,
            head_dim=4,
            rotary_dim=4,
            theta=100.0,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_native_pool_score_select_match_cpu() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        build_qwen4_exp_qsa,
        qwen4_exp_qsa_pool_norm_rope_f32,
        qwen4_exp_qsa_score_f32,
        qwen4_exp_qsa_select_blocks_f32_i64,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_qsa(load=True)
    rng = np.random.default_rng(408)
    ratio, blocks, index_dim, index_heads = 4, 6, 128, 4
    tokens = blocks * ratio + 3
    permutation = rng.permutation(tokens)
    positions = np.arange(tokens, dtype=np.int64)[permutation]
    raw_keys = rng.normal(0.0, 0.2, size=(tokens, index_dim)).astype(np.float32)
    gamma = rng.normal(1.0, 0.05, size=(index_dim,)).astype(np.float32)
    prepared = qsa_prepare_index_keys(
        raw_keys,
        positions,
        gamma,
        compression_ratio=ratio,
        rotary_dim=64,
        theta=10_000_000.0,
    )
    queries = rng.normal(0.0, 0.2, size=(3, index_heads, index_dim)).astype(np.float32)
    expected_scores = qsa_index_scores(queries, prepared.keys)
    query_positions = np.array([7, 19, 26], dtype=np.int64)
    expected_selection = qsa_select_positions(
        expected_scores,
        prepared.block_starts,
        query_positions=query_positions,
        available_positions=np.arange(tokens),
        compression_ratio=ratio,
        block_budget=2,
    )

    allocations = []
    try:
        d_raw = _upload(raw_keys, runtime, allocations)
        d_members = _upload(prepared.member_indices.astype(np.int32), runtime, allocations)
        d_starts = _upload(prepared.block_starts, runtime, allocations)
        d_gamma = _upload(gamma, runtime, allocations)
        d_prepared = _alloc(prepared.keys.shape, np.float32, runtime, allocations)
        qwen4_exp_qsa_pool_norm_rope_f32(
            d_raw.ptr,
            d_members.ptr,
            d_starts.ptr,
            d_gamma.ptr,
            d_prepared.ptr,
            blocks,
            ratio,
            index_dim,
            64,
            10_000_000.0,
            1e-6,
            library=library,
            runtime=runtime,
        )
        d_queries = _upload(queries, runtime, allocations)
        d_scores = _alloc(expected_scores.shape, np.float32, runtime, allocations)
        qwen4_exp_qsa_score_f32(
            d_queries.ptr,
            d_prepared.ptr,
            d_scores.ptr,
            queries.shape[0],
            blocks,
            index_heads,
            index_dim,
            library=library,
            runtime=runtime,
        )
        d_query_positions = _upload(query_positions, runtime, allocations)
        selected_shape = (queries.shape[0], 2)
        d_selected = _alloc(selected_shape, np.int64, runtime, allocations)
        d_counts = _alloc((queries.shape[0],), np.int32, runtime, allocations)
        qwen4_exp_qsa_select_blocks_f32_i64(
            d_scores.ptr,
            d_starts.ptr,
            d_query_positions.ptr,
            d_selected.ptr,
            d_counts.ptr,
            queries.shape[0],
            blocks,
            ratio,
            2,
            library=library,
            runtime=runtime,
        )
        tie_scores = np.ones_like(expected_scores)
        tie_positions = np.full(queries.shape[0], 26, dtype=np.int64)
        d_tie_scores = _upload(tie_scores, runtime, allocations)
        d_tie_positions = _upload(tie_positions, runtime, allocations)
        d_tie_selected = _alloc(selected_shape, np.int64, runtime, allocations)
        d_tie_counts = _alloc((queries.shape[0],), np.int32, runtime, allocations)
        qwen4_exp_qsa_select_blocks_f32_i64(
            d_tie_scores.ptr,
            d_starts.ptr,
            d_tie_positions.ptr,
            d_tie_selected.ptr,
            d_tie_counts.ptr,
            queries.shape[0],
            blocks,
            ratio,
            2,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_prepared = _download(
            d_prepared,
            prepared.keys.shape,
            np.float32,
            runtime,
        )
        actual_scores = _download(d_scores, expected_scores.shape, np.float32, runtime)
        actual_selected = _download(d_selected, selected_shape, np.int64, runtime)
        actual_counts = _download(d_counts, (queries.shape[0],), np.int32, runtime)
        actual_tie_selected = _download(
            d_tie_selected,
            selected_shape,
            np.int64,
            runtime,
        )
        actual_tie_counts = _download(
            d_tie_counts,
            (queries.shape[0],),
            np.int32,
            runtime,
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_prepared, prepared.keys, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=2e-5, atol=2e-5)
    for row, expected in enumerate(expected_selection.selected_block_starts):
        np.testing.assert_array_equal(actual_selected[row, : actual_counts[row]], expected)
    np.testing.assert_array_equal(actual_tie_counts, 2)
    np.testing.assert_array_equal(actual_tie_selected, [[0, 4], [0, 4], [0, 4]])


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
