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
        qwen4_exp_qsa_scatter_index_key_device_position_f32,
        qwen4_exp_qsa_scatter_index_keys_f32,
        qwen4_exp_qsa_split_norm_rope_rows_f32,
        qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32,
        qwen4_exp_qsa_topk_expand_f32_i64,
        qwen4_exp_qsa_topk_expand_rows_f32_i64,
        register_qwen4_exp_qsa_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_qsa_build()
    assert artifact.output_path.name == "qwen4_exp_qsa.so"
    register_qwen4_exp_qsa_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_index_append",
            quant="f32",
            variant="strict_rows_paged",
        )
        is qwen4_exp_qsa_scatter_index_keys_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_index_append",
            quant="f32",
            variant="strict_device_position_c1",
        )
        is qwen4_exp_qsa_scatter_index_key_device_position_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_index_score",
            quant="f32",
            variant="strict",
        )
        is qwen4_exp_qsa_score_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_split_norm_rope",
            quant="f32",
            variant="strict_rows",
        )
        is qwen4_exp_qsa_split_norm_rope_rows_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_select_blocks",
            quant="f32_i64",
            variant="strict_device_expand",
        )
        is qwen4_exp_qsa_topk_expand_f32_i64
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_select_blocks",
            quant="f32_i64",
            variant="strict_device_expand_rows",
        )
        is qwen4_exp_qsa_topk_expand_rows_f32_i64
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="qsa_sparse_attention",
            quant="bf16_kv",
            variant="production_wave32_h128_spans",
        )
        is qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32
    )

@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_index_row_scatter_matches_paged_positions() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        build_qwen4_exp_qsa,
        qwen4_exp_qsa_scatter_index_keys_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_qsa(load=True)
    rows, index_dim, block_size = 4, 3, 4
    start = 2
    source = np.arange(rows * index_dim, dtype=np.float32).reshape(rows, index_dim)
    block_table = np.asarray([1, 0], dtype=np.int32)
    destination = np.full((8, index_dim), -1.0, dtype=np.float32)
    expected = destination.copy()
    for row in range(rows):
        logical = start + row
        physical = (
            int(block_table[logical // block_size]) * block_size
            + logical % block_size
        )
        expected[physical] = source[row]
    allocations = []
    try:
        d_source = _upload(source, runtime, allocations)
        d_blocks = _upload(block_table, runtime, allocations)
        d_destination = _upload(destination, runtime, allocations)
        qwen4_exp_qsa_scatter_index_keys_f32(
            d_source.ptr,
            d_destination.ptr,
            d_blocks.ptr,
            start,
            rows,
            block_size,
            index_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_destination, destination.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_device_position_index_scatter_matches_strict() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        build_qwen4_exp_qsa,
        qwen4_exp_qsa_scatter_index_key_device_position_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_qsa(load=True)
    index_dim, block_size = 128, 4
    source = np.arange(index_dim, dtype=np.float32)
    block_table = np.asarray([1, 0], dtype=np.int32)
    position = np.asarray([5], dtype=np.int64)
    destination = np.full((8, index_dim), -1.0, dtype=np.float32)
    expected = destination.copy()
    physical = int(block_table[position[0] // block_size]) * block_size + position[0] % block_size
    expected[physical] = source
    allocations = []
    try:
        d_source = _upload(source, runtime, allocations)
        d_blocks = _upload(block_table, runtime, allocations)
        d_position = _upload(position, runtime, allocations)
        d_destination = _upload(destination, runtime, allocations)
        qwen4_exp_qsa_scatter_index_key_device_position_f32(
            d_source.ptr,
            d_destination.ptr,
            d_blocks.ptr,
            d_position.ptr,
            block_size,
            block_table.size,
            index_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_destination, destination.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_mrope_rows_match_interleaved_three_axis_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_norm_mrope_rows_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4388)
    rows, heads, head_dim, rotary_dim = 3, 2, 64, 64
    values = rng.normal(0.0, 0.2, size=(rows, heads, head_dim)).astype(np.float32)
    weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    positions = np.asarray(
        [[7, 8, 9], [17, 18, 19], [27, 28, 29]], dtype=np.int64
    )
    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1e-6)
    ) * weight
    expected = normalized.copy()
    half = rotary_dim // 2
    for row in range(rows):
        for pair in range(half):
            axis = (
                1
                if pair % 3 == 1 and pair < 33
                else 2
                if pair % 3 == 2 and pair < 30
                else 0
            )
            angle = np.float32(positions[axis, row]) * np.float32(
                10_000_000.0 ** (-2.0 * pair / rotary_dim)
            )
            c, s = np.cos(angle), np.sin(angle)
            x = normalized[row, :, pair].copy()
            y = normalized[row, :, pair + half].copy()
            expected[row, :, pair] = x * c - y * s
            expected[row, :, pair + half] = x * s + y * c

    allocations = []
    try:
        d_values = _upload(values, runtime, allocations)
        d_weight = _upload(weight, runtime, allocations)
        d_positions = _upload(positions, runtime, allocations)
        d_output = _alloc(values.shape, np.float32, runtime, allocations)
        qwen4_exp_qsa_norm_mrope_rows_f32(
            d_values.ptr,
            d_weight.ptr,
            d_positions.ptr,
            d_output.ptr,
            rows,
            heads,
            head_dim,
            rotary_dim,
            10_000_000.0,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, values.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("query_position", [4095, 4094])
def test_qwen4_exp_qsa_gpu_topk_expands_blocks_tail_and_breaks_ties(
    query_position: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_topk_expand_f32_i64,
    )

    runtime = get_hip_runtime()
    ratio, budget = 4, 512
    blocks = (query_position + 1) // ratio
    scores = np.zeros(blocks, dtype=np.float32)
    high = min(124, blocks)
    scores[-high:] = np.arange(1, high + 1, dtype=np.float32)
    selected_high = np.arange(blocks - high, blocks, dtype=np.int64)
    selected_ties = np.arange(budget - high, dtype=np.int64)
    selected_blocks = np.sort(np.concatenate((selected_ties, selected_high)))
    expected = np.concatenate(
        [
            (selected_blocks[:, None] * ratio + np.arange(ratio)).reshape(-1),
            np.arange(blocks * ratio, query_position + 1, dtype=np.int64),
        ]
    )

    allocations = []
    try:
        d_scores = _upload(scores, runtime, allocations)
        output_shape = (budget * ratio + ratio - 1,)
        d_selected = _alloc(output_shape, np.int64, runtime, allocations)
        d_count = _alloc((1,), np.int32, runtime, allocations)
        qwen4_exp_qsa_topk_expand_f32_i64(
            d_scores.ptr,
            d_selected.ptr,
            d_count.ptr,
            blocks,
            query_position,
            ratio,
            budget,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_count = _download(d_count, (1,), np.int32, runtime)
        actual = _download(d_selected, output_shape, np.int64, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    assert int(actual_count[0]) == expected.size
    np.testing.assert_array_equal(actual[: expected.size], expected)
    np.testing.assert_array_equal(actual[expected.size :], -1)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("blocks", [513, 4_096, 65_536])
def test_qwen4_exp_qsa_gpu_topk_matches_host_lexsort(blocks: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_topk_expand_f32_i64,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4347 + blocks)
    ratio, budget = 4, 512
    scores = rng.uniform(0.0, 10.0, size=blocks).astype(np.float32)
    selected_blocks = np.sort(
        np.lexsort((np.arange(blocks, dtype=np.int64), -scores))[:budget]
    )
    expected = (
        selected_blocks[:, None] * ratio + np.arange(ratio, dtype=np.int64)
    ).reshape(-1)
    output_shape = (budget * ratio + ratio - 1,)

    allocations = []
    try:
        d_scores = _upload(scores, runtime, allocations)
        d_selected = _alloc(output_shape, np.int64, runtime, allocations)
        d_count = _alloc((1,), np.int32, runtime, allocations)
        qwen4_exp_qsa_topk_expand_f32_i64(
            d_scores.ptr,
            d_selected.ptr,
            d_count.ptr,
            blocks,
            blocks * ratio - 1,
            ratio,
            budget,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_count = _download(d_count, (1,), np.int32, runtime)
        actual = _download(d_selected, output_shape, np.int64, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    assert int(actual_count[0]) == expected.size
    np.testing.assert_array_equal(actual[: expected.size], expected)
    np.testing.assert_array_equal(actual[expected.size :], -1)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_qsa_gpu_topk_rows_match_variable_host_prefixes() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_topk_expand_rows_f32_i64,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4351)
    ratio, budget, stride = 4, 512, 1_024
    positions = np.asarray([2_051, 2_998, 4_095], dtype=np.int64)
    scores = rng.uniform(0.0, 10.0, size=(positions.size, stride)).astype(np.float32)
    output_shape = (positions.size, budget * ratio + ratio - 1)
    expected: list[np.ndarray] = []
    for row, position in enumerate(positions):
        blocks = (int(position) + 1) // ratio
        selected_blocks = np.sort(
            np.lexsort(
                (np.arange(blocks, dtype=np.int64), -scores[row, :blocks])
            )[:budget]
        )
        expected.append(
            np.concatenate(
                (
                    (
                        selected_blocks[:, None] * ratio
                        + np.arange(ratio, dtype=np.int64)
                    ).reshape(-1),
                    np.arange(blocks * ratio, int(position) + 1, dtype=np.int64),
                )
            )
        )

    allocations = []
    try:
        d_scores = _upload(scores, runtime, allocations)
        d_positions = _upload(positions, runtime, allocations)
        d_selected = _alloc(output_shape, np.int64, runtime, allocations)
        d_counts = _alloc(positions.shape, np.int32, runtime, allocations)
        qwen4_exp_qsa_topk_expand_rows_f32_i64(
            d_scores.ptr,
            d_positions.ptr,
            d_selected.ptr,
            d_counts.ptr,
            positions.size,
            stride,
            output_shape[1],
            ratio,
            budget,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_selected, output_shape, np.int64, runtime)
        counts = _download(d_counts, positions.shape, np.int32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    for row, values in enumerate(expected):
        assert int(counts[row]) == values.size
        np.testing.assert_array_equal(actual[row, : values.size], values)
        np.testing.assert_array_equal(actual[row, values.size :], -1)


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
def test_qwen4_exp_qsa_row_transforms_are_exact_to_c1() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_norm_rope_f32,
        qwen4_exp_qsa_norm_rope_rows_f32,
        qwen4_exp_qsa_split_norm_rope_f32,
        qwen4_exp_qsa_split_norm_rope_rows_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(409)
    rows, q_heads, kv_heads, head_dim = 3, 2, 1, 8
    index_heads, index_dim = 2, 8
    q_projected = rng.normal(
        0.0, 0.2, size=(rows, q_heads, 2, head_dim)
    ).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(rows, kv_heads, head_dim)).astype(np.float32)
    q_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    index = rng.normal(0.0, 0.2, size=(rows, index_heads, index_dim)).astype(np.float32)
    index_weight = rng.normal(1.0, 0.05, size=index_dim).astype(np.float32)
    positions = np.array([3, 7, 11], dtype=np.int64)

    allocations = []
    try:
        d_q = _upload(q_projected, runtime, allocations)
        d_k = _upload(key, runtime, allocations)
        d_q_weight = _upload(q_weight, runtime, allocations)
        d_k_weight = _upload(k_weight, runtime, allocations)
        d_index = _upload(index, runtime, allocations)
        d_index_weight = _upload(index_weight, runtime, allocations)
        d_positions = _upload(positions, runtime, allocations)
        d_serial_query = _alloc((rows, q_heads, head_dim), np.float32, runtime, allocations)
        d_serial_key = _alloc((rows, kv_heads, head_dim), np.float32, runtime, allocations)
        d_serial_gate = _alloc((rows, q_heads, head_dim), np.float32, runtime, allocations)
        d_bulk_query = _alloc((rows, q_heads, head_dim), np.float32, runtime, allocations)
        d_bulk_key = _alloc((rows, kv_heads, head_dim), np.float32, runtime, allocations)
        d_bulk_gate = _alloc((rows, q_heads, head_dim), np.float32, runtime, allocations)
        d_serial_index = _alloc(index.shape, np.float32, runtime, allocations)
        d_bulk_index = _alloc(index.shape, np.float32, runtime, allocations)
        for row in range(rows):
            qwen4_exp_qsa_split_norm_rope_f32(
                d_q.ptr + row * q_heads * 2 * head_dim * 4,
                d_k.ptr + row * kv_heads * head_dim * 4,
                d_q_weight.ptr,
                d_k_weight.ptr,
                d_positions.ptr + row * 8,
                d_serial_query.ptr + row * q_heads * head_dim * 4,
                d_serial_key.ptr + row * kv_heads * head_dim * 4,
                d_serial_gate.ptr + row * q_heads * head_dim * 4,
                q_heads,
                kv_heads,
                head_dim,
                4,
                100.0,
                runtime=runtime,
            )
            qwen4_exp_qsa_norm_rope_f32(
                d_index.ptr + row * index_heads * index_dim * 4,
                d_index_weight.ptr,
                d_positions.ptr + row * 8,
                d_serial_index.ptr + row * index_heads * index_dim * 4,
                index_heads,
                index_dim,
                4,
                100.0,
                runtime=runtime,
            )
        qwen4_exp_qsa_split_norm_rope_rows_f32(
            d_q.ptr,
            d_k.ptr,
            d_q_weight.ptr,
            d_k_weight.ptr,
            d_positions.ptr,
            d_bulk_query.ptr,
            d_bulk_key.ptr,
            d_bulk_gate.ptr,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            4,
            100.0,
            runtime=runtime,
        )
        qwen4_exp_qsa_norm_rope_rows_f32(
            d_index.ptr,
            d_index_weight.ptr,
            d_positions.ptr,
            d_bulk_index.ptr,
            rows,
            index_heads,
            index_dim,
            4,
            100.0,
            runtime=runtime,
        )
        runtime.device_synchronize()
        serial_query = _download(
            d_serial_query, (rows, q_heads, head_dim), np.float32, runtime
        )
        serial_key = _download(
            d_serial_key, (rows, kv_heads, head_dim), np.float32, runtime
        )
        serial_gate = _download(
            d_serial_gate, (rows, q_heads, head_dim), np.float32, runtime
        )
        serial_index = _download(d_serial_index, index.shape, np.float32, runtime)
        bulk_query = _download(
            d_bulk_query, (rows, q_heads, head_dim), np.float32, runtime
        )
        bulk_key = _download(d_bulk_key, (rows, kv_heads, head_dim), np.float32, runtime)
        bulk_gate = _download(
            d_bulk_gate, (rows, q_heads, head_dim), np.float32, runtime
        )
        bulk_index = _download(d_bulk_index, index.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(bulk_query, serial_query)
    np.testing.assert_array_equal(bulk_key, serial_key)
    np.testing.assert_array_equal(bulk_gate, serial_gate)
    np.testing.assert_array_equal(bulk_index, serial_index)


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
