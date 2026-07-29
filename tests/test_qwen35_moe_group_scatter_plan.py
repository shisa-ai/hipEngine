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

from hipengine.kernels.hip_gfx1100.moe import (
    plan_qwen35_moe_group_scatter_build,
    qwen35_moe_gather_packed_hidden_lowp,
    qwen35_moe_group_compact_active,
    qwen35_moe_group_compact_active_parallel,
    qwen35_moe_group_compact_active_source_rows,
    qwen35_moe_group_compact_active_source_rows_parallel,
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_prefix_active,
    qwen35_moe_group_scatter,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_prefill_grouped_compact,
    qwen35_moe_prefill_selected_c1_rows,
    qwen35_moe_mmq32_tile_map,
    qwen35_moe_mmq64_tile_map,
    qwen35_moe_mmq128_tile_map,
    qwen35_moe_wmma_tile_map,
    register_qwen35_moe_group_scatter_kernels,
    register_qwen35_moe_prefill_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_moe_group_scatter_registers_prefill_metadata_variants() -> None:
    register_qwen35_moe_group_scatter_kernels()
    register_qwen35_moe_prefill_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_count",
            quant="w4_paro",
            variant="qwen35",
        )
        is qwen35_moe_group_count
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_prefix",
            quant="w4_paro",
            variant="qwen35",
        )
        is qwen35_moe_group_prefix
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_compact",
            quant="generic",
            variant="active_experts",
        )
        is qwen35_moe_group_compact_active
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_compact",
            quant="generic",
            variant="active_experts_source_rows",
        )
        is qwen35_moe_group_compact_active_source_rows
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_compact",
            quant="generic",
            variant="active_experts_parallel",
        )
        is qwen35_moe_group_compact_active_parallel
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_compact",
            quant="generic",
            variant="active_experts_source_rows_parallel",
        )
        is qwen35_moe_group_compact_active_source_rows_parallel
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_prefix",
            quant="generic",
            variant="active_experts",
        )
        is qwen35_moe_group_prefix_active
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_wmma_tile_map",
            quant="w4_paro",
            variant="qwen35",
        )
        is qwen35_moe_wmma_tile_map
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_mmq_tile_map",
            quant="generic",
            variant="tile32",
        )
        is qwen35_moe_mmq32_tile_map
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_mmq_tile_map",
            quant="generic",
            variant="tile64",
        )
        is qwen35_moe_mmq64_tile_map
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_mmq_tile_map",
            quant="generic",
            variant="tile128",
        )
        is qwen35_moe_mmq128_tile_map
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_wmma_tile_map",
            quant="generic",
            variant="tile16",
        )
        is qwen35_moe_wmma_tile_map
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_scatter",
            quant="w4_paro",
            variant="qwen35",
        )
        is qwen35_moe_group_scatter
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_scatter_gather",
            quant="w4_paro",
            variant="qwen35_lowp",
        )
        is qwen35_moe_group_scatter_gather_lowp
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_gather_packed_hidden",
            quant="w4_paro",
            variant="qwen35_lowp",
        )
        is qwen35_moe_gather_packed_hidden_lowp
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_prefill",
            quant="w4_paro",
            variant="qwen35_grouped_compact",
        )
        is qwen35_moe_prefill_grouped_compact
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_prefill",
            quant="w4_paro",
            variant="qwen35_selected_c1_rows",
        )
        is qwen35_moe_prefill_selected_c1_rows
    )


def test_qwen35_moe_group_scatter_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_moe_group_scatter_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 moe group scatter test version",
    )

    assert artifact.family == "qwen35_moe_group_scatter"
    assert artifact.profile.name == "prefill"
    assert artifact.profile.wavefront == 32
    assert artifact.output_path.name == "qwen35_moe_group_scatter.so"
    assert artifact.compiler_version == "hipcc qwen35 moe group scatter test version"
    assert any(str(path).endswith("group_scatter.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_moe_group_scatter_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="total_lanes"):
        qwen35_moe_group_count(0, 0, 0, 1)
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_group_prefix(0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="pad_multiple"):
        qwen35_moe_group_prefix(0, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_group_prefix_active(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="total_lanes"):
        qwen35_moe_group_compact_active(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="top_k"):
        qwen35_moe_group_compact_active_source_rows(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0
        )
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_wmma_tile_map(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="tile_capacity"):
        qwen35_moe_wmma_tile_map(0, 0, 0, 0, 1, tile_capacity=-1)
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_mmq32_tile_map(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="tile_capacity"):
        qwen35_moe_mmq128_tile_map(0, 0, 0, 0, 1, tile_capacity=-1)
    with pytest.raises(ValueError, match="total_lanes"):
        qwen35_moe_group_scatter(0, 0, 0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="top_k"):
        qwen35_moe_group_scatter_gather_lowp(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4)
    with pytest.raises(ValueError, match="total_elements"):
        qwen35_moe_gather_packed_hidden_lowp(0, 0, 0, 0, 1, 1, 4)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("parallel", [False, True])
def test_group_compact_active_matches_stable_cpu_oracle(parallel: bool) -> None:
    selected = np.asarray([2, 1, 4, 1, 2, 4, 4, 1], dtype=np.int64)
    weights = np.asarray([0.2, 0.1, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    expected_starts = np.asarray([0, 0, 3, 5, 5, 8], dtype=np.int64)
    expected_active = np.asarray([1, 2, 4], dtype=np.int64)
    expected_lanes = np.asarray([1, 3, 7, 0, 4, 2, 5, 6], dtype=np.int64)
    buffers = []
    try:
        selected_buffer = malloc(selected.nbytes)
        weights_buffer = malloc(weights.nbytes)
        buffers.extend((selected_buffer, weights_buffer))
        copy_host_to_device(selected_buffer, host_array_ptr(selected), selected.nbytes)
        copy_host_to_device(weights_buffer, host_array_ptr(weights), weights.nbytes)
        starts_buffer = malloc(expected_starts.nbytes)
        active_buffer = malloc(5 * np.dtype(np.int64).itemsize)
        active_count_buffer = malloc(np.dtype(np.int64).itemsize)
        lanes_buffer = malloc(selected.nbytes)
        experts_buffer = malloc(selected.nbytes)
        sorted_weights_buffer = malloc(weights.nbytes)
        buffers.extend(
            (
                starts_buffer,
                active_buffer,
                active_count_buffer,
                lanes_buffer,
                experts_buffer,
                sorted_weights_buffer,
            )
        )

        qwen35_moe_group_compact_active(
            selected_buffer.ptr,
            weights_buffer.ptr,
            starts_buffer.ptr,
            active_buffer.ptr,
            active_count_buffer.ptr,
            lanes_buffer.ptr,
            experts_buffer.ptr,
            sorted_weights_buffer.ptr,
            selected.size,
            5,
            **({"parallel": True} if parallel else {}),
        )
        starts = np.empty_like(expected_starts)
        active = np.empty(5, dtype=np.int64)
        active_count = np.empty(1, dtype=np.int64)
        lanes = np.empty_like(selected)
        experts = np.empty_like(selected)
        sorted_weights = np.empty_like(weights)
        for array, buffer in (
            (starts, starts_buffer),
            (active, active_buffer),
            (active_count, active_count_buffer),
            (lanes, lanes_buffer),
            (experts, experts_buffer),
            (sorted_weights, sorted_weights_buffer),
        ):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
        np.testing.assert_array_equal(starts, expected_starts)
        assert active_count.tolist() == [3]
        np.testing.assert_array_equal(active[:3], expected_active)
        np.testing.assert_array_equal(lanes, expected_lanes)
        np.testing.assert_array_equal(experts, selected[expected_lanes])
        np.testing.assert_array_equal(sorted_weights, weights[expected_lanes])
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("parallel", [False, True])
def test_group_compact_source_rows_and_mmq32_tile_map_match_cpu_oracle(
    parallel: bool,
) -> None:
    selected = np.asarray([2, 1, 4, 1, 2, 4, 4, 1], dtype=np.int64)
    weights = np.asarray([0.2, 0.1, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    expected_starts = np.asarray([0, 0, 3, 5, 5, 8], dtype=np.int64)
    expected_starts32 = np.asarray([0, 0, 32, 64, 64, 96], dtype=np.int64)
    expected_starts64 = np.asarray([0, 0, 64, 128, 128, 192], dtype=np.int64)
    expected_starts128 = np.asarray([0, 0, 128, 256, 256, 384], dtype=np.int64)
    expected_active = np.asarray([1, 2, 4], dtype=np.int64)
    expected_lanes = np.asarray([1, 3, 7, 0, 4, 2, 5, 6], dtype=np.int64)
    expected_sources = expected_lanes // 2
    expected_tiles = np.asarray([1, 2, 4, -1, -1], dtype=np.int64)
    buffers = []
    try:
        selected_buffer = malloc(selected.nbytes)
        weights_buffer = malloc(weights.nbytes)
        buffers.extend((selected_buffer, weights_buffer))
        copy_host_to_device(selected_buffer, host_array_ptr(selected), selected.nbytes)
        copy_host_to_device(weights_buffer, host_array_ptr(weights), weights.nbytes)
        starts_buffer = malloc(expected_starts.nbytes)
        starts32_buffer = malloc(expected_starts32.nbytes)
        starts64_buffer = malloc(expected_starts64.nbytes)
        starts128_buffer = malloc(expected_starts128.nbytes)
        active_buffer = malloc(5 * np.dtype(np.int64).itemsize)
        active_count_buffer = malloc(np.dtype(np.int64).itemsize)
        lanes_buffer = malloc(selected.nbytes)
        source_rows_buffer = malloc(selected.nbytes)
        sorted_weights_buffer = malloc(weights.nbytes)
        tiles_buffer = malloc(expected_tiles.nbytes)
        total32_buffer = malloc(np.dtype(np.int64).itemsize)
        total64_buffer = malloc(np.dtype(np.int64).itemsize)
        total128_buffer = malloc(np.dtype(np.int64).itemsize)
        buffers.extend(
            (
                starts_buffer,
                starts32_buffer,
                starts64_buffer,
                starts128_buffer,
                active_buffer,
                active_count_buffer,
                lanes_buffer,
                source_rows_buffer,
                sorted_weights_buffer,
                tiles_buffer,
                total32_buffer,
                total64_buffer,
                total128_buffer,
            )
        )

        qwen35_moe_group_compact_active_source_rows(
            selected_buffer.ptr,
            weights_buffer.ptr,
            starts_buffer.ptr,
            active_buffer.ptr,
            active_count_buffer.ptr,
            lanes_buffer.ptr,
            source_rows_buffer.ptr,
            sorted_weights_buffer.ptr,
            selected.size,
            5,
            2,
            **({"parallel": True} if parallel else {}),
        )
        qwen35_moe_mmq32_tile_map(
            starts_buffer.ptr,
            starts32_buffer.ptr,
            tiles_buffer.ptr,
            total32_buffer.ptr,
            5,
            tile_capacity=expected_tiles.size,
        )
        qwen35_moe_mmq64_tile_map(
            starts_buffer.ptr,
            starts64_buffer.ptr,
            tiles_buffer.ptr,
            total64_buffer.ptr,
            5,
            tile_capacity=expected_tiles.size,
        )
        qwen35_moe_mmq128_tile_map(
            starts_buffer.ptr,
            starts128_buffer.ptr,
            tiles_buffer.ptr,
            total128_buffer.ptr,
            5,
            tile_capacity=expected_tiles.size,
        )

        starts = np.empty_like(expected_starts)
        starts32 = np.empty_like(expected_starts32)
        starts64 = np.empty_like(expected_starts64)
        starts128 = np.empty_like(expected_starts128)
        active = np.empty(5, dtype=np.int64)
        active_count = np.empty(1, dtype=np.int64)
        lanes = np.empty_like(selected)
        source_rows = np.empty_like(selected)
        sorted_weights = np.empty_like(weights)
        tiles = np.empty_like(expected_tiles)
        total32 = np.empty(1, dtype=np.int64)
        total64 = np.empty(1, dtype=np.int64)
        total128 = np.empty(1, dtype=np.int64)
        for array, buffer in (
            (starts, starts_buffer),
            (starts32, starts32_buffer),
            (starts64, starts64_buffer),
            (starts128, starts128_buffer),
            (active, active_buffer),
            (active_count, active_count_buffer),
            (lanes, lanes_buffer),
            (source_rows, source_rows_buffer),
            (sorted_weights, sorted_weights_buffer),
            (tiles, tiles_buffer),
            (total32, total32_buffer),
            (total64, total64_buffer),
            (total128, total128_buffer),
        ):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(starts32, expected_starts32)
        np.testing.assert_array_equal(starts64, expected_starts64)
        np.testing.assert_array_equal(starts128, expected_starts128)
        assert active_count.tolist() == [3]
        np.testing.assert_array_equal(active[:3], expected_active)
        np.testing.assert_array_equal(lanes, expected_lanes)
        np.testing.assert_array_equal(source_rows, expected_sources)
        np.testing.assert_array_equal(sorted_weights, weights[expected_lanes])
        np.testing.assert_array_equal(tiles, expected_tiles)
        assert total32.tolist() == [96]
        assert total64.tolist() == [192]
        assert total128.tolist() == [384]
    finally:
        for buffer in reversed(buffers):
            free(buffer)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_group_prefix_active_matches_compact_cpu_oracle() -> None:
    counts = np.asarray([0, 3, 1, 0, 4], dtype=np.int32)
    expected_starts = np.asarray([0, 0, 3, 4, 4, 8], dtype=np.int64)
    expected_active = np.asarray([1, 2, 4], dtype=np.int64)
    buffers = []
    try:
        counts_buffer = malloc(counts.nbytes)
        buffers.append(counts_buffer)
        copy_host_to_device(
            counts_buffer, host_array_ptr(counts), counts.nbytes
        )
        starts_buffer = malloc(expected_starts.nbytes)
        active_buffer = malloc(counts.size * np.dtype(np.int64).itemsize)
        active_count_buffer = malloc(np.dtype(np.int64).itemsize)
        buffers.extend((starts_buffer, active_buffer, active_count_buffer))

        qwen35_moe_group_prefix_active(
            counts_buffer.ptr,
            starts_buffer.ptr,
            active_buffer.ptr,
            active_count_buffer.ptr,
            counts.size,
        )
        starts = np.empty_like(expected_starts)
        active = np.empty(counts.size, dtype=np.int64)
        active_count = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(starts), starts_buffer, starts.nbytes
        )
        copy_device_to_host(
            host_array_ptr(active), active_buffer, active.nbytes
        )
        copy_device_to_host(
            host_array_ptr(active_count),
            active_count_buffer,
            active_count.nbytes,
        )
        np.testing.assert_array_equal(starts, expected_starts)
        assert active_count.tolist() == [3]
        np.testing.assert_array_equal(active[: active_count[0]], expected_active)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
