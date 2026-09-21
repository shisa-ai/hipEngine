"""The 32-row M tile must be bit-identical to the 16-row owner.

The routed-MoE gate/up kernel is bound by expert-weight traffic: cost is
``sum over experts of ceil(rows_in_expert / 16)`` WMMA tiles x 1.843 MB per
launch, measured at a constant 104-126 GB/s
(``benchmarks/results/2026-09-22-moe-main-kernel-cost/``). A 32-row tile reads
each expert's weights once for 32 rows instead of once for 16, so it halves
that traffic whenever an expert holds 16 or fewer rows per tile.

The tile changes only *which rows share a weight read*. Each output row keeps
its own K-loop order, iu8 residual planes, Kahan accumulation and risk bound,
so the contract is exact parent parity between the two tile heights - not a
numerical envelope. These tests hold the activations, weights and expert
partition fixed and vary only the tile height, so any difference is a tiling
bug and not arithmetic drift.

The 32-row tile map is the existing ``qwen35_moe_mmq32_tile_map`` sibling of the
16-row ``qwen35_moe_wmma_tile_map``; the kernels take the map's
``expert_start_wmma``/``tile_expert``/``wmma_total_rows`` directly.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.kernels.registry import resolve
from tests.test_gpu_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture

J32_VARIANT = "selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out"
J16_VARIANT = "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
REPAIR_VARIANT = "selected_dual_sparse_exact_repair_bf16"

# Uneven experts, an empty expert, 16-row tails, and counts that straddle the
# 32-row boundary so one expert needs two 32-row tiles and another exactly one.
_TILE_CASES = [
    ([17, 0, 64, 3, 129, 32], 256, 128, 128),
    ([1, 16, 31, 32, 33, 64], 512, 128, 128),
    ([20, 20, 20, 20], 2560, 128, 128),
    ([5, 8, 13, 21, 34, 55], 2560, 128, 128),
]


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _tile_map(counts: list[int], tile_rows: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the padded expert starts, tile->expert map and padded row count."""

    padded = [((count + tile_rows - 1) // tile_rows) * tile_rows for count in counts]
    expert_start = np.zeros(len(counts) + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(np.asarray(padded, dtype=np.int64))
    tile_expert = np.asarray(
        [
            expert
            for expert, rows in enumerate(padded)
            for _ in range(rows // tile_rows)
        ],
        dtype=np.int64,
    )
    total = int(expert_start[-1])
    assert tile_expert.size == total // tile_rows
    return expert_start, tile_expert, total


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(count: int, dtype, runtime, allocations):
    device = malloc(int(count) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _run_tile(
    fixture,
    runtime,
    library,
    *,
    tile_rows: int,
    risk_multiplier: float,
    allocations: list,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run one tile height with its own tile map; return (out, queued, count)."""

    counts = np.diff(fixture.expert_start_compact).tolist()
    expert_start_wmma, tile_expert, wmma_total_rows = _tile_map(counts, tile_rows)

    dx = _upload(fixture.x_host, runtime, allocations)
    d_start = _upload(fixture.expert_start_compact, runtime, allocations)
    d_wmma = _upload(expert_start_wmma, runtime, allocations)
    d_tile = _upload(tile_expert, runtime, allocations)
    d_qa = _upload(fixture.qweight_a, runtime, allocations)
    d_qb = _upload(fixture.qweight_b, runtime, allocations)
    total_out = fixture.out_features_a + fixture.out_features_b
    d_out = _alloc(fixture.compact_rows * total_out, np.uint16, runtime, allocations)
    capacity = fixture.compact_rows * total_out
    d_count = _alloc(1, np.int32, runtime, allocations)
    d_indices = _alloc(capacity, np.int32, runtime, allocations)
    copy_host_to_device(
        d_count, host_array_ptr(np.zeros(1, dtype=np.int32)), runtime=runtime
    )

    launcher = (
        q4.gguf_q4_k_selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out
        if tile_rows == 32
        else q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out
    )
    launcher(
        dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr, d_qb.ptr,
        d_out.ptr, d_count.ptr, d_indices.ptr, capacity, risk_multiplier,
        fixture.compact_rows, fixture.in_features, fixture.out_features_a,
        fixture.out_features_b, fixture.num_experts, wmma_total_rows,
        library=library, runtime=runtime,
    )
    count = int(_download(d_count, (1,), np.int32, runtime)[0])
    assert count <= capacity, f"risk count {count} exceeds capacity {capacity}"
    queued = _download(d_indices, (capacity,), np.int32, runtime)[:count]
    out = _download(d_out, (fixture.compact_rows, total_out), np.uint16, runtime)
    return out, queued, count


def test_registry_binds_both_tile_heights() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    for variant in (J16_VARIANT, J32_VARIANT):
        assert resolve(
            backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
            variant=variant,
        ) is not None
    assert resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant=J16_VARIANT,
    ) is getattr(q4, "gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
    assert resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant=J32_VARIANT,
    ) is getattr(q4, "gguf_q4_k_selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out")


def test_tile_rows_must_divide_the_padded_row_count() -> None:
    """A 16-aligned row count at a 32-row tile would silently drop rows."""

    with pytest.raises(ValueError, match="multiple of tile_rows"):
        q4.gguf_q4_k_selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4.0, 16, 256, 128, 128, 4, 16,
            library=object(), runtime=object(),
        )
    with pytest.raises(ValueError, match="tile_rows must be 16 or 32"):
        q4._launch_wmma_iu8_risk(
            3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 4.0, 16, 256, 128, 128, 4, 64,
            stream=0, library=object(), runtime=object(), tile_rows=8,
        )


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("counts,in_features,out_a,out_b", _TILE_CASES)
@pytest.mark.parametrize("multiplier", [4.0, 1e30])
def test_j32_tile_matches_the_16_row_owner_bit_for_bit(
    counts, in_features, out_a, out_b, multiplier
) -> None:
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_a,
        out_features_b=out_b,
        dtype="bf16",
        seed=23,
    )
    allocations: list = []
    try:
        out16, queued16, count16 = _run_tile(
            fixture, runtime, library, tile_rows=16,
            risk_multiplier=multiplier, allocations=allocations,
        )
        out32, queued32, count32 = _run_tile(
            fixture, runtime, library, tile_rows=32,
            risk_multiplier=multiplier, allocations=allocations,
        )
    finally:
        for device in allocations:
            free(device)

    np.testing.assert_array_equal(
        out32, out16, err_msg=f"32-row tile changed the output at multiplier {multiplier}"
    )
    # 1e30 queues every element, so the queues must agree exactly; at 4.0 they
    # must agree as sets, because the queue is published in output-index order
    # within a block and the two tile heights partition the work differently.
    assert count32 == count16
    np.testing.assert_array_equal(
        np.sort(queued32), np.sort(queued16),
        err_msg=f"32-row tile changed which elements are queued at {multiplier}",
    )


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_j32_tile_survives_the_exact_repair_chain() -> None:
    """The published arithmetic after the repair pass is unchanged too."""

    counts, in_features, out_a, out_b = [17, 0, 64, 3, 129, 32], 256, 128, 128
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=counts, in_features=in_features, out_features_a=out_a,
        out_features_b=out_b, dtype="bf16", seed=23,
    )
    allocations: list = []
    try:
        results = []
        for tile_rows in (16, 32):
            out, _queued, count = _run_tile(
                fixture, runtime, library, tile_rows=tile_rows,
                risk_multiplier=4.0, allocations=allocations,
            )
            counts_arr = np.diff(fixture.expert_start_compact).tolist()
            expert_start_wmma, tile_expert, wmma_total_rows = _tile_map(
                counts_arr, tile_rows
            )
            dx = _upload(fixture.x_host, runtime, allocations)
            d_start = _upload(fixture.expert_start_compact, runtime, allocations)
            d_qa = _upload(fixture.qweight_a, runtime, allocations)
            d_qb = _upload(fixture.qweight_b, runtime, allocations)
            total_out = out_a + out_b
            d_out = _alloc(fixture.compact_rows * total_out, np.uint16, runtime, allocations)
            copy_host_to_device(
                d_out, host_array_ptr(np.ascontiguousarray(out)), runtime=runtime
            )
            d_count = _alloc(1, np.int32, runtime, allocations)
            d_indices = _alloc(max(count, 1), np.int32, runtime, allocations)
            copy_host_to_device(
                d_count, host_array_ptr(np.asarray([count], dtype=np.int32)),
                runtime=runtime,
            )
            # Re-run the risk prefill to fill the index buffer, then repair.
            d_tile = _upload(tile_expert, runtime, allocations)
            d_wmma = _upload(expert_start_wmma, runtime, allocations)
            d_out2 = _alloc(fixture.compact_rows * total_out, np.uint16, runtime, allocations)
            launcher = (
                q4.gguf_q4_k_selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out
                if tile_rows == 32
                else q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out
            )
            launcher(
                dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr, d_qb.ptr,
                d_out2.ptr, d_count.ptr, d_indices.ptr, fixture.compact_rows * total_out,
                4.0, fixture.compact_rows, in_features, out_a, out_b,
                fixture.num_experts, wmma_total_rows, library=library, runtime=runtime,
            )
            q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                dx.ptr, d_start.ptr, d_qa.ptr, d_qb.ptr, d_out2.ptr,
                d_count.ptr, d_indices.ptr, fixture.compact_rows * total_out,
                fixture.compact_rows, in_features, out_a, out_b,
                fixture.num_experts, library=library, runtime=runtime,
            )
            results.append(
                _download(d_out2, (fixture.compact_rows, total_out), np.uint16, runtime)
            )
    finally:
        for device in allocations:
            free(device)

    np.testing.assert_array_equal(
        results[1], results[0],
        err_msg="repaired output differs between the 16-row and 32-row tiles",
    )


# --------------------------------------------------------------------------
# The down kernel (Q5_1) is the second half of the routed-MoE main pair and has
# the same traffic structure, so it takes the same tile height.
# --------------------------------------------------------------------------

DOWN_J16 = "qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out"
DOWN_J32 = "qwen4_exp_q5_1_selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out"


def _down_fixture(counts, in_features, out_features, seed=23):
    from tests.test_gpu_qwen4exp_q51_iu8_exact import Fixture as _Q51Fixture

    return _Q51Fixture(
        counts=counts, in_features=in_features, out_features=out_features, seed=seed
    )


def _run_down_tile(
    fixture, runtime, library, *, tile_rows: int, risk_multiplier: float,
    allocations: list,
):
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51

    counts = fixture.counts.tolist()
    expert_start_wmma, tile_expert, wmma_total_rows = _tile_map(counts, tile_rows)
    dx = _upload(fixture.x_host, runtime, allocations)
    ds = _upload(fixture.expert_start, runtime, allocations)
    dws = _upload(expert_start_wmma, runtime, allocations)
    dte = _upload(tile_expert, runtime, allocations)
    dw = _upload(fixture.qweight, runtime, allocations)
    total = fixture.compact_rows * fixture.out_features
    d_out = _alloc(total, np.uint16, runtime, allocations)
    d_count = _alloc(1, np.int32, runtime, allocations)
    d_indices = _alloc(total, np.int32, runtime, allocations)
    copy_host_to_device(
        d_count, host_array_ptr(np.zeros(1, dtype=np.int32)), runtime=runtime
    )
    launcher = (
        q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out
        if tile_rows == 32
        else q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out
    )
    launcher(
        dx.ptr, ds.ptr, dws.ptr, dte.ptr, dw.ptr, d_out.ptr,
        d_count.ptr, d_indices.ptr, total, risk_multiplier,
        fixture.compact_rows, fixture.in_features, fixture.out_features,
        fixture.num_experts, wmma_total_rows, library=library, runtime=runtime,
    )
    count = int(_download(d_count, (1,), np.int32, runtime)[0])
    assert count <= total
    queued = _download(d_indices, (total,), np.int32, runtime)[:count]
    out = _download(
        d_out, (fixture.compact_rows, fixture.out_features), np.uint16, runtime
    )
    return out, queued, count


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "counts,in_features,out_features",
    [
        ([17, 0, 64, 3, 129, 32], 256, 128),
        ([1, 16, 31, 32, 33, 64], 512, 128),
        ([20, 20, 20, 20], 2560, 128),
    ],
)
@pytest.mark.parametrize("multiplier", [4.0, 1e30])
def test_down_j32_tile_matches_the_16_row_owner_bit_for_bit(
    counts, in_features, out_features, multiplier
) -> None:
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51

    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = _down_fixture(counts, in_features, out_features)
    allocations: list = []
    try:
        out16, queued16, count16 = _run_down_tile(
            fixture, runtime, library, tile_rows=16,
            risk_multiplier=multiplier, allocations=allocations,
        )
        out32, queued32, count32 = _run_down_tile(
            fixture, runtime, library, tile_rows=32,
            risk_multiplier=multiplier, allocations=allocations,
        )
    finally:
        for device in allocations:
            free(device)

    np.testing.assert_array_equal(
        out32, out16, err_msg="32-row down tile changed the output"
    )
    assert count32 == count16
    np.testing.assert_array_equal(np.sort(queued32), np.sort(queued16))


def test_down_registry_binds_both_tile_heights() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51

    register_gfx1151_kernels(replace=True)
    for variant, name in ((DOWN_J16, "selected_wmma_iu8_risk_prefill_bf16_bf16_out"),
                          (DOWN_J32, "selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out")):
        assert resolve(
            backend="hip_gfx1151", layer="moe_linear", quant="gguf_q5_1",
            variant=name,
        ) is getattr(q51, variant)
