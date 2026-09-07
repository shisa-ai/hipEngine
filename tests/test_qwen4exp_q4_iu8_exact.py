"""Exactness tests for the risk-collecting iu8-WMMA MoE gate/up prefill.

The candidate chain is the production iu8-WMMA kernel extended with a
Kahan-compensated accumulation-error probe plus a sparse exact repair
kernel that reproduces the grouped pair2 parent's exact arithmetic and
reduction tree. Promotion target: bit-identical BF16 outputs versus the
exact ``selected_dual_grouped_pair2`` parent, enabling early-layer
admission through the exact-trajectory route.
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
from tests.test_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture

PARENT = "gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out"
CANDIDATE = "gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
REPAIR = "gguf_q4_k_selected_dual_sparse_exact_repair_bf16"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry_resolves_new_variants_and_keeps_incumbents() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    candidate = resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant="selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
    )
    repair = resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant="selected_dual_sparse_exact_repair_bf16",
    )
    assert candidate is getattr(q4, CANDIDATE)
    assert repair is getattr(q4, REPAIR)
    # Incumbents stay bound to the production owners.
    assert resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant="selected_dual_wmma_iu8_prefill_bf16_bf16_out",
    ) is getattr(q4, "gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out")
    assert resolve(
        backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
        variant="selected_dual_grouped_pair2_bf16_bf16_out",
    ) is getattr(q4, PARENT)


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


def _run_chain(
    fixture,
    runtime,
    library,
    *,
    risk_multiplier: float,
    allocations: list,
) -> tuple[np.ndarray, int]:
    """Run candidate risk prefill + sparse exact repair; return (out, risks)."""

    dx = _upload(fixture.x_host, runtime, allocations)
    d_start = _upload(fixture.expert_start_compact, runtime, allocations)
    d_wmma = _upload(fixture.expert_start_wmma, runtime, allocations)
    d_tile = _upload(fixture.tile_expert, runtime, allocations)
    d_qa = _upload(fixture.qweight_a, runtime, allocations)
    d_qb = _upload(fixture.qweight_b, runtime, allocations)
    total = fixture.out_features_a + fixture.out_features_b
    d_out = _alloc(fixture.compact_rows * total, np.uint16, runtime, allocations)
    capacity = fixture.compact_rows * total
    d_risk_count = _alloc(1, np.int32, runtime, allocations)
    d_risk_indices = _alloc(capacity, np.int32, runtime, allocations)
    zero = np.zeros(1, dtype=np.int32)
    copy_host_to_device(d_risk_count, host_array_ptr(zero), runtime=runtime)
    q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
        dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr, d_qb.ptr,
        d_out.ptr, d_risk_count.ptr, d_risk_indices.ptr, capacity,
        risk_multiplier, fixture.compact_rows, fixture.in_features,
        fixture.out_features_a, fixture.out_features_b, fixture.num_experts,
        fixture.wmma_total_rows, library=library, runtime=runtime,
    )
    q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
        dx.ptr, d_start.ptr, d_qa.ptr, d_qb.ptr, d_out.ptr,
        d_risk_count.ptr, d_risk_indices.ptr, capacity, fixture.compact_rows,
        fixture.in_features, fixture.out_features_a, fixture.out_features_b,
        fixture.num_experts, library=library, runtime=runtime,
    )
    count = int(_download(d_risk_count, (1,), np.int32, runtime)[0])
    if count > capacity:
        raise AssertionError(f"risk count {count} exceeds capacity {capacity}")
    out = _download(d_out, (fixture.compact_rows, total), np.uint16, runtime)
    return out, count


def _run_parent(fixture, runtime, library, *, allocations: list) -> np.ndarray:
    dx = _upload(fixture.x_host, runtime, allocations)
    d_start = _upload(fixture.expert_start_compact, runtime, allocations)
    d_qa = _upload(fixture.qweight_a, runtime, allocations)
    d_qb = _upload(fixture.qweight_b, runtime, allocations)
    d_out_a = _alloc(fixture.compact_rows * fixture.out_features_a, np.uint16,
                     runtime, allocations)
    d_out_b = _alloc(fixture.compact_rows * fixture.out_features_b, np.uint16,
                     runtime, allocations)
    q4.gguf_q4_k_selected_dual_grouped_pair2_bf16_bf16_out(
        dx.ptr, d_start.ptr, d_qa.ptr, d_qb.ptr, d_out_a.ptr, d_out_b.ptr,
        fixture.compact_rows, fixture.num_experts, fixture.in_features,
        fixture.out_features_a, library=library, runtime=runtime,
    )
    a = _download(d_out_a, (fixture.compact_rows, fixture.out_features_a),
                  np.uint16, runtime)
    b = _download(d_out_b, (fixture.compact_rows, fixture.out_features_b),
                  np.uint16, runtime)
    return np.concatenate([a, b], axis=1)


_FIXTURE_CASES = [
    # uneven counts, an empty expert, and tile-boundary row counts.
    # out_features per tensor stays <= 128: the shared synthetic Q4_K weight
    # helper wraps its uint8 scale arithmetic for larger widths under
    # numpy 2.x; the production 640-column shape is screened separately on
    # actual model weights.
    ([17, 0, 64, 3, 129, 32], 256, 128, 128),
    ([1, 16, 31, 32, 33, 64], 512, 128, 128),
    ([5, 8, 13, 21, 34, 55], 2560, 128, 128),
]


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("counts,in_features,out_a,out_b", _FIXTURE_CASES)
@pytest.mark.parametrize("multiplier", [32.0, 1e30])
def test_candidate_chain_matches_pair2_exactly(
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
        seed=7,
    )
    allocations: list = []
    try:
        parent = _run_parent(fixture, runtime, library, allocations=allocations)
        candidate, risks = _run_chain(
            fixture, runtime, library,
            risk_multiplier=multiplier, allocations=allocations,
        )
        np.testing.assert_array_equal(
            candidate, parent,
            err_msg="candidate iu8+repair chain must be bit-identical to pair2",
        )
        total = fixture.compact_rows * (out_a + out_b)
        if multiplier >= 1e30:
            assert risks == total, "huge multiplier must queue every output"
        else:
            assert 0 <= risks <= total
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_repair_alone_is_idempotent_and_bounded() -> None:
    """A second repair launch over a stale non-zero counter is a no-op
    once the counter is zeroed by the caller, and invalid indices are
    rejected without touching memory."""

    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=[7, 12, 33],
        in_features=256,
        out_features_a=128,
        out_features_b=128,
        dtype="bf16",
        seed=11,
    )
    allocations: list = []
    try:
        out, risks = _run_chain(
            fixture, runtime, library, risk_multiplier=32.0,
            allocations=allocations,
        )
        assert risks >= 0
        # Re-run the chain; determinism must hold bit-for-bit.
        out2, risks2 = _run_chain(
            fixture, runtime, library, risk_multiplier=32.0,
            allocations=allocations,
        )
        np.testing.assert_array_equal(out, out2)
        assert risks == risks2
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_risk_buffers_validated_before_launch() -> None:
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=[16],
        in_features=256,
        out_features_a=128,
        out_features_b=128,
        dtype="bf16",
        seed=3,
    )
    allocations: list = []
    try:
        dx = _upload(fixture.x_host, runtime, allocations)
        d_start = _upload(fixture.expert_start_compact, runtime, allocations)
        d_wmma = _upload(fixture.expert_start_wmma, runtime, allocations)
        d_tile = _upload(fixture.tile_expert, runtime, allocations)
        d_qa = _upload(fixture.qweight_a, runtime, allocations)
        d_qb = _upload(fixture.qweight_b, runtime, allocations)
        d_out = _alloc(fixture.compact_rows * 256, np.uint16, runtime, allocations)
        with pytest.raises(ValueError):
            q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
                dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr,
                d_qb.ptr, d_out.ptr, 0, 0, 0, 32.0,
                fixture.compact_rows, fixture.in_features,
                fixture.out_features_a, fixture.out_features_b,
                fixture.num_experts, fixture.wmma_total_rows,
                library=library, runtime=runtime,
            )
        with pytest.raises(ValueError):
            q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out(
                dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr,
                d_qb.ptr, d_out.ptr, 4, 0, 0, -1.0,
                fixture.compact_rows, fixture.in_features,
                fixture.out_features_a, fixture.out_features_b,
                fixture.num_experts, fixture.wmma_total_rows,
                library=library, runtime=runtime,
            )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
