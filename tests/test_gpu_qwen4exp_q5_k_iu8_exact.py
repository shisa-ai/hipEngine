"""Exactness tests for the Q5_K risk-collecting iu8-WMMA MoE gate/up prefill.

#15 Q5_K bundle KL-shaving variant: the candidate chain is the
production-enveloped iu8-WMMA kernel (3-plane residual int8 activations,
Kahan-compensated risk probe) plus a sparse exact repair that reproduces
the strict row4 parent's exact arithmetic and reduction tree
(``gguf_q5_k_selected_grouped_row4_bf16_kernel``: 128 threads, strided k
ownership, shuffle tree + serial wave sum). Promotion target: bit-identical
BF16 outputs versus the row4 parent, so the route is arithmetic-preserving
and the T2 f16-WMMA drift cannot recur.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out as row4_parent,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
    build_gguf_q5_k_q8_1_selected_prefill,
    gguf_q5_k_selected_dual_sparse_exact_repair_bf16 as sparse_repair,
    gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out as iu8_risk,
    register_gguf_q5_k_q8_1_selected_prefill_kernels,
)
from hipengine.kernels.registry import resolve
from tests.test_gpu_gguf_k_gemv import make_q5_k_weight


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry_resolves_q5_k_iu8_variants() -> None:
    register_gguf_q5_k_q8_1_selected_prefill_kernels(replace=True)
    candidate = resolve(
        backend="hip_gfx1100", layer="moe_linear", quant="gguf_q5_k",
        variant="selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
    )
    repair = resolve(
        backend="hip_gfx1100", layer="moe_linear", quant="gguf_q5_k",
        variant="selected_dual_sparse_exact_repair_bf16",
    )
    assert candidate is iu8_risk
    assert repair is sparse_repair
    # The strict row4 parent stays bound to its production owner.
    assert resolve(
        backend="hip_gfx1100", layer="linear", quant="gguf_q5_k",
        variant="selected_grouped_row4_gemv_bf16_bf16_out",
    ) is row4_parent


class _Fixture:
    pass


def _build_fixture(
    *,
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    seed: int = 0,
):
    num_experts = len(counts)
    compact_rows = int(sum(counts))
    assert compact_rows > 0
    expert_start_compact = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    padded_counts = [((count + 15) // 16) * 16 for count in counts]
    expert_start_wmma = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_wmma[1:] = np.cumsum(np.asarray(padded_counts, dtype=np.int64))
    wmma_total_rows = int(expert_start_wmma[-1])
    tile_expert = np.asarray(
        [e for e, padded in enumerate(padded_counts) for _ in range(padded // 16)],
        dtype=np.int64,
    )
    assert tile_expert.size == wmma_total_rows // 16

    rng = np.random.default_rng(seed)
    x_f32 = rng.standard_normal((compact_rows, in_features)).astype(np.float32)
    x_host = x_f32.view(np.uint16) if False else (
        x_f32.astype(np.float32).view(np.int32) >> 16
    ).astype(np.uint16)  # truncate-to-bf16 (ties never on random data)

    qweight_a = np.ascontiguousarray(
        make_q5_k_weight(out_features_a, in_features)
    )[None, :, :].repeat(num_experts, axis=0)
    qweight_b = np.ascontiguousarray(
        make_q5_k_weight(out_features_b, in_features)
    )[None, :, :].repeat(num_experts, axis=0)

    fixture = _Fixture()
    fixture.counts = counts
    fixture.num_experts = num_experts
    fixture.compact_rows = compact_rows
    fixture.in_features = in_features
    fixture.out_features_a = out_features_a
    fixture.out_features_b = out_features_b
    fixture.x_host = np.ascontiguousarray(x_host)
    fixture.expert_start_compact = expert_start_compact
    fixture.expert_start_wmma = expert_start_wmma
    fixture.tile_expert = tile_expert
    fixture.wmma_total_rows = wmma_total_rows
    fixture.qweight_a = np.ascontiguousarray(qweight_a)
    fixture.qweight_b = np.ascontiguousarray(qweight_b)
    return fixture


class _Allocs:
    def __init__(self):
        self.items = []

    def upload(self, array, runtime):
        host = np.ascontiguousarray(array)
        device = malloc(host.nbytes, runtime=runtime)
        self.items.append(device)
        copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
        return device

    def alloc(self, count, dtype, runtime):
        device = malloc(int(count) * np.dtype(dtype).itemsize, runtime=runtime)
        self.items.append(device)
        return device

    def download(self, device, shape, dtype, runtime):
        host = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
        return host

    def free(self):
        for device in self.items:
            free(device)


def _run_candidate(fixture, runtime, library, *, risk_multiplier, allocs):
    dx = allocs.upload(fixture.x_host, runtime)
    d_start = allocs.upload(fixture.expert_start_compact, runtime)
    d_wmma = allocs.upload(fixture.expert_start_wmma, runtime)
    d_tile = allocs.upload(fixture.tile_expert, runtime)
    d_qa = allocs.upload(fixture.qweight_a, runtime)
    d_qb = allocs.upload(fixture.qweight_b, runtime)
    total = fixture.out_features_a + fixture.out_features_b
    d_out = allocs.alloc(fixture.compact_rows * total, np.uint16, runtime)
    capacity = fixture.compact_rows * total
    d_risk_count = allocs.alloc(1, np.int32, runtime)
    d_risk_indices = allocs.alloc(capacity, np.int32, runtime)
    zero = np.zeros(1, dtype=np.int32)
    copy_host_to_device(d_risk_count, host_array_ptr(zero), runtime=runtime)
    iu8_risk(
        dx.ptr, d_start.ptr, d_wmma.ptr, d_tile.ptr, d_qa.ptr, d_qb.ptr,
        d_out.ptr, d_risk_count.ptr, d_risk_indices.ptr, capacity,
        risk_multiplier, fixture.compact_rows, fixture.in_features,
        fixture.out_features_a, fixture.out_features_b, fixture.num_experts,
        fixture.wmma_total_rows, library=library, runtime=runtime,
    )
    sparse_repair(
        dx.ptr, d_start.ptr, d_qa.ptr, d_qb.ptr, d_out.ptr,
        d_risk_count.ptr, d_risk_indices.ptr, capacity, fixture.compact_rows,
        fixture.in_features, fixture.out_features_a, fixture.out_features_b,
        fixture.num_experts, library=library, runtime=runtime,
    )
    count = int(allocs.download(d_risk_count, (1,), np.int32, runtime)[0])
    if count > capacity:
        raise AssertionError(f"risk count {count} exceeds capacity {capacity}")
    out = allocs.download(d_out, (fixture.compact_rows, total), np.uint16, runtime)
    return out, count


def _run_parent(fixture, runtime, library, allocs):
    dx = allocs.upload(fixture.x_host, runtime)
    d_start = allocs.upload(fixture.expert_start_compact, runtime)
    outs = []
    for out_features, qweight in (
        (fixture.out_features_a, fixture.qweight_a),
        (fixture.out_features_b, fixture.qweight_b),
    ):
        d_qw = allocs.upload(qweight, runtime)
        d_out = allocs.alloc(fixture.compact_rows * out_features, np.uint16, runtime)
        row4_parent(
            dx.ptr, d_start.ptr, None, d_qw.ptr, d_out.ptr,
            fixture.compact_rows, fixture.compact_rows, fixture.num_experts,
            fixture.in_features, out_features,
            library=None, runtime=runtime,
        )
        outs.append(
            allocs.download(d_out, (fixture.compact_rows, out_features), np.uint16, runtime)
        )
    return np.concatenate(outs, axis=1)


_FIXTURE_CASES = [
    # uneven counts, an empty expert, and tile-boundary row counts.
    ([17, 0, 64, 3, 129, 32], 256, 128, 128),
    ([1, 16, 31, 32, 33, 64], 512, 128, 128),
    ([5, 8, 13, 21, 34, 55], 768, 128, 128),
    ([256, 1, 0, 0, 0, 0], 256, 128, 128),
]


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("counts,in_features,out_a,out_b", _FIXTURE_CASES)
def test_candidate_chain_matches_row4_parent_exactly(
    counts, in_features, out_a, out_b
) -> None:
    """iu8-WMMA + sparse exact repair == strict row4 parent, bit for bit."""

    register_gguf_q5_k_q8_1_selected_prefill_kernels(replace=True)
    runtime = get_hip_runtime()
    library = build_gguf_q5_k_q8_1_selected_prefill(load=True)
    fixture = _build_fixture(
        counts=counts, in_features=in_features,
        out_features_a=out_a, out_features_b=out_b,
    )
    candidate_allocs = _Allocs()
    parent_allocs = _Allocs()
    try:
        candidate, risk_count = _run_candidate(
            fixture, runtime, library, risk_multiplier=16.0, allocs=candidate_allocs
        )
        parent = _run_parent(fixture, runtime, library, parent_allocs)
    finally:
        candidate_allocs.free()
        parent_allocs.free()
    assert candidate.shape == parent.shape
    mismatches = np.count_nonzero(candidate != parent)
    assert mismatches == 0, (
        f"{mismatches} of {candidate.size} BF16 outputs differ from the "
        f"row4 parent (risk count {risk_count})"
    )


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("counts,in_features,out_a,out_b", _FIXTURE_CASES[:1])
def test_huge_risk_multiplier_forces_full_repair(
    counts, in_features, out_a, out_b
) -> None:
    """At a huge multiplier every output is queued; the chain still
    matches the parent exactly (repair dominates)."""

    register_gguf_q5_k_q8_1_selected_prefill_kernels(replace=True)
    runtime = get_hip_runtime()
    library = build_gguf_q5_k_q8_1_selected_prefill(load=True)
    fixture = _build_fixture(
        counts=counts, in_features=in_features,
        out_features_a=out_a, out_features_b=out_b,
    )
    candidate_allocs = _Allocs()
    parent_allocs = _Allocs()
    try:
        candidate, risk_count = _run_candidate(
            fixture, runtime, library, risk_multiplier=1.0e30, allocs=candidate_allocs
        )
        parent = _run_parent(fixture, runtime, library, parent_allocs)
    finally:
        candidate_allocs.free()
        parent_allocs.free()
    assert np.count_nonzero(candidate != parent) == 0
    assert risk_count > 0
