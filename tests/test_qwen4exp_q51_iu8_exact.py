"""Exactness tests for the risk-collecting iu8-WMMA Q5_1 MoE down prefill.

The candidate chain is a weight-exact iu8-WMMA kernel (raw 5-bit Q5_1 codes
as the integer operand, three residual activation planes staged in fp32, the
m-offset reconstructed from staged plane sums) plus a Kahan-compensated
accumulation-error probe with row-level at-risk guards and a sparse exact
repair that reproduces the production pair2 row-publish arithmetic.
Promotion target: bit-identical BF16 outputs versus the exact parent.
"""

from __future__ import annotations

import ctypes
import dataclasses

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
from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51
from tests.test_qwen4_exp_pf3_moe_schedules import (
    _alloc,
    _download,
    _make_activation,
    _make_expert_q5_1_weights,
    _upload,
)

PARENT = "qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"
PRODUCTION_MULT = 4.0


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _tile_map(counts: np.ndarray):
    experts = counts.shape[0]
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    tiles_per = (counts + 15) // 16
    total_tiles = int(tiles_per.sum())
    tile_expert = np.full(total_tiles, -1, dtype=np.int64)
    wmma_start = np.zeros(experts + 1, dtype=np.int64)
    tile = 0
    for e in range(experts):
        wmma_start[e] = tile * 16
        if counts[e] > 0:
            tile_expert[tile:tile + tiles_per[e]] = e
            tile += int(tiles_per[e])
    wmma_start[experts] = tile * 16
    return starts, wmma_start, tile * 16, tile_expert


class Fixture:
    def __init__(self, *, counts, in_features, out_features, seed):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.num_experts = len(counts)
        self.compact_rows = int(self.counts.sum())
        self.in_features = in_features
        self.out_features = out_features
        (
            self.expert_start,
            self.expert_start_wmma,
            self.wmma_total_rows,
            self.tile_expert,
        ) = _tile_map(self.counts)
        x, _ = _make_activation(self.compact_rows, in_features, seed=seed)
        self.x_host = x
        self.qweight = _make_expert_q5_1_weights(
            num_experts=self.num_experts,
            out_features=out_features,
            in_features=in_features,
            seed=seed + 1,
        )


def _run_chain(
    fixture, runtime, library, *, risk_multiplier: float, allocations: list,
    x_host=None,
) -> tuple[np.ndarray, int]:
    dx = _upload(x_host if x_host is not None else fixture.x_host,
                 runtime, allocations)
    ds = _upload(fixture.expert_start, runtime, allocations)
    dws = _upload(fixture.expert_start_wmma, runtime, allocations)
    dte = _upload(fixture.tile_expert, runtime, allocations)
    dw = _upload(fixture.qweight, runtime, allocations)
    total = fixture.compact_rows * fixture.out_features
    d_out = _alloc(total, np.uint16, runtime, allocations)
    d_rc = _alloc(1, np.int32, runtime, allocations)
    d_ri = _alloc(total, np.int32, runtime, allocations)
    zero = np.zeros(1, dtype=np.int32)
    copy_host_to_device(d_rc, host_array_ptr(zero), runtime=runtime)
    q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
        dx.ptr, ds.ptr, dws.ptr, dte.ptr, dw.ptr, d_out.ptr,
        d_rc.ptr, d_ri.ptr, total, risk_multiplier, fixture.compact_rows,
        fixture.in_features, fixture.out_features, fixture.num_experts,
        fixture.wmma_total_rows, library=library, runtime=runtime,
    )
    q51.qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
        dx.ptr, ds.ptr, dw.ptr, d_out.ptr, d_rc.ptr, d_ri.ptr, total,
        fixture.compact_rows, fixture.in_features, fixture.out_features,
        fixture.num_experts, library=library, runtime=runtime,
    )
    count = int(_download(d_rc, (1,), np.int32, runtime)[0])
    if count > total:
        raise AssertionError(f"risk count {count} exceeds capacity {total}")
    out = _download(
        d_out, (fixture.compact_rows, fixture.out_features), np.uint16, runtime
    )
    return out, count


def _run_parent(
    fixture, runtime, library, *, allocations: list, x_host=None
) -> np.ndarray:
    dx = _upload(x_host if x_host is not None else fixture.x_host,
                 runtime, allocations)
    ds = _upload(fixture.expert_start, runtime, allocations)
    dw = _upload(fixture.qweight, runtime, allocations)
    d_out = _alloc(
        fixture.compact_rows * fixture.out_features, np.uint16,
        runtime, allocations,
    )
    q51.qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out(
        dx.ptr, ds.ptr, dw.ptr, d_out.ptr, fixture.compact_rows,
        fixture.num_experts, fixture.in_features, fixture.out_features,
        library=library, runtime=runtime,
    )
    return _download(
        d_out, (fixture.compact_rows, fixture.out_features), np.uint16, runtime
    )


_FIXTURE_CASES = [
    # uneven counts, an empty expert, and tile-boundary row counts.
    ([17, 0, 64, 3, 129, 32], 640, 256),
    ([1, 16, 31, 32, 33, 64], 640, 256),
    ([5, 8, 13, 21, 34, 55], 640, 384),
    # heavily skewed experts and tile tails at the production multiplier.
    ([256, 1, 0, 0, 0, 0], 640, 256),
    ([33, 47, 0, 16], 640, 256),
]


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("counts,in_features,out_features", _FIXTURE_CASES)
@pytest.mark.parametrize("multiplier", [PRODUCTION_MULT, 1e30])
def test_candidate_chain_matches_parent_exactly(
    counts, in_features, out_features, multiplier
) -> None:
    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=counts, in_features=in_features, out_features=out_features,
        seed=13,
    )
    allocations: list = []
    try:
        parent = _run_parent(fixture, runtime, library, allocations=allocations)
        candidate, risks = _run_chain(
            fixture, runtime, library, risk_multiplier=multiplier,
            allocations=allocations,
        )
        np.testing.assert_array_equal(
            candidate, parent,
            err_msg="candidate iu8+repair chain must be bit-identical to "
                    "the pair2 row-publish parent",
        )
        total = fixture.compact_rows * out_features
        if multiplier >= 1e30:
            assert risks == total, "huge multiplier must queue every output"
        else:
            assert 0 <= risks <= total
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _adversarial_x(kind: str, rows: int, in_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "cancel":
        base = rng.integers(0, 4, size=(rows, in_features // 2))
        v = (2.0 ** -base).astype(np.float32)
        signs = rng.choice([-1.0, 1.0], size=(rows, in_features // 2))
        x = np.empty((rows, in_features), dtype=np.float32)
        x[:, 0::2] = v * signs
        x[:, 1::2] = -v * signs
        return _bf16_bits(x)
    if kind == "ties":
        return _bf16_bits(
            rng.choice([-1.0, 1.0], size=(rows, in_features)).astype(np.float32)
        )
    if kind == "wide_scale":
        exponents = rng.integers(-14, 14, size=(rows, in_features))
        return _bf16_bits(
            (rng.choice([-1.0, 1.0], size=(rows, in_features))
             * np.float32(2.0) ** exponents).astype(np.float32)
        )
    if kind == "tiny":
        return _bf16_bits(
            (rng.normal(0.0, 1.0, size=(rows, in_features))
             * np.float32(2.0) ** -120).astype(np.float32)
        )
    if kind == "subtiny":
        return _bf16_bits(
            (rng.normal(0.0, 1.0, size=(rows, in_features))
             * np.float32(2.0) ** -128).astype(np.float32)
        )
    if kind == "huge":
        return _bf16_bits(
            (rng.normal(0.0, 1.0, size=(rows, in_features))
             * np.float32(2.0) ** 100).astype(np.float32)
        )
    if kind == "nonfinite_rows":
        x = rng.normal(0.0, 0.5, size=(rows, in_features)).astype(np.float32)
        x[0, :] = np.float32("nan")
        x[1, :] = np.float32("inf")
        x[2, :] = np.float32("-inf")
        x[3, 0] = np.float32("inf")
        x[4, in_features // 2] = np.float32("nan")
        return _bf16_bits(x)
    raise AssertionError(f"unknown adversarial kind {kind}")


_ADVERSARIAL_KINDS = [
    "cancel", "ties", "wide_scale", "tiny", "subtiny", "huge",
]


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("kind", _ADVERSARIAL_KINDS)
def test_production_multiplier_holds_on_adversarial_inputs(kind) -> None:
    """R4b: at the production multiplier 4.0 the repaired chain must stay
    bit-identical to the parent on cancellation-heavy, tie-targeted,
    extreme-scale, subnormal-adjacent and huge-magnitude activations."""

    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=[19, 0, 4, 33], in_features=640, out_features=256, seed=17,
    )
    x_host = _adversarial_x(
        kind, fixture.compact_rows, fixture.in_features, 19
    )
    allocations: list = []
    try:
        parent = _run_parent(
            fixture, runtime, library, allocations=allocations, x_host=x_host
        )
        candidate, risks = _run_chain(
            fixture, runtime, library, risk_multiplier=PRODUCTION_MULT,
            allocations=allocations, x_host=x_host,
        )
        np.testing.assert_array_equal(
            candidate, parent,
            err_msg=f"adversarial kind {kind}: chain differs from the parent "
                    "at the production multiplier 4.0",
        )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_nonfinite_activations_match_parent_exactly() -> None:
    """NaN/Inf activations propagate through the chain exactly as through
    the pair2 row-publish parent (the staging must queue, not drop them)."""

    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=[7, 12, 33], in_features=640, out_features=256, seed=23,
    )
    x_host = _adversarial_x(
        "nonfinite_rows", fixture.compact_rows, fixture.in_features, 29
    )
    allocations: list = []
    try:
        parent = _run_parent(
            fixture, runtime, library, allocations=allocations, x_host=x_host
        )
        candidate, risks = _run_chain(
            fixture, runtime, library, risk_multiplier=PRODUCTION_MULT,
            allocations=allocations, x_host=x_host,
        )
        np.testing.assert_array_equal(
            candidate, parent,
            err_msg="nonfinite activations must match the parent exactly",
        )
        assert risks >= 3 * fixture.out_features
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_risk_queue_overflow_is_bounded() -> None:
    """A queue smaller than the risk count keeps the true count and writes
    only within capacity."""

    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=[9, 12, 3], in_features=640, out_features=256, seed=31,
    )
    total = fixture.compact_rows * fixture.out_features
    capacity = 17
    sentinel = np.full(4 * total, -5, dtype=np.int32)
    allocations: list = []
    try:
        dx = _upload(fixture.x_host, runtime, allocations)
        ds = _upload(fixture.expert_start, runtime, allocations)
        dws = _upload(fixture.expert_start_wmma, runtime, allocations)
        dte = _upload(fixture.tile_expert, runtime, allocations)
        dw = _upload(fixture.qweight, runtime, allocations)
        d_out = _alloc(total, np.uint16, runtime, allocations)
        d_rc = _alloc(1, np.int32, runtime, allocations)
        d_ri = _alloc(sentinel.size, np.int32, runtime, allocations)
        copy_host_to_device(
            d_ri, host_array_ptr(sentinel), runtime=runtime
        )
        zero = np.zeros(1, dtype=np.int32)
        copy_host_to_device(d_rc, host_array_ptr(zero), runtime=runtime)
        q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
            dx.ptr, ds.ptr, dws.ptr, dte.ptr, dw.ptr, d_out.ptr,
            d_rc.ptr, d_ri.ptr, capacity, 1e30, fixture.compact_rows,
            fixture.in_features, fixture.out_features,
            fixture.num_experts, fixture.wmma_total_rows,
            library=library, runtime=runtime,
        )
        q51.qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
            dx.ptr, ds.ptr, dw.ptr, d_out.ptr, d_rc.ptr, d_ri.ptr, capacity,
            fixture.compact_rows, fixture.in_features,
            fixture.out_features, fixture.num_experts,
            library=library, runtime=runtime,
        )
        count = int(_download(d_rc, (1,), np.int32, runtime)[0])
        queue = _download(d_ri, (sentinel.size,), np.int32, runtime)
        assert count == total
        sanctioned = queue[:capacity]
        assert np.all((sanctioned >= 0) & (sanctioned < total))
        assert np.all(queue[capacity:] == -5), (
            "risk queue wrote beyond its capacity"
        )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_repair_is_deterministic_at_production_multiplier() -> None:
    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=[17, 0, 64, 3, 129, 32], in_features=640,
        out_features=256, seed=37,
    )
    allocations: list = []
    try:
        out_a, risks_a = _run_chain(
            fixture, runtime, library, risk_multiplier=PRODUCTION_MULT,
            allocations=allocations,
        )
        out_b, risks_b = _run_chain(
            fixture, runtime, library, risk_multiplier=PRODUCTION_MULT,
            allocations=allocations,
        )
        np.testing.assert_array_equal(out_a, out_b)
        assert risks_a == risks_b
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP runtime is not available")
def test_risk_buffers_validated_before_launch() -> None:
    runtime = get_hip_runtime()
    library = q51.build_qwen4_exp_q5_1(load=True)
    fixture = Fixture(
        counts=[16], in_features=640, out_features=256, seed=41,
    )
    allocations: list = []
    try:
        dx = _upload(fixture.x_host, runtime, allocations)
        ds = _upload(fixture.expert_start, runtime, allocations)
        dws = _upload(fixture.expert_start_wmma, runtime, allocations)
        dte = _upload(fixture.tile_expert, runtime, allocations)
        dw = _upload(fixture.qweight, runtime, allocations)
        d_out = _alloc(
            fixture.compact_rows * fixture.out_features, np.uint16,
            runtime, allocations,
        )
        d_rc = _alloc(1, np.int32, runtime, allocations)
        d_ri = _alloc(
            fixture.compact_rows * fixture.out_features, np.int32,
            runtime, allocations,
        )
        with pytest.raises(ValueError):
            q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
                dx.ptr, ds.ptr, dws.ptr, dte.ptr, dw.ptr, d_out.ptr,
                0, d_ri.ptr, 0, PRODUCTION_MULT, fixture.compact_rows,
                fixture.in_features, fixture.out_features,
                fixture.num_experts, fixture.wmma_total_rows,
                library=library, runtime=runtime,
            )
        with pytest.raises(ValueError):
            q51.qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out(
                dx.ptr, ds.ptr, dws.ptr, dte.ptr, dw.ptr, d_out.ptr,
                d_rc.ptr, d_ri.ptr, 0, -1.0, fixture.compact_rows,
                fixture.in_features, fixture.out_features,
                fixture.num_experts, fixture.wmma_total_rows,
                library=library, runtime=runtime,
            )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
