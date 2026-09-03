"""PF-4 lever-2 RED tests: T0 fused weighted_lanes_sum + shared_gate_combine.

The candidate kernel `hipengine_weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w`
does not exist yet on the unmodified path — these tests RED via ImportError.
Once implemented they pin the T0 contract: the fused kernel must be
bit-identical to the unfused production chain

    weighted_lanes_sum_out_bf16_f32w      (routed = bf16(sum_k w_k * v_lane_k))
    shared_gate_combine_batch_out_bf16    (out = bf16(routed + sigmoid(logit_t) * shared))

including the intermediate BF16 rounding of the routed sum (the rounding
boundary is the contract; see worklog entry
20260903T214325.735150Z-lhl-pf-4-m5-declaration-3ff110).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


_NUM_EXPERTS = 512
_TOP_K = 10
_HIDDEN = 2560


def _routed_lanes(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0, 1.0, size=_NUM_EXPERTS) ** 4
    p = weights / weights.sum()
    return np.stack(
        [rng.choice(_NUM_EXPERTS, size=_TOP_K, replace=False, p=p)
         for _ in range(rows)]
    ).astype(np.int64).reshape(rows * _TOP_K)


def _sorted_weights(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, size=rows * _TOP_K).astype(np.float32)
    return (w / w.sum()).astype(np.float32)


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """Round F32 to BF16 (round-to-nearest-even) via uint32 manipulation."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    rounding = 0x7FFF + ((u >> 16) & 1)
    u = (u + rounding) >> 16
    return u.astype(np.uint16)


def _from_bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
class TestPf4Lever2FusedCombine:
    @pytest.fixture(scope="class")
    def runtime(self):
        return get_hip_runtime()

    def _alloc(self, array, runtime, allocations):
        from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
        host = np.ascontiguousarray(array)
        device = malloc(host.nbytes, runtime=runtime)
        allocations.append(device)
        copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
        return device

    def _download(self, device, shape, dtype, runtime):
        from hipengine.core.memory import copy_device_to_host, host_array_ptr
        host = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
        return host

    @pytest.mark.parametrize("rows", [1, 16, 64, 512])
    def test_fused_bit_identical_to_unfused_chain(self, runtime, rows):
        # ImportError RED: the candidate wrapper does not exist yet.
        from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
            shared_gate_combine_batch_out_bf16,
            weighted_lanes_sum_out_bf16_f32w,
            weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w,
        )

        top_k, features = _TOP_K, _HIDDEN
        allocations: list = []
        try:
            rng = np.random.default_rng(7100 + rows)
            values = _to_bf16(rng.standard_normal(rows * top_k * features) * 0.5)
            weights = _sorted_weights(rows, seed=7200 + rows)
            # Production grouped path: sorted_lanes index the per-lane weights
            # and lane_to_row[lane] maps to the expert_down row. Identity
            # mapping keeps the fixture production-faithful (weights are
            # already per-lane sorted weights).
            sorted_lanes = np.arange(rows * top_k, dtype=np.int64)
            lane_to_row_scratch = np.zeros(rows * top_k, dtype=np.int64)
            shared = _to_bf16(rng.standard_normal(rows * features) * 0.5)
            gate_logits = rng.standard_normal(rows).astype(np.float32)

            d_values = self._alloc(values, runtime, allocations)
            d_weights = self._alloc(weights, runtime, allocations)
            d_sorted_lanes = self._alloc(sorted_lanes, runtime, allocations)
            d_lane_to_row = self._alloc(lane_to_row_scratch, runtime, allocations)
            d_shared = self._alloc(shared, runtime, allocations)
            d_gate = self._alloc(gate_logits, runtime, allocations)
            d_routed = self._alloc(np.zeros(rows * features, np.uint16), runtime, allocations)
            d_out = self._alloc(np.zeros(rows * features, np.uint16), runtime, allocations)

            # Unfused production chain (strict fallback), exactly as the
            # runner drives it (qwen4_exp_runner.py grouped path).
            weighted_lanes_sum_out_bf16_f32w(
                d_values.ptr, d_weights.ptr, d_sorted_lanes.ptr,
                d_lane_to_row.ptr, d_routed.ptr, rows, top_k, features,
                runtime=runtime)
            shared_gate_combine_batch_out_bf16(
                d_routed.ptr, d_shared.ptr, d_gate.ptr, d_out.ptr,
                rows, features, runtime=runtime)
            runtime.device_synchronize()

            # Fused candidate (fresh lane_to_row scratch, same semantics).
            d_lane_to_row2 = self._alloc(lane_to_row_scratch, runtime, allocations)
            d_out_fused = self._alloc(np.zeros(rows * features, np.uint16), runtime, allocations)
            weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w(
                d_values.ptr, d_weights.ptr, d_sorted_lanes.ptr,
                d_lane_to_row2.ptr, d_shared.ptr, d_gate.ptr, d_out_fused.ptr,
                rows, top_k, features, runtime=runtime)
            runtime.device_synchronize()

            expected = self._download(d_out, (rows * features,), np.uint16, runtime)
            actual = self._download(d_out_fused, (rows * features,), np.uint16, runtime)
            np.testing.assert_array_equal(actual, expected)
        finally:
            from hipengine.core.memory import free
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)

    def test_registry_entry_present(self):
        from hipengine.kernels.registry import KernelKey, get_kernel
        fn = get_kernel(KernelKey("hip_gfx1100", "weighted_lanes_sum+shared_gate_combine", "bf16", "out"))
        assert callable(fn)
