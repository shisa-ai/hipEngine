"""Numerical contract for the LDS-staged f32 GR-up prototype.

The prototype exists to answer one question that gates two ledger items
(HC-1 hyper_connection, DP-1 dense_projection scope): the promoted
``dense_wide256`` tile is 4.6x faster than the production fused GR-up kernel and
fails the production gate, and the recorded explanation is that its matrix-core
multiply rounds both operands to f16. If that were the whole story, the traffic
mechanism itself -- stage a dequantized weight tile in LDS once per K-tile and
reuse it across every row in the block -- would be unusable at production
precision, and any route that needs it would have to change arithmetic.

The prototype separates the two: it keeps the wide tile's staging and tiling and
does the multiply in f32 FMA on f32 operands. These tests pin the claim that the
only arithmetic difference from the exact chain is the summation order, so a
future promotion cannot silently inherit f16 operand rounding.

The comparison is against the production fused kernel rather than a CPU oracle
because that kernel is already gated against ``kernels/cpu_reference``; what is
under test here is the *delta* against it.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_f32,
    qwen4_exp_sigmoid_f32,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_gr_staged_f32 as staged
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PRODUCTION = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32"

# The f32 summation-order floor. A staged f32 kernel sums K=320 in a different
# order than the production kernel's shuffle tree, so it cannot be bit-equal,
# but it must stay within a couple of f32 ulps of it. The f16-operand route
# misses this by more than two orders of magnitude (measured relative_l2
# 4.7e-5 against this kernel's 7.4e-8), which is the distinction under test.
SUMMATION_ORDER_RELATIVE_L2 = 1e-6


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _q8_0_weights(rng, rows, k):
    raw = rng.integers(0, 256, (rows, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (rows, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(rows, k // 32, 2)
    return raw.reshape(rows, -1)


def _relative_l2(reference, actual):
    reference = np.asarray(reference, np.float64)
    actual = np.asarray(actual, np.float64)
    return float(
        np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30)
    )


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "rows,k,h",
    [
        (1, 32, 6),
        (7, 64, 16),
        (37, 320, 64),
        (129, 320, 128),
        (257, 320, 2560),
    ],
)
def test_staged_f32_stays_at_the_summation_order_floor(rows, k, h):
    """The staged f32 arm must match the exact fused kernel to f32 rounding.

    ``h`` is the per-branch hidden width, so the up projection is over
    ``4*h`` outputs. ``k`` is a multiple of the 32-element Q8_0 block.
    """

    rng = np.random.default_rng(20260929)
    x = rng.normal(0, 0.2, (rows, k)).astype(np.float32)
    norm = rng.normal(0, 0.2, (rows, 4, h)).astype(np.float32)
    raw = _q8_0_weights(rng, 4 * h, k)
    runtime = get_hip_runtime()
    allocations = []
    try:
        dx, dn, dw = [_upload(v, runtime, allocations) for v in (x, norm, raw)]
        gates = [_alloc((rows, 4, h), np.float32, runtime, allocations) for _ in range(3)]
        mixed = [_alloc((rows, h), np.float32, runtime, allocations) for _ in range(3)]
        raw_out = _alloc((rows, 4 * h), np.float32, runtime, allocations)

        def production(i):
            getattr(q8, PRODUCTION)(
                dx.ptr, dw.ptr, dn.ptr, gates[i].ptr, mixed[i].ptr, rows, k, 4, h,
                runtime=runtime,
            )
            runtime.device_synchronize()
            return (
                _download(gates[i], (rows, 4, h), np.float32, runtime),
                _download(mixed[i], (rows, h), np.float32, runtime),
            )

        def staged_f32(i):
            staged.gr_staged_f32_up(
                dx.ptr, dw.ptr, raw_out.ptr, rows, k, 4 * h, runtime=runtime
            )
            qwen4_exp_sigmoid_f32(
                raw_out.ptr, raw_out.ptr, rows * 4 * h, runtime=runtime
            )
            qwen4_exp_gated_mean_f32(
                dn.ptr, raw_out.ptr, mixed[i].ptr, rows, 4, h, runtime=runtime
            )
            runtime.device_synchronize()
            return (
                _download(raw_out, (rows, 4 * h), np.float32, runtime).reshape(
                    rows, 4, h
                ),
                _download(mixed[i], (rows, h), np.float32, runtime),
            )

        expected = production(0)
        for index in (1, 2):
            actual = staged_f32(index)
            for name, got, want in zip(("gate", "mixed"), actual, expected):
                assert _relative_l2(want, got) <= SUMMATION_ORDER_RELATIVE_L2, (
                    f"{name} left the f32 summation-order floor at "
                    f"{_relative_l2(want, got):.3e}"
                )
        # Same arm, twice: the reduction is a fixed order, not a race.
        repeat = staged_f32(1)
        for name, first, second in zip(("gate", "mixed"), actual, repeat):
            np.testing.assert_array_equal(
                first.view(np.uint32), second.view(np.uint32)
            )
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
