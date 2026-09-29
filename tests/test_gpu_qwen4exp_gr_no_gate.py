"""The GR gate store is dead, and removing it is measurably not a win.

Ledger HC-3 records an "unconsumed 168 MB/layer GR gate write" as an
arithmetic-free optimization. The consumer census in
``docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`` confirmed the first half: no production
caller reads the gate field of ``Qwen4ExpGRReadDeviceResult``. These tests pin
the second half, which is the part that decides whether the item is worth
implementing:

- the suppressed-store variant produces a **bit-identical** ``mixed``, so the
  store is purely additive and removing it cannot change any downstream result;
- therefore the store's cost is the only question, and it was measured at
  ``scripts/qwen4exp_gr_projection_screen.py`` as a **2% regression** when
  removed (6.6551 ms against 6.5198 ms, non-overlapping sample sets over five
  repetitions).

The bit-identity test is the load-bearing one: it is what lets a future reader
trust that the screen's timing comparison is a pure store-cost comparison and
not a comparison of two different computations.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32"
NO_GATE = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32"


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


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "rows,k,h",
    [
        (1, 32, 6),
        (7, 96, 18),
        (37, 320, 2560),
        (257, 320, 2560),
    ],
)
def test_suppressing_the_gate_store_leaves_mixed_bit_identical(rows, k, h):
    """``mixed`` must be identical bit for bit with and without the gate store.

    If this ever fails, the store is not dead code in the arithmetic sense and
    the HC-3 no-win conclusion no longer applies.
    """

    rng = np.random.default_rng(20260930)
    x = rng.normal(0, 0.2, (rows, k)).astype(np.float32)
    norm = rng.normal(0, 0.2, (rows, 4, h)).astype(np.float32)
    raw = _q8_0_weights(rng, 4 * h, k)
    runtime = get_hip_runtime()
    allocations = []
    try:
        dx, dn, dw = [_upload(v, runtime, allocations) for v in (x, norm, raw)]
        gates = [_alloc((rows, 4, h), np.float32, runtime, allocations) for _ in range(2)]
        mixed = [_alloc((rows, h), np.float32, runtime, allocations) for _ in range(2)]

        getattr(q8, PARENT)(
            dx.ptr, dw.ptr, dn.ptr, gates[0].ptr, mixed[0].ptr, rows, k, 4, h,
            runtime=runtime,
        )
        runtime.device_synchronize()
        with_gate = _download(mixed[0], (rows, h), np.float32, runtime)

        # The no-gate wrapper takes no gate buffer at all: it passes a null
        # pointer, so the kernel must not dereference it.
        getattr(q8, NO_GATE)(
            dx.ptr, dw.ptr, dn.ptr, mixed[1].ptr, rows, k, 4, h, runtime=runtime
        )
        runtime.device_synchronize()
        without_gate = _download(mixed[1], (rows, h), np.float32, runtime)

        np.testing.assert_array_equal(
            with_gate.view(np.uint32), without_gate.view(np.uint32)
        )

        # The gate-writing arm must still be a working reference: a variant that
        # silently produced garbage in both arms would pass the equality above.
        assert np.all(np.isfinite(with_gate))
        assert np.any(with_gate != 0.0)
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
def test_no_gate_variant_rejects_a_zero_mixed_buffer():
    """The suppressed-store variant still requires its real output buffer."""

    runtime = get_hip_runtime()
    with pytest.raises(Exception):
        q8.gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32(
            0, 0, 0, 0, 8, 32, 4, 4, runtime=runtime
        )
