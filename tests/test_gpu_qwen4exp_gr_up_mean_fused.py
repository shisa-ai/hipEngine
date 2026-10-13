"""The fused up projection + gated mean equals the projection followed by the epilogue."""
import ctypes
import time

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_sigmoid_f32,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def run_pair(rows, k, hidden, *, iters=0):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows + 7 * hidden)
    x = rng.normal(0, 0.3, (rows, k)).astype(np.float16)
    w = make_q8_0_weight(4 * hidden, k)
    normalized = rng.normal(0, 1.0, (rows, 4, hidden)).astype(np.float32)
    guard = 16
    mixed = np.full(rows * hidden + 2 * guard, np.float32(9.5), np.float32)
    sizes = (x.nbytes, w.nbytes, normalized.nbytes, rows * 4 * hidden * 4,
             mixed.nbytes, mixed.nbytes)
    xd, wd, nd, gd, pd, cd = (malloc(n, runtime=runtime) for n in sizes)
    try:
        for dev, host in ((xd, x), (wd, w), (nd, normalized), (pd, mixed), (cd, mixed)):
            copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)

        def parent():
            dw._WRAPPERS["dense_wide256_f16in_f32_f32_out"](
                xd.ptr, wd.ptr, gd.ptr, rows, k, 4 * hidden, runtime=runtime)
            qwen4_exp_gated_mean_sigmoid_f32(
                nd.ptr, gd.ptr, pd.ptr + guard * 4, rows, 4, hidden,
                runtime=runtime, store_gate=False)

        def fused():
            dw.qwen4_exp_gr_up_mean_f16in_f32_out(
                xd.ptr, wd.ptr, nd.ptr, cd.ptr + guard * 4, rows, k, 4, hidden,
                runtime=runtime)

        timings = []
        for fn in (parent, fused):
            fn()
            runtime.device_synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            runtime.device_synchronize()
            timings.append((time.perf_counter() - start) / max(iters, 1) * 1e3)
        p, c = np.empty_like(mixed), np.empty_like(mixed)
        copy_device_to_host(host_array_ptr(p), pd, runtime=runtime)
        copy_device_to_host(host_array_ptr(c), cd, runtime=runtime)
        return p, c, guard, timings
    finally:
        for b in (xd, wd, nd, gd, pd, cd):
            free(b, runtime=runtime)


@pytest.mark.parametrize("rows,k,hidden", [(512, 320, 2560), (300, 64, 96), (257, 320, 64)])
def test_fused_up_mean_is_bit_identical_and_bounded(rows, k, hidden):
    parent, fused, guard, _ = run_pair(rows, k, hidden)
    np.testing.assert_array_equal(fused.view(np.uint32), parent.view(np.uint32))
    assert (fused[:guard] == 9.5).all() and (fused[-guard:] == 9.5).all()
    assert np.isfinite(fused).all()


if __name__ == "__main__":
    p, c, _, t = run_pair(4096, 320, 2560, iters=30)
    print("identical", np.array_equal(p.view(np.uint32), c.view(np.uint32)),
          "parent %.3f ms fused %.3f ms" % tuple(t))
