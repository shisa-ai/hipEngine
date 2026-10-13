"""The eight-column BF16 weighted lane sum equals the scalar kernel bit for bit."""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    weighted_lanes_sum_out_bf16_f32w,
)


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("tokens,top_k,features", [(37, 8, 2560), (5, 4, 72), (64, 8, 512)])
def test_vec8_matches_the_scalar_kernel(tokens, top_k, features):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(tokens + features)
    rows = tokens * top_k
    values = rng.integers(0, 1 << 16, (rows, features), dtype=np.uint16)
    # Keep every value finite: clear the all-ones exponent.
    values[(values & 0x7F80) == 0x7F80] &= 0xBFFF
    weights = rng.normal(0, 0.3, rows).astype(np.float32)
    sorted_lanes = rng.permutation(rows).astype(np.int64)
    guard = 16
    out_words = tokens * features + 2 * guard
    vd, wd, sd = (malloc(a.nbytes, runtime=runtime) for a in (values, weights, sorted_lanes))
    ld = malloc(rows * 8, runtime=runtime)
    od = malloc(out_words * 2, runtime=runtime)
    try:
        for dev, host in ((vd, values), (wd, weights), (sd, sorted_lanes)):
            copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
        results = []
        # A 16-byte aligned output takes the eight-column kernel; one that is
        # 2 bytes off falls back to the scalar kernel.
        for shift in (guard, guard + 1):
            host = np.full(out_words, 0x1234, np.uint16)
            copy_host_to_device(od, host_array_ptr(host), runtime=runtime)
            weighted_lanes_sum_out_bf16_f32w(
                vd.ptr, wd.ptr, sd.ptr, ld.ptr, od.ptr + shift * 2,
                tokens, top_k, features, runtime=runtime)
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(host), od, runtime=runtime)
            assert (host[:shift] == 0x1234).all()
            assert (host[shift + tokens * features:] == 0x1234).all()
            results.append(host[shift:shift + tokens * features].copy())
        np.testing.assert_array_equal(results[0], results[1])
    finally:
        for buf in (vd, wd, sd, ld, od):
            free(buf, runtime=runtime)
