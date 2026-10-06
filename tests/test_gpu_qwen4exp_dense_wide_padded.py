"""T0 padded-fragment sibling: parent parity, independent CPU floor and ownership."""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _copy_from(device, template, runtime):
    result = np.empty_like(template)
    copy_device_to_host(host_array_ptr(result), device, runtime=runtime)
    return result


@pytest.mark.parametrize("f16in", [False, True])
@pytest.mark.parametrize("rows,k,n", [
    (1, 64, 1), (15, 192, 63), (17, 256, 129),
    (257, 256, 200), (300, 192, 128), (256, 256, 256),
])
def test_padded_fragments_preserve_output_and_bounds(rows, k, n, f16in):
    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    suffix = "f16in_f32_f32_out" if f16in else "f32_f32_out"
    parent = dw._WRAPPERS[f"dense_wide256_{suffix}"]
    candidate = dw._WRAPPERS[f"dense_wide256_padded_{suffix}"]
    rng = np.random.default_rng(97123)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    if f16in:
        x = x.astype(np.float16)
    w = make_q8_0_weight(n, k)
    guard = 16
    poison = np.float32(-923.5)
    out_host = np.full(rows * n + guard * 2, poison, np.float32)
    buffers = [malloc(a.nbytes, runtime=runtime) for a in (x, w, out_host, out_host)]
    xd, wd, pd, cd = buffers
    try:
        for dev, host in zip(buffers, (x, w, out_host, out_host)):
            copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
        common = (xd.ptr, wd.ptr)
        snapshots = []
        for fn, dev in ((parent, pd), (candidate, cd), (candidate, cd)):
            fn(*common, dev.ptr + guard * 4, rows, k, n,
               library=library, runtime=runtime)
            runtime.device_synchronize()
            snapshots.append(_copy_from(dev, out_host, runtime))
        p, c, repeat = snapshots
        np.testing.assert_array_equal(c.view(np.uint32), p.view(np.uint32))
        np.testing.assert_array_equal(repeat.view(np.uint32), c.view(np.uint32))
        np.testing.assert_array_equal(c[:guard], out_host[:guard])
        np.testing.assert_array_equal(c[-guard:], out_host[-guard:])
        np.testing.assert_array_equal(_copy_from(xd, x, runtime), x)
        np.testing.assert_array_equal(_copy_from(wd, w, runtime), w)
        actual = c[guard:-guard].reshape(rows, n)
        reference = gguf_q8_0_gemv(x.astype(np.float32), w)
        assert np.isfinite(actual).all()
        logits = actual.astype(np.float64)
        ref = reference.astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        ref -= ref.max(axis=-1, keepdims=True)
        lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        lr = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
        kl = np.sum(np.exp(lr) * (lr - lp), axis=-1)
        assert float(np.max(kl)) <= 0.05
        assert np.mean(np.argmax(actual, axis=-1) == np.argmax(reference, axis=-1)) >= 0.9
    finally:
        for dev in buffers:
            free(dev, runtime=runtime)
