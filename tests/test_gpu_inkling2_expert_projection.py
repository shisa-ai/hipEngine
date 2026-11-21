import ctypes

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.quant.inkling2_experts import decode_mxfp8_expert, decode_nvfp4_expert, widen_nvfp4_scales


@pytest.mark.parametrize("nvfp4,columns", [(True, 1152), (True, 2304), (False, 1152), (False, 2304)])
def test_original_expert_projection_against_cpu(nvfp4, columns):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP runtime unavailable: {exc}")
    from hipengine.kernels.backends import resolve_backend
    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import launch_inkling2_expert

    rng = np.random.default_rng(1582)
    rows = 17
    if nvfp4:
        packed = rng.integers(0, 256, size=(rows, columns // 2), dtype=np.uint8)
        scales = rng.integers(32, 64, size=(rows, columns // 16), dtype=np.uint8)
        dense = decode_nvfp4_expert(packed, scales, amax=1.5, denominator=4)
    else:
        packed = rng.integers(0, 127, size=(rows, columns), dtype=np.uint8)
        packed |= rng.integers(0, 2, size=packed.shape, dtype=np.uint8) << 7
        scales = rng.integers(119, 124, size=(rows, columns // 32), dtype=np.uint8)
        dense = decode_mxfp8_expert(packed, scales)
    x = rng.standard_normal(columns, dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)
    rt = get_hip_runtime()
    ptrs = []
    try:
        for arr in (packed, scales, x, out):
            ptrs.append(rt.malloc(arr.nbytes))
            rt.memcpy(ptrs[-1], arr.ctypes.data, arr.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        launch_inkling2_expert(*ptrs, rows, columns, nvfp4=nvfp4, outer_scale=0.375 if nvfp4 else 1.0)
        rt.memcpy(out.ctypes.data, ptrs[-1], out.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        if nvfp4:
            expanded = widen_nvfp4_scales(scales)
            expanded_ptr = rt.malloc(expanded.nbytes)
            ptrs.append(expanded_ptr)
            rt.memcpy(expanded_ptr, expanded.ctypes.data, expanded.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            result = np.empty_like(out)
            launch_inkling2_expert(ptrs[0], expanded_ptr, ptrs[2], ptrs[3], rows, columns,
                                   nvfp4=True, outer_scale=0.375, expanded_scales=True)
            rt.memcpy(result.ctypes.data, ptrs[3], result.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
            np.testing.assert_array_equal(result.view(np.uint32), out.view(np.uint32))
    finally:
        for ptr in reversed(ptrs):
            rt.free(ptr)
    ref = dense.astype(np.float64) @ x.astype(np.float64)
    abs_products = np.abs(dense.astype(np.float64)) @ np.abs(x.astype(np.float64))
    # FP32 FMA lane accumulations followed by seven tree additions. Bound
    # roundoff by absolute products, not by the potentially cancelled sum.
    operations = (columns + 127) // 128 + 7
    unit_roundoff = np.finfo(np.float32).eps / 2
    gamma = (operations * unit_roundoff) / (1 - operations * unit_roundoff)
    assert np.all(np.abs(out - ref) <= gamma * abs_products)
    assert np.argmax(out) == np.argmax(ref)
    rp = np.exp(ref - np.max(ref)); rp /= rp.sum()
    gp = out.astype(np.float64); gp = np.exp(gp - np.max(gp)); gp /= gp.sum()
    valid = rp > 0
    assert np.sum(rp[valid] * (np.log(rp[valid]) - np.log(gp[valid]))) <= 0.05
