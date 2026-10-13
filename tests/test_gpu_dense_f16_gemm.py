"""The small-matrix F16 WMMA GEMM matches the FP32 reference product.

Operands are rounded to F16 and accumulated in F32, so the oracle is the
FP32 matmul of the F16-rounded operands at a tight tolerance (accumulation
order only), plus a looser bound against the unrounded FP32 product.
"""

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
from hipengine.kernels.hip_gfx1100.linear.dense_f16_gemm import (
    build_dense_f16_gemm,
    dense_f16_gemm_f32_bf16w_f32_out,
    dense_f16_gemm_f32_f32w_f32_out,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.mark.parametrize("weight_dtype", ["f32", "bf16"])
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (64, 64, 128),
        (130, 2560, 4),      # hyper-connection inject shape class: 4 columns
        (257, 2560, 48),     # GDN gate projections
        (100, 2560, 512),    # indexer query projection
        (3, 320, 131),       # ragged rows and columns
    ],
)
def test_dense_f16_gemm_matches_reference(
    weight_dtype, rows, in_features, out_features
) -> None:
    rng = np.random.default_rng(rows * 7 + out_features)
    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    w = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float32)
    if weight_dtype == "bf16":
        w_host = _to_bf16(w)
        w_ref = (w_host.astype(np.uint32) << 16).view(np.float32)
        fn = dense_f16_gemm_f32_bf16w_f32_out
    else:
        w_host = w
        w_ref = w
        fn = dense_f16_gemm_f32_f32w_f32_out
    rounded = x.astype(np.float16).astype(np.float64) @ w_ref.astype(
        np.float16
    ).astype(np.float64).T
    exact = x.astype(np.float64) @ w_ref.astype(np.float64).T

    runtime = get_hip_runtime()
    library = build_dense_f16_gemm(load=True)
    sentinel = np.float32(-777.0)
    out_host = np.full((rows + 1, out_features), sentinel, dtype=np.float32)
    buffers = []
    try:
        for arr in (x, np.ascontiguousarray(w_host), out_host):
            dev = malloc(arr.nbytes, runtime=runtime)
            buffers.append(dev)
            copy_host_to_device(dev, host_array_ptr(arr), runtime=runtime)
        fn(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
            rows, in_features, out_features,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), buffers[2], runtime=runtime)
    finally:
        for dev in reversed(buffers):
            free(dev, runtime=runtime)

    assert (out_host[rows:] == sentinel).all()
    actual = out_host[:rows].astype(np.float64)
    assert np.isfinite(actual).all()
    scale = np.abs(rounded).max()
    assert np.abs(actual - rounded).max() <= 2e-5 * scale
    assert np.linalg.norm(actual - exact) / np.linalg.norm(exact) <= 2e-3


@pytest.mark.parametrize("weight_dtype", ["f32", "bf16"])
@pytest.mark.parametrize("tile", [1, 2, 3])
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [(64, 64, 128), (257, 2560, 48), (300, 640, 131), (513, 1280, 512), (130, 2560, 4)],
)
def test_staged_tiles_are_bit_identical_to_the_scalar_kernel(
    weight_dtype, tile, rows, in_features, out_features
) -> None:
    rng = np.random.default_rng(rows + 31 * out_features + tile)
    x = (rng.standard_normal((rows, in_features)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float32)
    w_host = w if weight_dtype == "f32" else _to_bf16(w)
    runtime = get_hip_runtime()
    lib = build_dense_f16_gemm(load=True)
    fn = getattr(lib, f"hipengine_dense_f16_gemm_f32_{weight_dtype}w_f32_out_tile")
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_void_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    guard = 8
    xd = malloc(x.nbytes, runtime=runtime)
    wd = malloc(w_host.nbytes, runtime=runtime)
    od = malloc((rows * out_features + 2 * guard) * 4, runtime=runtime)
    try:
        copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(wd, host_array_ptr(w_host), runtime=runtime)
        results = []
        for selected in (0, tile):  # 0 names the scalar kernel on this entry
            host = np.full(rows * out_features + 2 * guard, np.float32(-3.5), np.float32)
            copy_host_to_device(od, host_array_ptr(host), runtime=runtime)
            assert fn(xd.ptr, wd.ptr, od.ptr + guard * 4, rows, in_features,
                      out_features, None, selected) == 0
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(host), od, runtime=runtime)
            results.append(host)
        scalar, staged = results
        np.testing.assert_array_equal(staged.view(np.uint32), scalar.view(np.uint32))
        assert (staged[:guard] == -3.5).all() and (staged[-guard:] == -3.5).all()
    finally:
        for buf in (xd, wd, od):
            free(buf, runtime=runtime)


@pytest.mark.parametrize("weight_dtype", ["f32", "bf16"])
@pytest.mark.parametrize("rows,in_features,out_features", [(257, 2560, 48), (513, 1280, 512)])
def test_a_declared_f16_image_gives_the_same_bits(
    weight_dtype, rows, in_features, out_features
) -> None:
    from hipengine.kernels.hip_gfx1100.linear.f16_mirror import (
        declare_f16_mirror, invalidate_f16_mirror,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide import f32_to_f16

    rng = np.random.default_rng(rows + out_features)
    x = (rng.standard_normal((rows, in_features)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float32)
    w_host = w if weight_dtype == "f32" else _to_bf16(w)
    gemm = (dense_f16_gemm_f32_f32w_f32_out if weight_dtype == "f32"
            else dense_f16_gemm_f32_bf16w_f32_out)
    runtime = get_hip_runtime()
    xd = malloc(x.nbytes, runtime=runtime)
    hd = malloc(x.nbytes // 2, runtime=runtime)
    wd = malloc(w_host.nbytes, runtime=runtime)
    od = malloc(rows * out_features * 4, runtime=runtime)
    try:
        copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(wd, host_array_ptr(w_host), runtime=runtime)
        f32_to_f16(xd.ptr, hd.ptr, rows * in_features, runtime=runtime)
        results = []
        for mirrored in (False, True):
            invalidate_f16_mirror()
            if mirrored:
                declare_f16_mirror(xd.ptr, rows * in_features, hd.ptr)
            host = np.zeros(rows * out_features, np.float32)
            gemm(xd.ptr, wd.ptr, od.ptr, rows, in_features, out_features, runtime=runtime)
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(host), od, runtime=runtime)
            results.append(host)
        np.testing.assert_array_equal(results[1].view(np.uint32), results[0].view(np.uint32))
    finally:
        invalidate_f16_mirror()
        for buf in (xd, hd, wd, od):
            free(buf, runtime=runtime)
