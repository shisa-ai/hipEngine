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
