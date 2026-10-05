"""Numerical RED for real hi/lo WMMA, not the identical-product timing probe.

This is T1/T2 production arithmetic. The leaf must recover F32 operand
precision and satisfy the independent CPU outer floor; full-model promotion
requires the separate unchanged execution-profile envelope.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kg
from tests.test_gpu_gguf_q8_0_dense_wide import _dequantize_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return get_hip_runtime().current_device() >= 0
    except (OSError, RuntimeError):
        return False


def make_split_fixture(rows: int, k: int, n: int, *, cancel: bool = False):
    rng = np.random.default_rng(27091 + rows + k + n)
    blocks = np.empty((n, k // 32, 34), dtype=np.uint8)
    # Non-power-of-two F16 scales make d*q non-F16-exact: the original test
    # generator's dyadic scales would leave the weight-low plane untested.
    scales = rng.uniform(0.003, 0.037, (n, k // 32)).astype(np.float16)
    scales[0] = 0
    blocks[:, :, :2] = scales[:, :, None].view(np.uint8)
    codes = rng.integers(-128, 128, (n, k // 32, 32), dtype=np.int16).astype(np.int8)
    codes[1, :, 0] = -128
    codes[1, :, 1] = 127
    blocks[:, :, 2:] = codes.view(np.uint8)
    x = rng.standard_normal((rows, k), dtype=np.float32)
    if cancel:
        # Paired opposite rows create cancellation-heavy outputs with a small
        # residual. The oracle still sees the real F32 inputs, not rounded hi.
        codes = blocks[:, :, 2:].view(np.int8)
        codes[:, :, 1::2] = codes[:, :, 0::2]
        x[:, 1::2] = -x[:, 0::2] + np.float32(3e-4)
    return x, blocks.reshape(n, -1)


def _launch(x, weight, wrapper, library):
    runtime = get_hip_runtime()
    out = np.full((x.shape[0], weight.shape[0]), np.nan, dtype=np.float32)
    buffers = [malloc(x.nbytes), malloc(weight.nbytes), malloc(out.nbytes)]
    try:
        copy_host_to_device(buffers[0], host_array_ptr(x), runtime=runtime)
        copy_host_to_device(buffers[1], host_array_ptr(weight), runtime=runtime)
        wrapper(*(b.ptr for b in buffers), x.shape[0], x.shape[1], weight.shape[0],
                library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), buffers[2], runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    return out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime/device not available")
@pytest.mark.parametrize("rows,k,n,cancel", [(128, 256, 64, False),
    (137, 192, 70, False), (257, 1024, 129, False), (73, 256, 64, True),
    (129, 2560, 72, False), (131, 6144, 80, False)])
def test_split_wmma_recovers_f32_operand_precision(rows, k, n, cancel):
    x, weight = make_split_fixture(rows, k, n, cancel=cancel)
    cpu_reference = gguf_q8_0_gemv(x, weight)
    wh = _dequantize_weight(weight, k)
    exact = x.astype(np.float64) @ wh.astype(np.float64).T
    conditioning_scale = float(np.max(np.abs(x.astype(np.float64)) @ np.abs(wh.astype(np.float64)).T))
    np.testing.assert_allclose(cpu_reference, exact, rtol=0, atol=1e-6 * conditioning_scale)
    rounded = x.astype(np.float16).astype(np.float64) @ wh.astype(np.float16).astype(np.float64).T
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    candidate = getattr(dw, "gguf_q8_0_dense_split64x128_f32_f32_out")
    out = _launch(x, weight, candidate, library)
    repeat = _launch(x, weight, candidate, library)
    assert np.isfinite(out).all()
    assert np.array_equal(out.view(np.uint32), repeat.view(np.uint32))
    error = np.linalg.norm(out.astype(np.float64) - exact) / np.linalg.norm(exact)
    rounding_error = np.linalg.norm(rounded - exact) / np.linalg.norm(exact)
    # A cancellation fixture must use a forward-error bound relative to the
    # unsigned dot, not to the deliberately near-zero result. This same bound
    # applies to the CPU oracle; report relative recovery separately.
    if cancel:
        assert np.max(np.abs(out-exact)) < 1e-6 * conditioning_scale
    else:
        assert error < 1e-6, (error, rounding_error)
    assert error < rounding_error / 40, (error, rounding_error)
    # Independent CPU-reference distribution outer floor, not model promotion.
    a = exact - exact.max(axis=1, keepdims=True)
    b = out.astype(np.float64) - out.max(axis=1, keepdims=True)
    pa = np.exp(a); pa /= pa.sum(axis=1, keepdims=True)
    logpa = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
    logpb = b - np.log(np.exp(b).sum(axis=1, keepdims=True))
    assert np.mean(np.sum(pa * (logpa - logpb), axis=1)) <= 0.05
    assert np.mean(exact.argmax(axis=1) == out.argmax(axis=1)) >= 0.90


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime/device not available")
@pytest.mark.parametrize("k", [2560, 10240])
def test_split_wmma_long_k_accuracy_beats_uncompensated_drift(k):
    """Numerical repair target, not a parent-bit-parity promotion requirement."""
    x, weight = make_split_fixture(129, k, 72)
    wh = _dequantize_weight(weight, k)
    exact = x.astype(np.float64) @ wh.astype(np.float64).T
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    out = _launch(x, weight, dw.gguf_q8_0_dense_split64x128_f32_f32_out, library)
    relative_l2 = np.linalg.norm(out.astype(np.float64) - exact) / np.linalg.norm(exact)
    # Independently declared tighter primitive target after natural-input
    # attribution: ordinary split is 2e-7--7e-7 at these K regimes.
    assert relative_l2 < 1.2e-7, relative_l2


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime/device not available")
@pytest.mark.parametrize("large_weight", [False, True])
def test_split_wmma_range_miss_runs_the_strict_f32_fallback(large_weight):
    x, weight = make_split_fixture(137, 192, 72)
    if large_weight:
        blocks = weight.reshape(72, 6, 34)
        scales = np.full((72, 6), 600, dtype=np.float16)
        blocks[:, :, :2] = scales[:, :, None].view(np.uint8)
        blocks[:, :, 2] = np.uint8(127)
    else:
        # One outlier in each 128-row tile forces fallback for every column
        # tile, while most operands stay ordinary. F16 cannot represent it.
        x[::128, 0] = np.float32(70000)
    out = _launch(x, weight, dw.gguf_q8_0_dense_split64x128_f32_f32_out,
                  dw.build_gguf_q8_0_dense_wide(load=True))
    parent = _launch(x, weight, kg.gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out,
                     kg.build_gguf_k_gemv(load=True))
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out.view(np.uint32), parent.view(np.uint32))
