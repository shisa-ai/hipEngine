"""T0 Q5_K grouped row4: parent bits, independent dequant oracle, row ownership."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def make_weights(experts, outputs, inputs, seed=903):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (experts, outputs, inputs // 256, 176), dtype=np.uint8)
    scales = rng.uniform(0.0002, 0.002, (*raw.shape[:-1], 2)).astype(np.float16)
    raw[..., :4] = scales.view(np.uint8).reshape(*raw.shape[:-1], 4)
    return np.ascontiguousarray(raw.reshape(experts, outputs, -1))


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def test_grouped_row4_registry_keeps_strict_parent():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    candidate = resolve(
        backend="hip_gfx1151", layer="linear", quant="gguf_q5_k",
        variant="selected_grouped_row4_gemv_bf16_bf16_out")
    parent = resolve(
        backend="hip_gfx1151", layer="linear", quant="gguf_q5_k",
        variant="selected_gemv_bf16_bf16_out")
    assert candidate is gemv.gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out
    assert parent is gemv.gguf_q5_k_selected_gemv_bf16_bf16_out


@pytest.mark.skipif(not _hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("x_rows,topk,experts,k,n,sorted_rows", [
    (1, 10, 16, 256, 7, False),
    (9, 3, 8, 512, 17, False),
    (27, 1, 8, 512, 17, True),
    (64, 10, 512, 2560, 640, False),
])
def test_grouped_row4_exact(x_rows, topk, experts, k, n, sorted_rows):
    candidate = getattr(gemv, "gguf_q5_k_selected_grouped_row4_gemv_bf16_bf16_out")
    rows = x_rows * topk
    rng = np.random.default_rng(403)
    selected = rng.integers(0, experts - 1, rows, dtype=np.int64)
    selected[:min(rows, 5)] = 0
    if sorted_rows:
        selected.sort()
    order = np.argsort(selected, kind="stable").astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(np.bincount(selected, minlength=experts)))).astype(np.int64)
    x = bf16(rng.normal(0, 0.2, (x_rows, k)))
    raw = make_weights(experts, n, k)
    runtime = get_hip_runtime()
    library = gemv.build_gguf_k_gemv(load=True)
    allocations = []

    def upload(value):
        value = np.ascontiguousarray(value)
        ptr = malloc(value.nbytes, runtime=runtime)
        allocations.append(ptr)
        copy_host_to_device(ptr, host_array_ptr(value), runtime=runtime)
        return ptr

    def download(ptr):
        result = np.empty((rows, n), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(result), ptr, runtime=runtime)
        return result

    try:
        dx, dw, ds, dp, dm = map(upload, (x, raw, selected, starts, order))
        parent = upload(np.full((rows, n), 0x7FC1, np.uint16))
        output = upload(np.full((rows, n), 0x7FC1, np.uint16))
        gemv.gguf_q5_k_selected_gemv_bf16_bf16_out(
            dx.ptr, ds.ptr, dw.ptr, parent.ptr, x_rows, rows, experts, k, n,
            library=library, runtime=runtime,
        )
        expected = download(parent)
        for _ in range(2):
            candidate(
                dx.ptr, dp.ptr, None if sorted_rows else dm.ptr, dw.ptr, output.ptr,
                x_rows, rows, experts, k, n, library=library, runtime=runtime,
            )
            got = download(output)
            np.testing.assert_array_equal(got, expected)
        # Check all columns of a small row sample against independent GGUF math.
        for row in range(min(rows, 3)):
            weights = dequantize_gguf_data(raw[selected[row]], GGMLQuantizationType.Q5_K)
            oracle = weights.astype(np.float64) @ f32(x[row // topk]).astype(np.float64)
            np.testing.assert_allclose(f32(got[row]), oracle, rtol=0.008, atol=0.005)
    finally:
        runtime.device_synchronize()
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
