"""Exact GR-up reduction contract; the existing shuffle chain is the oracle."""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32"
DPP = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_f32"


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry_preserves_parent_and_resolves_dpp():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        for suffix, name in (("wave_scale", PARENT), ("wave_scale_dpp", DPP)):
            assert resolve(
                backend=backend, layer="linear+gr_gated_mean", quant="gguf_q8_0",
                variant=f"coltile2_branch4_rowbatch4_{suffix}_f32_exact",
            ) is getattr(q8, name)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,h", [
    (1, 32, 6), (7, 96, 18), (17, 160, 64), (37, 320, 2560),
    (257, 320, 2560), (511, 320, 2560), (1023, 320, 2560),
    (1024, 320, 2560), (4096, 320, 2560),
])
def test_dpp_gate_and_mixed_match_parent_bits(rows, k, h):
    candidate = getattr(q8, DPP)  # RED: absent before implementation.
    rng = np.random.default_rng(20261005)
    x = rng.normal(0, 0.2, (rows, k)).astype(np.float32)
    normalized = rng.normal(0, 0.2, (rows, 4, h)).astype(np.float32)
    raw = rng.integers(0, 256, (4 * h, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, raw.shape[:2]).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(*raw.shape[:2], 2)
    raw = raw.reshape(4 * h, -1)
    runtime = get_hip_runtime()
    allocations = []
    try:
        dx, dn, dw = [_upload(v, runtime, allocations) for v in (x, normalized, raw)]
        gates = [_alloc((rows, 4, h), np.float32, runtime, allocations) for _ in range(2)]
        mixed = [_alloc((rows, h), np.float32, runtime, allocations) for _ in range(2)]

        def run(fn, slot):
            fn(dx.ptr, dw.ptr, dn.ptr, gates[slot].ptr, mixed[slot].ptr,
               rows, k, 4, h, runtime=runtime)
            runtime.device_synchronize()
            return (_download(gates[slot], (rows, 4, h), np.float32, runtime),
                    _download(mixed[slot], (rows, h), np.float32, runtime))

        parent = run(getattr(q8, PARENT), 0)
        for _ in range(2):
            actual = run(candidate, 1)
            for got, want in zip(actual, parent, strict=True):
                np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
        unscaled = run(q8.gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32, 0)
        for got, want in zip(actual, unscaled, strict=True):
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
        if h < 32:
            from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
                qwen4_exp_gated_mean_f32, qwen4_exp_sigmoid_f32,
            )
            q8.gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
                dx.ptr, dw.ptr, gates[0].ptr, rows, k, 4 * h, runtime=runtime)
            qwen4_exp_sigmoid_f32(gates[0].ptr, gates[0].ptr, rows * 4 * h, runtime=runtime)
            qwen4_exp_gated_mean_f32(dn.ptr, gates[0].ptr, mixed[0].ptr, rows, 4, h,
                                   runtime=runtime)
            runtime.device_synchronize()
            chain = (_download(gates[0], (rows, 4, h), np.float32, runtime),
                     _download(mixed[0], (rows, h), np.float32, runtime))
            for got, want in zip(actual, chain, strict=True):
                np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
            from hipengine.kernels.cpu_reference.ops import linear
            from hipengine.kernels.cpu_reference.qwen4_exp import _sigmoid

            logits = linear(x, dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0))
            gate = _sigmoid(logits).reshape(rows, 4, h)
            for got, want in zip(actual, (gate, (normalized * gate).mean(axis=1)), strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                def log_softmax(a):
                    a = a.astype(np.float64)
                    a -= a.max(axis=-1, keepdims=True)
                    return a - np.log(np.exp(a).sum(axis=-1, keepdims=True))
                p, q = log_softmax(want), log_softmax(got)
                assert np.max(np.sum(np.exp(p) * (p - q), axis=-1)) <= 0.05
                assert np.mean(want.argmax(-1) == got.argmax(-1)) >= 0.9
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
