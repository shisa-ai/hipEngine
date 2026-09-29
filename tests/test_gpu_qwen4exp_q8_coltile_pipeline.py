"""Exactness gate for the software-pipelined Q8_0 coltile prefill projection.

The candidate is a scheduling-only change: the next K iteration's weight and
activation loads are issued before the current iteration's FMAs retire. It must
reproduce the production ``coltile8_rowbatch4_wave_scale`` parent bit for bit,
because the per-output FMA order, the wave shuffle tree and the four-wave
publication are unchanged.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out"
PIPELINED = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
PIPELINED_UNSCALED = "gguf_q8_0_gemv_coltile8_rowbatch4_pipelined_f32_f32_out"
PIPELINED_COLTILE4 = "gguf_q8_0_gemv_coltile4_rowbatch8_pipelined_f32_f32_out"
UNSCALED_PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out"
COLTILE4_PARENT = "gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry_resolves_pipelined_variants() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    for name in (
        PARENT,
        PIPELINED,
        PIPELINED_UNSCALED,
        PIPELINED_COLTILE4,
    ):
        assert (
            resolve(
                backend="hip_gfx1151",
                layer="linear",
                quant="gguf_q8_0",
                variant=name.removeprefix("gguf_q8_0_gemv_"),
            )
            is getattr(q8, name)
        )


def _make_operands(rows: int, k: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (n, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (n, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(n, k // 32, 2)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    return x, raw.reshape(n, -1)


# Rows cover a partial row tile; K covers a single iteration, a K that is not a
# multiple of the 128-thread stride, and the production K; N covers a single
# column tile and a production width.
_CASES = [
    (1, 32, 8),
    (5, 32, 8),
    (7, 96, 24),
    (3, 160, 32),
    (17, 2560, 32),
    (129, 2560, 256),
    (512, 2560, 6144),
    (1023, 2560, 10240),
]


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n", _CASES)
def test_pipelined_matches_parent_bit_exactly(rows: int, k: int, n: int) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20260929)
    allocations = []
    try:
        dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
        outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
        getattr(q8, PARENT)(dx.ptr, dw.ptr, outputs[0].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        parent = _download(outputs[0], (rows, n), np.float32, runtime)
        for _ in range(2):
            getattr(q8, PIPELINED)(
                dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime
            )
            runtime.device_synchronize()
            actual = _download(outputs[1], (rows, n), np.float32, runtime)
            np.testing.assert_array_equal(actual.view(np.uint32), parent.view(np.uint32))
        if n <= 32:
            expected = x @ dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0).T
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

            def log_softmax(a):
                a = a.astype(np.float64)
                a -= a.max(-1, keepdims=True)
                return a - np.log(np.exp(a).sum(-1, keepdims=True))

            p, q = log_softmax(expected), log_softmax(actual)
            assert np.max(np.sum(np.exp(p) * (p - q), axis=-1)) <= 0.05
            assert np.mean(expected.argmax(-1) == actual.argmax(-1)) >= 0.9
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "parent,pipelined,rows,k,n",
    [
        (UNSCALED_PARENT, PIPELINED_UNSCALED, 17, 2560, 32),
        (UNSCALED_PARENT, PIPELINED_UNSCALED, 512, 2560, 10240),
        (COLTILE4_PARENT, PIPELINED_COLTILE4, 17, 2560, 32),
        (COLTILE4_PARENT, PIPELINED_COLTILE4, 512, 2560, 10240),
    ],
)
def test_pipelined_siblings_match_their_parents(
    parent: str, pipelined: str, rows: int, k: int, n: int
) -> None:
    runtime = get_hip_runtime()
    x, raw = _make_operands(rows, k, n, seed=20260930)
    allocations = []
    try:
        dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
        outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
        getattr(q8, parent)(dx.ptr, dw.ptr, outputs[0].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        expected = _download(outputs[0], (rows, n), np.float32, runtime)
        getattr(q8, pipelined)(dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        actual = _download(outputs[1], (rows, n), np.float32, runtime)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
