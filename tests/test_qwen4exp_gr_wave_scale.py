import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32"
CANDIDATE = "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32"


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    for variant, fn in (
        ("coltile2_branch4_rowbatch4_f32_exact", PARENT),
        ("coltile2_branch4_rowbatch4_wave_scale_f32_exact", CANDIDATE),
    ):
        assert resolve(
            backend="hip_gfx1151", layer="linear+gr_gated_mean", quant="gguf_q8_0", variant=variant
        ) is getattr(q8, fn)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize(
    "rows,k,h",
    [
        (1, 32, 6),
        (7, 96, 18),
        (37, 320, 2560),
        (257, 320, 2560),
        (511, 320, 2560),
        (512, 320, 2560),
    ],
)
def test_exact_gate_and_mixed(rows, k, h):
    rng = np.random.default_rng(6342)
    x = rng.normal(0, 0.2, (rows, k)).astype(np.float32)
    norm = rng.normal(0, 0.2, (rows, 4, h)).astype(np.float32)
    raw = rng.integers(0, 256, (4 * h, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (4 * h, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(4 * h, k // 32, 2)
    raw = raw.reshape(4 * h, -1)
    runtime = get_hip_runtime()
    allocations = []
    try:
        dx, dn, dw = [_upload(v, runtime, allocations) for v in (x, norm, raw)]
        gates = [_alloc((rows, 4, h), np.float32, runtime, allocations) for _ in range(2)]
        mixed = [_alloc((rows, h), np.float32, runtime, allocations) for _ in range(2)]

        def run(name, i):
            getattr(q8, name)(
                dx.ptr, dw.ptr, dn.ptr, gates[i].ptr, mixed[i].ptr, rows, k, 4, h, runtime=runtime
            )
            runtime.device_synchronize()
            return (
                _download(gates[i], (rows, 4, h), np.float32, runtime),
                _download(mixed[i], (rows, h), np.float32, runtime),
            )

        parent = run(PARENT, 0)
        for _ in range(2):
            for actual, expected in zip(run(CANDIDATE, 1), parent):
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        if h < 32:
            logits = (x @ dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0).T).reshape(
                rows, 4, h
            )
            expected = 1 / (1 + np.exp(-logits))
            np.testing.assert_allclose(parent[0], expected, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(
                parent[1], (norm * expected).mean(axis=1), rtol=1e-4, atol=1e-5
            )
            for actual, reference in zip(parent, (expected, (norm * expected).mean(axis=1))):

                def log_softmax(a):
                    a = a.astype(np.float64)
                    a -= a.max(axis=-1, keepdims=True)
                    return a - np.log(np.exp(a).sum(axis=-1, keepdims=True))

                lp, lq = log_softmax(reference), log_softmax(actual)
                assert np.max(np.sum(np.exp(lp) * (lp - lq), axis=-1)) <= 0.05
                assert np.mean(reference.argmax(-1) == actual.argmax(-1)) >= 0.9
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
