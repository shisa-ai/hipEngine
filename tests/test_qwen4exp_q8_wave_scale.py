import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out"
CANDIDATE = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out"


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
    for name in (PARENT, CANDIDATE):
        assert resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q8_0",
            variant=name.removeprefix("gguf_q8_0_gemv_"),
        ) is getattr(q8, name)


def test_screen_rejects_unbalanced_pairs_without_gpu(monkeypatch):
    from scripts.qwen4exp_q8_wave_scale_screen import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "screen",
            "--model-root",
            "/unused",
            "--compiler-version-file",
            "/unused",
            "--output",
            "/unused",
            "--pairs",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n", [(1, 32, 8), (7, 96, 24), (17, 2560, 32), (512, 2560, 6144)])
def test_exact_and_cpu(rows, k, n):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(12687)
    raw = rng.integers(0, 256, (n, k // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.0001, 0.003, (n, k // 32)).astype(np.float16)
    raw[..., :2] = scales.view(np.uint8).reshape(n, k // 32, 2)
    raw = raw.reshape(n, -1)
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    allocations = []
    try:
        dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
        outputs = [_alloc((rows, n), np.float32, runtime, allocations) for _ in range(2)]
        getattr(q8, PARENT)(dx.ptr, dw.ptr, outputs[0].ptr, rows, k, n, runtime=runtime)
        runtime.device_synchronize()
        parent = _download(outputs[0], (rows, n), np.float32, runtime)
        for _ in range(2):
            getattr(q8, CANDIDATE)(dx.ptr, dw.ptr, outputs[1].ptr, rows, k, n, runtime=runtime)
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
