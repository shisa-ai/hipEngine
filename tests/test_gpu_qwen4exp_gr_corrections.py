import ctypes

import numpy as np
import pytest

from hipengine.core.memory import free
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload


def available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not available(), reason="HIP runtime unavailable")


def test_compensated_blocks_recover_small_terms_between_large_cancelling_terms():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels

    runtime = get_hip_runtime()
    rows, k, n = 3, 10240, 5
    x = np.ones((rows, k), np.float32)
    blocks = k // 32
    raw = np.zeros((n, blocks, 34), np.uint8)
    scales = np.full((n, blocks), 2**-16, np.float16)
    scales[:, (0, -1)] = 1024
    raw[:, :, :2] = scales.view(np.uint8).reshape(n, blocks, 2)
    raw[:, :, 2:] = 1
    raw[:, -1, 2:] = 255
    expected = np.full((rows, n), (blocks - 2) * 32 * 2**-16, np.float64)
    allocations = []
    try:
        dx, dw = _upload(x, runtime, allocations), _upload(raw, runtime, allocations)
        out = _alloc(rows * n, np.float32, runtime, allocations)
        errors = []
        for fn in (
            kernels.gguf_q8_0_iu8_wmma_prefill_f32_f32,
            kernels.gguf_q8_0_iu8_compensated_prefill_f32_f32_t,
        ):
            fn(dx.ptr, dw.ptr, out.ptr, rows, k, n, runtime=runtime)
            runtime.device_synchronize()
            actual = _download(out, (rows, n), np.float32, runtime)
            errors.append(float(np.max(np.abs(actual - expected))))
        assert errors[0] > 0.1
        assert errors[1] < 0.005
        assert errors[1] < errors[0] / 10
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.parametrize("variant", ["p4", "compensated"])
def test_corrections_match_raw_q8_oracle_at_tails_and_repeat(variant):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels
    from tests.test_gpu_qwen4exp_gr_iu8 import _q8_0_pack
    from scripts.qwen4exp_q8_boundary_replay import dequant

    runtime = get_hip_runtime()
    rng = np.random.default_rng(93241)
    rows, k, n = 17, 320, 130
    x = rng.normal(0, 0.1, (rows, k)).astype(np.float32)
    raw = _q8_0_pack(rng.normal(0, 0.05, (n, k)).astype(np.float32))
    reference = x.astype(np.float64) @ dequant(raw, k).T
    fn = getattr(kernels, f"gguf_q8_0_iu8_{variant}_prefill_f32_f32_t")
    allocations = []
    try:
        dx, dw = _upload(x, runtime, allocations), _upload(raw, runtime, allocations)
        out = _alloc(rows * n, np.float32, runtime, allocations)
        results = []
        for _ in range(3):
            fn(dx.ptr, dw.ptr, out.ptr, rows, k, n, runtime=runtime)
            runtime.device_synchronize()
            results.append(_download(out, (rows, n), np.float32, runtime))
        assert np.isfinite(results[0]).all()
        np.testing.assert_allclose(results[0], reference, rtol=2e-5, atol=2e-7)
        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
