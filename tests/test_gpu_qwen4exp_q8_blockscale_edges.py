"""Non-finite risk estimates must not bypass sparse correction."""

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
)
from tests.test_gpu_qwen4exp_q8_blockscale import available, bf16
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download


@pytest.mark.skipif(not available(), reason="HIP unavailable")
def test_nan_risk_bound_for_finite_result_repairs_every_output():
    x = np.zeros((1, 64), dtype=np.float32)
    x[:, :32] = 1e38
    x[:, 57] = 1
    raw = np.zeros((1, 32, 2, 34), dtype=np.uint8)
    scales = np.zeros((1, 32, 2), dtype=np.float16)
    scales[..., 1] = np.float16(0.0467529296875)
    raw[..., :2] = scales.view(np.uint8).reshape(1, 32, 2, 2)
    raw[:, :, 1, 2:] = np.int8(-1).view(np.uint8)
    raw[:, :, 1, 2 + 25] = 16
    runtime = get_hip_runtime()
    allocations = []
    try:
        inputs = [_upload(value, runtime, allocations) for value in (
            bf16(x), np.array([0, 1], np.int64), np.array([0, 16], np.int64),
            np.array([0], np.int64), raw)]
        output = _alloc(32, np.uint16, runtime, allocations)
        count = _alloc(1, np.int32, runtime, allocations)
        indices = _alloc(32, np.int32, runtime, allocations)
        gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out(
            *(value.ptr for value in inputs), output.ptr, 1, 1, 64, 32, 16,
            risk_count_ptr=count.ptr, risk_indices_ptr=indices.ptr,
            risk_capacity=32, runtime=runtime)
        runtime.device_synchronize()
        assert int(_download(count, (1,), np.int32, runtime)[0]) == 32
        np.testing.assert_array_equal(
            _download(output, (1, 32), np.uint16, runtime),
            bf16(np.full((1, 32), 16 * float(scales[0, 0, 1]), np.float32)))
    finally:
        for allocation in reversed(allocations):
            free(allocation)
