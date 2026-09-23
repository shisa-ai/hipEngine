"""Attention reduction geometry regressions on gfx1100."""

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("head_dim", [6, 48, 64, 96, 192, 256, 512])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("keys", [3, 8192, 15616])
def test_uniform_attention_preserves_unit_values(head_dim, dtype, keys):
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32, gemma4_attention_prefill_bf16,
    )

    storage = np.float32 if dtype == "f32" else np.uint16
    one = 1.0 if dtype == "f32" else 0x3F80
    query = np.zeros((1, 1, head_dim), dtype=storage)
    key = np.zeros((keys, 1, head_dim), dtype=storage)
    value = np.full_like(key, one)
    mask = np.ones((1, keys), dtype=np.uint8)
    output = np.empty_like(query)
    buffers = []
    try:
        for array in (query, key, value, mask, output):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        launch = gemma4_attention_prefill_f32 if dtype == "f32" else gemma4_attention_prefill_bf16
        launch(*(b.ptr for b in buffers), tokens=1, keys=keys, num_heads=1,
               num_kv_heads=1, head_dim=head_dim, scale=1.0)
        copy_device_to_host(host_array_ptr(output), buffers[-1], output.nbytes)
        # Zero scores give equal weights. Their weighted average must be one,
        # independently of reduction block width or output representation.
        np.testing.assert_array_equal(output, np.full_like(output, one))
    finally:
        for buffer in buffers:
            free(buffer)
