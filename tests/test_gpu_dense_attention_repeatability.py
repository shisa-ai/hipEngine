"""Strict attention broadcasts must be consumed before scratch reuse."""
import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests.test_gpu_llama_dense_prefill import _attention_oracle, _bf16_bits_to_f32, _f32_to_bf16_bits

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("tokens,heads,kv_heads,head_dim", [(33, 4, 2, 32), (83, 56, 8, 128), (137, 4, 2, 128)])
def test_strict_attention_repeated_broadcasts(tokens, heads, kv_heads, head_dim):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention import causal_keep_mask, dense_causal_prefill_bf16

    runtime = get_hip_runtime()
    rng = np.random.default_rng(12)
    q, k, v = [
        _f32_to_bf16_bits(rng.normal(size=(tokens, width * head_dim)).astype(np.float32) * .4)
        for width in (heads, kv_heads, kv_heads)
    ]
    mask = causal_keep_mask(tokens)
    buffers = []
    try:
        for host in (q, k, v, mask):
            buffer = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(buffer, host_array_ptr(host), host.nbytes, runtime=runtime)
            buffers.append(buffer)
        output = malloc(q.nbytes, runtime=runtime)
        buffers.append(output)
        first = None
        for _ in range(30):
            dense_causal_prefill_bf16(
                *(buffer.ptr for buffer in buffers), tokens=tokens,
                num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
                scale=head_dim**-.5, strict=True, runtime=runtime,
            )
            host = np.empty_like(q)
            copy_device_to_host(host_array_ptr(host), output, host.nbytes, runtime=runtime)
            if first is None:
                first = host.copy()
            np.testing.assert_array_equal(host, first)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
    oracle = _attention_oracle(q, k, v, mask, heads=heads, kv_heads=kv_heads, head_dim=head_dim)
    assert np.max(np.abs(_bf16_bits_to_f32(first) - oracle)) <= .005 * np.abs(oracle).max()
