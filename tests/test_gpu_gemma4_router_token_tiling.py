"""Router token reuse must preserve the 512-thread parent reduction tree."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest


@pytest.fixture(scope="module")
def hip_runtime():
    try:
        lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    count = ctypes.c_int()
    if lib.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("HIP device unavailable")
    from hipengine.core.hip import get_hip_runtime
    return get_hip_runtime()


@pytest.mark.parametrize("tokens,hidden_size,experts", [
    (1, 2816, 128), (15, 2816, 128), (16, 2816, 128),
    (17, 2816, 128), (63, 2816, 128), (512, 2816, 128),
    (777, 2816, 128), (4096, 2816, 128), (35, 1027, 19),
])
def test_tile16_preserves_tile8_tree_and_bounds(hip_runtime, tokens, hidden_size, experts):
    from hipengine.core import memory as mem
    from hipengine.kernels.hip_gfx1100.moe import router

    rng = np.random.default_rng(20261003 + tokens)
    hidden = rng.normal(size=(tokens, hidden_size)).astype(np.float32)
    hidden_bits = np.ascontiguousarray((hidden.view(np.uint32) >> 16).astype(np.uint16))
    weight = np.ascontiguousarray(rng.normal(scale=0.02, size=(experts, hidden_size)).astype(np.float32))
    hbuf, wbuf = mem.malloc(hidden_bits.nbytes), mem.malloc(weight.nbytes)
    mem.copy_host_to_device(hbuf, mem.host_array_ptr(hidden_bits))
    mem.copy_host_to_device(wbuf, mem.host_array_ptr(weight))
    sentinel = np.uint32(0x7FA12345)
    size = tokens * experts
    initial = np.full(size + 64, sentinel, dtype=np.uint32)
    outputs = []
    stream = hip_runtime.stream_create()
    try:
        for fn in (router.qwen35_router_logits_bf16_f32w_token_tile_8,
                   router.qwen35_router_logits_bf16_f32w_token_tile_16):
            out = mem.malloc(initial.nbytes)
            mem.copy_host_to_device(out, mem.host_array_ptr(initial))
            fn(hbuf.ptr, wbuf.ptr, out.ptr + 32 * 4, tokens, hidden_size, experts,
               threads=512, stream=stream, runtime=hip_runtime)
            hip_runtime.stream_synchronize(stream)
            actual = np.empty_like(initial)
            mem.copy_device_to_host(mem.host_array_ptr(actual), out)
            np.testing.assert_array_equal(actual[:32], initial[:32])
            np.testing.assert_array_equal(actual[-32:], initial[-32:])
            assert not np.any(actual[32:-32] == sentinel)
            outputs.append(actual[32:-32].copy())
        np.testing.assert_array_equal(outputs[0], outputs[1])
        decoded_hidden = (hidden_bits.astype(np.uint32) << 16).view(np.float32)
        reference = decoded_hidden.astype(np.float64) @ weight.astype(np.float64).T
        np.testing.assert_allclose(outputs[1].view(np.float32).reshape(tokens, experts),
                                   reference, rtol=2e-5, atol=2e-5)
    finally:
        hip_runtime.stream_destroy(stream)
