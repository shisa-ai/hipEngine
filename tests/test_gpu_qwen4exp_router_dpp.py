"""RED contract for a same-tree tile4 router DPP implementation substitute."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference.ops import linear
from hipengine.kernels.hip_gfx1100.moe import router
from hipengine.kernels.registry import clear_registry_for_tests, resolve


@pytest.fixture(scope="module")
def library():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    return router.build_qwen35_router()


def test_router_dpp_registry_keeps_parent():
    clear_registry_for_tests()
    try:
        router.register_qwen35_router_kernels()
        assert resolve(backend="hip_gfx1100", layer="router_logits", quant="f32",
                       variant="f32_hidden_token_tile4_dpp_exact") is getattr(
                           router, "qwen35_router_logits_f32_f32w_token_tile4_dpp_exact")
        assert resolve(backend="hip_gfx1100", layer="router_logits", quant="f32",
                       variant="f32_hidden_token_tile4_dense_exact") is (
                           router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact)
    finally:
        clear_registry_for_tests()


@pytest.mark.parametrize("tokens,k,experts,mode", [
    (1, 1, 17, "random"), (3, 255, 17, "random"),
    (5, 2047, 33, "random"), (7, 2048, 33, "random"),
    (17, 2560, 512, "random"), (513, 2560, 512, "random"),
    (1024, 2560, 512, "random"), (4096, 2560, 512, "random"),
    (5, 2560, 33, "cancellation"), (9, 4097, 33, "cancellation"),
    (5, 2560, 33, "zeros"),
])
def test_router_dpp_parent_bits_and_selection(library, tokens, k, experts, mode):
    candidate = getattr(router, "qwen35_router_logits_f32_f32w_token_tile4_dpp_exact")
    rng = np.random.default_rng(20261005 + tokens + k)
    x = rng.normal(0, .1, (tokens, k)).astype(np.float32)
    weight = rng.normal(0, .1, (experts, k)).astype(np.float32)
    if mode == "cancellation":
        x = (rng.choice([-1., 1.], (tokens, k)) *
             np.exp2(rng.integers(-12, 12, (tokens, k)))).astype(np.float32)
        weight[:, 1::2] = -weight[:, :k // 2 * 2:2]
    elif mode == "zeros":
        x.fill(-0.)
    # Duplicate expert rows exercise deterministic tied selection.
    weight[1] = weight[0]
    outputs = [np.empty((tokens, experts), np.float32) for _ in range(3)]
    topk = min(10, experts)
    ids = [np.empty((tokens, topk), np.int64) for _ in range(3)]
    routing = [np.empty((tokens, topk), np.float32) for _ in range(3)]
    arrays = [x, weight, *outputs, *ids, *routing]
    buffers = []
    try:
        for arr in arrays:
            buffers.append(malloc(arr.nbytes))
        for arr, buf in zip(arrays[:2], buffers[:2], strict=True):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        for i, fn in enumerate([router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact,
                                candidate, candidate]):
            fn(buffers[0].ptr, buffers[1].ptr, buffers[2+i].ptr,
               tokens, k, experts, library=library)
            router.qwen35_router_select(buffers[2+i].ptr, buffers[5+i].ptr,
                                       buffers[8+i].ptr, tokens, experts, experts,
                                       topk, threads=128, library=library)
        for arr, buf in zip(arrays[2:], buffers[2:], strict=True):
            copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)
    assert np.isfinite(outputs[1]).all()
    for i in (1, 2):
        np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[i].view(np.uint32))
        np.testing.assert_array_equal(ids[0], ids[i])
        np.testing.assert_array_equal(routing[0].view(np.uint32), routing[i].view(np.uint32))
    if mode == "random":
        reference = linear(x, weight).astype(np.float64)
        logits = outputs[1].astype(np.float64)
        def logsoftmax(v):
            v = v - v.max(axis=1, keepdims=True)
            return v - np.log(np.exp(v).sum(axis=1, keepdims=True))
        ref_logp, candidate_logp = logsoftmax(reference), logsoftmax(logits)
        kl = np.sum(np.exp(ref_logp) * (ref_logp - candidate_logp), axis=1)
        assert float(kl.max()) <= .05
        assert float(np.mean(reference.argmax(axis=1) == logits.argmax(axis=1))) >= .9
