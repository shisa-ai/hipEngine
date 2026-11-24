"""Device route selection (laguna sigmoid top-k kernel) vs the Inkling 2 host contract.

The packed GGUF path selects routes on the host with
``hipengine.kernels.cpu_reference.inkling2_mini.select_expert_routes``.
Consuming a device-selected route requires the device kernel to agree on the
*set* of experts, and on the normalized unbiased mix, for realistic and
adversarial logits. The kernel returns ids in selection order; the host
contract returns ids sorted ascending, so comparisons sort both before
comparing mixes.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.inkling2_mini import select_expert_routes
from hipengine.kernels.hip_gfx1100.moe.laguna_router import (
    build_laguna_router,
    laguna_sigmoid_correction_topk_f32,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()

EXPERTS = 128
TOP_K = 8
ROUTE_SCALE = 8.0


@pytest.fixture(scope="module")
def router_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_laguna_router(load=True)


def _run_device(logits, bias, library):
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    tokens = logits.shape[0]
    probabilities = np.empty_like(logits)
    selected_scores = np.empty_like(logits)
    selected = np.full((tokens, TOP_K), -1, dtype=np.int64)
    routing = np.zeros((tokens, TOP_K), dtype=np.float32)
    scaled = np.zeros((tokens, TOP_K), dtype=np.float32)
    arrays = (logits, bias, probabilities, selected_scores, selected, routing, scaled)
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        laguna_sigmoid_correction_topk_f32(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr,
            buffers[4].ptr, buffers[5].ptr, buffers[6].ptr,
            tokens, EXPERTS, TOP_K, ROUTE_SCALE,
            library=library,
        )
        for array, buffer in zip(arrays[2:], buffers[2:], strict=True):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
    return probabilities, selected_scores, selected, routing, scaled


def _sorted_view(ids, mix):
    order = np.argsort(ids, axis=-1)
    return np.take_along_axis(ids, order, -1), np.take_along_axis(mix, order, -1)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tokens", [1, 4, 64])
def test_device_selection_matches_host_contract(router_library, tokens):
    rng = np.random.default_rng(20261124)
    logits = (rng.normal(0.0, 2.0, size=(tokens, EXPERTS))).astype(np.float32)
    bias = rng.normal(0.0, 0.5, size=EXPERTS).astype(np.float32)

    _, _, selected, routing, scaled = _run_device(logits, bias, router_library)
    host_ids, host_mix = select_expert_routes(logits, bias, top_k=TOP_K, route_scale=ROUTE_SCALE)

    device_ids, device_mix = _sorted_view(selected, routing)
    np.testing.assert_array_equal(device_ids, host_ids)
    np.testing.assert_allclose(device_mix, host_mix / ROUTE_SCALE, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(scaled, routing * ROUTE_SCALE, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_device_selection_handles_exact_ties(router_library):
    logits = np.full((1, EXPERTS), 0.25, dtype=np.float32)
    bias = np.zeros(EXPERTS, dtype=np.float32)
    _, _, selected, routing, _ = _run_device(logits, bias, router_library)
    host_ids, host_mix = select_expert_routes(logits, bias, top_k=TOP_K, route_scale=ROUTE_SCALE)
    device_ids, device_mix = _sorted_view(selected, routing)
    np.testing.assert_array_equal(device_ids, host_ids)  # lowest ids win the ties
    np.testing.assert_allclose(device_mix, host_mix / ROUTE_SCALE, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_device_selection_matches_host_with_extreme_logits(router_library):
    rng = np.random.default_rng(7)
    logits = (rng.uniform(-30.0, 30.0, size=(2, EXPERTS))).astype(np.float32)
    logits[0, ::3] = 1e4  # saturate sigmoid high
    logits[1, ::3] = -1e4  # saturate sigmoid low
    bias = np.zeros(EXPERTS, dtype=np.float32)
    _, _, selected, routing, _ = _run_device(logits, bias, router_library)
    host_ids, host_mix = select_expert_routes(logits, bias, top_k=TOP_K, route_scale=ROUTE_SCALE)
    device_ids, device_mix = _sorted_view(selected, routing)
    np.testing.assert_array_equal(device_ids, host_ids)
    np.testing.assert_allclose(device_mix, host_mix / ROUTE_SCALE, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_device_selection_zero_bias_matches_none_bias(router_library):
    rng = np.random.default_rng(99)
    logits = rng.normal(0.0, 2.0, size=(16, EXPERTS)).astype(np.float32)
    zero_bias = np.zeros(EXPERTS, dtype=np.float32)
    _, _, selected, routing, _ = _run_device(logits, zero_bias, router_library)
    host_ids, host_mix = select_expert_routes(logits, None, top_k=TOP_K, route_scale=ROUTE_SCALE)
    device_ids, device_mix = _sorted_view(selected, routing)
    np.testing.assert_array_equal(device_ids, host_ids)
    np.testing.assert_allclose(device_mix, host_mix / ROUTE_SCALE, rtol=1e-5, atol=1e-7)
