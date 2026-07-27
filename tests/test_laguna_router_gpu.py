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
from hipengine.kernels.cpu_reference.laguna import (
    laguna_prune_tail_routes,
    laguna_sigmoid_correction_topk_from_logits,
)
from hipengine.kernels.hip_gfx1100.moe.laguna_router import (
    build_laguna_router,
    laguna_prune_tail_routes_f32,
    laguna_sigmoid_correction_topk_f32,
    plan_laguna_router_build,
    register_laguna_router_kernels,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    build_qwen35_router,
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def laguna_router_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_laguna_router(load=True)


@pytest.fixture(scope="module")
def logits_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_qwen35_router(load=True)


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(value.shape)


def _bf16_u16_to_f32(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.uint16)
    return (value.astype(np.uint32) << 16).view(np.float32).reshape(value.shape).copy()


def test_laguna_router_build_and_registry_contract() -> None:
    artifact = plan_laguna_router_build(
        compiler_version="hipcc laguna router test version",
    )
    assert artifact.family == "laguna_router"
    assert artifact.output_path.name == "laguna_router.so"
    assert artifact.sources[0].name == "laguna_router.hip"
    assert artifact.profile.name == "decode"
    assert "-mcumode" in artifact.flags

    register_laguna_router_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_sigmoid_router_topk",
            quant="f32",
            variant="correction_bias",
        )
        is laguna_sigmoid_correction_topk_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_router_prune",
            quant="f32",
            variant="tail_mass",
        )
        is laguna_prune_tail_routes_f32
    )


def test_laguna_router_wrapper_rejects_non_production_contracts_before_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        laguna_sigmoid_correction_topk_f32(0, 0, 0, 0, 0, 0, 0, 0, 256, 10, 2.5)
    with pytest.raises(ValueError, match="num_experts must be <= 256"):
        laguna_sigmoid_correction_topk_f32(0, 0, 0, 0, 0, 0, 0, 1, 257, 10, 2.5)
    with pytest.raises(ValueError, match="top_k must be <= 16"):
        laguna_sigmoid_correction_topk_f32(0, 0, 0, 0, 0, 0, 0, 1, 256, 17, 2.5)
    with pytest.raises(ValueError, match="routed_scaling_factor"):
        laguna_sigmoid_correction_topk_f32(0, 0, 0, 0, 0, 0, 0, 1, 256, 10, 0.0)
    with pytest.raises(ValueError, match="requires 256 threads"):
        laguna_sigmoid_correction_topk_f32(0, 0, 0, 0, 0, 0, 0, 1, 256, 10, 2.5, threads=128)
    with pytest.raises(ValueError, match="drop_count"):
        laguna_prune_tail_routes_f32(0, 0, 0, 0, 0, 2, 10, 10, 0.15, 2.5)
    with pytest.raises(ValueError, match="max_tail_mass"):
        laguna_prune_tail_routes_f32(0, 0, 0, 0, 0, 2, 10, 2, 1.0, 2.5)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_bf16_f32_router_logits_match_cpu_at_production_shape(logits_library) -> None:
    rng = np.random.default_rng(20260722)
    tokens, hidden_size, experts = 2, 3_072, 256
    hidden_bits = _f32_to_bf16_u16(
        rng.normal(0.0, 0.04, size=(tokens, hidden_size)).astype(np.float32)
    )
    hidden = _bf16_u16_to_f32(hidden_bits)
    weight = rng.normal(0.0, 0.03, size=(experts, hidden_size)).astype(np.float32)
    logits = np.zeros((tokens, experts), dtype=np.float32)
    arrays = (hidden_bits, weight, logits)
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        qwen35_router_logits_bf16_f32w_auto_256(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            tokens,
            hidden_size,
            experts,
            library=logits_library,
        )
        copy_device_to_host(host_array_ptr(logits), buffers[2], logits.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)

    expected = (hidden @ weight.T).astype(np.float32)
    np.testing.assert_allclose(logits, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_router_matches_adversarial_cpu_semantics(laguna_router_library) -> None:
    experts, top_k = 256, 10
    logits = np.empty((3, experts), dtype=np.float32)
    logits[0] = np.linspace(-100.0, 100.0, experts, dtype=np.float32)
    logits[1] = np.float32(0.0)
    logits[2] = np.sin(np.arange(experts, dtype=np.float32) * np.float32(0.17)) * np.float32(7.0)
    # Exact ties crossing wave32 boundaries must retain lower expert IDs.
    logits[1, [1, 33, 65, 97, 129, 161, 193, 225, 17, 49, 81, 113]] = np.float32(20.0)
    correction = np.zeros(experts, dtype=np.float32)
    # Selection-only correction flips ordinary top-k on rows 0 and 2.
    correction[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = np.linspace(3.0, 2.1, top_k, dtype=np.float32)
    expected = laguna_sigmoid_correction_topk_from_logits(
        logits,
        correction,
        experts_used=top_k,
        routed_scaling_factor=2.5,
    )

    probabilities = np.zeros_like(logits)
    selection_scores = np.zeros_like(logits)
    selected = np.full((logits.shape[0], top_k), -1, dtype=np.int64)
    routing = np.zeros((logits.shape[0], top_k), dtype=np.float32)
    scaled = np.zeros_like(routing)
    arrays = (
        logits,
        correction,
        probabilities,
        selection_scores,
        selected,
        routing,
        scaled,
    )
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        laguna_sigmoid_correction_topk_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            buffers[6].ptr,
            logits.shape[0],
            experts,
            top_k,
            2.5,
            library=laguna_router_library,
        )
        for array, buffer in zip(arrays[2:], buffers[2:], strict=True):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)

    np.testing.assert_array_equal(selected, expected.selected_experts)
    np.testing.assert_allclose(probabilities, expected.routing_scores, rtol=2.0e-6, atol=2.0e-7)
    np.testing.assert_allclose(
        selection_scores,
        expected.selection_scores,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(routing, expected.routing_weights, rtol=2.0e-6, atol=2.0e-7)
    np.testing.assert_allclose(
        scaled,
        expected.scaled_routing_weights,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(routing.sum(axis=-1), np.ones(3), rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(scaled.sum(axis=-1), np.full(3, 2.5), rtol=0.0, atol=5.0e-7)
    # Prove correction never leaked into the gathered route values.
    np.testing.assert_allclose(
        routing,
        np.take_along_axis(probabilities, selected, axis=-1)
        / np.take_along_axis(probabilities, selected, axis=-1).sum(axis=-1, keepdims=True),
        rtol=2.0e-6,
        atol=2.0e-7,
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_tail_prune_matches_cpu_semantics(laguna_router_library) -> None:
    selected = np.arange(30, dtype=np.int64).reshape(3, 10)
    routing = np.asarray(
        [
            [0.16, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07, 0.05, 0.04],
            [0.15, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06],
            [0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.08, 0.08],
        ],
        dtype=np.float32,
    )
    routing /= routing.sum(axis=1, keepdims=True, dtype=np.float32)
    scaled = (routing * np.float32(2.5)).astype(np.float32)
    expected = laguna_prune_tail_routes(
        selected,
        routing,
        drop_count=2,
        max_tail_mass=0.15,
        routed_scaling_factor=2.5,
    )
    arrays = (
        selected.copy(),
        routing.copy(),
        scaled,
        np.full(selected.size, 91, dtype=np.int64),
        np.full(selected.size, 92, dtype=np.int64),
    )
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        laguna_prune_tail_routes_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            selected.shape[0],
            selected.shape[1],
            2,
            0.15,
            2.5,
            library=laguna_router_library,
        )
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)

    np.testing.assert_array_equal(arrays[0], expected[0])
    np.testing.assert_allclose(arrays[1], expected[1], rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(arrays[2], expected[2], rtol=0.0, atol=5.0e-7)
    np.testing.assert_array_equal(arrays[3], -1)
    np.testing.assert_array_equal(arrays[4], -1)
