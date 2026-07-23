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
    laguna_sigmoid_correction_topk_from_logits,
)
from hipengine.kernels.hip_gfx1100.moe.laguna_router import (
    build_laguna_router,
    laguna_router_topk_bf16_hidden_correction_bias_persistent,
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
            layer="laguna_router_topk",
            quant="f32",
            variant="bf16_hidden_correction_bias_persistent",
        )
        is laguna_router_topk_bf16_hidden_correction_bias_persistent
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
    with pytest.raises(ValueError, match="decode-only"):
        laguna_router_topk_bf16_hidden_correction_bias_persistent(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 17, 256, 10, 2.5
        )
    with pytest.raises(ValueError, match="completion_counter_ptr"):
        laguna_router_topk_bf16_hidden_correction_bias_persistent(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 17, 256, 10, 2.5
        )
    with pytest.raises(ValueError, match="hidden_size must be <= 3072"):
        laguna_router_topk_bf16_hidden_correction_bias_persistent(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3073, 256, 10, 2.5
        )


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
@pytest.mark.parametrize("hidden_size", [17, 3_072])
def test_laguna_persistent_router_is_bit_exact_and_self_rearms(
    hidden_size: int,
    laguna_router_library,
    logits_library,
) -> None:
    rng = np.random.default_rng(20260724 + hidden_size)
    experts, top_k = 256, 10
    if hidden_size == 17:
        hidden = np.zeros((1, hidden_size), dtype=np.float32)
        hidden[0, 0] = np.float32(1.0)
        weight = np.zeros((experts, hidden_size), dtype=np.float32)
        weight[:, 0] = np.linspace(-100.0, 100.0, experts, dtype=np.float32)
        weight[[1, 33, 65, 97, 129, 161, 193, 225], 0] = np.float32(20.0)
    else:
        hidden = rng.normal(0.0, 0.04, size=(1, hidden_size)).astype(np.float32)
        weight = rng.normal(0.0, 0.03, size=(experts, hidden_size)).astype(np.float32)
    hidden_bits = _f32_to_bf16_u16(hidden)
    correction = rng.normal(0.0, 0.25, size=experts).astype(np.float32)
    correction[np.arange(0, 20, 2)] = np.linspace(3.0, 2.1, top_k, dtype=np.float32)

    split_logits = np.full((1, experts), np.nan, dtype=np.float32)
    split_probabilities = np.full((1, experts), np.nan, dtype=np.float32)
    split_selection = np.full((1, experts), np.nan, dtype=np.float32)
    split_selected = np.full((1, top_k), -1, dtype=np.int64)
    split_routing = np.full((1, top_k), np.nan, dtype=np.float32)
    split_scaled = np.full((1, top_k), np.nan, dtype=np.float32)
    fused_logits = split_logits.copy()
    fused_probabilities = split_probabilities.copy()
    fused_selection = split_selection.copy()
    fused_selected = split_selected.copy()
    fused_routing = split_routing.copy()
    fused_scaled = split_scaled.copy()
    counter = np.zeros(1, dtype=np.int32)
    arrays = (
        hidden_bits,
        weight,
        correction,
        split_logits,
        split_probabilities,
        split_selection,
        split_selected,
        split_routing,
        split_scaled,
        fused_logits,
        fused_probabilities,
        fused_selection,
        fused_selected,
        fused_routing,
        fused_scaled,
        counter,
    )
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        qwen35_router_logits_bf16_f32w_auto_256(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[3].ptr,
            1,
            hidden_size,
            experts,
            library=logits_library,
        )
        laguna_sigmoid_correction_topk_f32(
            buffers[3].ptr,
            buffers[2].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            buffers[6].ptr,
            buffers[7].ptr,
            buffers[8].ptr,
            1,
            experts,
            top_k,
            2.5,
            library=laguna_router_library,
        )

        fused_args = (
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[9].ptr,
            buffers[10].ptr,
            buffers[11].ptr,
            buffers[12].ptr,
            buffers[13].ptr,
            buffers[14].ptr,
            buffers[15].ptr,
            1,
            hidden_size,
            experts,
            top_k,
            2.5,
        )
        counter_after_launches = []
        for _ in range(2):
            laguna_router_topk_bf16_hidden_correction_bias_persistent(
                *fused_args,
                library=laguna_router_library,
            )
            copy_device_to_host(host_array_ptr(counter), buffers[15], counter.nbytes)
            counter_after_launches.append(int(counter[0]))
        for array, buffer in zip(arrays[3:], buffers[3:], strict=True):
            copy_device_to_host(host_array_ptr(array), buffer, array.nbytes)
    finally:
        for buffer in reversed(buffers):
            free(buffer)

    for actual, expected in zip(
        (
            fused_logits,
            fused_probabilities,
            fused_selection,
            fused_routing,
            fused_scaled,
        ),
        (
            split_logits,
            split_probabilities,
            split_selection,
            split_routing,
            split_scaled,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(fused_selected, split_selected)
    assert counter_after_launches == [0, 0]
    np.testing.assert_array_equal(counter, np.zeros(1, dtype=np.int32))
