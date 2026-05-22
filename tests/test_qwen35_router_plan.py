from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.moe import (
    build_qwen35_router,
    plan_qwen35_router_build,
    qwen35_router_logits_bf16,
    qwen35_router_logits_fp16,
    qwen35_router_select,
    qwen35_router_topk_shared_coop_out_bf16,
    qwen35_router_topk_shared_coop_out_fp16,
    qwen35_router_topk_split_shared_coop_out_bf16,
    qwen35_router_topk_split_shared_coop_out_fp16,
    qwen35_router_topk_shared_out_bf16,
    qwen35_router_topk_shared_out_fp16,
    qwen35_router_topk_shared_sigmoid_out_bf16,
    qwen35_router_topk_shared_sigmoid_out_fp16,
    register_qwen35_router_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def router_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_qwen35_router(load=True)


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_router_registers_bf16_and_w4_paro() -> None:
    register_qwen35_router_kernels()

    assert resolve(backend="hip_gfx1100", layer="router_logits", quant="bf16") is qwen35_router_logits_bf16
    assert resolve(backend="hip_gfx1100", layer="router_logits", quant="fp16") is qwen35_router_logits_fp16
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="w4_paro", variant="fp16_hidden")
        is qwen35_router_logits_fp16
    )
    assert resolve(backend="hip_gfx1100", layer="router_select", quant="fp32") is qwen35_router_select
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="bf16", variant="out")
        is qwen35_router_topk_shared_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="out")
        is qwen35_router_topk_shared_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="out_fp16_hidden")
        is qwen35_router_topk_shared_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="prefill_sigmoid_out")
        is qwen35_router_topk_shared_sigmoid_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_topk_shared",
            quant="w4_paro",
            variant="prefill_sigmoid_out_fp16_hidden",
        )
        is qwen35_router_topk_shared_sigmoid_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="coop_out")
        is qwen35_router_topk_shared_coop_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="coop_out_fp16_hidden")
        is qwen35_router_topk_shared_coop_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="fp16", variant="out")
        is qwen35_router_topk_shared_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="fp16", variant="prefill_sigmoid_out")
        is qwen35_router_topk_shared_sigmoid_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="fp16", variant="coop_out")
        is qwen35_router_topk_shared_coop_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_split_shared", quant="bf16", variant="coop_out")
        is qwen35_router_topk_split_shared_coop_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_topk_split_shared",
            quant="w4_paro",
            variant="coop_out_fp16_hidden",
        )
        is qwen35_router_topk_split_shared_coop_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_split_shared", quant="fp16", variant="coop_out")
        is qwen35_router_topk_split_shared_coop_out_fp16
    )


def test_qwen35_router_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_router_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc router test version",
    )

    assert artifact.family == "qwen35_router"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_router.so"
    assert artifact.compiler_version == "hipcc router test version"
    assert any(str(path).endswith("router.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_router_wrappers_validate_shape_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_bf16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        qwen35_router_logits_bf16(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_fp16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="top_k must be <= 16"):
        qwen35_router_select(0, 0, 0, 1, 8, 8, 17)
    with pytest.raises(ValueError, match="top_k must be <= num_experts"):
        qwen35_router_select(0, 0, 0, 1, 8, 2, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_out_bf16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="prefill shared-gate sigmoid"):
        qwen35_router_topk_shared_sigmoid_out_fp16(0, 0, 0, 0, 0, 1, 16, 9, 8, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_sigmoid_out_bf16(0, 0, 0, 0, 0, 2, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="decode-only"):
        qwen35_router_topk_shared_coop_out_bf16(0, 0, 0, 0, 0, 2, 16, 9, 8, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_coop_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="split cooperative router is decode-only"):
        qwen35_router_topk_split_shared_coop_out_bf16(0, 0, 0, 0, 0, 0, 2, 16, 8, 4)
    with pytest.raises(ValueError, match="top_k must be <= num_experts"):
        qwen35_router_topk_split_shared_coop_out_fp16(0, 0, 0, 0, 0, 0, 1, 16, 2, 4)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_split_shared_coop_bf16_matches_cpu_router(router_library) -> None:
    rng = np.random.default_rng(20260520)
    hidden_size = 128
    num_experts = 11
    top_k = 4
    hidden_f32 = rng.normal(0.0, 0.2, size=(hidden_size,)).astype(np.float32)
    expert_f32 = rng.normal(0.0, 0.2, size=(num_experts, hidden_size)).astype(np.float32)
    shared_f32 = rng.normal(0.0, 0.2, size=(hidden_size,)).astype(np.float32)
    hidden = _f32_to_bf16_u16(hidden_f32)
    expert = _f32_to_bf16_u16(expert_f32)
    shared = _f32_to_bf16_u16(shared_f32)

    logits = np.zeros((num_experts + 1,), dtype=np.float32)
    selected = np.zeros((top_k,), dtype=np.int64)
    routing = np.zeros((top_k,), dtype=np.float32)

    buffers = [malloc(arr.nbytes) for arr in (hidden, expert, shared, logits, selected, routing)]
    try:
        for arr, buf in zip((hidden, expert, shared, logits, selected, routing), buffers, strict=True):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        qwen35_router_topk_split_shared_coop_out_bf16(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            1,
            hidden_size,
            num_experts,
            top_k,
            threads=128,
            library=router_library,
        )
        copy_device_to_host(host_array_ptr(logits), buffers[3], logits.nbytes)
        copy_device_to_host(host_array_ptr(selected), buffers[4], selected.nbytes)
        copy_device_to_host(host_array_ptr(routing), buffers[5], routing.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    hidden_ref = _bf16_u16_to_f32(hidden)
    expert_ref = _bf16_u16_to_f32(expert)
    shared_ref = _bf16_u16_to_f32(shared)
    expert_logits = expert_ref @ hidden_ref
    expected_logits = np.concatenate([expert_logits, [np.float32(np.dot(shared_ref, hidden_ref))]])
    expected_selected = []
    work = expert_logits.copy()
    for _ in range(top_k):
        idx = int(np.argmax(work))
        expected_selected.append(idx)
        work[idx] = -np.inf
    expected_selected = np.asarray(expected_selected, dtype=np.int64)
    top_vals = expert_logits[expected_selected]
    expected_routing = np.exp(top_vals - top_vals[0]).astype(np.float32)
    expected_routing /= np.maximum(expected_routing.sum(dtype=np.float32), np.float32(1.0e-20))

    np.testing.assert_allclose(logits, expected_logits, atol=2.0e-5, rtol=2.0e-5)
    np.testing.assert_array_equal(selected, expected_selected)
    np.testing.assert_allclose(routing, expected_routing, atol=1.0e-6, rtol=1.0e-6)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()
