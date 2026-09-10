from __future__ import annotations

import ctypes

import numpy as np
import pytest

import hipengine.kernels.hip_gfx1100.moe.router as router_module
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.moe import (
    build_qwen35_router,
    plan_qwen35_router_build,
    qwen35_router_logits_bf16,
    qwen35_router_logits_bf16_f32w,
    qwen35_router_logits_bf16_f32w_auto_256,
    qwen35_router_logits_bf16_f32w_token_tile_8,
    qwen35_router_logits_bf16_f32w_token_tile_16,
    qwen35_router_logits_f32_f32w,
    qwen35_router_logits_f32_f32w_token_tile4_dense_exact,
    qwen35_router_logits_fp16,
    qwen35_router_logits_fp16_f32w,
    qwen35_router_select,
    qwen35_router_topk_shared_coop_out_bf16,
    qwen35_router_topk_shared_coop_out_fp16,
    qwen35_router_topk_split_shared_coop_out_bf16,
    qwen35_router_topk_split_shared_coop_out_bf16_f32w,
    qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent,
    qwen35_router_topk_split_shared_coop_out_fp16,
    qwen35_router_topk_shared_out_bf16,
    qwen35_router_topk_shared_out_fp16,
    qwen35_router_topk_shared_sigmoid_out_bf16,
    qwen35_router_topk_shared_sigmoid_out_fp16,
    register_qwen35_router_kernels,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_f32
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
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="bf16", variant="bf16_hidden")
        is qwen35_router_logits_bf16
    )
    assert resolve(backend="hip_gfx1100", layer="router_logits", quant="fp16") is qwen35_router_logits_fp16
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="f32", variant="bf16_hidden")
        is qwen35_router_logits_bf16_f32w_auto_256
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden_token_tile_8",
        )
        is qwen35_router_logits_bf16_f32w_token_tile_8
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden_token_tile_16",
        )
        is qwen35_router_logits_bf16_f32w_token_tile_16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="f32", variant="fp16_hidden")
        is qwen35_router_logits_fp16_f32w
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="f32", variant="f32_hidden")
        is qwen35_router_logits_f32_f32w
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_logits",
            quant="f32",
            variant="f32_hidden_token_tile4_dense_exact",
        )
        is qwen35_router_logits_f32_f32w_token_tile4_dense_exact
    )
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
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_topk_split_shared",
            quant="f32",
            variant="coop_out_bf16_hidden",
        )
        is qwen35_router_topk_split_shared_coop_out_bf16_f32w
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_topk_split_shared",
            quant="f32",
            variant="coop_out_bf16_hidden_persistent",
        )
        is qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent
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


def test_lcp4_auto_router_wrapper_uses_256_threads_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        router_module,
        "qwen35_router_logits_bf16_f32w",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    qwen35_router_logits_bf16_f32w_auto_256(1, 2, 3, 512, 2048, 256)
    assert calls == [((1, 2, 3, 512, 2048, 256), {"threads": 256, "stream": 0, "library": None, "runtime": None})]


def test_qwen35_router_wrappers_validate_shape_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_bf16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        qwen35_router_logits_bf16(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_fp16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_bf16_f32w(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_f32_f32w(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        qwen35_router_logits_fp16_f32w(0, 0, 0, 1, 16, 8, threads=32)
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
    with pytest.raises(ValueError, match="F32-weight cooperative router requires 256 threads"):
        qwen35_router_topk_split_shared_coop_out_bf16_f32w(
            0, 0, 0, 0, 0, 0, 1, 2048, 256, 8, threads=512
        )
    with pytest.raises(ValueError, match="F32-weight cooperative router requires hidden_size <= 2048"):
        qwen35_router_topk_split_shared_coop_out_bf16_f32w(
            0, 0, 0, 0, 0, 0, 1, 4096, 256, 8, threads=256
        )
    with pytest.raises(ValueError, match="F32-weight cooperative router requires 256 threads"):
        qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent(
            0, 0, 0, 0, 0, 0, 0, 1, 2048, 256, 8, threads=512
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_router_f32_weight_wider_token_tiles_are_exact(router_library) -> None:
    rng = np.random.default_rng(20260726)
    tokens, hidden_size, experts = 17, 3_072, 256
    hidden = _f32_to_bf16_u16(
        rng.normal(0.0, 0.04, size=(tokens, hidden_size)).astype(np.float32)
    )
    weight = rng.normal(
        0.0,
        0.03,
        size=(experts, hidden_size),
    ).astype(np.float32)
    outputs = [
        np.zeros((tokens, experts), dtype=np.float32)
        for _ in range(3)
    ]
    arrays = (hidden, weight, *outputs)
    buffers = [malloc(array.nbytes) for array in arrays]
    try:
        for array, buffer in zip(arrays, buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        for launch, output_buffer in zip(
            (
                qwen35_router_logits_bf16_f32w_auto_256,
                qwen35_router_logits_bf16_f32w_token_tile_8,
                qwen35_router_logits_bf16_f32w_token_tile_16,
            ),
            buffers[2:],
            strict=True,
        ):
            launch(
                buffers[0].ptr,
                buffers[1].ptr,
                output_buffer.ptr,
                tokens,
                hidden_size,
                experts,
                library=router_library,
            )
        for output, output_buffer in zip(
            outputs,
            buffers[2:],
            strict=True,
        ):
            copy_device_to_host(
                host_array_ptr(output),
                output_buffer,
                output.nbytes,
            )
    finally:
        for buffer in reversed(buffers):
            free(buffer)

    assert np.array_equal(outputs[1], outputs[0])
    assert np.array_equal(outputs[2], outputs[0])


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_split_shared_coop_bf16_f32w_persistent_is_replay_exact_at_production_shape(router_library) -> None:
    rng = np.random.default_rng(20260714)
    hidden_size = 2048
    num_experts = 256
    top_k = 8
    hidden = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(hidden_size,)).astype(np.float32))
    expert = rng.normal(0.0, 0.2, size=(num_experts, hidden_size)).astype(np.float32)
    expert[0] = _bf16_u16_to_f32(hidden)
    expert[1] = expert[0]
    shared = rng.normal(0.0, 0.2, size=(hidden_size,)).astype(np.float32)
    control_logits = np.zeros((num_experts + 1,), dtype=np.float32)
    candidate_logits = np.zeros_like(control_logits)
    control_selected = np.zeros((top_k,), dtype=np.int64)
    candidate_selected = np.zeros_like(control_selected)
    control_routing = np.zeros((top_k,), dtype=np.float32)
    candidate_routing = np.zeros_like(control_routing)
    counter = np.zeros((1,), dtype=np.int32)
    arrays = (
        hidden,
        expert,
        shared,
        control_logits,
        candidate_logits,
        control_selected,
        candidate_selected,
        control_routing,
        candidate_routing,
        counter,
    )
    buffers = [malloc(arr.nbytes) for arr in arrays]
    try:
        for arr, buf in zip(arrays, buffers, strict=True):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        qwen35_router_logits_bf16_f32w(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[3].ptr,
            1,
            hidden_size,
            num_experts,
            threads=512,
            library=router_library,
        )
        qwen35_router_logits_bf16_f32w(
            buffers[0].ptr,
            buffers[2].ptr,
            buffers[3].ptr + num_experts * np.dtype(np.float32).itemsize,
            1,
            hidden_size,
            1,
            threads=512,
            library=router_library,
        )
        qwen35_router_select(
            buffers[3].ptr,
            buffers[5].ptr,
            buffers[7].ptr,
            1,
            num_experts,
            num_experts,
            top_k,
            threads=256,
            library=router_library,
        )
        qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[4].ptr,
            buffers[6].ptr,
            buffers[8].ptr,
            buffers[9].ptr,
            1,
            hidden_size,
            num_experts,
            top_k,
            threads=256,
            library=router_library,
        )
        candidate_logits.fill(np.nan)
        candidate_selected.fill(-1)
        candidate_routing.fill(np.nan)
        for arr, buf in (
            (candidate_logits, buffers[4]),
            (candidate_selected, buffers[6]),
            (candidate_routing, buffers[8]),
        ):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[4].ptr,
            buffers[6].ptr,
            buffers[8].ptr,
            buffers[9].ptr,
            1,
            hidden_size,
            num_experts,
            top_k,
            threads=256,
            library=router_library,
        )
        for arr, buf in (
            (control_logits, buffers[3]),
            (candidate_logits, buffers[4]),
            (control_selected, buffers[5]),
            (candidate_selected, buffers[6]),
            (control_routing, buffers[7]),
            (candidate_routing, buffers[8]),
            (counter, buffers[9]),
        ):
            copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    np.testing.assert_array_equal(candidate_logits.view(np.uint32), control_logits.view(np.uint32))
    np.testing.assert_array_equal(candidate_selected, control_selected)
    np.testing.assert_array_equal(candidate_routing.view(np.uint32), control_routing.view(np.uint32))
    assert counter.tolist() == [0]
    assert list(candidate_selected[:2]) == [0, 1]


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


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tokens", [1, 5, 508])
def test_router_logits_f32_token_tile4_is_bit_exact_to_dense(
    router_library, tokens: int
) -> None:
    rng = np.random.default_rng(20260831 + tokens)
    hidden_size = 2560
    num_experts = 512
    hidden = rng.normal(0.0, 0.1, size=(tokens, hidden_size)).astype(np.float32)
    weight = rng.normal(0.0, 0.1, size=(num_experts, hidden_size)).astype(np.float32)
    dense = np.empty((tokens, num_experts), dtype=np.float32)
    tiled = np.empty_like(dense)
    buffers = [malloc(arr.nbytes) for arr in (hidden, weight, dense, tiled)]
    try:
        for arr, buf in zip((hidden, weight), buffers, strict=False):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        dense_gemv_out_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            tokens,
            hidden_size,
            num_experts,
            threads=256,
        )
        qwen35_router_logits_f32_f32w_token_tile4_dense_exact(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[3].ptr,
            tokens,
            hidden_size,
            num_experts,
            threads=256,
            library=router_library,
        )
        copy_device_to_host(host_array_ptr(dense), buffers[2], dense.nbytes)
        copy_device_to_host(host_array_ptr(tiled), buffers[3], tiled.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    np.testing.assert_array_equal(tiled.view(np.uint32), dense.view(np.uint32))


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_router_logits_f32_f32w_matches_cpu(router_library) -> None:
    rng = np.random.default_rng(20260702)
    tokens = 2
    hidden_size = 128
    num_experts = 17
    hidden = rng.normal(0.0, 0.2, size=(tokens, hidden_size)).astype(np.float32)
    weight = rng.normal(0.0, 0.2, size=(num_experts, hidden_size)).astype(np.float32)
    logits = np.zeros((tokens, num_experts), dtype=np.float32)
    selected = np.zeros((tokens, 4), dtype=np.int64)
    routing = np.zeros((tokens, 4), dtype=np.float32)

    buffers = [malloc(arr.nbytes) for arr in (hidden, weight, logits, selected, routing)]
    try:
        for arr, buf in zip((hidden, weight, logits, selected, routing), buffers, strict=True):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
        qwen35_router_logits_f32_f32w(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            tokens,
            hidden_size,
            num_experts,
            threads=128,
            library=router_library,
        )
        qwen35_router_select(
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            tokens,
            num_experts,
            num_experts,
            4,
            threads=128,
            library=router_library,
        )
        copy_device_to_host(host_array_ptr(logits), buffers[2], logits.nbytes)
        copy_device_to_host(host_array_ptr(selected), buffers[3], selected.nbytes)
        copy_device_to_host(host_array_ptr(routing), buffers[4], routing.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    expected_logits = hidden @ weight.T
    expected_selected = np.argsort(-expected_logits, axis=-1, kind="stable")[:, :4].astype(np.int64)
    top_vals = np.take_along_axis(expected_logits, expected_selected, axis=-1)
    expected_routing = np.exp(top_vals - top_vals[:, :1]).astype(np.float32)
    expected_routing /= np.maximum(
        expected_routing.sum(axis=-1, keepdims=True, dtype=np.float32),
        np.float32(1.0e-20),
    )

    np.testing.assert_allclose(logits, expected_logits, atol=3.0e-5, rtol=3.0e-5)
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
