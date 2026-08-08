"""CUDA sm_120a Maple peer-backend contracts and GPU0 oracle gates."""

from __future__ import annotations

import ctypes
import inspect
import os

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey, resolve

_RUN_CUDA = os.environ.get("HIPENGINE_RUN_CUDA_MAPLE", "0") != "0"


def _require_cuda() -> None:
    if not _RUN_CUDA:
        pytest.skip("set HIPENGINE_RUN_CUDA_MAPLE=1 for the CUDA Maple gate")
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")


def test_cuda_sm120a_maple_build_plans_are_architecture_qualified(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.attention.maple_attention import (
        plan_maple_attention_build,
    )
    from hipengine.kernels.cuda_sm120a.linear.maple_lm_head import (
        plan_lm_head_build,
    )
    from hipengine.kernels.cuda_sm120a.moe.group_scatter import (
        plan_qwen35_moe_group_scatter_build,
    )
    from hipengine.kernels.cuda_sm120a.moe.maple_moe import plan_maple_moe_build
    from hipengine.kernels.cuda_sm120a.norm.maple_rmsnorm import (
        plan_qwen35_rmsnorm_build,
    )
    from hipengine.kernels.cuda_sm120a.quant.maple_ternary import (
        plan_maple_ternary_build,
    )

    plans = (
        plan_maple_ternary_build(cache_root=tmp_path, compiler_version="nvcc test"),
        plan_maple_attention_build(cache_root=tmp_path, compiler_version="nvcc test"),
        plan_maple_moe_build(cache_root=tmp_path, compiler_version="nvcc test"),
        plan_qwen35_moe_group_scatter_build(
            cache_root=tmp_path, compiler_version="nvcc test"
        ),
        plan_qwen35_rmsnorm_build(cache_root=tmp_path, compiler_version="nvcc test"),
        plan_lm_head_build(cache_root=tmp_path, compiler_version="nvcc test"),
    )
    assert all("-arch=sm_120a" in plan.command for plan in plans)
    assert all(any(str(source).endswith(".cu") for source in plan.sources) for plan in plans)
    assert len({plan.output_path for plan in plans}) == len(plans)


def test_cuda_sm120a_maple_c1_registry_keys_resolve() -> None:
    from hipengine.kernels.backends import backend_package_capability
    from hipengine.kernels.cuda_sm120a import register_backend_kernels
    from hipengine.runtime.maple_cuda import MapleCudaRunner

    register_backend_kernels(replace=True)
    runner_type = backend_package_capability(
        "cuda_sm120a", "maple_runner_type"
    )
    assert runner_type() is MapleCudaRunner
    keys = (
        KernelKey("cuda_sm120a", "maple_ternary_gemv", "maple_ternary2", "row_alpha"),
        KernelKey("cuda_sm120a", "maple_affine4_gemv", "maple_ternary2", "group64_wave32_exact"),
        KernelKey(
            "cuda_sm120a", "maple_qknorm_rope_kv_write", "maple_ternary2",
            "partial_rotate_half_bf16",
        ),
        KernelKey(
            "cuda_sm120a", "maple_attention_decode", "maple_ternary2",
            "gqa_spans_bf16",
        ),
        KernelKey(
            "cuda_sm120a", "maple_router_topk", "maple_ternary2",
            "bf16_fp32_single_dispatch",
        ),
        KernelKey(
            "cuda_sm120a", "maple_clamped_swiglu", "maple_ternary2",
            "clamp7_bf16",
        ),
        KernelKey(
            "cuda_sm120a", "maple_weighted_residual", "maple_ternary2",
            "two_bf16_boundaries",
        ),
        KernelKey(
            "cuda_sm120a", "moe_group_compact", "generic",
            "active_experts_i32_parallel",
        ),
        KernelKey("cuda_sm120a", "rmsnorm", "w4_paro", "paro_out"),
        KernelKey("cuda_sm120a", "argmax", "w4_paro", "f32_rows_i32"),
    )
    assert all(
        callable(
            resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )
        )
        for key in keys
    )


def test_cuda_sm120a_maple_native_prefill_uses_complete_bulk_chain() -> None:
    from hipengine.runtime.maple_cuda import MapleCudaRunner

    source = inspect.getsource(MapleCudaRunner.prefill_native)
    assert "return self.prefill(" not in source
    for required_call in (
        "maple_affine4_embed_batched_bf16",
        "maple_ternary_qkv_gemm_bf16",
        "maple_qknorm_rope_kv_write_batched_bf16",
        "maple_attention_prefill_ring_gqa4_bf16",
        "qwen35_moe_group_compact_active_i32_parallel",
        "maple_selected_ternary_dual_grouped_bf16",
        "maple_selected_ternary_grouped_bf16",
        "maple_weighted_residual_batched_bf16",
    ):
        assert required_call in source


def test_cuda_sm120a_maple_i32_stable_compaction_matches_cpu_order() -> None:
    _require_cuda()
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.cuda_sm120a.moe.group_scatter import (
        build_qwen35_moe_group_scatter,
        qwen35_moe_group_compact_active_i32_parallel,
    )

    selected = np.asarray(
        [[3, 1], [0, 3], [1, 1], [3, 0], [2, 3]], dtype=np.int32
    )
    routing = np.arange(selected.size, dtype=np.float32).reshape(selected.shape) / 17.0
    flat = selected.reshape(-1)
    expected_lanes = np.argsort(flat, kind="stable").astype(np.int64)
    expected_experts = flat[expected_lanes].astype(np.int64)
    counts = np.bincount(flat, minlength=4).astype(np.int64)
    expected_starts = np.zeros(5, dtype=np.int64)
    expected_starts[1:] = np.cumsum(counts, dtype=np.int64)
    expected_active = np.flatnonzero(counts).astype(np.int64)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    buffers = []
    try:
        def put(array: np.ndarray):
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
            return buffer

        selected_d = put(selected)
        routing_d = put(routing)
        outputs = (
            np.empty(5, dtype=np.int64),
            np.empty(4, dtype=np.int64),
            np.empty(1, dtype=np.int64),
            np.empty(flat.shape, dtype=np.int64),
            np.empty(flat.shape, dtype=np.int64),
            np.empty(flat.shape, dtype=np.float32),
        )
        output_buffers = []
        for array in outputs:
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            output_buffers.append(buffer)
        qwen35_moe_group_compact_active_i32_parallel(
            selected_d.ptr,
            routing_d.ptr,
            *(buffer.ptr for buffer in output_buffers),
            flat.size,
            4,
            library=build_qwen35_moe_group_scatter(load=True),
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in zip(outputs, output_buffers, strict=True):
            copy_device_to_host(
                host_array_ptr(host), device, nbytes=host.nbytes, runtime=runtime
            )

        starts, active, active_count, lanes, experts, weights = outputs
        assert int(active_count[0]) == expected_active.size
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(active[: expected_active.size], expected_active)
        np.testing.assert_array_equal(lanes, expected_lanes)
        np.testing.assert_array_equal(experts, expected_experts)
        np.testing.assert_array_equal(weights, routing.reshape(-1)[expected_lanes])
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def test_cuda_sm120a_maple_native_prefill_matches_serial_state() -> None:
    _require_cuda()
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        host_array_ptr,
        memory_stats,
    )
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple_cuda import MapleCudaRunner

    try:
        checkpoint = load_maple_checkpoint("/models/hf/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - optional public checkpoint gate
        pytest.skip(f"Maple checkpoint unavailable: {exc}")
    prompt = (9707, 13, 358, 1093, 220, 3100, 1066, 13, 366, 264, 1156, 15)
    baseline_bytes = memory_stats()["current_allocated_bytes"]

    serial = MapleCudaRunner.load(checkpoint, max_context=64)
    native = MapleCudaRunner.load(checkpoint, max_context=64)
    try:
        serial_result = serial.prefill(prompt)
        native_result = native.prefill_native(prompt)

        def copy_bytes(runner, buffer, *, offset=0, nbytes=None):
            size = buffer.nbytes - offset if nbytes is None else int(nbytes)
            host = np.empty(size, dtype=np.uint8)
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(ptr=buffer.ptr + offset, nbytes=size),
                nbytes=size,
                runtime=runner.runtime,
            )
            return host

        spec = checkpoint.spec
        hidden_bytes = spec.hidden_size * 2
        final_row_offset = (len(prompt) - 1) * hidden_bytes
        np.testing.assert_array_equal(
            copy_bytes(serial, serial.buffers.hidden, nbytes=hidden_bytes),
            copy_bytes(
                native,
                native.buffers.pf.hidden,
                offset=final_row_offset,
                nbytes=hidden_bytes,
            ),
        )
        np.testing.assert_array_equal(serial.copy_logits(), native.copy_logits())
        live_cache_bytes = len(prompt) * spec.kv_size * 2
        for serial_layer, native_layer in zip(
            serial.buffers.layers, native.buffers.layers, strict=True
        ):
            for name in ("key_cache", "value_cache"):
                np.testing.assert_array_equal(
                    copy_bytes(
                        serial,
                        getattr(serial_layer, name),
                        nbytes=live_cache_bytes,
                    ),
                    copy_bytes(
                        native,
                        getattr(native_layer, name),
                        nbytes=live_cache_bytes,
                    ),
                )
        for owner_name in ("sliding_span_owner", "global_span_owner"):
            serial_owner = getattr(serial.buffers, owner_name)
            native_owner = getattr(native.buffers, owner_name)
            for name in (
                "base_offsets",
                "live_counts",
                "token_positions",
                "evict_mask",
                "row_positions",
            ):
                np.testing.assert_array_equal(
                    copy_bytes(serial, getattr(serial_owner, name)),
                    copy_bytes(native, getattr(native_owner, name)),
                )
        assert native_result.position == serial_result.position == len(prompt) - 1
        assert native_result.token_id == serial_result.token_id
        assert native_result.top_logit == serial_result.top_logit

        serial_continuation = serial.step(serial_result.token_id)
        native_continuation = native.step(native_result.token_id)
        assert native_continuation.token_id == serial_continuation.token_id
        assert native_continuation.top_logit == serial_continuation.top_logit
        np.testing.assert_array_equal(serial.copy_logits(), native.copy_logits())
    finally:
        native.close()
        serial.close()
    assert memory_stats()["current_allocated_bytes"] == baseline_bytes


def test_cuda_sm120a_maple_ternary_norm_and_moe_match_numpy() -> None:
    _require_cuda()
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.cpu_reference.maple import (
        bf16_to_f32,
        clamped_swiglu,
        f32_to_bf16_bits,
        rmsnorm,
        ternary_gemv,
        weighted_residual,
    )
    from hipengine.kernels.cuda_sm120a.moe.maple_moe import (
        build_maple_moe,
        maple_clamped_swiglu_bf16,
        maple_weighted_residual_bf16,
    )
    from hipengine.kernels.cuda_sm120a.norm.maple_rmsnorm import (
        build_qwen35_rmsnorm,
        paro_rmsnorm_out_bf16,
    )
    from hipengine.kernels.cuda_sm120a.quant.maple_ternary import (
        build_maple_ternary,
        maple_ternary_gemv_bf16,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    rng = np.random.default_rng(120)
    hidden = 64
    out_features = 11
    top_k = 3
    x = f32_to_bf16_bits(rng.normal(0, 0.4, hidden).astype(np.float32))
    codes = rng.integers(0, 3, size=(out_features, hidden), dtype=np.uint32)
    packed = np.zeros((out_features, hidden // 16), dtype=np.uint32)
    for lane in range(16):
        packed |= codes[:, lane::16] << np.uint32(2 * lane)
    alpha = f32_to_bf16_bits(rng.uniform(0.02, 0.2, out_features).astype(np.float32))
    norm_weight = f32_to_bf16_bits(rng.uniform(0.8, 1.2, hidden).astype(np.float32))
    gate = f32_to_bf16_bits(rng.normal(0, 2, (top_k, hidden)).astype(np.float32))
    up = f32_to_bf16_bits(rng.normal(0, 2, (top_k, hidden)).astype(np.float32))
    residual = f32_to_bf16_bits(rng.normal(0, 0.5, hidden).astype(np.float32))
    routing = rng.uniform(0, 1, top_k).astype(np.float32)
    routing /= routing.sum()

    host_inputs = (x, packed, alpha, norm_weight, gate, up, residual, routing)
    buffers = []
    try:
        dev = []
        for array in host_inputs:
            buf = malloc(array.nbytes, runtime=runtime)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), runtime=runtime)
            dev.append(buf)
        ternary_out = malloc(out_features * 2, runtime=runtime)
        norm_out = malloc(hidden * 2, runtime=runtime)
        swiglu_out = malloc(top_k * hidden * 2, runtime=runtime)
        weighted_out = malloc(hidden * 2, runtime=runtime)
        buffers.extend((ternary_out, norm_out, swiglu_out, weighted_out))

        ternary_lib = build_maple_ternary(load=True)
        norm_lib = build_qwen35_rmsnorm(load=True)
        moe_lib = build_maple_moe(load=True)
        maple_ternary_gemv_bf16(
            dev[0].ptr, dev[1].ptr, dev[2].ptr, ternary_out.ptr,
            hidden, out_features, library=ternary_lib, runtime=runtime,
        )
        paro_rmsnorm_out_bf16(
            dev[0].ptr, dev[3].ptr, norm_out.ptr, 1, hidden,
            library=norm_lib, runtime=runtime,
        )
        maple_clamped_swiglu_bf16(
            dev[4].ptr, dev[5].ptr, swiglu_out.ptr, top_k, hidden,
            library=moe_lib, runtime=runtime,
        )
        maple_weighted_residual_bf16(
            dev[6].ptr, swiglu_out.ptr, dev[7].ptr, weighted_out.ptr,
            top_k, hidden, library=moe_lib, runtime=runtime,
        )
        runtime.device_synchronize()

        got_ternary = np.empty(out_features, dtype=np.uint16)
        got_norm = np.empty(hidden, dtype=np.uint16)
        got_swiglu = np.empty((top_k, hidden), dtype=np.uint16)
        got_weighted = np.empty(hidden, dtype=np.uint16)
        for output, buf in (
            (got_ternary, ternary_out),
            (got_norm, norm_out),
            (got_swiglu, swiglu_out),
            (got_weighted, weighted_out),
        ):
            copy_device_to_host(host_array_ptr(output), buf, nbytes=output.nbytes, runtime=runtime)

        expected_ternary = f32_to_bf16_bits(ternary_gemv(bf16_to_f32(x), packed, alpha))
        expected_norm = f32_to_bf16_bits(rmsnorm(bf16_to_f32(x), bf16_to_f32(norm_weight)))
        expected_swiglu = f32_to_bf16_bits(clamped_swiglu(bf16_to_f32(gate), bf16_to_f32(up)))
        expected_weighted = f32_to_bf16_bits(
            weighted_residual(
                bf16_to_f32(residual), bf16_to_f32(expected_swiglu), routing
            )
        )
        np.testing.assert_array_equal(got_ternary, expected_ternary)
        np.testing.assert_array_equal(got_norm, expected_norm)
        np.testing.assert_array_equal(got_swiglu, expected_swiglu)
        np.testing.assert_array_equal(got_weighted, expected_weighted)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def test_cuda_sm120a_maple_attention_consumes_kv_live_spans() -> None:
    _require_cuda()
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.cpu_reference.maple import (
        attention_decode,
        bf16_to_f32,
        f32_to_bf16_bits,
        qk_norm_rope,
    )
    from hipengine.kernels.cuda_sm120a.attention.maple_attention import (
        build_maple_attention,
        maple_attention_decode_bf16,
        maple_kv_span_update,
        maple_qknorm_rope_kv_write_bf16,
    )
    from hipengine.kvcache import KVLiveSpans

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    rng = np.random.default_rng(121)
    q_heads, kv_heads, head_dim, rope_dim = 4, 2, 4, 2
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    capacity = 4
    qkv = f32_to_bf16_bits(
        rng.normal(0, 0.4, q_size + 2 * kv_size).astype(np.float32)
    )
    q_weight = f32_to_bf16_bits(rng.uniform(0.8, 1.2, head_dim).astype(np.float32))
    k_weight = f32_to_bf16_bits(rng.uniform(0.8, 1.2, head_dim).astype(np.float32))
    arrays = (
        np.arange(capacity, dtype=np.int32),
        np.zeros(1, dtype=np.int64),
        np.full(capacity, -1, dtype=np.int64),
        np.ones(capacity, dtype=np.bool_),
        np.full(1, -1, dtype=np.int64),
        qkv,
        q_weight,
        k_weight,
    )
    buffers = []
    try:
        dev = []
        for array in arrays:
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
            dev.append(buffer)
        key_cache = malloc(capacity * kv_size * 2, runtime=runtime)
        value_cache = malloc(capacity * kv_size * 2, runtime=runtime)
        attention = malloc(q_size * 2, runtime=runtime)
        buffers.extend((key_cache, value_cache, attention))
        device = Device("cuda", 0)
        spans = KVLiveSpans.sliding_ring(
            base_offsets=Tensor.from_handle(dev[0].ptr, (capacity,), DType.INT32, device),
            live_counts=Tensor.from_handle(dev[1].ptr, (1,), DType.INT64, device),
            token_positions=Tensor.from_handle(dev[2].ptr, (capacity,), DType.INT64, device),
            evict_mask=Tensor.from_handle(dev[3].ptr, (capacity,), DType.BOOL, device),
            row_positions=Tensor.from_handle(dev[4].ptr, (1,), DType.INT64, device),
            capacity=capacity,
        )
        library = build_maple_attention(load=True)
        maple_kv_span_update(spans, position=0, library=library, runtime=runtime)
        maple_qknorm_rope_kv_write_bf16(
            dev[5].ptr,
            dev[6].ptr,
            dev[7].ptr,
            key_cache.ptr,
            value_cache.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            eps=1e-6,
            rope_theta=10_000.0,
            library=library,
            runtime=runtime,
        )
        maple_attention_decode_bf16(
            dev[5].ptr,
            key_cache.ptr,
            value_cache.ptr,
            attention.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        got_qkv = np.empty_like(qkv)
        got_attention = np.empty(q_size, dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(got_qkv), dev[5], nbytes=got_qkv.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(got_attention), attention,
            nbytes=got_attention.nbytes, runtime=runtime,
        )

        q_expected, k_expected = qk_norm_rope(
            bf16_to_f32(qkv[:q_size]).reshape(q_heads, head_dim),
            bf16_to_f32(qkv[q_size : q_size + kv_size]).reshape(kv_heads, head_dim),
            bf16_to_f32(q_weight),
            bf16_to_f32(k_weight),
            pos=0,
            rope_theta=10_000.0,
            rope_dim=rope_dim,
        )
        v_expected = bf16_to_f32(qkv[q_size + kv_size :]).reshape(
            1, kv_heads, head_dim
        )
        attention_expected = f32_to_bf16_bits(
            attention_decode(
                q_expected,
                k_expected.reshape(1, kv_heads, head_dim),
                v_expected,
                scale=head_dim**-0.5,
            )
        ).reshape(-1)
        np.testing.assert_array_equal(got_qkv[:q_size], f32_to_bf16_bits(q_expected).reshape(-1))
        np.testing.assert_array_equal(
            got_qkv[q_size : q_size + kv_size], f32_to_bf16_bits(k_expected).reshape(-1)
        )
        np.testing.assert_array_equal(got_attention, attention_expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def test_cuda_sm120a_maple_affine4_head_and_argmax_match_numpy() -> None:
    _require_cuda()
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.cpu_reference.maple import (
        affine4_gemv_f32,
        bf16_to_f32,
        f32_to_bf16_bits,
    )
    from hipengine.kernels.cuda_sm120a.linear.maple_lm_head import (
        argmax_f32,
        build_lm_head,
    )
    from hipengine.kernels.cuda_sm120a.quant.maple_ternary import (
        build_maple_ternary,
        maple_affine4_gemv_f32,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    rng = np.random.default_rng(122)
    hidden, vocab = 64, 37
    x = f32_to_bf16_bits(rng.normal(0, 0.3, hidden).astype(np.float32))
    codes = rng.integers(0, 16, size=(vocab, hidden), dtype=np.uint32)
    packed = np.zeros((vocab, hidden // 8), dtype=np.uint32)
    for lane in range(8):
        packed |= codes[:, lane::8] << np.uint32(4 * lane)
    scales = f32_to_bf16_bits(rng.uniform(0.01, 0.08, (vocab, 1)).astype(np.float32))
    biases = f32_to_bf16_bits(rng.uniform(-0.2, 0.2, (vocab, 1)).astype(np.float32))
    inputs = (x, packed, scales, biases)
    buffers = []
    try:
        dev = []
        for array in inputs:
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
            dev.append(buffer)
        logits = malloc(vocab * 4, runtime=runtime)
        stage1_blocks = (vocab + 1023) // 1024
        block_values = malloc(stage1_blocks * 4, runtime=runtime)
        block_indices = malloc(stage1_blocks * 8, runtime=runtime)
        out_index = malloc(8, runtime=runtime)
        out_value = malloc(4, runtime=runtime)
        buffers.extend((logits, block_values, block_indices, out_index, out_value))
        maple_affine4_gemv_f32(
            dev[0].ptr,
            dev[1].ptr,
            dev[2].ptr,
            dev[3].ptr,
            logits.ptr,
            hidden,
            vocab,
            library=build_maple_ternary(load=True),
            runtime=runtime,
        )
        argmax_f32(
            logits.ptr,
            block_values.ptr,
            block_indices.ptr,
            out_index.ptr,
            out_value.ptr,
            vocab,
            library=build_lm_head(load=True),
            runtime=runtime,
        )
        runtime.device_synchronize()
        got_logits = np.empty(vocab, dtype=np.float32)
        got_index = np.empty(1, dtype=np.int64)
        got_value = np.empty(1, dtype=np.float32)
        for output, buffer in (
            (got_logits, logits),
            (got_index, out_index),
            (got_value, out_value),
        ):
            copy_device_to_host(
                host_array_ptr(output), buffer, nbytes=output.nbytes, runtime=runtime
            )
        expected = affine4_gemv_f32(
            bf16_to_f32(x), packed, scales, biases
        )
        np.testing.assert_allclose(got_logits, expected, atol=2e-4, rtol=2e-4)
        assert int(got_index[0]) == int(np.argmax(expected))
        assert got_value[0] == got_logits[got_index[0]]
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
