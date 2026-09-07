"""Native prefill must not allocate a context-sized score array in LDS."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans


@pytest.mark.parametrize("context", [15872, 16128, 81920])
def test_native_prefill_long_context_uniform_attention(context, hip_test_target_arch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
    )
    runtime = get_hip_runtime()
    bufs = []
    def upload(array):
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes, runtime=runtime)
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
        return buf
    def allocate(nbytes):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf
    def tensor(buf, shape, dtype):
        return Tensor.from_handle(buf.ptr, shape=shape, dtype=dtype, device=Device("hip", 0))
    heads, dim, block = 2, 256, 256
    blocks = (context + block - 1) // block
    try:
        # Q=0 gives uniform attention. Exact BF16-valued V and gate=0 make
        # the independent oracle mean(V)/2; no model or quant noise involved.
        rng = np.random.default_rng(1701)
        values = rng.integers(-8, 9, size=(blocks * block, 1, dim)).astype(np.float32)
        bits = (values.view(np.uint32) >> 16).astype(np.uint16)
        query = upload(np.zeros((1, heads, dim), np.float32))
        key = upload(np.zeros_like(bits))
        value = upload(bits)
        gate = upload(np.zeros((1, heads, dim), np.uint16))
        out = allocate(heads * dim * 2)
        table = upload(np.arange(blocks, dtype=np.int32).reshape(1, blocks))
        counts = upload(np.array([context], np.int64))
        positions = upload(np.array([context - 1], np.int64))
        spans = KVLiveSpans.paged_uniform(
            block_table=tensor(table, (1, blocks), "int32"),
            live_counts=tensor(counts, (1,), "int64"),
            max_live_count=context,
            storage_dtype="bf16",
            row_positions=tensor(positions, (1,), "int64"),
            span_role="prefill",
        )
        partial = allocate(heads * blocks * dim * 4)
        partial_m = allocate(heads * blocks * 4)
        partial_l = allocate(heads * blocks * 4)
        with hip_target_arch_environment(hip_test_target_arch):
            library = build_qwen35_paged_attn_decode(load=True)
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans(
            query.ptr, key.ptr, value.ptr, gate.ptr, out.ptr, spans,
            1, context, block, heads, 1, dim, dim, 1, dim ** -0.5,
            split_partial_out_ptr=partial.ptr,
            split_partial_m_ptr=partial_m.ptr,
            split_partial_l_ptr=partial_l.ptr,
            split_batch_rows=1, split_count=blocks,
            runtime=runtime, library=library,
        )
        runtime.device_synchronize()
        result = np.empty((heads, dim), np.uint16)
        copy_device_to_host(host_array_ptr(result), out, result.nbytes, runtime=runtime)
        actual = (result.astype(np.uint32) << 16).view(np.float32)
        expected = np.broadcast_to(values[:context, 0].mean(axis=0) * 0.5, actual.shape)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=0.006, atol=2e-5)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


@pytest.mark.parametrize("context", [128, 512, 1024, 2048, 15872, 16064, 16128, 81920])
def test_global_scores_parent_parity_and_cpu_reference(context, hip_test_target_arch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.cpu_reference.ops import full_attn_prefill
    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        build_qwen35_paged_attn_decode,
        qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans as launch,
    )
    runtime = get_hip_runtime()
    bufs = []
    def upload(array):
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes, runtime=runtime)
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=runtime)
        return buf
    def allocate(nbytes):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf
    def tensor(buf, shape, dtype):
        return Tensor.from_handle(buf.ptr, shape, dtype, Device("hip", 0))
    def bf16(array):
        bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    def f32(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32)
    rows, heads, kv_heads, dim, block = 3, 4, 2, 256, 256
    blocks = (context + block - 1) // block
    rng = np.random.default_rng(4709)
    query = f32(bf16(rng.normal(0, 0.2, (rows, heads, dim))))
    keys = bf16(rng.normal(0, 0.2, (blocks, block, kv_heads, dim)))
    values = bf16(rng.normal(0, 0.2, keys.shape))
    gates = bf16(rng.normal(0, 0.2, (rows, heads, dim)))
    # Ragged visibility and physical-page permutations exercise row slicing;
    # the last row crosses a workspace batch boundary (capacity = two rows).
    counts = np.array([context - 9, context, context - 3], np.int64)
    positions = counts - np.array([1, 5, 2], np.int64)
    tables = np.stack([rng.permutation(blocks).astype(np.int32) for _ in range(rows)])
    try:
        q, k, v, g = [upload(x) for x in (query, keys, values, gates)]
        table, count, position = [upload(x) for x in (tables, counts, positions)]
        spans = KVLiveSpans.paged_uniform(
            block_table=tensor(table, tables.shape, "int32"),
            live_counts=tensor(count, counts.shape, "int64"),
            row_positions=tensor(position, positions.shape, "int64"),
            max_live_count=context, storage_dtype="bf16", span_role="prefill",
        )
        out = allocate(rows * heads * dim * 2)
        partial = allocate(2 * heads * blocks * dim * 4)
        partial_m = allocate(2 * heads * blocks * 4)
        partial_l = allocate(2 * heads * blocks * 4)
        with hip_target_arch_environment(hip_test_target_arch):
            lib = build_qwen35_paged_attn_decode(load=True)
        def run(force_global, batch_rows=2):
            launch(
                q.ptr, k.ptr, v.ptr, g.ptr, out.ptr, spans,
                rows, context, block, heads, kv_heads, dim, dim, 1, dim ** -0.5,
                split_partial_out_ptr=partial.ptr, split_partial_m_ptr=partial_m.ptr,
                split_partial_l_ptr=partial_l.ptr, split_batch_rows=batch_rows,
                split_count=blocks, runtime=runtime, library=lib,
                global_score_workspace=force_global,
            )
            runtime.device_synchronize()
            result = np.empty((rows, heads, dim), np.uint16)
            copy_device_to_host(host_array_ptr(result), out, result.nbytes, runtime=runtime)
            return result
        actual = run(True)
        np.testing.assert_array_equal(run(True), actual)  # repeat bytes
        np.testing.assert_array_equal(run(True, 1), actual)  # workspace batching
        if context <= 16064:
            np.testing.assert_array_equal(run(False), actual)  # shared-score parent
        reference = np.concatenate([
            full_attn_prefill(
                query[row:row+1], f32(gates[row:row+1]), keys, values,
                positions[row:row+1], context_counts=counts[row:row+1],
                block_table=tables[row], block_size=block,
                output_dtype=None,
            )
            for row in range(rows)
        ])
        output = f32(actual)
        assert np.isfinite(output).all()
        # CPU oracle rounds its attention result before the sigmoid gate;
        # native fuses that boundary. Allow those two BF16 rounding errors.
        np.testing.assert_allclose(output, reference, rtol=0.015, atol=2e-5)
        def probabilities(x):
            x = x.reshape(rows, -1).astype(np.float64)
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        teacher, candidate = probabilities(reference), probabilities(output)
        kl = np.sum(teacher * np.log(teacher / candidate), axis=1)
        assert np.max(kl) <= 0.05
        assert np.mean(teacher.argmax(axis=1) == candidate.argmax(axis=1)) >= 0.9
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
