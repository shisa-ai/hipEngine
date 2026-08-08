"""gfx11 correctness gates for Maple QK/RoPE/KV/attention kernels."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest

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
from hipengine.kernels.backends import (
    hip_target_arch_environment,
    load_backend_kernel_package,
)
from hipengine.kernels.cpu_reference.maple import (
    attention_decode,
    bf16_round,
    bf16_to_f32,
    f32_to_bf16_bits,
    qk_norm_rope,
    rmsnorm,
)
from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
    build_maple_attention,
    maple_attention_decode_bf16,
    maple_attention_fused_qknorm_decode_bf16,
    maple_kv_span_update,
    maple_qknorm_rope_kv_write_bf16,
    plan_maple_attention_build,
)
from hipengine.kernels.hip_gfx1100.norm.rmsnorm import (
    build_qwen35_rmsnorm,
    paro_rmsnorm_out_bf16,
)
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans


class DeviceArrays:
    def __init__(self) -> None:
        self.buffers = []

    def put(self, array: np.ndarray):
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        self.buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(host))
        return buffer

    def empty(self, shape: tuple[int, ...], dtype: np.dtype):
        host = np.empty(shape, dtype=dtype)
        buffer = malloc(host.nbytes)
        self.buffers.append(buffer)
        return host, buffer

    def get(self, host: np.ndarray, buffer) -> np.ndarray:
        copy_device_to_host(host_array_ptr(host), buffer)
        return host

    def close(self) -> None:
        for buffer in reversed(self.buffers):
            free(buffer)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def make_spans(
    dev: DeviceArrays,
    *,
    base_offsets: np.ndarray,
    live_counts: np.ndarray,
    token_positions: np.ndarray,
    evict_mask: np.ndarray,
    row_positions: np.ndarray,
) -> KVLiveSpans:
    device = Device("hip", 0)
    base_d = dev.put(base_offsets)
    live_d = dev.put(live_counts)
    token_d = dev.put(token_positions)
    evict_d = dev.put(evict_mask)
    row_d = dev.put(row_positions)
    capacity = int(base_offsets.size)
    return KVLiveSpans.sliding_ring(
        base_offsets=Tensor.from_handle(base_d.ptr, (capacity,), DType.INT32, device),
        live_counts=Tensor.from_handle(live_d.ptr, (1,), DType.INT64, device),
        token_positions=Tensor.from_handle(token_d.ptr, (capacity,), DType.INT64, device),
        evict_mask=Tensor.from_handle(evict_d.ptr, (capacity,), DType.BOOL, device),
        row_positions=Tensor.from_handle(row_d.ptr, (1,), DType.INT64, device),
        capacity=capacity,
    )


def test_maple_attention_build_plan_and_gfx1151_registry_alias() -> None:
    plan = plan_maple_attention_build(compiler_version="hipcc-test")
    assert plan.family == "maple_attention"
    assert plan.output_path.name == "maple_attention.so"
    assert plan.sources[0].name == "maple_attention.hip"

    load_backend_kernel_package("hip_gfx1151")
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_attention_decode",
        quant="maple_ternary2",
        variant="gqa_spans_bf16",
    ) is maple_attention_decode_bf16
    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_prefill_ring_gqa4_bf16,
    )

    assert resolve(
        backend="hip_gfx1151",
        layer="maple_attention_prefill",
        quant="maple_ternary2",
        variant="gqa4_wave32_causal_ring_bf16",
    ) is maple_attention_prefill_ring_gqa4_bf16


@pytest.fixture(scope="module")
def maple_attention_lib(hip_test_target_arch):
    with hip_target_arch_environment(hip_test_target_arch):
        return build_maple_attention(load=True)


@pytest.fixture(scope="module")
def maple_norm_lib(hip_test_target_arch):
    with hip_target_arch_environment(hip_test_target_arch):
        return build_qwen35_rmsnorm(load=True)


def test_maple_standard_rmsnorm_is_bf16_bit_exact(maple_norm_lib) -> None:
    rng = np.random.default_rng(33)
    rows, hidden = 3, 2048
    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    weight_f32 = rng.uniform(0.5, 1.5, size=hidden).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    weight = f32_to_bf16_bits(weight_f32)
    expected = f32_to_bf16_bits(
        rmsnorm(bf16_round(x_f32), bf16_round(weight_f32))
    )

    with DeviceArrays() as dev:
        x_d, weight_d = dev.put(x), dev.put(weight)
        out, out_d = dev.empty((rows, hidden), np.dtype(np.uint16))
        paro_rmsnorm_out_bf16(
            x_d.ptr,
            weight_d.ptr,
            out_d.ptr,
            rows,
            hidden,
            library=maple_norm_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


def test_maple_span_update_publishes_ring_metadata(maple_attention_lib) -> None:
    capacity = 4
    base = np.asarray([2, 0, 3, 1], dtype=np.int32)
    live = np.asarray([0], dtype=np.int64)
    positions = np.full(capacity, -1, dtype=np.int64)
    evict = np.ones(capacity, dtype=np.bool_)
    row = np.asarray([-1], dtype=np.int64)
    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=live,
            token_positions=positions,
            evict_mask=evict,
            row_positions=row,
        )
        for position in range(6):
            maple_kv_span_update(
                spans, position=position, library=maple_attention_lib
            )
        live_out = np.empty_like(live)
        positions_out = np.empty_like(positions)
        evict_out = np.empty_like(evict)
        row_out = np.empty_like(row)
        dev.get(live_out, dev.buffers[1])
        dev.get(positions_out, dev.buffers[2])
        dev.get(evict_out, dev.buffers[3])
        dev.get(row_out, dev.buffers[4])

    assert live_out.tolist() == [4]
    assert row_out.tolist() == [5]
    assert positions_out.tolist() == [4, 5, 2, 3]
    assert not np.any(evict_out)


def test_maple_kv_span_update_batched_publishes_range(maple_attention_lib) -> None:
    """Batched span publish marks [start, start+rows) valid (P4)."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_kv_span_update_batched,
    )

    capacity, start, rows = 8, 3, 4
    base = np.asarray(list(range(capacity)), dtype=np.int32)
    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([0], dtype=np.int64),
            token_positions=np.full(capacity, -1, dtype=np.int64),
            evict_mask=np.ones(capacity, dtype=np.bool_),
            row_positions=np.asarray([-1], dtype=np.int64),
        )
        maple_kv_span_update_batched(
            spans,
            start=start,
            rows=rows,
            library=maple_attention_lib,
        )
        live_out = np.empty(1, dtype=np.int64)
        positions_out = np.empty(capacity, dtype=np.int64)
        evict_out = np.empty(capacity, dtype=np.bool_)
        row_out = np.empty(1, dtype=np.int64)
        dev.get(live_out, dev.buffers[1])
        dev.get(positions_out, dev.buffers[2])
        dev.get(evict_out, dev.buffers[3])
        dev.get(row_out, dev.buffers[4])

    expected_positions = np.full(capacity, -1, dtype=np.int64)
    for p in range(start, start + rows):
        expected_positions[p % capacity] = p
    assert live_out.tolist() == [start + rows]
    assert row_out.tolist() == [start + rows - 1]
    assert positions_out.tolist() == expected_positions.tolist()
    assert evict_out.tolist() == [not (start <= p < start + rows) for p in range(capacity)]



def test_maple_qknorm_partial_rope_and_kv_write_match_oracle(maple_attention_lib) -> None:
    rng = np.random.default_rng(44)
    q_heads, kv_heads, head_dim, rope_dim = 4, 2, 4, 2
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    qkv_f32 = rng.normal(size=q_size + 2 * kv_size).astype(np.float32)
    qkv = f32_to_bf16_bits(qkv_f32)
    q_weight_f32 = rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32)
    k_weight_f32 = rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32)
    q_weight = f32_to_bf16_bits(q_weight_f32)
    k_weight = f32_to_bf16_bits(k_weight_f32)
    position, capacity = 1, 4
    expected_q, expected_k = qk_norm_rope(
        bf16_round(qkv_f32[:q_size]).reshape(q_heads, head_dim),
        bf16_round(qkv_f32[q_size : q_size + kv_size]).reshape(kv_heads, head_dim),
        bf16_round(q_weight_f32),
        bf16_round(k_weight_f32),
        pos=position,
        rope_theta=10000.0,
        rope_dim=rope_dim,
    )
    expected_v = qkv[q_size + kv_size :]
    base = np.asarray([2, 0, 3, 1], dtype=np.int32)

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([0], dtype=np.int64),
            token_positions=np.full(capacity, -1, dtype=np.int64),
            evict_mask=np.ones(capacity, dtype=np.bool_),
            row_positions=np.asarray([-1], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        qw_d, kw_d = dev.put(q_weight), dev.put(k_weight)
        key_cache, key_d = dev.empty((capacity, kv_size), np.dtype(np.uint16))
        value_cache, value_d = dev.empty((capacity, kv_size), np.dtype(np.uint16))
        maple_kv_span_update(spans, position=position, library=maple_attention_lib)
        maple_qknorm_rope_kv_write_bf16(
            qkv_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            key_d.ptr,
            value_d.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            eps=1e-6,
            rope_theta=10000.0,
            library=maple_attention_lib,
        )
        qkv_out = np.empty_like(qkv)
        dev.get(qkv_out, qkv_d)
        dev.get(key_cache, key_d)
        dev.get(value_cache, value_d)

    physical_slot = int(base[position % capacity])
    assert np.array_equal(qkv_out[:q_size], f32_to_bf16_bits(expected_q.reshape(-1)))
    assert np.array_equal(
        qkv_out[q_size : q_size + kv_size], f32_to_bf16_bits(expected_k.reshape(-1))
    )
    assert np.array_equal(qkv_out[q_size + kv_size :], expected_v)
    assert np.array_equal(key_cache[physical_slot], f32_to_bf16_bits(expected_k.reshape(-1)))
    assert np.array_equal(value_cache[physical_slot], expected_v)


def test_maple_qknorm_rope_kv_write_batched_matches_oracle(maple_attention_lib) -> None:
    """Batched qknorm+RoPE+KV write over T rows is bit-exact vs per-row oracle (P2)."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_qknorm_rope_kv_write_batched_bf16,
    )

    rng = np.random.default_rng(88)
    q_heads, kv_heads, head_dim, rope_dim = 4, 2, 4, 2
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    rows, start, capacity = 5, 2, 8
    qkv_f32 = rng.normal(size=(rows, q_size + 2 * kv_size)).astype(np.float32)
    qkv = f32_to_bf16_bits(qkv_f32)
    qw_f32 = bf16_round(rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32))
    kw_f32 = bf16_round(rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32))
    q_weight = f32_to_bf16_bits(qw_f32)
    k_weight = f32_to_bf16_bits(kw_f32)
    base = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    token_positions = np.full(capacity, -1, dtype=np.int64)
    for r in range(rows):
        token_positions[(start + r) % capacity] = start + r
    expected_q = np.zeros((rows, q_size), dtype=np.uint16)
    expected_key = np.zeros((rows, kv_size), dtype=np.uint16)
    expected_val = np.zeros((rows, kv_size), dtype=np.float32)
    for r in range(rows):
        eq, ek = qk_norm_rope(
            bf16_round(qkv_f32[r, :q_size]).reshape(q_heads, head_dim),
            bf16_round(qkv_f32[r, q_size : q_size + kv_size]).reshape(kv_heads, head_dim),
            qw_f32,
            kw_f32,
            pos=start + r,
            rope_theta=10000.0,
            rope_dim=rope_dim,
        )
        expected_q[r] = f32_to_bf16_bits(eq.reshape(-1))
        expected_key[r] = f32_to_bf16_bits(ek.reshape(-1))
        expected_val[r] = qkv_f32[r, q_size + kv_size :]

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([rows], dtype=np.int64),
            token_positions=token_positions,
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([start + rows - 1], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        qw_d, kw_d = dev.put(q_weight), dev.put(k_weight)
        key_cache, key_d = dev.empty((capacity, kv_size), np.dtype(np.uint16))
        value_cache, value_d = dev.empty((capacity, kv_size), np.dtype(np.uint16))
        maple_qknorm_rope_kv_write_batched_bf16(
            qkv_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            key_d.ptr,
            value_d.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            eps=1e-6,
            rope_theta=10000.0,
            start=start,
            rows=rows,
            library=maple_attention_lib,
        )
        qkv_out = np.empty_like(qkv)
        dev.get(qkv_out, qkv_d)
        dev.get(key_cache, key_d)
        dev.get(value_cache, value_d)

    assert np.array_equal(qkv_out[:, :q_size], expected_q)
    for r in range(rows):
        physical = int(base[(start + r) % capacity])
        assert np.array_equal(key_cache[physical], expected_key[r])
        assert np.array_equal(value_cache[physical], f32_to_bf16_bits(expected_val[r]))


def test_maple_prefill_attention_ring_matches_causal_oracle(maple_attention_lib) -> None:
    """Ring-aware batched prefill attention is bit-exact vs the causal oracle (P2)."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_prefill_ring_bf16,
    )

    rng = np.random.default_rng(99)
    rows, q_heads, kv_heads, head_dim = 5, 4, 2, 4
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    start, capacity = 3, 8
    q = bf16_round(rng.normal(size=(rows, q_heads, head_dim)).astype(np.float32))
    total = start + rows
    keys = bf16_round(rng.normal(size=(total, kv_heads, head_dim)).astype(np.float32))
    values = bf16_round(rng.normal(size=(total, kv_heads, head_dim)).astype(np.float32))
    scale = head_dim**-0.5
    expected = np.stack(
        [
            attention_decode(
                q[r], keys[: start + r + 1], values[: start + r + 1], scale=scale
            )
            for r in range(rows)
        ]
    ).reshape(-1)
    expected = f32_to_bf16_bits(expected)
    base = np.asarray(list(range(capacity)), dtype=np.int32)
    token_positions = np.arange(capacity, dtype=np.int64)
    key_cache = keys.copy()
    value_cache = values.copy()
    qkv = np.zeros((rows, q_size + 2 * kv_size), dtype=np.uint16)
    qkv[:, :q_size] = f32_to_bf16_bits(q.reshape(rows, -1))

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([total], dtype=np.int64),
            token_positions=token_positions,
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([start + rows - 1], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        key_d = dev.put(f32_to_bf16_bits(key_cache))
        value_d = dev.put(f32_to_bf16_bits(value_cache))
        out, out_d = dev.empty((rows, q_size), np.uint16)
        maple_attention_prefill_ring_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            out_d.ptr,
            spans,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=scale,
            start=start,
            library=maple_attention_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out.reshape(-1), expected)


def test_maple_prefill_attention_gqa4_wave32_matches_local128_bits(
    maple_attention_lib,
) -> None:
    """P2 GQA4 must preserve the local128 tree on production head geometry."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_prefill_ring_gqa4_bf16,
        maple_attention_prefill_ring_bf16,
    )

    rng = np.random.default_rng(20260808)
    rows, q_heads, kv_heads, head_dim = 4, 4, 1, 128
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    start, capacity = 256, 512
    total = start + rows
    scale = head_dim**-0.5
    q = bf16_round(rng.normal(size=(rows, q_heads, head_dim)).astype(np.float32))
    keys = bf16_round(
        rng.normal(size=(total, kv_heads, head_dim)).astype(np.float32)
    )
    values = bf16_round(
        rng.normal(size=(total, kv_heads, head_dim)).astype(np.float32)
    )
    expected = np.stack(
        [
            attention_decode(
                q[row],
                keys[: start + row + 1],
                values[: start + row + 1],
                scale=scale,
            )
            for row in range(rows)
        ]
    ).reshape(rows, q_size)

    base = rng.permutation(capacity).astype(np.int32)
    token_positions = np.arange(capacity, dtype=np.int64)
    key_cache = np.zeros((capacity, kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_bits = f32_to_bf16_bits(keys)
    value_bits = f32_to_bf16_bits(values)
    for absolute_position in range(total):
        physical_slot = int(base[absolute_position % capacity])
        key_cache[physical_slot] = key_bits[absolute_position]
        value_cache[physical_slot] = value_bits[absolute_position]
    qkv = np.zeros((rows, q_size + 2 * kv_size), dtype=np.uint16)
    qkv[:, :q_size] = f32_to_bf16_bits(q.reshape(rows, q_size))

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([total], dtype=np.int64),
            token_positions=token_positions,
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([start + rows - 1], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        key_d = dev.put(key_cache)
        value_d = dev.put(value_cache)
        reference, reference_d = dev.empty((rows, q_size), np.uint16)
        candidate, candidate_d = dev.empty((rows, q_size), np.uint16)
        maple_attention_prefill_ring_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            reference_d.ptr,
            spans,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=scale,
            start=start,
            library=maple_attention_lib,
        )
        maple_attention_prefill_ring_gqa4_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            candidate_d.ptr,
            spans,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=scale,
            start=start,
            library=maple_attention_lib,
        )
        dev.get(reference, reference_d)
        dev.get(candidate, candidate_d)

    mismatch = np.flatnonzero(candidate.reshape(-1) != reference.reshape(-1))
    assert not mismatch.size, (
        f"{mismatch.size} BF16 mismatches; first={mismatch[:8].tolist()} "
        f"candidate={candidate.reshape(-1)[mismatch[:8]].tolist()} "
        f"reference={reference.reshape(-1)[mismatch[:8]].tolist()}"
    )
    np.testing.assert_allclose(
        bf16_to_f32(candidate), expected, rtol=2e-2, atol=2e-2
    )


def test_maple_prefill_attention_gqa4_consumes_complete_spans(
    maple_attention_lib,
) -> None:
    """P2 GQA4 honors live count, position, eviction, and physical mapping."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_prefill_ring_gqa4_bf16,
    )

    rng = np.random.default_rng(20260809)
    rows, q_heads, kv_heads, head_dim = 1, 4, 1, 128
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    query_position, capacity = 7, 8
    qkv = f32_to_bf16_bits(
        rng.normal(size=(rows, q_size + 2 * kv_size)).astype(np.float32)
    )
    base = np.asarray([3, 1, 6, 0, 5, 2, 7, 4], dtype=np.int32)
    token_positions = np.full(capacity, -1, dtype=np.int64)
    evict_mask = np.ones(capacity, dtype=np.bool_)
    key_cache = np.zeros((capacity, kv_heads, head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    for absolute_position in range(4, 8):
        logical_slot = absolute_position % capacity
        physical_slot = int(base[logical_slot])
        token_positions[logical_slot] = absolute_position
        evict_mask[logical_slot] = absolute_position == 5
        key_cache[physical_slot] = f32_to_bf16_bits(
            rng.normal(size=(kv_heads, head_dim)).astype(np.float32)
        )
        value_cache[physical_slot] = f32_to_bf16_bits(
            rng.normal(size=(kv_heads, head_dim)).astype(np.float32)
        )

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([4], dtype=np.int64),
            token_positions=token_positions,
            evict_mask=evict_mask,
            row_positions=np.asarray([query_position], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        key_d = dev.put(key_cache)
        value_d = dev.put(value_cache)
        reference, reference_d = dev.empty((q_size,), np.uint16)
        candidate, candidate_d = dev.empty((rows, q_size), np.uint16)
        maple_attention_decode_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            reference_d.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            library=maple_attention_lib,
        )
        maple_attention_prefill_ring_gqa4_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            candidate_d.ptr,
            spans,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            start=query_position,
            library=maple_attention_lib,
        )
        dev.get(reference, reference_d)
        dev.get(candidate, candidate_d)

    assert np.array_equal(candidate.reshape(-1), reference)


def test_maple_gqa_attention_reads_wrapped_kv_live_spans(maple_attention_lib) -> None:
    rng = np.random.default_rng(55)
    q_heads, kv_heads, head_dim, capacity = 4, 2, 4, 4
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    current_position = 5
    absolute_positions = np.arange(2, 6, dtype=np.int64)
    q = bf16_round(rng.normal(size=(q_heads, head_dim)).astype(np.float32))
    keys = bf16_round(
        rng.normal(size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    values = bf16_round(
        rng.normal(size=(capacity, kv_heads, head_dim)).astype(np.float32)
    )
    base = np.asarray([2, 0, 3, 1], dtype=np.int32)
    token_positions = np.full(capacity, -1, dtype=np.int64)
    key_cache = np.zeros((capacity, kv_heads, head_dim), dtype=np.float32)
    value_cache = np.zeros_like(key_cache)
    for logical_index, absolute in enumerate(absolute_positions):
        ring_slot = int(absolute % capacity)
        physical = int(base[ring_slot])
        token_positions[ring_slot] = absolute
        key_cache[physical] = keys[logical_index]
        value_cache[physical] = values[logical_index]
    expected = f32_to_bf16_bits(
        attention_decode(q, keys, values, scale=head_dim**-0.5).reshape(-1)
    )
    qkv = np.zeros(q_size + 2 * kv_size, dtype=np.uint16)
    qkv[:q_size] = f32_to_bf16_bits(q.reshape(-1))

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([capacity], dtype=np.int64),
            token_positions=token_positions,
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([current_position], dtype=np.int64),
        )
        qkv_d = dev.put(qkv)
        key_d = dev.put(f32_to_bf16_bits(key_cache))
        value_d = dev.put(f32_to_bf16_bits(value_cache))
        out, out_d = dev.empty((q_size,), np.dtype(np.uint16))
        maple_attention_decode_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            out_d.ptr,
            spans,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            library=maple_attention_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


def test_maple_prefill_attention_matches_causal_oracle(maple_attention_lib) -> None:
    """Batched causal prefill attention is bit-exact vs the CPU oracle (P2)."""

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_prefill_bf16,
    )

    rng = np.random.default_rng(66)
    rows, q_heads, kv_heads, head_dim = 5, 4, 2, 4
    q_size = q_heads * head_dim
    q = bf16_round(rng.normal(size=(rows, q_heads, head_dim)).astype(np.float32))
    keys = bf16_round(
        rng.normal(size=(rows, kv_heads, head_dim)).astype(np.float32)
    )
    values = bf16_round(
        rng.normal(size=(rows, kv_heads, head_dim)).astype(np.float32)
    )
    # Causal: row r attends to keys/values rows [0, r].
    expected = np.stack(
        [
            attention_decode(q[r], keys[: r + 1], values[: r + 1], scale=head_dim**-0.5)
            for r in range(rows)
        ]
    ).reshape(-1)
    expected = f32_to_bf16_bits(expected)
    qkv = np.zeros((rows, q_size), dtype=np.uint16)
    qkv[:, :q_size] = f32_to_bf16_bits(q.reshape(rows, -1))

    with DeviceArrays() as dev:
        qkv_d = dev.put(qkv)
        key_d = dev.put(f32_to_bf16_bits(keys))
        value_d = dev.put(f32_to_bf16_bits(values))
        out, out_d = dev.empty((rows, q_size), np.dtype(np.uint16))
        maple_attention_prefill_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            out_d.ptr,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            library=maple_attention_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out.reshape(-1), expected)


def test_maple_fused_qknorm_attention_decode_matches_unfused_chain(
    maple_attention_lib,
) -> None:
    """M2 fused QK-norm+RoPE+KV-write + attention is bit-exact with unfused.

    Compares maple_attention_fused_qknorm_decode against the standalone
    qknorm_rope_kv_write + attention_decode chain. The attention output AND the
    KV cache (current-token K/V writes) must match bit-for-bit.
    """
    rng = np.random.default_rng(66)
    q_heads, kv_heads, head_dim, capacity = 4, 2, 8, 6
    rope_dim = 8
    eps = 1e-5
    rope_theta = 10000.0
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    current_position = 5
    live = 4

    # Raw current-token Q/K/V (pre-norm), in the shared qkv buffer.
    qkv = np.zeros(q_size + 2 * kv_size, dtype=np.uint16)
    qkv[:q_size] = f32_to_bf16_bits(
        rng.normal(size=(q_heads, head_dim)).astype(np.float32).reshape(-1)
    )
    qkv[q_size : q_size + kv_size] = f32_to_bf16_bits(
        rng.normal(size=(kv_heads, head_dim)).astype(np.float32).reshape(-1)
    )
    qkv[q_size + kv_size :] = f32_to_bf16_bits(
        rng.normal(size=(kv_heads, head_dim)).astype(np.float32).reshape(-1)
    )
    q_norm_weight = f32_to_bf16_bits(rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32))
    k_norm_weight = f32_to_bf16_bits(rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32))

    # Prior history positions 2,3,4 written to the ring (physical slots).
    base = np.asarray([5, 2, 0, 3, 1, 4], dtype=np.int32)
    token_positions = np.full(capacity, -1, dtype=np.int64)
    key_cache_f = np.zeros((capacity, kv_heads, head_dim), dtype=np.float32)
    value_cache_f = np.zeros_like(key_cache_f)
    for idx, pos in enumerate([2, 3, 4]):
        logical = int(pos % capacity)
        physical = int(base[logical])
        token_positions[logical] = pos
        key_cache_f[physical] = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)
        value_cache_f[physical] = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)

    key_cache_bits = f32_to_bf16_bits(key_cache_f)
    value_cache_bits = f32_to_bf16_bits(value_cache_f)

    with DeviceArrays() as dev:
        # ---- Unfused chain. ----
        spans_u = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([live], dtype=np.int64),
            token_positions=token_positions.copy(),
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([current_position], dtype=np.int64),
        )
        qkv_u = dev.put(qkv.copy())
        key_u = dev.put(key_cache_bits.copy())
        value_u = dev.put(value_cache_bits.copy())
        qn_u = dev.put(q_norm_weight)
        kn_u = dev.put(k_norm_weight)
        out_u, out_u_d = dev.empty((q_size,), np.dtype(np.uint16))
        maple_qknorm_rope_kv_write_bf16(
            qkv_u.ptr, qn_u.ptr, kn_u.ptr, key_u.ptr, value_u.ptr, spans_u,
            q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            rope_dim=rope_dim, eps=eps, rope_theta=rope_theta,
            library=maple_attention_lib,
        )
        maple_attention_decode_bf16(
            qkv_u.ptr, key_u.ptr, value_u.ptr, out_u_d.ptr, spans_u,
            q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            scale=head_dim**-0.5, library=maple_attention_lib,
        )
        dev.get(out_u, out_u_d)
        key_res_u, _ = dev.empty((capacity, kv_heads, head_dim), np.uint16)
        value_res_u, _ = dev.empty((capacity, kv_heads, head_dim), np.uint16)
        dev.get(key_res_u, key_u)
        dev.get(value_res_u, value_u)

        # ---- Fused kernel on fresh caches. ----
        spans_f = make_spans(
            dev,
            base_offsets=base,
            live_counts=np.asarray([live], dtype=np.int64),
            token_positions=token_positions.copy(),
            evict_mask=np.zeros(capacity, dtype=np.bool_),
            row_positions=np.asarray([current_position], dtype=np.int64),
        )
        qkv_f = dev.put(qkv.copy())
        key_f = dev.put(key_cache_bits.copy())
        value_f = dev.put(value_cache_bits.copy())
        qn_f = dev.put(q_norm_weight)
        kn_f = dev.put(k_norm_weight)
        out_f, out_f_d = dev.empty((q_size,), np.dtype(np.uint16))
        maple_attention_fused_qknorm_decode_bf16(
            qkv_f.ptr, qn_f.ptr, kn_f.ptr, key_f.ptr, value_f.ptr, out_f_d.ptr,
            spans_f,
            q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            rope_dim=rope_dim, eps=eps, rope_theta=rope_theta,
            scale=head_dim**-0.5, library=maple_attention_lib,
        )
        dev.get(out_f, out_f_d)
        key_res_f, _ = dev.empty((capacity, kv_heads, head_dim), np.uint16)
        value_res_f, _ = dev.empty((capacity, kv_heads, head_dim), np.uint16)
        dev.get(key_res_f, key_f)
        dev.get(value_res_f, value_f)

    assert np.array_equal(out_f, out_u), "fused vs unfused attention output mismatch"
    assert np.array_equal(key_res_f, key_res_u), "fused vs unfused K-cache mismatch"
    assert np.array_equal(value_res_f, value_res_u), "fused vs unfused V-cache mismatch"


def test_maple_batched_decode_qknorm_write_and_attention_match_oracle(
    maple_attention_lib,
) -> None:
    """M6 batched per-request QK-norm+KV write + attention decode (D5).

    Each request row owns a disjoint position range in the shared identity
    arena; the batched kernels must be bit-exact with running the c1 kernels
    per row.
    """

    from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
        maple_attention_decode_batched_bf16,
        maple_qknorm_rope_kv_write_batched_decode_bf16,
    )

    rng = np.random.default_rng(77)
    rows, q_heads, kv_heads, head_dim = 3, 4, 2, 8
    rope_dim = 8
    eps = 1e-6
    rope_theta = 10000.0
    per_cap = 32
    capacity = rows * per_cap
    q_size, kv_size = q_heads * head_dim, kv_heads * head_dim
    qkv_stride = q_size + 2 * kv_size

    # Identity arena: physical slot == arena slot. Each request owns a disjoint
    # arena range [row_base_offsets[r], +per_cap) and uses LOCAL RoPE positions.
    base = np.arange(capacity, dtype=np.int32)
    token_positions = np.full(capacity, -1, dtype=np.int64)
    evict_mask = np.zeros(capacity, dtype=np.bool_)
    row_base_offsets = np.asarray([0, 32, 64], dtype=np.int64)
    row_positions = np.asarray([35, 68, 101], dtype=np.int64)  # wrapped local positions
    live_counts = np.full(rows, per_cap, dtype=np.int64)
    q_norm_weight = f32_to_bf16_bits(
        rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32)
    )
    k_norm_weight = f32_to_bf16_bits(
        rng.uniform(0.5, 1.5, size=head_dim).astype(np.float32)
    )

    qkv = np.zeros((rows, qkv_stride), dtype=np.uint16)
    key_cache_f = np.zeros((capacity, kv_heads, head_dim), dtype=np.float32)
    value_cache_f = np.zeros_like(key_cache_f)
    expected_kv = []
    for row in range(rows):
        pos = int(row_positions[row])
        apos = int(row_base_offsets[row])
        q_raw = rng.normal(size=(q_heads, head_dim)).astype(np.float32)
        k_raw = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)
        v_raw = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)
        qkv[row, :q_size] = f32_to_bf16_bits(q_raw.reshape(-1))
        qkv[row, q_size : q_size + kv_size] = f32_to_bf16_bits(k_raw.reshape(-1))
        qkv[row, q_size + kv_size :] = f32_to_bf16_bits(v_raw.reshape(-1))

        live = int(live_counts[row])
        start_position = pos - live + 1
        hist_keys = rng.normal(size=(live, kv_heads, head_dim)).astype(np.float32)
        hist_vals = rng.normal(size=(live, kv_heads, head_dim)).astype(np.float32)
        for offset, token_position in enumerate(range(start_position, pos + 1)):
            slot = apos + token_position % per_cap
            token_positions[slot] = token_position
            key_cache_f[slot] = hist_keys[offset]
            value_cache_f[slot] = hist_vals[offset]

        qn_ref, kn_ref = qk_norm_rope(
            bf16_round(q_raw),
            bf16_round(k_raw),
            bf16_to_f32(q_norm_weight),
            bf16_to_f32(k_norm_weight),
            pos=pos,
            rope_theta=rope_theta,
            rope_dim=rope_dim,
            eps=eps,
        )
        expected_kv.append((qn_ref, kn_ref, v_raw))

    with DeviceArrays() as dev:
        spans = make_spans(
            dev,
            base_offsets=base,
            live_counts=live_counts,
            token_positions=token_positions,
            evict_mask=evict_mask,
            row_positions=row_positions,
        )
        rbo = dev.put(row_base_offsets)
        qkv_d = dev.put(qkv)
        key_d = dev.put(f32_to_bf16_bits(key_cache_f))
        value_d = dev.put(f32_to_bf16_bits(value_cache_f))
        qn_d = dev.put(q_norm_weight)
        kn_d = dev.put(k_norm_weight)

        maple_qknorm_rope_kv_write_batched_decode_bf16(
            qkv_d.ptr,
            qn_d.ptr,
            kn_d.ptr,
            key_d.ptr,
            value_d.ptr,
            spans,
            row_base_offsets=rbo.ptr,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            eps=eps,
            rope_theta=rope_theta,
            library=maple_attention_lib,
        )
        qkv_res, _ = dev.empty((rows, qkv_stride), np.dtype(np.uint16))
        dev.get(qkv_res, qkv_d)
        key_res_bits, _ = dev.empty(
            (capacity, kv_heads, head_dim), np.uint16
        )
        dev.get(key_res_bits, key_d)

        out, out_d = dev.empty((rows, q_size), np.dtype(np.uint16))
        maple_attention_decode_batched_bf16(
            qkv_d.ptr,
            key_d.ptr,
            value_d.ptr,
            out_d.ptr,
            spans,
            row_base_offsets=rbo.ptr,
            rows=rows,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            library=maple_attention_lib,
        )
        dev.get(out, out_d)

    for row in range(rows):
        pos = int(row_positions[row])
        apos = int(row_base_offsets[row])
        qn_ref, kn_ref, v_raw = expected_kv[row]
        assert np.array_equal(
            np.array(qkv_res[row, :q_size], dtype=np.uint16),
            f32_to_bf16_bits(qn_ref.reshape(-1)),
        ), f"row {row} normalized Q mismatch"
        current_slot = apos + pos % per_cap
        assert np.array_equal(
            np.array(key_res_bits[current_slot], dtype=np.uint16).reshape(-1),
            f32_to_bf16_bits(kn_ref.reshape(-1)),
        ), f"row {row} K-cache write mismatch"

        # Attention oracle over the last per_cap local positions, ordered by age.
        start_position = pos - per_cap + 1
        slots = [apos + token % per_cap for token in range(start_position, pos + 1)]
        hist_k = bf16_round(key_cache_f[slots])
        hist_k[-1] = bf16_round(kn_ref)
        hist_v = bf16_round(value_cache_f[slots])
        hist_v[-1] = bf16_round(v_raw)
        expected = f32_to_bf16_bits(
            attention_decode(qn_ref, hist_k, hist_v, scale=head_dim**-0.5).reshape(-1)
        )
        got = np.array(out[row, :q_size], dtype=np.uint16)
        assert np.array_equal(got, expected), f"row {row} attention decode mismatch"
