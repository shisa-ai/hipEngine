from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _ring_spans(capacity: int = 512) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x1000, (capacity,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_kv_live_spans_sliding_ring_requires_complete_absolute_metadata() -> None:
    spans = _ring_spans()

    assert spans.spans_mode == "sliding_ring"
    assert spans.max_live_count == 512
    assert spans.token_positions is not None
    assert spans.token_positions.dtype is DType.INT64
    assert spans.evict_mask is not None
    assert spans.row_positions is not None

    with pytest.raises(ValueError, match="token_positions"):
        KVLiveSpans(
            base_offsets=_tensor(1, (512,), "int32"),
            live_counts=_tensor(2, (1,), "int64"),
            max_live_count=512,
            token_positions=None,
            evict_mask=_tensor(4, (512,), "bool"),
            row_positions=_tensor(5, (1,), "int64"),
            storage_dtype=DType.BF16,
            spans_mode="sliding_ring",
        )
    with pytest.raises(ValueError, match="capacity"):
        KVLiveSpans.sliding_ring(
            base_offsets=_tensor(1, (511,), "int32"),
            live_counts=_tensor(2, (1,), "int64"),
            token_positions=_tensor(3, (512,), "int64"),
            evict_mask=_tensor(4, (512,), "bool"),
            row_positions=_tensor(5, (1,), "int64"),
            capacity=512,
            storage_dtype="bf16",
        )


def test_laguna_swa_build_plan_registry_and_validation(tmp_path) -> None:
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_attention_decode_bf16_spans,
        laguna_global_attention_prefill_bf16_spans,
        laguna_global_write_kv_rows_f32_spans,
        laguna_swa_attention_decode_bf16_spans,
        laguna_swa_attention_prefill_bf16_spans,
        laguna_swa_attention_prefill_wave32_exact_bf16_spans,
        laguna_swa_write_kv_f32_spans,
        laguna_swa_write_kv_rows_f32_spans,
        plan_laguna_kv_attention_build,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import resolve

    artifact = plan_laguna_kv_attention_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc laguna swa test version",
    )
    assert artifact.family == "laguna_kv_attention"
    assert artifact.output_path.name == "laguna_kv_attention.so"
    assert any(str(path).endswith("laguna_kv_attention.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()

    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="global_context_spans",
        )
        is laguna_global_attention_decode_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_spans",
        )
        is laguna_swa_attention_decode_bf16_spans
    )
    load_backend_kernel_package("hip_gfx1151")
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_spans",
        )
        is laguna_swa_attention_decode_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_spans",
        )
        is laguna_global_attention_prefill_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_spans",
        )
        is laguna_swa_attention_prefill_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_wave32_exact_spans",
        )
        is laguna_swa_attention_prefill_wave32_exact_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_kv_write",
            quant="bf16",
            variant="global_f32_rows_spans",
        )
        is laguna_global_write_kv_rows_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_kv_write",
            quant="bf16",
            variant="swa_f32_rows_spans",
        )
        is laguna_swa_write_kv_rows_f32_spans
    )

    with pytest.raises(ValueError, match="num_kv_heads"):
        laguna_swa_write_kv_f32_spans(0, 0, 0, 0, _ring_spans(), 0, 128)
    with pytest.raises(ValueError, match="divisible"):
        laguna_swa_attention_decode_bf16_spans(
            0,
            0,
            0,
            0,
            _ring_spans(),
            70,
            8,
            128,
            128**-0.5,
        )


class _FakeRuntime:
    def __init__(self, *, fail_malloc_at: int | None = None) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind]] = []
        self.memsets: list[tuple[int, int, int]] = []
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic Laguna KV allocation failure")
        ptr = self.next_ptr
        self.next_ptr += 0x1000
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert int(dst) in self.allocations or any(
            base < int(dst) < base + nbytes for base, nbytes in self.allocations.items()
        )
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.copies.append((int(dst), int(count), kind))

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert int(dst) in self.allocations or any(
            base < int(dst) < base + size for base, size in self.allocations.items()
        )
        self.memsets.append((int(dst), int(value), int(nbytes)))


def _production_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION for layer_id in range(48)
    )
    head_counts = tuple(
        48 if attention_type == FULL_ATTENTION else 72 for attention_type in layer_types
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=head_counts,
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )


def test_laguna_kv_owner_allocates_12_global_36_bounded_rings_and_tears_down() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    bytes_per_token = 8 * 128 * 2 * 2
    expected_payload = bytes_per_token * (12 * 4096 + 36 * 512)

    assert len(cache.layers) == 48
    assert sum(layer.attention_type == FULL_ATTENTION for layer in cache.layers) == 12
    assert sum(layer.attention_type == SLIDING_ATTENTION for layer in cache.layers) == 36
    assert cache.payload_nbytes == expected_payload == 264 * 1024 * 1024
    assert all(
        layer.capacity == (4096 if layer.attention_type == FULL_ATTENTION else 512)
        for layer in cache.layers
    )
    assert all(
        layer.spans.spans_mode
        == ("uniform" if layer.attention_type == FULL_ATTENTION else "sliding_ring")
        for layer in cache.layers
    )
    assert all(layer.spans.token_positions is not None for layer in cache.layers)
    assert all(layer.spans.evict_mask is not None for layer in cache.layers)
    assert all(
        layer.attention_prefill_variant == "swa_context_rows_spans"
        for layer in cache.layers
        if layer.attention_type == SLIDING_ATTENTION
    )
    assert cache.allocation_count == 243

    cache.prepare_position(0)
    cache.prepare_position(1)
    with pytest.raises(ValueError, match="token-serial"):
        cache.prepare_position(3)
    assert cache.position == 1

    cache.prepare_rows((2, 3, 4))
    assert cache.pending_positions == (2, 3, 4)
    with pytest.raises(RuntimeError, match="bulk positions are pending"):
        cache.prepare_position(2)
    cache.discard_rows()
    assert cache.position == 1
    assert cache.pending_positions == ()
    cache.prepare_rows((2, 3, 4))
    cache.commit_rows()
    assert cache.position == 4
    assert cache.pending_positions == ()
    with pytest.raises(ValueError, match="consecutive"):
        cache.prepare_rows((5, 7))
    with pytest.raises(ValueError, match="capacity"):
        cache.prepare_rows(tuple(range(5, 5 + 513)))

    cache.reset()
    assert cache.position == -1
    assert cache.pending_positions == ()
    assert len(runtime.memsets) == 48 * 3
    assert {value for _, value, _ in runtime.memsets} == {0, 1, 0xFF}

    allocated_count = len(runtime.allocations)
    assert allocated_count > 96
    cache.free()
    assert len(runtime.freed) == allocated_count
    assert runtime.allocations == {}
    cache.free()


def test_laguna_kv_owner_cleans_partial_allocation_failure() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    with pytest.raises(ValueError, match="SWA prefill variant"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=_FakeRuntime(),
            swa_prefill_variant="missing",
        )

    runtime = _FakeRuntime(fail_malloc_at=12)
    with pytest.raises(MemoryError, match="synthetic Laguna KV"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=runtime,
        )
    assert runtime.allocations == {}


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_production_kv_owner_live_allocation_and_teardown() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    before = memory_stats()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        during = memory_stats()
        assert cache.payload_nbytes == 264 * 1024 * 1024
        assert (
            during["current_allocated_bytes"] - before["current_allocated_bytes"]
            == cache.resident_nbytes
        )
        assert during["active_allocations"] - before["active_allocations"] == cache.allocation_count
    finally:
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_global_and_swa_token_serial_attention_match_cpu_across_wraps() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    kv_library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=1026,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    global_offsets = np.roll(
        np.arange(cache.layer(0).spans.base_offsets.numel, dtype=np.int32),
        1,
    )
    swa_offsets = np.arange(511, -1, -1, dtype=np.int32)
    runtime.memcpy(
        cache.layer(0).spans.base_offsets.ptr,
        host_array_ptr(global_offsets),
        global_offsets.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    runtime.memcpy(
        cache.layer(1).spans.base_offsets.ptr,
        host_array_ptr(swa_offsets),
        swa_offsets.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    rng = np.random.default_rng(290)
    total_tokens = 1026
    keys = rng.normal(0.0, 0.12, size=(total_tokens, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(total_tokens, 8, 128)).astype(np.float32)
    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    selected = {510, 511, 512, 513, 1024, 1025}
    allocations = []
    evicted_position = 700
    try:
        key_device = malloc(keys[0].nbytes, runtime=runtime)
        value_device = malloc(values[0].nbytes, runtime=runtime)
        query_device = malloc(72 * 128 * 4, runtime=runtime)
        output_device = malloc(72 * 128 * 4, runtime=runtime)
        allocations.extend((key_device, value_device, query_device, output_device))

        for position in range(total_tokens):
            key_row = np.ascontiguousarray(keys[position])
            value_row = np.ascontiguousarray(values[position])
            copy_host_to_device(
                key_device,
                host_array_ptr(key_row),
                runtime=runtime,
            )
            copy_host_to_device(
                value_device,
                host_array_ptr(value_row),
                runtime=runtime,
            )
            cache.prepare_position(position)
            cache.append(
                0,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )
            cache.append(
                1,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )

            if position not in selected:
                continue
            query = rng.normal(0.0, 0.12, size=(72, 128)).astype(np.float32)
            copy_host_to_device(
                query_device,
                host_array_ptr(query),
                runtime=runtime,
            )
            if position == 1025:
                cache.evict_swa_position(1, evicted_position)
            cache.attend(
                1,
                query_device.ptr,
                output_device.ptr,
                library=kv_library,
            )
            runtime.device_synchronize()
            actual = np.empty((72, 128), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(actual),
                output_device,
                runtime=runtime,
            )
            visible = np.arange(max(0, position - 511), position + 1)
            if position == 1025:
                visible = visible[visible != evicted_position]
            expected = _attention_reference(
                query,
                keys_bf16[visible],
                values_bf16[visible],
                num_kv_heads=8,
            )
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)

            if position == 513:
                global_evicted_position = 100
                cache.evict_position(0, global_evicted_position)
                global_query = np.ascontiguousarray(query[:48])
                copy_host_to_device(
                    query_device,
                    host_array_ptr(global_query),
                    nbytes=global_query.nbytes,
                    runtime=runtime,
                )
                cache.attend(
                    0,
                    query_device.ptr,
                    output_device.ptr,
                    library=kv_library,
                )
                runtime.device_synchronize()
                global_actual = np.empty((48, 128), dtype=np.float32)
                copy_device_to_host(
                    host_array_ptr(global_actual),
                    output_device,
                    nbytes=global_actual.nbytes,
                    runtime=runtime,
                )
                global_visible = np.arange(position + 1)
                global_visible = global_visible[global_visible != global_evicted_position]
                global_expected = _attention_reference(
                    global_query,
                    keys_bf16[global_visible],
                    values_bf16[global_visible],
                    num_kv_heads=8,
                )
                np.testing.assert_allclose(
                    global_actual,
                    global_expected,
                    rtol=3e-4,
                    atol=3e-4,
                )

        token_positions = np.empty(512, dtype=np.int64)
        mask = np.empty(512, dtype=np.bool_)
        swa_spans = cache.layer(1).spans
        runtime.memcpy(
            host_array_ptr(token_positions),
            swa_spans.token_positions.ptr,
            token_positions.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        runtime.memcpy(
            host_array_ptr(mask),
            swa_spans.evict_mask.ptr,
            mask.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        assert token_positions[0] == 1024
        assert token_positions[1] == 1025
        assert token_positions[evicted_position % 512] == evicted_position
        assert mask[evicted_position % 512]

        physical_slot = int(swa_offsets[1025 % 512])
        physical_key = np.empty((8, 128), dtype=np.uint16)
        runtime.memcpy(
            host_array_ptr(physical_key),
            cache.layer(1).key_cache.ptr + physical_slot * physical_key.nbytes,
            physical_key.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        np.testing.assert_array_equal(
            physical_key,
            float_array_to_bf16_bits(keys[1025]),
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_bulk_global_and_swa_prefill_match_serial_across_ring_wrap() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    serial = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    bulk = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    wave32 = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    rng = np.random.default_rng(1207)
    seed_rows = 508
    rows = 8
    keys = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    query_global = rng.normal(0.0, 0.12, size=(rows, 48, 128)).astype(np.float32)
    query_swa = rng.normal(0.0, 0.12, size=(rows, 72, 128)).astype(np.float32)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        global_query_rows = malloc(query_global.nbytes, runtime=runtime)
        swa_query_rows = malloc(query_swa.nbytes, runtime=runtime)
        global_bulk_out = malloc(query_global.nbytes, runtime=runtime)
        swa_bulk_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_wave32_out = malloc(query_swa.nbytes, runtime=runtime)
        global_serial_out = malloc(query_global[0].nbytes, runtime=runtime)
        swa_serial_out = malloc(query_swa[0].nbytes, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                global_query_rows,
                swa_query_rows,
                global_bulk_out,
                swa_bulk_out,
                swa_wave32_out,
                global_serial_out,
                swa_serial_out,
            )
        )
        for buffer, array in (
            (key_rows, keys),
            (value_rows, values),
            (global_query_rows, query_global),
            (swa_query_rows, query_swa),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        # Populate both owners through the bulk write path, then compare one
        # wrap-crossing 508..515 chunk against the established token-serial path.
        seed_positions = tuple(range(seed_rows))
        for cache in (serial, bulk, wave32):
            cache.prepare_rows(seed_positions)
            for layer_id in range(2):
                cache.append_rows(
                    layer_id,
                    key_rows.ptr,
                    value_rows.ptr,
                    seed_rows,
                    library=library,
                )
            cache.commit_rows()
            cache.evict_position(0, 200)
            cache.evict_swa_position(1, 200)

        row_bytes = 8 * 128 * np.dtype(np.float32).itemsize
        positions = tuple(range(seed_rows, seed_rows + rows))
        bulk.prepare_rows(positions)
        bulk.attend_prefill(
            0,
            global_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            global_bulk_out.ptr,
            rows,
            library=library,
        )
        bulk.append_rows(
            0,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        bulk.attend_prefill(
            1,
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_bulk_out.ptr,
            rows,
            library=library,
        )
        bulk.append_rows(
            1,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        bulk.commit_rows()

        wave32.prepare_rows(positions)
        wave32.attend_prefill(
            1,
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_wave32_out.ptr,
            rows,
            library=library,
        )
        wave32.append_rows(
            1,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        wave32.commit_rows()

        expected_global = np.empty_like(query_global)
        expected_swa = np.empty_like(query_swa)
        for row, position in enumerate(positions):
            serial.prepare_position(position)
            for layer_id, (query, output) in enumerate(
                (
                    (query_global[row], global_serial_out),
                    (query_swa[row], swa_serial_out),
                )
            ):
                serial.append(
                    layer_id,
                    key_rows.ptr + position * row_bytes,
                    value_rows.ptr + position * row_bytes,
                    library=library,
                )
                serial.attend(
                    layer_id,
                    (global_query_rows if layer_id == 0 else swa_query_rows).ptr
                    + row * query.nbytes,
                    output.ptr,
                    library=library,
                )
                runtime.device_synchronize()
                destination = expected_global[row] if layer_id == 0 else expected_swa[row]
                copy_device_to_host(
                    host_array_ptr(destination),
                    output,
                    destination.nbytes,
                    runtime=runtime,
                )

        actual_global = np.empty_like(query_global)
        actual_swa = np.empty_like(query_swa)
        actual_swa_wave32 = np.empty_like(query_swa)
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual_global),
            global_bulk_out,
            actual_global.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa),
            swa_bulk_out,
            actual_swa.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_wave32),
            swa_wave32_out,
            actual_swa_wave32.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual_global, expected_global)
        np.testing.assert_array_equal(actual_swa, expected_swa)
        np.testing.assert_array_equal(actual_swa_wave32, actual_swa)
        assert bulk.position == serial.position == 515

        for layer_id in range(2):
            bulk_state = bulk.layer(layer_id)
            serial_state = serial.layer(layer_id)
            capacity = bulk_state.capacity
            for field, dtype in (("token_positions", np.int64), ("evict_mask", np.bool_)):
                bulk_values = np.empty(capacity, dtype=dtype)
                serial_values = np.empty(capacity, dtype=dtype)
                bulk_tensor = getattr(bulk_state.spans, field)
                serial_tensor = getattr(serial_state.spans, field)
                runtime.memcpy(
                    host_array_ptr(bulk_values),
                    bulk_tensor.ptr,
                    bulk_values.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                runtime.memcpy(
                    host_array_ptr(serial_values),
                    serial_tensor.ptr,
                    serial_values.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                np.testing.assert_array_equal(bulk_values, serial_values)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        wave32.free()
        bulk.free()
        serial.free()


def _attention_reference(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    num_kv_heads: int,
) -> np.ndarray:
    num_q_heads, head_dim = query.shape
    group = num_q_heads // num_kv_heads
    kv_for_q = np.arange(num_q_heads, dtype=np.int64) // group
    expanded_keys = keys[:, kv_for_q, :]
    expanded_values = values[:, kv_for_q, :]
    scores = np.einsum("hd,thd->ht", query, expanded_keys, dtype=np.float32)
    scores *= np.float32(head_dim**-0.5)
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores, dtype=np.float32)
    weights /= np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
    return np.einsum("ht,thd->hd", weights, expanded_values, dtype=np.float32)
