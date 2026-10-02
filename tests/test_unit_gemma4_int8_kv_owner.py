"""CPU contracts for the Gemma 4 INT8 KV cache owner and layer selection.

No device is touched: the owner's allocator and copies are replaced with
recorders, so layout, allocated bytes, lifecycle, span shapes and the registry
selection are pinned without HIP. The GPU route (writer -> consumer vs the CPU
representation oracle) is in the GPU-tier module.
"""

from __future__ import annotations

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.kvcache import KVLiveSpans
from hipengine.runtime import gemma4_int8_kv as owner_module
from hipengine.runtime.gemma4_int8_kv import (
    Gemma4Int8KVCache,
    int8_kv_block_table_len,
    int8_kv_resident_bytes,
)


class _FakeAllocator:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.next_ptr = 0x1000
        self.allocated: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []
        self.calls = 0
        self.fail_at = fail_at
        self.copies: list[tuple[int, int]] = []
        self.enqueues: list[tuple[int, int, int]] = []

    def malloc(self, nbytes: int) -> DeviceBuffer:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise MemoryError(f"fake allocator exhausted at call {self.calls}")
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=nbytes)
        self.next_ptr += nbytes + 0x1000
        self.allocated.append(buffer)
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self.freed.append(buffer)

    def copy(self, buffer, host_ptr, nbytes, **kwargs) -> None:
        self.copies.append((buffer.ptr, int(nbytes)))

    def enqueue(self, buffer, host_ptr, nbytes, **kwargs) -> None:
        self.enqueues.append((buffer.ptr, int(nbytes), int(kwargs.get("stream", 0))))


def _install(monkeypatch: pytest.MonkeyPatch, allocator: _FakeAllocator) -> None:
    monkeypatch.setattr(owner_module, "malloc", allocator.malloc)
    monkeypatch.setattr(owner_module, "free", allocator.free)
    monkeypatch.setattr(owner_module, "copy_host_to_device", allocator.copy)
    monkeypatch.setattr(owner_module, "enqueue_host_to_device", allocator.enqueue)


_ATTENTIONS = ((16, 8, 256), (16, 2, 512))


def _cache(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allocator: _FakeAllocator | None = None,
    capacity: int = 512,
    max_block: int = 8,
    block_size: int = 64,
    scale_dtype: DType = DType.FP16,
    attentions: tuple[tuple[int, int, int], ...] = _ATTENTIONS,
) -> tuple[Gemma4Int8KVCache, _FakeAllocator]:
    allocator = allocator or _FakeAllocator()
    _install(monkeypatch, allocator)
    cache = Gemma4Int8KVCache(
        capacity=capacity,
        max_block=max_block,
        attentions=attentions,
        scale_dtype=scale_dtype,
        block_size=block_size,
    )
    return cache, allocator


def test_block_table_len_strictly_covers_capacity() -> None:
    for capacity in (1, 255, 256, 257, 512, 1024, 8192):
        blocks = int8_kv_block_table_len(capacity, 256)
        assert blocks == capacity // 256 + 1
        assert blocks * 256 > capacity, (
            f"block table {blocks} x 256 does not strictly exceed capacity {capacity}; "
            "a full cache would trip the writer's bound"
        )


def test_allocated_bytes_match_the_planner_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, allocator = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    expected = int8_kv_resident_bytes(
        capacity=512,
        max_block=8,
        attentions=_ATTENTIONS,
        scale_dtype=DType.FP16,
        block_size=64,
    )
    try:
        assert cache.allocated_bytes == expected
        assert cache.allocated_bytes == sum(b.nbytes for b in allocator.allocated)
    finally:
        cache.close()


def test_scale_dtype_changes_scale_plane_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    fp16, _ = _cache(monkeypatch, scale_dtype=DType.FP16)
    fp32, _ = _cache(monkeypatch, scale_dtype=DType.FP32)
    try:
        delta = int8_kv_resident_bytes(
            capacity=512,
            max_block=8,
            attentions=_ATTENTIONS,
            scale_dtype=DType.FP32,
            block_size=64,
        ) - int8_kv_resident_bytes(
            capacity=512,
            max_block=8,
            attentions=_ATTENTIONS,
            scale_dtype=DType.FP16,
            block_size=64,
        )
        assert fp32.allocated_bytes - fp16.allocated_bytes == delta
        assert delta > 0
    finally:
        fp16.close()
        fp32.close()


def test_scale_metadata_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, _ = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        assert cache.blocks == int8_kv_block_table_len(512, 64)
        assert len(cache.scale_metadata) == len(_ATTENTIONS)
        for index, (num_heads, num_kv_heads, head_dim) in enumerate(_ATTENTIONS):
            metadata = cache.scale_metadata[index]
            assert metadata.granularity == "per_token_head"
            assert metadata.scale_dtype == DType.FP16
            assert metadata.k_scale.shape == (cache.blocks, 64, num_kv_heads)
            assert metadata.v_scale.shape == (cache.blocks, 64, num_kv_heads)
    finally:
        cache.close()


def test_begin_block_builds_writer_and_consumer_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, allocator = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        block = cache.begin_block(write_offset=4, rows=3, stream=0)
        assert block.rows == 3 and block.write_offset == 4
        # Append positions are absolute; consumer counts are causal prefix lengths.
        assert block.positions.shape == (3,)
        assert block.counts.shape == (3,)
        assert block.base_offsets_1d.shape == (cache.blocks,)
        assert block.base_offsets_2d.shape == (3, cache.blocks)

        # Both views alias one resident identity page table.
        assert block.base_offsets_1d.ptr == block.base_offsets_2d.ptr
        assert block.positions.ptr != block.counts.ptr
        assert len(allocator.enqueues) == 2
        assert all(entry[2] == 0 for entry in allocator.enqueues)
        assert [entry[1] for entry in allocator.enqueues] == [3 * 8, 3 * 8]

        layer = cache.layer_kv(0, block)
        writer_spans = layer.writer_spans
        consumer_spans = layer.consumer_spans
        # Writer: row-major table, positions as live_counts.
        assert writer_spans.base_offsets.shape == (3, cache.blocks)
        assert writer_spans.live_counts.ptr == block.positions.ptr
        assert writer_spans.row_positions is None
        # Consumer: shared 1-D table, counts as live_counts, positions as row_positions.
        assert consumer_spans.base_offsets.shape == (cache.blocks,)
        assert consumer_spans.live_counts.ptr == block.counts.ptr
        assert consumer_spans.row_positions.ptr == block.positions.ptr
        for spans in (writer_spans, consumer_spans):
            assert isinstance(spans, KVLiveSpans)
            assert spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
            assert spans.spans_mode == "uniform"
            assert spans.scale_metadata is not None
            assert spans.max_live_count == cache.capacity
    finally:
        cache.close()


def test_positions_and_counts_are_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, _ = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        # A decode step at position 7: one row, count 8.
        block = cache.begin_block(write_offset=7, rows=1, stream=0)
        assert cache._positions_host[:1].tolist() == [7]
        assert cache._counts_host[:1].tolist() == [8]
        # The writer spans carry the append position, the consumer its prefix.
        assert block.positions.numel == 1 and block.counts.numel == 1
    finally:
        cache.close()


def test_reset_clears_host_staging_without_freeing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, allocator = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        cache.begin_block(write_offset=4, rows=3, stream=0)
        assert cache._positions_host[:3].tolist() == [4, 5, 6]
        cache.reset()
        assert cache._positions_host[:3].tolist() == [0, 0, 0]
        assert cache._counts_host[:3].tolist() == [0, 0, 0]
        # Reset never frees; the buffers are reused.
        assert allocator.freed == []
        # The cache is still usable after reset.
        cache.begin_block(write_offset=0, rows=2, stream=0)
        assert cache._positions_host[:2].tolist() == [0, 1]
    finally:
        cache.close()


def test_close_is_idempotent_and_frees_every_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, allocator = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    cache.close()
    cache.close()
    assert len(allocator.freed) == len(allocator.allocated)
    assert sorted(b.ptr for b in allocator.freed) == sorted(b.ptr for b in allocator.allocated)
    assert len({b.ptr for b in allocator.freed}) == len(allocator.freed)
    assert cache.closed is True


def test_constructor_failure_frees_partial_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeAllocator(fail_at=5)
    _install(monkeypatch, allocator)
    with pytest.raises(MemoryError):
        Gemma4Int8KVCache(
            capacity=512, max_block=8, attentions=_ATTENTIONS, block_size=64
        )
    assert len(allocator.allocated) == 4
    assert sorted(b.ptr for b in allocator.freed) == sorted(b.ptr for b in allocator.allocated)


def test_unsupported_scale_dtype_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator = _FakeAllocator()
    _install(monkeypatch, allocator)
    with pytest.raises(ValueError):
        Gemma4Int8KVCache(
            capacity=512, max_block=8, attentions=_ATTENTIONS, scale_dtype=DType.BF16
        )


def test_oversized_context_is_a_named_capability_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeAllocator()
    _install(monkeypatch, allocator)
    # capacity + head_dim + 2*num_warps must fit the consumer's 64 KiB shared
    # memory. A context that cannot is refused at construction, before any
    # forward pass could fail deep inside the kernel.
    with pytest.raises(ValueError, match="shared memory"):
        Gemma4Int8KVCache(
            capacity=20000, max_block=8, attentions=((16, 2, 512),), block_size=64
        )


def test_begin_block_refuses_over_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, _ = _cache(monkeypatch, capacity=16, max_block=4, block_size=8)
    try:
        with pytest.raises(ValueError):
            cache.begin_block(write_offset=14, rows=4, stream=0)
    finally:
        cache.close()


def test_layer_kv_rejects_out_of_range_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    cache, _ = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        block = cache.begin_block(write_offset=0, rows=2, stream=0)
        with pytest.raises(IndexError):
            cache.layer_kv(len(_ATTENTIONS), block)
    finally:
        cache.close()


def test_layer_selects_writer_and_consumer_by_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layer resolves both kernels from the registry, by shape not identity.

    Prefill and decode must select different writer variants and different
    consumer layers from the same spans; nothing here touches a device.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        _select_int8_kv_kernels,
    )

    cache, _ = _cache(monkeypatch, capacity=512, max_block=8, block_size=64)
    try:
        prefill_block = cache.begin_block(write_offset=0, rows=4, stream=0)
        prefill = _select_int8_kv_kernels(cache.layer_kv(0, prefill_block), 4)
        assert prefill[0].quant == "int8_per_token_head"
        assert prefill[0].variant == "per_token_head_bf16_prompt_spans"
        assert prefill[2] == "paged_attn_prefill"
        assert prefill[3].__name__ == "gemma4_attention_prefill_int8_per_token_head_spans"

        decode_block = cache.begin_block(write_offset=4, rows=1, stream=0)
        decode = _select_int8_kv_kernels(cache.layer_kv(0, decode_block), 1)
        assert decode[0].variant == "per_token_head_bf16_spans"
        assert decode[2] == "paged_attn_decode"
        assert decode[3].__name__ == "gemma4_attention_decode_int8_per_token_head_spans"
        # Different writer variant and different consumer, selected from spans.
        assert prefill[0].variant != decode[0].variant
        assert prefill[3] is not decode[3]
    finally:
        cache.close()
