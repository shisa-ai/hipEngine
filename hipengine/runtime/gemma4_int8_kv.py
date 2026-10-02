"""INT8 per-token/head KV cache owner for the Gemma 4 runner.

This owns the storage the BF16-source INT8 writer writes and the direct INT8
attention consumers read: an INT8 payload cache, a per-token/per-KV-head FP16
or FP32 scale plane, a shared identity page table, and the per-row live
counts/positions the consumers validate. It is the runtime half of D12 that was
missing -- the writer (``paged_kv_write.hip``) and the consumers
(``gemma4_attention_int8.hip``) already exist and are registered; this object
allocates and releases the buffers they operate on and hands the layer one
``Gemma4LayerInt8KV`` per layer per block.

Layout, chosen so the writer and both consumers agree byte-for-byte:

* ``key_cache``/``value_cache`` are ``[blocks, block_size, num_kv_heads,
  head_dim]`` INT8.
* ``k_scale``/``v_scale`` are ``[blocks, block_size, num_kv_heads]`` FP16 or
  FP32, exactly the shape the writer stores and the consumers reconstruct from.
* ``blocks`` is one more than the minimum to cover ``capacity``, so a full cache
  never trips the writer's ``max_live_count >= block_size * block_table_len``
  bound.

Two shapes of the same page table are exposed. The writer's prompt/batch ABI
indexes a row-major ``[rows, block_table_len]`` table, while the multi-row
prefill consumer declares one shared 1-D ``[block_table_len]`` table. Both are
views of one resident identity table whose rows all repeat, so they describe the
same shared arena rather than two prefixes.

There is no persistent BF16 shadow: the layer quantizes the block's BF16 K/V
once and the consumers reconstruct ``float32(int8) * float32(scale)``. The FP32
query/context scratch the consumers require lives here too, sized to the widest
block, so the BF16 layer scratch is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    enqueue_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

_INT8_BYTES = 1
_I32_BYTES = 4
_I64_BYTES = 8
_F32_BYTES = 4

#: Page size for the INT8 cache. Independent of the runner's prefill block; it
#: only sets how the cache is addressed. The KV policy default is 256.
DEFAULT_INT8_KV_BLOCK_SIZE = 256

#: 64 KiB of shared memory per consumer block, in floats. The consumer sizes its
#: logit scratch as ``capacity + head_dim + 2 * num_warps`` floats; a context
#: whose count would exceed this is a named capability miss, reported at load.
_MAX_CONSUMER_SHARED_FLOATS = 64 * 1024 // _F32_BYTES

#: Warps per consumer block. The kernel's logit scratch is sized for one partial
#: reduction slot per warp on each side of the row.
_CONSUMER_NUM_WARPS = 8


def int8_kv_consumer_shared_bytes(*, capacity: int, head_dim: int) -> int:
    """Shared memory the direct INT8 consumer needs for ``capacity`` and ``head_dim``.

    One formula, used by the cache owner and by the runner's admission check, so
    the two cannot drift: the kernel keeps ``capacity`` logits plus ``head_dim``
    staging slots plus two partial-reduction slots per warp in LDS. A context
    that needs more than the 64 KiB gfx1100 block limit is a named capability
    miss -- reported here, at load, rather than as a launch failure deep inside a
    forward pass.
    """

    shared_floats = int(capacity) + int(head_dim) + 2 * _CONSUMER_NUM_WARPS
    required = shared_floats * _F32_BYTES
    if shared_floats > _MAX_CONSUMER_SHARED_FLOATS:
        raise ValueError(
            "the Gemma 4 direct INT8 attention consumer needs "
            f"{required} bytes of shared memory for capacity={int(capacity)}, "
            f"head_dim={int(head_dim)}; this kernel supports at most "
            f"{_MAX_CONSUMER_SHARED_FLOATS * _F32_BYTES} bytes, so reduce the "
            "context length"
        )
    return required

#: The two consumer variants the layer resolves. Kept here so the owner and its
#: tests name the same keys the layer selects.
INT8_ATTENTION_QUANT = "int8_per_token_head"
INT8_ATTENTION_VARIANT = "gemma4_direct_spans"

_DEVICE = Device("hip", 0)


def int8_kv_block_table_len(capacity: int, block_size: int) -> int:
    """Logical page count for ``capacity`` at ``block_size``.

    One more than ``ceil(capacity / block_size)`` so ``block_size * len`` is
    strictly greater than ``capacity``. The writer refuses a block table whose
    ``block_size * block_table_len`` does not strictly exceed ``max_live_count``;
    a full cache reaches exactly ``capacity``, so the extra page is what keeps a
    fully populated request runnable rather than a boundary refusal.
    """

    if int(capacity) <= 0:
        raise ValueError("capacity must be positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    return int(capacity) // int(block_size) + 1


def int8_kv_resident_bytes(
    *,
    capacity: int,
    max_block: int,
    attentions: Sequence[tuple[int, int, int]],
    scale_dtype: DType = DType.FP16,
    block_size: int = DEFAULT_INT8_KV_BLOCK_SIZE,
) -> int:
    """Bytes the owner holds for ``attentions`` ``(num_heads, num_kv_heads, head_dim)``.

    A pure size computation with no allocation, so the runner's block-fit
    planner can ask before anything is taken.
    """

    scale_dtype = DType.parse(scale_dtype)
    blocks = int8_kv_block_table_len(capacity, block_size)
    scale_bytes = scale_dtype.itemsize
    total = max_block * blocks * _I32_BYTES  # identity page table
    total += 2 * max_block * _I64_BYTES  # append positions and live counts
    for num_heads, num_kv_heads, head_dim in attentions:
        payload = blocks * block_size * num_kv_heads * head_dim
        total += 2 * payload * _INT8_BYTES
        total += 2 * blocks * block_size * num_kv_heads * scale_bytes
        total += 2 * max_block * num_heads * head_dim * _F32_BYTES
    return total


@dataclass(frozen=True)
class Gemma4Int8Block:
    """The per-block views shared by every layer of one forward block."""

    rows: int
    write_offset: int
    positions: Tensor  # (rows,) int64 absolute append positions
    counts: Tensor  # (rows,) int64 causal prefix lengths (positions + 1)
    base_offsets_1d: Tensor  # (blocks,) int32 shared prefix page table
    base_offsets_2d: Tensor  # (rows, blocks) int32 repeated identity table


@dataclass(frozen=True)
class Gemma4LayerInt8KV:
    """One layer's INT8 cache pointers and the spans for this block.

    Duck-typed by the layer, which resolves the writer and consumer from the
    registry using ``writer_spans``/``consumer_spans`` and the backend. Kept as a
    plain value object so the layer package does not import the runtime.
    """

    backend: str
    key_cache: int
    value_cache: int
    k_scale: int
    v_scale: int
    query_f32: int
    context_f32: int
    writer_spans: KVLiveSpans
    consumer_spans: KVLiveSpans
    block_size: int
    max_context_len: int


@dataclass
class Gemma4Int8KVCache:
    """Owns the INT8 KV payload, scales, page table and metadata for one runner."""

    capacity: int
    max_block: int
    attentions: tuple[tuple[int, int, int], ...]
    backend: str = "hip_gfx1100"
    scale_dtype: DType = DType.FP16
    block_size: int = DEFAULT_INT8_KV_BLOCK_SIZE
    device: Device = _DEVICE
    blocks: int = field(init=False)
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _key_caches: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _value_caches: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _k_scales: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _v_scales: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _query_f32: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _context_f32: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _scale_metadata: list[KVScaleMetadata] = field(default_factory=list, repr=False)
    _page_table: DeviceBuffer | None = field(default=None, repr=False)
    _positions: DeviceBuffer | None = field(default=None, repr=False)
    _counts: DeviceBuffer | None = field(default=None, repr=False)
    _positions_host: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _counts_host: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _base_offsets_1d: Tensor | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.capacity) <= 0:
            raise ValueError("capacity must be positive")
        if int(self.max_block) <= 0:
            raise ValueError("max_block must be positive")
        if int(self.max_block) > int(self.capacity):
            raise ValueError("max_block must not exceed capacity")
        self.scale_dtype = DType.parse(self.scale_dtype)
        if self.scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError("INT8 KV scale dtype must be fp16 or fp32")
        if not self.attentions:
            raise ValueError("INT8 KV cache needs at least one attention geometry")
        self.blocks = int8_kv_block_table_len(self.capacity, self.block_size)
        for num_heads, num_kv_heads, head_dim in self.attentions:
            if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
                raise ValueError("attention geometry entries must be positive")
            if num_heads % num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
            int8_kv_consumer_shared_bytes(capacity=self.capacity, head_dim=head_dim)
        try:
            self._allocate()
        except BaseException:
            self._closed = True
            self._release_owned()
            raise

    # --- allocation -------------------------------------------------------

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        if nbytes <= 0:
            raise ValueError(f"refusing to allocate {nbytes} bytes")
        buffer = malloc(nbytes)
        self._buffers.append(buffer)
        return buffer

    def _allocate(self) -> None:
        scale_bytes = self.scale_dtype.itemsize
        payload_elements = self.blocks * self.block_size
        scale_shape = (self.blocks, self.block_size)

        for num_heads, num_kv_heads, head_dim in self.attentions:
            key = self._alloc(payload_elements * num_kv_heads * head_dim * _INT8_BYTES)
            value = self._alloc(payload_elements * num_kv_heads * head_dim * _INT8_BYTES)
            k_scale = self._alloc(payload_elements * num_kv_heads * scale_bytes)
            v_scale = self._alloc(payload_elements * num_kv_heads * scale_bytes)
            self._key_caches.append(key)
            self._value_caches.append(value)
            self._k_scales.append(k_scale)
            self._v_scales.append(v_scale)
            self._scale_metadata.append(
                KVScaleMetadata(
                    k_scale=Tensor.from_handle(
                        k_scale.ptr,
                        (scale_shape[0], scale_shape[1], num_kv_heads),
                        self.scale_dtype,
                        self.device,
                    ),
                    v_scale=Tensor.from_handle(
                        v_scale.ptr,
                        (scale_shape[0], scale_shape[1], num_kv_heads),
                        self.scale_dtype,
                        self.device,
                    ),
                    scale_dtype=self.scale_dtype,
                    granularity="per_token_head",
                )
            )
            self._query_f32.append(
                self._alloc(self.max_block * num_heads * head_dim * _F32_BYTES)
            )
            self._context_f32.append(
                self._alloc(self.max_block * num_heads * head_dim * _F32_BYTES)
            )

        # Identity page table, repeated for every row of the writer's row-major
        # view. One resident buffer; the 1-D and row-major views are aliases.
        identity = np.tile(
            np.arange(self.blocks, dtype=np.int32), (self.max_block, 1)
        ).reshape(-1)
        page_table = self._alloc(identity.nbytes)
        copy_host_to_device(page_table, host_array_ptr(identity), identity.nbytes)
        self._page_table = page_table
        self._base_offsets_1d = Tensor.from_handle(
            page_table.ptr, (self.blocks,), DType.INT32, self.device
        )

        self._positions_host = np.zeros(self.max_block, dtype=np.int64)
        self._counts_host = np.zeros(self.max_block, dtype=np.int64)
        self._positions = self._alloc(self.max_block * _I64_BYTES)
        self._counts = self._alloc(self.max_block * _I64_BYTES)

    # --- per-block spans --------------------------------------------------

    def begin_block(
        self, *, write_offset: int, rows: int, stream: int = 0
    ) -> Gemma4Int8Block:
        """Stage this block's append positions and live counts on ``stream``.

        ``positions[i] = write_offset + i`` is the absolute slot row ``i`` is
        appended at; the writer consumes it as the append position and the
        consumer as the query's row position. ``counts[i] = positions[i] + 1``
        is the consumer's causal prefix length. Both are enqueued before the
        writer/consumer launches that read them, so the checked consumer's
        readback sees this block's values.
        """

        if self._closed:
            raise RuntimeError("INT8 KV cache is closed")
        rows = int(rows)
        write_offset = int(write_offset)
        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows > self.max_block:
            raise ValueError(f"rows={rows} exceeds max_block {self.max_block}")
        if write_offset < 0 or write_offset + rows > self.capacity:
            raise ValueError(
                f"{rows} tokens from position {write_offset} exceeds capacity {self.capacity}"
            )
        positions = write_offset + np.arange(rows, dtype=np.int64)
        self._positions_host[:rows] = positions
        self._counts_host[:rows] = positions + 1
        assert self._positions is not None and self._counts is not None
        assert self._page_table is not None and self._base_offsets_1d is not None
        enqueue_host_to_device(
            self._positions,
            host_array_ptr(np.ascontiguousarray(self._positions_host[:rows])),
            rows * _I64_BYTES,
            stream=stream,
        )
        enqueue_host_to_device(
            self._counts,
            host_array_ptr(np.ascontiguousarray(self._counts_host[:rows])),
            rows * _I64_BYTES,
            stream=stream,
        )
        return Gemma4Int8Block(
            rows=rows,
            write_offset=write_offset,
            positions=Tensor.from_handle(
                self._positions.ptr, (rows,), DType.INT64, self.device
            ),
            counts=Tensor.from_handle(
                self._counts.ptr, (rows,), DType.INT64, self.device
            ),
            base_offsets_1d=self._base_offsets_1d,
            base_offsets_2d=Tensor.from_handle(
                self._page_table.ptr, (rows, self.blocks), DType.INT32, self.device
            ),
        )

    def layer_kv(self, index: int, block: Gemma4Int8Block) -> Gemma4LayerInt8KV:
        """Build one layer's cache pointers and writer/consumer spans for ``block``."""

        if self._closed:
            raise RuntimeError("INT8 KV cache is closed")
        if not 0 <= index < len(self._scale_metadata):
            raise IndexError(f"layer index {index} out of range")
        metadata = self._scale_metadata[index]
        role = "prefill" if block.rows > 1 else "decode"
        # Writer ABI: row-major [rows, blocks] table, positions as live_counts.
        writer_spans = KVLiveSpans(
            base_offsets=block.base_offsets_2d,
            live_counts=block.positions,
            max_live_count=self.capacity,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            row_positions=None,
            span_role=role,
            scale_metadata=metadata,
        )
        # Consumer ABI: shared 1-D table, per-row counts, per-row positions.
        consumer_spans = KVLiveSpans(
            base_offsets=block.base_offsets_1d,
            live_counts=block.counts,
            max_live_count=self.capacity,
            token_positions=None,
            evict_mask=None,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            spans_mode="uniform",
            row_positions=block.positions,
            span_role=role,
            scale_metadata=metadata,
        )
        return Gemma4LayerInt8KV(
            backend=self.backend,
            key_cache=self._key_caches[index].ptr,
            value_cache=self._value_caches[index].ptr,
            k_scale=self._k_scales[index].ptr,
            v_scale=self._v_scales[index].ptr,
            query_f32=self._query_f32[index].ptr,
            context_f32=self._context_f32[index].ptr,
            writer_spans=writer_spans,
            consumer_spans=consumer_spans,
            block_size=self.block_size,
            max_context_len=self.capacity,
        )

    # --- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Rewind to an empty sequence.

        No device copy: the payload and scale planes are overwritten by the
        writer for every slot the consumers read. A request reads only
        ``[0, live_count)`` slots, and each of those was written by this
        request's own append before the read, so a shorter sequence after a
        longer one cannot read a previous request's bytes. The staging arrays
        are zeroed so a stale host value cannot be re-enqueued by a later
        ``begin_block`` that writes fewer rows than the last one.
        """

        if self._closed:
            return
        if self._positions_host is not None:
            self._positions_host[:] = 0
        if self._counts_host is not None:
            self._counts_host[:] = 0

    def _release_owned(self) -> None:
        for buffer in self._buffers:
            free(buffer)
        self._buffers.clear()
        self._key_caches.clear()
        self._value_caches.clear()
        self._k_scales.clear()
        self._v_scales.clear()
        self._query_f32.clear()
        self._context_f32.clear()
        self._scale_metadata.clear()
        self._page_table = None
        self._positions = None
        self._counts = None
        self._base_offsets_1d = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_owned()

    # --- introspection ----------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def scale_metadata(self) -> tuple[KVScaleMetadata, ...]:
        return tuple(self._scale_metadata)

    @property
    def key_caches(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._key_caches)

    @property
    def value_caches(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._value_caches)

    @property
    def k_scale_buffers(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._k_scales)

    @property
    def v_scale_buffers(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._v_scales)

    @property
    def query_f32_buffers(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._query_f32)

    @property
    def context_f32_buffers(self) -> tuple[DeviceBuffer, ...]:
        return tuple(self._context_f32)

    @property
    def allocated_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers)
