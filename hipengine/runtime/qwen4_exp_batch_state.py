"""Per-request ownership for Qwen4Exp paged dense-attention state.

``Qwen4ExpDenseAttentionState`` in ``runtime/qwen4_exp_runner.py`` owns exactly
one request: one K/V cache, one page table, one int64 position/context scalar.
That is the c1 shape the runner dispatches today, and it is why a second
resident request needs a second full copy of the per-layer state.

This module is the ownership prerequisite for sharing one paged cache across
independent requests. Every physical state slot owns a *disjoint* block range
with a full ``max_positions`` capacity, so a request that lands in slot ``s``
keeps the capacity it would have had alone. ``prepare`` packs the active rows
into the leading rows of the metadata buffers while the physical slots those
rows name stay arbitrary (sparse and permuted sets are supported), so the
active width (``1 .. state_slots``) is independent of the physical capacity
(``state_slots``) and shrinking or regrowing the active set never reallocates
or remaps a slot.

Nothing here dispatches. It produces validated, uploaded per-row page tables
plus ``KVLiveSpans`` views for the existing batch paged-KV-write and
dense-attention decode wrappers. The runner still calls the c1 wrappers, so
this is an ownership/ABI unit, not a serving path.

Two properties the batch wrappers require and this module guarantees:

* ``base_offsets`` is a row-major ``(rows, blocks_per_request)`` int32 table, so
  row ``r`` addresses only slot ``slot_ids[r]``'s blocks. The batch write kernel
  reads ``block_table[row * block_table_len + logical_block]`` and the batch
  dense-attention kernel reads ``block_table + row * block_table_len``; both
  accept this layout, and the shared-table variants do not.
* ``live_counts`` is an int64 vector holding one append position per row (for
  ``append_spans``) or one context length per row (for ``decode_spans``).

The state is block-size agnostic; the dense-attention decode wrappers
themselves require ``block_size == 256``, and that check stays with them.

``Qwen4ExpBatchStepMetadata`` is the one packed upload that carries every QSA
owner's dynamic per-step metadata (position, context, index query position).
It owns one buffer and hands each layer stable, non-owning views of it, so a
step publishes validated rows through fixed addresses and a captured graph
keeps addressing the same bytes at every replay and every active width.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans

# Column order inside one layer's slice of the packed per-step metadata buffer.
# One int64 per slot per column; the runner stages every QSA owner's dynamic
# per-step metadata here and publishes it with a single host-to-device copy.
STEP_POSITION_COLUMN = 0
STEP_CONTEXT_COLUMN = 1
STEP_INDEX_POSITION_COLUMN = 2
#: Columns per layer: append position, decode context, index query position.
STEP_METADATA_COLUMNS = 3

# Integer ranges the paged kernels actually index with, not arbitrary limits.
# Page-table entries are int32 physical block ids; live_counts/position/context
# tensors are int64; the write and dense-context kernels build K/V element
# offsets as (physical_block * block_size + block_offset) * kv_heads * head_dim
# in int64, and the host sizes the allocation as a size_t byte count.
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class Qwen4ExpBatchActiveRows:
    """One prepared active row set, valid until the next ``prepare``."""

    rows: int
    slot_ids: tuple[int, ...]
    block_table: Tensor
    position: Tensor
    context: Tensor
    append_spans: KVLiveSpans
    decode_spans: KVLiveSpans
    max_context_len: int


@dataclass(frozen=True)
class Qwen4ExpBatchAttentionRowPlan:
    """One attention owner's validated row set, before anything is published.

    ``validate_rows`` builds it without touching a host mirror or a device
    buffer, so a caller can validate every owner's rows and only then perform
    its single packed metadata upload. ``publish`` consumes it.

    A plan describes the owner's geometry as it was when the plan was built, so
    it is valid only until that slot's page table or the caller's row set
    changes. The batch runner builds and publishes a plan inside one step.
    """

    slot_ids: tuple[int, ...]
    positions: np.ndarray
    contexts: np.ndarray
    block_table: np.ndarray
    max_context_len: int

    @property
    def rows(self) -> int:
        return len(self.slot_ids)


@dataclass(frozen=True)
class Qwen4ExpBatchPackedMetadataViews:
    """One layer's stable, non-owning views of the packed step metadata.

    All three views are slices of a single ``Qwen4ExpBatchStepMetadata`` buffer,
    so their addresses are fixed for that buffer's lifetime. A graph captured
    through them keeps addressing the same device bytes at every replay and at
    every active width; the row stride is the physical slot count, not the
    active width.
    """

    layer: int
    position: DeviceBuffer
    context: DeviceBuffer
    index_position: DeviceBuffer


@dataclass
class Qwen4ExpBatchStepMetadata:
    """One packed upload for every QSA owner's dynamic per-step metadata.

    A step must publish, for each QSA layer, one int64 append position, one
    int64 decode context, and one int64 index query position per active row.
    Uploading those per owner costs ``3 * layers`` small host-to-device copies
    per step; this owner holds all of them in one buffer and publishes them with
    one copy.

    The layout is ``[layer, column, slot]`` with the *physical* slot count as
    the row stride, never the active width. That is what makes a view's address
    independent of how many rows a step carries, and it is why ``upload`` copies
    the whole buffer instead of a prefix: a fixed size and fixed addresses are
    what a captured graph can replay. Rows past the active width are staged
    copies of nothing any kernel reads.

    Staging is a host-mirror write only. ``upload`` is the single device write,
    so a caller validates every owner's rows first, stages them all, and only
    then publishes them; a rejected stage cannot leave a half-uploaded device
    buffer behind.
    """

    buffer: DeviceBuffer
    host: np.ndarray
    layers: int
    state_slots: int
    runtime: HipRuntime
    closed: bool = False
    closing: bool = False
    _views: dict[int, Qwen4ExpBatchPackedMetadataViews] = field(default_factory=dict)
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def allocate(
        cls,
        *,
        layers: int,
        state_slots: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpBatchStepMetadata":
        """Allocate the packed buffer for ``layers`` owners of ``state_slots`` rows.

        ``layers`` may be zero (a configuration with no QSA layer); the buffer is
        then a single unused int64 so no caller has to special-case a null
        pointer, and ``views`` rejects every layer index.
        """

        parsed_layers = _int_index(layers, "layers")
        if parsed_layers < 0:
            raise ValueError(f"layers must not be negative, got {parsed_layers}")
        slots = _positive_int(state_slots, "state_slots")
        host = np.zeros(
            (parsed_layers, STEP_METADATA_COLUMNS, slots), dtype=np.int64
        )
        active_runtime = runtime or get_hip_runtime()
        buffer = malloc(
            max(host.nbytes, DType.INT64.itemsize), runtime=active_runtime
        )
        try:
            # A row that was never staged must read as a checkable value rather
            # than as uninitialized memory.
            active_runtime.memset(buffer.ptr, 0, buffer.nbytes)
        except BaseException as error:
            try:
                _free_pending([buffer], active_runtime)
            except Exception as cleanup_error:  # pragma: no cover - defensive
                error.add_note(
                    f"cleanup after the failed allocation also failed: {cleanup_error!r}"
                )
            raise
        return cls(
            buffer=buffer,
            host=host,
            layers=parsed_layers,
            state_slots=slots,
            runtime=active_runtime,
            _pending_free=[buffer],
        )

    # -- geometry ---------------------------------------------------------

    @property
    def bytes_per_column(self) -> int:
        """Bytes one layer's column holds (one int64 per physical slot)."""

        return self.state_slots * DType.INT64.itemsize

    def views(self, layer: int) -> Qwen4ExpBatchPackedMetadataViews:
        """``layer``'s views; the same object, and the same pointers, every call."""

        self._require_open()
        parsed = self._require_layer(layer)
        cached = self._views.get(parsed)
        if cached is None:
            stride = self.bytes_per_column
            base = self.buffer.ptr + parsed * STEP_METADATA_COLUMNS * stride
            cached = Qwen4ExpBatchPackedMetadataViews(
                layer=parsed,
                position=DeviceBuffer(base + STEP_POSITION_COLUMN * stride, stride),
                context=DeviceBuffer(base + STEP_CONTEXT_COLUMN * stride, stride),
                index_position=DeviceBuffer(
                    base + STEP_INDEX_POSITION_COLUMN * stride, stride
                ),
            )
            self._views[parsed] = cached
        return cached

    # -- staging and upload -----------------------------------------------

    def stage(
        self,
        layer: int,
        *,
        positions: object,
        contexts: object,
        index_positions: object,
    ) -> int:
        """Stage one layer's rows in host memory and return the row count.

        Every column is validated before the host mirror is written, so a
        rejected call leaves the staging array exactly as it was. Nothing here
        reaches the device: ``upload`` is the single device write.
        """

        self._require_open()
        parsed = self._require_layer(layer)
        columns = (
            (STEP_POSITION_COLUMN, "positions", positions),
            (STEP_CONTEXT_COLUMN, "contexts", contexts),
            (STEP_INDEX_POSITION_COLUMN, "index_positions", index_positions),
        )
        staged: list[tuple[int, np.ndarray]] = []
        rows = -1
        for column, name, values in columns:
            array = np.ascontiguousarray(
                np.asarray(_index_sequence(values, name), dtype=np.int64)
            )
            if rows < 0:
                rows = int(array.size)
                if rows <= 0:
                    raise ValueError("stage requires at least one active row")
                if rows > self.state_slots:
                    raise ValueError(
                        f"stage packs at most {self.state_slots} rows, got {rows}"
                    )
            elif array.size != rows:
                raise ValueError(
                    f"{name} must hold {rows} values, got {int(array.size)}"
                )
            staged.append((column, array))
        for column, array in staged:
            self.host[parsed, column, :rows] = array
        return rows

    def upload(self) -> None:
        """Publish every staged layer with one host-to-device copy."""

        self._require_open()
        if not self.host.size:
            return
        copy_host_to_device(
            self.buffer,
            host_array_ptr(self.host),
            self.host.nbytes,
            runtime=self.runtime,
        )

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        # Poison uploads before the first free: a partial close leaves the buffer
        # released and no upload may target it.
        self.closing = True
        pending = self._pending_free
        if not pending:
            pending.append(self.buffer)
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchStepMetadata":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp batch step metadata is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp batch step metadata is closing; its buffer may already be "
                "released"
            )

    def _require_layer(self, layer: object) -> int:
        parsed = _int_index(layer, "layer")
        if not 0 <= parsed < self.layers:
            raise ValueError(
                f"layer {parsed} is outside [0, {self.layers}) of the packed step "
                "metadata"
            )
        return parsed


@dataclass
class Qwen4ExpBatchAttentionState:
    """Disjoint per-slot paged K/V storage plus active-row span views."""

    key_cache: DeviceBuffer
    value_cache: DeviceBuffer
    block_table: DeviceBuffer
    position: DeviceBuffer
    context: DeviceBuffer
    block_host: np.ndarray
    block_table_host: np.ndarray
    position_host: np.ndarray
    context_host: np.ndarray
    block_size: int
    max_positions: int
    state_slots: int
    kv_heads: int
    head_dim: int
    runtime: HipRuntime
    active: Qwen4ExpBatchActiveRows | None = field(default=None)
    #: Non-owning device views the packed per-step metadata is published through.
    #: When set, ``publish`` reads positions and contexts from here and leaves
    #: the owner's own ``position``/``context`` buffers unused, so one packed
    #: upload can serve every layer. The views keep their addresses for the
    #: lifetime of the buffer that owns them.
    position_view: DeviceBuffer | None = field(default=None)
    context_view: DeviceBuffer | None = field(default=None)
    #: Physical blocks in the K/V slabs. Equal to ``state_slots *
    #: blocks_per_request`` for the static layout; smaller in pooled mode, where
    #: the batch runner assigns blocks to a slot at admission.
    pool_blocks: int = field(default=0)
    pooled: bool = field(default=False)
    closed: bool = field(default=False)
    # Set before the first free of a close attempt. Uploads are rejected from that
    # point on, because a partial close can leave earlier buffers released while
    # later ones are still owned; ``close`` itself may retry the remainder.
    closing: bool = field(default=False)
    # Buffers still owned, in allocation order. ``close`` frees the last entry
    # first (newest first) and drops an entry only after its free returns, so a
    # failed close can be retried without a double free.
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def allocate(
        cls,
        *,
        state_slots: int,
        max_positions: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        runtime: HipRuntime | None = None,
        pool_blocks: int | None = None,
        block_host: np.ndarray | None = None,
        position_view: DeviceBuffer | None = None,
        context_view: DeviceBuffer | None = None,
    ) -> "Qwen4ExpBatchAttentionState":
        """Allocate ``state_slots`` disjoint paged caches of ``max_positions``.

        Static layout (``pool_blocks=None``): total capacity is
        ``state_slots * ceil(max_positions / block_size)`` blocks, and slot
        ``s`` owns exactly blocks ``[s * blocks_per_request, (s + 1) *
        blocks_per_request)``, so capacity is additive in requests.

        Pooled layout: the slabs hold ``pool_blocks`` physical blocks shared by
        every slot, and ``block_host`` (``int32[state_slots,
        blocks_per_request]``) is the page table the owner fills when it assigns
        blocks to a slot. Passing the same array to every layer's state keeps
        their page tables identical. A slot still addresses at most
        ``max_positions`` logical positions.

        Geometry whose products the kernels cannot address (int32 physical block
        ids, int64 positions, int64 K/V element offsets and byte sizes) is
        rejected before a runtime is resolved or a byte is allocated.

        ``position_view`` and ``context_view`` are non-owning device views the
        owner publishes through instead of its own ``position``/``context``
        buffers. The batch runner passes slices of one
        ``Qwen4ExpBatchStepMetadata`` buffer, so a single packed upload serves
        every layer and the published addresses never move. Each view must hold
        exactly one int64 per state slot.
        """

        slots = _positive_int(state_slots, "state_slots")
        capacity = _positive_int(max_positions, "max_positions")
        parsed_block = _positive_int(block_size, "block_size")
        parsed_kv_heads = _positive_int(kv_heads, "kv_heads")
        parsed_head_dim = _positive_int(head_dim, "head_dim")
        blocks_per_request = (capacity + parsed_block - 1) // parsed_block
        pooled = pool_blocks is not None
        total_blocks = (
            _positive_int(pool_blocks, "pool_blocks")
            if pooled
            else slots * blocks_per_request
        )
        # Bound every product the kernels and the host index with before taking a
        # runtime or allocating anything, so an impossible shape fails clearly.
        _check_shape_products(
            total_blocks=total_blocks,
            block_size=parsed_block,
            kv_heads=parsed_kv_heads,
            head_dim=parsed_head_dim,
            max_positions=capacity,
        )
        if pooled:
            if block_host is None:
                block_host = np.zeros((slots, blocks_per_request), dtype=np.int32)
            if block_host.shape != (slots, blocks_per_request) or block_host.dtype != np.int32:
                raise ValueError(
                    "a pooled page table must be int32[state_slots, blocks_per_request]"
                )
        elif block_host is not None:
            raise ValueError("block_host is only meaningful with pool_blocks")
        else:
            block_host = np.arange(
                slots * blocks_per_request, dtype=np.int32
            ).reshape(slots, blocks_per_request)
        for name, view in (
            ("position_view", position_view),
            ("context_view", context_view),
        ):
            if view is None:
                continue
            if not isinstance(view, DeviceBuffer):
                raise TypeError(f"{name} must be a DeviceBuffer")
            if int(view.nbytes) != slots * DType.INT64.itemsize:
                raise ValueError(
                    f"{name} must hold exactly one int64 per state slot "
                    f"({slots * DType.INT64.itemsize} bytes), got {int(view.nbytes)}"
                )
        active_runtime = runtime or get_hip_runtime()

        block_table_host = np.zeros((slots, blocks_per_request), dtype=np.int32)
        position_host = np.zeros(slots, dtype=np.int64)
        context_host = np.zeros(slots, dtype=np.int64)

        token_bytes = parsed_kv_heads * parsed_head_dim * 2
        cache_bytes = total_blocks * parsed_block * token_bytes
        buffers: list[DeviceBuffer] = []
        try:
            key_cache = malloc(cache_bytes, runtime=active_runtime)
            buffers.append(key_cache)
            value_cache = malloc(cache_bytes, runtime=active_runtime)
            buffers.append(value_cache)
            block_table = malloc(block_table_host.nbytes, runtime=active_runtime)
            buffers.append(block_table)
            position = malloc(position_host.nbytes, runtime=active_runtime)
            buffers.append(position)
            context = malloc(context_host.nbytes, runtime=active_runtime)
            buffers.append(context)
            # Publish known-zero metadata so an inactive row can never address a
            # stale table, then zero K/V so "never written" is a checkable value.
            copy_host_to_device(
                block_table, host_array_ptr(block_table_host), runtime=active_runtime
            )
            copy_host_to_device(
                position, host_array_ptr(position_host), runtime=active_runtime
            )
            copy_host_to_device(
                context, host_array_ptr(context_host), runtime=active_runtime
            )
            active_runtime.memset(key_cache.ptr, 0, key_cache.nbytes)
            active_runtime.memset(value_cache.ptr, 0, value_cache.nbytes)
        except BaseException as error:
            try:
                _free_pending(buffers, active_runtime)
            except Exception as cleanup_error:  # pragma: no cover - defensive
                error.add_note(
                    f"cleanup after the failed allocation also failed: {cleanup_error!r}"
                )
            raise
        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            position=position,
            context=context,
            block_host=block_host,
            block_table_host=block_table_host,
            position_host=position_host,
            context_host=context_host,
            block_size=parsed_block,
            max_positions=capacity,
            state_slots=slots,
            kv_heads=parsed_kv_heads,
            head_dim=parsed_head_dim,
            runtime=active_runtime,
            pool_blocks=total_blocks,
            pooled=pooled,
            position_view=position_view,
            context_view=context_view,
            _pending_free=list(buffers),
        )

    # -- geometry ---------------------------------------------------------

    @property
    def blocks_per_request(self) -> int:
        return int(self.block_host.shape[1])

    @property
    def slot_capacity_tokens(self) -> int:
        """Paged capacity of one slot, in tokens (``>= max_positions``)."""

        return self.blocks_per_request * self.block_size

    @property
    def total_capacity_tokens(self) -> int:
        """Physical positions in the slabs (the pool, in pooled mode)."""

        blocks = self.pool_blocks or self.state_slots * self.blocks_per_request
        return blocks * self.block_size

    @property
    def cache_bytes_per_slot(self) -> int:
        return self.key_cache.nbytes // self.state_slots

    @property
    def active_rows(self) -> int:
        return 0 if self.active is None else self.active.rows

    @property
    def append_spans(self) -> KVLiveSpans:
        return self._require_active().append_spans

    @property
    def decode_spans(self) -> KVLiveSpans:
        return self._require_active().decode_spans

    def blocks_for_slot(self, slot: int) -> np.ndarray:
        """Physical block ids owned by ``slot`` (a host copy)."""

        parsed = _int_index(slot, "slot")
        if not 0 <= parsed < self.state_slots:
            raise ValueError(f"slot {parsed} is outside [0, {self.state_slots})")
        return self.block_host[parsed].copy()

    # -- active rows ------------------------------------------------------

    def validate_rows(
        self,
        slot_ids: Sequence[int],
        positions: Sequence[int],
    ) -> Qwen4ExpBatchAttentionRowPlan:
        """Validate one row set and return its plan without publishing it.

        ``slot_ids[r]`` is the physical slot that serves row ``r``, so the row
        order is the caller's (sparse and permuted slot sets are supported) and
        every slot may appear at most once. ``positions[r]`` is the row's next
        append position; the decode context length is ``positions[r] + 1``.

        Nothing here writes a host mirror or a device buffer, so a caller that
        must publish several owners atomically can validate all of them before
        its first device write.
        """

        self._require_open()
        slots = _index_sequence(slot_ids, "slot_ids")
        parsed_positions = _index_sequence(positions, "positions")
        if not slots:
            raise ValueError("a row set requires at least one active row")
        if len(slots) != len(parsed_positions):
            raise ValueError(
                f"slot_ids and positions must have equal length, got {len(slots)} and "
                f"{len(parsed_positions)}"
            )
        if len(slots) > self.state_slots:
            raise ValueError(
                f"a row set admits at most {self.state_slots} active rows, got "
                f"{len(slots)}"
            )
        seen: set[int] = set()
        for slot in slots:
            if not 0 <= slot < self.state_slots:
                raise ValueError(f"slot {slot} is outside [0, {self.state_slots})")
            if slot in seen:
                raise ValueError(f"slot {slot} appears twice in one active row set")
            seen.add(slot)
        for row, position in enumerate(parsed_positions):
            if not 0 <= position < self.max_positions:
                raise ValueError(
                    f"row {row} position {position} is outside [0, {self.max_positions})"
                )

        table = np.ascontiguousarray(self.block_host[np.asarray(slots, dtype=np.int64)])
        positions_array = np.ascontiguousarray(
            np.asarray(parsed_positions, dtype=np.int64)
        )
        contexts = positions_array + 1
        max_context_len = int(contexts.max())
        needed_blocks = (max_context_len + self.block_size - 1) // self.block_size
        if needed_blocks > self.blocks_per_request:
            raise ValueError(
                f"context {max_context_len} needs {needed_blocks} blocks but each row "
                f"table holds {self.blocks_per_request}"
            )
        return Qwen4ExpBatchAttentionRowPlan(
            slot_ids=tuple(slots),
            positions=positions_array,
            contexts=contexts,
            block_table=table,
            max_context_len=max_context_len,
        )

    def publish(
        self,
        plan: Qwen4ExpBatchAttentionRowPlan,
        *,
        packed: Qwen4ExpBatchPackedMetadataViews | None = None,
    ) -> Qwen4ExpBatchActiveRows:
        """Publish one validated row set as this owner's active rows.

        ``packed`` names the stable views that already hold this row set's
        positions and contexts, uploaded by the batch runner as part of one
        packed copy. With it, this owner uploads only its page table, and only
        when that table changed, so steady decode performs no copy here at all.

        A device-copy failure clears the active set instead of publishing a
        half-uploaded row table.
        """

        self._require_open()
        rows = _require_row_plan(
            plan,
            state_slots=self.state_slots,
            max_positions=self.max_positions,
            blocks_per_request=self.blocks_per_request,
        )
        if packed is not None:
            self._require_packed_views(packed)
        table = plan.block_table
        try:
            # ``block_table_host`` mirrors the device table, and a row's blocks
            # change only with its slot, so steady decode skips this upload.
            if not np.array_equal(self.block_table_host[:rows], table):
                copy_host_to_device(
                    self.block_table,
                    host_array_ptr(table),
                    table.nbytes,
                    runtime=self.runtime,
                )
            if packed is None:
                copy_host_to_device(
                    self.position,
                    host_array_ptr(plan.positions),
                    plan.positions.nbytes,
                    runtime=self.runtime,
                )
                copy_host_to_device(
                    self.context,
                    host_array_ptr(plan.contexts),
                    plan.contexts.nbytes,
                    runtime=self.runtime,
                )
        except BaseException:
            self.active = None
            raise

        self.block_table_host[:rows] = table
        self.position_host[:rows] = plan.positions
        self.context_host[:rows] = plan.contexts
        device = Device("hip", 0)
        block_table_tensor = Tensor.from_handle(
            self.block_table.ptr, table.shape, DType.INT32, device
        )
        position_buffer = self.position if packed is None else packed.position
        context_buffer = self.context if packed is None else packed.context
        position_tensor = Tensor.from_handle(
            position_buffer.ptr, (rows,), DType.INT64, device
        )
        context_tensor = Tensor.from_handle(
            context_buffer.ptr, (rows,), DType.INT64, device
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=position_tensor,
            max_live_count=self.max_positions - 1,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=self.max_positions,
            storage_dtype=DType.BF16,
        )
        active = Qwen4ExpBatchActiveRows(
            rows=rows,
            slot_ids=plan.slot_ids,
            block_table=block_table_tensor,
            position=position_tensor,
            context=context_tensor,
            append_spans=append_spans,
            decode_spans=decode_spans,
            max_context_len=plan.max_context_len,
        )
        self.active = active
        return active

    def prepare(
        self,
        slot_ids: Sequence[int],
        positions: Sequence[int],
        *,
        packed: Qwen4ExpBatchPackedMetadataViews | None = None,
    ) -> Qwen4ExpBatchActiveRows:
        """Validate and publish one row set in a single call.

        This is the standalone form of ``validate_rows`` followed by
        ``publish``: it uploads its own positions and contexts unless ``packed``
        names views that already hold them.
        """

        return self.publish(self.validate_rows(slot_ids, positions), packed=packed)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        # Poison uploads before the first free: a partial close leaves some
        # buffers released and some owned, and no upload may target either.
        self.closing = True
        self.active = None
        pending = self._pending_free
        if not pending:
            # Ownership was not handed over by ``allocate`` (direct construction);
            # take it now so closing cannot silently leak.
            pending.extend(
                (
                    self.key_cache,
                    self.value_cache,
                    self.block_table,
                    self.position,
                    self.context,
                )
            )
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchAttentionState":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp batch attention state is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp batch attention state is closing; its buffers may already "
                "be released"
            )

    def _require_active(self) -> Qwen4ExpBatchActiveRows:
        self._require_open()
        if self.active is None:
            raise RuntimeError(
                "Qwen4Exp batch attention state has no active rows; call prepare first"
            )
        return self.active

    def _require_packed_views(self, packed: Qwen4ExpBatchPackedMetadataViews) -> None:
        """Refuse packed views that cannot carry one int64 per state slot."""

        if not isinstance(packed, Qwen4ExpBatchPackedMetadataViews):
            raise TypeError("packed must be Qwen4ExpBatchPackedMetadataViews")
        expected = self.state_slots * DType.INT64.itemsize
        for name, view in (
            ("packed.position", packed.position),
            ("packed.context", packed.context),
        ):
            if not isinstance(view, DeviceBuffer):
                raise TypeError(f"{name} must be a DeviceBuffer")
            if int(view.nbytes) != expected:
                raise ValueError(
                    f"{name} must hold exactly one int64 per state slot "
                    f"({expected} bytes), got {int(view.nbytes)}"
                )


def _require_row_plan(
    plan: object,
    *,
    state_slots: int,
    max_positions: int,
    blocks_per_request: int,
) -> int:
    """Check a plan's shape and return its row count, before any device write.

    ``publish`` accepts only a plan ``validate_rows`` produced, so this re-checks
    the plan's own scalar invariants rather than rebuilding the page table it
    already carries. A plan built for another geometry (a different slot count or
    page-table width) or hand-built with inconsistent arrays is refused here
    instead of being published as a mismatched row table.
    """

    if not isinstance(plan, Qwen4ExpBatchAttentionRowPlan):
        raise TypeError("publish requires a plan returned by validate_rows")
    rows = plan.rows
    if rows <= 0:
        raise ValueError("a row plan requires at least one active row")
    if rows > state_slots:
        raise ValueError(
            f"a row plan admits at most {state_slots} active rows, got {rows}"
        )
    if plan.positions.shape != (rows,) or plan.contexts.shape != (rows,):
        raise ValueError(
            "a row plan must hold exactly one position and one context per row"
        )
    if plan.block_table.shape != (rows, blocks_per_request):
        raise ValueError(
            f"a row plan must hold an int32[{rows}, {blocks_per_request}] page table, "
            f"got {plan.block_table.shape}"
        )
    seen: set[int] = set()
    for slot in plan.slot_ids:
        if not 0 <= slot < state_slots:
            raise ValueError(f"slot {slot} is outside [0, {state_slots})")
        if slot in seen:
            raise ValueError(f"slot {slot} appears twice in one active row set")
        seen.add(slot)
    for row, position in enumerate(plan.positions.tolist()):
        if not 0 <= int(position) < max_positions:
            raise ValueError(
                f"row {row} position {int(position)} is outside [0, {max_positions})"
            )
    return rows


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first (last entry first), popping after each free.

    A failed ``free`` leaves that buffer and everything older in ``pending``, so
    retrying releases exactly the remainder instead of double-freeing the
    buffers that are already gone.
    """

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _check_shape_products(
    *,
    total_blocks: int,
    block_size: int,
    kv_heads: int,
    head_dim: int,
    max_positions: int,
) -> None:
    """Reject shapes whose products the kernels' own integer types cannot hold."""

    if max_positions > _INT64_MAX:
        raise ValueError(
            f"max_positions {max_positions} exceeds the int64 position/context "
            f"range (max {_INT64_MAX})"
        )
    if total_blocks - 1 > _INT32_MAX:
        raise ValueError(
            f"{total_blocks} physical block ids exceed the int32 page-table range "
            f"(max block id {_INT32_MAX})"
        )
    kv_elements = total_blocks * block_size * kv_heads * head_dim
    if kv_elements > _INT64_MAX:
        raise ValueError(
            f"{kv_elements} K/V elements exceed the int64 element index the paged "
            f"kernels compute (max {_INT64_MAX})"
        )
    if kv_elements * 2 > _INT64_MAX:
        raise ValueError(
            f"{kv_elements * 2} K/V bytes exceed the int64/size_t allocation bound "
            f"(max {_INT64_MAX})"
        )


def _int_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _positive_int(value: object, name: str) -> int:
    parsed = _int_index(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _index_sequence(values: object, name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        items = list(values)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None
    return [_int_index(item, name) for item in items]
