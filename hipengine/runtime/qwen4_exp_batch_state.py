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
    ) -> "Qwen4ExpBatchAttentionState":
        """Allocate ``state_slots`` disjoint paged caches of ``max_positions``.

        Total capacity is ``state_slots * ceil(max_positions / block_size)``
        blocks, and slot ``s`` owns exactly blocks
        ``[s * blocks_per_request, (s + 1) * blocks_per_request)``. Capacity is
        therefore additive in requests rather than shared.

        Geometry whose products the kernels cannot address (int32 physical block
        ids, int64 positions, int64 K/V element offsets and byte sizes) is
        rejected before a runtime is resolved or a byte is allocated.
        """

        slots = _positive_int(state_slots, "state_slots")
        capacity = _positive_int(max_positions, "max_positions")
        parsed_block = _positive_int(block_size, "block_size")
        parsed_kv_heads = _positive_int(kv_heads, "kv_heads")
        parsed_head_dim = _positive_int(head_dim, "head_dim")
        blocks_per_request = (capacity + parsed_block - 1) // parsed_block
        # Bound every product the kernels and the host index with before taking a
        # runtime or allocating anything, so an impossible shape fails clearly.
        _check_shape_products(
            state_slots=slots,
            blocks_per_request=blocks_per_request,
            block_size=parsed_block,
            kv_heads=parsed_kv_heads,
            head_dim=parsed_head_dim,
            max_positions=capacity,
        )
        active_runtime = runtime or get_hip_runtime()

        block_host = np.arange(
            slots * blocks_per_request, dtype=np.int32
        ).reshape(slots, blocks_per_request)
        block_table_host = np.zeros((slots, blocks_per_request), dtype=np.int32)
        position_host = np.zeros(slots, dtype=np.int64)
        context_host = np.zeros(slots, dtype=np.int64)

        token_bytes = parsed_kv_heads * parsed_head_dim * 2
        slot_bytes = blocks_per_request * parsed_block * token_bytes
        buffers: list[DeviceBuffer] = []
        try:
            key_cache = malloc(slots * slot_bytes, runtime=active_runtime)
            buffers.append(key_cache)
            value_cache = malloc(slots * slot_bytes, runtime=active_runtime)
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
        return self.state_slots * self.slot_capacity_tokens

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

    def prepare(
        self,
        slot_ids: Sequence[int],
        positions: Sequence[int],
    ) -> Qwen4ExpBatchActiveRows:
        """Admit active rows and upload their tables, positions, and contexts.

        ``slot_ids[r]`` is the physical slot that serves row ``r``, so the row
        order is the caller's (sparse and permuted slot sets are supported) and
        every slot may appear at most once. ``positions[r]`` is the row's next
        append position; the decode context length is ``positions[r] + 1``.

        Metadata is validated in full before any device write, so a rejected
        call leaves both host mirrors and device buffers exactly as they were.
        A device-copy failure clears the active set instead of publishing a
        half-uploaded row table.
        """

        self._require_open()
        slots = _index_sequence(slot_ids, "slot_ids")
        parsed_positions = _index_sequence(positions, "positions")
        if not slots:
            raise ValueError("prepare requires at least one active row")
        if len(slots) != len(parsed_positions):
            raise ValueError(
                f"slot_ids and positions must have equal length, got {len(slots)} and "
                f"{len(parsed_positions)}"
            )
        if len(slots) > self.state_slots:
            raise ValueError(
                f"prepare admits at most {self.state_slots} active rows, got {len(slots)}"
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

        rows = len(slots)
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
            copy_host_to_device(
                self.position,
                host_array_ptr(positions_array),
                positions_array.nbytes,
                runtime=self.runtime,
            )
            copy_host_to_device(
                self.context,
                host_array_ptr(contexts),
                contexts.nbytes,
                runtime=self.runtime,
            )
        except BaseException:
            self.active = None
            raise

        self.block_table_host[:rows] = table
        self.position_host[:rows] = positions_array
        self.context_host[:rows] = contexts
        device = Device("hip", 0)
        block_table_tensor = Tensor.from_handle(
            self.block_table.ptr, table.shape, DType.INT32, device
        )
        position_tensor = Tensor.from_handle(
            self.position.ptr, (rows,), DType.INT64, device
        )
        context_tensor = Tensor.from_handle(
            self.context.ptr, (rows,), DType.INT64, device
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
            slot_ids=tuple(slots),
            block_table=block_table_tensor,
            position=position_tensor,
            context=context_tensor,
            append_spans=append_spans,
            decode_spans=decode_spans,
            max_context_len=max_context_len,
        )
        self.active = active
        return active

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
    state_slots: int,
    blocks_per_request: int,
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
    total_blocks = state_slots * blocks_per_request
    if total_blocks - 1 > _INT32_MAX:
        raise ValueError(
            f"{state_slots} slots x {blocks_per_request} blocks = {total_blocks} "
            f"physical block ids exceed the int32 page-table range "
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
