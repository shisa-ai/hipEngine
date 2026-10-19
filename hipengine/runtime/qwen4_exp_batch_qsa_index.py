"""Independent-request sparse QSA index state and block selection.

``Qwen4ExpQSAIndexDeviceState`` (``runtime/qwen4_exp_runner.py:1149``) is the
strict single-request index owner: one raw-key slab addressed through one page
table, one member map, one cursor pair (``count``/``pooled_count``), one score
row, and one selection. It is the owner the c1 mixer drives, and it is the
reference this module is measured against.

This module is the N-request form on top of ``Qwen4ExpBatchAttentionState``.
Rows are independent requests; a row's physical storage is its slot, and slot
ownership comes from the batch attention owner's page tables, which partition
the global physical block range disjointly (slot ``s`` owns blocks
``[s * blocks_per_request, (s + 1) * blocks_per_request)``). The raw-key slab is
therefore indexed by *global* physical position, exactly the value the scatter
kernel computes, and two slots can never alias a key.

## What is actually batched

Per step with ``N`` active rows, using only the committed primitives:

============  ==============  ==================================================
stage         dispatches     why
============  ==============  ==================================================
append keys   ``N``           ``scatter_index_key_device_position_f32`` takes one
                              page table and one position scalar, so a row-indexed
                              append needs one launch per row.
pool blocks   ``<= N``        ``pool_norm_rope_f32`` writes to a contiguous output
                              range from one member map, so per-slot output
                              offsets are not expressible; one launch per slot
                              that completes a block.
score         ``N``           Every committed score ABI (``score_f32``,
                              ``score_rows_f32``) shares **one** ``pooled_keys``
                              array across its queries. Passing ``query_count=N``
                              would score N queries against one request's keys,
                              which is not independent-request scoring.
select        ``1``           ``topk_expand_rows_f32_i64`` is already row-batched:
                              per-row query positions, per-row score stride,
                              per-row output slice and count.
============  ==============  ==================================================

So this module is **not** a physical batching of the index path. It is a
correct request-owned index state whose selection stage is genuinely batched and
whose append/pool/score stages are per-slot serial dispatches. Selection results
are bit-identical to the c1 owner because the batched kernel is the same kernel
body with the same per-row inputs; the per-slot stages are the same primitives
with the same arguments the c1 owner passes.

Real physical batching needs three primitives that do not exist yet. Their exact
boundary, stated as the ABI they would need (nothing here assumes they exist):

1. ``qsa_scatter_index_keys_rows_f32(source, destination, block_tables,
   positions, rows, block_table_stride, block_size, index_dim, stream)`` —
   row ``r`` reads ``block_tables[r * block_table_stride ...]`` and
   ``positions[r]`` instead of one table and one scalar.
2. ``qsa_pool_norm_rope_rows_f32(raw_keys, member_indices, block_starts, weight,
   output, block_offsets, blocks, ratio, index_dim, rotary_dim, theta, eps,
   stream)`` — a per-block output offset so N slots' newly complete blocks land
   in their own pooled-key slabs in one launch.
3. ``qsa_score_rows_indexed_f32(queries, pooled_keys, scores, block_counts, rows,
   pooled_stride, score_stride, heads, index_dim, stream)`` — row ``r`` scores
   ``block_counts[r]`` keys starting at ``pooled_keys[r * pooled_stride]`` into
   ``scores[r * score_stride]``. This is the essential one: without a per-row key
   base there is no independent-request scoring.

## Scope

Decode-shaped selection only: one index key appended per active row per step, and
a sparse selection for a row whose complete-block count exceeds the budget. A
prefill chunk's rows are one request's consecutive positions, not independent
requests, and stay on the c1 owner. Rows whose complete-block count is within the
budget are inside the dense equivalence bound (``dense_equivalent_limit``) and are
rejected rather than silently given dense-equivalent selection; ``partition_rows``
splits an active set so a caller can route dense rows to the dense mixer and
sparse rows here.

Nothing in this module dispatches by backend or quant: every device call is a
public kernel wrapper.

## Rollback is a bounded transaction, not a snapshot

``begin_rollback`` issues one single-use token per owner; ``rollback`` applies it
and consumes it; ``discard_rollback`` abandons it. The token is valid only for
append-only rollback within the current per-slot generation on the owner that
issued it, because only the counters are captured (see
:class:`Qwen4ExpBatchQsaIndexRollback` for why that is sufficient). A slot reset
advances that slot's generation and therefore permanently invalidates the token,
a token presented to another owner is refused, a token that would move a slot
forward is refused, and a token can never be applied twice, so a rewind followed
by an overwrite cannot be re-advanced by replaying an old token. Arbitrary
snapshots, multiple live tokens, and cross-generation restores are deliberately
unsupported rather than approximated.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_pool_norm_rope_f32,
    qwen4_exp_qsa_scatter_index_key_device_position_f32,
    qwen4_exp_qsa_score_f32,
    qwen4_exp_qsa_topk_expand_rows_f32_i64,
)
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState

# ``member_indices`` is int32, so a global physical position must fit it.
_QSA_INDEX_MAX_PHYSICAL_POSITION = 2**31 - 1
# Rollback tokens name their owner by a monotonic identity rather than ``id()``,
# so a token cannot be accepted by a later owner that reuses a freed address.
_IDENTITY = itertools.count(1)


@dataclass(frozen=True)
class Qwen4ExpBatchQsaIndexRollback:
    """Single-use rollback token for one speculative step on one index owner.

    Only the per-slot counters are captured, and only as immutable tuples, so the
    token cannot be mutated into false provenance after it is issued. The
    counters are sufficient because every read is bounded by them: a speculated
    append writes raw position ``cursor`` and a speculated pool writes pooled
    block ``pooled_count``, both strictly beyond the token, and both are
    re-written by the next append at that position.

    This is deliberately *not* a general snapshot. It is valid only for
    append-only rollback within one slot generation on the owner that issued it:
    the owner rejects a token after a slot reset, after the token was already
    used, when the token would move a slot forward, and when the token belongs to
    another owner. Rewinding and then overwriting the rewound positions makes an
    already-used token stale, which single use already covers.
    """

    owner: int
    generation: tuple[int, ...]
    cursors: tuple[int, ...]
    pooled_counts: tuple[int, ...]


@dataclass(frozen=True)
class Qwen4ExpBatchQsaIndexActiveRows:
    """Row-to-slot mapping and per-row selection metadata for one step."""

    rows: int
    slot_ids: tuple[int, ...]
    slots: np.ndarray
    positions: np.ndarray
    blocks: np.ndarray

    def row_of(self, slot: int) -> int:
        matches = np.nonzero(self.slots == int(slot))[0]
        if matches.size != 1:
            raise ValueError(f"slot {slot} is not an active row")
        return int(matches[0])


@dataclass
class Qwen4ExpBatchQsaIndexState:
    """One QSA layer's per-slot index keys, pooled keys, and selection output."""

    attention: Qwen4ExpBatchAttentionState
    raw_keys: DeviceBuffer
    member_indices: DeviceBuffer
    block_starts: DeviceBuffer
    pooled_keys: DeviceBuffer
    scores: DeviceBuffer
    positions: DeviceBuffer
    selected_positions: DeviceBuffer
    selected_counts: DeviceBuffer
    member_host: np.ndarray
    block_starts_host: np.ndarray
    cursor_host: np.ndarray
    pooled_count_host: np.ndarray
    reset_generation: np.ndarray
    selected_positions_host: np.ndarray
    selected_counts_host: np.ndarray
    capacity: int
    complete_blocks: int
    blocks_per_request: int
    block_size: int
    slot_capacity_tokens: int
    state_slots: int
    index_heads: int
    index_dim: int
    compression_ratio: int
    block_budget: int
    runtime: HipRuntime
    identity: int
    active: Qwen4ExpBatchQsaIndexActiveRows | None = None
    pending_rollback: Qwen4ExpBatchQsaIndexRollback | None = None
    closed: bool = False
    closing: bool = False
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    @classmethod
    def allocate(
        cls,
        *,
        attention_state: Qwen4ExpBatchAttentionState,
        index_heads: int,
        index_dim: int,
        compression_ratio: int,
        block_budget: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpBatchQsaIndexState":
        """Allocate per-slot index slabs matching the batch attention owner.

        Every slot gets ``complete_blocks = max_positions // ratio`` pooled-key
        rows, one ``ratio``-wide member map, and a cursor pair. The raw-key slab
        covers the attention owner's *total* physical position range so the
        scatter kernel's global physical index is used unchanged and slot
        windows cannot overlap.
        """

        if attention_state.closed or attention_state.closing:
            raise RuntimeError("QSA index state requires an open attention state")
        active_runtime = runtime or attention_state.runtime
        if active_runtime is not attention_state.runtime:
            raise ValueError("runtime must match the mirrored attention state")
        heads = _positive_int(index_heads, "index_heads")
        dimension = _positive_int(index_dim, "index_dim")
        ratio = _positive_int(compression_ratio, "compression_ratio")
        budget = _positive_int(block_budget, "block_budget")
        capacity = int(attention_state.max_positions)
        slots = int(attention_state.state_slots)
        block_size = int(attention_state.block_size)
        blocks_per_request = int(attention_state.blocks_per_request)
        slot_capacity_tokens = int(attention_state.slot_capacity_tokens)
        # Raw keys are addressed by physical position, so the slab follows the
        # attention owner's physical extent: every slot's window in the static
        # layout, the shared block pool in pooled mode.
        total_positions = int(attention_state.total_capacity_tokens)
        pooled = bool(getattr(attention_state, "pooled", False))
        complete_blocks = capacity // ratio
        if complete_blocks <= 0:
            raise ValueError(
                f"max_positions {capacity} holds no complete {ratio}-token index block"
            )
        if complete_blocks < budget:
            # ``topk_expand_rows_f32_i64`` requires score_stride >= budget, so a
            # narrower index space could not even be scanned by the selection.
            raise ValueError(
                f"max_positions {capacity} gives {complete_blocks} complete index "
                f"blocks, fewer than the selection budget {budget}"
            )
        if total_positions - 1 > _QSA_INDEX_MAX_PHYSICAL_POSITION:
            raise ValueError(
                f"total physical positions {total_positions} exceed the int32 "
                "member-index range"
            )
        _check_slab_products(
            state_slots=slots,
            complete_blocks=complete_blocks,
            ratio=ratio,
            index_dim=dimension,
            total_positions=total_positions,
            output_stride=budget * ratio + ratio - 1,
        )

        # Slot s owns blocks [s*bpr, (s+1)*bpr), so its logical positions map to
        # global physical positions inside a disjoint window. Both the member map
        # and the pooled-key block starts repeat the same logical starts per slot:
        # a compression block is defined by logical positions, not by the page
        # table, so the starts are slot-independent and only the members differ.
        logical_positions = np.arange(complete_blocks * ratio, dtype=np.int64)
        logical_blocks = logical_positions // block_size
        within_block = logical_positions % block_size
        member_host = np.zeros((slots, complete_blocks, ratio), dtype=np.int32)
        if not pooled:
            for slot in range(slots):
                physical = (
                    attention_state.block_host[slot][logical_blocks].astype(np.int64)
                    * block_size
                    + within_block
                )
                member_host[slot] = physical.reshape(complete_blocks, ratio)
            if np.unique(member_host).size != member_host.size:
                raise ValueError("QSA index member maps overlap between slots")
        member_flat = np.ascontiguousarray(
            member_host.reshape(slots * complete_blocks, ratio)
        )
        block_starts_host = np.tile(
            np.arange(complete_blocks, dtype=np.int64) * ratio, slots
        )
        cursor_host = np.zeros(slots, dtype=np.int64)
        pooled_count_host = np.zeros(slots, dtype=np.int64)
        reset_generation = np.zeros(slots, dtype=np.int64)
        output_stride = budget * ratio + ratio - 1
        selected_positions_host = np.empty(
            (slots, output_stride), dtype=np.int64
        )
        selected_counts_host = np.zeros(slots, dtype=np.int32)

        sizes = (
            total_positions * dimension * DType.FP32.itemsize,
            member_flat.nbytes,
            block_starts_host.nbytes,
            slots * complete_blocks * dimension * DType.FP32.itemsize,
            slots * complete_blocks * DType.FP32.itemsize,
            slots * DType.INT64.itemsize,
            selected_positions_host.nbytes,
            selected_counts_host.nbytes,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in sizes:
                buffers.append(malloc(size, runtime=active_runtime))
            copy_host_to_device(
                buffers[1], host_array_ptr(member_flat), runtime=active_runtime
            )
            copy_host_to_device(
                buffers[2], host_array_ptr(block_starts_host), runtime=active_runtime
            )
            copy_host_to_device(
                buffers[7], host_array_ptr(selected_counts_host), runtime=active_runtime
            )
            # A key that was never appended must read as a checkable value.
            active_runtime.memset(buffers[0].ptr, 0, buffers[0].nbytes)
            active_runtime.memset(buffers[3].ptr, 0, buffers[3].nbytes)
        except BaseException:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(
            attention=attention_state,
            raw_keys=buffers[0],
            member_indices=buffers[1],
            block_starts=buffers[2],
            pooled_keys=buffers[3],
            scores=buffers[4],
            positions=buffers[5],
            selected_positions=buffers[6],
            selected_counts=buffers[7],
            member_host=member_host,
            block_starts_host=block_starts_host,
            cursor_host=cursor_host,
            pooled_count_host=pooled_count_host,
            reset_generation=reset_generation,
            selected_positions_host=selected_positions_host,
            selected_counts_host=selected_counts_host,
            capacity=capacity,
            complete_blocks=complete_blocks,
            blocks_per_request=blocks_per_request,
            block_size=block_size,
            slot_capacity_tokens=slot_capacity_tokens,
            state_slots=slots,
            index_heads=heads,
            index_dim=dimension,
            compression_ratio=ratio,
            block_budget=budget,
            runtime=active_runtime,
            identity=next(_IDENTITY),
            _pending_free=list(buffers),
        )

    # -- geometry ---------------------------------------------------------

    @property
    def output_stride(self) -> int:
        """Selected positions written per row (``budget`` blocks plus a tail)."""

        return self.block_budget * self.compression_ratio + self.compression_ratio - 1

    @property
    def dense_equivalent_limit(self) -> int:
        """Largest context the budgeted selection is equivalent to dense."""

        return self.output_stride

    @property
    def active_rows(self) -> int:
        return 0 if self.active is None else self.active.rows

    def raw_position_ptr(self, slot: int, position: int) -> int:
        """Device address of ``slot``'s raw index key at logical ``position``."""

        parsed_slot = self._require_slot(slot)
        parsed_position = _index_int(position, "position")
        if not 0 <= parsed_position < self.slot_capacity_tokens:
            raise ValueError(
                f"position {parsed_position} is outside [0, {self.slot_capacity_tokens})"
            )
        logical_block = parsed_position // self.block_size
        physical = (
            int(self.attention.block_host[parsed_slot][logical_block]) * self.block_size
            + parsed_position % self.block_size
        )
        return self.raw_keys.ptr + physical * self.index_dim * DType.FP32.itemsize

    def pooled_slot_ptr(self, slot: int) -> int:
        """Device address of ``slot``'s pooled-key slab base."""

        parsed = self._require_slot(slot)
        stride = self.complete_blocks * self.index_dim * DType.FP32.itemsize
        return self.pooled_keys.ptr + parsed * stride

    def selected_positions_row_ptr(self, row: int) -> int:
        parsed = self._require_row(row)
        return self.selected_positions.ptr + parsed * self.output_stride * DType.INT64.itemsize

    def selected_counts_row_ptr(self, row: int) -> int:
        parsed = self._require_row(row)
        return self.selected_counts.ptr + parsed * DType.INT32.itemsize

    # -- active rows ------------------------------------------------------

    def prepare(self, slot_ids: Sequence[int]) -> Qwen4ExpBatchQsaIndexActiveRows:
        """Admit active rows and upload their query positions.

        ``slot_ids[r]`` is the physical slot serving row ``r``, so the row order
        is the caller's and every slot appears at most once. A row's query
        position is the last index key appended to its slot; a slot with no
        appended key is admitted here so its first append has a row, and
        ``select_active`` refuses it until a key exists. Validation completes
        before the upload; a failed upload clears the active set instead of
        publishing half a row table.
        """

        self._require_open()
        slots = _index_sequence(slot_ids, "slot_ids")
        if not slots:
            raise ValueError("prepare requires at least one active row")
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
        rows = len(slots)
        parsed_slots = np.asarray(slots, dtype=np.int64)
        positions = self.cursor_host[parsed_slots] - 1
        blocks = self.cursor_host[parsed_slots] // self.compression_ratio
        try:
            copy_host_to_device(
                self.positions,
                host_array_ptr(np.ascontiguousarray(positions)),
                positions.nbytes,
                runtime=self.runtime,
            )
        except BaseException:
            self.active = None
            raise
        active = Qwen4ExpBatchQsaIndexActiveRows(
            rows=rows,
            slot_ids=tuple(slots),
            slots=parsed_slots,
            positions=positions,
            blocks=blocks,
        )
        self.active = active
        return active

    def partition_rows(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Split active rows into ``(dense, sparse)`` row-index groups.

        A row is dense when its complete-block count is within the budget, i.e.
        the sparse selection is equivalent to dense attention. The dense group
        belongs on the dense mixer; the sparse group is what ``select_active``
        accepts.
        """

        active = self._require_active()
        blocks = self.cursor_host[active.slots] // self.compression_ratio
        sparse = tuple(int(row) for row in np.nonzero(blocks > self.block_budget)[0])
        dense = tuple(int(row) for row in np.nonzero(blocks <= self.block_budget)[0])
        return dense, sparse

    def append_active(self, raw_keys_ptr: int, *, stream: int = 0) -> None:
        """Append one raw index key per active row at its own position.

        ``raw_keys_ptr`` is a device ``[rows, index_dim]`` FP32 block. Row ``r``
        is scattered through row ``r``'s page table at row ``r``'s device
        position, so the key lands at exactly the logical position the KV write
        for that row uses. The host cursor mirror advances immediately; a graph
        owner that replays a captured step must use ``begin_rollback`` /
        ``rollback`` rather than calling this twice.

        One dispatch per active row: the committed scatter takes a single page
        table and a single position scalar.
        """

        self._require_open()
        active = self._require_active()
        source = _positive_ptr(raw_keys_ptr, "raw_keys_ptr")
        if active.rows > self.state_slots:
            raise ValueError("active row count exceeds the declared state slab")
        pending: list[int] = []
        for row, slot in enumerate(active.slot_ids):
            cursor = int(self.cursor_host[slot])
            expected = int(self.attention.position_host[row])
            if cursor != expected:
                raise ValueError(
                    f"row {row} slot {slot} index cursor {cursor} does not match the "
                    f"attention position {expected}; append the index key in the same "
                    "step as the KV write"
                )
            if cursor >= self.capacity:
                raise ValueError(
                    f"row {row} slot {slot} index cursor {cursor} reached the "
                    f"per-request capacity {self.capacity}"
                )
            pending.append(slot)
        table_stride = self.blocks_per_request * DType.INT32.itemsize
        dim_bytes = self.index_dim * DType.FP32.itemsize
        for row, slot in enumerate(pending):
            qwen4_exp_qsa_scatter_index_key_device_position_f32(
                source + row * dim_bytes,
                self.raw_keys.ptr,
                self.attention.active.block_table.ptr + row * table_stride,
                self.attention.active.position.ptr + row * DType.INT64.itemsize,
                self.block_size,
                self.blocks_per_request,
                self.index_dim,
                stream=stream,
                runtime=self.runtime,
            )
            self.cursor_host[slot] += 1

    # -- selection --------------------------------------------------------

    def select_active(
        self,
        prepared_queries_ptr: int,
        *,
        key_norm_weight_ptr: int,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
        stream: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score and select for every active row, then return host metadata.

        ``prepared_queries_ptr`` is a device ``[rows, index_heads, index_dim]``
        FP32 block already normalized and rotated by the caller (the same
        contract as the c1 owner's ``prepared_query_ptr``).

        Returns ``(counts, positions)``: ``counts`` is ``int32[rows]`` and
        ``positions`` is ``int64[rows, output_stride]``; entry ``[r, :counts[r]]``
        is row ``r``'s selection in the order the strict online-softmax consumer
        expects, matching the c1 owner exactly. The device buffers stay valid for
        the sparse attention kernel via ``selected_positions_row_ptr``.

        A row whose complete-block count is within the budget is rejected: the
        sparse path is only defined past the dense equivalence bound, and
        ``partition_rows`` is how a caller separates the two groups.
        """

        self._require_open()
        active = self._require_active()
        queries = _positive_ptr(prepared_queries_ptr, "prepared_queries_ptr")
        norm = _positive_ptr(key_norm_weight_ptr, "key_norm_weight_ptr")
        rotate = _positive_int(rotary_dim, "rotary_dim")
        if rotate > self.index_dim:
            raise ValueError(
                f"rotary_dim {rotate} exceeds index_dim {self.index_dim}"
            )
        if rotate % 2:
            raise ValueError(f"rotary_dim {rotate} must be even")
        theta_value = _finite_positive(theta, "theta")
        eps_value = _finite_positive(eps, "eps")
        slots = active.slots
        cursors = self.cursor_host[slots]
        empty = np.nonzero(cursors <= 0)[0]
        if empty.size:
            first = int(empty[0])
            raise ValueError(
                f"row {first} slot {int(slots[first])} has no appended index key, so "
                "it has no query position to select from"
            )
        positions = cursors - 1
        if not np.array_equal(positions, active.positions):
            raise ValueError(
                "active query positions do not match the published row metadata; "
                "re-run prepare after appending index keys"
            )
        blocks = cursors // self.compression_ratio
        if not np.array_equal(blocks, active.blocks):
            raise ValueError(
                "active complete-block counts do not match the published row metadata"
            )
        shallow = np.nonzero(blocks <= self.block_budget)[0]
        if shallow.size:
            first = int(shallow[0])
            slot = int(slots[first])
            raise ValueError(
                f"row {first} slot {slot} has {int(blocks[first])} complete index "
                f"blocks, not more than the budget {self.block_budget}; dense "
                "attention covers this context and the sparse path is undefined"
            )

        # Per-slot pool of newly completed blocks (one dispatch per slot with work).
        for slot in slots.tolist():
            wanted = int(self.cursor_host[slot]) // self.compression_ratio
            first = int(self.pooled_count_host[slot])
            if wanted <= first:
                continue
            qwen4_exp_qsa_pool_norm_rope_f32(
                self.raw_keys.ptr,
                self.member_indices.ptr
                + (slot * self.complete_blocks + first)
                * self.compression_ratio
                * DType.INT32.itemsize,
                self.block_starts.ptr
                + (slot * self.complete_blocks + first) * DType.INT64.itemsize,
                norm,
                self.pooled_keys.ptr
                + (slot * self.complete_blocks + first)
                * self.index_dim
                * DType.FP32.itemsize,
                wanted - first,
                self.compression_ratio,
                self.index_dim,
                rotate,
                theta_value,
                eps_value,
                stream=stream,
                runtime=self.runtime,
            )
            self.pooled_count_host[slot] = wanted

        # Per-row score against that row's own slot keys (one dispatch per row).
        query_stride = self.index_heads * self.index_dim * DType.FP32.itemsize
        pooled_stride = self.complete_blocks * self.index_dim * DType.FP32.itemsize
        score_stride = self.complete_blocks * DType.FP32.itemsize
        for row, slot in enumerate(slots.tolist()):
            qwen4_exp_qsa_score_f32(
                queries + row * query_stride,
                self.pooled_keys.ptr + slot * pooled_stride,
                self.scores.ptr + row * score_stride,
                1,
                int(blocks[row]),
                self.index_heads,
                self.index_dim,
                stream=stream,
                runtime=self.runtime,
            )

        # One batched selection over every row.
        qwen4_exp_qsa_topk_expand_rows_f32_i64(
            self.scores.ptr,
            self.positions.ptr,
            self.selected_positions.ptr,
            self.selected_counts.ptr,
            active.rows,
            self.complete_blocks,
            self.output_stride,
            self.compression_ratio,
            self.block_budget,
            stream=stream,
            runtime=self.runtime,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        rows = active.rows
        copy_device_to_host(
            host_array_ptr(self.selected_counts_host[:rows]),
            self.selected_counts,
            rows * DType.INT32.itemsize,
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(self.selected_positions_host[:rows]),
            self.selected_positions,
            rows * self.output_stride * DType.INT64.itemsize,
            runtime=self.runtime,
        )
        counts = self.selected_counts_host[:rows].copy()
        widest = int(counts.max())
        return counts, self.selected_positions_host[:rows, :widest].copy()

    def begin_rollback(self) -> Qwen4ExpBatchQsaIndexRollback:
        """Issue the single rollback token for the next speculative step.

        One token exists per owner at a time. It is valid for append-only
        rollback within the current slot generation, and it is consumed by
        ``rollback``: after a rewind and an overwrite of the rewound positions,
        the same token can no longer be re-applied. A slot reset invalidates it
        instead of silently making it wrong.
        """

        self._require_open()
        if self.pending_rollback is not None:
            raise RuntimeError(
                "a rollback token is already pending; roll it back, discard it, or "
                "reset the affected slot before capturing another"
            )
        token = Qwen4ExpBatchQsaIndexRollback(
            owner=self.identity,
            generation=tuple(int(value) for value in self.reset_generation),
            cursors=tuple(int(value) for value in self.cursor_host),
            pooled_counts=tuple(int(value) for value in self.pooled_count_host),
        )
        self.pending_rollback = token
        return token

    def rollback(self, token: Qwen4ExpBatchQsaIndexRollback) -> None:
        """Apply ``token``, rewinding the counters and dropping the active rows.

        Device contents are not copied: every read is bounded by the counters,
        and a speculated append or pool writes strictly beyond them. Every
        rejection happens before any assignment, so a refused rollback leaves
        the counters, the slot generations, the pending token, and the active row
        set exactly as they were.
        """

        self._require_open()
        if not isinstance(token, Qwen4ExpBatchQsaIndexRollback):
            raise TypeError("rollback requires a Qwen4ExpBatchQsaIndexRollback")
        if token.owner != self.identity:
            raise ValueError(
                "rollback token belongs to a different index state; it was issued by "
                f"owner {token.owner} but presented to owner {self.identity}"
            )
        if token.generation != tuple(int(value) for value in self.reset_generation):
            reset = [
                slot
                for slot, (before, after) in enumerate(
                    zip(token.generation, self.reset_generation.tolist())
                )
                if before != after
            ]
            raise ValueError(
                f"slot(s) {reset} were reset after this rollback token was issued; "
                "their counters now describe a different request, so the token is "
                "stale and cannot rewind them"
            )
        # Materialize and validate both counter arrays before any assignment: a
        # wrong-length, non-integer, or truncating tuple must be refused before
        # the first slice write, or a half-applied rollback would corrupt the
        # state it claims to reject. ``type(...) is int`` is deliberate: bool and
        # float would otherwise slip through numpy's silent conversion.
        counters = token.cursors + token.pooled_counts
        if not all(type(value) is int for value in counters):
            raise ValueError("rollback token counters must be integer tuples")
        if (
            len(token.cursors) != self.state_slots
            or len(token.pooled_counts) != self.state_slots
        ):
            raise ValueError(
                f"rollback token must carry {self.state_slots} cursors and pooled "
                f"counts, got {len(token.cursors)} and {len(token.pooled_counts)}"
            )
        cursors = np.asarray(token.cursors, dtype=np.int64)
        pooled = np.asarray(token.pooled_counts, dtype=np.int64)
        forward = [
            slot
            for slot, cursor in enumerate(token.cursors)
            if cursor > int(self.cursor_host[slot])
        ]
        if forward:
            raise ValueError(
                f"rollback token would move slot(s) {forward} forward, past the "
                "current cursor; a rewind can only go backwards"
            )
        if token is not self.pending_rollback:
            raise RuntimeError(
                "rollback token is not the pending one; it was already used, "
                "discarded, or invalidated"
            )
        self.cursor_host[:] = cursors
        self.pooled_count_host[:] = pooled
        self.active = None
        self.pending_rollback = None

    def discard_rollback(self) -> bool:
        """Abandon the pending token without rewinding; returns whether one existed."""

        self._require_open()
        existed = self.pending_rollback is not None
        self.pending_rollback = None
        return existed

    def map_slot(self, slot: int, blocks: int) -> None:
        """Publish ``slot``'s member map for its first ``blocks`` paged blocks.

        Pooled mode assigns physical blocks at admission, so the member map that
        turns a slot's logical index-block members into physical raw-key rows is
        rebuilt from the attention owner's page table and uploaded for the
        covered prefix. Rows past it are never read: a slot's cursor cannot pass
        the positions its blocks hold.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        count = _index_int(blocks, "blocks")
        if not 0 <= count <= self.blocks_per_request:
            raise ValueError(
                f"blocks {count} is outside [0, {self.blocks_per_request}]"
            )
        complete = min(self.complete_blocks, count * self.block_size // self.compression_ratio)
        if complete <= 0:
            return
        logical = np.arange(complete * self.compression_ratio, dtype=np.int64)
        table = self.attention.block_host[parsed]
        physical = (
            table[logical // self.block_size].astype(np.int64) * self.block_size
            + logical % self.block_size
        )
        rows = physical.reshape(complete, self.compression_ratio).astype(np.int32)
        self.member_host[parsed, :complete] = rows
        row_bytes = self.compression_ratio * DType.INT32.itemsize
        offset = parsed * self.complete_blocks * row_bytes
        self.runtime.memcpy(
            self.member_indices.ptr + offset,
            host_array_ptr(np.ascontiguousarray(self.member_host[parsed, :complete])),
            complete * row_bytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )

    def reset_slot(self, slot: int) -> None:
        """Release one slot for a new request (its keys become unreadable).

        This advances the slot's generation, which permanently invalidates any
        pending rollback token: a token captured before the reset describes
        counters of the previous request, and applying it would re-expose that
        request's keys to the new one.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        self.cursor_host[parsed] = 0
        self.pooled_count_host[parsed] = 0
        self.reset_generation[parsed] += 1
        if self.active is not None and parsed in self.active.slot_ids:
            self.active = None

    def close(self) -> None:
        if self.closed:
            return
        # Poison uploads before the first free: a partial close leaves some
        # buffers released and some owned, and no upload may target either.
        self.closing = True
        self.active = None
        self.pending_rollback = None
        pending = self._pending_free
        if not pending:
            pending.extend(
                (
                    self.raw_keys,
                    self.member_indices,
                    self.block_starts,
                    self.pooled_keys,
                    self.scores,
                    self.positions,
                    self.selected_positions,
                    self.selected_counts,
                )
            )
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchQsaIndexState":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp batch QSA index state is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp batch QSA index state is closing; its buffers may already "
                "be released"
            )
        if self.attention.closed or self.attention.closing:
            raise RuntimeError(
                "Qwen4Exp batch QSA index state mirrors a closing attention state"
            )

    def _require_active(self) -> Qwen4ExpBatchQsaIndexActiveRows:
        self._require_open()
        if self.active is None:
            raise RuntimeError(
                "Qwen4Exp batch QSA index state has no active rows; call prepare first"
            )
        return self.active

    def _require_slot(self, slot: object) -> int:
        parsed = _index_int(slot, "slot")
        if not 0 <= parsed < self.state_slots:
            raise ValueError(f"slot {parsed} is outside [0, {self.state_slots})")
        return parsed

    def _require_row(self, row: object) -> int:
        parsed = _index_int(row, "row")
        active = self._require_active()
        if not 0 <= parsed < active.rows:
            raise ValueError(f"row {parsed} is outside [0, {active.rows})")
        return parsed


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first, popping only after each free returns."""

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _check_slab_products(
    *,
    state_slots: int,
    complete_blocks: int,
    ratio: int,
    index_dim: int,
    total_positions: int,
    output_stride: int,
) -> None:
    """Reject geometry whose byte products or int64 offsets cannot be addressed."""

    products = {
        "raw index keys": total_positions * index_dim * DType.FP32.itemsize,
        "pooled index keys": (
            state_slots * complete_blocks * index_dim * DType.FP32.itemsize
        ),
        "index member map": state_slots * complete_blocks * ratio * DType.INT32.itemsize,
        "index block starts": state_slots * complete_blocks * DType.INT64.itemsize,
        "index scores": state_slots * complete_blocks * DType.FP32.itemsize,
        "index selected positions": state_slots * output_stride * DType.INT64.itemsize,
    }
    for name, nbytes in products.items():
        if nbytes <= 0 or nbytes > np.iinfo(np.int64).max:
            raise ValueError(f"{name} byte size {nbytes} is not addressable")


def _index_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _positive_int(value: object, name: str) -> int:
    parsed = _index_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_ptr(value: object, name: str) -> int:
    parsed = _index_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive device pointer")
    return parsed


def _finite_positive(value: object, name: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a float") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _index_sequence(values: object, name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        items = list(values)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None
    return [_index_int(item, name) for item in items]


def batch_qsa_index_call_table() -> dict[str, Callable[..., Any]]:
    """The exact direct callables this module uses for index maintenance.

    Append, pooled-key normalization, scoring, and selection are called directly
    here; their matching registry keys are recorded as fallback identities, not
    as evidence that this module resolved them through the registry.
    """

    return {
        "qsa_index_append": qwen4_exp_qsa_scatter_index_key_device_position_f32,
        "qsa_index_pool_norm_rope": qwen4_exp_qsa_pool_norm_rope_f32,
        "qsa_index_score": qwen4_exp_qsa_score_f32,
        "qsa_index_select": qwen4_exp_qsa_topk_expand_rows_f32_i64,
    }
