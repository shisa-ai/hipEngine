"""Ownership/layout tests for the per-slot Qwen4Exp batch attention state."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_state as batch_state_module
from hipengine.runtime.qwen4_exp_batch_state import (
    STEP_CONTEXT_COLUMN,
    STEP_INDEX_POSITION_COLUMN,
    STEP_METADATA_COLUMNS,
    STEP_POSITION_COLUMN,
    Qwen4ExpBatchActiveRows,
    Qwen4ExpBatchAttentionRowPlan,
    Qwen4ExpBatchAttentionState,
    Qwen4ExpBatchStepMetadata,
)

SLOTS = 4
MAX_POSITIONS = 1280
BLOCK_SIZE = 256
KV_HEADS = 2
HEAD_DIM = 64


class FakeRuntime:
    """Records allocation, memset, and copy traffic without touching a device."""

    def __init__(
        self, *, fail_malloc_call: int | None = None, fail_free_call: int | None = None
    ) -> None:
        self.next_ptr = 0xA000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.memsets: list[tuple[int, int, int]] = []
        self.copies: list[tuple[int, bytes, int]] = []
        self.fail_malloc_call = fail_malloc_call
        self.fail_free_call = fail_free_call
        self.malloc_calls = 0
        self.free_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_call == self.malloc_calls:
            raise MemoryError("simulated device allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.free_calls += 1
        if self.fail_free_call == self.free_calls:
            raise RuntimeError("simulated free failure")
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations
        assert nbytes <= self.allocations[dst]
        self.memsets.append((dst, value, nbytes))

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        assert dst in self.allocations
        assert nbytes <= self.allocations[dst]
        self.copies.append((dst, ctypes.string_at(src, nbytes), nbytes))

    # Copy traffic grouped per destination buffer keeps assertions readable.
    def copies_to(self, buffer: object) -> list[bytes]:
        ptr = getattr(buffer, "ptr", buffer)
        return [payload for dest, payload, _ in self.copies if dest == ptr]


def _allocate(**overrides) -> tuple[Qwen4ExpBatchAttentionState, FakeRuntime]:
    runtime = overrides.pop("runtime", None) or FakeRuntime()
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
        **overrides,
    )
    return state, runtime


def test_allocate_gives_every_slot_a_disjoint_full_capacity_range() -> None:
    state, _ = _allocate()
    try:
        assert state.blocks_per_request == 5
        assert state.block_host.shape == (SLOTS, 5)
        # Slot s owns a contiguous disjoint range, so capacity is additive.
        assert state.block_host.tolist() == [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ]
        assert len({int(block) for block in state.block_host.ravel()}) == SLOTS * 5
        assert state.slot_capacity_tokens == 1280 == MAX_POSITIONS
        assert state.total_capacity_tokens == SLOTS * MAX_POSITIONS
        token_bytes = KV_HEADS * HEAD_DIM * 2
        assert state.cache_bytes_per_slot == 5 * BLOCK_SIZE * token_bytes
        assert state.key_cache.nbytes == SLOTS * 5 * BLOCK_SIZE * token_bytes
        assert state.value_cache.nbytes == state.key_cache.nbytes
        # The page tables, positions, and contexts are sized for the physical
        # slot capacity, not for the active width.
        assert state.block_table.nbytes == SLOTS * 5 * 4
        assert state.position.nbytes == SLOTS * 8
        assert state.context.nbytes == SLOTS * 8
        assert state.active is None
        assert state.active_rows == 0
    finally:
        state.close()


def test_allocate_zeroes_metadata_and_kv() -> None:
    state, runtime = _allocate()
    try:
        assert runtime.copies_to(state.block_table) == [bytes(SLOTS * 5 * 4)]
        assert runtime.copies_to(state.position) == [bytes(SLOTS * 8)]
        assert runtime.copies_to(state.context) == [bytes(SLOTS * 8)]
        assert (state.key_cache.ptr, 0, state.key_cache.nbytes) in runtime.memsets
        assert (state.value_cache.ptr, 0, state.value_cache.nbytes) in runtime.memsets
    finally:
        state.close()


def test_prepare_uploads_only_the_packed_metadata_rows() -> None:
    state, runtime = _allocate()
    try:
        active = state.prepare([2, 0], [255, 3])
        assert isinstance(active, Qwen4ExpBatchActiveRows)
        assert active.rows == 2 == state.active_rows
        assert active.slot_ids == (2, 0)
        assert active.max_context_len == 256
        # Row r carries slot_ids[r]'s own page table, so no row can address
        # another slot's blocks.
        uploaded = np.frombuffer(runtime.copies_to(state.block_table)[-1], dtype=np.int32)
        assert uploaded.tolist() == [10, 11, 12, 13, 14, 0, 1, 2, 3, 4]
        assert np.frombuffer(
            runtime.copies_to(state.position)[-1], dtype=np.int64
        ).tolist() == [255, 3]
        assert np.frombuffer(
            runtime.copies_to(state.context)[-1], dtype=np.int64
        ).tolist() == [256, 4]
        # Only the packed metadata rows are written; the inactive slot rows keep
        # the zero tables published by allocate.
        assert len(runtime.copies_to(state.block_table)[-1]) == 2 * 5 * 4
        assert state.block_table_host[:2].tolist() == [
            [10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4],
        ]
        assert state.block_table_host[2:].tolist() == [[0] * 5, [0] * 5]
        assert active.block_table.shape == (2, 5)
        assert active.position.numel == 2
        assert active.context.numel == 2
        assert active.append_spans.base_offsets.ptr == active.block_table.ptr
        assert active.append_spans.live_counts.ptr == active.position.ptr
        assert active.decode_spans.live_counts.ptr == active.context.ptr
        assert active.append_spans.max_live_count == MAX_POSITIONS - 1
        assert active.decode_spans.max_live_count == MAX_POSITIONS
    finally:
        state.close()


def test_permuted_slot_order_keeps_row_to_slot_mapping() -> None:
    state, _ = _allocate()
    try:
        active = state.prepare([3, 1, 0], [700, 255, 0])
        assert active.slot_ids == (3, 1, 0)
        assert state.block_table_host[:3].tolist() == [
            [15, 16, 17, 18, 19],
            [5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4],
        ]
        assert active.max_context_len == 701
        assert state.blocks_for_slot(3).tolist() == [15, 16, 17, 18, 19]
        with pytest.raises(ValueError):
            state.blocks_for_slot(SLOTS)
    finally:
        state.close()


def test_active_width_shrinks_and_regrows_without_reallocation() -> None:
    state, runtime = _allocate()
    try:
        allocations = len(runtime.allocations)
        widths = [4, 1, 2, 4, 3]
        for width in widths:
            active = state.prepare(list(range(width)), [10] * width)
            assert active.rows == width == state.active_rows
            assert active.block_table.shape == (width, 5)
            assert state.block_table_host[:width].tolist() == state.block_host[
                :width
            ].tolist()
        # No reallocation across the whole width sweep, and the physical
        # capacity never shrank with the active width.
        assert len(runtime.allocations) == allocations
        assert state.state_slots == SLOTS
        assert state.slot_capacity_tokens == MAX_POSITIONS
        assert state.key_cache.nbytes == SLOTS * 5 * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2
    finally:
        state.close()


def test_prepare_rejects_duplicate_slots_without_touching_state() -> None:
    state, runtime = _allocate()
    try:
        state.prepare([0, 1], [4, 5])
        copies_before = len(runtime.copies)
        with pytest.raises(ValueError, match="appears twice"):
            state.prepare([0, 0], [4, 5])
        assert len(runtime.copies) == copies_before
        assert state.active_rows == 2
        assert state.active is not None
        assert state.active.slot_ids == (0, 1)
        assert state.position_host[:2].tolist() == [4, 5]
    finally:
        state.close()


@pytest.mark.parametrize(
    ("slots", "positions", "match"),
    [
        ([SLOTS], [0], "outside"),
        ([-1], [0], "outside"),
        ([0], [MAX_POSITIONS], "outside"),
        ([0], [-1], "outside"),
        ([0], [], "equal length"),
        ([], [], "at least one active row"),
        ([0, 1, 1], [0, 0, 0], "appears twice"),
        (list(range(SLOTS + 1)), [0] * (SLOTS + 1), "at most"),
        ([0], [0, 1], "equal length"),
    ],
)
def test_prepare_rejects_invalid_metadata_before_any_device_write(
    slots: list[int], positions: list[int], match: str
) -> None:
    state, runtime = _allocate()
    try:
        state.prepare([0], [0])
        copies_before = len(runtime.copies)
        memsets_before = len(runtime.memsets)
        with pytest.raises(ValueError, match=match):
            state.prepare(slots, positions)
        assert len(runtime.copies) == copies_before
        assert len(runtime.memsets) == memsets_before
        # The previously prepared row set is still the published one.
        assert state.active_rows == 1
        assert state.active is not None
        assert state.active.slot_ids == (0,)
    finally:
        state.close()


@pytest.mark.parametrize("bad", [1.5, True, "0", None])
def test_prepare_rejects_non_integer_metadata(bad: object) -> None:
    state, _ = _allocate()
    try:
        with pytest.raises(TypeError):
            state.prepare([bad], [0])
        with pytest.raises(TypeError):
            state.prepare([0], [bad])
        assert state.active is None
    finally:
        state.close()


def test_failed_upload_clears_active_rows_instead_of_publishing_a_partial_set() -> None:
    state, runtime = _allocate()
    try:
        state.prepare([0, 1], [4, 5])
        real_memcpy = runtime.memcpy
        calls = {"n": 0}

        def failing_memcpy(dst: int, src: int, nbytes: int, kind) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated copy failure")
            real_memcpy(dst, src, nbytes, kind)

        runtime.memcpy = failing_memcpy
        with pytest.raises(RuntimeError, match="simulated copy failure"):
            state.prepare([2, 3], [7, 8])
        assert state.active is None
        assert state.active_rows == 0
        with pytest.raises(RuntimeError, match="no active rows"):
            state.decode_spans
        # Recovering is a normal prepare; nothing was left half-published.
        runtime.memcpy = real_memcpy
        active = state.prepare([2, 3], [7, 8])
        assert active.rows == 2
        assert state.position_host[:2].tolist() == [7, 8]
    finally:
        state.close()


def test_allocate_failure_frees_every_partial_allocation() -> None:
    for fail_call in range(1, 6):
        runtime = FakeRuntime(fail_malloc_call=fail_call)
        with pytest.raises(MemoryError):
            _allocate(runtime=runtime)
        assert runtime.allocations == {}
        # Every allocation made before the failure is released, newest first.
        assert len(runtime.freed) == fail_call - 1
        assert runtime.freed == sorted(runtime.freed, reverse=True)
        assert runtime.memsets == []
        assert runtime.copies == []


def test_close_is_idempotent_and_frees_every_buffer() -> None:
    state, runtime = _allocate()
    pointers = [
        state.key_cache.ptr,
        state.value_cache.ptr,
        state.block_table.ptr,
        state.position.ptr,
        state.context.ptr,
    ]
    state.prepare([0], [0])
    state.close()
    assert sorted(runtime.freed) == sorted(pointers)
    assert runtime.allocations == {}
    state.close()
    assert sorted(runtime.freed) == sorted(pointers)
    with pytest.raises(RuntimeError, match="closed"):
        state.prepare([0], [0])
    with pytest.raises(RuntimeError, match="closed"):
        state.append_spans


def test_context_manager_closes_on_exit() -> None:
    runtime = FakeRuntime()
    with _allocate(runtime=runtime)[0] as state:
        state.prepare([1], [2])
        assert state.active_rows == 1
    assert runtime.allocations == {}
    assert state.closed is True


def test_prepare_is_deterministic_for_identical_metadata() -> None:
    state, runtime = _allocate()
    try:
        first = state.prepare([1, 3], [300, 4])
        first_payloads = runtime.copies_to(state.block_table)[-1]
        second = state.prepare([1, 3], [300, 4])
        assert runtime.copies_to(state.block_table)[-1] == first_payloads
        assert first.slot_ids == second.slot_ids
        assert first.max_context_len == second.max_context_len == 301
        assert first.append_spans.live_counts.ptr == second.append_spans.live_counts.ptr
    finally:
        state.close()


def test_spans_require_an_active_row_set() -> None:
    state, _ = _allocate()
    try:
        with pytest.raises(RuntimeError, match="no active rows"):
            state.append_spans
        with pytest.raises(RuntimeError, match="no active rows"):
            state.decode_spans
    finally:
        state.close()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"state_slots": 0}, ValueError),
        ({"max_positions": 0}, ValueError),
        ({"block_size": 0}, ValueError),
        ({"kv_heads": 0}, ValueError),
        ({"head_dim": -1}, ValueError),
        ({"state_slots": 1.5}, TypeError),
        ({"state_slots": True}, TypeError),
        ({"max_positions": False}, TypeError),
        ({"block_size": True}, TypeError),
        ({"kv_heads": True}, TypeError),
        ({"head_dim": False}, TypeError),
    ],
)
def test_allocate_rejects_invalid_geometry(kwargs: dict, error: type) -> None:
    geometry = {
        "state_slots": SLOTS,
        "max_positions": MAX_POSITIONS,
        "block_size": BLOCK_SIZE,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
    }
    geometry.update(kwargs)
    with pytest.raises(error):
        Qwen4ExpBatchAttentionState.allocate(runtime=FakeRuntime(), **geometry)


def test_partial_trailing_block_still_gives_full_request_capacity() -> None:
    runtime = FakeRuntime()
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=2,
        max_positions=300,
        block_size=BLOCK_SIZE,
        kv_heads=1,
        head_dim=8,
        runtime=runtime,
    )
    try:
        # 300 positions round up to 2 blocks, so the paged capacity exceeds the
        # request capacity instead of being clamped to it.
        assert state.blocks_per_request == 2
        assert state.slot_capacity_tokens == 512 > state.max_positions
        active = state.prepare([1], [299])
        assert active.max_context_len == 300
        with pytest.raises(ValueError, match="outside"):
            state.prepare([1], [300])
    finally:
        state.close()


_HUGE_GEOMETRY = [
    # Physical block ids are int32 page-table entries.
    (
        dict(state_slots=(1 << 31) + 1, max_positions=1, block_size=1, kv_heads=1, head_dim=1),
        "int32 page-table",
    ),
    # Positions and context lengths are int64 live_counts entries.
    (
        dict(state_slots=1, max_positions=(1 << 63) + 1, block_size=1, kv_heads=1, head_dim=1),
        "int64 position",
    ),
    # K/V element offsets are int64 products of block, block_size, and head size.
    (
        dict(
            state_slots=4,
            max_positions=1 << 20,
            block_size=256,
            kv_heads=1 << 22,
            head_dim=1 << 22,
        ),
        "int64 element",
    ),
    # The byte count handed to the allocator is bounded the same way.
    (
        dict(state_slots=1, max_positions=1, block_size=1, kv_heads=1 << 31, head_dim=1 << 31),
        "int64/size_t",
    ),
]


def test_huge_geometry_is_rejected_before_runtime_resolution_or_numpy(monkeypatch) -> None:
    def _fail_runtime_resolution():
        raise AssertionError("allocate resolved a runtime before validating the shape")

    def _fail_allocation(*args, **kwargs):
        raise AssertionError("allocate built a host array before validating the shape")

    monkeypatch.setattr(batch_state_module, "get_hip_runtime", _fail_runtime_resolution)
    monkeypatch.setattr(batch_state_module.np, "arange", _fail_allocation)
    monkeypatch.setattr(batch_state_module.np, "zeros", _fail_allocation)
    for geometry, match in _HUGE_GEOMETRY:
        # No runtime argument: rejection has to happen before resolution.
        with pytest.raises(ValueError, match=match):
            Qwen4ExpBatchAttentionState.allocate(**geometry)


def test_huge_geometry_takes_no_allocation_or_copy_side_effect() -> None:
    for geometry, match in _HUGE_GEOMETRY:
        runtime = FakeRuntime()
        with pytest.raises(ValueError, match=match):
            Qwen4ExpBatchAttentionState.allocate(runtime=runtime, **geometry)
        assert runtime.malloc_calls == 0
        assert runtime.allocations == {}
        assert runtime.copies == []
        assert runtime.memsets == []


def test_close_retries_after_a_free_failure_without_double_free() -> None:
    runtime = FakeRuntime(fail_free_call=3)
    state, _ = _allocate(runtime=runtime)
    pointers = [
        state.key_cache.ptr,
        state.value_cache.ptr,
        state.block_table.ptr,
        state.position.ptr,
        state.context.ptr,
    ]
    with pytest.raises(RuntimeError, match="simulated free failure"):
        state.close()
    # The failure must not mark the state closed, and the buffers released before
    # it must leave the owned set so the retry cannot free them again.
    assert state.closed is False
    assert len(runtime.freed) == 2
    assert set(runtime.allocations) == set(pointers) - set(runtime.freed)
    state.close()
    assert state.closed is True
    assert runtime.allocations == {}
    assert sorted(runtime.freed) == sorted(pointers)
    assert len(runtime.freed) == len(set(runtime.freed))


def test_allocate_cleanup_failure_is_reported_without_masking_the_cause() -> None:
    runtime = FakeRuntime(fail_malloc_call=4, fail_free_call=1)
    with pytest.raises(MemoryError) as info:
        _allocate(runtime=runtime)
    assert any("cleanup" in note for note in getattr(info.value, "__notes__", []))
    # The three buffers taken before the failure are still tracked, not dropped.
    assert runtime.freed == []
    assert len(runtime.allocations) == 3


def test_close_reclaims_buffers_when_the_ownership_list_was_not_handed_over() -> None:
    state, runtime = _allocate()
    pointers = [
        state.key_cache.ptr,
        state.value_cache.ptr,
        state.block_table.ptr,
        state.position.ptr,
        state.context.ptr,
    ]
    # Simulate a construction path that did not pass ``_pending_free``: closing
    # must still release everything instead of silently leaking.
    state._pending_free.clear()
    state.close()
    assert state.closed is True
    assert runtime.allocations == {}
    assert sorted(runtime.freed) == sorted(pointers)


def test_partial_close_rejects_uploads_and_entry_before_any_runtime_operation() -> None:
    runtime = FakeRuntime(fail_free_call=3)
    state, _ = _allocate(runtime=runtime)
    pointers = [
        state.key_cache.ptr,
        state.value_cache.ptr,
        state.block_table.ptr,
        state.position.ptr,
        state.context.ptr,
    ]
    state.prepare([0, 1], [4, 5])
    # Frees run newest first, so the two buffers a prepare uploads into (context
    # and position) are already released when the third free fails.
    with pytest.raises(RuntimeError, match="simulated free failure"):
        state.close()
    assert state.closed is False
    assert state.closing is True
    released = list(runtime.freed)
    assert released == [state.context.ptr, state.position.ptr]
    copies_before = len(runtime.copies)
    memsets_before = len(runtime.memsets)
    mallocs_before = runtime.malloc_calls

    # Every upload path must refuse before touching a released buffer.
    with pytest.raises(RuntimeError, match="closing"):
        state.prepare([0, 1], [4, 5])
    with pytest.raises(RuntimeError, match="closing"):
        state.append_spans
    with pytest.raises(RuntimeError, match="closing"):
        state.decode_spans
    with pytest.raises(RuntimeError, match="closing"):
        with state:
            pass

    assert len(runtime.copies) == copies_before
    assert len(runtime.memsets) == memsets_before
    assert runtime.malloc_calls == mallocs_before
    assert runtime.freed == released

    # Retrying releases exactly the remainder, once each, and stays idempotent.
    state.close()
    assert state.closed is True
    state.close()
    assert runtime.allocations == {}
    assert sorted(runtime.freed) == sorted(pointers)
    assert len(runtime.freed) == len(set(runtime.freed))


# -- packed per-step metadata ------------------------------------------------
#
# The batch runner publishes every QSA owner's dynamic per-step metadata through
# one buffer and one upload. These tests pin the parts a device test cannot
# isolate cheaply: the layout, the stability of the view addresses a captured
# graph depends on, the single copy per step, and the fact that a rejected row
# set publishes nothing.

LAYERS = 2


def _allocate_packed(**overrides) -> tuple[Qwen4ExpBatchStepMetadata, FakeRuntime]:
    runtime = overrides.pop("runtime", None) or FakeRuntime()
    metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=overrides.pop("layers", LAYERS),
        state_slots=overrides.pop("state_slots", SLOTS),
        runtime=runtime,
        **overrides,
    )
    return metadata, runtime


def test_packed_metadata_layout_is_layer_major_with_a_slot_row_stride() -> None:
    metadata, runtime = _allocate_packed()
    try:
        stride = SLOTS * 8
        assert metadata.bytes_per_column == stride
        assert metadata.buffer.nbytes == LAYERS * STEP_METADATA_COLUMNS * stride
        for layer in range(LAYERS):
            views = metadata.views(layer)
            assert views.layer == layer
            assert views.position.ptr == (
                metadata.buffer.ptr
                + (layer * STEP_METADATA_COLUMNS + STEP_POSITION_COLUMN) * stride
            )
            assert views.context.ptr == views.position.ptr + stride
            assert views.index_position.ptr == views.context.ptr + stride
            assert views.position.nbytes == views.context.nbytes == stride
        # Two layers never alias one another's metadata.
        assert metadata.views(0).position.ptr != metadata.views(1).position.ptr
        assert runtime.copies == []
    finally:
        metadata.close()


def test_packed_metadata_views_keep_their_addresses_across_widths() -> None:
    metadata, runtime = _allocate_packed()
    try:
        before = {
            (layer, column): ptr
            for layer in range(LAYERS)
            for column, ptr in (
                ("position", metadata.views(layer).position.ptr),
                ("context", metadata.views(layer).context.ptr),
                ("index", metadata.views(layer).index_position.ptr),
            )
        }
        for width in (1, SLOTS, 2, SLOTS):
            for layer in range(LAYERS):
                metadata.stage(
                    layer,
                    positions=[width] * width,
                    contexts=[width + 1] * width,
                    index_positions=[width - 1] * width,
                )
            metadata.upload()
            assert metadata.views(layer) is metadata.views(layer)
            after = {
                (layer, column): ptr
                for layer in range(LAYERS)
                for column, ptr in (
                    ("position", metadata.views(layer).position.ptr),
                    ("context", metadata.views(layer).context.ptr),
                    ("index", metadata.views(layer).index_position.ptr),
                )
            }
            assert after == before, "a width change moved a published address"
        # Every upload is the whole buffer, so the copy size never depends on the
        # active width either.
        assert {nbytes for _, _, nbytes in runtime.copies} == {metadata.buffer.nbytes}
        assert len(runtime.copies) == 4
    finally:
        metadata.close()


def test_packed_metadata_upload_publishes_every_staged_layer_in_one_copy() -> None:
    metadata, runtime = _allocate_packed()
    try:
        metadata.stage(
            0,
            positions=[4, 9],
            contexts=[5, 10],
            index_positions=[3, 8],
        )
        metadata.stage(
            1,
            positions=[11, 12],
            contexts=[12, 13],
            index_positions=[10, 11],
        )
        metadata.upload()
        payloads = runtime.copies_to(metadata.buffer)
        assert len(payloads) == 1, "one packed upload per step"
        uploaded = np.frombuffer(payloads[-1], dtype=np.int64).reshape(
            LAYERS, STEP_METADATA_COLUMNS, SLOTS
        )
        # Row r carries row r's own value; slots past the active width keep the
        # zero the allocation published and no kernel reads.
        assert uploaded[0, STEP_POSITION_COLUMN].tolist() == [4, 9, 0, 0]
        assert uploaded[0, STEP_CONTEXT_COLUMN].tolist() == [5, 10, 0, 0]
        assert uploaded[0, STEP_INDEX_POSITION_COLUMN].tolist() == [3, 8, 0, 0]
        assert uploaded[1, STEP_POSITION_COLUMN].tolist() == [11, 12, 0, 0]
        assert uploaded[1, STEP_CONTEXT_COLUMN].tolist() == [12, 13, 0, 0]
        assert uploaded[1, STEP_INDEX_POSITION_COLUMN].tolist() == [10, 11, 0, 0]
        assert metadata.host.shape == (LAYERS, STEP_METADATA_COLUMNS, SLOTS)
    finally:
        metadata.close()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(positions=[1, 2], contexts=[1], index_positions=[0, 0]), ValueError, "must hold"),
        (
            dict(positions=[], contexts=[], index_positions=[]),
            ValueError,
            "at least one",
        ),
        (
            dict(
                positions=[1] * (SLOTS + 1),
                contexts=[1] * (SLOTS + 1),
                index_positions=[0] * (SLOTS + 1),
            ),
            ValueError,
            "at most",
        ),
        (dict(positions=[1.5], contexts=[2], index_positions=[0]), TypeError, "integer"),
        (dict(positions=[True], contexts=[2], index_positions=[0]), TypeError, "integer"),
    ],
)
def test_packed_metadata_stage_rejects_bad_rows_before_writing_host(
    kwargs: dict, error: type, match: str
) -> None:
    metadata, runtime = _allocate_packed()
    try:
        metadata.stage(0, positions=[7], contexts=[8], index_positions=[6])
        staged = metadata.host.copy()
        with pytest.raises(error, match=match):
            metadata.stage(0, **kwargs)
        assert np.array_equal(metadata.host, staged), "a rejected stage wrote the mirror"
        assert runtime.copies == []
        assert metadata.stage(0, positions=[7], contexts=[8], index_positions=[6]) == 1
        with pytest.raises(ValueError, match="outside"):
            metadata.stage(LAYERS, positions=[1], contexts=[2], index_positions=[0])
        with pytest.raises(ValueError, match="outside"):
            metadata.views(LAYERS)
    finally:
        metadata.close()


def test_packed_metadata_close_frees_once_and_retries_a_partial_failure() -> None:
    runtime = FakeRuntime(fail_free_call=1)
    metadata, _ = _allocate_packed(runtime=runtime)
    with pytest.raises(RuntimeError, match="simulated free failure"):
        metadata.close()
    assert metadata.closed is False
    assert metadata.closing is True
    copies_before = len(runtime.copies)
    with pytest.raises(RuntimeError, match="closing"):
        metadata.upload()
    with pytest.raises(RuntimeError, match="closing"):
        metadata.views(0)
    with pytest.raises(RuntimeError, match="closing"):
        metadata.stage(0, positions=[1], contexts=[2], index_positions=[0])
    assert len(runtime.copies) == copies_before
    metadata.close()
    assert metadata.closed is True
    assert runtime.allocations == {}
    metadata.close()
    assert runtime.freed == [metadata.buffer.ptr]
    with pytest.raises(RuntimeError, match="closed"):
        metadata.upload()


def test_packed_metadata_with_no_layers_allocates_but_rejects_every_layer() -> None:
    metadata, _ = _allocate_packed(layers=0)
    try:
        assert metadata.layers == 0
        assert metadata.host.size == 0
        assert metadata.buffer.nbytes >= 8
        # Nothing is staged, so the upload is a no-op rather than a zero-byte copy.
        metadata.upload()
        with pytest.raises(ValueError, match="outside"):
            metadata.views(0)
    finally:
        metadata.close()


def test_attention_owner_publishes_through_packed_views_without_its_own_upload() -> None:
    runtime = FakeRuntime()
    metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=1, state_slots=SLOTS, runtime=runtime
    )
    state = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
        position_view=metadata.views(0).position,
        context_view=metadata.views(0).context,
    )
    try:
        views = metadata.views(0)
        metadata.stage(
            0,
            positions=[255, 3],
            contexts=[256, 4],
            index_positions=[254, 2],
        )
        metadata.upload()
        copies_before = len(runtime.copies)
        active = state.prepare([2, 0], [255, 3], packed=views)
        # The first admission changes the page table away from the zero table the
        # allocation published, so exactly that one table upload happens here.
        assert len(runtime.copies) == copies_before + 1
        assert active.slot_ids == (2, 0)
        assert active.position.ptr == views.position.ptr
        assert active.context.ptr == views.context.ptr
        assert active.append_spans.live_counts.ptr == views.position.ptr
        assert active.decode_spans.live_counts.ptr == views.context.ptr
        assert active.position.numel == 2
        assert state.position_host[:2].tolist() == [255, 3]
        assert state.context_host[:2].tolist() == [256, 4]
        # Re-publishing the same row set copies nothing at all: no metadata upload
        # (the packed copy carried it) and no table upload (unchanged).
        copies_before = len(runtime.copies)
        state.prepare([2, 0], [255, 3], packed=views)
        assert len(runtime.copies) == copies_before, "packed publish uploaded metadata"
        # The owner's own buffers stay untouched: the packed copy is the only one.
        assert runtime.copies_to(state.position) == [bytes(SLOTS * 8)]
        assert runtime.copies_to(state.context) == [bytes(SLOTS * 8)]
        # A changed page table still uploads, packed views or not: the allocation
        # published one zero table, the first admission published the real one,
        # and this reassignment publishes a third.
        state.block_host[0] = np.asarray([99, 1, 2, 3, 4], dtype=np.int32)
        state.prepare([2, 0], [255, 3], packed=views)
        assert len(runtime.copies_to(state.block_table)) == 3
    finally:
        state.close()
        metadata.close()


def test_attention_publish_rejects_a_malformed_plan_before_any_device_write() -> None:
    runtime = FakeRuntime()
    state, _ = _allocate(runtime=runtime)
    try:
        plan = state.validate_rows([0, 1], [4, 5])
        state.prepare([0], [0])
        copies_before = len(runtime.copies)
        bad_plans = [
            Qwen4ExpBatchAttentionRowPlan(
                slot_ids=(0, 1),
                positions=plan.positions,
                contexts=plan.contexts[:1],
                block_table=plan.block_table,
                max_context_len=plan.max_context_len,
            ),
            Qwen4ExpBatchAttentionRowPlan(
                slot_ids=(0, 1),
                positions=plan.positions,
                contexts=plan.contexts,
                block_table=plan.block_table[:1],
                max_context_len=plan.max_context_len,
            ),
            Qwen4ExpBatchAttentionRowPlan(
                slot_ids=(0, 0),
                positions=plan.positions,
                contexts=plan.contexts,
                block_table=plan.block_table,
                max_context_len=plan.max_context_len,
            ),
            Qwen4ExpBatchAttentionRowPlan(
                slot_ids=(0, SLOTS),
                positions=plan.positions,
                contexts=plan.contexts,
                block_table=plan.block_table,
                max_context_len=plan.max_context_len,
            ),
            Qwen4ExpBatchAttentionRowPlan(
                slot_ids=(0, 1),
                positions=np.asarray([4, MAX_POSITIONS], dtype=np.int64),
                contexts=np.asarray([5, MAX_POSITIONS + 1], dtype=np.int64),
                block_table=plan.block_table,
                max_context_len=plan.max_context_len,
            ),
        ]
        for bad in bad_plans:
            with pytest.raises(ValueError):
                state.publish(bad)
            assert len(runtime.copies) == copies_before
            assert state.active_rows == 1
        with pytest.raises(TypeError, match="validate_rows"):
            state.publish({"slot_ids": (0,)})  # type: ignore[arg-type]
        # An empty row set is refused too.
        with pytest.raises(ValueError, match="at least one"):
            state.publish(
                Qwen4ExpBatchAttentionRowPlan(
                    slot_ids=(),
                    positions=np.zeros(0, dtype=np.int64),
                    contexts=np.zeros(0, dtype=np.int64),
                    block_table=np.zeros((0, state.blocks_per_request), dtype=np.int32),
                    max_context_len=0,
                )
            )
    finally:
        state.close()


def test_attention_allocate_rejects_a_packed_view_of_the_wrong_width() -> None:
    runtime = FakeRuntime()
    metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=1, state_slots=SLOTS + 1, runtime=runtime
    )
    try:
        views = metadata.views(0)
        with pytest.raises(ValueError, match="one int64 per state slot"):
            Qwen4ExpBatchAttentionState.allocate(
                state_slots=SLOTS,
                max_positions=MAX_POSITIONS,
                block_size=BLOCK_SIZE,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                runtime=runtime,
                position_view=views.position,
            )
        with pytest.raises(TypeError, match="DeviceBuffer"):
            Qwen4ExpBatchAttentionState.allocate(
                state_slots=SLOTS,
                max_positions=MAX_POSITIONS,
                block_size=BLOCK_SIZE,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                runtime=runtime,
                context_view=object(),
            )
    finally:
        metadata.close()
