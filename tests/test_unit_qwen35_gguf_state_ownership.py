"""Ownership contracts for GGUF resident state that grows in place.

Three resident families replace live device buffers when a request outgrows
their current capacity: the verify block buffers, the verify linear-state row
buffers, and the packed AR split-K attention workspace. A resident slot view
additionally grows its own copy of the verify block family while sharing the
batch owner's prefix-snapshot arena pool.

Growth is all-or-nothing. A failed replacement must leave the previous buffers
live (same objects, same capacity), must not leak any buffer it acquired, and
must leave teardown able to release every pointer exactly once. A slot view
never owns the batch owner's prefix-snapshot arena pool, so closing a view must
not close it.

Everything here runs on CPU through the real session methods with a recording
stand-in for the module's ``malloc``/``free`` bindings.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner

_VERIFY_BLOCK_FIELDS = (
    "_verify_hidden_seed_buf",
    "_verify_hidden_f32_a",
    "_verify_hidden_f32_b",
    "_verify_conv_out_f32",
    "_verify_token_ids_i64",
    "_verify_token_counter_i64",
)

_VERIFY_LINEAR_STAGING_FIELDS = (
    "_verify_linear_state_src_conv_table_buf",
    "_verify_linear_state_src_recurrent_table_buf",
    "_verify_linear_state_dst_conv_table_buf",
    "_verify_linear_state_dst_recurrent_table_buf",
    "_verify_linear_state_commit_row_i32_buf",
    "_verify_linear_state_src_conv_host",
    "_verify_linear_state_src_recurrent_host",
    "_verify_linear_state_src_conv_cached",
    "_verify_linear_state_src_recurrent_cached",
    "_verify_linear_state_dst_conv_host",
    "_verify_linear_state_dst_recurrent_host",
)

_INJECTED_FAILURE = "injected device allocation failure"


class _FakeDeviceAllocator:
    """Recording stand-in for the module's ``malloc``/``free`` bindings.

    ``free`` refuses a pointer that is not currently live, so a double free or
    a free of a buffer another owner still holds fails the test at the exact
    call site instead of passing silently.
    """

    def __init__(self) -> None:
        self._next_ptr = 0x400000
        self.calls = 0
        self.fail_on_call: int | None = None
        self.live: dict[int, DeviceBuffer] = {}
        self.released: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gguf_runner, "malloc", self.malloc)
        monkeypatch.setattr(gguf_runner, "free", self.free)

    def fail_nth_next(self, offset: int) -> None:
        """Make the ``offset``-th allocation counted from now fail."""

        self.fail_on_call = self.calls + int(offset)

    def malloc(self, nbytes: int, *, runtime: object | None = None) -> DeviceBuffer:
        del runtime
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError(f"{_INJECTED_FAILURE} on call {self.calls}")
        buffer = DeviceBuffer(ptr=self._next_ptr, nbytes=int(nbytes))
        self._next_ptr += 0x1000
        self.live[int(buffer.ptr)] = buffer
        return buffer

    def free(self, buffer: DeviceBuffer, *, runtime: object | None = None) -> None:
        del runtime
        ptr = int(buffer.ptr)
        if ptr not in self.live:
            raise RuntimeError(
                f"released device pointer {ptr:#x} that is not a live allocation"
                " (double free, or a buffer owned by someone else)"
            )
        del self.live[ptr]
        self.released.append(ptr)

    def release_count(self, buffer: DeviceBuffer | None) -> int:
        if buffer is None:
            return 0
        return self.released.count(int(buffer.ptr))

    def assert_live(
        self,
        expected: tuple[DeviceBuffer | None, ...],
        *,
        context: str,
    ) -> None:
        """The only live allocations are exactly ``expected``."""

        expected_ptrs = {int(buffer.ptr) for buffer in expected if buffer is not None}
        live = set(self.live)
        released_expected = sorted(expected_ptrs - live)
        assert not released_expected, (
            f"{context}: previous state was released instead of retained:"
            f" {[hex(ptr) for ptr in released_expected]}"
        )
        leaked = sorted(live - expected_ptrs)
        assert not leaked, (
            f"{context}: {len(leaked)} buffer(s) acquired by the attempt leaked:"
            f" {[hex(ptr) for ptr in leaked]}"
        )

    def assert_no_replacement_leak(
        self,
        retained: tuple[DeviceBuffer | None, ...],
        *,
        context: str,
    ) -> None:
        """Nothing the failed attempt acquired is still live off the retained set."""

        retained_ptrs = {int(buffer.ptr) for buffer in retained if buffer is not None}
        leaked = sorted(set(self.live) - retained_ptrs)
        assert not leaked, (
            f"{context}: {len(leaked)} buffer(s) acquired by the failed attempt were"
            f" neither released nor part of the retained state:"
            f" {[hex(ptr) for ptr in leaked]}"
        )


class _RecordingArenaPool:
    """Minimal prefix-snapshot arena pool that records closure."""

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _FakeSlotScratchOwner:
    """Target scratch owner exposing one scratch object per resident slot."""

    def __init__(self, *, slot_count: int = 3) -> None:
        self.slot_count = int(slot_count)
        self.buffers: tuple[object, ...] = ()
        self._slot_scratch: dict[int, object] = {}

    def for_slot(self, index: int) -> object:
        return self._slot_scratch.setdefault(
            int(index), SimpleNamespace(slot=int(index))
        )


def _fake_runner(
    *,
    hidden_size: int = 8,
    linear_qkv_width: int = 4,
    head_count: int = 2,
    q_width: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=int(hidden_size),
        q_width=int(q_width),
        linear_qkv_width=int(linear_qkv_width),
        weights=SimpleNamespace(config=SimpleNamespace(head_count=int(head_count))),
    )


def _minimal_session(**overrides: object) -> gguf_runner.Qwen35GGUFResidentSession:
    """Session carrying the state the resident teardown and slot-view paths touch."""

    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.runner = _fake_runner()
    session.runtime = SimpleNamespace()
    session.scratch = None
    session.max_batch_size = 2
    session._resident_batch_owner = session
    session._target_scratch_owner = None
    session._target_layout = gguf_runner.Qwen35GGUFResidentTargetLayout(
        max_batch_size=2,
        hidden_size=8,
        vocab_size=16,
        max_sequence_length=256,
    )
    session._owns_runner = False
    session._buffers = ()
    session._prefill_f16_staging_buf = None
    session._iq_dense_mmq_buf = None
    session._q6_integer_mmq_library = None
    session._gapped_slot_block_table_cache = None
    session._gapped_head_major_cache = None
    for name in _VERIFY_BLOCK_FIELDS:
        setattr(session, name, None)
    session._verify_block_rows_capacity = 0
    session._verify_hidden_seed_rows_populated = 0
    session._verify_lm_out_values = None
    session._verify_lm_out_indices_i32 = None
    session._verify_lm_block_indices_i32 = None
    session._verify_lm_block_values = None
    session._verify_lm_q8_1 = None
    session._verify_logits_buf = None
    session._verify_lm_rows_capacity = 0
    session._verify_linear_conv_state_rows = ()
    session._verify_linear_recurrent_state_rows = ()
    session._verify_linear_state_rows_capacity = 0
    session._verify_linear_conv_initial_snapshots = ()
    session._verify_linear_recurrent_initial_snapshots = ()
    session._verify_linear_initial_snapshot_users = 0
    for name in _VERIFY_LINEAR_STAGING_FIELDS:
        setattr(session, name, None)
    session._verify_linear_state_conv_row_nbytes = 0
    session._verify_linear_state_recurrent_row_nbytes = 0
    session._verify_linear_state_layer_count = 0
    for name, value in overrides.items():
        setattr(session, name, value)
    return session


def _fake_linear_scratch(*, layers: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        layer_conv_states=tuple(
            SimpleNamespace(nbytes=32) for _ in range(int(layers))
        ),
        layer_recurrent_states=tuple(
            SimpleNamespace(nbytes=64) for _ in range(int(layers))
        ),
    )


def _verify_block_snapshot(
    session: gguf_runner.Qwen35GGUFResidentSession,
) -> dict[str, DeviceBuffer | None]:
    return {name: getattr(session, name) for name in _VERIFY_BLOCK_FIELDS}


def _assert_same_buffers(
    session: gguf_runner.Qwen35GGUFResidentSession,
    snapshot: dict[str, DeviceBuffer | None],
    *,
    context: str,
) -> None:
    for name, buffer in snapshot.items():
        assert getattr(session, name) is buffer, (
            f"{context}: {name} was replaced instead of retained"
        )


def _assert_same_rows(
    session: gguf_runner.Qwen35GGUFResidentSession,
    conv_rows: tuple[DeviceBuffer | None, ...],
    recurrent_rows: tuple[DeviceBuffer | None, ...],
    *,
    context: str,
) -> None:
    for name, expected in (
        ("_verify_linear_conv_state_rows", conv_rows),
        ("_verify_linear_recurrent_state_rows", recurrent_rows),
    ):
        current = getattr(session, name)
        assert len(current) == len(expected), f"{context}: {name} lost rows"
        for index, (live, retained) in enumerate(zip(current, expected)):
            assert live is retained, (
                f"{context}: {name}[{index}] was replaced instead of retained"
            )


# --------------------------------------------------------------------------
# verify block buffers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fail_offset", [1, 3, 6], ids=["first", "middle", "last"])
def test_verify_block_growth_failure_retains_previous_state(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_verify_block_buffers(2, runtime=runtime)
    previous = _verify_block_snapshot(session)
    previous_buffers = tuple(previous.values())
    assert session._verify_block_rows_capacity == 2
    assert all(buffer is not None for buffer in previous_buffers)
    allocator.assert_live(previous_buffers, context="verify block baseline")

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_verify_block_buffers(4, runtime=runtime)
    allocator.fail_on_call = None

    _assert_same_buffers(session, previous, context="failed verify block growth")
    assert session._verify_block_rows_capacity == 2
    allocator.assert_live(previous_buffers, context="failed verify block growth")


@pytest.mark.parametrize("fail_offset", [3, 6], ids=["middle", "last"])
def test_verify_block_growth_failure_leaks_no_partial_allocation(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_verify_block_buffers(2, runtime=runtime)
    previous_buffers = tuple(_verify_block_snapshot(session).values())

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_verify_block_buffers(4, runtime=runtime)
    allocator.fail_on_call = None

    allocator.assert_no_replacement_leak(
        previous_buffers, context="failed verify block growth"
    )


@pytest.mark.parametrize("fail_offset", [1, 6], ids=["first", "last"])
def test_verify_block_cleanup_after_failed_growth_frees_each_pointer_once(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_verify_block_buffers(2, runtime=runtime)
    previous_buffers = tuple(_verify_block_snapshot(session).values())

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_verify_block_buffers(4, runtime=runtime)
    allocator.fail_on_call = None

    session._close_resident_slot_view_buffers(runtime=runtime)

    for buffer in previous_buffers:
        assert allocator.release_count(buffer) == 1, (
            f"cleanup released {hex(int(buffer.ptr))} "
            f"{allocator.release_count(buffer)} times"
        )
    assert allocator.live == {}, (
        "cleanup left live allocations behind:"
        f" {[hex(ptr) for ptr in sorted(allocator.live)]}"
    )


def test_verify_block_growth_success_releases_previous_once_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_verify_block_buffers(2, runtime=runtime)
    previous = _verify_block_snapshot(session)

    session._ensure_verify_block_buffers(4, runtime=runtime)
    grown = _verify_block_snapshot(session)

    assert session._verify_block_rows_capacity == 4
    for name, buffer in previous.items():
        assert allocator.release_count(buffer) == 1, (
            f"growing past capacity did not release the previous {name} exactly once"
        )
        assert grown[name] is not buffer, f"{name} was not replaced on growth"
        assert grown[name] is not None
    assert int(grown["_verify_hidden_seed_buf"].nbytes) == 2 * int(
        previous["_verify_hidden_seed_buf"].nbytes
    ), "row-scaled verify storage did not follow the requested row count"
    allocator.assert_live(tuple(grown.values()), context="verify block growth")

    calls_after_growth = allocator.calls
    session._ensure_verify_block_buffers(3, runtime=runtime)
    assert allocator.calls == calls_after_growth, (
        "a request within capacity must not allocate"
    )
    _assert_same_buffers(session, grown, context="verify block reuse")
    assert session._verify_block_rows_capacity == 4


# --------------------------------------------------------------------------
# verify linear-state row buffers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fail_offset", [1, 4, 6], ids=["first", "middle", "last"])
def test_linear_state_row_growth_failure_retains_previous_rows(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session(scratch=_fake_linear_scratch(layers=3))
    runtime = SimpleNamespace()

    session._ensure_verify_linear_state_row_buffers(2, runtime=runtime)
    previous_conv = session._verify_linear_conv_state_rows
    previous_recurrent = session._verify_linear_recurrent_state_rows
    previous = (*previous_conv, *previous_recurrent)
    assert session._verify_linear_state_rows_capacity == 2
    assert len(previous_conv) == 3 and all(buffer is not None for buffer in previous_conv)
    allocator.assert_live(previous, context="linear-state row baseline")

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_verify_linear_state_row_buffers(4, runtime=runtime)
    allocator.fail_on_call = None

    _assert_same_rows(
        session,
        previous_conv,
        previous_recurrent,
        context="failed linear-state row growth",
    )
    assert session._verify_linear_state_rows_capacity == 2
    allocator.assert_live(previous, context="failed linear-state row growth")

    session._free_verify_linear_state_row_buffers(runtime=runtime)
    for buffer in previous:
        assert allocator.release_count(buffer) == 1
    assert allocator.live == {}
    assert session._verify_linear_state_rows_capacity == 0


@pytest.mark.parametrize("fail_offset", [4, 6], ids=["middle", "last"])
def test_linear_state_row_growth_failure_leaks_no_partial_allocation(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session(scratch=_fake_linear_scratch(layers=3))
    runtime = SimpleNamespace()

    session._ensure_verify_linear_state_row_buffers(2, runtime=runtime)
    previous = (
        *session._verify_linear_conv_state_rows,
        *session._verify_linear_recurrent_state_rows,
    )

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_verify_linear_state_row_buffers(4, runtime=runtime)
    allocator.fail_on_call = None

    allocator.assert_no_replacement_leak(
        previous, context="failed linear-state row growth"
    )


def test_linear_state_row_growth_success_releases_previous_once_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session(scratch=_fake_linear_scratch(layers=3))
    runtime = SimpleNamespace()

    session._ensure_verify_linear_state_row_buffers(2, runtime=runtime)
    previous = (
        *session._verify_linear_conv_state_rows,
        *session._verify_linear_recurrent_state_rows,
    )

    session._ensure_verify_linear_state_row_buffers(4, runtime=runtime)
    grown_conv = session._verify_linear_conv_state_rows
    grown_recurrent = session._verify_linear_recurrent_state_rows
    grown = (*grown_conv, *grown_recurrent)

    assert session._verify_linear_state_rows_capacity == 4
    assert len(grown) == 6 and all(buffer is not None for buffer in grown)
    for buffer in previous:
        assert allocator.release_count(buffer) == 1, (
            "growing past capacity did not release the previous row exactly once"
        )
    assert not set(map(id, previous)) & set(map(id, grown))
    allocator.assert_live(grown, context="linear-state row growth")

    calls_after_growth = allocator.calls
    session._ensure_verify_linear_state_row_buffers(3, runtime=runtime)
    assert allocator.calls == calls_after_growth, (
        "a request within capacity must not allocate"
    )
    _assert_same_rows(
        session, grown_conv, grown_recurrent, context="linear-state row reuse"
    )
    assert session._verify_linear_state_rows_capacity == 4


# --------------------------------------------------------------------------
# packed AR split-K attention workspace
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fail_offset", [1, 2, 3], ids=["first", "middle", "last"])
def test_packed_ar_workspace_growth_failure_retains_previous_workspace(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_packed_ar_attention_workspace(
        rows=2, max_context_len=256, runtime=runtime
    )
    previous = session._packed_ar_attention_workspace
    assert previous is not None
    previous_buffers = tuple(previous.buffers)
    assert len(previous_buffers) == 3
    assert all(buffer is not None for buffer in previous_buffers)
    allocator.assert_live(previous_buffers, context="packed AR workspace baseline")

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_packed_ar_attention_workspace(
            rows=4, max_context_len=512, runtime=runtime
        )
    allocator.fail_on_call = None

    assert session._packed_ar_attention_workspace is previous, (
        "a failed growth dropped the live packed AR workspace"
    )
    assert (int(previous.rows), int(previous.num_splits)) == (2, 1)
    allocator.assert_live(previous_buffers, context="failed packed AR workspace growth")


@pytest.mark.parametrize("fail_offset", [2, 3], ids=["middle", "last"])
def test_packed_ar_workspace_growth_failure_leaks_no_partial_allocation(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_packed_ar_attention_workspace(
        rows=2, max_context_len=256, runtime=runtime
    )
    previous_buffers = tuple(session._packed_ar_attention_workspace.buffers)

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_packed_ar_attention_workspace(
            rows=4, max_context_len=512, runtime=runtime
        )
    allocator.fail_on_call = None

    allocator.assert_no_replacement_leak(
        previous_buffers, context="failed packed AR workspace growth"
    )


@pytest.mark.parametrize("fail_offset", [1, 3], ids=["first", "last"])
def test_packed_ar_workspace_cleanup_after_failed_growth_frees_each_pointer_once(
    monkeypatch: pytest.MonkeyPatch, fail_offset: int
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_packed_ar_attention_workspace(
        rows=2, max_context_len=256, runtime=runtime
    )
    previous_buffers = tuple(session._packed_ar_attention_workspace.buffers)

    allocator.fail_nth_next(fail_offset)
    with pytest.raises(RuntimeError, match=_INJECTED_FAILURE):
        session._ensure_packed_ar_attention_workspace(
            rows=4, max_context_len=512, runtime=runtime
        )
    allocator.fail_on_call = None

    session._free_packed_verify_workspace(runtime=runtime)

    for buffer in previous_buffers:
        assert allocator.release_count(buffer) == 1, (
            f"teardown released {hex(int(buffer.ptr))} "
            f"{allocator.release_count(buffer)} times"
        )
    assert allocator.live == {}, (
        "teardown left live allocations behind:"
        f" {[hex(ptr) for ptr in sorted(allocator.live)]}"
    )
    assert session._packed_ar_attention_workspace is None


def test_packed_ar_workspace_growth_success_releases_previous_once_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    session = _minimal_session()
    runtime = SimpleNamespace()

    session._ensure_packed_ar_attention_workspace(
        rows=2, max_context_len=256, runtime=runtime
    )
    previous = session._packed_ar_attention_workspace
    previous_buffers = tuple(previous.buffers)

    session._ensure_packed_ar_attention_workspace(
        rows=4, max_context_len=512, runtime=runtime
    )
    grown = session._packed_ar_attention_workspace

    assert grown is not previous
    assert (int(grown.rows), int(grown.num_splits)) == (4, 2)
    for buffer in previous_buffers:
        assert allocator.release_count(buffer) == 1, (
            "growing past capacity did not release the previous workspace buffer"
            " exactly once"
        )
    allocator.assert_live(tuple(grown.buffers), context="packed AR workspace growth")

    calls_after_growth = allocator.calls
    reused = session._ensure_packed_ar_attention_workspace(
        rows=3, max_context_len=400, runtime=runtime
    )
    assert reused is grown, "a request within capacity must reuse the workspace"
    assert allocator.calls == calls_after_growth, (
        "a request within capacity must not allocate"
    )


# --------------------------------------------------------------------------
# resident slot views
# --------------------------------------------------------------------------


def test_resident_slot_view_does_not_inherit_or_close_owner_prefix_arena_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeDeviceAllocator()
    allocator.install(monkeypatch)
    runtime = SimpleNamespace()
    pool = _RecordingArenaPool()

    owner = _minimal_session()
    owner._gguf_prefix_snapshot_arena_pool = pool
    owner._target_scratch_owner = _FakeSlotScratchOwner(slot_count=3)
    owner.scratch = owner._target_scratch_owner.for_slot(0)
    owner._ensure_verify_block_buffers(2, runtime=runtime)
    owner_buffers = _verify_block_snapshot(owner)
    allocator.assert_live(
        tuple(owner_buffers.values()), context="batch owner baseline"
    )

    view = owner.resident_slot_view(1)
    assert view.__dict__.get("_gguf_prefix_snapshot_arena_pool", None) is None, (
        "a resident slot view must not inherit the batch owner's"
        " prefix-snapshot arena pool"
    )

    view._ensure_verify_block_buffers(2, runtime=runtime)
    view_buffers = tuple(_verify_block_snapshot(view).values())
    assert all(buffer is not None for buffer in view_buffers)
    assert not set(map(id, view_buffers)) & set(map(id, owner_buffers.values())), (
        "the view reused the batch owner's verify buffers"
    )

    view._close_resident_slot_view_buffers(runtime=runtime)

    assert pool.close_calls == 0, (
        "closing a resident slot view closed the batch owner's"
        " prefix-snapshot arena pool"
    )
    assert not pool.closed
    assert owner.__dict__.get("_gguf_prefix_snapshot_arena_pool") is pool
    assert owner._prefix_snapshot_arena_pool() is pool, (
        "the batch owner's prefix-snapshot arena pool was replaced after a slot"
        " view closed"
    )

    for buffer in view_buffers:
        assert allocator.release_count(buffer) == 1, (
            f"view cleanup released {hex(int(buffer.ptr))} "
            f"{allocator.release_count(buffer)} times"
        )
    _assert_same_buffers(owner, owner_buffers, context="owner after view cleanup")
    allocator.assert_live(
        tuple(owner_buffers.values()), context="owner after view cleanup"
    )

    released_before = len(allocator.released)
    view._close_resident_slot_view_buffers(runtime=runtime)
    assert len(allocator.released) == released_before, (
        "repeating slot-view cleanup released buffers a second time"
    )
    assert pool.close_calls == 0
