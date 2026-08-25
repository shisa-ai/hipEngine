"""Stability contracts for the GGUF packed verify workspace.

These tests pin the serving-load requirements reviewed in the concurrency2
load-fault investigation:

1. Interleaved prefill/decode geometry requests must not free/reallocate the
   packed workspace (the churn that wedged the HIP allocator under load).
2. Any workspace free must fail closed while packed decode graphs still bind
   the buffers (use-after-free via graph replay page-faulted the load gate).
3. Scratch allocation must be atomic on failure (no leaked buffers).
4. Resident slot views must share the batch owner's packed workspace.
5. Packed prefill must invalidate decode graphs before reusing their bound
   private slots, then flush canonical state before the overwrite.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner


class _AllocRecorder:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.state_allocations: list[SimpleNamespace] = []
        self.scratch_allocations: list[SimpleNamespace] = []
        self.freed: list[int] = []
        self._monkeypatch = monkeypatch

    def install(self) -> None:
        self._monkeypatch.setattr(
            gguf_runner._GGUFPackedTargetState,
            "allocate",
            lambda runner, **kwargs: self._allocate_state(**kwargs),
        )
        self._monkeypatch.setattr(
            gguf_runner._GGUFFullAttentionPrefillScratch,
            "allocate",
            lambda runner, **kwargs: self._allocate_scratch(**kwargs),
        )

    def _allocate_state(self, *, slot_count, max_sequence_length, **_kwargs):
        state = SimpleNamespace(
            slot_count=int(slot_count),
            max_sequence_length=int(max_sequence_length),
            kv_layout=None,
            ptr=id(("state", len(self.state_allocations))),
        )
        self.state_allocations.append(state)
        return state

    def _allocate_scratch(self, *, rows, capacity, **_kwargs):
        scratch = SimpleNamespace(
            rows=int(rows),
            max_positions=int(capacity),
            gdn_segment_capacity=int(_kwargs.get("segments", 1)),
            ptr=id(("scratch", len(self.scratch_allocations))),
        )
        self.scratch_allocations.append(scratch)
        return scratch


def _make_owner(recorder: _AllocRecorder) -> gguf_runner.Qwen35GGUFResidentSession:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(weights=None)
    owner.scratch = None
    owner._device_kv_layout = None
    owner._packed_verify_state = None
    owner._packed_verify_scratch = None
    owner._packed_ar_attention_workspace = None
    owner._packed_verify_session_ids = ()
    owner._packed_verify_max_written_positions = ()
    owner._packed_decode_sessions = ()
    owner._packed_decode_last_layout = None
    owner._packed_decode_state_dirty = False
    owner._packed_decode_session_ids = ()
    owner._packed_decode_positions = ()
    owner._decode_graphs = []
    owner._device_kv_graph_handles = {}

    def fake_free(self, *, runtime):
        for workspace in (
            getattr(self, "_packed_ar_attention_workspace", None),
            self._packed_verify_scratch,
            self._packed_verify_state,
        ):
            if workspace is not None and hasattr(workspace, "ptr"):
                recorder.freed.append(workspace.ptr)
        self._packed_ar_attention_workspace = None
        self._packed_verify_scratch = None
        self._packed_verify_state = None

    owner._free_packed_verify_workspace = MethodType(fake_free, owner)
    return owner


def test_packed_workspace_interleave_is_alloc_stable(monkeypatch) -> None:
    """Alternating prefill (1x128) and decode (NxN) geometry must not churn."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()

    owner._ensure_packed_verify_workspace(
        slot_count=1, rows=128, max_sequence_length=1024, runtime=runtime
    )
    assert len(recorder.state_allocations) == 1
    assert len(recorder.scratch_allocations) == 1

    # Decode-shaped request, then prefill-shaped again, several cycles.
    for _ in range(3):
        owner._ensure_packed_verify_workspace(
            slot_count=4, rows=4, max_sequence_length=1024, runtime=runtime
        )
        owner._ensure_packed_verify_workspace(
            slot_count=2, rows=2, max_sequence_length=1024, runtime=runtime
        )
        owner._ensure_packed_verify_workspace(
            slot_count=1, rows=128, max_sequence_length=1024, runtime=runtime
        )

    assert len(recorder.state_allocations) == 1, "packed state must not be reallocated"
    assert len(recorder.scratch_allocations) == 1, "packed scratch must not be reallocated"
    assert recorder.freed == [], "no workspace buffer may be freed during interleave"


def test_packed_workspace_growth_keeps_union_geometry(monkeypatch) -> None:
    """A larger request grows axes monotonically; smaller requests reuse it."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()

    owner._ensure_packed_verify_workspace(
        slot_count=1, rows=4, max_sequence_length=1024, runtime=runtime
    )
    grown_state, grown_scratch = owner._ensure_packed_verify_workspace(
        slot_count=8, rows=128, max_sequence_length=1024, runtime=runtime
    )
    assert int(grown_state.slot_count) >= 8
    assert int(grown_scratch.rows) >= 128
    assert int(grown_scratch.gdn_segment_capacity) >= 8

    reused_state, reused_scratch = owner._ensure_packed_verify_workspace(
        slot_count=1, rows=4, max_sequence_length=1024, runtime=runtime
    )
    assert reused_state is grown_state, "workspace must not shrink back"
    assert reused_scratch is grown_scratch


def test_packed_workspace_long_packed_prefill_covers_total_rows(monkeypatch) -> None:
    """Scratch capacity covers packed rows even when per-slot context is smaller."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)

    _state, scratch = owner._ensure_packed_verify_workspace(
        slot_count=2,
        rows=4096,
        max_sequence_length=1024,
        runtime=SimpleNamespace(),
    )

    assert int(scratch.rows) >= 4096
    assert int(scratch.max_positions) >= 4096


def test_packed_workspace_growth_invalidates_live_graph_first(monkeypatch) -> None:
    """Growth closes binding graphs before freeing; a close-less graph fails closed."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()

    owner._ensure_packed_verify_workspace(
        slot_count=1, rows=4, max_sequence_length=1024, runtime=runtime
    )
    graph = SimpleNamespace(closed=False)
    graph.close = lambda: setattr(graph, "closed", True)
    owner._decode_graphs.append(graph)

    # Growth beyond the default union capacity closes the binding graph and
    # proceeds (the scheduler re-captures); it never frees while it is open.
    state, _ = owner._ensure_packed_verify_workspace(
        slot_count=16, rows=128, max_sequence_length=1024, runtime=runtime
    )
    assert graph.closed is True
    assert int(state.slot_count) >= 16
    reused_state, _ = owner._ensure_packed_verify_workspace(
        slot_count=1, rows=4, max_sequence_length=1024, runtime=runtime
    )
    assert reused_state is state

    # A graph that cannot be closed still fails the resize closed.
    uncloseable = SimpleNamespace(closed=False)
    owner._decode_graphs.append(uncloseable)
    with pytest.raises(RuntimeError, match="close"):
        owner._ensure_packed_verify_workspace(
            slot_count=64, rows=128, max_sequence_length=1024, runtime=runtime
        )
    assert uncloseable.closed is False


def test_slot_views_delegate_packed_workspace_to_batch_owner(monkeypatch) -> None:
    """resident_slot_view sessions share one packed workspace via the owner."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()
    delegated: list[tuple[int, int, int]] = []

    original_ensure = gguf_runner.Qwen35GGUFResidentSession._ensure_packed_verify_workspace

    def owner_ensure(self, *, slot_count, rows, max_sequence_length, runtime, stream=0):
        delegated.append((int(slot_count), int(rows), int(max_sequence_length)))
        return original_ensure(
            self,
            slot_count=slot_count,
            rows=rows,
            max_sequence_length=max_sequence_length,
            runtime=runtime,
            stream=stream,
        )

    owner._ensure_packed_verify_workspace = MethodType(owner_ensure, owner)
    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view.runner = owner.runner
    view.scratch = None
    view._device_kv_layout = None
    view._packed_verify_state = None
    view._packed_verify_scratch = None
    view._packed_ar_attention_workspace = None
    view._packed_verify_session_ids = ()
    view._packed_verify_max_written_positions = ()
    view._packed_decode_sessions = ()
    view._packed_decode_last_layout = None
    view._packed_decode_state_dirty = False
    view._packed_decode_session_ids = ()
    view._packed_decode_positions = ()
    view._decode_graphs = []
    view._device_kv_graph_handles = {}
    view._resident_batch_owner = owner

    view._ensure_packed_verify_workspace(
        slot_count=4, rows=4, max_sequence_length=1024, runtime=runtime
    )

    assert delegated == [(4, 4, 1024)], "slot views must delegate to the batch owner"
    assert view._packed_verify_state is owner._packed_verify_state
    assert view._packed_verify_scratch is owner._packed_verify_scratch


def test_slot_view_graph_invalidation_delegates_to_batch_owner() -> None:
    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._resident_batch_owner = None
    owner._resident_slot_views = []
    owner._device_kv_graph_handles = {}
    graph = SimpleNamespace(closed=False)
    graph.close = lambda: setattr(graph, "closed", True)
    owner._decode_graphs = [graph]

    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view._resident_batch_owner = owner

    assert view._invalidate_live_packed_decode_graphs() == 1
    assert graph.closed is True


def test_packed_prefill_invalidates_graph_before_flush_and_slot_reuse() -> None:
    """A replay graph must not survive a prefill overwrite of its private slots."""

    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner._packed_decode_state_dirty = True
    owner._resident_batch_owner = None
    owner._device_kv_graph_handles = {}
    events: list[object] = []
    result = [SimpleNamespace(token_id=7)]
    graph = SimpleNamespace(closed=False)

    def close_graph():
        events.append("invalidate")
        graph.closed = True

    graph.close = close_graph
    owner._decode_graphs = [graph]

    def flush(self, *, stream=0):
        assert graph.closed is True
        events.append(("flush", int(stream)))
        self._packed_decode_state_dirty = False
        return True

    def prefill(self, *args, **kwargs):
        events.append("prefill")
        return result

    owner.flush_packed_decode_state = MethodType(flush, owner)
    owner._prefill_batch_native_impl = MethodType(prefill, owner)
    owner._release_int8_prefill_oracle_buffers = MethodType(
        lambda self: None, owner
    )

    observed = owner.prefill_batch_native([[1]], sessions=(owner,), stream=9)

    assert observed is result
    assert events == ["invalidate", ("flush", 9), "prefill"]


def test_prefill_scratch_allocate_is_atomic_on_failure(monkeypatch) -> None:
    """A mid-allocation malloc failure must free every earlier buffer."""

    next_ptr = 0x100000
    live: dict[int, int] = {}

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        from hipengine.core.memory import DeviceBuffer

        if len(live) == 4:
            raise MemoryError("injected allocation failure")
        ptr = next_ptr
        next_ptr += max(8, int(nbytes) + 8)
        buffer = DeviceBuffer(ptr=ptr, nbytes=int(nbytes))
        live[ptr] = int(nbytes)
        return buffer

    freed: list[int] = []

    def fake_free(buffer, *, runtime):
        if int(buffer.ptr) in live:
            freed.append(int(buffer.ptr))

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "free", fake_free)
    cfg = SimpleNamespace(
        expert_used_count=2,
        is_moe=True,
        expert_count=4,
        expert_shared_feed_forward_length=8,
        ssm_inner_size=6,
        ssm_conv_kernel=4,
        ssm_group_count=2,
        ssm_time_step_rank=2,
        ssm_state_size=3,
        head_count_kv=2,
        key_length=4,
        rope_dimension_count=4,
        rope_freq_base=10000.0,
        head_count=4,
    )
    runner = SimpleNamespace(
        hidden_size=8,
        q_width=16,
        kv_width=8,
        ffn_size=12,
        linear_qkv_width=10,
        ssm_value_dim=2,
        backend="hip_gfx1151",
        weights=SimpleNamespace(config=cfg),
    )

    with pytest.raises(MemoryError):
        gguf_runner._GGUFFullAttentionPrefillScratch.allocate(
            runner,
            rows=6,
            capacity=1024,
            allocate_kv_cache=False,
            segments=4,
            runtime=SimpleNamespace(),
        )

    leaked = [ptr for ptr in live if ptr not in freed]
    assert leaked == [], f"{len(leaked)} buffer(s) leaked by partial allocation failure"
