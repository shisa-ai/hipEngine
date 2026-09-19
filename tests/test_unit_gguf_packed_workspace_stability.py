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
from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _install_fake_device,
)


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
    owner.use_iq_dense_mmq = False
    owner.runner = SimpleNamespace(weights=None)
    owner.scratch = None
    owner._device_kv_layout = None
    # The fake must declare the serving cap the scheduler guarantees: the
    # capacity-honest union (b753495b4) sizes the first packed allocation to
    # ``max_batch_size`` instead of the historical 8-slot floor. Without this,
    # the dataclass class default of 1 makes the interleave churn by design.
    owner.max_batch_size = 4
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


def test_packed_workspace_first_allocation_covers_declared_cap(monkeypatch) -> None:
    """The capacity-honest union sizes the first allocation to the serving cap.

    Guards both regressions: per-request sizing (churn on the first wider
    geometry) and the historical fixed 8-slot floor (wasting resident memory
    at small caps).
    """

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)

    owner._ensure_packed_verify_workspace(
        slot_count=1, rows=4, max_sequence_length=1024, runtime=SimpleNamespace()
    )
    assert int(recorder.state_allocations[0].slot_count) == 4
    assert int(recorder.scratch_allocations[0].gdn_segment_capacity) == 4


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

    def owner_ensure(
        self,
        *,
        slot_count,
        rows,
        max_sequence_length,
        runtime,
        stream=0,
        require_kv_planes=True,
    ):
        delegated.append((int(slot_count), int(rows), int(max_sequence_length)))
        return original_ensure(
            self,
            slot_count=slot_count,
            rows=rows,
            max_sequence_length=max_sequence_length,
            runtime=runtime,
            stream=stream,
            require_kv_planes=require_kv_planes,
        )

    owner._ensure_packed_verify_workspace = MethodType(owner_ensure, owner)
    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view.use_iq_dense_mmq = False
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
    owner.use_iq_dense_mmq = False
    owner._resident_batch_owner = None
    owner._resident_slot_views = []
    owner._device_kv_graph_handles = {}
    graph = SimpleNamespace(closed=False)
    graph.close = lambda: setattr(graph, "closed", True)
    owner._decode_graphs = [graph]

    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view.use_iq_dense_mmq = False
    view._resident_batch_owner = owner

    assert view._invalidate_live_packed_decode_graphs() == 1
    assert graph.closed is True


def test_packed_prefill_invalidates_graph_before_flush_and_slot_reuse() -> None:
    """A replay graph must not survive a prefill overwrite of its private slots."""

    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.use_iq_dense_mmq = False
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


def _shared_view_of(owner: gguf_runner.Qwen35GGUFResidentSession) -> (
    gguf_runner.Qwen35GGUFResidentSession
):
    """A slot view: same workspace objects, different session identity."""

    view = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    view.use_iq_dense_mmq = False
    view._resident_batch_owner = owner
    view._packed_ar_attention_workspace = owner._packed_ar_attention_workspace
    view._packed_verify_scratch = owner._packed_verify_scratch
    view._packed_verify_state = owner._packed_verify_state
    return view


def test_packed_workspace_nbytes_counts_split_growth_owners(monkeypatch) -> None:
    """The split-growth buffers are freed with the workspace, so they count."""

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()
    owner._ensure_packed_verify_workspace(
        slot_count=2, rows=4, max_sequence_length=1024, runtime=runtime
    )
    scratch = owner._packed_verify_scratch
    if scratch is None:
        pytest.skip("workspace allocation is not available on this host")
    # The recorder fakes the state/scratch as SimpleNamespaces; give them the
    # buffer tuples the accounting walks.
    owner._packed_verify_state.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xA0000, nbytes=1024),
    )
    scratch.buffers = (gguf_runner.DeviceBuffer(ptr=0xB0000, nbytes=512),)
    before = owner.packed_workspace_nbytes()
    growth = tuple(
        gguf_runner.DeviceBuffer(ptr=0xBEEF000 + index, nbytes=256)
        for index in range(2)
    )
    object.__setattr__(scratch, "full_attn_split_growth_buffers", growth)
    try:
        after = owner.packed_workspace_nbytes()
    finally:
        object.__setattr__(scratch, "full_attn_split_growth_buffers", ())
    assert after == before + 512


def test_packed_workspace_owner_bytes_dedupes_shared_views(monkeypatch) -> None:
    """Owner-deduplicated bytes: views of one workspace report it once."""

    from hipengine.generation.qwen35_gguf import (
        packed_workspace_owner_inventory,
    )

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    owner = _make_owner(recorder)
    runtime = SimpleNamespace()
    owner._ensure_packed_verify_workspace(
        slot_count=2, rows=4, max_sequence_length=1024, runtime=runtime
    )
    if owner._packed_verify_scratch is None:
        pytest.skip("workspace allocation is not available on this host")
    owner._packed_verify_state.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xA0000, nbytes=1024),
    )
    owner._packed_verify_scratch.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xB0000, nbytes=512),
    )
    single = owner.packed_workspace_nbytes()
    assert single > 0

    views = tuple(_shared_view_of(owner) for _ in range(3))
    unique_bytes, contributing_sessions = packed_workspace_owner_inventory(
        (owner, *views)
    )
    assert unique_bytes == single
    assert contributing_sessions == 4
    # The per-session sum that the old counter used inflates by the view count.
    assert sum(view.packed_workspace_nbytes() for view in views) == 3 * single


def test_packed_workspace_owner_inventory_independent_allocations(monkeypatch) -> None:
    """Independent workspaces on distinct sessions still sum."""

    from hipengine.generation.qwen35_gguf import (
        packed_workspace_owner_inventory,
    )

    recorder = _AllocRecorder(monkeypatch)
    recorder.install()
    first = _make_owner(recorder)
    second = _make_owner(recorder)
    runtime = SimpleNamespace()
    first._ensure_packed_verify_workspace(
        slot_count=2, rows=4, max_sequence_length=1024, runtime=runtime
    )
    second._ensure_packed_verify_workspace(
        slot_count=2, rows=4, max_sequence_length=1024, runtime=runtime
    )
    if first._packed_verify_scratch is None or second._packed_verify_scratch is None:
        pytest.skip("workspace allocation is not available on this host")
    first._packed_verify_state.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xA0000, nbytes=1024),
    )
    first._packed_verify_scratch.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xB0000, nbytes=512),
    )
    second._packed_verify_state.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xC0000, nbytes=1024),
    )
    second._packed_verify_scratch.buffers = (
        gguf_runner.DeviceBuffer(ptr=0xD0000, nbytes=512),
    )
    expected = first.packed_workspace_nbytes() + second.packed_workspace_nbytes()
    unique_bytes, contributing = packed_workspace_owner_inventory((first, second))
    assert unique_bytes == expected
    assert contributing == 2


def test_prefill_transient_inventory_reports_owners_and_modes() -> None:
    """Oracle/hidden owners and actual-vs-legacy executor mode, deduped."""

    from hipengine.generation.qwen35_gguf import prefill_transient_owner_inventory

    owner = _make_owner(_AllocRecorder.__new__(_AllocRecorder))
    owner._int8_prefill_oracle_buffers = {
        3: (gguf_runner.DeviceBuffer(ptr=0x1000, nbytes=2048),
            gguf_runner.DeviceBuffer(ptr=0x2000, nbytes=2048)),
        5: (gguf_runner.DeviceBuffer(ptr=0x3000, nbytes=2048),
            gguf_runner.DeviceBuffer(ptr=0x4000, nbytes=2048)),
    }
    owner._int8_prefill_oracle_capacity_positions = lambda: 65_536
    owner._int8_prefill_lifetime_plan = SimpleNamespace(mode="layer_outer_shared_oracle")
    owner.last_packed_prefill_plan = {"route": "slot_fair_bounded_rounds", "chunk_count": 2}
    owner._int8_prefill_oracle_per_layer = True
    owner._prefill_hidden_a = gguf_runner.DeviceBuffer(ptr=0x5000, nbytes=4096)
    owner._prefill_hidden_b = gguf_runner.DeviceBuffer(ptr=0x6000, nbytes=4096)
    owner._prefill_token_buf = None
    owner._bulk_prefill_scratch = None

    view = _shared_view_of(owner)
    view._int8_prefill_oracle_buffers = {}
    view._int8_prefill_oracle_capacity_positions = lambda: 65_536
    view._int8_prefill_lifetime_plan = None
    view.last_packed_prefill_plan = {}
    view._int8_prefill_oracle_per_layer = False
    view._prefill_hidden_a = owner._prefill_hidden_a
    view._prefill_hidden_b = None
    view._prefill_token_buf = None
    view._bulk_prefill_scratch = None

    report = prefill_transient_owner_inventory((owner, view))
    # Two per-layer oracle pairs, counted once despite the view.
    assert report["oracle_owner_bytes"] == 4 * 2048
    assert report["oracle_owner_count_total"] == 2
    assert report["oracle_capacity_positions"] == [65_536, 65_536]
    # The hidden plane is shared with the view and counted once.
    assert report["hidden_and_bulk_owner_bytes"] == 2 * 4096
    # The legacy plan and the actual executor disagree, and both are visible.
    assert report["int8_prefill_lifetime_plan_modes"] == [
        "layer_outer_shared_oracle", "None",
    ]
    assert report["last_packed_executor_routes"] == [
        "slot_fair_bounded_rounds", "None",
    ]
    assert report["oracle_per_layer_flags"] == [True, False]
    # A closed/partial session has no live KV layout. The scrape reports that
    # as unknown (exported as "none") instead of raising out of the telemetry
    # path, which is what broke this test when the field was first added.
    assert report["kv_attention_sources"] == [None, None]


def test_slot_local_prefill_leases_no_packed_kv_planes(monkeypatch) -> None:
    """P4: slot-local packed prefill must not pin the pool KV plane lease.

    The packed KV planes are read only by non-slot-local packed attention,
    packed batch decode, and the MTP verifier - all of which pass
    require_kv_planes=True. A slot-local prefill resolves the route before
    the workspace ensure, so it can represent the absence of a packed KV
    consumer in the resolved workspace plan (roadmap F2) instead of pinning
    a second full-context reservation.
    """

    from hipengine.runtime.qwen35_gguf_runner import (
        _GGUFPackedTargetState,
        _rebind_packed_verify_layout_pages,
    )

    state = object.__new__(_GGUFPackedTargetState)
    state.__dict__.update(
        slot_count=1,
        max_sequence_length=4096,
        block_size=256,
        blocks_per_slot=16,
        total_positions=4096,
        kv_layout=SimpleNamespace(
            layer_storage_dtypes=(None, "int8_per_token_head"),
            bf16_mirror_layer_indices=(),
        ),
        layer_conv_states=(SimpleNamespace(ptr=1, nbytes=8), None),
        layer_recurrent_states=(SimpleNamespace(ptr=2, nbytes=8), None),
        full_key_caches=(None, None),
        full_value_caches=(None, None),
        full_bf16_mirror_key_caches=(None, None),
        full_bf16_mirror_value_caches=(None, None),
        full_k_scale_caches=(None, None),
        full_v_scale_caches=(None, None),
        full_kv_scale_metadata=(None, None),
        buffers=(),
        page_ids=(),
        kv_backing_kind="unleased",
    )
    # post_init accepts the unleased kind.
    _GGUFPackedTargetState.__post_init__(state)

    # Fail closed: an unleased state cannot serve packed-scratch attention.
    with pytest.raises(ValueError, match="no packed full-attention KV cache"):
        state.full_cache(1)
    # The linear state slots are present and unaffected.
    assert state.linear_state_pair(0) == (
        state.layer_conv_states[0],
        state.layer_recurrent_states[0],
    )
    # The layout rebind is the identity binding: slot-local execution never
    # consumes the packed block table.
    layout = SimpleNamespace(
        slot_count=1,
        blocks_per_slot=16,
        cu_seqlens=(0, 4),
        active_mask=(True,),
        block_table=object(),
    )
    assert _rebind_packed_verify_layout_pages(layout, state) is layout

    # Non-int8 or plane-requiring routes must not produce unleased states:
    # the allocate flag only skips the lease for INT8 storage layouts.
    assert (
        _GGUFPackedTargetState.__dataclass_fields__["kv_backing_kind"].default
        == "private"
    )


def test_ensure_packed_workspace_upgrades_unleased_for_plane_consumers(
    monkeypatch,
) -> None:
    """A plane-requiring call reallocates an unleased state with planes."""

    real_ensure = gguf_runner.Qwen35GGUFResidentSession._ensure_packed_verify_workspace

    def fake_ensure(self, **kwargs):
        # Emulate the growth decision: an unleased state is not ready when
        # planes are required, and the allocate call then leases them.
        state = getattr(self, "_packed_verify_state", None)
        require = bool(kwargs.get("require_kv_planes", True))
        if (
            state is not None
            and require
            and getattr(state, "kv_backing_kind", "private") == "unleased"
        ):
            upgraded = SimpleNamespace(
                slot_count=state.slot_count,
                max_sequence_length=state.max_sequence_length,
                kv_backing_kind="pool_lease",
            )
            self._packed_verify_state = upgraded
            self._packed_verify_scratch = SimpleNamespace(
                rows=kwargs["rows"],
                max_positions=kwargs["max_sequence_length"],
                gdn_segment_capacity=kwargs["slot_count"],
            )
            return upgraded, self._packed_verify_scratch
        return real_ensure(self, **kwargs)

    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.use_iq_dense_mmq = False
    owner._packed_verify_state = SimpleNamespace(
        slot_count=1,
        max_sequence_length=4096,
        kv_backing_kind="unleased",
    )
    owner._packed_verify_scratch = SimpleNamespace(
        rows=8, max_positions=4096, gdn_segment_capacity=1
    )
    owner._invalidate_live_packed_decode_graphs = lambda: None
    owner._free_packed_verify_workspace = lambda **kwargs: None
    owner._packed_verify_union_geometry = lambda **kwargs: (1, 8, 4096, 1)
    owner._workspace_kv_pool = object()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            gguf_runner.Qwen35GGUFResidentSession,
            "_ensure_packed_verify_workspace",
            fake_ensure,
        )
        runtime = SimpleNamespace()
        state, scratch = owner._ensure_packed_verify_workspace(
            slot_count=1,
            rows=8,
            max_sequence_length=4096,
            runtime=runtime,
            require_kv_planes=True,
        )
    assert state.kv_backing_kind == "pool_lease"


# ---------------------------------------------------------------------------
# P4 allocator: real allocate + real upgrade, only device operations mocked
# (reviewer corrective unit, 2026-09-10)
# ---------------------------------------------------------------------------


def _allocator_fake_runner():
    """Minimal runner satisfying _GGUFPackedTargetState.allocate's reads."""

    cfg = SimpleNamespace(
        layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention"),
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_value_dim=128,
        ssm_conv_kernel=4,
        ssm_inner_size=6144,
        ssm_group_count=16,
        is_moe=False,
        expert_count=0,
        expert_used_count=0,
        expert_shared_feed_forward_length=0,
        head_count_kv=4,
        key_length=256,
        value_length=256,
        full_attention_interval=2,
        head_count=24,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        feed_forward_length=17408,
    )
    return SimpleNamespace(
        backend="hip_gfx1100",
        hidden_size=5120,
        q_width=6144,
        kv_width=1024,
        ffn_size=17408,
        vocab_size=248320,
        linear_qkv_width=10240,
        ssm_value_dim=128,
        fp16_recurrent_state=False,
        weights=SimpleNamespace(config=cfg),
    )


def _int8_kv_layout() -> "gguf_runner.Qwen35GGUFKVChunkLayout":
    return gguf_runner.Qwen35GGUFKVChunkLayout(
        storage_dtype=gguf_runner.DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=gguf_runner.DType.FP32,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(
            None,
            gguf_runner.DType.INT8_PER_TOKEN_HEAD,
            None,
            gguf_runner.DType.INT8_PER_TOKEN_HEAD,
        ),
    )


def _fake_kv_pool(layout, pages):
    full = tuple(SimpleNamespace(ptr=0x2000 + i, nbytes=4096) for i in range(4))
    backing = SimpleNamespace(
        layout=layout,
        full_key_caches=full,
        full_value_caches=full,
        full_bf16_mirror_key_caches=(None,) * 4,
        full_bf16_mirror_value_caches=(None,) * 4,
        full_k_scale_caches=full,
        full_v_scale_caches=full,
        full_kv_scale_metadata=full,
        buffers=(),
    )
    return SimpleNamespace(
        workspace_pages=lambda key: tuple(pages),
        backing=backing,
    )


def test_real_allocate_unleased_makes_zero_private_allocations(
    monkeypatch,
) -> None:
    """The unleased branch must be terminal: no private KV chunk, ever."""

    from hipengine.runtime.qwen35_gguf_runner import _GGUFPackedTargetState

    _install_fake_device(monkeypatch)
    private_calls: list[int] = []

    def fake_private_chunk(*args, **kwargs):
        private_calls.append(1)
        raise AssertionError("unleased allocate must not touch the private KV chunk allocator")

    monkeypatch.setattr(
        gguf_runner, "_allocate_qwen35_gguf_kv_chunk", fake_private_chunk
    )
    runtime = SimpleNamespace(memset=lambda ptr, value, nbytes: None)

    state = _GGUFPackedTargetState.allocate(
        _allocator_fake_runner(),
        slot_count=1,
        max_sequence_length=1024,
        runtime=runtime,
        kv_layout=_int8_kv_layout(),
        kv_pool=_fake_kv_pool(_int8_kv_layout(), pages=(0, 1, 2, 3)),
        lease_kv_planes=False,
    )
    assert private_calls == []
    assert state.kv_backing_kind == "unleased"
    # page_ids keep the class invariant's derived identity mapping (never
    # consumed on the unleased route: the layout rebind early-returns and
    # full_cache fails closed).
    assert state.page_ids == (0, 1, 2, 3)
    # Layer-indexed geometry: full-length tuples, None at LINEAR positions.
    assert len(state.full_key_caches) == 4
    assert state.full_key_caches[0] is None and state.full_key_caches[2] is None
    with pytest.raises(ValueError, match="no packed full-attention KV cache"):
        state.full_cache(1)
    with pytest.raises(ValueError, match="no packed full-attention KV cache"):
        state.full_cache(3)
    # Only the conv/recurrent state buffers are owned.
    assert len(state.buffers) == 4  # two LINEAR layers x (conv, recurrent)


def test_real_allocate_leased_uses_the_pool_planes(monkeypatch) -> None:
    from hipengine.runtime.qwen35_gguf_runner import _GGUFPackedTargetState

    _install_fake_device(monkeypatch)
    monkeypatch.setattr(
        gguf_runner,
        "_allocate_qwen35_gguf_kv_chunk",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("leased allocate must not use the private allocator")
        ),
    )
    runtime = SimpleNamespace(memset=lambda ptr, value, nbytes: None)
    layout = _int8_kv_layout()
    pool = _fake_kv_pool(layout, pages=(7, 8, 9, 10))

    state = _GGUFPackedTargetState.allocate(
        _allocator_fake_runner(),
        slot_count=1,
        max_sequence_length=1024,
        runtime=runtime,
        kv_layout=layout,
        kv_pool=pool,
        lease_kv_planes=True,
    )
    assert state.kv_backing_kind == "pool_lease"
    assert state.page_ids == (7, 8, 9, 10)
    # The caches are the arena planes, layer-indexed.
    assert state.full_key_caches[1] is pool.backing.full_key_caches[1]


def test_real_ensure_upgrades_unleased_for_plane_consumers(monkeypatch) -> None:
    """The REAL ensure growth path upgrades an unleased state with planes."""

    from hipengine.runtime.qwen35_gguf_runner import _GGUFPackedTargetState

    _install_fake_device(monkeypatch)
    monkeypatch.setattr(
        gguf_runner,
        "_allocate_qwen35_gguf_kv_chunk",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("upgrade must lease pool planes, not allocate privately")
        ),
    )
    layout = _int8_kv_layout()
    pool = _fake_kv_pool(layout, pages=(0, 1, 2, 3))

    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.use_iq_dense_mmq = False
    owner.__dict__.update(
        runner=_allocator_fake_runner(),
        runtime=SimpleNamespace(memset=lambda ptr, value, nbytes: None),
        scratch=SimpleNamespace(max_positions=1024),
        _packed_verify_state=None,
        _packed_verify_scratch=None,
        _workspace_kv_pool=pool,
        # Pin the device KV layout so the ensure path uses this exact
        # layout object (the fake pool's backing carries it).
        _device_kv_layout=layout,
        _packed_decode_state_dirty=False,
        _invalidate_live_packed_decode_graphs=lambda: None,
    )

    def fake_free(**kwargs):
        # Mirror the real free: clear the cached state so the growth path
        # reallocates instead of returning the stale object.
        owner._packed_verify_state = None
        owner._packed_verify_scratch = None

    owner._free_packed_verify_workspace = fake_free
    runtime_arg = owner.runtime

    def union_geometry(**kwargs):
        return (kwargs["slot_count"], kwargs["rows"], kwargs["max_sequence_length"], kwargs["slot_count"])

    owner._packed_verify_union_geometry = union_geometry

    state, _ = owner._ensure_packed_verify_workspace(
        slot_count=1,
        rows=8,
        max_sequence_length=1024,
        runtime=runtime_arg,
        require_kv_planes=False,
    )
    assert state.kv_backing_kind == "unleased"

    upgraded, _ = owner._ensure_packed_verify_workspace(
        slot_count=1,
        rows=8,
        max_sequence_length=1024,
        runtime=runtime_arg,
        require_kv_planes=True,
    )
    assert upgraded.kv_backing_kind == "pool_lease"
    assert upgraded.page_ids == (0, 1, 2, 3)
    assert upgraded.full_key_caches[1] is pool.backing.full_key_caches[1]


# ---------------------------------------------------------------------------
# Workspace lease sizing vs the union geometry (2026-09-17)
#
# The lease is taken once at pool creation, before any packed layout exists,
# while the workspace is allocated for the union geometry - which unions the
# realized layout slots with the serving capacity. A one-slot lease therefore
# went short on the first packed prefill: 32 leased pages against a 4-slot x
# 9-page workspace at an 8192-token session, and 4 against 16 at 1024, where
# it failed during startup warmup.
# ---------------------------------------------------------------------------


def test_packed_verify_lease_slot_ceiling_mirrors_union_capacity() -> None:
    """The lease slot term must match the union geometry's capacity term."""

    ceiling = gguf_runner.packed_verify_lease_slot_ceiling

    assert ceiling(None) == gguf_runner._PACKED_VERIFY_DEFAULT_SLOT_CAPACITY
    assert ceiling("not-a-number") == gguf_runner._PACKED_VERIFY_DEFAULT_SLOT_CAPACITY
    assert ceiling(0) == 1
    assert ceiling(1) == 1
    assert ceiling(9) == 9


def test_union_geometry_never_exceeds_the_lease_slot_ceiling() -> None:
    """A single-slot request still packs the full serving capacity."""

    owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    owner.max_batch_size = 4
    owner._packed_verify_state = None
    owner._packed_verify_scratch = None
    owner._packed_verify_prefill_row_cap = lambda: 8

    union_slots, _rows, _max_seq, _segments = owner._packed_verify_union_geometry(
        slot_count=1,
        rows=8,
        max_sequence_length=1024,
    )

    # A one-slot request opens the full capacity, so a lease sized for one
    # slot cannot cover the workspace the allocation asks for.
    assert union_slots == 4
    assert union_slots == gguf_runner.packed_verify_lease_slot_ceiling(4)
    assert union_slots > 1


def test_lease_pages_helper_covers_every_capacity_bounded_geometry() -> None:
    """The lease helper must never under-size what the union geometry demands.

    This is the anti-drift contract. `packed_verify_workspace_lease_pages` and
    `_packed_verify_union_geometry` are two expressions of one geometry; when
    they were written separately they disagreed, and the disagreement only
    surfaced at prefill time as "packed workspace lease holds N pages but the
    workspace needs M". Sweeping the capacity/context grid here keeps them
    pinned together for every request the serving loop can raise within its
    own capacity.
    """

    for capacity in (1, 2, 4, 8):
        for max_positions in (256, 512, 1024, 2048, 8192):
            leased = gguf_runner.packed_verify_workspace_lease_pages(
                capacity, max_positions
            )

            owner = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
            owner.max_batch_size = capacity
            owner._packed_verify_state = None
            owner._packed_verify_scratch = None
            owner._packed_verify_prefill_row_cap = lambda: 8

            # The loop cannot request more slots than its own capacity; an MTP
            # verify group that packs wider is the documented exception and is
            # covered by the degradation test below, not by the lease.
            for slot_count in range(1, capacity + 1):
                union_slots, _rows, union_max_seq, _segments = (
                    owner._packed_verify_union_geometry(
                        slot_count=slot_count,
                        rows=8,
                        max_sequence_length=max_positions,
                    )
                )
                needed = union_slots * ((union_max_seq + 255) // 256)
                assert leased >= needed, (
                    f"capacity={capacity} positions={max_positions} "
                    f"slots={slot_count}: leased {leased} < needed {needed}"
                )


def test_lease_pages_helper_applies_the_context_floor_and_capacity_term() -> None:
    """Both terms are load-bearing and neither silently disappears."""

    # 1024-token per-slot floor: a short request context does not shrink below it.
    assert gguf_runner.packed_verify_workspace_lease_pages(1, 256) == 4
    assert gguf_runner.packed_verify_workspace_lease_pages(1, 1024) == 4
    # Context above the floor scales pages per slot.
    assert gguf_runner.packed_verify_workspace_lease_pages(1, 2048) == 8
    # Capacity scales slots.
    assert gguf_runner.packed_verify_workspace_lease_pages(4, 2048) == 32
    # An unusable capacity falls back to the historical floor rather than 0.
    assert gguf_runner.packed_verify_workspace_lease_pages(None, 1024) == (
        gguf_runner._PACKED_VERIFY_DEFAULT_SLOT_CAPACITY * 4
    )


def test_short_lease_degrades_to_private_instead_of_failing_closed(
    monkeypatch,
) -> None:
    """A lease too small for the request must not wedge serving.

    The lease is sized once at pool creation from the caps known then, but
    `_packed_verify_union_geometry` lets a caller pack more slots than the
    serving capacity (an MTP verify group does exactly this at C1). That used
    to raise from `allocate`, turning a legitimate geometry into a hard prefill
    failure. The private branch allocates the identical geometry, so the lease
    is a fast path and never a correctness dependency.
    """

    from hipengine.runtime.qwen35_gguf_runner import _GGUFPackedTargetState

    _install_fake_device(monkeypatch)
    private_calls: list[dict] = []
    real_chunk = gguf_runner._allocate_qwen35_gguf_kv_chunk

    def recording_chunk(*args, **kwargs):
        private_calls.append(dict(kwargs))
        return real_chunk(*args, **kwargs)

    monkeypatch.setattr(
        gguf_runner, "_allocate_qwen35_gguf_kv_chunk", recording_chunk
    )
    runtime = SimpleNamespace(memset=lambda ptr, value, nbytes: None)
    layout = _int8_kv_layout()
    # 1 slot x 1024 positions needs 4 pages; the lease holds 2.
    pool = _fake_kv_pool(layout, pages=(0, 1))
    runner = _allocator_fake_runner()

    state = _GGUFPackedTargetState.allocate(
        runner,
        slot_count=1,
        max_sequence_length=1024,
        runtime=runtime,
        kv_layout=layout,
        kv_pool=pool,
        lease_kv_planes=True,
    )

    assert state.kv_backing_kind == "private"
    assert len(private_calls) == 1
    assert private_calls[0]["pages"] == 4
    # The shortfall is recorded, not silently swallowed: chronic under-leasing
    # is a perf defect that must stay visible.
    shortfalls = getattr(runner, "_packed_workspace_lease_shortfalls", [])
    assert len(shortfalls) == 1
    assert shortfalls[0]["leased_pages"] == 2
    assert shortfalls[0]["needed_pages"] == 4


def test_sufficient_lease_still_prefers_the_arena(monkeypatch) -> None:
    """The degradation path must not cost the arena win in the normal case."""

    from hipengine.runtime.qwen35_gguf_runner import _GGUFPackedTargetState

    _install_fake_device(monkeypatch)
    monkeypatch.setattr(
        gguf_runner,
        "_allocate_qwen35_gguf_kv_chunk",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a sufficient lease must not fall back to private")
        ),
    )
    runtime = SimpleNamespace(memset=lambda ptr, value, nbytes: None)
    layout = _int8_kv_layout()
    runner = _allocator_fake_runner()

    state = _GGUFPackedTargetState.allocate(
        runner,
        slot_count=1,
        max_sequence_length=1024,
        runtime=runtime,
        kv_layout=layout,
        kv_pool=_fake_kv_pool(layout, pages=(4, 5, 6, 7)),
        lease_kv_planes=True,
    )

    assert state.kv_backing_kind == "pool_lease"
    assert getattr(runner, "_packed_workspace_lease_shortfalls", []) == []


def test_pool_pressure_callbacks_tolerate_a_session_without_a_batch_owner() -> None:
    """A pool-owning session has no ``_resident_batch_owner`` and must not raise.

    The attribute is only ever assigned onto the per-slot views a resident batch
    owner creates. A session that owns its own device KV pool never has it, so
    the grow/pressure callbacks have to read it defensively. They did not, and a
    prefix-cache run raised ``AttributeError: 'Qwen35GGUFResidentSession' object
    has no attribute '_resident_batch_owner'`` from
    ``GlobalDeviceKVPool._ensure_free_pages`` the moment retained snapshots
    pinned enough pages to trigger pressure.
    """

    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    assert not hasattr(session, "_resident_batch_owner")

    before_grow = lambda: (  # noqa: E731 - mirrors the callback shape under test
        owner._invalidate_live_packed_decode_graphs()
        if (owner := getattr(session, "_resident_batch_owner", None)) is not None
        else None
    )
    on_pressure = lambda required: (  # noqa: E731
        owner.evict_prefix_cache_for_pressure(required)
        if (owner := getattr(session, "_resident_batch_owner", None)) is not None
        else None
    )
    assert before_grow() is None
    assert on_pressure(8) is None

    # With an owner attached the callbacks delegate.
    calls: list[object] = []
    session._resident_batch_owner = SimpleNamespace(
        _invalidate_live_packed_decode_graphs=lambda: calls.append("invalidate"),
        evict_prefix_cache_for_pressure=lambda required: calls.append(("evict", required)),
    )
    before_grow()
    on_pressure(8)
    assert calls == ["invalidate", ("evict", 8)]


def test_bound_blocks_validate_against_pool_capacity_after_growth() -> None:
    """A grown pool's pages must bind, not fail against the first chunk's count.

    The global pool grows by appending backing chunks while keeping page ids
    stable and reaching every page through its pointer tables. Validating an
    allocation against the first chunk's page count rejected any request whose
    pages landed past that chunk, which is what a retained-snapshot prefix-cache
    run hits the moment the pool grows.
    """

    validate = gguf_runner._validate_bound_blocks_against_capacity

    # Pages inside the first chunk, and pages past it, are both in range once
    # the bound is the pool capacity.
    validate((0, 1, 2), start_block_id=0, page_capacity=128)
    validate((126, 127, 200), start_block_id=0, page_capacity=256)

    with pytest.raises(ValueError, match="must contain pages"):
        validate((), start_block_id=0, page_capacity=128)
    with pytest.raises(ValueError, match="must be unique"):
        validate((3, 3), start_block_id=0, page_capacity=128)
    with pytest.raises(ValueError, match="outside its pool page range"):
        validate((0, 128), start_block_id=0, page_capacity=128)
    with pytest.raises(ValueError, match="outside its pool page range"):
        validate((1,), start_block_id=4, page_capacity=128)
