"""Depth-wave GDN/Conv state adapter for one mixed target row layout.

These tests pin the adapter contract, not the kernels: the two indexed owners and
every device copy are replaced with recorders, so what is asserted is which rows
reach which dispatch, which slot each row's state came from, that the live state
owner is never written during verification, and that a rejected call mutates
nothing. No GPU, no real allocation, no kernel build.

Coverage: ragged chains, AR peers, topological physical permutations, every
selected accepted depth, the root state source mapping, live-state immutability,
invalid layouts failing before mutation, and retry after a failed cleanup.
"""

from __future__ import annotations

import pytest

import hipengine.runtime.qwen4_exp_mixed_gdn as mixed_gdn
from hipengine.core.memory import DeviceBuffer
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.qwen4_exp_batch_gdn import Qwen4ExpBatchGdnState
from hipengine.runtime.qwen4_exp_mixed_gdn import (
    Qwen4ExpMixedGdnAdapter,
    Qwen4ExpMixedGdnGeometry,
    Qwen4ExpMixedGdnPlan,
    Qwen4ExpMixedGdnProjectedRows,
    Qwen4ExpMixedGdnWave,
    Qwen4ExpMixedGdnWaveScratch,
    plan_qwen4_exp_mixed_gdn,
)
from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedRequest,
    compile_mixed_layout,
)

LIVE_SLOTS = 8
K_HEADS = 2
V_HEADS = 4
HEAD_DIM = 128
KERNEL = 4
QKV_WIDTH = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM
CORE_WIDTH = V_HEADS * HEAD_DIM
SCALAR_WIDTH = V_HEADS

# Semantic rows are request 1 depths 0..2, request 2 depth 0, request 3 depths
# 0..1. This permutation interleaves the requests and keeps every chain ascending.
PERMUTATION = (3, 0, 4, 1, 5, 2)


# -- recorders ---------------------------------------------------------------


class Timeline:
    """One ordered event log shared by the fake runtime and the dispatch recorders."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def add(self, event: tuple) -> None:
        self.events.append(event)


class FakeRuntime:
    """Records allocation, memset, copy and free traffic without a device."""

    def __init__(
        self,
        *,
        timeline: Timeline,
        fail_free_call: int | None = None,
        fail_copy_call: int | None = None,
    ) -> None:
        self.timeline = timeline
        self.next_ptr = 0x400000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.memsets: list[tuple[int, int]] = []
        self.copies: list[tuple[int, int, int, int]] = []
        self.fail_free_call = fail_free_call
        self.fail_copy_call = fail_copy_call
        self.malloc_calls = 0
        self.free_calls = 0
        self.copy_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.free_calls += 1
        if self.fail_free_call == self.free_calls:
            raise RuntimeError("simulated device free failure")
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations
        self.memsets.append((dst, int(nbytes)))

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        self.copy_calls += 1
        if self.fail_copy_call == self.copy_calls:
            raise RuntimeError("simulated device copy failure")
        assert self._owns(int(dst), int(nbytes)), (
            "destination must be an owned device range"
        )
        self.copies.append((dst, int(src), int(nbytes), int(kind)))
        if int(kind) == int(MemcpyKind.DEVICE_TO_DEVICE):
            self.timeline.add(("d2d", dst, int(src), int(nbytes)))
        else:
            self.timeline.add(("h2d", dst, int(nbytes)))

    def _owns(self, ptr: int, nbytes: int) -> bool:
        """Whether ``[ptr, ptr + nbytes)`` lies inside one allocation.

        Copies target whole slabs and single slots inside them, so the check is a
        range check rather than an exact allocation base match.
        """

        return any(
            start <= ptr and ptr + nbytes <= start + size
            for start, size in self.allocations.items()
        )


@pytest.fixture
def timeline() -> Timeline:
    return Timeline()


@pytest.fixture
def runtime(timeline: Timeline) -> FakeRuntime:
    return FakeRuntime(timeline=timeline)


@pytest.fixture
def dispatches(monkeypatch, timeline: Timeline) -> Timeline:
    def conv(*args, **kwargs):
        timeline.add(("conv", kwargs["rows"], args, kwargs))

    def recurrence(*args, **kwargs):
        timeline.add(("recurrence", kwargs["rows"], args, kwargs))

    monkeypatch.setattr(
        mixed_gdn, "qwen4_exp_linear_attn_conv_decode_indexed_f32", conv
    )
    monkeypatch.setattr(
        mixed_gdn, "qwen4_exp_gdn_decode_rowsplit_indexed_f32", recurrence
    )
    return timeline


@pytest.fixture
def live_state(runtime: FakeRuntime):
    state = Qwen4ExpBatchGdnState.allocate(
        state_slots=LIVE_SLOTS,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
        runtime=runtime,
    )
    try:
        yield state
    finally:
        state.close()


# -- fixtures ----------------------------------------------------------------


def _request(
    request_id: int,
    *,
    slot: int,
    start: int = 2,
    root: int = 10,
    candidates=(),
) -> Qwen4ExpMixedRequest:
    return Qwen4ExpMixedRequest(
        request_id=request_id,
        slot=slot,
        owner_token=request_id + 100,
        start_position=start,
        root_token=root,
        candidates=tuple(candidates),
    )


def _layout(*, physical_row_order=None):
    """Request 1 speculates two tokens, request 2 is an AR peer, 3 one token."""

    return compile_mixed_layout(
        [
            _request(1, slot=5, start=2, root=10, candidates=(11, 12)),
            _request(2, slot=0, start=40, root=20),
            _request(3, slot=7, start=9, root=30, candidates=(31,)),
        ],
        physical_row_order=physical_row_order,
    )


def _geometry() -> Qwen4ExpMixedGdnGeometry:
    return Qwen4ExpMixedGdnGeometry(
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
    )


def _projected(rows: int) -> Qwen4ExpMixedGdnProjectedRows:
    return Qwen4ExpMixedGdnProjectedRows(
        qkv=DeviceBuffer(ptr=0x900000, nbytes=rows * QKV_WIDTH * 4),
        gate=DeviceBuffer(ptr=0xA00000, nbytes=rows * CORE_WIDTH * 4),
        alpha=DeviceBuffer(ptr=0xB00000, nbytes=rows * SCALAR_WIDTH * 4),
        beta=DeviceBuffer(ptr=0xC00000, nbytes=rows * SCALAR_WIDTH * 4),
    )


def _adapter(live_state, runtime, *, physical_row_order=None) -> Qwen4ExpMixedGdnAdapter:
    return Qwen4ExpMixedGdnAdapter.allocate(
        layout=_layout(physical_row_order=physical_row_order),
        live_state=live_state,
        runtime=runtime,
    )


def _owner_tokens(layout) -> dict[int, int]:
    """The batch owner's current slot -> owner token map for one layout."""

    return {request.slot: request.owner_token for request in layout.requests}


@pytest.fixture
def adapter(live_state, runtime):
    layout = _layout(physical_row_order=PERMUTATION)
    owners = _owner_tokens(layout)
    built = Qwen4ExpMixedGdnAdapter.allocate(
        layout=layout,
        live_state=live_state,
        runtime=runtime,
        owner_of=owners.get,
    )
    try:
        yield built
    finally:
        built.close()


KERNEL_POINTERS = dict(
    conv_weight_ptr=0x910,
    dt_bias_ptr=0x920,
    a_ptr=0x930,
    norm_weight_ptr=0x940,
)


def _live_state_writes(runtime: FakeRuntime, live: Qwen4ExpBatchGdnState) -> list[tuple]:
    """Copies whose destination lies inside the live owner's state slabs."""

    slabs = (
        (live.recurrent_state.ptr, live.recurrent_state.nbytes),
        (live.conv_state.ptr, live.conv_state.nbytes),
    )
    return [
        copy
        for copy in runtime.copies
        if any(
            start <= copy[0] and copy[0] + copy[2] <= start + size
            for start, size in slabs
        )
    ]


def _recurrent_slot(state: Qwen4ExpBatchGdnState, slot: int) -> int:
    return state.recurrent_state.ptr + slot * state.recurrent_bytes_per_slot


def _conv_slot(state: Qwen4ExpBatchGdnState, slot: int) -> int:
    return state.conv_state.ptr + slot * state.conv_bytes_per_slot


def _expected_run_events(
    adapter: Qwen4ExpMixedGdnAdapter,
    projected: Qwen4ExpMixedGdnProjectedRows,
) -> list[tuple]:
    """The exact copy/dispatch order one pass must produce."""

    plan = adapter.plan
    live = adapter.live_state
    working = adapter.working_state
    events: list[tuple] = []
    for request in plan.layout.requests:
        root = plan.snapshot_row(request.request_id, 0)
        events.append(
            (
                "d2d",
                _recurrent_slot(working, root),
                _recurrent_slot(live, request.slot),
                working.recurrent_bytes_per_slot,
            )
        )
        events.append(
            (
                "d2d",
                _conv_slot(working, root),
                _conv_slot(live, request.slot),
                working.conv_bytes_per_slot,
            )
        )
    widths = (
        ("qkv", QKV_WIDTH),
        ("gate", CORE_WIDTH),
        ("alpha", SCALAR_WIDTH),
        ("beta", SCALAR_WIDTH),
    )
    for wave in plan.waves:
        if wave.depth:
            for row, parent in zip(wave.rows, wave.parent_rows, strict=True):
                events.append(
                    (
                        "d2d",
                        _recurrent_slot(working, row),
                        _recurrent_slot(working, parent),
                        working.recurrent_bytes_per_slot,
                    )
                )
                events.append(
                    (
                        "d2d",
                        _conv_slot(working, row),
                        _conv_slot(working, parent),
                        working.conv_bytes_per_slot,
                    )
                )
        for index, physical in enumerate(wave.rows):
            for name, width in widths:
                events.append(
                    (
                        "d2d",
                        adapter.wave.row_ptr(name, index),
                        getattr(projected, name).ptr + physical * width * 4,
                        width * 4,
                    )
                )
        events.append(("h2d", working.indices.ptr, wave.width * 8))
        events.append(("conv", wave.width))
        events.append(("recurrence", wave.width))
        for index, physical in enumerate(wave.rows):
            events.append(
                (
                    "d2d",
                    adapter.core.ptr + physical * CORE_WIDTH * 4,
                    adapter.wave.row_ptr("core", index),
                    CORE_WIDTH * 4,
                )
            )
    return events


# -- wave plan ---------------------------------------------------------------


def test_plan_puts_one_chain_row_per_wave_and_keeps_ar_peers_in_wave_zero(
    live_state,
) -> None:
    plan = plan_qwen4_exp_mixed_gdn(
        _layout(physical_row_order=PERMUTATION),
        geometry=_geometry(),
        live_state_slots=live_state.state_slots,
    )
    assert isinstance(plan, Qwen4ExpMixedGdnPlan)
    assert [wave.depth for wave in plan.waves] == [0, 1, 2]
    assert [wave.width for wave in plan.waves] == [3, 2, 1]
    assert plan.max_wave_rows == 3
    assert plan.state_slots == plan.total_rows == 6
    assert plan.request_width == 3
    # Wave 0 holds every request once, ordered by physical position; the AR peer
    # contributes no other row.
    assert plan.wave(0).request_ids == (2, 1, 3)
    assert plan.wave(1).request_ids == (1, 3)
    assert plan.wave(2).request_ids == (1,)
    assert plan.wave(0).rows == (0, 1, 2)
    assert plan.wave(1).rows == (3, 4)
    assert plan.wave(2).rows == (5,)
    # Roots read the live slot; deeper rows read their parent's snapshot row.
    assert plan.wave(0).parent_rows == (-1, -1, -1)
    assert plan.wave(1).parent_rows == (1, 2)
    assert plan.wave(2).parent_rows == (3,)
    # Provisional slot == physical row, so a snapshot is addressable by row.
    for wave in plan.waves:
        assert wave.slots == wave.rows
    assert plan.live_slot(1) == 5
    assert plan.live_slot(2) == 0
    assert plan.live_slot(3) == 7
    assert plan.snapshot_row(1, 2) == 5
    assert plan.snapshot_row(3, 0) == 2
    with pytest.raises(KeyError, match="depths 0..2"):
        plan.wave(3)


@pytest.mark.parametrize(
    "physical_row_order",
    [None, PERMUTATION, (0, 3, 1, 4, 2, 5)],
)
def test_topological_physical_permutations_keep_the_same_chain_rows(
    live_state, physical_row_order
) -> None:
    layout = _layout(physical_row_order=physical_row_order)
    plan = plan_qwen4_exp_mixed_gdn(
        layout, geometry=_geometry(), live_state_slots=live_state.state_slots
    )
    # Depth membership is a property of the chains, never of physical placement;
    # the order inside a wave follows physical position instead.
    assert [set(wave.request_ids) for wave in plan.waves] == [
        {1, 2, 3},
        {1, 3},
        {1},
    ]
    for wave in plan.waves:
        assert wave.rows == tuple(sorted(wave.rows))
        assert len(set(wave.rows)) == wave.width
        for request, row, parent in zip(
            wave.request_ids, wave.rows, wave.parent_rows, strict=True
        ):
            assert row == layout.physical_row(request, wave.depth)
            if wave.depth:
                assert parent == layout.physical_row(request, wave.depth - 1)
                assert parent < row
            else:
                assert parent == -1
    # Every physical row is claimed exactly once across the waves.
    assert sorted(row for wave in plan.waves for row in wave.rows) == list(
        range(plan.total_rows)
    )


def test_wave_bound_is_the_indexed_row_bound(live_state) -> None:
    layout = compile_mixed_layout(
        [_request(request_id, slot=request_id) for request_id in range(65)]
    )
    with pytest.raises(ValueError, match=r"1\.\.64"):
        plan_qwen4_exp_mixed_gdn(layout, geometry=_geometry(), live_state_slots=65)


def test_plan_rejects_geometry_the_primitives_cannot_address() -> None:
    layout = _layout()
    cases = [
        (
            dict(num_k_heads=4, num_v_heads=6, head_dim=HEAD_DIM, conv_kernel=KERNEL),
            "divisible",
        ),
        (
            dict(num_k_heads=K_HEADS, num_v_heads=V_HEADS, head_dim=64, conv_kernel=KERNEL),
            "head_dim 128",
        ),
        (
            dict(num_k_heads=K_HEADS, num_v_heads=V_HEADS, head_dim=HEAD_DIM, conv_kernel=0),
            "conv_kernel",
        ),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            Qwen4ExpMixedGdnGeometry(**kwargs)
    with pytest.raises(ValueError, match="live state slab"):
        plan_qwen4_exp_mixed_gdn(layout, geometry=_geometry(), live_state_slots=7)
    with pytest.raises(ValueError, match="conv state slab width"):
        plan_qwen4_exp_mixed_gdn(layout, geometry=_geometry(), live_state_slots=4097)
    with pytest.raises(TypeError, match="Qwen4ExpMixedLayout"):
        plan_qwen4_exp_mixed_gdn(
            "layout", geometry=_geometry(), live_state_slots=LIVE_SLOTS
        )
    with pytest.raises(ValueError, match="positive"):
        plan_qwen4_exp_mixed_gdn(layout, geometry=_geometry(), live_state_slots=0)


def test_plan_rejects_waves_that_do_not_match_its_layout(live_state) -> None:
    plan = plan_qwen4_exp_mixed_gdn(
        _layout(), geometry=_geometry(), live_state_slots=live_state.state_slots
    )
    with pytest.raises(ValueError, match="do not match this layout's rows"):
        Qwen4ExpMixedGdnPlan(
            layout=plan.layout,
            geometry=plan.geometry,
            live_state_slots=plan.live_state_slots,
            waves=plan.waves[:1],
        )
    with pytest.raises(ValueError, match="distinct slot"):
        Qwen4ExpMixedGdnWave(
            depth=0, request_ids=(1, 2), rows=(0, 1), parent_rows=(-1, -1), slots=(0, 0)
        )
    with pytest.raises(ValueError, match="ascending"):
        Qwen4ExpMixedGdnWave(
            depth=0, request_ids=(1, 2), rows=(1, 0), parent_rows=(-1, -1), slots=(1, 0)
        )
    with pytest.raises(ValueError, match="roots"):
        Qwen4ExpMixedGdnWave(
            depth=0, request_ids=(1,), rows=(0,), parent_rows=(0,), slots=(0,)
        )


# -- one pass ----------------------------------------------------------------


def test_run_copies_each_root_from_its_requests_live_slot(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    runtime.copies.clear()
    runtime.timeline.events.clear()
    adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)

    roots = {
        request.request_id: adapter.plan.snapshot_row(request.request_id, 0)
        for request in adapter.plan.layout.requests
    }
    assert roots == {1: 1, 2: 0, 3: 2}
    root_copies = runtime.copies[: 2 * len(roots)]
    expected = []
    for request in adapter.plan.layout.requests:
        root = roots[request.request_id]
        expected.append(
            (
                _recurrent_slot(adapter.working_state, root),
                _recurrent_slot(adapter.live_state, request.slot),
                adapter.working_state.recurrent_bytes_per_slot,
                int(MemcpyKind.DEVICE_TO_DEVICE),
            )
        )
        expected.append(
            (
                _conv_slot(adapter.working_state, root),
                _conv_slot(adapter.live_state, request.slot),
                adapter.working_state.conv_bytes_per_slot,
                int(MemcpyKind.DEVICE_TO_DEVICE),
            )
        )
    assert root_copies == expected


def test_run_dispatches_one_wave_per_depth_over_the_provisional_slab(
    adapter: Qwen4ExpMixedGdnAdapter, dispatches: Timeline
) -> None:
    output = adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
    assert output is adapter.core
    assert output.nbytes == adapter.plan.total_rows * CORE_WIDTH * 4

    names = [event[0] for event in dispatches.events if event[0] in ("conv", "recurrence")]
    assert names == [
        "conv",
        "recurrence",
        "conv",
        "recurrence",
        "conv",
        "recurrence",
    ]
    assert [
        event[1] for event in dispatches.events if event[0] in ("conv", "recurrence")
    ] == [3, 3, 2, 2, 1, 1]

    convs = [event for event in dispatches.events if event[0] == "conv"]
    recurrences = [event for event in dispatches.events if event[0] == "recurrence"]
    for wave, (_, rows, args, kwargs), (_, gdn_rows, gdn_args, gdn_kwargs) in zip(
        adapter.plan.waves, convs, recurrences, strict=True
    ):
        assert rows == gdn_rows == wave.width
        # The wave scratch is the operand block: rows 0..width-1 of one dispatch.
        assert args[0] == adapter.wave.qkv.ptr
        assert args[1] == adapter.working_state.conv_state.ptr
        assert args[2] == KERNEL_POINTERS["conv_weight_ptr"]
        assert args[3] == adapter.wave.conv.ptr
        assert args[4] == adapter.working_state.indices.ptr
        assert kwargs["state_slots"] == adapter.plan.state_slots
        assert kwargs["channels"] == QKV_WIDTH
        assert kwargs["kernel_size"] == KERNEL
        assert kwargs["runtime"] is adapter.runtime

        assert gdn_args[0] == adapter.wave.conv.ptr
        assert gdn_args[1] == adapter.wave.gate.ptr
        assert gdn_args[2] == adapter.wave.alpha.ptr
        assert gdn_args[3] == adapter.wave.beta.ptr
        assert gdn_args[7] == adapter.working_state.recurrent_state.ptr
        assert gdn_args[8] == adapter.wave.core.ptr
        assert gdn_args[9] == adapter.working_state.indices.ptr
        assert gdn_kwargs["num_k_heads"] == K_HEADS
        assert gdn_kwargs["num_v_heads"] == V_HEADS
        assert gdn_kwargs["head_k_dim"] == HEAD_DIM
        assert gdn_kwargs["state_slots"] == adapter.plan.state_slots


def test_run_order_pins_snapshot_and_gather_copies_before_each_wave(
    adapter: Qwen4ExpMixedGdnAdapter, dispatches: Timeline
) -> None:
    projected = _projected(adapter.plan.total_rows)
    dispatches.events.clear()
    adapter.run(projected, **KERNEL_POINTERS)
    # Dispatch arguments are pinned by the previous test; here the wave width and
    # the copy before it are what the order assertion reads.
    recorded = [
        (event[0], event[1]) if event[0] in ("conv", "recurrence") else event
        for event in dispatches.events
    ]
    assert recorded == _expected_run_events(adapter, projected)


def test_a_repeated_pass_reproduces_the_same_copies(
    adapter: Qwen4ExpMixedGdnAdapter, dispatches: Timeline
) -> None:
    projected = _projected(adapter.plan.total_rows)
    dispatches.events.clear()
    adapter.run(projected, **KERNEL_POINTERS)
    first = list(dispatches.events)
    dispatches.events.clear()
    adapter.run(projected, **KERNEL_POINTERS)
    # Every snapshot is written from a validated source before it is read, so a
    # second pass needs no reset and starts from the same live bytes.
    assert dispatches.events == first


def test_verification_never_writes_the_live_state(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    live = adapter.live_state
    indices_before = live.indices_host.copy()
    # Ignore the setup uploads the two owners performed while allocating.
    runtime.copies.clear()
    dispatches.events.clear()
    adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)

    # No copy targets the live owner, and no dispatch is handed its slabs.
    assert _live_state_writes(runtime, live) == []
    for event in dispatches.events:
        if event[0] in ("conv", "recurrence"):
            pointers = set(event[2])
            assert live.recurrent_state.ptr not in pointers
            assert live.conv_state.ptr not in pointers
            assert live.indices.ptr not in pointers
    # The live row set was not re-prepared either: admission is the caller's.
    assert live.active is None
    assert live.indices_host.tolist() == indices_before.tolist()

    # Commit is the only path that touches live state: two blocks per request.
    runtime.copies.clear()
    adapter.commit([2, 0, 1])
    assert len(_live_state_writes(runtime, live)) == 2 * len(
        adapter.plan.layout.requests
    )


# -- commit ------------------------------------------------------------------


@pytest.mark.parametrize(
    "accepted",
    [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 0, 1),
        (2, 0, 1),
    ],
)
def test_commit_copies_the_selected_snapshot_for_each_accepted_depth(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline, accepted
) -> None:
    adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
    runtime.copies.clear()
    plan = adapter.commit(list(accepted))

    assert plan.accepted_counts == accepted
    assert plan.commit_rows == tuple(
        adapter.plan.snapshot_row(request.request_id, count)
        for request, count in zip(adapter.plan.layout.requests, accepted, strict=True)
    )
    expected = []
    for request, count in zip(adapter.plan.layout.requests, accepted, strict=True):
        snapshot = adapter.plan.snapshot_row(request.request_id, count)
        expected.append(
            (
                _recurrent_slot(adapter.live_state, request.slot),
                _recurrent_slot(adapter.working_state, snapshot),
                adapter.live_state.recurrent_bytes_per_slot,
                int(MemcpyKind.DEVICE_TO_DEVICE),
            )
        )
        expected.append(
            (
                _conv_slot(adapter.live_state, request.slot),
                _conv_slot(adapter.working_state, snapshot),
                adapter.live_state.conv_bytes_per_slot,
                int(MemcpyKind.DEVICE_TO_DEVICE),
            )
        )
    assert runtime.copies == expected


def test_commit_rejects_counts_before_any_copy(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
    runtime.copies.clear()
    cases = [
        ((3, 0, 0), ValueError, "prepares only 2"),
        ((0, 1, 0), ValueError, "autoregressive"),
        ((1, 0), ValueError, "one count per request"),
        ((-1, 0, 0), ValueError, "non-negative"),
    ]
    for counts, error, match in cases:
        with pytest.raises(error, match=match):
            adapter.commit(counts)
        assert runtime.copies == [], counts
    # A rejected count is the same transaction, not a consumed one.
    assert adapter.verification_pending is True
    adapter.commit([0, 0, 0])
    assert adapter.verification_pending is False


def test_commit_requires_a_completed_pass(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime
) -> None:
    runtime.copies.clear()
    assert adapter.verification_pending is False
    with pytest.raises(RuntimeError, match="no completed verification pass"):
        adapter.commit([0, 0, 0])
    assert runtime.copies == []


def test_failed_commit_is_not_repeatable(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    projected = _projected(adapter.plan.total_rows)
    adapter.run(projected, **KERNEL_POINTERS)
    assert adapter.verification_pending is True

    # Fail the third snapshot block: two requests' worth of live state lands.
    runtime.copies.clear()
    runtime.fail_copy_call = runtime.copy_calls + 3
    with pytest.raises(RuntimeError, match="device copy failure"):
        adapter.commit([2, 0, 1])
    applied = len(runtime.copies)
    assert applied == 2
    # The transaction is consumed, so the partial application cannot be replayed
    # as if it were still pending.
    assert adapter.verification_pending is False
    runtime.fail_copy_call = None
    with pytest.raises(RuntimeError, match="no completed verification pass"):
        adapter.commit([2, 0, 1])
    assert len(runtime.copies) == applied

    # Recovery is a fresh pass, which rebuilds every snapshot from live state.
    adapter.run(projected, **KERNEL_POINTERS)
    assert adapter.verification_pending is True
    adapter.commit([2, 0, 1])
    assert adapter.verification_pending is False


def test_failed_pass_is_not_committable(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime
) -> None:
    runtime.fail_copy_call = runtime.copy_calls + 2
    with pytest.raises(RuntimeError, match="device copy failure"):
        adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
    runtime.fail_copy_call = None
    assert adapter.verification_pending is False
    runtime.copies.clear()
    with pytest.raises(RuntimeError, match="no completed verification pass"):
        adapter.commit([0, 0, 0])
    assert runtime.copies == []


# -- admission epoch ---------------------------------------------------------


def test_a_reused_slot_is_rejected_before_any_copy(
    live_state, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    layout = _layout(physical_row_order=PERMUTATION)
    owners = _owner_tokens(layout)
    adapter = Qwen4ExpMixedGdnAdapter.allocate(
        layout=layout, live_state=live_state, runtime=runtime, owner_of=owners.get
    )
    projected = _projected(adapter.plan.total_rows)
    try:
        adapter.run(projected, **KERNEL_POINTERS)
        adapter.commit([2, 0, 1])
        runtime.copies.clear()
        dispatches.events.clear()

        # Slot 5 is released and handed to another request after the layout was
        # compiled: request 1's rows would read and commit the new occupant's
        # state.
        owners[5] = 999
        with pytest.raises(ValueError, match="slot 5 is owned by token 999"):
            adapter.run(projected, **KERNEL_POINTERS)
        assert runtime.copies == []
        assert dispatches.events == []
        with pytest.raises(ValueError, match="slot 5 is owned by token 999"):
            adapter.commit([0, 0, 0])
        assert runtime.copies == []

        # A released slot reads as unowned rather than as the old admission.
        owners[5] = 0
        with pytest.raises(ValueError, match="slot 5 is unowned now"):
            adapter.run(projected, **KERNEL_POINTERS)
        assert runtime.copies == []

        # The check is per request, not just for the first one seen.
        owners[5] = 101
        owners[7] = 404
        with pytest.raises(ValueError, match="slot 7 is owned by token 404"):
            adapter.run(projected, **KERNEL_POINTERS)
        assert runtime.copies == []

        # Restoring the admission restores the pass.
        owners[7] = 103
        adapter.run(projected, **KERNEL_POINTERS)
        adapter.commit([2, 0, 1])
    finally:
        adapter.close()


def test_stale_slot_is_rejected_before_allocation(live_state, runtime: FakeRuntime) -> None:
    before = runtime.malloc_calls
    with pytest.raises(ValueError, match="slot 5 is unowned now"):
        Qwen4ExpMixedGdnAdapter.allocate(
            layout=_layout(physical_row_order=PERMUTATION),
            live_state=live_state,
            runtime=runtime,
            owner_of=lambda slot: 0,
        )
    assert runtime.malloc_calls == before
    with pytest.raises(TypeError, match="owner_of must be callable"):
        Qwen4ExpMixedGdnAdapter.allocate(
            layout=_layout(physical_row_order=PERMUTATION),
            live_state=live_state,
            runtime=runtime,
            owner_of=42,
        )
    assert runtime.malloc_calls == before
    with pytest.raises(TypeError, match="owner token must be an integer"):
        Qwen4ExpMixedGdnAdapter.allocate(
            layout=_layout(physical_row_order=PERMUTATION),
            live_state=live_state,
            runtime=runtime,
            owner_of=lambda slot: None,
        )
    assert runtime.malloc_calls == before


def test_without_an_owner_callback_the_adapter_trusts_the_layout(
    live_state, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    adapter = _adapter(live_state, runtime, physical_row_order=PERMUTATION)
    try:
        assert adapter.owner_of is None
        adapter.require_owner_epoch()
        adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
        adapter.commit([0, 0, 0])
    finally:
        adapter.close()


# -- rejection and lifecycle -------------------------------------------------


def test_invalid_layouts_fail_before_any_allocation(
    live_state, runtime: FakeRuntime
) -> None:
    # A live owner narrower than the layout's widest request slot.
    narrow = Qwen4ExpBatchGdnState.allocate(
        state_slots=5,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
        runtime=runtime,
    )
    try:
        cases = [
            (
                dict(layout=_layout(), live_state=narrow, runtime=runtime),
                ValueError,
                "live state slab",
            ),
            (
                dict(layout="layout", live_state=live_state, runtime=runtime),
                TypeError,
                "Qwen4ExpMixedLayout",
            ),
            (
                dict(layout=_layout(), live_state=object(), runtime=runtime),
                TypeError,
                "Qwen4ExpBatchGdnState",
            ),
            (
                dict(layout=_layout(), live_state=live_state, runtime=object()),
                ValueError,
                "runtime must be the state owner",
            ),
        ]
        before = runtime.malloc_calls
        for kwargs, error, match in cases:
            with pytest.raises(error, match=match):
                Qwen4ExpMixedGdnAdapter.allocate(**kwargs)
            assert runtime.malloc_calls == before, match
    finally:
        narrow.close()


def test_head_geometry_the_recurrence_cannot_run_is_rejected_first(runtime) -> None:
    narrow = Qwen4ExpBatchGdnState.allocate(
        state_slots=LIVE_SLOTS,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=64,
        conv_kernel=KERNEL,
        runtime=runtime,
    )
    try:
        before = runtime.malloc_calls
        with pytest.raises(ValueError, match="head_dim 128"):
            Qwen4ExpMixedGdnAdapter.allocate(
                layout=_layout(), live_state=narrow, runtime=runtime
            )
        assert runtime.malloc_calls == before
    finally:
        narrow.close()


def test_run_rejects_bad_input_before_mutating(
    adapter: Qwen4ExpMixedGdnAdapter, runtime: FakeRuntime, dispatches: Timeline
) -> None:
    rows = adapter.plan.total_rows
    small = _projected(rows)
    undersized = Qwen4ExpMixedGdnProjectedRows(
        qkv=DeviceBuffer(ptr=0x900000, nbytes=rows * QKV_WIDTH * 4),
        gate=DeviceBuffer(ptr=0xA00000, nbytes=(rows - 1) * CORE_WIDTH * 4),
        alpha=small.alpha,
        beta=small.beta,
    )
    cases = [
        (undersized, dict(KERNEL_POINTERS), ValueError, "projected gate rows"),
        (small, dict(KERNEL_POINTERS, conv_weight_ptr=0), ValueError, "conv_weight_ptr"),
        (small, dict(KERNEL_POINTERS, stream=-1), ValueError, "stream"),
        (object(), dict(KERNEL_POINTERS), TypeError, "Qwen4ExpMixedGdnProjectedRows"),
    ]
    dispatches.events.clear()
    for projected, kwargs, error, match in cases:
        with pytest.raises(error, match=match):
            adapter.run(projected, **kwargs)
        assert dispatches.events == [], match

    adapter.closing = True
    with pytest.raises(RuntimeError, match="closing"):
        adapter.run(small, **KERNEL_POINTERS)
    assert dispatches.events == []
    adapter.closing = False

    adapter.live_state.closing = True
    with pytest.raises(RuntimeError, match="live"):
        adapter.run(small, **KERNEL_POINTERS)
    assert dispatches.events == []
    adapter.live_state.closing = False


def test_closed_owners_are_rejected_before_mutating(
    adapter: Qwen4ExpMixedGdnAdapter, dispatches: Timeline
) -> None:
    projected = _projected(adapter.plan.total_rows)
    dispatches.events.clear()
    adapter.wave.close()
    with pytest.raises(RuntimeError, match="wave scratch"):
        adapter.run(projected, **KERNEL_POINTERS)
    assert dispatches.events == []
    with pytest.raises(RuntimeError, match="wave scratch"):
        adapter.commit([0, 0, 0])
    assert dispatches.events == []


def test_close_retries_a_partial_failure_and_never_frees_live_state(
    timeline: Timeline,
) -> None:
    runtime = FakeRuntime(timeline=timeline, fail_free_call=1)
    live_state = Qwen4ExpBatchGdnState.allocate(
        state_slots=LIVE_SLOTS,
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel=KERNEL,
        runtime=runtime,
    )
    adapter = _adapter(live_state, runtime, physical_row_order=PERMUTATION)
    owned = sorted(
        [
            adapter.core.ptr,
            adapter.wave.qkv.ptr,
            adapter.wave.gate.ptr,
            adapter.wave.alpha.ptr,
            adapter.wave.beta.ptr,
            adapter.wave.conv.ptr,
            adapter.wave.core.ptr,
            adapter.working_state.recurrent_state.ptr,
            adapter.working_state.conv_state.ptr,
            adapter.working_state.indices.ptr,
        ]
    )
    live_owned = {
        live_state.recurrent_state.ptr,
        live_state.conv_state.ptr,
        live_state.indices.ptr,
    }
    assert sorted(runtime.allocations) == sorted(owned + sorted(live_owned))

    with pytest.raises(RuntimeError, match="device free failure"):
        adapter.close()
    assert adapter.closed is False
    assert adapter.closing is True
    # Execution is poisoned once cleanup has begun, even with nothing freed yet.
    with pytest.raises(RuntimeError, match="closing"):
        adapter.run(_projected(adapter.plan.total_rows), **KERNEL_POINTERS)
    assert sorted(runtime.allocations) == sorted(owned + sorted(live_owned))

    adapter.close()
    assert adapter.closed is True
    assert sorted(runtime.freed) == sorted(owned)
    # The borrowed live owner is untouched and still owns its slabs.
    assert sorted(runtime.allocations) == sorted(live_owned)
    adapter.close()
    assert sorted(runtime.freed) == sorted(owned)


def test_accessors_reject_rows_outside_the_layout(
    adapter: Qwen4ExpMixedGdnAdapter,
) -> None:
    assert adapter.core_row_ptr(5) == adapter.core.ptr + 5 * CORE_WIDTH * 4
    assert adapter.recurrent_snapshot_ptr(2) == adapter.working_state.recurrent_slot_ptr(2)
    assert adapter.conv_snapshot_ptr(4) == adapter.working_state.conv_slot_ptr(4)
    with pytest.raises(ValueError, match="outside"):
        adapter.core_row_ptr(adapter.plan.total_rows)
    with pytest.raises(TypeError, match="integer"):
        adapter.conv_snapshot_ptr(True)
    with pytest.raises(ValueError, match="outside"):
        adapter.wave.row_ptr("qkv", adapter.wave.rows)
    with pytest.raises(KeyError, match="unknown wave buffer"):
        adapter.wave.row_ptr("output", 0)


def test_wave_scratch_allocates_one_packed_block_per_operand(runtime: FakeRuntime) -> None:
    scratch = Qwen4ExpMixedGdnWaveScratch.allocate(
        rows=3,
        qkv_width=QKV_WIDTH,
        core_width=CORE_WIDTH,
        scalar_width=SCALAR_WIDTH,
        runtime=runtime,
    )
    try:
        assert scratch.rows == 3
        assert scratch.qkv.nbytes == 3 * QKV_WIDTH * 4
        assert scratch.gate.nbytes == 3 * CORE_WIDTH * 4
        assert scratch.alpha.nbytes == 3 * SCALAR_WIDTH * 4
        assert scratch.beta.nbytes == 3 * SCALAR_WIDTH * 4
        assert scratch.conv.nbytes == 3 * QKV_WIDTH * 4
        assert scratch.core.nbytes == 3 * CORE_WIDTH * 4
        assert scratch.row_ptr("qkv", 2) == scratch.qkv.ptr + 2 * QKV_WIDTH * 4
        assert scratch.row_ptr("alpha", 1) == scratch.alpha.ptr + SCALAR_WIDTH * 4
        assert scratch.width("core") == CORE_WIDTH
        # No stray zero-fill: the wave is fully gathered before a dispatch.
        assert runtime.memsets == []
    finally:
        scratch.close()
    assert sorted(runtime.freed) == sorted(
        [scratch.qkv.ptr, scratch.gate.ptr, scratch.alpha.ptr, scratch.beta.ptr,
         scratch.conv.ptr, scratch.core.ptr]
    )
    with pytest.raises(ValueError, match="positive"):
        Qwen4ExpMixedGdnWaveScratch.allocate(
            rows=0,
            qkv_width=QKV_WIDTH,
            core_width=CORE_WIDTH,
            scalar_width=SCALAR_WIDTH,
            runtime=runtime,
        )
