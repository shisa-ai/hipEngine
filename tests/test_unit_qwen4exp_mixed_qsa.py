"""Depth-wave mixed QSA adapter: waves, transactions, and pre-write refusal.

The adapter's control flow is what these tests pin. Every device call the
committed mixer makes is a module-level name, so they are replaced with
recorders; the attention and index owners, their host mirrors, and their
validation contracts are used for real against a fake runtime, because those
contracts are what the adapter has to satisfy.

Covered here: ragged chains, autoregressive peers, a physical row permutation,
dense-to-sparse depth transitions inside one wave and across waves, root row
mapping, every accepted depth, cancellation and mid-pass failure rollback, a
stale ownership token refused before any write, unaffected slots outside the
layout, and retry-safe allocation and cleanup. No GPU work and no runner,
scheduler, or serving path is involved.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_batch_qsa as batch_qsa
import hipengine.runtime.qwen4_exp_batch_qsa_index as batch_index
from hipengine.core.memory import malloc
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedRequest,
    compile_mixed_layout,
)
from hipengine.runtime.qwen4_exp_mixed_qsa import (
    Qwen4ExpMixedQsaScratch,
    abort_qwen4_exp_mixed_qsa_attention,
    commit_qwen4_exp_mixed_qsa_attention,
    plan_qwen4_exp_mixed_qsa_waves,
    run_qwen4_exp_mixed_qsa_attention,
    validate_qwen4_exp_mixed_qsa_pass,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
)

SLOTS = 6
#: A slot no layout in this file uses, so "unaffected peer" is checkable.
PEER_SLOT = SLOTS - 1
MAX_POSITIONS = 16
BLOCK_SIZE = 256
HIDDEN = 8
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 4
ROTARY_DIM = 4
THETA = 100.0
Q_WIDTH = QUERY_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
INDEX_HEADS = 2
INDEX_DIM = 4
INDEX_ROTARY_DIM = 4
RATIO = 2
BUDGET = 3
#: ``output_stride = budget * ratio + ratio - 1``; a row is dense while its
#: post-append cursor // ratio is within the budget, i.e. while position <= 6.
STRIDE = BUDGET * RATIO + RATIO - 1
#: What the fake selection read-back hands the adapter for a sparse row.
SELECT_COUNT = 3
#: Largest position whose row is still dense (context ``STRIDE`` is the bound).
DENSE_POSITION_LIMIT = STRIDE - 1


class FakeRuntime:
    """Allocation bookkeeping only: no device work, no real pointers."""

    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, int, object]] = []
        self.malloc_calls = 0
        self.free_calls = 0
        self.fail_malloc_on_call: int | None = None
        self.fail_free_on_call: int | None = None

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_on_call == self.malloc_calls:
            raise RuntimeError("injected malloc failure")
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.free_calls += 1
        if self.fail_free_on_call == self.free_calls:
            raise RuntimeError("injected free failure")
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert dst in self.allocations

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        # Both sides of a gather are owned slabs, so check that each pointer
        # lands inside an allocation rather than on its base.
        def owned(pointer: int) -> bool:
            return any(
                base <= pointer < base + size
                for base, size in self.allocations.items()
            )

        assert owned(dst) or owned(src), (dst, src)
        self.copies.append((dst, src, int(nbytes), kind))

    def device_to_device(self) -> list[tuple[int, int, int, object]]:
        return [entry for entry in self.copies if entry[3] == MemcpyKind.DEVICE_TO_DEVICE]


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def stage(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def all(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for label, args, kwargs in self.calls if label == name]

    def one(self, name: str) -> tuple[tuple, dict]:
        matches = self.all(name)
        assert len(matches) == 1, f"expected one {name} dispatch, got {len(matches)}"
        return matches[0]

    def projection(self, slot: str) -> tuple[tuple, dict]:
        """The one projection dispatch for ``slot``."""

        matches = [
            (args, kwargs)
            for args, kwargs in self.all("projection")
            if args[0] == f"{slot}-weight"
        ]
        assert len(matches) == 1, f"expected one {slot} projection, got {len(matches)}"
        return matches[0]


def _write(dst_ptr: int, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    ctypes.memmove(dst_ptr, contiguous.ctypes.data, contiguous.nbytes)


def _fake_readback(dst_ptr, src, nbytes, *, runtime=None) -> None:
    """Give the index owner a deterministic selection for every sparse row.

    ``select_active`` reads ``int32[rows]`` counts then ``int64[rows, stride]``
    positions. The row count comes from the transfer size, which is enough here
    because the adapter runs one wave at a time and the tests assert the wave
    structure separately.
    """

    if nbytes > 0 and nbytes % (STRIDE * 8) == 0:
        rows = nbytes // (STRIDE * 8)
        positions = np.tile(
            np.arange(STRIDE, dtype=np.int64), (rows, 1)
        )
        _write(dst_ptr, positions)
        return
    rows = nbytes // 4
    _write(dst_ptr, np.full(rows, SELECT_COUNT, dtype=np.int32))


@pytest.fixture(autouse=True)
def recorder(monkeypatch) -> Recorder:
    record = Recorder()
    for name in (
        "qwen4_exp_qsa_split_norm_rope_rows_f32",
        "qwen4_exp_qsa_split_norm_mrope_rows_f32",
        "qwen4_exp_qsa_norm_rope_rows_f32",
        "qwen4_exp_qsa_norm_mrope_rows_f32",
        "qwen35_write_paged_kv_f32_batch_spans",
        "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32",
        "qwen4_exp_qsa_gate_context_f32",
    ):
        monkeypatch.setattr(batch_qsa, name, record.stage(name))
    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", record.stage("projection"))
    for name in (
        "qwen4_exp_qsa_scatter_index_key_device_position_f32",
        "qwen4_exp_qsa_pool_norm_rope_f32",
        "qwen4_exp_qsa_score_f32",
        "qwen4_exp_qsa_topk_expand_rows_f32_i64",
    ):
        monkeypatch.setattr(batch_index, name, record.stage(name))
    monkeypatch.setattr(batch_index, "copy_device_to_host", _fake_readback)
    return record


# -- layouts ----------------------------------------------------------------


def _request(
    request_id: int,
    *,
    slot: int,
    owner: int,
    start: int,
    root: int,
    candidates=(),
) -> Qwen4ExpMixedRequest:
    return Qwen4ExpMixedRequest(
        request_id=request_id,
        slot=slot,
        owner_token=owner,
        start_position=start,
        root_token=root,
        candidates=tuple(candidates),
    )


def _ragged_layout(*, physical_row_order=None):
    """Request 1: two candidates. Request 2: an AR peer. Request 3: one candidate.

    Distinct start positions and chain lengths, so every wave's row count, slot
    order and cursor advance is distinguishable.
    """

    return compile_mixed_layout(
        [
            _request(1, slot=0, owner=11, start=2, root=10, candidates=(11, 12)),
            _request(2, slot=1, owner=12, start=6, root=20),
            _request(3, slot=2, owner=13, start=1, root=30, candidates=(31,)),
        ],
        physical_row_order=physical_row_order,
    )


def _transition_layout():
    """Wave 0 holds one dense row and one sparse row; later waves are sparse.

    A row is dense while its post-append cursor // ratio is within the budget,
    so a chain starting at position 6 is dense at its root and sparse one row
    later, and a chain starting at position 7 is already sparse at its root.
    """

    return compile_mixed_layout(
        [
            _request(6, slot=1, owner=31, start=6, root=60, candidates=(61, 62, 63)),
            _request(7, slot=2, owner=32, start=7, root=70, candidates=(71,)),
        ]
    )


# -- fixtures ---------------------------------------------------------------


def _owners(runtime: FakeRuntime) -> tuple[Qwen4ExpBatchAttentionState, Qwen4ExpBatchQsaIndexState]:
    attention = Qwen4ExpBatchAttentionState.allocate(
        state_slots=SLOTS,
        max_positions=MAX_POSITIONS,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=runtime,
    )
    index = Qwen4ExpBatchQsaIndexState.allocate(
        attention_state=attention,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        compression_ratio=RATIO,
        block_budget=BUDGET,
        runtime=runtime,
    )
    return attention, index


def _weights() -> Qwen4ExpQSAMixerDeviceWeights:
    return Qwen4ExpQSAMixerDeviceWeights(
        projections={
            name: f"{name}-weight"
            for name in (
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "index_q",
                "index_k",
            )
        },
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )


class Fixture:
    def __init__(self, layout) -> None:
        self.runtime = FakeRuntime()
        self.layout = layout
        self.attention, self.index = _owners(self.runtime)
        self.shared = Qwen4ExpQSAScratch.allocate(
            rows=layout.total_rows,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            runtime=self.runtime,
        )
        self.mixed = malloc(layout.total_rows * HIDDEN * 4, runtime=self.runtime)
        self.scratch = Qwen4ExpMixedQsaScratch.allocate(
            layout=layout,
            attention_state=self.attention,
            index_state=self.index,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=self.runtime,
        )
        self.weights = _weights()
        self.owners = {request.slot: request.owner_token for request in layout.requests}
        # Committed prefix: every layout slot's index cursor sits at its
        # request's start position, and one unrelated peer keeps its own state.
        for request in layout.requests:
            self.index.cursor_host[request.slot] = request.start_position
            self.index.pooled_count_host[request.slot] = request.start_position // RATIO
        self.index.cursor_host[PEER_SLOT] = 5
        self.index.pooled_count_host[PEER_SLOT] = 2

    def owner_of(self, slot: int) -> int:
        return self.owners[slot]

    def cursors(self) -> dict[int, int]:
        return {
            request.slot: int(self.index.cursor_host[request.slot])
            for request in self.layout.requests
        }

    def pooled(self) -> dict[int, int]:
        return {
            request.slot: int(self.index.pooled_count_host[request.slot])
            for request in self.layout.requests
        }

    def peer(self) -> tuple[int, int]:
        return (
            int(self.index.cursor_host[PEER_SLOT]),
            int(self.index.pooled_count_host[PEER_SLOT]),
        )

    def close(self) -> None:
        self.scratch.close()
        self.shared.close()
        self.index.close()
        self.attention.close()


@pytest.fixture
def fixture() -> Fixture:
    built = Fixture(_ragged_layout())
    try:
        yield built
    finally:
        built.close()


@pytest.fixture
def transition_fixture() -> Fixture:
    built = Fixture(_transition_layout())
    try:
        yield built
    finally:
        built.close()


def _run(fixture: Fixture, **overrides):
    kwargs = dict(
        layout=fixture.layout,
        shared=fixture.shared,
        attention_state=fixture.attention,
        index_state=fixture.index,
        scratch=fixture.scratch,
        owner_of=fixture.owner_of,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        theta=THETA,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        index_rotary_dim=INDEX_ROTARY_DIM,
        runtime=fixture.runtime,
    )
    kwargs.update(overrides)
    weights = kwargs.pop("weights", fixture.weights)
    return run_qwen4_exp_mixed_qsa_attention(fixture.mixed.ptr, weights, **kwargs)


# -- wave structure ---------------------------------------------------------


def test_ragged_chains_split_into_depth_subsets_in_physical_order() -> None:
    layout = _ragged_layout()

    waves = plan_qwen4_exp_mixed_qsa_waves(layout)

    assert [wave.depth for wave in waves] == [0, 1, 2]
    # Semantic order here: r1d0, r1d1, r1d2, r2d0, r3d0, r3d1.
    assert [wave.physical_rows for wave in waves] == [(0, 3, 4), (1, 5), (2,)]
    assert [wave.rows for wave in waves] == [3, 2, 1]
    assert [wave.slot_ids for wave in waves] == [(0, 1, 2), (0, 2), (0,)]
    assert [wave.positions for wave in waves] == [(2, 6, 1), (3, 2), (4,)]
    assert [wave.request_ids for wave in waves] == [(1, 2, 3), (1, 3), (1,)]
    # Every wave is a strict subset of the previous one, so wave 0 is the widest.
    assert waves[0].rows == layout.physical_request_width
    assert len(waves) == layout.max_chain_rows


def test_root_wave_is_every_requests_root_row() -> None:
    built = Fixture(_ragged_layout())
    try:
        layout = built.layout
        plan = validate_qwen4_exp_mixed_qsa_pass(
            built.mixed.ptr,
            built.weights,
            layout=layout,
            shared=built.shared,
            attention_state=built.attention,
            index_state=built.index,
            scratch=built.scratch,
            owner_of=built.owner_of,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            theta=THETA,
            index_heads=INDEX_HEADS,
            index_dim=INDEX_DIM,
            index_rotary_dim=INDEX_ROTARY_DIM,
            runtime=built.runtime,
        )

        assert plan.root_rows == layout.verify_batch.root_rows
        assert plan.wave(0).positions == tuple(
            request.start_position for request in layout.requests
        )
        # Every root row is at its own request's start position, and the root
        # wave names each request exactly once.
        for row in layout.rows:
            if row.depth == 0:
                index = plan.wave(0).physical_rows.index(row.physical_row)
                assert plan.wave(0).positions[index] == row.position
                assert layout.request(row.request_id).start_position == row.position
        assert sorted(plan.wave(0).request_ids) == sorted(layout.request_ids)
    finally:
        built.close()


def test_physical_permutation_reorders_the_wave_not_the_row_metadata() -> None:
    # physical_row_order[p] is the semantic row at physical row p: the AR peer
    # moves to the front and every chain still ascends.
    layout = _ragged_layout(physical_row_order=(3, 0, 1, 4, 2, 5))

    waves = plan_qwen4_exp_mixed_qsa_waves(layout)

    assert [wave.physical_rows for wave in waves] == [(0, 1, 3), (2, 5), (4,)]
    assert [wave.slot_ids for wave in waves] == [(1, 0, 2), (0, 2), (0,)]
    assert [wave.positions for wave in waves] == [(6, 2, 1), (3, 2), (4,)]
    # The permutation never rewrites a row's request, depth, position or token.
    assert layout.physical_row(1, 0) == 1
    assert layout.physical_row(1, 1) == 2
    assert layout.physical_row(1, 2) == 4
    assert layout.physical_row(2, 0) == 0


def test_permuted_wave_gathers_and_scatters_at_physical_positions(
    recorder: Recorder,
) -> None:
    built = Fixture(_ragged_layout(physical_row_order=(3, 0, 1, 4, 2, 5)))
    try:
        run = _run(built)
        copies = built.runtime.device_to_device()
        row_bytes = HIDDEN * 4
        hidden = built.scratch.hidden_rows
        hidden_gathers = [
            entry for entry in copies
            if hidden.ptr <= entry[0] < hidden.ptr + hidden.nbytes
        ]
        # Wave 0 gathers physical rows 0, 1, 3 into wave rows 0, 1, 2: the wave
        # order is physical, and the permutation is read out of the layout.
        assert [entry[0] - hidden.ptr for entry in hidden_gathers[:3]] == [
            0,
            row_bytes,
            2 * row_bytes,
        ]
        assert [entry[1] - built.mixed.ptr for entry in hidden_gathers[:3]] == [
            0,
            row_bytes,
            3 * row_bytes,
        ]
        assert {entry[2] for entry in hidden_gathers} == {row_bytes}
        # Every wave scatters its gated rows to their own physical positions, and
        # the six rows of the pass are covered exactly once, in physical order.
        gated_row_bytes = Q_WIDTH * 4
        gated_rows = built.scratch.gated_rows
        scatters = [
            entry for entry in copies
            if gated_rows.ptr <= entry[0] < gated_rows.ptr + gated_rows.nbytes
        ]
        assert [
            (entry[0] - gated_rows.ptr) // gated_row_bytes for entry in scatters
        ] == [0, 1, 3, 2, 5, 4]
        assert [entry[2] for entry in scatters] == [gated_row_bytes] * 6
        assert sorted(
            (entry[0] - gated_rows.ptr) // gated_row_bytes for entry in scatters
        ) == list(range(built.layout.total_rows))
        assert run.wave_rows == (3, 2, 1)
        # One full-row projection after the last wave, reading the scattered gate
        # output and writing the pass result: the physical mapping above is what
        # makes its row order correct.
        (args, kwargs) = recorder.projection("attn_output")
        assert args[:6] == (
            "attn_output-weight",
            built.scratch.gated_rows.ptr,
            built.scratch.output.ptr,
            built.layout.total_rows,
            Q_WIDTH,
            HIDDEN,
        )
        assert run.output is built.scratch.output
    finally:
        built.close()


# -- the pass itself --------------------------------------------------------


def test_pass_gathers_every_wave_and_scatters_the_pass_result(
    fixture: Fixture, recorder: Recorder
) -> None:
    run = _run(fixture)

    assert run.waves == 3
    assert run.wave_rows == (3, 2, 1)
    assert run.rows == fixture.layout.total_rows == 6
    assert run.output is fixture.scratch.output
    # Five gathered slabs plus one gated scatter per row: 5 * 6 + 6 copies.
    assert len(fixture.runtime.device_to_device()) == 6 * 6
    # One attention stage per wave.
    assert len(recorder.all("qwen4_exp_qsa_split_norm_rope_rows_f32")) == 3
    assert [
        kwargs["rows"]
        for _, kwargs in recorder.all("qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32")
    ] == [3, 2, 1]
    # The caller's shared input stage already projected attn_q/k/v and index_k
    # over every physical row, and the pass projects attn_output once over every
    # row after the last wave. index_q is skipped: no wave holds a sparse row.
    assert len(recorder.all("projection")) == 1
    (args, _) = recorder.projection("attn_output")
    assert args[1] == fixture.scratch.gated_rows.ptr
    assert args[2] == fixture.scratch.output.ptr
    assert args[3] == fixture.layout.total_rows
    assert run.index_query_rows == (0, 0, 0)
    assert run.index_query_total == 0
    # Every wave appended exactly one index key per row.
    assert len(
        recorder.all("qwen4_exp_qsa_scatter_index_key_device_position_f32")
    ) == 6


def test_pass_publishes_each_wave_and_advances_every_chain(
    fixture: Fixture, monkeypatch
) -> None:
    published: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = fixture.attention.prepare

    def spy(slot_ids, positions, **kwargs):
        published.append((tuple(slot_ids), tuple(positions)))
        return original(slot_ids, positions, **kwargs)

    monkeypatch.setattr(fixture.attention, "prepare", spy)

    run = _run(fixture)

    assert published == [
        ((0, 1, 2), (2, 6, 1)),
        ((0, 2), (3, 2)),
        ((0,), (4,)),
    ]
    # Each request's cursor advanced by exactly its chain length: the root row
    # plus one row per candidate.
    assert fixture.cursors() == {0: 5, 1: 7, 2: 3}
    assert run.layout.request(1).start_position + run.layout.request(1).rows == 5
    # The attention owner ends on the last wave's row set.
    assert fixture.attention.active is not None
    assert fixture.attention.active.slot_ids == (0,)
    assert tuple(fixture.attention.position_host[:1]) == (4,)


def test_autoregressive_peer_advances_exactly_once(
    fixture: Fixture, recorder: Recorder
) -> None:
    peer = fixture.layout.request(2)

    run = _run(fixture)
    plan = commit_qwen4_exp_mixed_qsa_attention(
        run, (2, 0, 1), index_state=fixture.index, owner_of=fixture.owner_of
    )

    assert peer.is_autoregressive and peer.depth == 0
    assert fixture.cursors()[peer.slot] == peer.start_position + 1
    assert plan.accepted_counts == (2, 0, 1)
    assert plan.commit_positions == (4, 6, 2)
    assert plan.commit_rows == (2, 3, 5)


def test_dense_and_sparse_rows_share_a_wave_and_switch_by_depth(
    transition_fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    built = transition_fixture
    snapshots: list[tuple[int, list[int]]] = []
    staged = batch_qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32

    def spy(*args, **kwargs):
        # Which counts buffer the wave reads, and what the batch-wide staging
        # mirror holds while it does. A mixed wave stages host metadata; an
        # all-sparse wave reads the index owner's own device buffer instead.
        snapshots.append((args[4], built.scratch.selection.counts_host.tolist()))
        return staged(*args, **kwargs)

    monkeypatch.setattr(
        batch_qsa, "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32", spy
    )

    run = _run(built)

    # Wave 0 holds a dense root (position 6, so its selection is all 7 visible
    # positions) and a sparse root; both are attended by one rows dispatch, but
    # only the sparse row is scored and selected, so the wave stages host
    # metadata into the batch-wide buffer.
    assert run.wave_rows == (2, 2, 1, 1)
    wave0_mirror = snapshots[0][1]
    assert snapshots[0][0] == built.scratch.selection.counts.ptr
    assert wave0_mirror[:2] == [STRIDE, SELECT_COUNT]
    # From wave 1 on, every row is past the bound and selects, so each wave hands
    # the index owner's own counts buffer to the kernel and leaves the staging
    # mirror exactly as wave 0 left it.
    assert [ptr for ptr, _ in snapshots[1:]] == [built.index.selected_counts.ptr] * 3
    assert [counts for _, counts in snapshots[1:]] == [wave0_mirror] * 3
    # One index-query projection per wave, sized by the whole wave: the mixer's
    # own rule, which is what the adapter reports rather than replaces.
    assert run.index_query_rows == (2, 2, 1, 1)
    assert run.index_query_total == run.rows == 6
    # One index_q per wave plus the single pass-wide attn_output: the input
    # projections were the caller's.
    assert len(recorder.all("projection")) == 4 + 1
    (args, _) = recorder.projection("attn_output")
    assert args[1] == built.scratch.gated_rows.ptr
    assert args[3] == built.layout.total_rows == 6
    # One score dispatch per sparse row: 1 + 2 + 1 + 1, not one per row, and the
    # batched selection runs over the sparse subset of each wave.
    assert len(recorder.all("qwen4_exp_qsa_score_f32")) == 5
    assert [
        args[4] for args, _ in recorder.all("qwen4_exp_qsa_topk_expand_rows_f32_i64")
    ] == [1, 2, 1, 1]


def test_the_output_projection_runs_once_after_every_wave(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    """One full-row projection, after the last gather and gated scatter."""

    observed: list[int] = []
    projection = batch_qsa.launch_gguf_linear

    def spy(*args, **kwargs):
        if args[0] == "attn_output-weight":
            observed.append(len(fixture.runtime.device_to_device()))
        return projection(*args, **kwargs)

    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", spy)

    run = _run(fixture)

    assert len(observed) == 1, "one attn_output projection for the whole pass"
    # Five gathers plus one gated scatter for every row were already recorded, and
    # the projection adds none, so it is the last device work of the pass.
    assert observed[0] == 6 * 6
    assert len(fixture.runtime.device_to_device()) == 6 * 6
    assert run.output is fixture.scratch.output


def test_a_failed_output_projection_rolls_the_pass_back(
    fixture: Fixture, monkeypatch
) -> None:
    before_cursors = fixture.cursors()
    before_pooled = fixture.pooled()

    def boom(*args, **kwargs):
        if args[0] == "attn_output-weight":
            raise RuntimeError("injected projection failure")

    monkeypatch.setattr(batch_qsa, "launch_gguf_linear", boom)

    with pytest.raises(RuntimeError, match="injected projection failure"):
        _run(fixture)

    # The projection runs inside the pass transaction, so its failure withdraws
    # every appended chain row instead of leaving the pass half-committed.
    assert fixture.cursors() == before_cursors
    assert fixture.pooled() == before_pooled
    assert fixture.index.active is None
    assert fixture.index.pending_rollback is None


def test_an_autoregressive_only_pass_is_one_wave(recorder: Recorder) -> None:
    layout = compile_mixed_layout(
        [
            _request(10, slot=2, owner=100, start=3, root=40),
            _request(11, slot=0, owner=101, start=0, root=41),
        ]
    )
    built = Fixture(layout)
    try:
        assert plan_qwen4_exp_mixed_qsa_waves(layout) == (
            plan_qwen4_exp_mixed_qsa_waves(layout)[0],
        )
        run = _run(built)

        assert run.waves == 1
        assert run.wave_rows == (2,)
        assert run.index_query_rows == (0,)
        commit_qwen4_exp_mixed_qsa_attention(
            run, (0, 0), index_state=built.index, owner_of=built.owner_of
        )
        # Each peer advances exactly one token, so each cursor is its root
        # position plus one.
        assert built.cursors() == {2: 4, 0: 1}
    finally:
        built.close()


def test_pass_keeps_an_unrelated_slot_and_every_prefix(fixture: Fixture) -> None:
    before_peer = fixture.peer()
    before_prefixes = fixture.cursors()

    run = _run(fixture)
    commit_qwen4_exp_mixed_qsa_attention(
        run, (0, 0, 0), index_state=fixture.index, owner_of=fixture.owner_of
    )

    assert fixture.peer() == before_peer
    # A slot outside the layout never appears in a published row set, so its
    # cursor cannot move; the layout slots each advanced by their root row only.
    assert fixture.cursors() == {
        slot: start + 1 for slot, start in before_prefixes.items()
    }
    assert fixture.layout.request(2).slot != PEER_SLOT


# -- commit and abort -------------------------------------------------------


@pytest.mark.parametrize(
    "accepted, cursors, pooled",
    [
        # Every wave of this layout is dense, so no block was pooled during the
        # pass and the committed counter is the pre-pass one clamped by the
        # committed cursor's complete blocks.
        ((0, 0, 0), {0: 3, 1: 7, 2: 2}, {0: 1, 1: 3, 2: 0}),
        ((1, 0, 0), {0: 4, 1: 7, 2: 2}, {0: 1, 1: 3, 2: 0}),
        ((2, 0, 1), {0: 5, 1: 7, 2: 3}, {0: 1, 1: 3, 2: 0}),
    ],
)
def test_commit_publishes_exactly_the_accepted_depth(
    fixture: Fixture, accepted, cursors, pooled
) -> None:
    run = _run(fixture)

    plan = commit_qwen4_exp_mixed_qsa_attention(
        run,
        accepted,
        index_state=fixture.index,
        owner_of=fixture.owner_of,
        transaction_id=7,
    )

    assert plan.transaction_id == 7
    assert plan.accepted_counts == accepted
    assert fixture.cursors() == cursors
    assert fixture.pooled() == pooled
    # A committed pooled counter never claims a block that the committed cursor
    # does not cover, or a later step would read a rejected row's pooled key.
    for slot, cursor in cursors.items():
        assert fixture.pooled()[slot] <= cursor // RATIO
    # The destination is the last consumed row of each request, and no rejected
    # suffix row is visible past the committed cursor.
    for request, keep in zip(fixture.layout.requests, accepted):
        assert plan.commit_positions[fixture.layout.requests.index(request)] == (
            request.start_position + keep
        )
        assert fixture.cursors()[request.slot] == request.start_position + keep + 1
        assert fixture.cursors()[request.slot] <= (
            request.start_position + request.rows
        )
    assert fixture.index.active is None
    assert fixture.index.pending_rollback is None


def test_commit_clamps_a_pooled_block_that_spanned_a_rejected_row(
    transition_fixture: Fixture,
) -> None:
    built = transition_fixture
    run = _run(built)
    # The pass pooled every complete block of both chains.
    assert built.pooled() == {1: 5, 2: 4}

    commit_qwen4_exp_mixed_qsa_attention(
        run, (0, 0), index_state=built.index, owner_of=built.owner_of
    )

    # Request 6 keeps its root row only: cursor 7, three complete blocks, so the
    # blocks pooled from the rejected candidates must be pooled again later.
    assert built.cursors() == {1: 7, 2: 8}
    assert built.pooled() == {1: 3, 2: 4}
    assert built.peer() == (5, 2)


def test_abort_restores_every_cursor_and_pooled_counter(fixture: Fixture) -> None:
    before_cursors = fixture.cursors()
    before_pooled = fixture.pooled()
    before_peer = fixture.peer()

    run = _run(fixture)
    assert fixture.cursors() != before_cursors

    abort_qwen4_exp_mixed_qsa_attention(run, index_state=fixture.index)

    assert fixture.cursors() == before_cursors
    assert fixture.pooled() == before_pooled
    assert fixture.peer() == before_peer
    assert fixture.index.active is None
    assert fixture.index.pending_rollback is None
    with pytest.raises((ValueError, RuntimeError)):
        abort_qwen4_exp_mixed_qsa_attention(run, index_state=fixture.index)


def test_commit_after_abort_is_refused(fixture: Fixture) -> None:
    run = _run(fixture)
    abort_qwen4_exp_mixed_qsa_attention(run, index_state=fixture.index)
    before = fixture.cursors()

    with pytest.raises(RuntimeError, match="no longer pending"):
        commit_qwen4_exp_mixed_qsa_attention(
            run, (0, 0, 0), index_state=fixture.index, owner_of=fixture.owner_of
        )
    assert fixture.cursors() == before


def test_a_failed_wave_rolls_the_pass_back_and_can_be_retried(
    fixture: Fixture, recorder: Recorder, monkeypatch
) -> None:
    before_cursors = fixture.cursors()
    before_pooled = fixture.pooled()
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected wave failure")

    monkeypatch.setattr(batch_qsa, "qwen4_exp_qsa_gate_context_f32", boom)

    with pytest.raises(RuntimeError, match="injected wave failure"):
        _run(fixture)

    # Wave 0 appended its keys before the failure; the rollback withdrew them.
    assert calls["n"] == 2
    assert fixture.cursors() == before_cursors
    assert fixture.pooled() == before_pooled
    assert fixture.index.active is None
    assert fixture.index.pending_rollback is None

    monkeypatch.setattr(
        batch_qsa, "qwen4_exp_qsa_gate_context_f32", recorder.stage("gate")
    )
    run = _run(fixture)

    assert run.wave_rows == (3, 2, 1)
    assert fixture.cursors() == {0: 5, 1: 7, 2: 3}


def test_a_second_pass_refuses_while_the_token_is_pending(fixture: Fixture) -> None:
    run = _run(fixture)

    with pytest.raises(RuntimeError, match="pending rollback token"):
        _run(fixture)

    commit_qwen4_exp_mixed_qsa_attention(
        run, (0, 0, 0), index_state=fixture.index, owner_of=fixture.owner_of
    )
    # Resolving the token releases the owner for the next pass.
    fixture.index.cursor_host[[0, 1, 2]] = [2, 6, 1]
    _run(fixture)


# -- refusal before mutation ------------------------------------------------


def test_a_stale_owner_token_is_refused_before_any_write(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.runtime.copies.clear()
    stale = dict(fixture.owners)
    stale[1] = 99

    with pytest.raises(ValueError, match="stale for this slot"):
        _run(fixture, owner_of=lambda slot: stale[slot])

    assert fixture.runtime.copies == []
    assert recorder.calls == []
    assert fixture.cursors() == {0: 2, 1: 6, 2: 1}
    assert fixture.index.pending_rollback is None
    assert fixture.attention.active is None


def test_owner_of_must_return_an_int(fixture: Fixture) -> None:
    tokens = dict(fixture.owners)
    tokens[0] = np.int64(11)

    with pytest.raises(TypeError, match="an owner token is an int"):
        _run(fixture, owner_of=lambda slot: tokens[slot])
    with pytest.raises(TypeError, match="owner_of must be a callable"):
        _run(fixture, owner_of=None)


def test_commit_rechecks_ownership_and_leaves_the_token_usable(
    fixture: Fixture,
) -> None:
    run = _run(fixture)
    after_pass = fixture.cursors()
    stale = dict(fixture.owners)
    stale[2] = 77

    with pytest.raises(ValueError, match="stale for this slot"):
        commit_qwen4_exp_mixed_qsa_attention(
            run, (0, 0, 0), index_state=fixture.index, owner_of=lambda slot: stale[slot]
        )

    assert fixture.cursors() == after_pass
    assert fixture.index.pending_rollback is run.rollback
    abort_qwen4_exp_mixed_qsa_attention(run, index_state=fixture.index)
    assert fixture.cursors() == {0: 2, 1: 6, 2: 1}


def test_a_cursor_that_is_not_at_the_start_position_is_refused(
    fixture: Fixture, recorder: Recorder
) -> None:
    fixture.index.cursor_host[0] = 3

    with pytest.raises(ValueError, match="start position 2"):
        _run(fixture)

    assert recorder.calls == []
    assert fixture.index.pending_rollback is None


def test_a_stale_cursor_at_commit_time_is_refused(fixture: Fixture) -> None:
    run = _run(fixture)
    # Another caller appended to the slot after the pass.
    fixture.index.cursor_host[0] += 1

    with pytest.raises(ValueError, match="the run is stale"):
        commit_qwen4_exp_mixed_qsa_attention(
            run, (0, 0, 0), index_state=fixture.index, owner_of=fixture.owner_of
        )


def test_an_ar_peer_cannot_accept_a_candidate(fixture: Fixture) -> None:
    run = _run(fixture)
    after_pass = fixture.cursors()

    with pytest.raises(ValueError, match="advances exactly once"):
        commit_qwen4_exp_mixed_qsa_attention(
            run, (0, 1, 0), index_state=fixture.index, owner_of=fixture.owner_of
        )
    assert fixture.cursors() == after_pass


def test_commit_requires_the_owner_that_ran_the_pass(
    fixture: Fixture, recorder: Recorder
) -> None:
    run = _run(fixture)
    other_runtime = FakeRuntime()
    _, other = _owners(other_runtime)
    try:
        with pytest.raises(ValueError, match="different index owner"):
            commit_qwen4_exp_mixed_qsa_attention(
                run, (0, 0, 0), index_state=other, owner_of=fixture.owner_of
            )
        with pytest.raises(ValueError, match="different index owner"):
            abort_qwen4_exp_mixed_qsa_attention(run, index_state=other)
    finally:
        other.close()


def test_pass_refuses_geometry_and_owner_mismatches(fixture: Fixture) -> None:
    cases = [
        (dict(kv_heads=2), ValueError, "must match the attention state cache"),
        (dict(head_dim=HEAD_DIM + 1), ValueError, "must match the attention state cache"),
        (dict(index_dim=INDEX_DIM + 1), ValueError, "must match its state owner"),
        (dict(rotary_dim=3), ValueError, "rotary_dim must be even"),
        (dict(index_rotary_dim=3), ValueError, "index_rotary_dim must be even"),
        (dict(theta=0.0), ValueError, "theta must be a finite positive number"),
        (dict(eps=-1.0), ValueError, "eps must be a finite positive number"),
        (dict(hidden=0), ValueError, "hidden must be positive"),
        (dict(attention_context_limit=4), ValueError, "does not cover"),
        (dict(attention_context_limit=MAX_POSITIONS + 1), ValueError, "exceeds"),
        (dict(runtime=FakeRuntime()), ValueError, "runtime must match"),
    ]
    for overrides, error, match in cases:
        with pytest.raises(error, match=match):
            _run(fixture, **overrides)


def test_pass_refuses_a_mismatched_index_owner_and_closed_owners(
    fixture: Fixture,
) -> None:
    other_runtime = FakeRuntime()
    other_attention, other_index = _owners(other_runtime)
    try:
        with pytest.raises(ValueError, match="must mirror the batch attention state"):
            _run(fixture, index_state=other_index)
    finally:
        other_index.close()
        other_attention.close()

    fixture.scratch.close()
    with pytest.raises(RuntimeError, match="wave scratch is closed"):
        _run(fixture)


def test_pass_refuses_a_position_past_the_attention_capacity() -> None:
    layout = compile_mixed_layout(
        [_request(9, slot=0, owner=90, start=MAX_POSITIONS - 1, root=9, candidates=(1,))]
    )
    built = Fixture(layout)
    try:
        with pytest.raises(ValueError, match="outside the attention capacity"):
            _run(built)
        assert built.index.pending_rollback is None
    finally:
        built.close()


def test_pass_refuses_a_shared_scratch_that_covers_too_few_rows(
    fixture: Fixture,
) -> None:
    small = Qwen4ExpQSAScratch.allocate(
        rows=1,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        runtime=fixture.runtime,
    )
    try:
        with pytest.raises(ValueError, match="for 6 physical rows"):
            _run(fixture, shared=small)
        with pytest.raises(TypeError, match="no q_projected"):
            _run(fixture, shared=object())
    finally:
        small.close()


def test_pass_refuses_a_scratch_allocated_for_another_layout(
    fixture: Fixture,
) -> None:
    narrow = compile_mixed_layout([_request(8, slot=3, owner=80, start=0, root=8)])
    other = Qwen4ExpMixedQsaScratch.allocate(
        layout=narrow,
        attention_state=fixture.attention,
        index_state=fixture.index,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        runtime=fixture.runtime,
    )
    try:
        with pytest.raises(ValueError, match="result rows"):
            _run(fixture, scratch=other)
    finally:
        other.close()


def test_pass_refuses_missing_or_mis_shaped_weights(fixture: Fixture) -> None:
    missing = Qwen4ExpQSAMixerDeviceWeights(
        projections={name: "w" for name in ("attn_q", "attn_k", "attn_v")},
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )
    with pytest.raises(ValueError, match="missing Qwen4Exp QSA weights"):
        _run(fixture, weights=missing)

    zero_norm = Qwen4ExpQSAMixerDeviceWeights(
        projections=fixture.weights.projections,
        q_norm_weight_ptr=0,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0x908,
        index_k_norm_weight_ptr=0x90C,
    )
    with pytest.raises(ValueError, match="Q/K norm weights are required"):
        _run(fixture, weights=zero_norm)

    zero_index_norm = Qwen4ExpQSAMixerDeviceWeights(
        projections=fixture.weights.projections,
        q_norm_weight_ptr=0x900,
        k_norm_weight_ptr=0x904,
        index_q_norm_weight_ptr=0,
        index_k_norm_weight_ptr=0x90C,
    )
    with pytest.raises(ValueError, match="index Q/K norm weights are required"):
        _run(fixture, weights=zero_index_norm)

    from types import SimpleNamespace

    mis_shaped = dict(fixture.weights.projections)
    mis_shaped["attn_k"] = SimpleNamespace(
        spec=SimpleNamespace(source=SimpleNamespace(shape=(KV_WIDTH + 1, HIDDEN)))
    )
    with pytest.raises(ValueError, match="attn_k weight shape"):
        _run(
            fixture,
            weights=Qwen4ExpQSAMixerDeviceWeights(
                projections=mis_shaped,
                q_norm_weight_ptr=0x900,
                k_norm_weight_ptr=0x904,
                index_q_norm_weight_ptr=0x908,
                index_k_norm_weight_ptr=0x90C,
            ),
        )


def test_validate_writes_nothing(fixture: Fixture) -> None:
    fixture.runtime.copies.clear()
    before = fixture.cursors()

    plan = validate_qwen4_exp_mixed_qsa_pass(
        fixture.mixed.ptr,
        fixture.weights,
        layout=fixture.layout,
        shared=fixture.shared,
        attention_state=fixture.attention,
        index_state=fixture.index,
        scratch=fixture.scratch,
        owner_of=fixture.owner_of,
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        theta=THETA,
        index_heads=INDEX_HEADS,
        index_dim=INDEX_DIM,
        index_rotary_dim=INDEX_ROTARY_DIM,
        runtime=fixture.runtime,
    )

    assert fixture.runtime.copies == []
    assert fixture.cursors() == before
    assert fixture.index.pending_rollback is None
    assert plan.owners == (11, 12, 13)
    assert plan.q_width == Q_WIDTH and plan.kv_width == KV_WIDTH
    assert plan.max_wave_rows == 3
    assert plan.stride == STRIDE


def test_wave_metadata_rejects_a_repeated_slot() -> None:
    from hipengine.runtime.qwen4_exp_mixed_qsa import Qwen4ExpMixedQsaWave

    with pytest.raises(ValueError, match="names a slot twice"):
        Qwen4ExpMixedQsaWave(
            depth=0,
            request_ids=(1, 2),
            physical_rows=(0, 1),
            slot_ids=(0, 0),
            positions=(1, 2),
        )
    with pytest.raises(ValueError, match="physical row order"):
        Qwen4ExpMixedQsaWave(
            depth=0,
            request_ids=(1, 2),
            physical_rows=(1, 0),
            slot_ids=(0, 1),
            positions=(1, 2),
        )


# -- allocation and cleanup -------------------------------------------------


def test_scratch_close_is_retry_safe(fixture: Fixture) -> None:
    scratch = fixture.scratch
    owned = [getattr(scratch, name).ptr for name in (
        "hidden_rows", "q_projected", "key_projected", "value_projected",
        "index_q_projected", "index_k_projected", "index_query", "query", "key",
        "gate", "context", "gated", "gated_rows", "output",
    )] + [
        scratch.selection.positions.ptr,
        scratch.selection.counts.ptr,
        scratch.selection.queries.ptr,
    ]
    assert len(owned) == 17
    freed_before = len(fixture.runtime.freed)
    fixture.runtime.fail_free_on_call = fixture.runtime.free_calls + 3

    with pytest.raises(RuntimeError, match="injected free failure"):
        scratch.close()

    assert not scratch.closed
    assert scratch.closing
    # A partially closed scratch refuses to be used as a mixer scratch.
    with pytest.raises(RuntimeError, match="closing"):
        scratch._require_open()

    scratch.close()

    assert scratch.closed
    released = fixture.runtime.freed[freed_before:]
    assert sorted(released) == sorted(owned)
    assert len(released) == len(set(released))
    assert all(ptr not in fixture.runtime.allocations for ptr in owned)


def test_scratch_allocation_failure_frees_what_it_took() -> None:
    runtime = FakeRuntime()
    attention, index = _owners(runtime)
    layout = _ragged_layout()
    before = set(runtime.allocations)
    runtime.fail_malloc_on_call = runtime.malloc_calls + 5

    with pytest.raises(RuntimeError, match="injected malloc failure"):
        Qwen4ExpMixedQsaScratch.allocate(
            layout=layout,
            attention_state=attention,
            index_state=index,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=runtime,
        )

    assert set(runtime.allocations) == before
    assert len(runtime.freed) == 4
    index.close()
    attention.close()
    assert runtime.allocations == {}


def test_scratch_allocation_refuses_bad_geometry() -> None:
    runtime = FakeRuntime()
    attention, index = _owners(runtime)
    layout = _ragged_layout()
    cases = [
        (dict(hidden=0), ValueError, "hidden must be positive"),
        (dict(query_heads=3, kv_heads=2), ValueError, "divisible by kv_heads"),
        (dict(head_dim=512), ValueError, "dense decode bound"),
        (dict(kv_heads=2), ValueError, "must match the attention state cache"),
        (dict(runtime=FakeRuntime()), ValueError, "runtime must match"),
        (dict(layout=None), TypeError, "requires a Qwen4ExpMixedLayout"),
    ]
    for overrides, error, match in cases:
        kwargs = dict(
            layout=layout,
            attention_state=attention,
            index_state=index,
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            runtime=runtime,
        )
        kwargs.update(overrides)
        with pytest.raises(error, match=match):
            Qwen4ExpMixedQsaScratch.allocate(**kwargs)
    index.close()
    attention.close()


def test_scratch_exposes_the_mixer_scratch_contract(fixture: Fixture) -> None:
    scratch = fixture.scratch
    wave_widths = {
        "q_projected": Q_WIDTH * 2,
        "key_projected": KV_WIDTH,
        "value_projected": KV_WIDTH,
        "query": Q_WIDTH,
        "key": KV_WIDTH,
        "gate": Q_WIDTH,
        "context": Q_WIDTH,
        "gated": Q_WIDTH,
    }
    # One row per request: the widest wave, never the pass's row count.
    assert scratch.rows == fixture.layout.physical_request_width == 3
    assert scratch.total_rows == fixture.layout.total_rows == 6
    for name, width in wave_widths.items():
        assert getattr(scratch, name).nbytes == scratch.rows * width * 4, name
    # The two pass-wide slabs cover every physical row.
    assert scratch.gated_rows.nbytes == 6 * Q_WIDTH * 4
    assert scratch.output.nbytes == 6 * HIDDEN * 4
    assert scratch.selection.rows == scratch.rows
    assert scratch.selection.stride == STRIDE
    assert scratch.selection.index_heads == INDEX_HEADS
    assert isinstance(scratch.selection, Qwen4ExpBatchQsaSelectionScratch)
    # The output projection reads the scattered full-row gate output, never the
    # wave staging slab.
    view = scratch.output_scratch()
    assert view.gated is scratch.gated_rows
    assert view.output is scratch.output
    assert view.runtime is scratch.runtime
    assert view.gated.ptr != scratch.gated.ptr
    # One row of each: the wave staging slab is the gate stage's own.
    assert scratch.gated.nbytes == scratch.rows * Q_WIDTH * 4
