"""Chunk-sized bulk prefill hidden/token storage with monotonic owner growth.

The bulk prefill workspace used to allocate its BF16 hidden planes and INT64
token buffer at the INT8 lifetime plan's ``required_hidden_capacity``, i.e. at
the full context of the session, even on a deferred-KV serving owner whose
requests are streamed through chunk-outer prefill and never touch more than one
chunk's worth of rows. That is real resident memory charged for capacity a
request may never use.

Contract under test (CPU, fake device):

- ``_gguf_bulk_prefill_initial_hidden_capacity`` is the single storage policy:
  a non-DMS deferred-KV owner with a full-prompt lifetime plan starts
  chunk-sized; standalone, DMS, and expert-sidecar sessions keep the
  full-prompt capacity they always had.
- ``_gguf_bulk_prefill_hidden_plane_count`` is the single plane policy, shared
  by the initial allocation and by growth, so growing never changes the
  route-scoped alias decision.
- ``_ensure_bulk_prefill_hidden_capacity`` grows the *owner* monotonically,
  allocates replacements before publishing them, keeps the plane count, rebinds
  every borrowed slot view, updates the owner's ``_buffers`` ledger, and frees
  each distinct old buffer exactly once. A failed growth leaves the workspace
  untouched and leaks nothing.
- Full-prompt consumers grow before they address the storage: scalar bulk
  prefill before it reads the chunk-outer decision, packed layer-outer prefill
  before its first embedding launch, and a resumed layer-outer segment before
  it restores the checkpoint.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _GGUFPackedARPrefillChunk,
    _GGUFResumablePrefillState,
)

from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)

HIDDEN = 5_120
ROW_NBYTES = HIDDEN * 2
CAPACITY = 8_192
CHUNK_ROWS = 768


class _GrowthProbe(RuntimeError):
    """Sentinel proving a call site reached the growth step."""


def _fake_session(
    monkeypatch,
    *,
    capacity: int = CAPACITY,
    chunk_rows: int = CHUNK_ROWS,
    plan_mode: str | None = "layer_outer_shared_oracle",
    required_hidden_capacity: int | None = None,
    **overrides,
):
    allocations = _install_fake_device(monkeypatch)
    freed: list[object] = []
    monkeypatch.setattr(
        gguf_runner,
        "free",
        lambda buffer, *, runtime: freed.append(buffer),
    )
    runner = _fake_dense_qwen36_runner()
    runner.vocab_size = 248_320
    session = object.__new__(Qwen35GGUFResidentSession)
    session.__dict__.update(
        runner=runner,
        runtime=SimpleNamespace(),
        backend="hip_gfx1100",
        use_expert_sidecar=False,
        dms_prefill_mode="dense_pool",
        dms_metadata_path=None,
        defer_kv_allocation=True,
        _runtime_state_library=None,
        prefill_chunk_tuning={},
        _int8_prefill_lifetime_plan=SimpleNamespace(
            mode=plan_mode,
            required_hidden_capacity=(
                capacity
                if required_hidden_capacity is None
                else int(required_hidden_capacity)
            ),
        ),
        _buffers=(),
        _prefill_token_buf=None,
        _prefill_hidden_a=None,
        _prefill_hidden_b=None,
        _bulk_prefill_scratch=None,
        _prefill_flight_recorder=None,
    )
    session.scratch = SimpleNamespace(max_positions=capacity)
    session.__dict__.update(overrides)
    session._prefill_scratch_rows = lambda capacity: chunk_rows  # noqa: ARG001
    return session, freed, allocations


def _workspace_buffers(session) -> tuple:
    hidden = (
        (session._prefill_hidden_a,)
        if session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
        else (session._prefill_hidden_a, session._prefill_hidden_b)
    )
    return (session._prefill_token_buf, *hidden)


# --------------------------------------------------------------------------
# Pure policy
# --------------------------------------------------------------------------


def test_initial_capacity_is_chunk_sized_for_a_deferred_serving_owner() -> None:
    assert (
        gguf_runner._gguf_bulk_prefill_initial_hidden_capacity(
            plan_mode="layer_outer_shared_oracle",
            plan_required_hidden_capacity=CAPACITY,
            prefill_capacity=CAPACITY,
            chunk_rows=CHUNK_ROWS,
            defer_kv_allocation=True,
        )
        == CHUNK_ROWS
    )


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        (
            "standalone_session_allocates_kv_up_front",
            {"defer_kv_allocation": False},
        ),
        ("external_dms_serving", {"dms_serving": True}),
        ("dms_layer_outer", {"dms_prefill_mode": "layer_outer"}),
        ("expert_sidecar", {"use_expert_sidecar": True}),
    ],
)
def test_initial_capacity_keeps_full_prompt_storage_on_every_other_route(
    label: str, kwargs: dict
) -> None:
    capacity = gguf_runner._gguf_bulk_prefill_initial_hidden_capacity(
        plan_mode="layer_outer_shared_oracle",
        plan_required_hidden_capacity=CAPACITY,
        prefill_capacity=CAPACITY,
        chunk_rows=CHUNK_ROWS,
        **{"defer_kv_allocation": True, **kwargs},
    )
    assert capacity == CAPACITY, label


def test_initial_capacity_leaves_chunk_plans_alone() -> None:
    """A chunk-outer plan already asks for chunk-sized storage; nothing moves."""

    assert (
        gguf_runner._gguf_bulk_prefill_initial_hidden_capacity(
            plan_mode="chunk_outer_layer_local_oracles",
            plan_required_hidden_capacity=CHUNK_ROWS,
            prefill_capacity=CAPACITY,
            chunk_rows=CHUNK_ROWS,
            defer_kv_allocation=True,
        )
        == CHUNK_ROWS
    )


def test_initial_capacity_rejects_impossible_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        gguf_runner._gguf_bulk_prefill_initial_hidden_capacity(
            plan_mode="layer_outer_shared_oracle",
            plan_required_hidden_capacity=0,
            prefill_capacity=CAPACITY,
            chunk_rows=CHUNK_ROWS,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        gguf_runner._gguf_bulk_prefill_initial_hidden_capacity(
            plan_mode="layer_outer_shared_oracle",
            plan_required_hidden_capacity=CAPACITY,
            prefill_capacity=CHUNK_ROWS,
            chunk_rows=CAPACITY,
        )


def test_plane_count_follows_the_route_scoped_alias_gates(monkeypatch) -> None:
    monkeypatch.delenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, raising=False)
    assert (
        gguf_runner._gguf_bulk_prefill_hidden_plane_count(
            plan_mode="layer_outer_shared_oracle",
            dms_prefill_mode="dense_pool",
        )
        == 1
    )
    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "0")
    assert (
        gguf_runner._gguf_bulk_prefill_hidden_plane_count(
            plan_mode="layer_outer_shared_oracle",
            dms_prefill_mode="dense_pool",
        )
        == 2
    )
    # The geometry policy can remove a plane; the route gates never add one back.
    assert (
        gguf_runner._gguf_bulk_prefill_hidden_plane_count(
            plan_mode="chunk_outer_layer_local_oracles",
            dms_prefill_mode="dense_pool",
            geometry_plane_count=1,
        )
        == 1
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_deferred_serving_owner_constructs_chunk_sized_hidden_and_tokens(
    monkeypatch,
) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())

    assert session._prefill_hidden_a.nbytes == CHUNK_ROWS * ROW_NBYTES
    assert session._prefill_token_buf.nbytes == CHUNK_ROWS * 8
    assert session._bulk_prefill_hidden_capacity_rows() == CHUNK_ROWS
    # The plan still owns the executor decision and the growth ceiling.
    assert session._bulk_prefill_full_prompt_capacity_rows == CAPACITY
    assert session._int8_prefill_lifetime_plan.mode == "layer_outer_shared_oracle"
    # The alias policy and the bulk scratch width are unchanged.
    assert session._bulk_prefill_hidden_planes == 1
    assert session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
    assert int(session._bulk_prefill_scratch.rows) == CHUNK_ROWS


def test_standalone_session_construction_is_unchanged(monkeypatch) -> None:
    session, _freed, _allocations = _fake_session(
        monkeypatch, defer_kv_allocation=False
    )
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._prefill_hidden_a.nbytes == CAPACITY * ROW_NBYTES
    assert session._prefill_token_buf.nbytes == CAPACITY * 8


def test_dms_and_expert_sidecar_construction_is_unchanged(monkeypatch) -> None:
    dms, _freed, _allocations = _fake_session(
        monkeypatch, dms_metadata_path="/tmp/dms.json"
    )
    dms._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert dms._prefill_hidden_a.nbytes == CAPACITY * ROW_NBYTES

    sidecar, _freed2, _allocations2 = _fake_session(
        monkeypatch, use_expert_sidecar=True
    )
    sidecar._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert sidecar._prefill_hidden_a.nbytes == CAPACITY * ROW_NBYTES


# --------------------------------------------------------------------------
# Growth
# --------------------------------------------------------------------------


def test_growth_reaches_the_prompt_and_is_monotonic(monkeypatch) -> None:
    session, freed, allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    initial_allocations = len(allocations)

    assert session._ensure_bulk_prefill_hidden_capacity(2_048) == 2_048
    assert session._prefill_hidden_a.nbytes == 2_048 * ROW_NBYTES
    assert session._prefill_token_buf.nbytes == 2_048 * 8
    assert session._bulk_prefill_hidden_capacity_rows() == 2_048
    grew_allocations = len(allocations)
    assert grew_allocations == initial_allocations + 2, "one token buffer + one aliased plane"

    # Reuse: a smaller request keeps the grown storage, and the same request
    # does not reallocate.
    assert session._ensure_bulk_prefill_hidden_capacity(1_024) == 2_048
    assert session._ensure_bulk_prefill_hidden_capacity(2_048) == 2_048
    assert len(allocations) == grew_allocations
    assert session._prefill_hidden_a.nbytes == 2_048 * ROW_NBYTES

    # The aliased plane is freed exactly once, and only the superseded set.
    superseded = [buffer for buffer in freed if buffer.nbytes == CHUNK_ROWS * ROW_NBYTES]
    assert len(superseded) == 1
    assert len({id(buffer) for buffer in freed}) == len(freed)


def test_growth_keeps_the_two_plane_policy(monkeypatch) -> None:
    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "0")
    session, freed, allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._bulk_prefill_hidden_planes == 2
    assert session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr
    initial_allocations = len(allocations)

    assert session._ensure_bulk_prefill_hidden_capacity(2_048) == 2_048
    assert len(allocations) == initial_allocations + 3, "token buffer + two planes"
    assert session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr
    assert session._prefill_hidden_b.nbytes == 2_048 * ROW_NBYTES
    # Both old planes are freed once each; no double free of either plane.
    old_plane_frees = [buffer for buffer in freed if buffer.nbytes == CHUNK_ROWS * ROW_NBYTES]
    assert len(old_plane_frees) == 2
    assert len({id(buffer) for buffer in freed}) == len(freed)


def test_growth_rebinds_every_borrowed_slot_view(monkeypatch) -> None:
    owner, _freed, _allocations = _fake_session(monkeypatch)
    owner._allocate_bulk_prefill_workspace(SimpleNamespace())
    stale = _workspace_buffers(owner)

    view = object.__new__(Qwen35GGUFResidentSession)
    view.__dict__.update(
        runner=owner.runner,
        runtime=owner.runtime,
        _resident_batch_owner=owner,
        _buffers=(),
        _prefill_token_buf=owner._prefill_token_buf,
        _prefill_hidden_a=owner._prefill_hidden_a,
        _prefill_hidden_b=owner._prefill_hidden_b,
        _bulk_prefill_scratch=owner._bulk_prefill_scratch,
    )
    view.scratch = SimpleNamespace(max_positions=CAPACITY)
    owner._resident_slot_views = [view]

    # Growth requested through the borrower resolves to the owner.
    assert view._ensure_bulk_prefill_hidden_capacity(2_048) == 2_048
    for attribute in (
        "_prefill_token_buf",
        "_prefill_hidden_a",
        "_prefill_hidden_b",
        "_bulk_prefill_scratch",
    ):
        assert getattr(view, attribute) is getattr(owner, attribute), attribute
    assert view._prefill_hidden_a.nbytes == 2_048 * ROW_NBYTES
    assert view._buffers == (), "a borrower never owns the ledger"
    assert view._bulk_prefill_full_prompt_capacity_rows == CAPACITY

    # Growth requested on the owner rebinds the borrower too.
    assert owner._ensure_bulk_prefill_hidden_capacity(4_096) == 4_096
    assert view._prefill_hidden_a is owner._prefill_hidden_a
    assert view._prefill_hidden_a.nbytes == 4_096 * ROW_NBYTES

    # The owner's ledger tracks exactly one entry per live workspace buffer.
    live = _workspace_buffers(owner)
    for buffer in live:
        assert sum(1 for entry in owner._buffers if entry is buffer) == 1
    for buffer in stale:
        assert all(entry is not buffer for entry in owner._buffers)


def test_growth_clamps_to_the_plans_full_prompt_capacity(monkeypatch) -> None:
    """A caller that skips the row validation still cannot over-allocate."""

    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._ensure_bulk_prefill_hidden_capacity(CAPACITY + 1) == CAPACITY
    assert session._bulk_prefill_hidden_capacity_rows() == CAPACITY


def test_growth_refuses_a_streamed_target_hidden_sink(monkeypatch) -> None:
    """The sink sizes its transform buffer from the workspace it was built on."""

    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    with pytest.raises(RuntimeError, match="before creating the sink"):
        session._ensure_bulk_prefill_hidden_capacity(
            2_048, target_hidden_chunk_sink=SimpleNamespace()
        )
    assert session._bulk_prefill_hidden_capacity_rows() == CHUNK_ROWS


@pytest.mark.parametrize("fail_at", [1, 2])
def test_partial_growth_failure_leaves_the_workspace_intact(
    monkeypatch, fail_at: int
) -> None:
    session, freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    before = _workspace_buffers(session)
    ledger_before = session._buffers
    freed_before = list(freed)

    real_malloc = gguf_runner.malloc
    attempts = {"count": 0}

    def failing_malloc(nbytes: int, *, runtime):
        attempts["count"] += 1
        if attempts["count"] == fail_at:
            raise MemoryError("device out of memory")
        return real_malloc(nbytes, runtime=runtime)

    monkeypatch.setattr(gguf_runner, "malloc", failing_malloc)

    with pytest.raises(MemoryError):
        session._ensure_bulk_prefill_hidden_capacity(2_048)

    # Nothing published and the old workspace is untouched: only the partial
    # replacement set is released.
    assert _workspace_buffers(session) == before
    assert session._buffers == ledger_before
    assert session._bulk_prefill_hidden_capacity_rows() == CHUNK_ROWS
    abandoned = [buffer for buffer in freed if buffer not in freed_before]
    assert len(abandoned) == fail_at - 1
    assert len({id(buffer) for buffer in abandoned}) == len(abandoned)
    for buffer in before:
        assert all(buffer is not entry for entry in abandoned)


def test_growth_failure_after_allocation_frees_the_replacements(monkeypatch) -> None:
    """A graph-invalidation failure must not leak the new allocation set."""

    session, freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    before = _workspace_buffers(session)
    freed_before = list(freed)

    def failing_invalidate() -> int:
        raise RuntimeError("cannot invalidate a live graph")

    session._invalidate_live_packed_decode_graphs = failing_invalidate

    with pytest.raises(RuntimeError, match="cannot invalidate"):
        session._ensure_bulk_prefill_hidden_capacity(2_048)

    assert _workspace_buffers(session) == before
    replacements = [buffer for buffer in freed if buffer not in freed_before]
    assert len(replacements) == 2
    assert len({id(buffer) for buffer in replacements}) == 2


def test_growth_requires_a_live_workspace(monkeypatch) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    with pytest.raises(RuntimeError, match="closed"):
        session._ensure_bulk_prefill_hidden_capacity(2_048)
    with pytest.raises(ValueError, match="positive"):
        session._ensure_bulk_prefill_hidden_capacity(0)


def test_release_and_reacquire_recompute_the_chunk_sized_workspace(
    monkeypatch,
) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session._ensure_bulk_prefill_hidden_capacity(2_048)
    session._release_bulk_prefill_workspace()
    assert session._bulk_prefill_hidden_capacity_rows() == 0
    session._ensure_bulk_prefill_workspace()
    assert session._bulk_prefill_hidden_capacity_rows() == CHUNK_ROWS
    assert session._bulk_prefill_full_prompt_capacity_rows == CAPACITY


# --------------------------------------------------------------------------
# Full-prompt call sites grow before they address the storage
# --------------------------------------------------------------------------


def _probe_growth(session, calls: list[tuple[int, str]]):
    def probe(rows, **kwargs):
        calls.append((int(rows), str(kwargs.get("consumer"))))
        raise _GrowthProbe("growth reached")

    session._ensure_bulk_prefill_hidden_capacity = probe
    return calls


def test_scalar_bulk_prefill_grows_before_reading_the_route(monkeypatch) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session.reset = lambda **kwargs: None
    calls = _probe_growth(session, [])

    with pytest.raises(_GrowthProbe):
        session._run_bulk_prefill_and_sample(list(range(2_048)))

    assert calls == [(2_048, "scalar bulk prefill")]


def test_scalar_bulk_prefill_without_growth_hits_the_shared_oracle_guard(
    monkeypatch,
) -> None:
    """The pre-change failure mode that makes growth mandatory, not optional."""

    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session.reset = lambda **kwargs: None
    session._ensure_bulk_prefill_hidden_capacity = (
        lambda rows, **kwargs: session._bulk_prefill_hidden_capacity_rows()
    )

    with pytest.raises(RuntimeError, match="full-capacity layer-outer hidden ownership"):
        session._run_bulk_prefill_and_sample(list(range(2_048)))


def test_scalar_bulk_prefill_keeps_chunk_outer_execution_for_chunk_plans(
    monkeypatch,
) -> None:
    """A chunk plan must not grow: its executor is selected by small storage.

    The recording wrapper calls the real growth step and then stops before any
    device work, so the assertion is on the capacity the route decision actually
    reads (``chunk_outer = allocated < rows``), not merely on the call.
    """

    session, _freed, allocations = _fake_session(
        monkeypatch,
        plan_mode="chunk_outer_layer_local_oracles",
        required_hidden_capacity=CHUNK_ROWS,
    )
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session.reset = lambda **kwargs: None
    observed: list[int] = []
    real_ensure = session._ensure_bulk_prefill_hidden_capacity

    def recording_ensure(rows, **kwargs):
        observed.append(int(real_ensure(rows, **kwargs)))
        raise _GrowthProbe("route decision reached")

    session._ensure_bulk_prefill_hidden_capacity = recording_ensure
    initial_allocations = len(allocations)

    with pytest.raises(_GrowthProbe):
        session._run_bulk_prefill_and_sample(list(range(2_048)))

    assert observed == [CHUNK_ROWS], "chunk plan must not grow to the prompt"
    assert len(allocations) == initial_allocations, "no replacement allocation"
    assert session._bulk_prefill_hidden_capacity_rows() == CHUNK_ROWS
    assert observed[0] < 2_048, "chunk_outer stays True, so the executor is unchanged"


@pytest.mark.parametrize(
    "plan_mode", ["bf16", "chunk_outer_layer_local_oracles", "bf16_mirror"]
)
def test_deferred_chunk_plan_scalar_prefill_does_not_grow(
    monkeypatch, plan_mode: str
) -> None:
    """Regression: a BF16/chunk-outer deferred owner keeps chunk-outer prefill."""

    session, _freed, allocations = _fake_session(
        monkeypatch, plan_mode=plan_mode, required_hidden_capacity=CHUNK_ROWS
    )
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session.reset = lambda **kwargs: None
    initial_allocations = len(allocations)

    capacity = session._ensure_bulk_prefill_hidden_capacity(4_096)

    assert capacity == CHUNK_ROWS
    assert len(allocations) == initial_allocations
    assert session._prefill_hidden_a.nbytes == CHUNK_ROWS * ROW_NBYTES
    assert capacity < 4_096, "chunk_outer must stay True for this plan"


def test_packed_layer_outer_prefill_grows_to_the_combined_prompt(monkeypatch) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session._prefill_scratch_rows = lambda capacity: CHUNK_ROWS  # noqa: ARG001
    calls = _probe_growth(session, [])
    chunks = (
        _GGUFPackedARPrefillChunk(
            slot_indices=(0, 1),
            start_offsets=(0, 0),
            prompt_token_ids=((1,) * 700, (2,) * 700),
        ),
        _GGUFPackedARPrefillChunk(
            slot_indices=(0, 1),
            start_offsets=(700, 700),
            prompt_token_ids=((3,) * 700, (4,) * 700),
        ),
    )

    with pytest.raises(_GrowthProbe):
        session._prefill_batch_native_layer_outer(
            [(1,) * 1_400, (2,) * 1_400],
            sessions=(session, session),
            chunks=chunks,
        )

    assert calls == [(2_800, "layer-outer packed prefill")]


def test_resumable_layer_outer_segment_grows_before_restoring(monkeypatch) -> None:
    session, _freed, _allocations = _fake_session(monkeypatch)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    session.last_packed_prefill_plan = {}
    restored: list[int] = []
    state = _GGUFResumablePrefillState(
        prompts=((1,) * 2_048,),
        sessions=(session,),
        chunk_plans=[],
        packed_state=SimpleNamespace(),
        packed_scratch_base=None,
        layer_types=("full_attention", "linear_attention"),
        next_layer=1,
        total_rows=2_048,
        runtime=session.runtime,
        scratch=SimpleNamespace(
            restore=lambda **kwargs: restored.append(1),
        ),
    )
    calls = _probe_growth(session, [])

    with pytest.raises(_GrowthProbe):
        session._prefill_batch_native_layer_outer_segment(state)

    assert calls == [(2_048, "resumable layer-outer prefill")]
    assert restored == [], "restore must never run into undersized planes"


if __name__ == "__main__":
    pytest.main([__file__])
