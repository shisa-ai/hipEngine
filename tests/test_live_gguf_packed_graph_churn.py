"""Live churn gate for packed decode graph binding (task #13).

A captured packed decode graph binds raw device pointers: weight allocations,
the prefill/verify scratch planes, and the packed state/scratch workspace.  Any
event that frees or rebinds one of those buffers must invalidate the graphs that
reference them, or the next replay writes through stale pointers.

This gate drives a real captured graph through the four churn events the
resident runner guards, and asserts each one closes the live graph before a
fresh capture reproduces the cold baseline exactly:

* slot reassignment -- packed prefill becomes the new occupant of the slots
  ``prefill_batch_native``);
* verify-workspace growth -- ``_ensure_packed_verify_workspace`` at a wider
  row/context geometry;
* lm-head plane growth -- ``_ensure_verify_lm_head_buffers`` past capacity;
* session cancellation/close -- the terminal path.

KV scale reallocation is the fifth case in the task and is unreachable by
design: ``capture_packed_decode_graph`` refuses non-BF16 storage up front, so no
INT8 session can ever bind a graph to a scale plane.  The last test pins that
refusal so the case cannot silently become reachable without this gate noticing.

Skips unless ROCm and the dense GGUF model are both present.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PROMPT = (9707, 11, 220, 264, 12)
NEXT_TOKEN = 264
STEPS = 2
# Growth target for the lm-head planes: far above the rows==1 capture so the
# ensure-path cannot satisfy it from existing holdings. The packed workspace is
# already prefill-shaped (max_sequence_length rows), so that case grows relative
# to the live scratch instead.
GROWTH_ROWS = 64
GROWTH_ROWS_MARGIN = 64
GROWTH_MAX_SEQUENCE = 4096


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm + the dense 27B GGUF model",
)


@pytest.fixture(scope="module")
def session():
    """One live resident session reused across the ordered churn cases."""

    from hipengine.core.dtype import DType
    from hipengine.kvcache import FixedPagedKVPolicy
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    # Keep the short-prompt prefill on the plain BF16 mirror route so the
    # captured graph binds the same planes the probe recipe exercised.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            runner_module, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 0
        )
        with runner_module.Qwen35GGUFResidentSession(
            MODEL,
            max_sequence_length=1536,
            kv_scale_dtype=DType.FP32,
            kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
        ) as resident:
            yield resident


def _capture_and_replay(resident):
    """Capture a rows==1 graph, replay it, and return its tokens and handle."""

    graph = resident.capture_packed_decode_graph(
        (NEXT_TOKEN,),
        sessions=(resident,),
        physical_rows=1,
        active_slot_indices=(0,),
        steps_per_replay=1,
        max_replay_steps=4,
        record_steps=4,
    )
    graph.replay(STEPS)
    tokens = tuple(int(token) for token in graph.read_latest_generated_token_ids())
    return tokens, graph


def _cold_cycle(resident):
    """Re-establish the cold state and capture: the comparison baseline."""

    resident.reset()
    resident.prefill(PROMPT)
    return _capture_and_replay(resident)


def _owner_plane_ptrs(resident):
    """The owner-owned plane set the graph key hashes.

    Capture-time feedback planes (generated tokens, record index, active mask)
    are allocated per capture, so they are excluded here: this is the binding
    set that only a reallocation can move.
    """

    from hipengine.runtime.gguf_packed_decode_graph import _packed_graph_buffer_ptrs

    state = resident._packed_verify_state
    scratch = resident._packed_verify_scratch
    assert state is not None and scratch is not None, (
        "the packed workspace must be allocated before this comparison"
    )
    return _packed_graph_buffer_ptrs(resident, state, scratch)


def _runtime(resident):
    runtime = resident.runtime
    assert runtime is not None, "a live resident session must expose its runtime"
    return runtime


def test_cold_capture_is_reproducible(session) -> None:
    """Control: the baseline must repeat, or no churn verdict below is usable."""

    first_tokens, _ = _cold_cycle(session)
    before = _owner_plane_ptrs(session)
    second_tokens, _ = _cold_cycle(session)
    after = _owner_plane_ptrs(session)

    assert first_tokens == second_tokens
    assert first_tokens, "the captured graph must record at least one token"
    # A plain reset/prefill cycle reallocates nothing, so the owner-owned
    # binding set is unchanged -- which is what makes the growth cases below
    # attributable to the growth rather than to the cycle.
    assert before == after


def test_prefill_reassignment_invalidates_the_live_graph(session) -> None:
    """Packed prefill takes the slots, so the previous occupant must not survive."""

    baseline, stale = _cold_cycle(session)
    stale_key = stale.bucket_key

    # The shipping AR prefill entry: same slots, different occupant.
    session.reset()
    session.prefill_batch_native((PROMPT,), sessions=(session,))

    assert stale.closed, "packed prefill left a graph bound to overwritten slots"
    with pytest.raises(RuntimeError, match="closed"):
        stale.replay(1)

    after, fresh = _capture_and_replay(session)
    assert after == baseline
    assert (
        fresh.bucket_key.state_generations != stale_key.state_generations
    ), "the re-captured graph still reports the invalidated cycle's generation"


def test_verify_workspace_growth_invalidates_the_live_graph(session) -> None:
    """Growth frees workspace planes the live graph replays against."""

    baseline, stale = _cold_cycle(session)
    before_planes = _owner_plane_ptrs(session)
    before_rows = int(session._packed_verify_scratch.rows)

    session._ensure_packed_verify_workspace(
        slot_count=1,
        rows=before_rows + GROWTH_ROWS_MARGIN,
        max_sequence_length=GROWTH_MAX_SEQUENCE,
        runtime=_runtime(session),
    )

    assert int(session._packed_verify_scratch.rows) > before_rows, (
        "the requested growth did not widen the packed workspace"
    )
    assert _owner_plane_ptrs(session) != before_planes, (
        "workspace growth must move the owner-owned binding set"
    )
    assert stale.closed, "workspace growth left a graph bound to freed planes"
    with pytest.raises(RuntimeError, match="closed"):
        stale.replay(1)

    after, fresh = _capture_and_replay(session)
    assert after == baseline
    assert (
        fresh.bucket_key.buffer_identity_sha256 != stale.bucket_key.buffer_identity_sha256
    )


def test_lm_head_growth_invalidates_the_live_graph(session) -> None:
    """The verify lm-head token/index planes are replay targets, not scratch."""

    baseline, stale = _cold_cycle(session)
    before_planes = _owner_plane_ptrs(session)
    before_capacity = int(session._verify_lm_rows_capacity)

    session._ensure_verify_lm_head_buffers(GROWTH_ROWS, runtime=_runtime(session))

    assert int(session._verify_lm_rows_capacity) > before_capacity, (
        "the requested growth did not widen the lm-head planes"
    )
    assert _owner_plane_ptrs(session) != before_planes, (
        "lm-head growth must move the owner-owned binding set"
    )
    assert stale.closed, "lm-head growth left a graph bound to freed token planes"
    with pytest.raises(RuntimeError, match="closed"):
        stale.replay(1)

    after, fresh = _capture_and_replay(session)
    assert after == baseline
    assert (
        fresh.bucket_key.buffer_identity_sha256 != stale.bucket_key.buffer_identity_sha256
    )


def test_int8_scale_reallocation_cannot_reach_a_capture(session) -> None:
    """Document why the scale-reallocation case is unreachable by design."""

    from hipengine.core.dtype import DType

    baseline, graph = _cold_cycle(session)
    assert baseline

    original = session.kv_storage_dtype
    try:
        session.kv_storage_dtype = DType.INT8
        with pytest.raises(NotImplementedError, match="BF16 KV only"):
            session.capture_packed_decode_graph(
                (NEXT_TOKEN,),
                sessions=(session,),
                physical_rows=1,
                active_slot_indices=(0,),
                steps_per_replay=1,
                max_replay_steps=4,
                record_steps=4,
            )
    finally:
        session.kv_storage_dtype = original

    # The refusal is up front: the pre-existing graph is untouched and still
    # replayable, because nothing was reallocated.
    assert not graph.closed
    assert tuple(
        int(token) for token in graph.read_latest_generated_token_ids()
    ) == baseline


def test_close_invalidates_every_live_graph(session) -> None:
    """Cancellation/teardown must leave no replayable handle behind."""

    baseline, graph = _cold_cycle(session)
    assert baseline

    session.close()

    assert graph.closed, "session close left a replayable graph behind"
    with pytest.raises(RuntimeError, match="closed"):
        graph.replay(1)
    assert session._decode_graphs == []
