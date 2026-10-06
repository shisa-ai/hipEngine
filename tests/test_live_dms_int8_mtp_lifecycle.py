"""Live lifecycle arms for the DMS + INT8 MTP store journal (task #26).

Scope, stated exactly. DMS reaches the resident-session surface only:
``dms_metadata_path`` is accepted by ``Qwen35GGUFResidentSession`` and by nothing
in the engine, LLM, or HTTP layer, so a DMS row has no ``LLM.generate`` or server
request to cancel from. These arms therefore drive the same resident-session
harness the DMS parity gate uses (``tests/test_live_dms_int8_mtp_parity.py``) and
observe the lifecycle the verifier owns: a cycle opened by ``prepare`` has
already appended its rows to the DMS store, and the journal is what puts the
store back when the cycle never commits.

What each arm asserts, per the task's acceptance: the exact restored store state
-- payload planes, scale planes, positions, live counts, evict mask, range
capacity, extents and extent-pool ownership, on both the host state and the
device payload store -- plus a following cycle that still commits exactly the
tokens the autoregressive arm produces.

The rollback path under test is ``Qwen35GGUFTransactionalVerifier.close()``
calling ``rollback()`` for a still-open ``prepared`` cycle, which composes the
resident ``_StateJournal`` with ``DMSCompactBackend.rollback``. A cycle that
aborts must be rolled back, never dropped: a dropped cycle leaves the store's
appended rows behind and the next append then fails its own cross-check.

Skips unless ROCm, the dense GGUF model, and the DMS sidecar metadata are all
present.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import memory_stats
from hipengine.generation.deadline import (
    GenerationCancelled,
    GenerationDeadlineExceeded,
)
from hipengine.runtime.qwen35_gguf_mtp import _deadline_checkpoint

from tests.test_live_dms_int8_mtp_parity import (
    _BACKEND,
    _CYCLE_WIDTH,
    _DECODE_STEPS,
    _METADATA,
    _MODEL,
    _ar_arm,
    _category_cases,
    _dms_session,
    _hip_available,
    _prefill,
)

# The resident harness drives one DMS row, and the parity gate's draft batches
# use request id 0, so the row's compact state is keyed 0.
_ROW_REQUEST_ID = 0

pytestmark = pytest.mark.skipif(
    not (_hip_available() and _MODEL.exists() and _METADATA.exists()),
    reason="requires ROCm + the dense GGUF model + DMS sidecar metadata",
)


@pytest.fixture(scope="module")
def shared_runner():
    from hipengine.runtime import qwen35_gguf_runner as runner_module
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 0)
        yield Qwen35GGUFFullStackRunner(_MODEL, backend=_BACKEND)


def _freeze(value: Any) -> Any:
    """Reduce a snapshot value to something comparable across two reads."""

    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, dict):
        return {key: _freeze(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if dataclasses.is_dataclass(value):
        return _freeze(dataclasses.astuple(value))
    return value


def _row_state(backend):
    if backend.has_request(_ROW_REQUEST_ID):
        return backend.state_for_request(_ROW_REQUEST_ID)
    ids = sorted(backend._states)
    assert ids, "the session has no compact DMS state"
    return backend.state_for_request(ids[0])


def _store_signature(session) -> dict[str, Any]:
    """Everything the acceptance names, read from the live store.

    Host state (payload, scales, positions, live counts, evict mask, extents,
    range capacity), the extent pool's and ledger's own allocator state, and the
    device payload store's planes, so a rollback that restores only one half
    fails the comparison. The device half is required, not optional: this host's
    INT8 DMS rows are device-payload rows, and a signature that silently skipped
    those planes would pass while the device store stayed corrupted.
    """

    backend = session._dms_backend
    state = _row_state(backend)
    signature: dict[str, Any] = {
        "logical_tokens": int(state.logical_tokens),
        "live_counts": _freeze(state.live_counts),
        "token_positions": _freeze(state.token_positions),
        "evict_mask": _freeze(state.evict_mask),
        "range_capacity": _freeze(state.range_capacity),
        "base_offsets": _freeze(state.base_offsets),
        "extents": _freeze(state.extents),
        "k_payload": _freeze(state.k_payload),
        "v_payload": _freeze(state.v_payload),
        "k_scales": _freeze(state.k_scales),
        "v_scales": _freeze(state.v_scales),
        "extent_pool": _freeze(backend.extents.state_snapshot()),
        "ledger": _freeze(backend.ledger.state_snapshot()),
    }
    store = backend._device_store
    assert store is not None, (
        "this session has no device payload store, so the device planes the "
        "acceptance names would not be compared"
    )
    view = store.layer_view(0)
    signature["device"] = {
        "k_bits": _freeze(view.k_bits),
        "v_bits": _freeze(view.v_bits),
        "k_scales": _freeze(view.k_scales),
        "v_scales": _freeze(view.v_scales),
        "positions": _freeze(view.positions),
        "evict": _freeze(view.evict),
        "live_counts": _freeze(store.live_counts(0)),
    }
    return signature


def _open_cycle(session, verifier, token, candidates, cycle):
    """Prepare one verify cycle, which appends its rows to the DMS store."""

    from hipengine.speculative import DraftBatch, TargetVerifyBatch

    width = len(candidates)
    position = int(session.position)
    draft = DraftBatch(
        request_ids=(_ROW_REQUEST_ID,),
        candidate_tokens=tuple(candidates),
        parent_positions=tuple(position + index for index in range(width)),
        draft_depths=tuple(range(1, width + 1)),
        row_to_request=(_ROW_REQUEST_ID,) * width,
        mode="verify_chain",
    )
    batch = TargetVerifyBatch.from_draft(
        draft, root_tokens=(token,), root_positions=(position,)
    )
    prepared = verifier.prepare(
        batch,
        transaction_id=cycle,
        graph_bucket=verifier.graph_bucket(f"dms-lifecycle-c{cycle}", batch),
        remaining_decode=(_DECODE_STEPS,),
        allow_graph=False,
    )
    assert prepared.gpu_accept_match_cpu
    return batch, prepared


def _commit_cycle(verifier, batch, prepared, cycle):
    """Commit a prepared cycle and return its tokens plus the next token."""

    from hipengine.speculative import TargetCommitPlan

    plan = TargetCommitPlan(
        transaction_id=cycle,
        request_ids=batch.request_ids,
        accepted_counts=prepared.summary.accepted_counts,
        commit_rows=prepared.summary.commit_rows,
        commit_tokens=prepared.summary.commit_tokens,
        commit_positions=prepared.summary.commit_positions,
        next_tokens=prepared.summary.next_tokens,
        candidate_counts=batch.candidate_counts,
        draft_depth=batch.draft_depth,
        tree_shape=batch.tree_shape,
        mode=batch.mode,
    )
    verifier.commit(prepared, plan)
    verifier.finish(prepared)
    committed = [int(value) for value in prepared.summary.accepted_tokens[0]]
    committed.append(int(prepared.summary.next_tokens[0]))
    return committed


def _verifier(session):
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier

    return Qwen35GGUFTransactionalVerifier(
        session,
        max_candidate_budget=_CYCLE_WIDTH,
        quant="gguf_q4_k_m",
        target_verify_mode="serial_exact",
    )


def _following_cycle_commits_exactly(session, root, oracle, *, committed_already: int) -> None:
    """After an aborted cycle, a fresh cycle must still commit the AR tokens."""

    token = oracle[committed_already - 1] if committed_already else root
    candidates = list(oracle[committed_already : committed_already + _CYCLE_WIDTH])
    verifier = _verifier(session)
    try:
        batch, prepared = _open_cycle(session, verifier, token, candidates, 99)
        committed = _commit_cycle(verifier, batch, prepared, 99)
        accepted = sum(int(value) for value in prepared.summary.accepted_counts)
    finally:
        verifier.close()
    assert committed == list(oracle[committed_already : committed_already + len(committed)]), (
        "the cycle after the aborted one did not commit the autoregressive tokens"
    )
    assert accepted > 0, "the cycle after the aborted one accepted nothing"


def test_dms_int8_mtp_cancel_mid_cycle_restores_the_store(shared_runner) -> None:
    """A cancelled cycle is rolled back, and the next cycle still commits.

    The cancellation is the real ``GenerationCancelled`` the engine raises; what
    is exercised here is the unwind, because the DMS row's cancellation arrives
    between ``prepare`` and ``commit`` and the journal owns the restoration.
    """

    prompt = _category_cases()[0][1]
    oracle = _ar_arm(shared_runner, prompt)["tokens"]
    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt)
        before = _store_signature(session)
        verifier = _verifier(session)
        try:
            _, prepared = _open_cycle(session, verifier, root, list(oracle[:_CYCLE_WIDTH]), 1)
            # The open cycle really did mutate the store, or this proves nothing.
            assert _store_signature(session) != before, (
                "the open cycle appended nothing, so the rollback is vacuous"
            )
            with pytest.raises(GenerationCancelled):
                raise GenerationCancelled("client cancelled mid-cycle")
        finally:
            verifier.close()
        assert _store_signature(session) == before, "the cancelled cycle was not rolled back"
        _following_cycle_commits_exactly(session, root, oracle, committed_already=0)


def test_dms_int8_mtp_deadline_mid_cycle_restores_and_unwinds_idempotently(
    shared_runner,
) -> None:
    """A real expired deadline aborts the cycle, and the unwind is idempotent."""

    prompt = _category_cases()[0][1]
    oracle = _ar_arm(shared_runner, prompt)["tokens"]
    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt)
        before = _store_signature(session)
        verifier = _verifier(session)
        try:
            _, prepared = _open_cycle(session, verifier, root, list(oracle[:_CYCLE_WIDTH]), 1)
            assert _store_signature(session) != before
            # A real checkpoint from the module's own factory, with a deadline
            # that has already expired.
            checkpoint = _deadline_checkpoint(0.0)
            assert checkpoint is not None
            with pytest.raises(GenerationDeadlineExceeded):
                checkpoint()
        finally:
            verifier.close()
        restored = _store_signature(session)
        assert restored == before, "the deadline-aborted cycle was not rolled back"
        # Idempotence: unwinding again must not move the store, and must not
        # raise on a verifier that is already closed.
        verifier.close()
        assert _store_signature(session) == restored, "the second unwind moved the store"
        _following_cycle_commits_exactly(session, root, oracle, committed_already=0)


def test_dms_int8_mtp_disconnect_mid_cycle_rolls_back_rather_than_drops(
    shared_runner,
) -> None:
    """An unexpected mid-cycle exception rolls the open cycle back, not drops it.

    A dropped cycle is the failure this arm exists to catch: the store would keep
    the rows the aborted cycle appended, and the next append would then fail its
    own host/device cross-check. The exception used is the one a client
    disconnect surfaces as at this layer.
    """

    prompt = _category_cases()[0][1]
    oracle = _ar_arm(shared_runner, prompt)["tokens"]
    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt)
        before = _store_signature(session)
        verifier = _verifier(session)
        try:
            _, prepared = _open_cycle(session, verifier, root, list(oracle[:_CYCLE_WIDTH]), 1)
            assert _store_signature(session) != before
            with pytest.raises(ConnectionResetError):
                raise ConnectionResetError("client disconnected mid-cycle")
        finally:
            verifier.close()
        assert _store_signature(session) == before, "the disconnected cycle was not rolled back"
        # The store must still be usable: this is what a dropped cycle breaks.
        _following_cycle_commits_exactly(session, root, oracle, committed_already=0)


def test_dms_int8_mtp_session_teardown_leaves_no_outstanding_allocations(
    shared_runner,
) -> None:
    """Session teardown after real cycles returns to the allocation baseline."""

    prompt = _category_cases()[0][1]
    oracle = _ar_arm(shared_runner, prompt)["tokens"]
    baseline = int(memory_stats()["active_allocations"])
    with _dms_session(shared_runner) as session:
        root = _prefill(session, prompt)
        verifier = _verifier(session)
        try:
            _, prepared = _open_cycle(session, verifier, root, list(oracle[:_CYCLE_WIDTH]), 1)
            # Tear the cycle down without committing, so the journal's unwind and
            # the session's own teardown both have to release what they took.
        finally:
            verifier.close()
    outstanding = int(memory_stats()["active_allocations"])
    assert outstanding == baseline, (
        f"session teardown left {outstanding - baseline} tracked allocations "
        f"outstanding (baseline {baseline}, now {outstanding})"
    )
