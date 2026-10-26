"""Mixed target publication, commit preflight and lifecycle containment.

The fakes here stand in for the live runner and its index owners so each refusal
and each cleanup failure can be asserted exactly: which host check rejected the
call, whether any live write happened, and whether the pass's own buffers were
released anyway.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_mixed_target as mixed_target
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedRequest, compile_mixed_layout
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

SLOTS = 3
# Slot 2 carries request 10 at position 5 with two prepared candidates, slot 0
# carries request 20 as an AR peer at position 7. Both chains end at index cursor
# 8 once the pass has appended them.
START_CURSORS = (8, 0, 8)


class _CommitRecorder:
    """A committing state adapter that records the counts it was handed."""

    def __init__(self):
        self.calls = []

    def commit(self, counts):
        self.calls.append(tuple(counts))


class _CloseRecorder:
    """An owned resource that records its own release."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _layout():
    return compile_mixed_layout((Qwen4ExpMixedRequest(10, 2, 8, 5, 11, (12, 13)),
                                 Qwen4ExpMixedRequest(20, 0, 9, 7, 14)))


def _batch(*, ple=False, slots=SLOTS):
    batch = SimpleNamespace(
        _positions=np.array([7, 0, 5]), _owners=[9, 0, 8], _poisoned=set(),
        _ple_hash_states={}, index_states=[], gdn_bindings={0: SimpleNamespace()},
        _ple_layers=frozenset({0}) if ple else frozenset(),
        max_positions=300, _slot_blocks=[2] * slots, closed=False,
        config=SimpleNamespace(vocab_size=1024), backend='hip_gfx1100', runtime=object(),
    )
    batch.resident = SimpleNamespace(
        weight=lambda name: SimpleNamespace(spec=SimpleNamespace(quant_key='q4_k_m')),
    )

    def require_open():
        if batch.closed:
            raise RuntimeError('Qwen4Exp batch runner is closed')

    def require_slot(slot):
        if not 0 <= int(slot) < slots:
            raise ValueError(f'slot {slot} is outside [0, {slots})')
        return int(slot)

    def require_owner(slot, token):
        if batch._owners[slot] != token:
            raise ValueError('stale owner')

    batch._require_open = require_open
    batch._require_slot = require_slot
    batch._require_owner = require_owner
    batch._require_healthy = lambda slot: None
    batch.blocks_for_positions = lambda positions: (int(positions) + 255) // 256
    return batch


def _pass(*, ple=False, slots=SLOTS):
    target = object.__new__(Qwen4ExpMixedTargetPass)
    target.layout = _layout()
    target.batch = _batch(ple=ple, slots=slots)
    target.runtime = target.batch.runtime
    target.closed = target.closing = target.resolved = False
    target.verified = True
    target.gdn = {}
    target.ple = {}
    target.qsa_runs = []
    target.hash_snapshots = {}
    target._owned = []
    return target


def _wide_pass(*, requests, depth, ple=False):
    """A pass whose row count can exceed a dispatch bound while each wave stays narrow.

    Request ``i`` owns slot ``i`` at position 1, so a depth-``d`` chain costs
    ``d + 1`` physical rows but still contributes one row to each of its waves.
    """

    owners = [index + 1 for index in range(requests)]
    layout = compile_mixed_layout(tuple(
        Qwen4ExpMixedRequest(100 + index, index, owners[index], start_position=1,
                             root_token=11, candidates=tuple(range(12, 12 + depth)))
        for index in range(requests)
    ))
    target = _pass(ple=ple, slots=requests)
    target.layout = layout
    target.batch._positions = np.ones(requests, dtype=np.int64)
    target.batch._owners = list(owners)
    target.batch._slot_blocks = [2] * requests
    return target


def _index_state(*, identity=7, token=None, cursors=START_CURSORS,
                 closed=False, closing=False):
    """A read-only index owner: no device buffer, only the counters a token names."""

    state = object.__new__(Qwen4ExpBatchQsaIndexState)
    state.identity = identity
    state.pending_rollback = token
    state.cursor_host = np.array(cursors, dtype=np.int64)
    state.pooled_count_host = np.zeros(SLOTS, dtype=np.int64)
    state.compression_ratio = 64
    state.closed = closed
    state.closing = closing
    return state


def _rollback_token(owner=7):
    return SimpleNamespace(owner=owner)


def _qsa_run(layout, token, owners=(8, 9)):
    return SimpleNamespace(layout=layout, rollback=token,
                           plan=SimpleNamespace(owners=tuple(owners)))


# -- admission bounds -------------------------------------------------------


def test_admission_accepts_a_deep_pass_wider_than_the_dispatch_row_bound():
    """A chain costs rows, not wave width: 20 requests x depth 3 is 80 rows, 20 wide."""

    target = _wide_pass(requests=20, depth=3)
    assert target.layout.total_rows == 80 > mixed_target.QWEN4_EXP_GDN_INDEXED_MAX_ROWS
    assert target.layout.physical_request_width == 20
    target._validate()


def test_admission_refuses_a_wave_wider_than_the_dispatch_row_bound():
    target = _wide_pass(requests=mixed_target.QWEN4_EXP_GDN_INDEXED_MAX_ROWS + 1, depth=0)
    assert target.layout.physical_request_width == 65
    with pytest.raises(ValueError, match='past the indexed GDN decode dispatch bound'):
        target._validate()


def test_admission_refuses_a_pass_past_the_state_slab_bound(monkeypatch):
    monkeypatch.setattr(mixed_target, 'QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS', 4)
    target = _wide_pass(requests=2, depth=3)
    assert target.layout.total_rows == 8
    assert target.layout.physical_request_width == 2
    with pytest.raises(ValueError, match='state slab bound of 4'):
        target._validate()


def test_admission_refuses_a_ple_wave_wider_than_its_dispatch_row_bound(monkeypatch):
    monkeypatch.setattr(mixed_target, 'QWEN4_EXP_PLE_INDEXED_MAX_ROWS', 2)
    target = _wide_pass(requests=3, depth=0, ple=True)
    assert target.layout.physical_request_width == 3
    with pytest.raises(ValueError, match='past the indexed PLE decode dispatch bound of 2'):
        target._validate()


# -- publication ------------------------------------------------------------


def test_commit_advances_ar_once_and_only_the_chosen_speculative_prefix():
    target = _pass()
    called = []
    target.gdn[0] = SimpleNamespace(commit=lambda counts: called.append(tuple(counts)))
    plan = target.commit((1, 0))
    assert called == [(1, 0)]
    assert tuple(target.batch._positions) == (8, 0, 7)
    assert plan.commit_rows == (1, 3)
    with pytest.raises(RuntimeError, match='unresolved'):
        target.commit((1, 0))


@pytest.mark.parametrize('fault', ['owner', 'position', 'depth'])
def test_invalid_commit_is_refused_before_any_state_publication(fault):
    target = _pass()
    called = []
    target.gdn[0] = SimpleNamespace(commit=lambda counts: called.append(counts))
    if fault == 'owner':
        target.batch._owners[2] = 12
    if fault == 'position':
        target.batch._positions[2] += 1
    with pytest.raises(ValueError):
        target.commit((3 if fault == 'depth' else 1, 0))
    assert called == []
    assert not target.resolved
    assert target.batch._poisoned == set()


def test_device_commit_failure_poisons_every_participant_and_cannot_retry():
    target = _pass()

    def fail(counts):
        raise RuntimeError('copy failed')

    target.gdn[0] = SimpleNamespace(commit=fail)
    with pytest.raises(RuntimeError, match='copy failed'):
        target.commit((1, 0))
    assert target.batch._poisoned == {0, 2}
    assert tuple(target.batch._positions) == (7, 0, 5)
    with pytest.raises(RuntimeError, match='unresolved'):
        target.commit((1, 0))


def test_abort_leaves_live_positions_and_state_unpublished():
    target = _pass()
    target.abort()
    target.abort()
    assert tuple(target.batch._positions) == (7, 0, 5)
    assert target.batch._poisoned == set()
    assert target.resolved


# -- commit preflight -------------------------------------------------------


def test_commit_on_a_closed_live_runner_is_refused_before_any_live_write():
    target = _pass()
    gdn = _CommitRecorder()
    target.gdn = {0: gdn}
    target.batch.closed = True
    with pytest.raises(RuntimeError, match='batch runner is closed'):
        target.commit((1, 0))
    assert gdn.calls == []
    assert target.batch._poisoned == set()
    assert not target.resolved
    assert tuple(target.batch._positions) == (7, 0, 5)


def test_commit_refuses_a_slot_the_live_runner_does_not_have():
    target = _pass()
    target.layout = compile_mixed_layout((Qwen4ExpMixedRequest(10, 7, 8, 5, 11),))
    gdn = _CommitRecorder()
    target.gdn = {0: gdn}
    with pytest.raises(ValueError, match='outside'):
        target.commit((0,))
    assert gdn.calls == []
    assert not target.resolved


def test_stale_index_cursor_is_refused_before_any_live_state_is_written(monkeypatch):
    target = _pass()
    token = _rollback_token()
    state = _index_state(token=token, cursors=(8, 0, 9))
    target.batch.index_states = [state]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]
    gdn = _CommitRecorder()
    target.gdn = {0: gdn}
    committed = []
    monkeypatch.setattr(mixed_target, 'commit_qwen4_exp_mixed_qsa_attention',
                        lambda run, counts, **kwargs: committed.append(tuple(counts)))

    with pytest.raises(ValueError, match='is not where the mixed pass left it'):
        target.commit((1, 0))

    assert gdn.calls == [] and committed == []
    assert target.batch._poisoned == set()
    assert not target.resolved
    assert tuple(target.batch._positions) == (7, 0, 5)

    # A refused preflight leaves the pass abortable, so the corrected call is
    # still the same transaction rather than a poisoned one.
    state.cursor_host[2] = 8
    plan = target.commit((1, 0))
    assert plan.commit_rows == (1, 3)
    assert gdn.calls == [(1, 0)]
    assert committed == [(1, 0)]
    assert tuple(target.batch._positions) == (8, 0, 7)


def test_index_transaction_already_resolved_is_refused_before_any_live_write():
    target = _pass()
    token = _rollback_token()
    target.batch.index_states = [_index_state(token=None)]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]
    gdn = _CommitRecorder()
    target.gdn = {0: gdn}
    with pytest.raises(RuntimeError, match='no longer pending'):
        target.commit((1, 0))
    assert gdn.calls == []
    assert target.batch._poisoned == set()
    assert tuple(target.batch._positions) == (7, 0, 5)


def test_index_owner_from_another_pass_is_refused_before_any_live_write():
    target = _pass()
    token = _rollback_token(owner=7)
    target.batch.index_states = [_index_state(identity=11, token=token)]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]
    gdn = _CommitRecorder()
    target.gdn = {0: gdn}
    with pytest.raises(ValueError, match='different index owner'):
        target.commit((1, 0))
    assert gdn.calls == []
    assert target.batch._poisoned == set()


# -- PLE n-gram history precondition ----------------------------------------


def test_validate_refuses_a_ple_history_that_does_not_start_where_the_chain_does():
    target = _pass(ple=True)
    target.batch._ple_hash_states = {2: SimpleNamespace(next_position=4),
                                    0: SimpleNamespace(next_position=7)}
    with pytest.raises(ValueError, match='PLE n-gram history starts at position 4'):
        target._validate()
    target.batch._ple_hash_states = {2: SimpleNamespace(next_position=5),
                                    0: SimpleNamespace(next_position=7)}
    target._validate()


def test_validate_refuses_a_missing_ple_history_beyond_position_zero():
    target = _pass(ple=True)
    target.batch._ple_hash_states = {0: SimpleNamespace(next_position=7)}
    with pytest.raises(ValueError, match='holds no PLE n-gram history'):
        target._validate()


def test_validate_accepts_a_fresh_request_with_no_ple_history():
    target = _pass(ple=True)
    target.layout = compile_mixed_layout((Qwen4ExpMixedRequest(10, 2, 8, 0, 11),))
    target.batch._positions = np.zeros(SLOTS, dtype=np.int64)
    target._validate()


# -- verify failure containment ---------------------------------------------


def test_verify_failure_keeps_its_primary_error_and_notes_the_failed_discard(monkeypatch):
    target = _pass()
    target.verified = False
    target.token_ids = None

    monkeypatch.setattr(mixed_target, 'resolve', lambda **kwargs: object())

    def failing_copy(*args, **kwargs):
        raise RuntimeError('device copy failed')

    def failing_abort(self):
        raise RuntimeError('rollback failed')

    monkeypatch.setattr(mixed_target, 'copy_host_array_to_device', failing_copy)
    monkeypatch.setattr(Qwen4ExpMixedTargetPass, 'abort', failing_abort)

    with pytest.raises(RuntimeError, match='device copy failed') as info:
        target.verify()

    notes = getattr(info.value, '__notes__', [])
    assert any('discarding the failed pass also failed' in note for note in notes)
    assert 'rollback failed' in ' '.join(notes)
    assert not target.resolved


def test_commit_failure_keeps_its_primary_error_and_notes_the_failed_abort(monkeypatch):
    target = _pass()
    token = _rollback_token()
    target.batch.index_states = [_index_state(token=token)]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]

    def fail(counts):
        raise RuntimeError('copy failed')

    def failing_abort(*args, **kwargs):
        raise RuntimeError('rollback failed')

    target.gdn[0] = SimpleNamespace(commit=fail)
    monkeypatch.setattr(mixed_target, 'abort_qwen4_exp_mixed_qsa_attention', failing_abort)

    with pytest.raises(RuntimeError, match='copy failed') as info:
        target.commit((1, 0))

    notes = getattr(info.value, '__notes__', [])
    assert any('aborting the failed commit also failed' in note for note in notes)
    assert 'rollback failed' in ' '.join(notes)
    assert target.batch._poisoned == {0, 2}


# -- release ----------------------------------------------------------------


@pytest.mark.parametrize('released', ['dropped', 'closed'])
def test_close_after_the_live_runner_released_its_index_owners_still_frees(
        monkeypatch, released):
    target = _pass()
    token = _rollback_token()
    if released == 'closed':
        target.batch.index_states = [_index_state(token=token, closed=True)]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]

    def forbidden(*args, **kwargs):
        raise AssertionError('a released index owner must not be rewound')

    monkeypatch.setattr(mixed_target, 'abort_qwen4_exp_mixed_qsa_attention', forbidden)
    owner = _CloseRecorder()
    target._owned = [owner]

    target.close()

    assert target.closed
    assert owner.closed
    assert target._owned == []
    assert target.qsa_runs == []


def test_close_releases_every_owned_resource_before_reporting_a_failed_discard(monkeypatch):
    target = _pass()
    token = _rollback_token()
    target.batch.index_states = [_index_state(token=token)]
    target.qsa_runs = [(0, _qsa_run(target.layout, token))]

    def failing(*args, **kwargs):
        raise RuntimeError('rollback failed')

    monkeypatch.setattr(mixed_target, 'abort_qwen4_exp_mixed_qsa_attention', failing)
    first, second = _CloseRecorder(), _CloseRecorder()
    target._owned = [first, second]

    with pytest.raises(RuntimeError, match='rollback failed'):
        target.close()

    assert first.closed and second.closed
    assert target._owned == []
    assert target.closed


def test_failed_release_leaves_the_pass_refusing_work_and_stays_retryable():
    target = _pass()

    class _Failing:
        def close(self):
            raise RuntimeError('free failed')

    target._owned = [_Failing()]

    with pytest.raises(RuntimeError, match='free failed'):
        target.close()
    assert not target.closed
    assert target.closing
    assert len(target._owned) == 1, 'a failed release stays pending for the retry'
    with pytest.raises(RuntimeError, match='fresh open pass'):
        target.verify()
    with pytest.raises(RuntimeError, match='unresolved verification'):
        target.commit((1, 0))

    with pytest.raises(RuntimeError, match='free failed'):
        target.close()
