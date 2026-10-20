"""Serving uses one shared target pass for a prepared MTP row and AR peer."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPProposal, Qwen4ExpMTPTargetRows
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetResult
from hipengine.speculative import SpecTransactionMode
from tests._qwen4exp_ar_batch_fixtures import FakeBatchRunner, FakeC1Runner, FakeMTPProvider, make_generator
from tests.test_unit_qwen4exp_ar_batch_decode import _register, _prefill_work, _decode


def test_explicit_mtp_prefill_imports_and_root_does_not_run_a_second_target(monkeypatch):
    monkeypatch.setattr(FakeBatchRunner, 'supports_mixed_target', True, raising=False)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET', '1')
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner', FakeBatchRunner)
    g = make_generator()
    g._speculative_provider = FakeMTPProvider()
    n = g.create_resident_model_runner(capacity=1)
    try:
        n._prepare_mtp()
        _register(n, 1, (1, 2))
        n.register_speculative_request(1, 2)
        n.reserve_admission(SimpleNamespace(request_id=1))
        assert n._row(1).mixed_mtp
        n.prefill_batch(_prefill_work(1, (1, 2)), commit=True)
        assert n._row(1).runner is None
        assert n._ar_batch.imported(1)
        assert n._row(1).mtp_steps is None
        assert g._speculative_provider.iter_calls == []
        root = _decode(n, (1,))
        assert len(root) == 1
        assert len(n._ar_batch.runner.step_calls) == 0
        assert n._row(1).generated_ids == [3]
    finally:
        n.close()


def test_explicit_mtp_keeps_separate_execution_unless_shared_target_is_selected(monkeypatch):
    # The shared pass measured slower than separate execution, so an explicit
    # MTP row keeps its c1 runner by default even on a mixed-capable owner.
    monkeypatch.setattr(FakeBatchRunner, 'supports_mixed_target', True, raising=False)
    monkeypatch.delenv('HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET', raising=False)
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner', FakeBatchRunner)
    g = make_generator()
    g._speculative_provider = FakeMTPProvider()
    n = g.create_resident_model_runner(capacity=1)
    try:
        n._prepare_mtp()
        _register(n, 1, (1, 2))
        n.register_speculative_request(1, 2)
        n.reserve_admission(SimpleNamespace(request_id=1))
        assert not n._row(1).mixed_mtp
        assert not n._ar_batch.owns(1)
    finally:
        n.close()


def _cycle_fixture(monkeypatch, *, truth=(12, 29, 31, 18), ar_max_tokens=8):
    from hipengine.generation.qwen4_exp_serving_mixed import Qwen4ExpServingMixed
    from hipengine.generation.qwen4_exp_gguf import _Qwen4ExpServingRow
    from hipengine.generation.registry import GenerationRequest
    g = make_generator()
    rows = [_Qwen4ExpServingRow(i, i - 1, GenerationRequest(prompts=((1, 2),), max_tokens=8,
                      temperature=0.0, top_p=1.0, ignore_eos=True), (1, 2), 0.0) for i in (1, 2)]
    for row in rows:
        row.generated_ids = [11]
        row.next_result = SimpleNamespace(token_id=17, logits=None)
    draft = SimpleNamespace(position=2)
    draft.snapshot = lambda: SimpleNamespace(position=draft.position)
    draft.restore = lambda snap: setattr(draft, 'position', snap.position)
    draft.trim = lambda pos: setattr(draft, 'position', pos)
    draft.forward = lambda *args, **kwargs: setattr(draft, 'position', draft.position + 1)
    rows[0].mixed_mtp = True
    rows[0].mtp_budget = 2
    rows[0].mixed_root_hidden = np.zeros(4)
    rows[0].draft = draft
    batch = SimpleNamespace(_positions=[2, 2])
    owner = SimpleNamespace(runner=batch, _leases={1: SimpleNamespace(slot=0, owner=1),
                                                 2: SimpleNamespace(slot=1, owner=2)})
    from hipengine.generation.qwen4_exp_gguf import _token_finish
    rows[1].request = replace(rows[1].request, max_tokens=ar_max_tokens)
    native = Qwen4ExpServingMixed()
    native.generator, native._ar_batch = g, owner
    native._row = lambda rid: rows[rid - 1]
    native._check_request_live = lambda row: None
    native._finish = lambda row: _token_finish(row.request, row.generated_ids, row.grammar,
                                              g.tokenizer.eos_token_id)
    native.request_cancel_reason = lambda rid: None
    calls = []
    def proposal(*args, **kwargs):
        calls.append('draft')
        draft.position += 2
        return Qwen4ExpMTPProposal(2, 11, (12, 13))
    monkeypatch.setattr('hipengine.generation.qwen4_exp_serving_mixed.prepare_proposal', proposal)
    class Target:
        def __init__(self, batch, layout):
            self.layout = layout
            calls.append(('target', layout.physical_request_width, layout.total_rows))
        def verify(self):
            # Root matches c1; c1 rejects c2. The remaining row is never sampled.
            positions = [truth[row.semantic_row] for row in self.layout.rows]
            return Qwen4ExpMixedTargetResult(self.layout, np.array(positions),
                                             np.zeros((len(positions), 64)), np.zeros((len(positions), 4)))
        def commit(self, depths):
            calls.append(('commit', tuple(depths)))
            for request, depth in zip(self.layout.requests, depths, strict=True):
                batch._positions[request.slot] += depth + 1
        def abort(self):
            calls.append('abort')
        def close(self):
            calls.append('close')
    monkeypatch.setattr('hipengine.generation.qwen4_exp_serving_mixed.Qwen4ExpMixedTargetPass', Target)
    plan = SimpleNamespace(request_ids=(1, 2), candidate_counts=(2, 0), operation_id='test', cycle_id=1,
                           target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
                           provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND)
    return SimpleNamespace(native=native, rows=rows, plan=plan, batch=batch, calls=calls, draft=draft)


def test_containment_refuses_device_fault_even_with_partial_wrapper():
    from hipengine.core.hip import HipError
    from hipengine.generation.qwen4_exp_serving_mixed import (
        Qwen4ExpMixedCycleError, Qwen4ExpServingMixed,
    )
    error = Qwen4ExpMixedCycleError((1,), HipError(700, 'device fault'), mutation='partial')
    assert Qwen4ExpServingMixed().contain_execution_failure(
        error, phase='speculative_cycle', request_ids=(1,), work_kind='verify_chain',
    ) is None


def test_shared_cycle_proposes_once_scores_one_pass_and_commits_peer_once(monkeypatch):
    f = _cycle_fixture(monkeypatch)
    native, rows, plan, batch, calls, draft = (f.native, f.rows, f.plan, f.batch, f.calls, f.draft)
    result = native._execute_mixed_cycle(plan)
    assert calls == ['draft', ('target', 2, 4), ('commit', (1, 0)), 'close']
    assert result.committed_output_ids == ((12, 29), (17,))
    assert rows[0].generated_ids == [11, 12, 29]
    assert rows[1].generated_ids == [11, 17]
    assert rows[1].next_result.token_id == 18
    assert draft.position == 4
    assert native._last_mixed_pass['request_width'] == 2
    assert native._last_mixed_pass['target_rows'] == 4
    assert native._last_mixed_pass['request_ids'] == [1, 2]
    assert native._last_mixed_pass['accepted_draft_tokens'] == [1, 0]
    stages = native._last_mixed_pass['stages_ms']
    assert set(stages) == {'checkpoint', 'proposal', 'target_setup', 'verify',
                           'acceptance_catchup', 'commit', 'cleanup'}
    assert all(value >= 0 for value in stages.values())
    assert sum(stages.values()) <= native._last_mixed_pass['wall_ms']


@pytest.mark.parametrize('truth,depth,tokens', [
    ((28, 29, 31, 18), 0, (28,)),
    ((12, 29, 31, 18), 1, (12, 29)),
    ((12, 13, 31, 18), 2, (12, 13, 31)),
])
def test_shared_acceptance_publishes_only_consumed_prefix(monkeypatch, truth, depth, tokens):
    f = _cycle_fixture(monkeypatch, truth=truth)
    result = f.native._execute_mixed_cycle(f.plan)
    assert result.committed_output_ids == (tokens, (17,))
    assert ('commit', (depth, 0)) in f.calls
    assert f.batch._positions == [3 + depth, 3]
    assert f.draft.position == 3 + depth


def test_terminal_ar_peer_does_not_enter_target_or_advance_cursor(monkeypatch):
    f = _cycle_fixture(monkeypatch, ar_max_tokens=2)
    result = f.native._execute_mixed_cycle(f.plan)
    assert ('target', 1, 3) in f.calls
    assert ('commit', (1,)) in f.calls
    assert f.batch._positions == [4, 2]
    assert result.accept_result.target_cursor_deltas == (2, 0)
    assert result.accept_result.finish_reasons == (None, 'length')
    assert result.committed_output_ids == ((12, 29), (17,))
    assert f.native._last_mixed_pass['request_ids'] == [1]
    assert f.native._last_mixed_pass['candidate_counts'] == [2]
    assert f.native._last_mixed_pass['committed_input_rows'] == [2]
    assert f.native._last_mixed_pass['planned_candidate_counts'] == [2, 0]
    assert f.rows[1].mixed_widths == {}
    assert f.rows[1].mixed_target_rows == 0
    assert f.rows[0].mixed_widths == {1: 1}


def test_cancellation_before_commit_restores_controls_and_draft_for_peer(monkeypatch):
    from hipengine.generation.qwen4_exp_serving_mtp import _RequestRetired
    from hipengine.speculative import SpecCycleStage
    from hipengine.generation.sampling import RowSamplingState
    f = _cycle_fixture(monkeypatch)
    for row in f.rows:
        row.sampling_state = RowSamplingState(prompt_tokens=(1, 2), generated_tokens=(11,),
                                              step_index=1, rng_draw_count=1, seed=row.request_id)
        row.request = replace(row.request, temperature=0.7, logprobs=True)
        row.next_result = SimpleNamespace(token_id=17, logits=np.zeros(64, dtype=np.float32))
    snapshots = [row.sampling_state.clone() for row in f.rows]
    visits = []
    def check(row):
        visits.append(row.request_id)
        if len(visits) == 3:
            raise _RequestRetired()
    f.native._check_request_live = check
    f.native.request_cancel_reason = lambda rid: 'disconnect' if rid == 1 else None
    result = f.native._execute_mixed_cycle(f.plan)
    assert result.stage is SpecCycleStage.CANCELLED
    assert result.cancelled_request_ids == (1,)
    assert result.transaction.rolled_back
    assert not any(isinstance(call, tuple) and call[0] == 'commit' for call in f.calls)
    assert f.batch._positions == [2, 2]
    assert f.draft.position == 2
    assert all(row.generated_ids == [11] for row in f.rows)
    for row, state in zip(f.rows, snapshots, strict=True):
        assert row.sampling_state.rng_draw_count == state.rng_draw_count
        assert row.sampling_state.step_index == state.step_index
        assert row.sampling_state.generated_tokens == state.generated_tokens
        assert row.sampling_state._rng.bit_generator.state == state._rng.bit_generator.state
        assert not row.token_logprobs


def test_failed_commit_is_quiesced_and_scoped_to_participating_rows(monkeypatch):
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: f.calls.append('synchronize'))
    target_type = module.Qwen4ExpMixedTargetPass
    def fail_commit(self, depths):
        raise RuntimeError('injected commit copy failure')
    monkeypatch.setattr(target_type, 'commit', fail_commit)
    with pytest.raises(module.Qwen4ExpMixedCycleError, match='injected') as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert f.calls[-3:] == ['abort', 'synchronize', 'close']
    assert f.draft.position == 2
    assert all(row.generated_ids == [11] for row in f.rows)
    assert caught.value.request_ids == (1, 2)
    assert caught.value.mutation == 'unknown'
    # The pass cannot establish how much of the commit reached device state, and
    # ``unknown`` is never a successful containment claim, so the failure stays
    # fatal for the whole service instead of retiring these two rows.
    assert f.native.contain_execution_failure(caught.value, phase='speculative_cycle',
                                              request_ids=(1, 2, 3), work_kind='verify_chain') is None
    assert f.native.contain_execution_failure(RuntimeError('unknown'), phase='decode',
                                              request_ids=(1,), work_kind='decode') is None


@pytest.mark.parametrize('failure', [
    # The shape a ``_preflight_commit`` host refusal takes.
    ValueError('mixed target live position changed before commit'),
    # The shape a partially applied device copy takes.
    RuntimeError('injected commit copy failure'),
])
def test_commit_preflight_refusal_and_copy_failure_are_equally_unresolved(monkeypatch, failure):
    """A commit failure is fatal whatever shape it arrives in.

    ``Qwen4ExpMixedTargetPass.commit`` runs its host preflight and its device
    copies inside one call, so the serving layer cannot narrow a failure there
    to ``none``. The only signal that separates them is the pass's own
    ``resolved`` flag, which this layer does not own: basing a containment
    decision on it would make an over-narrow claim the moment that ordering
    changes. Both shapes therefore stay ``unknown`` and stay fatal.
    """
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: None)
    monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'commit',
                        lambda self, depths: (_ for _ in ()).throw(failure))
    with pytest.raises(module.Qwen4ExpMixedCycleError) as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert caught.value.mutation == 'unknown'
    assert caught.value.cause is failure
    assert f.native.contain_execution_failure(caught.value, phase='speculative_cycle',
                                              request_ids=(1, 2), work_kind='verify_chain') is None


@pytest.mark.parametrize('stage', ['proposal', 'verify', 'commit'])
def test_hip_error_never_becomes_a_containment_claim(monkeypatch, stage):
    """A device fault stops the service; it never retires only some rows.

    A ``HipError`` means the device itself reported a fault, so shared device
    state is unproven even when the pass had not reached its canonical commit
    yet. ``stage='proposal'`` covers draft device work, ``verify`` the
    provisional target rows, and ``commit`` the published prefix.
    """
    from hipengine.core.hip import HipError
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: None)
    fault = HipError(719, 'unspecified launch failure')
    def raise_fault(*args, **kwargs):
        raise fault
    if stage == 'proposal':
        monkeypatch.setattr(module, 'prepare_proposal', raise_fault)
    elif stage == 'verify':
        monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'verify', raise_fault)
    else:
        monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'commit', raise_fault)
    with pytest.raises(module.Qwen4ExpMixedCycleError) as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert caught.value.mutation == 'unknown'
    assert caught.value.cause is fault
    assert f.native.contain_execution_failure(caught.value, phase='speculative_cycle',
                                              request_ids=(1, 2), work_kind='verify_chain') is None


def test_pre_commit_failure_stays_contained_as_partial(monkeypatch):
    """Preserve request-scoped containment for a pass that never reached commit."""
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: f.calls.append('synchronize'))
    monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'verify',
                        lambda self: (_ for _ in ()).throw(RuntimeError('injected verify failure')))
    with pytest.raises(module.Qwen4ExpMixedCycleError, match='injected') as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert f.calls[-3:] == ['abort', 'synchronize', 'close']
    assert f.draft.position == 2
    assert all(row.generated_ids == [11] for row in f.rows)
    assert caught.value.mutation == 'partial'
    failure = f.native.contain_execution_failure(caught.value, phase='speculative_cycle',
                                                request_ids=(1, 2, 3), work_kind='verify_chain')
    assert failure.request_ids == (1, 2)
    assert failure.mutation == 'partial'


def test_failed_target_does_not_retire_terminal_ar_peer(monkeypatch):
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch, ar_max_tokens=2)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: None)
    monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'commit',
                        lambda self, depths: (_ for _ in ()).throw(RuntimeError('copy failed')))
    with pytest.raises(module.Qwen4ExpMixedCycleError) as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert caught.value.request_ids == (1,)
    assert f.rows[1].generated_ids == [11]
    assert f.rows[1].next_result.token_id == 17


def test_failed_cleanup_remains_fatal_even_after_quiesced_copy_failure(monkeypatch):
    from hipengine.generation import qwen4_exp_serving_mixed as module
    f = _cycle_fixture(monkeypatch)
    f.batch.runtime = SimpleNamespace(device_synchronize=lambda: None)
    def fail(*args):
        raise RuntimeError('copy failed')
    def fail_close(*args):
        raise RuntimeError('free failed')
    monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'commit', fail)
    monkeypatch.setattr(module.Qwen4ExpMixedTargetPass, 'close', fail_close)
    with pytest.raises(RuntimeError, match='free failed') as caught:
        f.native._execute_mixed_cycle(f.plan)
    assert any('copy failed' in note for note in caught.value.__notes__)
    assert f.native.contain_execution_failure(caught.value, phase='decode',
                                              request_ids=(1,2),work_kind='decode') is None


def test_admission_counts_free_drafts_across_selected_rows(monkeypatch):
    monkeypatch.setattr('hipengine.generation.qwen4_exp_serving_mtp.Qwen4ExpGGUFMTPDraftRunner',
                        lambda *args, **kwargs: type(FakeMTPProvider().draft)())
    monkeypatch.setattr(FakeBatchRunner, 'supports_mixed_target', True, raising=False)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET', '1')
    monkeypatch.setattr('hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner', FakeBatchRunner)
    g = make_generator()
    g._speculative_provider = FakeMTPProvider()
    n = g.create_resident_model_runner(capacity=2)
    try:
        n._prepare_mtp()
        for rid in (1, 2):
            _register(n, rid, (1, 2))
            n.register_speculative_request(rid, 2)
        n._available_drafts[:] = n._available_drafts[:1]
        assert n.plan_admission([SimpleNamespace(request_id=rid) for rid in (1, 2)], max_items=2) == (1,)
    finally:
        n.close()
