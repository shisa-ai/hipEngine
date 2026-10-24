"""Atomic rollback lifecycle, independent of tensor layout or model identity."""
from types import SimpleNamespace

import pytest

from hipengine.speculative.transaction import SpeculativeStateJournal, SpecCycleTransaction, SpecCycleStage
from hipengine.speculative.frontier import SpecTransactionMode
from hipengine.kvcache import ResourceClaimSet, ClaimLifetime
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter, _ParoMTP2RequestState


def test_journal_restores_only_completed_captures_in_reverse_order():
    events=[]
    journal=SpeculativeStateJournal()
    assert journal.capture('provider',lambda:17,lambda value:events.append(('provider',value))) == 17
    journal.capture('target',lambda:23,lambda value:events.append(('target',value)))
    journal.rollback();journal.rollback()
    assert events == [('target',23),('provider',17)]
    assert not journal.pending
    with pytest.raises(RuntimeError,match='closed'):
        journal.capture('late',lambda:0,lambda value:None)


def test_failed_capture_never_registers_stale_restore():
    events=[];journal=SpeculativeStateJournal()
    journal.capture('provider',lambda:17,events.append)
    def fail(): raise RuntimeError('capture failed')
    with pytest.raises(RuntimeError,match='capture failed'):
        journal.capture('target',fail,lambda value:events.append('stale target'))
    journal.rollback()
    assert events == [17]


def test_restore_failure_does_not_skip_other_owner_and_can_retry():
    events=[];journal=SpeculativeStateJournal();fail=[True]
    journal.capture('provider',lambda:17,events.append)
    def restore(value):
        events.append(value)
        if fail[0]:raise RuntimeError('target restore failed')
    journal.capture('target',lambda:23,restore)
    with pytest.raises(ExceptionGroup,match='rollback'):
        journal.rollback()
    assert events == [23,17] and journal.pending
    with pytest.raises(RuntimeError,match='rollback'):
        journal.commit()
    fail[0]=False;journal.rollback()
    assert events == [23,17,23] and not journal.pending


def test_commit_releases_snapshots_without_restoring():
    journal=SpeculativeStateJournal()
    journal.capture('provider',lambda:1,lambda value:pytest.fail('restored committed state'))
    with pytest.raises(ValueError,match='duplicate'):
        journal.capture('provider',lambda:2,lambda value:None)
    journal.commit();journal.commit();journal.rollback()
    assert not journal.pending


def adapter_fixture():
    events=[]
    session=SimpleNamespace(position_arr=[91,0],context_arr=[92,1])
    session.copy_slot_state=lambda src,dst,**kw:events.append(('copy',src,dst,kw))
    def record(position,*,slot):
        session.position_arr[slot]=position;session.context_arr[slot]=position+1
    session._record_slot_position_host=record
    session.runtime=SimpleNamespace(device_synchronize=lambda:events.append(('sync',)))
    proposer=SimpleNamespace(save_state=lambda slot:('snapshot',slot),restore_state=lambda value:events.append(('provider',value)))
    owner=SimpleNamespace(generator=SimpleNamespace(),_session=session)
    return Qwen35ParoMTP2Adapter(owner),_ParoMTP2RequestState(1,proposer),session,events


def test_paro_rollback_before_target_capture_does_not_copy_stale_slot():
    adapter,state,session,events=adapter_fixture()
    state.journal=SpeculativeStateJournal()
    state.checkpoint=state.journal.capture('provider',lambda:state.proposer.save_state(0),state.proposer.restore_state)
    adapter._restore_target_and_provider(state,session)
    adapter._restore_target_and_provider(state,session)
    assert events == [('provider',('snapshot',0)),('sync',)]
    assert state.checkpoint is None and state.journal is None


def test_paro_target_checkpoint_copies_mutable_state_but_not_append_only_kv():
    adapter,state,session,events=adapter_fixture()
    state.journal=SpeculativeStateJournal()
    adapter._checkpoint_target(state,session)
    session.position_arr[0]=100;session.context_arr[0]=101
    adapter._restore_target_and_provider(state,session)
    assert events == [('copy',0,1,{'kv_rows':0}),('copy',1,0,{'kv_rows':0}),('sync',)]
    assert session.position_arr[0] == 91 and session.context_arr[0] == 92


def test_paro_failed_target_capture_never_restores_partial_backup():
    adapter,state,session,events=adapter_fixture()
    state.journal=SpeculativeStateJournal()
    state.journal.capture('provider',lambda:17,state.proposer.restore_state)
    def fail(*args,**kwargs):raise RuntimeError('copy failed')
    session.copy_slot_state=fail
    with pytest.raises(RuntimeError,match='copy failed'):adapter._checkpoint_target(state,session)
    adapter._restore_target_and_provider(state,session)
    assert events == [('provider',17),('sync',)]


def test_paro_cancellation_restores_both_owners_and_outer_rollback_is_idempotent():
    adapter,state,session,events=adapter_fixture()
    adapter._states[1]=state
    state.journal=SpeculativeStateJournal()
    state.checkpoint=state.journal.capture('provider',lambda:17,state.proposer.restore_state)
    adapter._checkpoint_target(state,session)
    transaction=SpecCycleTransaction(
        operation_id='cycle',transaction_id=1,cycle_id=1,request_ids=(1,),
        reserved_claims=ResourceClaimSet.from_mapping('cycle',{'target':1},lifetime=ClaimLifetime.TRANSACTION),
        pre_target_cursors=(91,),pre_rng_counters=(0,),
        target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        target_owner='target',target_checkpoint_ids=('target:91',),
        pre_provider_cursors=(91,),provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        provider_owner='provider',provider_checkpoint_ids=('provider:91',),
        target_open=True,provider_open=True,
    )
    result=adapter._cancelled_result(transaction,state,session)
    assert result.stage == SpecCycleStage.CANCELLED
    assert result.transaction.rolled_back and result.cancelled_request_ids == (1,)
    adapter.rollback_cycle(SimpleNamespace(request_ids=(1,)),None,RuntimeError('cancel'))
    assert events == [('copy',0,1,{'kv_rows':0}),('copy',1,0,{'kv_rows':0}),('provider',17),('sync',)]


def test_paro_failed_target_restore_still_restores_provider_and_does_not_recycle():
    adapter,state,session,events=adapter_fixture()
    adapter._states[1]=state
    state.journal=SpeculativeStateJournal()
    state.checkpoint=state.journal.capture('provider',lambda:17,state.proposer.restore_state)
    adapter._checkpoint_target(state,session)
    def fail(*args,**kwargs):raise RuntimeError('restore copy failed')
    session.copy_slot_state=fail
    with pytest.raises(ExceptionGroup,match='rollback'):
        adapter._restore_target_and_provider(state,session)
    assert ('provider',17) in events and events[-1] == ('sync',)
    state.proposer.closed=False
    state.proposer.close=lambda:events.append(('close',))
    adapter._drop_request(1,disable=False,recycle=True)
    assert events[-1] == ('close',) and not adapter._proposer_pool
