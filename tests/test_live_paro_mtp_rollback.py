"""Device rollback equality after public PARO MTP failure injection."""
from contextlib import closing
import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from hipengine.speculative.mtp_native import NativeMtpChainProposer


def _hip_available():
    try:
        runtime=ctypes.CDLL('libamdhip64.so');count=ctypes.c_int()
        return runtime.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        return False


pytestmark=pytest.mark.skipif(not _hip_available(),reason='HIP runtime is unavailable')


def _digest(runtime,ptr,nbytes):
    array=np.empty(nbytes,dtype=np.uint8)
    runtime.memcpy(host_array_ptr(array),int(ptr),nbytes,HipMemcpyKind.DEVICE_TO_HOST)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _snapshot(session,proposer):
    session.runtime.device_synchronize()
    runtime=session.runtime
    position=int(session.position_arr[0])
    values={'position':position,'context':int(session.context_arr[0]),
            'draft_cursor':proposer.cache_len,'draft_position':proposer.position}
    for name in ('batch_hidden','batch_next_hidden'):
        tensor=getattr(session,name)
        values[name]=_digest(runtime,tensor.ptr,int(session.config.hidden_size)*tensor.dtype.itemsize)
    for layer,(conv,recurrent,conv_buf,recurrent_buf,*_) in session.linear_states.items():
        for name,tensor,buffer in [('conv',conv,conv_buf),('recurrent',recurrent,recurrent_buf)]:
            values[f'{layer}:{name}']=_digest(runtime,buffer.ptr,int(np.prod(tensor.shape))*tensor.dtype.itemsize)
    for layer,(key,value,key_buf,value_buf) in session.full_caches.items():
        for name,tensor,buffer in [('key',key,key_buf),('value',value,value_buf)]:
            values[f'{layer}:{name}']=_digest(runtime,buffer.ptr,position*int(np.prod(tensor.shape[2:]))*tensor.dtype.itemsize)
    for name in ('position_buf','context_buf','token_id_buf'):
        values[name]=_digest(runtime,getattr(session,name).ptr,8)
    for name,width,size in [('final_hidden_buf',proposer.hidden,2),('key_cache_buf',proposer.cache_len*proposer.kv_features,4),('value_cache_buf',proposer.cache_len*proposer.kv_features,2)]:
        values[name]=_digest(runtime,getattr(proposer,name).ptr,width*size)
    return values


_PROVIDER_KEYS=('draft_cursor','draft_position','final_hidden_buf',
                'key_cache_buf','value_cache_buf')


def _snapshot_diff(before,after):
    return {key:(value,after.get(key)) for key,value in before.items()
            if after.get(key) != value}


def _result_probe(proposer):
    try:
        proposer.device_candidate_token_ids()
    except RuntimeError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize('length',[1,37,511,1025,4096])
@pytest.mark.parametrize('failure_stage',['target','provider'])
def test_public_failure_restores_mutable_state_cursors_and_live_kv(monkeypatch,length,failure_stage):
    model=Path(os.environ.get('HIPENGINE_TEST_PARO_MTP_MODEL','/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16'))
    if not model.is_dir():pytest.skip('packed PARO model with BF16 MTP weights is unavailable')
    original_propose=Qwen35ParoMTP2Adapter.propose_batch
    original_capture=Qwen35ParoMTP2Adapter._checkpoint_target
    original_restore=Qwen35ParoMTP2Adapter._restore_target_and_provider
    original_verify=Qwen35ParoMTP2Adapter._verify_target
    original_advance=NativeMtpChainProposer.advance_with_target_hidden
    pending={};observed=[];injected=[];injection_phase={}
    phases={'target_checkpoint':False,'target_verified':False}

    def propose(adapter,plan,*args,**kwargs):
        state=adapter._states[plan.request_ids[0]]
        # A new proposal starts a new phase. Target-hidden advances during
        # drafting must not be mistaken for post-target provider reconciliation.
        phases['target_checkpoint']=False
        phases['target_verified']=False
        # Provider rollback restores this pre-draft point, not the
        # post-proposal state that exists when the target checkpoint is taken.
        before=_snapshot(adapter.owner._session,state.proposer)
        pending['provider']={key:before[key] for key in _PROVIDER_KEYS}
        return original_propose(adapter,plan,*args,**kwargs)

    def capture(adapter,state,session):
        original_capture(adapter,state,session)
        snapshot=_snapshot(session,state.proposer)
        # Target checkpoint is pre-verify; provider keys are the pre-draft
        # rollback boundary because the two owners restore different points.
        snapshot.update(pending['provider'])
        pending['state']=snapshot
        pending['session']=session
        pending['proposer']=state.proposer
        phases['target_checkpoint']=True

    def restore(adapter,state,session):
        # Record instead of asserting in this hook: the engine catches a
        # rollback failure, recovers the request, and would otherwise turn a
        # real mismatch into an empty observation list.
        observation=None
        journal_before=state.journal is not None
        if injected and not observed:
            observation={}
            observed.append(observation)
        try:
            original_restore(adapter,state,session)
        except BaseException as exc:
            if observation is not None:
                observation['restore_error']=exc
            raise
        if observation is not None:
            try:
                after=_snapshot(session,state.proposer)
                observation.update(
                    journal_before=journal_before,
                    journal_after=state.journal is None,
                    result_valid=bool(state.proposer.result_valid),
                    candidate_error=_result_probe(state.proposer),
                    mismatches=_snapshot_diff(pending['state'],after),
                )
            except BaseException as exc:
                observation['observe_error']=exc
                raise

    def verify(*args,**kwargs):
        result=original_verify(*args,**kwargs)
        phases['target_verified']=True
        if failure_stage == 'target' and not injected:
            injected.append(True);injection_phase.update(phases)
            raise RuntimeError('injected after target verification')
        return result

    def advance(proposer,*args,**kwargs):
        result=original_advance(proposer,*args,**kwargs)
        # Only the post-verify provider reconciliation may inject here.
        # Drafting advances run before the target checkpoint and verification.
        if (failure_stage == 'provider' and phases['target_checkpoint']
                and phases['target_verified'] and not injected):
            injected.append(True);injection_phase.update(phases)
            raise RuntimeError('injected after provider update')
        return result

    before=memory_stats()
    with closing(LLM(str(model),backend='auto',quant='w4_paro',max_active_requests=1,
                     max_sequence_length=4352,prefix_cache='off')) as llm:
        params=SamplingParams(max_tokens=8,temperature=0,ignore_eos=True,
                              speculative_mtp=True,kv_storage='bf16')
        baseline=llm.generate_detailed([[9707]*length],params)[0]
        with monkeypatch.context() as patch:
            patch.setattr(Qwen35ParoMTP2Adapter,'propose_batch',propose)
            patch.setattr(Qwen35ParoMTP2Adapter,'_checkpoint_target',capture)
            patch.setattr(Qwen35ParoMTP2Adapter,'_restore_target_and_provider',restore)
            patch.setattr(Qwen35ParoMTP2Adapter,'_verify_target',staticmethod(verify))
            patch.setattr(NativeMtpChainProposer,'advance_with_target_hidden',advance)
            result=llm.generate_detailed([[9707]*length],params)[0]
        assert len(result.generated_token_ids) == 8
        recovered=llm.generate_detailed([[9707]*length],params)[0]
    assert recovered.generated_token_ids == baseline.generated_token_ids
    assert injected == [True]
    assert injection_phase == {'target_checkpoint': True, 'target_verified': True}
    assert pending.get('provider') and pending.get('state')
    assert set(pending['provider']) <= set(pending['state'])
    assert len(observed) == 1
    observation=observed[0]
    assert observation.get('restore_error') is None
    assert observation.get('observe_error') is None
    assert observation['journal_before'] is True
    assert observation['journal_after'] is True
    assert observation['result_valid'] is False
    assert observation['candidate_error'] is not None
    assert 'result-producing advance' in observation['candidate_error']
    assert observation['mismatches'] == {}
    after=memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
