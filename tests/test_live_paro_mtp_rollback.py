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


@pytest.mark.parametrize('length',[1,37,511,1025,4096])
@pytest.mark.parametrize('failure_stage',['target','provider'])
def test_public_failure_restores_mutable_state_cursors_and_live_kv(monkeypatch,length,failure_stage):
    model=Path(os.environ.get('HIPENGINE_TEST_PARO_MTP_MODEL','/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16'))
    if not model.is_dir():pytest.skip('packed PARO model with BF16 MTP weights is unavailable')
    original_capture=Qwen35ParoMTP2Adapter._checkpoint_target
    original_restore=Qwen35ParoMTP2Adapter._restore_target_and_provider
    original_verify=Qwen35ParoMTP2Adapter._verify_target
    original_advance=NativeMtpChainProposer.advance_with_target_hidden
    pending=[];checked=[];injected=[]

    def capture(adapter,state,session):
        original_capture(adapter,state,session)
        pending.append((session,state.proposer,_snapshot(session,state.proposer)))

    def restore(adapter,state,session):
        active=state.journal is not None
        original_restore(adapter,state,session)
        if active:
            assert not state.proposer.result_valid
            with pytest.raises(RuntimeError, match='result-producing advance'):
                state.proposer.device_candidate_token_ids()
            before=pending[-1][2];after=_snapshot(session,state.proposer)
            assert before == after, {key:(before[key],after[key]) for key in before if before[key]!=after[key]}
            checked.append(True)

    def verify(*args,**kwargs):
        result=original_verify(*args,**kwargs)
        if failure_stage == 'target' and not injected:
            injected.append(True);raise RuntimeError('injected after target verification')
        return result

    def advance(proposer,*args,**kwargs):
        result=original_advance(proposer,*args,**kwargs)
        if failure_stage == 'provider' and pending and not injected:
            injected.append(True);raise RuntimeError('injected after provider update')
        return result

    monkeypatch.setattr(Qwen35ParoMTP2Adapter,'_checkpoint_target',capture)
    monkeypatch.setattr(Qwen35ParoMTP2Adapter,'_restore_target_and_provider',restore)
    monkeypatch.setattr(Qwen35ParoMTP2Adapter,'_verify_target',staticmethod(verify))
    monkeypatch.setattr(NativeMtpChainProposer,'advance_with_target_hidden',advance)
    before=memory_stats()
    with closing(LLM(str(model),backend='auto',quant='w4_paro',max_active_requests=1,
                     max_sequence_length=4352,prefix_cache='off')) as llm:
        result=llm.generate_detailed([[9707]*length],SamplingParams(max_tokens=8,temperature=0,
                                     ignore_eos=True,speculative_mtp=True,kv_storage='bf16'))[0]
        assert len(result.generated_token_ids) == 8
    assert injected == checked == [True]
    after=memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
