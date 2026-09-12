"""EOS readback is request-local and absent from ordinary greedy requests."""
from types import SimpleNamespace as NS
import ctypes
import numpy as np
import pytest


@pytest.mark.parametrize('eos', [None, 22, 99])
def test_eos_readback_uses_candidate_offset_and_ignores_padding(monkeypatch, eos):
    from hipengine.generation import qwen35_gguf_mtp2 as m
    candidates = np.array([11,12,21,22,23],dtype=np.int32)
    top = np.array([21,22,23,99,666,666],dtype=np.int32)
    reads=[]
    def copy(host, device, nbytes=None, **kwargs):
        size = device.nbytes if nbytes is None else nbytes
        reads.append((device.ptr,size))
        ctypes.memmove(host,device.ptr,size)
    monkeypatch.setattr(m,'copy_device_to_host',copy)
    proposal=NS(request_ids=(4,9),candidate_counts=(2,3),token_ids=NS(ptr=candidates.ctypes.data))
    rows=[NS(request=NS(eos_token_id=None,ignore_eos=False)),
          NS(request=NS(eos_token_id=eos,ignore_eos=False))]
    results=[NS(request_id=4),NS(request_id=9,target_top1=NS(ptr=top.ctypes.data))]
    limit=m.Qwen35GGUFMTP2Adapter._limit_target_batch_eos(
        proposal,results,rows,(8,8),runtime=object())
    assert limit == (8,8 if eos is None else 2 if eos==22 else 4)
    assert reads == ([] if eos is None else [(candidates.ctypes.data+8,12),(top.ctypes.data,16)])
