"""KV writes/attention obey nonidentity physical mapping and eviction."""
import numpy as np
import pytest
from tests._rocm_guard import hip_runtime_available
if not hip_runtime_available():
    pytest.skip('no usable HIP runtime',allow_module_level=True)

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.core.memory import malloc,free,copy_host_array_to_device,copy_device_to_host,host_array_ptr
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.vibevoice.encoder import f32_to_bf16_bits


def test_spans_scatter_and_mask():
    from hipengine.kernels.hip_gfx1100.vibevoice.encoder import vv_kv_write_spans, vv_attention_spans
    owned=[]
    def put(x):
        x=np.ascontiguousarray(x); b=malloc(x.nbytes);owned.append(b);copy_host_array_to_device(b,x);return b
    def tensor(x,dtype):
        b=put(x);return Tensor.from_handle(b.ptr,x.shape,dtype,Device('hip'))
    try:
        mapping=np.array([2,0,3,1],np.int32)
        positions=np.argsort(mapping).astype(np.int64)
        spans=KVLiveSpans(tensor(mapping,'int32'),tensor(np.full(4,4,np.int64),'int64'),4,
            tensor(positions,'int64'),tensor(np.array([False,False,False,True]),'bool'),'bf16',
            row_positions=tensor(np.arange(4,dtype=np.int64),'int64'),span_role='prefill')
        keys=np.array([[[1.,0]],[[0,1]],[[8,8]],[[1,1]]],np.float32)
        values=np.array([[[1.,2]],[[3,4]],[[99,99]],[[5,6]]],np.float32)
        k,v=put(f32_to_bf16_bits(keys)),put(f32_to_bf16_bits(values))
        kc,vc=put(np.zeros((4,1,2),np.uint16)),put(np.zeros((4,1,2),np.uint16))
        vv_kv_write_spans(k.ptr,v.ptr,kc.ptr,vc.ptr,spans,4,1,2)
        q=put(np.ones((4,1,2),np.float32));out=put(np.zeros((4,1,2),np.float32))
        vv_attention_spans(q.ptr,kc.ptr,vc.ptr,out.ptr,spans,4,1,1,2,1.)
        got=np.empty((4,1,2),np.float32);copy_device_to_host(host_array_ptr(got),out)
        ref=[]
        for pos in range(4):
            live=[i for i in range(pos+1) if i != 2]
            scores=keys[live,0].sum(axis=1);weights=np.exp(scores-scores.max());weights/=weights.sum()
            ref.append(weights @ values[live,0])
        np.testing.assert_allclose(got[:,0],ref,atol=1e-6,rtol=1e-6)
        saved=np.empty((4,1,2),np.uint16);copy_device_to_host(host_array_ptr(saved),vc)
        np.testing.assert_array_equal(saved[3],0)
        np.testing.assert_array_equal(saved[2],f32_to_bf16_bits(values[0]))
    finally:
        for b in reversed(owned):free(b)
