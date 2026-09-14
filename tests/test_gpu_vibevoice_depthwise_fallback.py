"""Strict staged fallback must preserve the fused accumulator/epilogue contract."""
import numpy as np
import pytest
from tests._rocm_guard import hip_runtime_available
if not hip_runtime_available():
    pytest.skip('no usable HIP runtime',allow_module_level=True)


def test_depthwise_staged_fallback():
    from hipengine.kernels.vibevoice import resolve_vibevoice_kernels,resolve_vibevoice_kernel
    from hipengine.core.memory import free,copy_device_to_host,host_array_ptr
    from hipengine.runtime.vibevoice_encoder import _upload_u16
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    ops=resolve_vibevoice_kernels()
    fused=resolve_vibevoice_kernel(ops.backend,'vv_depthwise_conv_bf16','fused')
    staged=resolve_vibevoice_kernel(ops.backend,'vv_depthwise_conv_bf16','strict')
    assert fused is not staged
    rng=np.random.default_rng(82)
    rows,c,k=13,64,7
    host=[rng.normal(size=shape).astype(np.float32) for shape in ((k-1,c),(rows,c),(rows,c),(c,k),(c,),(c,))]
    buffers=[_upload_u16(f32_to_bf16_bits(x)) for x in host]
    outputs=[_upload_u16(np.zeros((rows,c),dtype=np.uint16)) for _ in range(2)]
    try:
        results=[]
        for fn,out in zip((fused,staged),outputs):
            fn(*(x.ptr for x in buffers),out.ptr,k-1,rows,c,k)
            result=np.empty((rows,c),dtype=np.uint16)
            copy_device_to_host(host_array_ptr(result),out,result.nbytes)
            results.append(result)
        np.testing.assert_array_equal(*results)
        from hipengine.kernels.cpu_reference.vibevoice_asr import vibevoice_causal_conv1d
        from scripts.quant_quality.metrics import per_row_metrics
        quantized=[(f32_to_bf16_bits(x).astype(np.uint32)<<16).view(np.float32) for x in host]
        prefix,normed,resid,w,b,gamma=quantized
        conv=vibevoice_causal_conv1d(np.concatenate((prefix,normed)).T[None],w[:,None,:],b,groups=c)
        reference=conv[0,:,k-1:].T*gamma+resid
        actual=(results[1].astype(np.uint32)<<16).view(np.float32)
        metrics=per_row_metrics(reference,actual,np.argmax(reference,axis=1),top_k=5)
        assert metrics['kl_nats'].max() <= .05
        assert metrics['top1_equal'].mean() >= .9
    finally:
        for buf in buffers+outputs: free(buf)
