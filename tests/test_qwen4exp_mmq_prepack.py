import ctypes
import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as mmq
from tests.test_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download
from tests.test_qwen4exp_pf1_dense_parity import make_q8_0_weight_large

PARENT = "gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
CANDIDATE = "gguf_q8_0_mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out"


def pack_reference(raw, n, k):
    blocks = np.ascontiguousarray(raw).reshape(n,k//256,8,34)
    padded_n = (n+127)//128*128
    packed = np.zeros((k//256,padded_n,304), dtype=np.uint8)
    packed[:,:n,:256] = blocks[:,:,:,2:].reshape(n,k//256,256).transpose(1,0,2)
    scales = blocks[:,:,:,:2].copy().view(np.float16).reshape(n,k//256,8).astype(np.float32)
    packed[:,:n,256:288] = np.ascontiguousarray(scales.transpose(1,0,2)).view(np.uint8)
    packed[:,n:] = packed[:,n-1:n]
    return packed


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_registry():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
        variant=CANDIDATE.removeprefix("gguf_q8_0_"),
    ) is getattr(mmq,CANDIDATE)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n", [(65,256,80), (257,10240,320), (512,2560,10240)])
def test_exact_prepacked(rows,k,n):
    runtime = get_hip_runtime()
    library = mmq.build_gguf_q8_0_mmq_prefill(load=True)
    allocations = []
    try:
        x = np.random.default_rng(9533).normal(0,.2,(rows,k)).astype(np.float32)
        w = make_q8_0_weight_large(n,k)
        dx,dw,dp = [_upload(v,runtime,allocations) for v in (x,w,pack_reference(w,n,k))]
        d4 = _alloc((mmq.q8_mmq_d4x3_nbytes(rows,k),),np.uint8,runtime,allocations)
        count = _alloc((1,),np.int32,runtime,allocations)
        indices = _alloc((rows*n,),np.int32,runtime,allocations)
        out = _alloc((rows,n),np.float32,runtime,allocations)
        mmq.gguf_q8_0_mmq128_quantize_f32_d4x3(dx.ptr,d4.ptr,rows,k,library=library,runtime=runtime)
        results = []
        for name,weight in ((PARENT,dw),(CANDIDATE,dp),(CANDIDATE,dp)):
            runtime.memset(count.ptr,0,4)
            getattr(mmq,name)(d4.ptr,weight.ptr,out.ptr,count.ptr,indices.ptr,
                rows*n,0.,rows,k,n,library=library,runtime=runtime)
            raw_result = _download(out,(rows,n),np.float32,runtime)
            mmq.gguf_q8_0_mmq128_sparse_exact_correct_f32(
                dx.ptr,dw.ptr,out.ptr,count.ptr,indices.ptr,rows*n,rows,k,n,
                library=library,runtime=runtime)
            results.append((raw_result,_download(out,(rows,n),np.float32,runtime)))
        for result in results[1:]:
            for expected,actual in zip(results[0],result):
                np.testing.assert_array_equal(expected.view(np.uint32),actual.view(np.uint32))
    finally:
        for ptr in reversed(allocations):
            free(ptr,runtime=runtime)
