import pytest
import numpy as np
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from tests.test_gpu_qwen4exp_q4_bundle import hip_available
from tests.test_gpu_qwen4exp_pf1_dense_parity import make_q8_0_weight_large
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload,_alloc,_download,_make_activation

NAME="gguf_q8_0_selected_grouped_row4_register_gemv_bf16_bf16_out"


def test_register_scope():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151",layer="linear",quant="gguf_q8_0",
                   variant=NAME.removeprefix("gguf_q8_0_")) is getattr(gemv,NAME)
    for k,threads in ((512,128),(640,64),(640,256)):
        with pytest.raises(ValueError,match="K/threads"):
            getattr(gemv,NAME)(1,2,None,3,4,1,1,1,k,1,threads=threads)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("mapped",[False,True])
def test_register_exact_mapped_and_compact(mapped):
    rt=get_hip_runtime()
    lib=gemv.build_gguf_k_gemv(load=True)
    counts=np.array([0,1,3,9,4,38],dtype=np.int64)
    starts=np.concatenate(([0],counts.cumsum())).astype(np.int64)
    rows,k,n=int(starts[-1]),640,19
    x,_=_make_activation(rows,k,9047)
    w=make_q8_0_weight_large(len(counts)*n,k)
    lanes=np.random.default_rng(8071).permutation(rows).astype(np.int64)
    allocations=[]
    try:
        dx,ds,dw,dl=[_upload(v,rt,allocations) for v in (x,starts,w,lanes)]
        out=[_alloc((rows,n),np.uint16,rt,allocations) for _ in range(2)]
        args=(dx.ptr,ds.ptr,dl.ptr if mapped else None,dw.ptr)
        gemv.gguf_q8_0_selected_grouped_row4_bundle_gemv_bf16_bf16_out(
            *args,out[0].ptr,rows,rows,len(counts),k,n,library=lib,runtime=rt)
        for _ in range(3):
            getattr(gemv,NAME)(*args,out[1].ptr,rows,rows,len(counts),k,n,library=lib,runtime=rt)
            np.testing.assert_array_equal(_download(out[0],(rows,n),np.uint16,rt),
                                          _download(out[1],(rows,n),np.uint16,rt))
    finally:
        for buf in reversed(allocations):
            free(buf,runtime=rt)
