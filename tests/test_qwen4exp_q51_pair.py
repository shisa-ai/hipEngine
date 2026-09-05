import ctypes
import numpy as np
import pytest
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q5
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data, bf16_to_float32
from tests.test_qwen4_exp_pf3_moe_schedules import (
    _upload, _alloc, _download, _make_activation, _make_expert_q5_1_weights)

PARENT="qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"
CANDIDATE="qwen4_exp_q5_1_selected_grouped_prefill_pair2_bf16_bf16_out"


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_pair_registry_and_bounds():
    from types import SimpleNamespace
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    fn=resolve(backend="hip_gfx1151",layer="moe_linear",quant="gguf_q5_1",
               variant="selected_grouped_prefill_pair2_bf16_bf16_out")
    assert fn is getattr(q5,CANDIDATE)
    assert callable(resolve(backend="hip_gfx1151",layer="moe_linear",quant="gguf_q5_1",
        variant="selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"))
    with pytest.raises(ValueError,match="4096"):
        fn(1,1,1,1,1,1,8192,1)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("rows,experts,k,n",[(17,8,96,7),(65,64,640,33),(5120,512,640,2560)])
def test_pair_exact(rows,experts,k,n):
    candidate=getattr(q5,CANDIDATE)
    runtime=get_hip_runtime()
    library=q5.build_qwen4_exp_q5_1(load=True)
    rng=np.random.default_rng(5151)
    counts=np.bincount(rng.integers(0,experts-1,rows),minlength=experts)
    starts=np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
    x,xref=_make_activation(rows,k,2451)
    w=_make_expert_q5_1_weights(num_experts=experts,out_features=n,in_features=k,seed=7451)
    allocations=[]
    try:
        dx,ds,dw=[_upload(v,runtime,allocations) for v in (x,starts,w)]
        before,after=[_alloc((rows,n),np.uint16,runtime,allocations) for _ in range(2)]
        getattr(q5,PARENT)(dx.ptr,ds.ptr,dw.ptr,before.ptr,rows,experts,k,n,library=library,runtime=runtime)
        expected=_download(before,(rows,n),np.uint16,runtime)
        for _ in range(2):
            candidate(dx.ptr,ds.ptr,dw.ptr,after.ptr,rows,experts,k,n,library=library,runtime=runtime)
            np.testing.assert_array_equal(_download(after,(rows,n),np.uint16,runtime),expected)
        if rows==17:
            for e,count in enumerate(counts):
                if not count:
                    continue
                weight=dequantize_gguf_data(w[e],GGMLQuantizationType.Q5_1)
                lo,hi=starts[e:e+2]
                cpu=xref[lo:hi] @ weight.T
                np.testing.assert_allclose(bf16_to_float32(expected[lo:hi]),cpu,rtol=.02,atol=.002)
    finally:
        for p in reversed(allocations):
            free(p,runtime=runtime)
