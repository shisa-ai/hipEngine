import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from tests.test_qwen4_exp_pf3_moe_schedules import (
    _upload, _alloc, _download, _make_activation, _q4_k_reference_per_expert,
)
from hipengine.quant.gguf import bf16_to_float32

PARENT = "gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"
CANDIDATE = "gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def raw_weights(experts, n, k, seed):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0,256,(experts,n,k//256,144),dtype=np.uint8)
    scales = rng.uniform(0.0001,0.001,(*raw.shape[:-1],2)).astype(np.float16)
    raw[...,:4] = scales.view(np.uint8).reshape(*raw.shape[:-1],4)
    return raw.reshape(experts,n,-1)


def test_bundle_registry_preserves_parent():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    for name in (PARENT,CANDIDATE):
        assert resolve(
            backend="hip_gfx1151",layer="moe_linear",quant="gguf_q4_k",
            variant=name.removeprefix("gguf_q4_k_")) is getattr(q4,name)


@pytest.mark.skipif(not hip_available(), reason="HIP unavailable")
@pytest.mark.parametrize("rows,experts,k,n", [(17,8,256,7),(65,64,512,33),(5120,512,2560,640)])
def test_bundle_exact(rows, experts, k, n):
    candidate = getattr(q4, CANDIDATE)
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    rng = np.random.default_rng(1215)
    selected = rng.integers(0,experts-1,rows)
    counts = np.bincount(selected,minlength=experts)
    starts = np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
    x, xref = _make_activation(rows,k,1217)
    a,b = raw_weights(experts,n,k,2),raw_weights(experts,n,k,3)
    allocations=[]
    try:
        dx,ds,da,db = [_upload(v,runtime,allocations) for v in (x,starts,a,b)]
        outputs = [_alloc((rows,n),np.uint16,runtime,allocations) for _ in range(4)]
        args = (dx.ptr,ds.ptr,da.ptr,db.ptr)
        getattr(q4,PARENT)(*args,outputs[0].ptr,outputs[1].ptr,rows,experts,k,n,library=library,runtime=runtime)
        expected = [_download(o,(rows,n),np.uint16,runtime) for o in outputs[:2]]
        for _ in range(2):
            candidate(*args,outputs[2].ptr,outputs[3].ptr,rows,experts,k,n,library=library,runtime=runtime)
            for o,e in zip(outputs[2:],expected):
                np.testing.assert_array_equal(_download(o,(rows,n),np.uint16,runtime),e)
        if rows==17:
            cpu = _q4_k_reference_per_expert(xref,a,counts)
            np.testing.assert_allclose(bf16_to_float32(expected[0]),cpu,rtol=0.02,atol=0.002)
    finally:
        for p in reversed(allocations):
            free(p,runtime=runtime)
