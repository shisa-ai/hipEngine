import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data, bf16_to_float32
from tests.test_qwen4exp_q4_bundle import hip_available, raw_weights, PAIR
from tests.test_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download, _make_activation

CANDIDATE = "gguf_q4_k_selected_dual_wmma_f16x2_bf16_bf16_out"


def tile_map(counts):
    starts = np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
    padded = (np.asarray(counts)+15)//16*16
    wmma_starts = np.concatenate(([0],padded.cumsum())).astype(np.int64)
    tiles = np.repeat(np.arange(len(counts)),padded//16).astype(np.int64)
    return starts,wmma_starts,tiles


def test_residual_registry_scope():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151",layer="moe_linear",quant="gguf_q4_k",
                   variant="selected_dual_wmma_f16x2_bf16_bf16_out") is getattr(q4,CANDIDATE)
    with pytest.raises(ValueError,match="16"):
        getattr(q4,CANDIDATE)(tile_n=32)


def test_weight_residual_reduces_reconstruction_error():
    raw = raw_weights(1,32,2560,613)[0]
    w = dequantize_gguf_data(raw,GGMLQuantizationType.Q4_K).astype(np.float32)
    hi = w.astype(np.float16).astype(np.float32)
    lo = (w-hi).astype(np.float16).astype(np.float32)
    assert np.linalg.norm(w-hi-lo) < .01*np.linalg.norm(w-hi)


def test_screen_metrics_fail_on_nonfinite_output():
    from scripts.qwen4exp_q4_residual_wmma_screen import metrics
    zeros = np.zeros((1,4),dtype=np.uint16)
    assert metrics(zeros,zeros)["bf16_element_agreement"] == 1.
    invalid = zeros.copy()
    invalid[0,0] = 0x7fc0
    with pytest.raises(ValueError,match="non-finite"):
        metrics(zeros,invalid)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("k,n",[(256,16),(2560,32),(4096,48)])
def test_residual_wmma_numerical_floor_and_repeatability(k,n):
    counts = np.array([1,17,0,31],dtype=np.int64)
    starts,wmma_starts,tiles = tile_map(counts)
    rows,experts = int(starts[-1]),len(counts)
    x,xref = _make_activation(rows,k,719)
    a,b = raw_weights(experts,n,k,817),raw_weights(experts,n,k,818)
    runtime = get_hip_runtime()
    lib = q4.build_gguf_q4_k_selected_prefill(load=True)
    allocations = []
    try:
        dx,ds,dws,dt,da,db = [_upload(v,runtime,allocations) for v in
                              (x,starts,wmma_starts,tiles,a,b)]
        exact = [_alloc((rows,n),np.uint16,runtime,allocations) for _ in range(2)]
        out = _alloc((rows,2*n),np.uint16,runtime,allocations)
        getattr(q4,PAIR)(dx.ptr,ds.ptr,da.ptr,db.ptr,exact[0].ptr,exact[1].ptr,
                        rows,experts,k,n,library=lib,runtime=runtime)
        parent = np.concatenate([_download(v,(rows,n),np.uint16,runtime)
                                 for v in exact],axis=1)
        result = []
        for tile in (16,16,32,64):
            getattr(q4,CANDIDATE)(dx.ptr,ds.ptr,dws.ptr,dt.ptr,da.ptr,db.ptr,out.ptr,
                                 rows,k,n,n,experts,int(wmma_starts[-1]),
                                 tile_m=tile,library=lib,runtime=runtime)
            result.append(_download(out,(rows,2*n),np.uint16,runtime))
        for actual_bits in result[1:]:
            np.testing.assert_array_equal(result[0],actual_bits)
        actual = bf16_to_float32(result[0])
        assert np.isfinite(actual).all()
        assert np.mean(result[0] == parent) >= .98
        cpu = np.empty((rows,2*n),dtype=np.float32)
        for e,count in enumerate(counts):
            if count:
                weights = np.concatenate([dequantize_gguf_data(w[e],GGMLQuantizationType.Q4_K)
                                          for w in (a,b)],axis=0)
                cpu[starts[e]:starts[e+1]] = xref[starts[e]:starts[e+1]] @ weights.T
        def logsoftmax(v):
            v = v.astype(np.float64)
            v -= v.max(axis=1,keepdims=True)
            return v-np.log(np.exp(v).sum(axis=1,keepdims=True))
        lp,lq = logsoftmax(cpu),logsoftmax(actual)
        assert np.max(np.sum(np.exp(lp)*(lp-lq),axis=1)) <= .05
        assert np.mean(cpu.argmax(1) == actual.argmax(1)) >= .9
    finally:
        for buf in reversed(allocations):
            free(buf,runtime=runtime)
