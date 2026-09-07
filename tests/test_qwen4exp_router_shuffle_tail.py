import ctypes
import numpy as np
import pytest
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.moe import router
from tests.test_qwen4_exp_pf3_moe_schedules import _upload,_alloc,_download

PARENT = "qwen35_router_logits_f32_f32w_token_tile4_dense_exact"
CANDIDATE = "qwen35_router_logits_f32_f32w_token_tile4_shuffle_exact"


def hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_router_shuffle_registry():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151",layer="router_logits",quant="f32",
                   variant="f32_hidden_token_tile4_shuffle_exact") is getattr(router,CANDIDATE)


@pytest.mark.skipif(not hip_available(),reason="HIP unavailable")
@pytest.mark.parametrize("rows,k,n",[(1,31,7),(5,257,17),(65,2560,512),(1024,2560,512)])
def test_router_shuffle_exact(rows,k,n):
    runtime = get_hip_runtime()
    library = router.build_qwen35_router(load=True)
    rng = np.random.default_rng(914)
    x = rng.normal(0,.2,(rows,k)).astype(np.float32)
    w = rng.normal(0,.05,(n,k)).astype(np.float32)
    allocations = []
    try:
        dx,dw = [_upload(v,runtime,allocations) for v in (x,w)]
        outputs = [_alloc((rows,n),np.float32,runtime,allocations) for _ in range(2)]
        getattr(router,PARENT)(dx.ptr,dw.ptr,outputs[0].ptr,rows,k,n,library=library,runtime=runtime)
        expected = _download(outputs[0],(rows,n),np.float32,runtime)
        for _ in range(3):
            getattr(router,CANDIDATE)(dx.ptr,dw.ptr,outputs[1].ptr,rows,k,n,library=library,runtime=runtime)
            actual = _download(outputs[1],(rows,n),np.float32,runtime)
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        cpu = x.astype(np.float64) @ w.astype(np.float64).T
        def logsoftmax(v):
            v = v.astype(np.float64)
            v -= v.max(axis=1,keepdims=True)
            return v-np.log(np.exp(v).sum(axis=1,keepdims=True))
        lp,lq = logsoftmax(cpu),logsoftmax(actual)
        assert np.max(np.sum(np.exp(lp)*(lp-lq),axis=1)) <= .05
        assert np.mean(cpu.argmax(1)==actual.argmax(1)) >= .9
    finally:
        for buf in reversed(allocations):
            free(buf,runtime=runtime)
