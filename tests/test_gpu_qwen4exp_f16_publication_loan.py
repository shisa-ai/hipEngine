"""DP-5 native producer/consumer uses the existing workspace without a cast."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import free, malloc, memory_stats
from hipengine.loading.materialize import float_array_to_bf16_bits
from tests.test_gpu_qwen4_exp_gr_hip import _upload, _download, _alloc

try:
    ctypes.CDLL("libamdhip64.so")
    HIP_AVAILABLE = True
except OSError:
    HIP_AVAILABLE = False
pytestmark = pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime unavailable")


def test_native_norm_consumer_single_use_and_nondefault_stream(monkeypatch):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as runner
    from hipengine.kernels.registry import KernelKey, register, resolve

    runtime = get_hip_runtime()
    register_gfx1151_kernels(replace=True)
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str,range(13,48))))
    monkeypatch.setenv(linear.DENSE_WIDE_TAIL_VARIANT_ENV, "dense_wide256_padded_f16in_f32_f32_out")
    rows, branches, hidden, n = 4096, 4, 2560, 320
    k = branches * hidden
    rng = np.random.default_rng(3007)
    x = float_array_to_bf16_bits(rng.normal(0,.3,(rows,k)).astype(np.float32))
    gamma = rng.normal(1,.05,(branches,hidden)).astype(np.float32)
    # Raw valid Q8_0 fixture: each block is F16 scale and32signed bytes.
    blocks = np.empty((n,k//32,34),np.uint8)
    blocks[:,:,:2] = np.array([.01],np.float16).view(np.uint8)
    blocks[:,:,2:] = rng.integers(-127,128,(n,k//32,32),dtype=np.int8).view(np.uint8)
    allocations=[]
    before=memory_stats()["current_allocated_bytes"]
    stream=runtime.stream_create()
    producer_key=KernelKey("hip_gfx1151","gr_grouped_rmsnorm","bf16_f32","f32_f16_publish")
    producer=resolve(backend=producer_key.backend,layer=producer_key.layer,
                     quant=producer_key.quant,variant=producer_key.variant)
    counted=[]
    register(producer_key,lambda *a,**kw: counted.append(a) or producer(*a,**kw),replace=True)
    casts=[]
    cast=wide.f32_to_f16
    monkeypatch.setattr(wide,"f32_to_f16",lambda *a,**kw: casts.append(a) or cast(*a,**kw))
    try:
        xd,gd,wd=(_upload(a,runtime,allocations) for a in (x,gamma,blocks))
        nd=_alloc((rows,k),np.float32,runtime,allocations)
        od=_alloc((rows,n),np.float32,runtime,allocations)
        hd=_alloc((rows,k),np.float16,runtime,allocations)
        weight=SimpleNamespace(backend="hip_gfx1151",spec=SimpleNamespace(
            layout="raw_gguf",quant_key="gguf_q8_0",slot_path="layers.16.hc_attn_down",
            source=SimpleNamespace(shape=(n,k))),
            allocation=lambda _:SimpleNamespace(tensor=SimpleNamespace(ptr=wd.ptr),buffer=wd))
        scratch=SimpleNamespace(normalized=nd,low_rank=od)
        snapshots=[]
        with linear.wide_f16_activation_session(workspace_ptr=hd.ptr,workspace_nbytes=hd.nbytes):
            for enabled in ("0","1","1","0"):
                monkeypatch.setenv(linear.NORM_F16_PUBLICATION_ENV,enabled)
                start=len(casts)
                live=memory_stats()["current_allocated_bytes"]
                runner._run_qwen4_exp_gr_norm_down(xd.ptr,gd.ptr,weight,scratch,rows=rows,
                    branches=branches,hidden=hidden,low_rank=n,eps=1e-6,stream=stream,runtime=runtime)
                runtime.stream_synchronize(stream)
                assert len(casts)-start == (enabled=="0")
                assert memory_stats()["current_allocated_bytes"]==live
                snapshots.append([_download(d,shape,dtype,runtime).tobytes() for d,shape,dtype in (
                    (nd,(rows,k),np.float32),(od,(rows,n),np.float32),(hd,(rows,k),np.float16))])
        assert all(s==snapshots[0] for s in snapshots)
        assert len(counted)==2
        np.testing.assert_array_equal(_download(xd,x.shape,np.uint16,runtime),x)
        np.testing.assert_array_equal(_download(wd,blocks.shape,np.uint8,runtime),blocks)
    finally:
        register(producer_key,producer,replace=True)
        runtime.stream_destroy(stream)
        for d in reversed(allocations): free(d,runtime=runtime)
    assert memory_stats()["current_allocated_bytes"]==before
