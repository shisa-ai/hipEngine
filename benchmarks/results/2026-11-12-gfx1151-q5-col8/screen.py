"""Isolated IQ4 tile screen; exact parent output before timing acceptance."""
import json
from pathlib import Path
import statistics
import numpy as np
from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as q5
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
from tests._gguf_synthetic_weights import make_q5_k_weight
from tests.test_gpu_gguf_iq_local32_decode import bf16

rt = get_hip_runtime()
lib = q5.build_gguf_t16_selected_gemv(load=True)
libs = {'parent': q5.gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
        'col8': q5.gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out}
rng=np.random.default_rng(19)
results=[]
for k,n in ((5120,17408),(17408,5120),(5120,6144),(5120,12288),(5120,10240),(6144,5120),(5120,1024)):
 weights=repack_gguf_q5_k_tile16(make_q5_k_weight(n,k)[None,...]).tiles
 for rows in (2,3,4,8):
    x=bf16(rng.normal(0,0.1,(rows,k)))
    bufs=[]
    def upload(a):
        b=malloc(a.nbytes);bufs.append(b);copy_host_array_to_device(b,a);return b
    try:
        w=upload(weights);xd=upload(x);outs={a:upload(np.zeros((rows,n),dtype=np.uint16)) for a in libs}
        def launch(a):libs[a](xd.ptr,w.ptr,outs[a].ptr,rows,k,n,library=lib)
        for a in libs:
            for _ in range(3):launch(a)
        rt.device_synchronize()
        host={a:np.empty((rows,n),dtype=np.uint16) for a in libs}
        for a in libs:copy_device_to_host(host_array_ptr(host[a]),outs[a])
        assert np.array_equal(host['parent'],host['col8']),(k,n,rows)
        timings={a:[] for a in libs}
        start=rt.event_create();end=rt.event_create()
        try:
            for rep in range(8):
                for a in (('parent','col8') if rep%2==0 else ('col8','parent')):
                    rt.event_record(start,0)
                    for _ in range(10):launch(a)
                    rt.event_record(end,0);rt.event_synchronize(end)
                    timings[a].append(rt.event_elapsed_time_ms(start,end)/10)
        finally:rt.event_destroy(start);rt.event_destroy(end)
        result=dict(k=k,n=n,rows=rows,exact=True,timings_ms=timings,medians_ms={a:statistics.median(t) for a,t in timings.items()})
        results.append(result);print(result['k'],n,rows,result['medians_ms'],flush=True)
    finally:
        for b in reversed(bufs):free(b)
Path('/tmp/qwen38-q5-screen.json').write_text(json.dumps(results,indent=2))
