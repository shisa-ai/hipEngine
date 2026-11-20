"""Isolated IQ4 tile screen; exact parent output before timing acceptance."""
import json
from pathlib import Path
import statistics
import numpy as np
from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense as iq
from tests.test_gpu_gguf_iq_local32_decode import bf16

rt = get_hip_runtime()
libs = {name: build_hip(sources=[Path('/tmp/qwen38-iq-'+name+'.hip')], family='iq_screen_'+name,
    profile='decode', include_dirs=[iq.SOURCE.parent], extra_flags=['-ffp-contract=off',f'-DGGUF_IQ_TABLE_HASH={iq.TABLE_HASH}'], load=True) for name in ('parent','col4')}
fixture=Path('tests/fixtures/gguf_ud')
entry=next(e for e in json.loads((fixture/'real_rows.json').read_text())['entries'] if e['type']=='IQ4_XS')
with np.load(fixture/'real_rows.npz') as data:
    blocks=data[entry['key']+'_raw'].reshape(-1,136).copy()
rng=np.random.default_rng(19)
results=[]
for k,n in ((5120,17408),(17408,5120),(5120,6144),(5120,12288)):
 for rows in (2,3,4):
    weights=np.ascontiguousarray(blocks[rng.integers(0,len(blocks),size=(n,k//256))])
    x=bf16(rng.normal(0,0.1,(rows,k)))
    bufs=[]
    def upload(a):
        b=malloc(a.nbytes);bufs.append(b);copy_host_array_to_device(b,a);return b
    try:
        w=upload(weights);xd=upload(x);outs={a:upload(np.zeros((rows,n),dtype=np.uint16)) for a in libs}
        def launch(a):iq.launch_local32_rows(xd.ptr,w.ptr,outs[a].ptr,rows,k,n,library=libs[a])
        for a in libs:
            for _ in range(3):launch(a)
        rt.device_synchronize()
        host={a:np.empty((rows,n),dtype=np.uint16) for a in libs}
        for a in libs:copy_device_to_host(host_array_ptr(host[a]),outs[a])
        assert np.array_equal(host['parent'],host['col4']),(k,n,rows)
        timings={a:[] for a in libs}
        start=rt.event_create();end=rt.event_create()
        try:
            for rep in range(8):
                for a in (('parent','col4') if rep%2==0 else ('col4','parent')):
                    rt.event_record(start,0)
                    for _ in range(10):launch(a)
                    rt.event_record(end,0);rt.event_synchronize(end)
                    timings[a].append(rt.event_elapsed_time_ms(start,end)/10)
        finally:rt.event_destroy(start);rt.event_destroy(end)
        result=dict(k=k,n=n,rows=rows,exact=True,timings_ms=timings,medians_ms={a:statistics.median(t) for a,t in timings.items()})
        results.append(result);print(result['k'],n,rows,result['medians_ms'],flush=True)
    finally:
        for b in reversed(bufs):free(b)
Path('/tmp/qwen38-iq-screen.json').write_text(json.dumps(results,indent=2))
