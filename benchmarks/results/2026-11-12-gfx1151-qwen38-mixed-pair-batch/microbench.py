import argparse, json, platform, statistics, time
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc,free,copy_host_array_to_device,copy_device_to_host,host_array_ptr
from hipengine.kernels.hip_gfx1100.fused.gguf_iq4_q4_pair import build_gguf_iq4_q4_pair,gguf_iq4_q4_pair_silu_bf16_bf16_out,gguf_q4_iq4_pair_silu_bf16_bf16_out
from tests.test_gpu_gguf_mixed_pair_batch import mixed_pair_inputs
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--batch',action='store_true');p.add_argument('--profile',action='store_true');args=p.parse_args()
r=get_hip_runtime();lib=build_gguf_iq4_q4_pair(require_cached=args.profile)
results=[]
for reverse in (False,True):
 for rows in (2,4,8):
  k,n=5120,17408
  arrays=mixed_pair_inputs(rows,k,n);bufs=[];events=[]
  try:
   for a in (*arrays,np.zeros((rows,n),np.uint16),np.zeros((rows,n),np.uint16)):
    b=malloc(a.nbytes,runtime=r);bufs.append(b);copy_host_array_to_device(b,a,runtime=r)
   x,iq,q,ref,out=bufs;a,b=(q,iq) if reverse else (iq,q)
   fn=gguf_q4_iq4_pair_silu_bf16_bf16_out if reverse else gguf_iq4_q4_pair_silu_bf16_bf16_out
   def launch(mode,dst):
    if mode=='batch':fn(x.ptr,a.ptr,b.ptr,dst.ptr,rows,k,n,library=lib,runtime=r)
    else:
     for row in range(rows):fn(x.ptr+row*k*2,a.ptr,b.ptr,dst.ptr+row*n*2,1,k,n,library=lib,runtime=r)
   launch('loop',ref)
   modes=('loop','batch') if args.batch else ('loop',)
   for mode in modes:
    for _ in range(3):launch(mode,out)
   r.device_synchronize()
   e0,e1=r.event_create(),r.event_create();events=[e0,e1]
   samples={m:[] for m in modes};walls={m:[] for m in modes}
   for rep in range(10):
    for mode in (modes if rep%2==0 else modes[::-1]):
     t=time.perf_counter();r.event_record(e0)
     for _ in range(10):launch(mode,out)
     r.event_record(e1);r.event_synchronize(e1)
     walls[mode].append((time.perf_counter()-t)*1000/10)
     samples[mode].append(r.event_elapsed_time_ms(e0,e1)/10)
     expected=np.empty((rows,n),np.uint16);actual=np.empty_like(expected)
     copy_device_to_host(host_array_ptr(expected),ref,expected.nbytes,runtime=r)
     copy_device_to_host(host_array_ptr(actual),out,actual.nbytes,runtime=r)
     np.testing.assert_array_equal(actual,expected)
   result=dict(reverse=reverse,rows=rows,k=k,n=n,event_ms=samples,wall_ms=walls,median_event_ms={m:statistics.median(v) for m,v in samples.items()},exact=True)
   results.append(result);print('CELL',reverse,rows,result['median_event_ms'],flush=True)
  finally:
   for e in events:r.event_destroy(e)
   for b in reversed(bufs):free(b,runtime=r)
Path(args.out).write_text(json.dumps(dict(host=platform.node(),shape='fixture-derived IQ4_XS / synthetic Q4, kernel diagnostic only',results=results),indent=2))
