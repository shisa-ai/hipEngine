#!/usr/bin/env python3
"""Reproduce the historical third-replay hazard with 36 independent GDN states."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from typing import Any,Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.qwen4exp_canonical_ar_bench import _git_metadata,_host_metadata

def _replay_summary(rows):
 return {'passed':all(r['state_exact'] and r['output_exact'] for r in rows),'first_state_mismatch':next((r['replay'] for r in rows if not r['state_exact']),None),'first_output_mismatch':next((r['replay'] for r in rows if not r['output_exact']),None),'replays':len(rows)}
def build_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--layers',type=int,default=36);p.add_argument('--replays',type=int,default=4);p.add_argument('--seed',type=int,default=4071);p.add_argument('--compiler-version-file',type=Path);p.add_argument('--require-cached-build',action='store_true');p.add_argument('--output',type=Path,required=True);return p

def run(args,*,command:Sequence[str])->dict[str,Any]:
 if args.layers<=0 or args.replays<4:raise ValueError('layers positive and at least four replays required')
 if args.compiler_version_file:os.environ['HIPENGINE_COMPILER_VERSION_FILE']=str(args.compiler_version_file)
 if args.require_cached_build:os.environ['HIPENGINE_REQUIRE_CACHED_BUILD']='1'
 os.environ.setdefault('HIPENGINE_HIP_ARCH','gfx1151')
 from hipengine.core.hip import get_hip_runtime,HipMemcpyKind
 from hipengine.core.memory import malloc,free,copy_host_to_device,copy_device_to_host,host_array_ptr,memory_stats,reset_memory_stats
 from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv,qwen35_linear_attn_conv_decode_f32
 from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import build_qwen4_exp_gdn,qwen4_exp_gdn_decode_f32
 rt=get_hip_runtime();lib=build_qwen4_exp_gdn(load=True);conv_lib=build_qwen35_linear_attn_conv(load=True);reset_memory_stats();rng=np.random.default_rng(args.seed)
 layers=int(args.layers);kh,vh,d=16,48,128;qkv=2*kh*d+vh*d;core=vh*d;kernel=4;state_elems=vh*d*d;conv_elems=qkv*kernel
 shapes=[qkv,core,vh,vh,vh,vh,d]
 inputs=[np.ascontiguousarray(rng.normal(0,.05,(layers,n)).astype(np.float32)) for n in shapes];inputs[5]=-np.abs(inputs[5]);conv_weights=np.ascontiguousarray(rng.normal(0,.05,(layers,conv_elems)).astype(np.float32));initial_conv=np.ascontiguousarray(rng.normal(0,.02,(layers,conv_elems)).astype(np.float32));initial=np.ascontiguousarray(rng.normal(0,.02,(layers,state_elems)).astype(np.float32))
 bufs=[];graph=exec_=stream=0
 def up(a):
  z=malloc(a.nbytes,runtime=rt);bufs.append(z);copy_host_to_device(z,host_array_ptr(a),a.nbytes,runtime=rt);return z
 dev=[up(a) for a in inputs];dev_conv_weights=up(conv_weights);eager_conv_state=up(initial_conv);graph_conv_state=up(initial_conv);eager_state=up(initial);graph_state=up(initial);eager_conv_out=malloc(layers*qkv*4,runtime=rt);graph_conv_out=malloc(layers*qkv*4,runtime=rt);eager_out=malloc(layers*core*4,runtime=rt);graph_out=malloc(layers*core*4,runtime=rt);bufs.extend((eager_conv_out,graph_conv_out,eager_out,graph_out))
 def launch(conv_state,state,conv_out,out,stream_id):
  for layer in range(layers):
   ptr=[z.ptr+layer*inputs[i].shape[1]*4 for i,z in enumerate(dev)]
   qwen35_linear_attn_conv_decode_f32(ptr[0],conv_state.ptr+layer*conv_elems*4,dev_conv_weights.ptr+layer*conv_elems*4,conv_out.ptr+layer*qkv*4,qkv,kernel,stream=stream_id,library=conv_lib,runtime=rt)
   ptr[0]=conv_out.ptr+layer*qkv*4
   qwen4_exp_gdn_decode_f32(*ptr,state.ptr+layer*state_elems*4,out.ptr+layer*core*4,kh,vh,d,d,stream=stream_id,library=lib,runtime=rt)
 def download(buffer,shape):
  a=np.empty(shape,np.float32);copy_device_to_host(host_array_ptr(a),buffer,a.nbytes,runtime=rt);return a
 try:
  # Warm dispatch/JIT using a throwaway eager transition, then restore.
  launch(eager_conv_state,eager_state,eager_conv_out,eager_out,0);rt.device_synchronize();copy_host_to_device(eager_conv_state,host_array_ptr(initial_conv),initial_conv.nbytes,runtime=rt);copy_host_to_device(eager_state,host_array_ptr(initial),initial.nbytes,runtime=rt)
  before_conv=download(graph_conv_state,initial_conv.shape);before=download(graph_state,initial.shape)
  stream=rt.stream_create(nonblocking=True);rt.stream_begin_capture(stream,2);launch(graph_conv_state,graph_state,graph_conv_out,graph_out,stream);graph=rt.stream_end_capture(stream);exec_=rt.graph_instantiate(graph)
  after_conv=download(graph_conv_state,initial_conv.shape);after_capture=download(graph_state,initial.shape);capture_nonexecuting=bool(np.array_equal(before_conv,after_conv) and np.array_equal(before,after_capture))
  rows=[]
  for replay in range(1,int(args.replays)+1):
   launch(eager_conv_state,eager_state,eager_conv_out,eager_out,0);rt.device_synchronize();rt.graph_launch(exec_,0);rt.device_synchronize()
   ec=download(eager_conv_state,initial_conv.shape);gc=download(graph_conv_state,initial_conv.shape);es=download(eager_state,initial.shape);gs=download(graph_state,initial.shape);eo=download(eager_out,(layers,core));go=download(graph_out,(layers,core))
   rows.append({'replay':replay,'conv_state_exact':bool(np.array_equal(ec,gc)),'recurrent_state_exact':bool(np.array_equal(es,gs)),'state_exact':bool(np.array_equal(ec,gc) and np.array_equal(es,gs)),'output_exact':bool(np.array_equal(eo,go)),'state_max_abs':float(max(np.max(np.abs(ec-gc)),np.max(np.abs(es-gs)))),'output_max_abs':float(np.max(np.abs(eo-go)))})
  for buffer,array in ((eager_conv_state,initial_conv),(graph_conv_state,initial_conv),(eager_state,initial),(graph_state,initial)):copy_host_to_device(buffer,host_array_ptr(array),array.nbytes,runtime=rt)
  launch(eager_conv_state,eager_state,eager_conv_out,eager_out,0);rt.device_synchronize();rt.graph_launch(exec_,0);rt.device_synchronize();reset_exact=bool(np.array_equal(download(eager_conv_state,initial_conv.shape),download(graph_conv_state,initial_conv.shape)) and np.array_equal(download(eager_state,initial.shape),download(graph_state,initial.shape)) and np.array_equal(download(eager_out,(layers,core)),download(graph_out,(layers,core))))
  summary=_replay_summary(rows);passed=bool(capture_nonexecuting and reset_exact and summary['passed'])
  payload={'schema':1,'kind':'qwen4exp_gdn_graph_replay_probe','status':'passed' if passed else 'reproduced_corruption','command':list(command),'source':_git_metadata(ROOT),'host':_host_metadata(),'protocol':{'layers':layers,'independent_conv_states':layers,'independent_recurrent_states':layers,'replays':args.replays,'shape':[kh,vh,d,d],'conv_channels':qkv,'conv_kernel':kernel,'captured_kernels_per_replay':2*layers,'capture_mode':'relaxed','state_dtype':'FP32'},'capture_nonexecuting':capture_nonexecuting,'rows':rows,'summary':summary,'reset_exact':reset_exact}
 finally:
  if exec_:rt.graph_exec_destroy(exec_)
  if graph:rt.graph_destroy(graph)
  if stream:rt.stream_destroy(stream)
  for z in reversed(bufs):free(z,runtime=rt)
 payload['lifecycle']={'after_close':memory_stats(),'passed':memory_stats()['current_allocated_bytes']==0};payload['status']='passed' if passed and payload['lifecycle']['passed'] else payload['status'];return payload

def main():
 a=build_parser().parse_args();p=run(a,command=[Path(sys.argv[0]).name,*sys.argv[1:]]);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':p['status'],'summary':p['summary'],'output':str(a.output)}));return 0 if p['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
