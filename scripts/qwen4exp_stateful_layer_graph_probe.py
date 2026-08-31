#!/usr/bin/env python3
"""Correctness/performance probe for one captured real Qwen4Exp GDN layer."""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.qwen4exp_canonical_ar_bench import _git_metadata,_host_metadata

def build_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--prompt-file',type=Path,required=True);p.add_argument('--layer',type=int,default=-1);p.add_argument('--replays',type=int,default=4);p.add_argument('--samples',type=int,default=30);p.add_argument('--max-sequence-length',type=int,default=64);p.add_argument('--prefill-chunk-size',type=int,default=64);p.add_argument('--output',type=Path,required=True);return p
def _first_mismatch(rows):
 for row in rows:
  if not row['state_exact']:
   return {'replay':row['replay'],'kind':'state','owner':next(name for name,exact in row['state_owners'].items() if not exact)}
  if not row['output_exact']:return {'replay':row['replay'],'kind':'output','owner':None}
 return None
def _state_hashes(snapshot):return {name:hashlib.sha256(np.ascontiguousarray(value,dtype=np.uint8)).hexdigest() for name,value in snapshot.decode_state.buffers.items()}
def _make_generator(args):
 from hipengine.execution_profiles import ExecutionProfile,resolve_runtime_profile
 from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
 from hipengine.generation.qwen4_exp_profiles import QWEN4_EXP_BACKEND,QWEN4_EXP_MODEL,QWEN4_EXP_QUANTS
 from hipengine.loading.gguf import discover_gguf_files,load_gguf_index
 from hipengine.models import resolve_model
 index=load_gguf_index(discover_gguf_files(args.model_root)[0]);plugin=resolve_model(index.architecture or '')
 resolved=resolve_runtime_profile(model=QWEN4_EXP_MODEL,backend=QWEN4_EXP_BACKEND,quant=QWEN4_EXP_QUANTS[1],profile=ExecutionProfile.STRICT)
 return resolved.construct_generator(lambda:Qwen4ExpGGUFTextGenerator(model_path=args.model_root,weight_index=index,model_plugin=plugin,backend=QWEN4_EXP_BACKEND,max_sequence_length=args.max_sequence_length,prefill_chunk_size=args.prefill_chunk_size)),resolved

def run(args,*,command:Sequence[str])->dict[str,Any]:
 if not args.model_root.is_dir() or not args.prompt_file.is_file():raise ValueError('model root and prompt file must exist')
 if args.replays<4 or args.samples<3:raise ValueError('at least four replays and three samples required')
 os.environ['HIPENGINE_QWEN4_EXP_MOE_GRAPH']='0';os.environ.setdefault('HIPENGINE_HIP_ARCH','gfx1151')
 from hipengine.core.hip import HipMemcpyKind
 from hipengine.core.memory import host_array_ptr,memory_stats,reset_memory_stats
 from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
 from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
 from hipengine.runtime.qwen4_exp_runner import run_qwen4_exp_gdn_layer
 register_gfx1151_kernels(replace=True);register_qwen4_exp_gfx1151_profiles();reset_memory_stats();generator=None;graph=exec_=stream=0
 try:
  generator,resolved=_make_generator(args);runner=generator.runner;rt=runner.runtime;cfg=runner.config
  ids=generator.tokenizer.encode(args.prompt_file.read_text())[:8]
  if not ids:raise ValueError('prompt tokenization is empty')
  runner.prefill(ids,capture_logits=False,capture_target_hidden=False)
  layer=int(args.layer)
  if layer<0:layer=next(i for i,kind in enumerate(cfg.layer_types) if kind=='gdn')
  if layer>=len(cfg.layer_types) or cfg.layer_types[layer]!='gdn':raise ValueError('selected layer must be GDN')
  binding=runner.gdn_bindings[layer];assert runner.state is not None and runner.gdn_scratch is not None
  conv_row=(2*cfg.gdn_group_count*cfg.gdn_state_size+cfg.gdn_inner_size)*cfg.gdn_conv_kernel*4;matrix_row=cfg.gdn_time_step_rank*cfg.gdn_state_size*cfg.gdn_state_size*4
  def launch(stream_id):
   return run_qwen4_exp_gdn_layer(runner.state.residual.ptr,binding,conv_state_ptr=runner.state.gdn_conv.ptr+binding.gdn_state_index*conv_row,recurrent_state_ptr=runner.state.gdn_matrix.ptr+binding.gdn_state_index*matrix_row,scratch=runner.gdn_scratch,rows=1,branches=cfg.residual_branch_count,hidden=cfg.hidden_size,low_rank=cfg.residual_low_rank,num_k_heads=cfg.gdn_group_count,num_v_heads=cfg.gdn_time_step_rank,head_dim=cfg.gdn_state_size,conv_kernel=cfg.gdn_conv_kernel,ffn=cfg.expert_feed_forward_length,experts=cfg.expert_count,top_k=cfg.expert_used_count,stream=stream_id,runtime=rt,moe_graph_cache=None,moe_graph_key=None)
  def output_bytes():
   out=np.empty(runner.gdn_scratch.output.nbytes,np.uint8);rt.memcpy(int(out.ctypes.data),runner.gdn_scratch.output.ptr,out.nbytes,HipMemcpyKind.DEVICE_TO_HOST);return out
  base=runner.snapshot();launch(0);rt.device_synchronize() # warm all dispatch/JIT paths
  runner.restore(base);refs=[]
  for replay in range(1,args.replays+1):
   launch(0);rt.device_synchronize();refs.append((_state_hashes(runner.snapshot()),hashlib.sha256(output_bytes()).hexdigest()))
  runner.restore(base);before=_state_hashes(runner.snapshot());stream=rt.stream_create(nonblocking=True);rt.stream_begin_capture(stream,2);launch(stream);graph=rt.stream_end_capture(stream);exec_=rt.graph_instantiate(graph);after=_state_hashes(runner.snapshot());capture_nonexecuting=before==after
  rows=[]
  for replay,(expected_state,expected_output) in enumerate(refs,1):
   rt.graph_launch(exec_,0);rt.device_synchronize();got_state=_state_hashes(runner.snapshot());got_output=hashlib.sha256(output_bytes()).hexdigest();owners={name:got_state[name]==digest for name,digest in expected_state.items()};rows.append({'replay':replay,'state_exact':all(owners.values()),'state_owners':owners,'output_exact':got_output==expected_output})
  runner.restore(base);eager_ms=[]
  for _ in range(args.samples):
   rt.device_synchronize();start=time.perf_counter();launch(0);rt.device_synchronize();eager_ms.append((time.perf_counter()-start)*1e3)
  runner.restore(base);graph_ms=[]
  for _ in range(args.samples):
   rt.device_synchronize();start=time.perf_counter();rt.graph_launch(exec_,0);rt.device_synchronize();graph_ms.append((time.perf_counter()-start)*1e3)
  eager_median=statistics.median(eager_ms);graph_median=statistics.median(graph_ms);mismatch=_first_mismatch(rows);correct=bool(capture_nonexecuting and mismatch is None)
  payload={'schema':1,'kind':'qwen4exp_stateful_layer_graph_probe','status':'passed' if correct else 'reproduced_corruption','command':list(command),'source':_git_metadata(ROOT),'host':_host_metadata(),'profile':{'name':'strict','manifest_sha256':resolved.manifest_sha256},'model':str(args.model_root),'layer':layer,'gdn_state_index':binding.gdn_state_index,'prompt_tokens':len(ids),'capture_nonexecuting':capture_nonexecuting,'rows':rows,'first_mismatch':mismatch,'timing':{'samples':args.samples,'eager_median_ms':eager_median,'graph_median_ms':graph_median,'speedup':eager_median/graph_median,'eager_ms':eager_ms,'graph_ms':graph_ms}}
 finally:
  if exec_ and generator:generator.runner.runtime.graph_exec_destroy(exec_)
  if graph and generator:generator.runner.runtime.graph_destroy(graph)
  if stream and generator:generator.runner.runtime.stream_destroy(stream)
  if generator:generator.close()
 after_close=memory_stats();payload['lifecycle']={'after_close':after_close,'passed':after_close['current_allocated_bytes']==0};return payload

def main():
 args=build_parser().parse_args();payload=run(args,command=[Path(sys.argv[0]).name,*sys.argv[1:]]);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':payload['status'],'layer':payload['layer'],'first_mismatch':payload['first_mismatch'],'timing':payload['timing'],'output':str(args.output)}));return 0 if payload['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
