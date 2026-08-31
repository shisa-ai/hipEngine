#!/usr/bin/env python3
"""Correctness/performance probe for captured real Qwen4Exp layer segments."""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
from typing import Any,Sequence
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.qwen4exp_canonical_ar_bench import _git_metadata,_host_metadata

def build_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--prompt-file',type=Path,required=True);p.add_argument('--layer',type=int,default=-1);p.add_argument('--segment-length',type=int,default=1);p.add_argument('--advance-position',action='store_true');p.add_argument('--omit-ple',action='store_true');p.add_argument('--replays',type=int,default=4);p.add_argument('--samples',type=int,default=30);p.add_argument('--max-sequence-length',type=int,default=64);p.add_argument('--prefill-chunk-size',type=int,default=64);p.add_argument('--output',type=Path,required=True);return p
def _first_mismatch(rows):
 for row in rows:
  if not row['state_exact']:
   return {'replay':row['replay'],'kind':'state','owner':next(name for name,exact in row['state_owners'].items() if not exact)}
  if not row['output_exact']:return {'replay':row['replay'],'kind':'output','owner':None}
 return None
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
 if args.replays<4 or args.samples<3 or args.segment_length<=0:raise ValueError('at least four replays, three samples, and one segment layer required')
 os.environ['HIPENGINE_QWEN4_EXP_MOE_GRAPH']='0';os.environ.setdefault('HIPENGINE_HIP_ARCH','gfx1151')
 from hipengine.core.hip import HipMemcpyKind
 from hipengine.core.memory import host_array_ptr,memory_stats,reset_memory_stats
 from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
 from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
 from hipengine.kernels.hip_gfx1100.runtime.state import advance_decode_position_i64
 from hipengine.runtime.qwen4_exp_runner import (
  run_qwen4_exp_dense_qsa_layer,
  run_qwen4_exp_gdn_layer,
  run_qwen4_exp_ple,
 )
 register_gfx1151_kernels(replace=True);register_qwen4_exp_gfx1151_profiles();reset_memory_stats();generator=None;graph=exec_=stream=0
 try:
  generator,resolved=_make_generator(args);runner=generator.runner;rt=runner.runtime;cfg=runner.config
  ids=generator.tokenizer.encode(args.prompt_file.read_text())[:8]
  if not ids:raise ValueError('prompt tokenization is empty')
  runner.prefill(ids,capture_logits=False,capture_target_hidden=False)
  layer=int(args.layer)
  if layer<0:layer=next(i for i,kind in enumerate(cfg.layer_types) if kind=='gdn')
  layers=tuple(range(layer,layer+int(args.segment_length)))
  if not layers or layers[-1]>=len(cfg.layer_types):raise ValueError('selected segment exceeds model layers')
  layer_kinds=tuple(cfg.layer_types[item] for item in layers)
  if any(kind not in {'gdn','qsa'} for kind in layer_kinds):raise ValueError('selected segment contains an unsupported layer kind')
  assert runner.state is not None and runner.gdn_scratch is not None and runner.qsa_scratch is not None and runner.ple_scratch is not None
  conv_row=(2*cfg.gdn_group_count*cfg.gdn_state_size+cfg.gdn_inner_size)*cfg.gdn_conv_kernel*4;matrix_row=cfg.gdn_time_step_rank*cfg.gdn_state_size*cfg.gdn_state_size*4
  qsa_indices=tuple(runner.qsa_bindings[item].qsa_state_index for item in layers if cfg.layer_types[item]=='qsa')
  position=int(runner.position)
  context_limit=position+max(int(args.replays),int(args.samples))
  if args.advance_position:
   if not qsa_indices:raise ValueError('advancing-position graph requires a QSA layer')
   if context_limit>runner.attention_states[qsa_indices[0]].max_positions:raise ValueError('advancing-position graph exceeds attention capacity')
   if any(context_limit>runner.index_states[item].dense_equivalent_limit for item in qsa_indices):raise ValueError('advancing-position graph cannot cross sparse QSA selection')
  base_index_cursors={qsa_index:(runner.index_states[qsa_index].count,runner.index_states[qsa_index].pooled_count) for qsa_index in qsa_indices}
  mutable_buffers={f'decode.{name}':buffer for name,buffer in runner.state.owned_buffers.items()}
  for qsa_index in qsa_indices:
   attention=runner.attention_states[qsa_index];index=runner.index_states[qsa_index]
   for name in ('key_cache','value_cache','position','context'):mutable_buffers[f'qsa.{qsa_index}.attention.{name}']=getattr(attention,name)
   for name in ('raw_keys','pooled_keys','scores','selected_starts','selected_count','query_position','selected_positions'):mutable_buffers[f'qsa.{qsa_index}.index.{name}']=getattr(index,name)
  def snapshot_state():
   snapshot={}
   for name,buffer in mutable_buffers.items():
    value=np.empty(buffer.nbytes,np.uint8);rt.memcpy(int(value.ctypes.data),buffer.ptr,value.nbytes,HipMemcpyKind.DEVICE_TO_HOST);snapshot[name]=value
   return snapshot
  def restore_state(snapshot):
   for name,buffer in mutable_buffers.items():
    value=snapshot[name];rt.memcpy(buffer.ptr,int(value.ctypes.data),value.nbytes,HipMemcpyKind.HOST_TO_DEVICE)
   for qsa_index in qsa_indices:
    attention=runner.attention_states[qsa_index];attention.position_host[0]=position;attention.context_host[0]=position+1
    index=runner.index_states[qsa_index];index.count,index.pooled_count=base_index_cursors[qsa_index]
  def state_hashes(snapshot):return {name:hashlib.sha256(value).hexdigest() for name,value in snapshot.items()}
  def prepare_fixed_qsa_control():
   for qsa_index in qsa_indices:
    runner.attention_states[qsa_index].set_position(position)
    runner.index_states[qsa_index].count=position
  def launch(stream_id,current_position=position,graph_owned=False):
   residual_ptr=runner.state.residual.ptr
   for item,kind in zip(layers,layer_kinds,strict=True):
    if item in cfg.ple_layers and not args.omit_ple:
     layer_prefix=f'layers.{item}.'
     residual_ptr=run_qwen4_exp_ple(residual_ptr,runner.ple_embedding_buffer.ptr,{'ple_key':runner.resident.weight(layer_prefix+'ple_key'),'ple_value':runner.resident.weight(layer_prefix+'ple_value')},norm_key_ptr=runner.resident.weight(layer_prefix+'ple_norm_key').allocation('raw').tensor.ptr,norm_query_ptr=runner.resident.weight(layer_prefix+'ple_norm_query').allocation('raw').tensor.ptr,norm_conv_ptr=runner.resident.weight(layer_prefix+'ple_norm_conv').allocation('raw').tensor.ptr,conv_weight_ptr=runner.resident.weight(layer_prefix+'ple_conv1d').allocation('raw').tensor.ptr,conv_history_ptr=runner.state.ple_conv.ptr,scratch=runner.ple_scratch,rows=1,branches=cfg.residual_branch_count,hidden=cfg.hidden_size,conv_kernel=cfg.ple_conv_kernel,dilation=cfg.ple_ngram_size,stream=stream_id,runtime=rt).ptr
    if kind=='gdn':
     binding=runner.gdn_bindings[item]
     residual_ptr=run_qwen4_exp_gdn_layer(residual_ptr,binding,conv_state_ptr=runner.state.gdn_conv.ptr+binding.gdn_state_index*conv_row,recurrent_state_ptr=runner.state.gdn_matrix.ptr+binding.gdn_state_index*matrix_row,scratch=runner.gdn_scratch,rows=1,branches=cfg.residual_branch_count,hidden=cfg.hidden_size,low_rank=cfg.residual_low_rank,num_k_heads=cfg.gdn_group_count,num_v_heads=cfg.gdn_time_step_rank,head_dim=cfg.gdn_state_size,conv_kernel=cfg.gdn_conv_kernel,ffn=cfg.expert_feed_forward_length,experts=cfg.expert_count,top_k=cfg.expert_used_count,stream=stream_id,runtime=rt,moe_graph_cache=None,moe_graph_key=None).ptr
    else:
     binding=runner.qsa_bindings[item]
     residual_ptr=run_qwen4_exp_dense_qsa_layer(residual_ptr,binding,attention_state=runner.attention_states[binding.qsa_state_index],index_state=runner.index_states[binding.qsa_state_index],scratch=runner.qsa_scratch,position=current_position,rows=1,branches=cfg.residual_branch_count,hidden=cfg.hidden_size,low_rank=cfg.residual_low_rank,query_heads=cfg.attention_head_count,kv_heads=cfg.attention_kv_head_count,head_dim=cfg.attention_key_length,rotary_dim=cfg.rope_dimension_count,theta=cfg.rope_freq_base,index_heads=cfg.indexer_head_count,index_dim=cfg.indexer_key_length,index_rotary_dim=cfg.rope_dimension_count,position_prepared=graph_owned or not args.advance_position,device_position_owned=graph_owned and args.advance_position,attention_context_limit=context_limit if graph_owned and args.advance_position else None,ffn=cfg.expert_feed_forward_length,experts=cfg.expert_count,top_k=cfg.expert_used_count,stream=stream_id,runtime=rt,moe_graph_cache=None,moe_graph_key=None).ptr
   if args.advance_position:
    for qsa_index in qsa_indices:
     attention=runner.attention_states[qsa_index]
     advance_decode_position_i64(attention.position.ptr,attention.context.ptr,stream=stream_id,runtime=rt)
   return residual_ptr
  final_buffer=runner.qsa_scratch.output if layer_kinds[-1]=='qsa' else runner.gdn_scratch.output
  def output_bytes():
   out=np.empty(final_buffer.nbytes,np.uint8);rt.memcpy(int(out.ctypes.data),final_buffer.ptr,out.nbytes,HipMemcpyKind.DEVICE_TO_HOST);return out
  base=snapshot_state();prepare_fixed_qsa_control();launch(0,position);rt.device_synchronize() # warm all dispatch/JIT paths
  restore_state(base);prepare_fixed_qsa_control();refs=[]
  for replay in range(1,args.replays+1):
   current_position=position+replay-1 if args.advance_position else position
   if not args.advance_position:
    for qsa_index in qsa_indices:runner.index_states[qsa_index].count=position
   launch(0,current_position);rt.device_synchronize();refs.append((state_hashes(snapshot_state()),hashlib.sha256(output_bytes()).hexdigest()))
  restore_state(base);prepare_fixed_qsa_control();before=state_hashes(snapshot_state());stream=rt.stream_create(nonblocking=True);rt.stream_begin_capture(stream,2);launch(stream,position,graph_owned=args.advance_position);graph=rt.stream_end_capture(stream);exec_=rt.graph_instantiate(graph);after=state_hashes(snapshot_state());capture_nonexecuting=before==after
  rows=[]
  for replay,(expected_state,expected_output) in enumerate(refs,1):
   rt.graph_launch(exec_,0);rt.device_synchronize()
   if args.advance_position:
    for qsa_index in qsa_indices:
     attention=runner.attention_states[qsa_index];attention.position_host[0]=position+replay;attention.context_host[0]=position+replay+1;runner.index_states[qsa_index].count=position+replay
   got_state=state_hashes(snapshot_state());got_output=hashlib.sha256(output_bytes()).hexdigest();owners={name:got_state[name]==digest for name,digest in expected_state.items()};rows.append({'replay':replay,'position':position+replay-1 if args.advance_position else position,'state_exact':all(owners.values()),'state_owners':owners,'output_exact':got_output==expected_output})
  restore_state(base);prepare_fixed_qsa_control();eager_ms=[]
  for sample in range(args.samples):
   current_position=position+sample if args.advance_position else position
   if not args.advance_position:
    for qsa_index in qsa_indices:runner.index_states[qsa_index].count=position
   rt.device_synchronize();start=time.perf_counter();launch(0,current_position);rt.device_synchronize();eager_ms.append((time.perf_counter()-start)*1e3)
  restore_state(base);prepare_fixed_qsa_control();graph_ms=[]
  for sample in range(args.samples):
   rt.device_synchronize();start=time.perf_counter();rt.graph_launch(exec_,0);rt.device_synchronize();graph_ms.append((time.perf_counter()-start)*1e3)
   if args.advance_position:
    for qsa_index in qsa_indices:runner.index_states[qsa_index].count=position+sample+1
  eager_median=statistics.median(eager_ms);graph_median=statistics.median(graph_ms);mismatch=_first_mismatch(rows);correct=bool(capture_nonexecuting and mismatch is None)
  payload={'schema':1,'kind':'qwen4exp_stateful_layer_graph_probe','status':'passed' if correct else 'reproduced_corruption','command':list(command),'source':_git_metadata(ROOT),'host':_host_metadata(),'profile':{'name':'strict','manifest_sha256':resolved.manifest_sha256},'model':str(args.model_root),'layers':list(layers),'layer_kinds':list(layer_kinds),'gdn_state_indices':[runner.gdn_bindings[item].gdn_state_index for item in layers if cfg.layer_types[item]=='gdn'],'qsa_state_indices':list(qsa_indices),'ple_layers':[item for item in layers if item in cfg.ple_layers and not args.omit_ple],'advancing_position':bool(args.advance_position),'start_position':position,'fixed_position':None if args.advance_position else position,'attention_context_limit':context_limit if args.advance_position else None,'host_cursor_replay_safe':not qsa_indices or bool(args.advance_position),'prompt_tokens':len(ids),'capture_nonexecuting':capture_nonexecuting,'rows':rows,'first_mismatch':mismatch,'timing':{'samples':args.samples,'eager_median_ms':eager_median,'graph_median_ms':graph_median,'speedup':eager_median/graph_median,'eager_ms':eager_ms,'graph_ms':graph_ms}}
 finally:
  if exec_ and generator:generator.runner.runtime.graph_exec_destroy(exec_)
  if graph and generator:generator.runner.runtime.graph_destroy(graph)
  if stream and generator:generator.runner.runtime.stream_destroy(stream)
  if generator:generator.close()
 after_close=memory_stats();payload['lifecycle']={'after_close':after_close,'passed':after_close['current_allocated_bytes']==0};return payload

def main():
 args=build_parser().parse_args();payload=run(args,command=[Path(sys.argv[0]).name,*sys.argv[1:]]);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':payload['status'],'layers':payload['layers'],'first_mismatch':payload['first_mismatch'],'timing':payload['timing'],'output':str(args.output)}));return 0 if payload['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
