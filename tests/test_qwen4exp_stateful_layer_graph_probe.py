from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'scripts/qwen4exp_stateful_layer_graph_probe.py'
def _load():
 spec=importlib.util.spec_from_file_location('qwen4exp_stateful_layer_graph_probe',SCRIPT);assert spec and spec.loader;module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def test_parser_requires_real_model_and_defaults_to_four_replays(tmp_path):
 args=_load().build_parser().parse_args(['--model-root',str(tmp_path),'--prompt-file',str(tmp_path/'p'),'--output',str(tmp_path/'o')]);assert args.layer==-1;assert args.segment_length==1;assert args.advance_position is False;assert args.replays==4;assert args.samples==30
def test_first_mismatch_localizes_owner_and_replay():
 rows=[{'replay':1,'output_exact':True,'state_exact':True,'state_owners':{'gdn_conv':True,'gdn_matrix':True}},{'replay':2,'output_exact':True,'state_exact':False,'state_owners':{'gdn_conv':True,'gdn_matrix':False}}]
 assert _load()._first_mismatch(rows)=={'replay':2,'kind':'state','owner':'gdn_matrix'}

@pytest.mark.parametrize(('prepared','expected'),((False,[7]),(True,[])))
def test_qsa_position_prepared_skips_legacy_stream_upload(monkeypatch,prepared,expected):
 import hipengine.runtime.qwen4_exp_runner as module
 owner=object();positions=[]
 state=SimpleNamespace(closed=False,runtime=owner,max_positions=64,set_position=positions.append)
 scratch=SimpleNamespace(closed=False,runtime=owner,q_projected=SimpleNamespace(ptr=1),key_projected=SimpleNamespace(ptr=2),value_projected=SimpleNamespace(ptr=3))
 weights=SimpleNamespace(projections={name:object() for name in ('attn_q','attn_k','attn_v','attn_output')})
 def stop(*_args,**_kwargs):raise RuntimeError('stop after position ownership')
 monkeypatch.setattr(module,'launch_gguf_linear',stop)
 with pytest.raises(RuntimeError,match='stop after position ownership'):
  module.run_qwen4_exp_dense_qsa_token_mixer(1,weights,attention_state=state,scratch=scratch,position=7,rows=1,hidden=1,query_heads=1,kv_heads=1,head_dim=1,rotary_dim=1,theta=1.0,position_prepared=prepared,runtime=owner)
 assert positions==expected

def test_qsa_device_position_requires_prepared_control():
 import hipengine.runtime.qwen4_exp_runner as module
 owner=object()
 state=SimpleNamespace(closed=False,runtime=owner,set_position=lambda _value:None)
 scratch=SimpleNamespace(closed=False,runtime=owner)
 weights=SimpleNamespace(projections={name:object() for name in ('attn_q','attn_k','attn_v','attn_output')})
 with pytest.raises(ValueError,match='must be prepared'):
  module.run_qwen4_exp_dense_qsa_token_mixer(1,weights,attention_state=state,scratch=scratch,position=7,rows=1,hidden=1,query_heads=1,kv_heads=1,head_dim=1,rotary_dim=1,theta=1.0,device_position_owned=True,runtime=owner)
