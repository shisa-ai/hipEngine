from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'scripts/qwen4exp_stateful_layer_graph_probe.py'
def _load():
 spec=importlib.util.spec_from_file_location('qwen4exp_stateful_layer_graph_probe',SCRIPT);assert spec and spec.loader;module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def test_parser_requires_real_model_and_defaults_to_four_replays(tmp_path):
 args=_load().build_parser().parse_args(['--model-root',str(tmp_path),'--prompt-file',str(tmp_path/'p'),'--output',str(tmp_path/'o')]);assert args.layer==-1;assert args.segment_length==1;assert args.replays==4;assert args.samples==30
def test_first_mismatch_localizes_owner_and_replay():
 rows=[{'replay':1,'output_exact':True,'state_exact':True,'state_owners':{'gdn_conv':True,'gdn_matrix':True}},{'replay':2,'output_exact':True,'state_exact':False,'state_owners':{'gdn_conv':True,'gdn_matrix':False}}]
 assert _load()._first_mismatch(rows)=={'replay':2,'kind':'state','owner':'gdn_matrix'}
