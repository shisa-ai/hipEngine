from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SCRIPT=ROOT/'scripts/qwen4exp_gdn_graph_replay_probe.py'
def _load():
 s=importlib.util.spec_from_file_location('qwen4exp_gdn_graph_replay_probe',SCRIPT); assert s and s.loader; m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_replay_summary_names_first_state_or_output_mismatch():
 m=_load(); rows=[{'replay':1,'state_exact':True,'output_exact':True},{'replay':2,'state_exact':False,'output_exact':True},{'replay':3,'state_exact':False,'output_exact':False}]
 assert m._replay_summary(rows)=={'passed':False,'first_state_mismatch':2,'first_output_mismatch':3,'replays':3}
def test_parser_defaults_to_historical_third_replay_scope(tmp_path):
 a=_load().build_parser().parse_args(['--output',str(tmp_path/'o.json')]);assert a.layers==36;assert a.replays==4
