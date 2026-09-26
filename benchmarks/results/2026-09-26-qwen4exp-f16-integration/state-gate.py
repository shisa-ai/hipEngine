import sys, os, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, '/home/lhl/hipEngine')
from scripts.qwen4exp_layer2_profile_gate import _make_generator
from scripts.qwen4exp_chunk_boundary_gate import trajectory
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.core.memory import memory_stats
os.environ['HIPENGINE_QWEN4_EXP_PLE_WARM']='1'
register_gfx1151_kernels(replace=True)
register_qwen4_exp_gfx1151_profiles()
args=SimpleNamespace(model_root=Path('/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL'),max_sequence_length=4224,prefill_chunk_size=1024)
fixture=Path('/home/lhl/hipEngine/benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json')
report=dict(host=_host_metadata(),source=_git_metadata(Path('/home/lhl/hipEngine')),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),steps=64,chunk=1024,profile='production',comparison='F32-input production versus F16-input production; not strict-teacher qualification',rows=[],status='running')
out=Path('/tmp/qwen4exp-f16-integration-20260926/state-gate.json')
generator,profile,_=_make_generator(args,'production')
report['manifest_sha256']=profile.manifest_sha256
r=generator.runner
buffer_id=None
try:
 for case in json.loads(fixture.read_bytes())['cases']:
  reference=None
  for rep,arm in enumerate(['0','1','1','1','0']):
   os.environ['HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT']=arm
   r.reset()
   logits,tokens,state=trajectory(r,case['prompt_token_ids'],steps=64,chunk=1024)
   if reference is None: reference=(logits.copy(),tokens,state)
   buffer=getattr(r,'_wide_f16_activation_buffer',None)
   if arm=='1':
    assert buffer is not None
    if buffer_id is None: buffer_id=id(buffer)
    assert id(buffer)==buffer_id, 'workspace reallocated'
   row=dict(case=case['id'],arm=arm,repeat=rep,logits_equal=bool(np.array_equal(logits,reference[0])),tokens_equal=tokens==reference[1],state_equal=state==reference[2],logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),state=state)
   report['rows'].append(row)
   print(case['id'],arm,rep,row['logits_equal'],row['state_equal'],flush=True)
   out.write_text(json.dumps(report,indent=2)+'\n')
   assert row['logits_equal'] and row['tokens_equal'] and row['state_equal'], 'production-parent comparison differs; review numerical gates'
 report['workspace_reused']=True
 report['status']='passed'
finally:
 generator.close()
 report['workspace_released']=getattr(r,'_wide_f16_activation_buffer',None) is None
 report['memory_after_close']=memory_stats()
 out.write_text(json.dumps(report,indent=2)+'\n')
