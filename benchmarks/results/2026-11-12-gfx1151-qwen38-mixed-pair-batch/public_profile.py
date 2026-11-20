import argparse,json
from pathlib import Path
from hipengine import LLM,SamplingParams
from hipengine.core.hip import get_hip_runtime
from scripts.qwen35_gguf_bench import _RoctxProfilerControl
p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');p.add_argument('--out',required=True);args=p.parse_args()
control=_RoctxProfilerControl(enabled=args.profile)
if args.profile and control._resume is None:raise RuntimeError('ROCTx unavailable')
llm=LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',max_sequence_length=2048,prefix_cache='off',speculative_candidate_budget=3)
r=get_hip_runtime();results=[]
try:
 params=SamplingParams(max_tokens=32,temperature=0,ignore_eos=True,speculative_mtp=True)
 prompt='Explain why transformer inference is usually memory-bandwidth limited on local GPUs.'
 llm.generate_detailed([prompt],params);r.device_synchronize()
 with control.region('request',selected='request' if args.profile else ''):
  result=llm.generate_detailed([prompt],params)[0];r.device_synchronize()
 results.append({'ids':list(result.generated_token_ids),'telemetry':result.telemetry.to_json_dict(),'profile':llm.resolved_execution_profile,'manifest':llm.execution_profile_manifest_sha256})
finally:llm.close()
Path(args.out).write_text(json.dumps(results,indent=2))
