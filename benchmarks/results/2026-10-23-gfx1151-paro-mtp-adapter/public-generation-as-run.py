import copy,json,hashlib,sys,subprocess
from contextlib import closing
from pathlib import Path
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.benchmark.provenance import collect_artifact_provenance
from tokenizers import Tokenizer

root=Path.cwd(); model=Path('/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16')
outpath=Path('/tmp/paro-mtp-adapter-validation.json')
tok=Tokenizer.from_file(str(model/'tokenizer.json'))
cases=[]
for fname in ('mtpbench-code-general-ja.jsonl','gdn-prefill-category-heldouts.jsonl'):
 for line in (root/'benchmarks/prompts'/fname).read_text().splitlines():
  if not line.strip(): continue
  c=json.loads(line); text='<|im_start|>user\n'+c['messages'][0]['content']+'<|im_end|>\n<|im_start|>assistant\n'
  cases.append({'id':c['id'],'category':c['category'],'tokens':tok.encode(text).ids})
for n in (511,513,1025,4096): cases.append({'id':f'p{n}','category':'length-control','tokens':[9707]*n})
def find_routes(d):
 if isinstance(d,dict):
  if 'routes' in d and 'recent_completed' in d['routes']: return d['routes']
  for v in d.values():
   r=find_routes(v)
   if r is not None: return r
 return None
result={'schema':1,'performance_claim':False,'cases':[], 'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with closing(LLM(str(model),backend='hip_gfx1151',quant='w4_paro',max_active_requests=1,max_sequence_length=4352,prefix_cache='off')) as llm:
 for case in cases:
  runs=[]
  for mode in (False,True,True):
   o=llm.generate_detailed([case['tokens']],SamplingParams(max_tokens=64,temperature=0,ignore_eos=True,speculative_mtp=mode,kv_storage='bf16'))[0]
   routes=find_routes(llm.live_loop_snapshot()); assert routes is not None
   m=copy.deepcopy(routes['recent_completed'][-1]); m.pop('scheduler_chunks',None)
   runs.append({'mtp':mode,'ids':list(o.generated_token_ids),'text':o.text,'route':m,'fallback_reasons':routes['fallback_reasons']})
  passed=all(r['ids']==runs[0]['ids'] and len(r['ids'])==64 for r in runs)
  used=all(r['route']['specdec2_mtp2_used'] and r['route']['specdec2_mtp2_cycles']>0 for r in runs[1:])
  item={'id':case['id'],'category':case['category'],'prompt_tokens':len(case['tokens']),'runs':runs,'exact':passed,'mtp_used':used}
  result['cases'].append(item); outpath.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
  print(case['id'],'exact',passed,'mtp_used',used,'cycles',runs[1]['route']['specdec2_mtp2_cycles'],flush=True)
 result['generate_text_matches']=llm.generate([cases[0]['tokens']],SamplingParams(max_tokens=64,temperature=0,ignore_eos=True,speculative_mtp=True,kv_storage='bf16'))[0]==result['cases'][0]['runs'][1]['text']
 result['final_snapshot']=llm.live_loop_snapshot()
result['memory_after_close']=memory_stats()
result['passed']=all(c['exact'] and c['mtp_used'] for c in result['cases']) and result['generate_text_matches'] and result['memory_after_close']['active_allocations']==0
result['provenance']=collect_artifact_provenance(repo_root=root,configured_backend='hip_gfx1151',resolved_backend='hip_gfx1151',target_arch='gfx1151',model_path=model,quant='w4_paro',kv_dtype='bf16',command=[sys.executable,*sys.argv],build_profile='PARO staged adapter correctness',timing_protocol='no performance claim',warmups=0,repetitions=2,profiler={'enabled':False,'reason':'control-path repair'},hipcc_version=Path('/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt').read_text())
outpath.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('PASSED',result['passed'],flush=True)
sys.exit(0 if result['passed'] else 1)
