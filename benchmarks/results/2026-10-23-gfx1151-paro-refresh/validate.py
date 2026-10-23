"""PARO refresh validation; run from the clean measured repository."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_native_prefill_fixture_gate import _read_logits, _compare_logits
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True)
p.add_argument('--mtp-model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mode', choices=['numerical', 'public'], required=True)
a = p.parse_args()
model = Path(a.model)
root = Path.cwd()
compiler = Path('/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt').read_text()
rows = [json.loads(line) for line in (root/'benchmarks/prompts/mtpbench-code-general-ja.jsonl').read_text().splitlines() if line.strip()]
tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'))
cases = []
for row in rows:
    text = row.get('prompt') or row['messages'][0]['content']
    rendered = '<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n'
    cases.append({'id':row['id'], 'category':row['category'], 'tokens':tokenizer.encode(rendered).ids})

def digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()

result = {'schema':1, 'performance_claim':False, 'mode':a.mode,
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'thresholds':{'serial_native_max_kl':0.05, 'serial_native_min_top1':0.90},
          'notes':['Serial/native uses the same checkpoint, not an independent quantization-quality oracle.',
                   'No arithmetic or production-profile promotion is performed by this refresh.'],
          'cases':[]}
started = time.perf_counter()
if a.mode == 'numerical':
    runner = Qwen35ParoNextTokenRunner(model, backend='hip_gfx1151', shared_expert_format='packed_paro_w4')
    with Qwen35ParoResidentSession(runner, max_sequence_length=4352,
            compiler_version=compiler, prefill_config=PrefillConfig(attn_aotriton_min_tokens=512)) as session:
        for case in cases:
            prompt = case['tokens']
            session.reset()
            session._resolve_prefill_config_for_length(len(prompt))
            for pos, token in enumerate(prompt):
                seed = session.step(token, position=pos, sample=pos == len(prompt)-1)
            teacher_ids = [int(seed.token_id)]
            teacher_logits = [_read_logits(session)]
            for offset in range(8):
                seed = session.step(teacher_ids[-1], position=len(prompt)+offset, sample=True)
                teacher_ids.append(int(seed.token_id))
                teacher_logits.append(_read_logits(session))
            session.reset()
            session._resolve_prefill_config_for_length(len(prompt))
            seed = session.prefill_native(prompt, sample=True)
            native_logits = [_read_logits(session)]
            for offset in range(8):
                session.step(teacher_ids[offset], position=len(prompt)+offset, sample=True)
                native_logits.append(_read_logits(session))
            comparison = _compare_logits(teacher_logits, native_logits)
            finite = all(np.isfinite(v).all() for v in teacher_logits+native_logits)
            result['cases'].append({'id':case['id'], 'category':case['category'], 'prompt_ids':prompt,
                'teacher_ids':teacher_ids, 'finite':bool(finite), 'comparison':comparison,
                'passed':bool(finite and comparison['max_kl'] <= .05 and comparison['top1_agreement'] >= .9)})
            print('numerical',case['id'],result['cases'][-1]['passed'],comparison,flush=True)
        graph_cases = cases + [{'id':f'p{n}', 'category':'repeated-token', 'tokens':[9707]*n} for n in (512,1024,4096)]
        result['graph_cases'] = []
        for case in graph_cases:
            prompt = case['tokens']
            runs = []
            for mode in ('eager', 'graph', 'graph'):
                session.reset()
                session._resolve_prefill_config_for_length(len(prompt))
                seed = session.prefill_native(prompt, sample=True)
                ids = [int(seed.token_id)]
                if mode == 'eager':
                    for offset in range(32):
                        seed = session.step(ids[-1],position=len(prompt)+offset,sample=True)
                        ids.append(int(seed.token_id))
                else:
                    graph = session.capture_decode_graph(position=len(prompt),steps_per_replay=1,max_replay_steps=32,record_steps=32)
                    try:
                        graph.replay(32)
                        ids.extend(int(i) for i in graph.read_generated_token_ids(32))
                    finally:
                        graph.close()
                logits = _read_logits(session)
                runs.append({'mode':mode,'ids':ids,'logits_sha256':digest(logits),'finite':bool(np.isfinite(logits).all())})
            passed = all(r['finite'] and r['ids']==runs[0]['ids'] and r['logits_sha256']==runs[0]['logits_sha256'] for r in runs)
            result['graph_cases'].append({'id':case['id'],'prompt_tokens':len(prompt),'runs':runs,'passed':passed})
            print('graph',case['id'],passed,flush=True)
    result['passed'] = all(c['passed'] for c in result['cases']+result['graph_cases'])
else:
    params = SamplingParams(max_tokens=33,temperature=0.0,ignore_eos=True,speculative_mtp=False,kv_storage='bf16')
    with closing(LLM(str(model),backend='hip_gfx1151',quant='w4_paro',max_active_requests=1,
             max_sequence_length=4352,prefix_cache='off')) as llm:
        result['profile'] = {'requested':llm.execution_profile,'resolved':llm.resolved_execution_profile,
                'manifest':llm.execution_profile_manifest,'manifest_sha256':llm.execution_profile_manifest_sha256,
                'strict_manifest_sha256':llm.execution_profile_strict_manifest_sha256}
        for case in cases + [{'id':'p512','category':'repeated-token','tokens':[9707]*512}]:
            runs=[]
            for _ in range(2):
                output=llm.generate_detailed([case['tokens']],params)[0]
                runs.append({'ids':list(output.generated_token_ids),'text':output.text,'telemetry':output.telemetry.to_json_dict()})
            result['cases'].append({'id':case['id'],'prompt_ids':case['tokens'],'runs':runs,
                    'passed':runs[0]['ids']==runs[1]['ids'] and len(runs[0]['ids'])==33})
            print('public',case['id'],result['cases'][-1]['passed'],flush=True)
        result['generate_text_matches_detailed'] = llm.generate([cases[0]['tokens']],params)[0] == result['cases'][0]['runs'][0]['text']
        result['generator_type'] = type(llm._get_text_generator()).__name__
        result['resolved_backend'] = llm.resolved_backend
        result['resolved_quant'] = llm.resolved_quant
    with closing(LLM(a.mtp_model,backend='hip_gfx1151',quant='w4_paro',max_active_requests=1,
             max_sequence_length=512,prefix_cache='off')) as llm:
        try:
            output=llm.generate_detailed([cases[0]['tokens']],SamplingParams(max_tokens=8,temperature=0.0,speculative_mtp=True,kv_storage='bf16'))
            result['mtp']={'status':'ran','outputs':[o.text for o in output], 'telemetry':[o.telemetry.to_json_dict() for o in output]}
        except (ValueError,NotImplementedError,RuntimeError) as exc:
            result['mtp']={'status':'unsupported','error_type':type(exc).__name__,'error':str(exc)}
        print('mtp',result['mtp'],flush=True)
    result['passed'] = all(c['passed'] for c in result['cases']) and result['generate_text_matches_detailed']
result['memory_after_close'] = memory_stats()
result['diagnostic_wall_seconds'] = time.perf_counter()-started
result['provenance'] = collect_artifact_provenance(repo_root=root, configured_backend='hip_gfx1151',
    resolved_backend='hip_gfx1151',target_arch='gfx1151',model_path=model,quant='w4_paro',kv_dtype='bf16',
    command=[sys.executable,*sys.argv],build_profile='PARO refresh correctness',timing_protocol='untimed correctness checks',
    warmups=0,repetitions=2 if a.mode=='public' else 1,profiler={'enabled':False,'reason':'correctness run'},hipcc_version=compiler)
Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('RESULT',a.output,'passed',result['passed'],flush=True)
sys.exit(0 if result['passed'] else 1)
