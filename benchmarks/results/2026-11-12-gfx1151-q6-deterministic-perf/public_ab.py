"""Balanced public comparison of parent timed and deterministic Q6 selection."""
import argparse, hashlib, json, os, platform, subprocess, time
from pathlib import Path
import numpy as np
from hipengine import LLM, SamplingParams
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as q6

SOURCE='hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_f16_rocblas_prefill.py'
BASE='197bf61ae^'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--passes',type=int,default=2)
    args=parser.parse_args()
    parent=subprocess.check_output(['git','show',f'{BASE}:{SOURCE}'],text=True)
    start=parent.index('def _launch_q6_fullgemm(')
    end=parent.index('\n# Dispatch passes',start)
    namespace=dict(vars(q6))
    exec(compile(parent[start:end],'<parent-q6-selector>','exec'),namespace)
    old=namespace['_launch_q6_fullgemm'];new=q6._launch_q6_fullgemm
    prompt_path=Path('benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl')
    prompts=[json.loads(s) for s in prompt_path.read_text().splitlines()]
    rt=get_hip_runtime()
    payload=dict(host=platform.node(),hardware='Radeon 8060S',backend='hip_gfx1151',
        model='/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',kv='bf16',budget=3,max_tokens=64,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline_commit=subprocess.check_output(['git','rev-parse',BASE],text=True).strip(),
        parent_source_sha256=hashlib.sha256(parent.encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        command=' '.join(__import__('sys').argv),
        env={k:v for k,v in os.environ.items() if k.startswith(('HIPENGINE_','HIP_VISIBLE','ROCM','HIP_PATH','LD_LIBRARY_PATH'))},
        initial_memory=memory_stats(),results=[])
    for arm in ('parent','fixed','fixed','parent'):
        q6._launch_q6_fullgemm=old if arm=='parent' else new
        started=time.perf_counter()
        llm=LLM(payload['model'],backend='hip_gfx1151',kv_storage='bf16',max_sequence_length=2048,
                prefix_cache='off',speculative_candidate_budget=3)
        rt.device_synchronize()
        result=dict(arm=arm,setup_s=time.perf_counter()-started,profile=llm.resolved_execution_profile,
                    manifest=llm.execution_profile_manifest_sha256,samples=[])
        payload['results'].append(result)
        params=SamplingParams(max_tokens=64,temperature=0,ignore_eos=True,speculative_mtp=True)
        try:
            start=time.perf_counter()
            llm.generate_detailed(['List three uses of a hash table.'],params)
            rt.device_synchronize()
            result['warmup_s']=time.perf_counter()-start
            for repeat in range(args.passes):
                for p in prompts:
                    text=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in p['messages'])+'<|im_start|>assistant\n'
                    start=time.perf_counter()
                    row=llm.generate_detailed([text],params)[0]
                    rt.device_synchronize()
                    sample=dict(id=p['id'],category=p['category'],repeat=repeat,wall_s=time.perf_counter()-start,
                                ids=list(row.generated_token_ids),telemetry=row.telemetry.to_json_dict())
                    result['samples'].append(sample)
                    print(arm,repeat,p['id'],sample['wall_s'],flush=True)
                    Path(args.out).write_text(json.dumps(payload,indent=2))
        finally:
            llm.close()
            q6._launch_q6_fullgemm=new
        result['memory_after_close']=memory_stats()
        Path(args.out).write_text(json.dumps(payload,indent=2))
    payload['complete']=True
    Path(args.out).write_text(json.dumps(payload,indent=2))

if __name__=='__main__':main()
