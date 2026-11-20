"""Fresh-owner public acceptance traces without extra GPU synchronization."""
import json
from pathlib import Path
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier as V
original=V.prepare
traces=[]
def prepare(self,*args,**kwargs):
    p=original(self,*args,**kwargs)
    traces.append(dict(positions=list(p.batch.positions),tokens=list(p.batch.tokens),
                       top1=list(p.target_top1),proposal_values=list(p.device_proposal_top1_values),
                       accepted=list(p.summary.accepted_counts),next_tokens=list(p.summary.next_tokens)))
    return p
V.prepare=prepare
prompt=next(json.loads(s) for s in Path('benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl').read_text().splitlines() if json.loads(s)['id']=='heldout_general_ja_speculative')
text=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in prompt['messages'])+'<|im_start|>assistant\n'
params=SamplingParams(max_tokens=64,temperature=0,ignore_eos=True,speculative_mtp=True)
results=[]
for owner in range(6):
    llm=LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',kv_storage='bf16',max_sequence_length=2048,prefix_cache='off',speculative_candidate_budget=3)
    try:
        llm.generate_detailed(['List three uses of a hash table.'],params)
        for repeat in range(3):
            traces.clear()
            row=llm.generate_detailed([text],params)[0]
            d=row.telemetry.to_json_dict()['diagnostics']['specdec2_mtp2']
            results.append(dict(owner=owner,repeat=repeat,ids=list(row.generated_token_ids),controls=d,traces=list(traces)))
            print(owner,repeat,d['cycles'],d['accepted_draft_tokens'],len(traces),flush=True)
            Path('/tmp/mtp-repeatability-traces.json').write_text(json.dumps(results,indent=2))
    finally:
        llm.close()
    print('closed',memory_stats(),flush=True)
V.prepare=original
