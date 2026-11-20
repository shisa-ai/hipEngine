"""Exercise public Q6 scratch lifetime with overlapping model sessions."""
import gc
import json
from pathlib import Path
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_f16_rocblas_prefill import Q6FullGemmWorkspace

owners=[]
original=Q6FullGemmWorkspace.__init__
def track(self):
    original(self)
    owners.append(self)
Q6FullGemmWorkspace.__init__=track
params=SamplingParams(max_tokens=32,temperature=0,ignore_eos=True,speculative_mtp=True)
def model():
    return LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf',backend='hip_gfx1151',
               max_sequence_length=4096,prefix_cache='off',speculative_candidate_budget=3,kv_storage='bf16')
def run(llm):
    tokens=(llm.tokenize('This is a neutral technical context. ')*2053)[:2053]
    result=llm.generate_detailed([tokens],params)[0]
    d=result.telemetry.to_json_dict()['diagnostics']
    d['specdec2_mtp2'].pop('cycle_timing_ms')
    return dict(ids=list(result.generated_token_ids),diagnostics=d)
records=[]
a=b=c=None
try:
    a=model(); first=run(a); a_owners=list(owners)
    b=model(); second=run(b); b_owners=[x for x in owners if x not in a_owners]
    assert first==second
    a.close();a=None
    assert all(x.closed and not x.resources for x in a_owners)
    assert any(x.resources and not x.closed for x in b_owners)
    records.append(dict(phase='first_closed_second_live',memory=memory_stats()))
    repeat=run(b);assert repeat==second
    b.close();b=None;gc.collect()
    assert memory_stats()['current_allocated_bytes']==0
    records.append(dict(phase='both_closed',memory=memory_stats()))
    c=model();third=run(c);assert third==first
    c.close();c=None;gc.collect()
    assert memory_stats()['current_allocated_bytes']==0
    records.append(dict(phase='reopened_closed',memory=memory_stats()))
    baseline=json.loads(Path('/tmp/qwen38-attribution-long-warm.json').read_text())
    assert first['ids']==baseline['result']['ids']
    Path('/tmp/qwen38-cache-public-lifecycle.json').write_text(json.dumps(dict(
        passed=True,outputs=first,records=records,owner_count=len(owners),
        all_owners_closed=all(x.closed and not x.resources for x in owners)),indent=2)+'\n')
    print('PASS public shared-process close/continue/reopen; no tracked allocations remain',flush=True)
finally:
    for obj in (c,b,a):
        if obj is not None:obj.close()
