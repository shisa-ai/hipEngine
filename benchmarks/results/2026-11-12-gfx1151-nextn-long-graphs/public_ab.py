"""Public category/held-out NextN schedule comparison, no target changes."""
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from collections import Counter
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor as E

original = E._launch_graph_chain_device
arm = 'graph'
calls = []
def launch(self, request_id, token_id, position, target_hidden, *, candidate_budget, cached_only):
    if arm == 'eager':
        return None
    result = original(self, request_id, token_id, position, target_hidden,
                      candidate_budget=candidate_budget, cached_only=cached_only)
    calls.append(dict(position=position, budget=candidate_budget, status=self._proposal_graph_last_status,
                      error=self._proposal_graph_last_error, launched=result is not None))
    return result
E._launch_graph_chain_device = launch
source=Path('benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl')
prompts=[json.loads(line) for line in source.read_text().splitlines()]
llm = LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf', backend='hip_gfx1151',
          kv_storage='bf16', max_sequence_length=4608, prefix_cache='off', speculative_candidate_budget=3)
params = SamplingParams(max_tokens=48, temperature=0, ignore_eos=True, speculative_mtp=True)
results=[]
lengths=(973, 1017, 1041, 1269, 1401, 4081)
try:
    texts=[]
    for i,prompt in enumerate(prompts):
        text=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in prompt['messages'])+'<|im_start|>assistant\n'
        target=lengths[i % len(lengths)]
        # Neutral context padding; task messages remain intact, no token-ID selection.
        pad='Reference notes.\n' + 'A software project has source files and tests.\n' * max(0,(target-llm.count_tokens(text))//10)
        text=pad+text
        texts.append(text)
    for arm in ('eager', 'graph', 'graph', 'eager'):
        llm.generate_detailed(['List three uses of a hash table.'], params)
        samples=[]
        for prompt,text in zip(prompts,texts,strict=True):
            calls.clear()
            start=time.perf_counter()
            row=llm.generate_detailed([text], params)[0]
            sample=dict(id=prompt['id'],category=prompt['category'],wall_s=time.perf_counter()-start,
                        ids=list(row.generated_token_ids),telemetry=row.telemetry.to_json_dict(),calls=list(calls))
            samples.append(sample)
            print(arm,prompt['id'],sample['telemetry']['decode_state']['prompt_tokens'],sample['wall_s'],Counter(c['status'] for c in calls),flush=True)
            Path('/tmp/nextn-long-public-progress.json').write_text(json.dumps(dict(results=results,current=samples),indent=2))
        results.append(dict(arm=arm,samples=samples))
finally:
    llm.close()
    E._launch_graph_chain_device=original
keys=('cycles','accepted_draft_tokens','generated_draft_tokens','selected_depth_histogram','failure_reason_counts','ar_step_reason_counts')
checks=[]
for r in results:
    for expected,got in zip(results[0]['samples'],r['samples'],strict=True):
        a=expected['telemetry']['diagnostics']['specdec2_mtp2']; b=got['telemetry']['diagnostics']['specdec2_mtp2']
        checks.append(dict(arm=r['arm'],id=got['id'],ids_equal=expected['ids']==got['ids'],
                           control_equal=all(a[k]==b[k] for k in keys),
                           graph_engaged=r['arm']=='eager' or any(c['launched'] for c in got['calls']),
                           no_capture_errors=not any(c['error'] for c in got['calls'])))
payload=dict(results=results,checks=checks,memory_after_close=memory_stats(),host=platform.node(),
             source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             prompt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             profile=llm.resolved_execution_profile,manifest=llm.execution_profile_manifest_sha256)
payload['passed']=all(all(c[k] for k in ('ids_equal','control_equal','graph_engaged','no_capture_errors')) for c in checks)
Path('/tmp/nextn-long-public.json').write_text(json.dumps(payload,indent=2))
assert payload['passed'], 'output/control/graph-path check failed'
