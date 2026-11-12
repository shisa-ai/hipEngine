"""Same-owner public copy A/B; optional live snapshot byte oracle (not timing evidence)."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from hipengine import LLM, SamplingParams
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr, memory_stats
from hipengine.runtime.qwen35_gguf_mtp import _StateJournal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
p.add_argument('--check-bytes', action='store_true')
a = p.parse_args()
prompts = [json.loads(s) for s in Path('benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl').read_text().splitlines()]
if a.check_bytes:
    prompts = [p for p in prompts if p['id'] == 'heldout_general_ja_speculative'] * 3
runtime = get_hip_runtime()
original = _StateJournal._copy_initial_state
arm = 'batch'
checked_bytes = 0
calls = 0

def read(ptr, nbytes):
    data = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(data), DeviceBuffer(ptr, nbytes), runtime=runtime)
    return data

def execute(self, *, restore, stream):
    global checked_bytes, calls
    if arm == 'loop':
        return False
    result = original(self, restore=restore, stream=stream)
    if result:
        calls += 1
    if result and a.check_bytes:
        runtime.device_synchronize()
        for state, snapshot in self.state_rows:
            np.testing.assert_array_equal(read(state.ptr, state.nbytes), read(snapshot.ptr, state.nbytes))
            checked_bytes += state.nbytes
    return result

_StateJournal._copy_initial_state = execute
results = []
llm = LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf', backend='hip_gfx1151',
          kv_storage='bf16', max_sequence_length=2048, prefix_cache='off', speculative_candidate_budget=3)
params = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True, speculative_mtp=True)
try:
    llm.generate_detailed(['List three uses of a hash table.'], params)
    for arm in ('loop', 'batch', 'batch', 'loop'):
        samples = []
        for prompt in prompts:
            text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in prompt['messages']) + '<|im_start|>assistant\n'
            start = time.perf_counter()
            row = llm.generate_detailed([text], params)[0]
            runtime.device_synchronize()
            samples.append(dict(id=prompt['id'], wall_s=time.perf_counter()-start, ids=list(row.generated_token_ids), telemetry=row.telemetry.to_json_dict()))
            print(arm, prompt['id'], samples[-1]['wall_s'], flush=True)
        results.append(dict(arm=arm, samples=samples))
finally:
    llm.close()
    _StateJournal._copy_initial_state = original
keys = ('cycles', 'accepted_draft_tokens', 'generated_draft_tokens', 'selected_depth_histogram', 'failure_reason_counts', 'ar_step_reason_counts')
checks = []
for result in results:
    for expected, got in zip(results[0]['samples'], result['samples'], strict=True):
        x = expected['telemetry']['diagnostics']['specdec2_mtp2']
        y = got['telemetry']['diagnostics']['specdec2_mtp2']
        checks.append(dict(arm=result['arm'], id=got['id'], ids_equal=expected['ids']==got['ids'], control_equal=all(x[k]==y[k] for k in keys)))
payload = dict(results=results, checks=checks, checked_bytes=checked_bytes, batch_calls=calls, memory_after_close=memory_stats(), passed=all(c['ids_equal'] and c['control_equal'] for c in checks))
Path(a.out).write_text(json.dumps(payload, indent=2))
assert payload['passed'], 'same-owner output/control mismatch'
