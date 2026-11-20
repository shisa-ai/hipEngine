"""Observe public dense copy/graph ownership and post-close tracked allocations.

This adds Python stack inspection, so its wall times are not performance claims.
"""
from collections import Counter
from functools import wraps
import argparse
import gc
import inspect
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core import memory
from hipengine.core.hip import HipRuntime, get_hip_runtime
from scripts.qwen35_gguf_bench import _RoctxProfilerControl

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True)
p.add_argument('--profile', action='store_true')
p.add_argument('--prompt-tokens', type=int, default=15)
a = p.parse_args()
control = _RoctxProfilerControl(enabled=a.profile)
if a.profile and control._resume is None:
    raise RuntimeError('ROCTx unavailable')
active = False
copies = Counter()
allocations = {}
graph_calls = Counter()

def frames():
    stack = inspect.stack(context=0)[2:]
    return tuple(f'{Path(f.filename).name}:{f.function}:{f.lineno}'
                 for f in stack if '/hipengine/' in f.filename)[:7]

original_record = memory._MEMORY_STATS.record_malloc
def record(buffer):
    allocations[buffer.ptr] = (buffer.nbytes, frames())
    return original_record(buffer)
memory._MEMORY_STATS.record_malloc = record
originals = {}
for name in ('memcpy', 'memcpy_async'):
    original = getattr(HipRuntime, name)
    originals[name] = original
    def make_wrapper(name, original):
        @wraps(original)
        def wrapped(self, dst, src, nbytes, kind, *args, **kwargs):
            if active:
                copies[(name, int(kind), int(nbytes), frames())] += 1
            return original(self, dst, src, nbytes, kind, *args, **kwargs)
        return wrapped
    setattr(HipRuntime, name, make_wrapper(name, original))
original_graph = HipRuntime.graph_launch
@wraps(original_graph)
def graph(self, graph_exec, stream):
    if active:
        graph_calls[(int(graph_exec), frames())] += 1
    return original_graph(self, graph_exec, stream)
HipRuntime.graph_launch = graph

runtime = get_hip_runtime()
llm = LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf', backend='hip_gfx1151',
          max_sequence_length=4096, prefix_cache='off', speculative_candidate_budget=3,
          kv_storage='bf16')
params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True, speculative_mtp=True)
text = 'Explain why transformer inference is usually memory-bandwidth limited on local GPUs.'
tokens = llm.tokenize(text)
if a.prompt_tokens != len(tokens):
    # Fixed-length context control, not a task-quality prompt.
    tokens = (llm.tokenize('This is a neutral technical context. ')
              * a.prompt_tokens)[:a.prompt_tokens]
try:
    llm.generate_detailed([tokens], params)
    runtime.device_synchronize()
    active = True
    with control.region('request', selected='request' if a.profile else ''):
        result = llm.generate_detailed([tokens], params)[0]
        runtime.device_synchronize()
    active = False
    result_data = dict(ids=list(result.generated_token_ids),telemetry=result.telemetry.to_json_dict(),
                       profile=llm.resolved_execution_profile,manifest=llm.execution_profile_manifest_sha256)
finally:
    active = False
    llm.close()
post_close = memory.memory_stats()
del llm
gc.collect()
post_gc = memory.memory_stats()
residual = Counter()
with memory._MEMORY_STATS._lock:
    for ptr,size in memory._MEMORY_STATS._live.items():
        residual[(size, allocations.get(ptr,(None,('unobserved',)))[1])] += 1
payload = dict(prompt_tokens=len(tokens),result=result_data,
               copies=[dict(api=k[0],kind=k[1],nbytes=k[2],stack=k[3],calls=v,total_bytes=k[2]*v)
                       for k,v in copies.most_common()],
               graphs=[dict(handle=k[0],stack=k[1],calls=v) for k,v in graph_calls.items()],
               post_close=post_close,post_gc=post_gc,
               residual_allocations=[dict(nbytes=k[0],stack=k[1],count=v,total_bytes=k[0]*v)
                                     for k,v in residual.most_common()])
Path(a.out).write_text(json.dumps(payload,indent=2)+'\n')
print('DONE',a.prompt_tokens,'copies',sum(copies.values()),'graphs',sum(graph_calls.values()),
      'post_close',post_close,'post_gc',post_gc,flush=True)
