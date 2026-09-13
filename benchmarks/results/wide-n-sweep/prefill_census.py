import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from collections import defaultdict
from hipengine.core.hip import get_hip_runtime
from hipengine.runtime import gguf_linear as gl

rt = get_hip_runtime()
events = []
DEPTH = {'n': 0}
_orig_linear = gl.launch_gguf_linear
_orig_pair = gl.launch_gguf_linear_pair
_orig_pair_silu = gl.launch_gguf_linear_pair_silu

def timed(label, fn, *a, **kw):
    DEPTH['n'] += 1
    outer = DEPTH['n'] == 1
    if outer:
        s = rt.event_create(); e = rt.event_create()
        rt.event_record(s)
        events.append((label, s, e))
        try:
            return fn(*a, **kw)
        finally:
            rt.event_record(e); DEPTH['n'] -= 1
    try:
        return fn(*a, **kw)
    finally:
        DEPTH['n'] -= 1

def linear(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw):
    return timed(f"single:{weight.spec.quant_key}:{rows}", _orig_linear,
                 weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)

def pair(a, b, x_ptr, oa, ob, rows, in_features, out_features, **kw):
    return timed(f"pair:{a.spec.quant_key}+{b.spec.quant_key}:{rows}", _orig_pair,
                 a, b, x_ptr, oa, ob, rows, in_features, out_features, **kw)

def pair_silu(a, b, x_ptr, out_ptr, rows, in_features, out_features, **kw):
    return timed(f"pair_silu:{a.spec.quant_key}+{b.spec.quant_key}:{rows}", _orig_pair_silu,
                 a, b, x_ptr, out_ptr, rows, in_features, out_features, **kw)

gl.launch_gguf_linear = linear
gl.launch_gguf_linear_pair = pair
gl.launch_gguf_linear_pair_silu = pair_silu
import hipengine.runtime.qwen35_gguf_runner as runner_mod
runner_mod.launch_gguf_linear = linear
runner_mod.launch_gguf_linear_pair = pair
runner_mod.launch_gguf_linear_pair_silu = pair_silu

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
model = sys.argv[1]
with Qwen35GGUFResidentSession(
    model, compiler_version=open('/tmp/ud-hipcc-version.txt').read(),
    max_sequence_length=600, use_wmma_prefill=True, use_gemv_decode=True,
) as session:
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)  # warm
    events.clear()
    t0 = time.perf_counter()
    session.reset()
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)
    rt.device_synchronize()
    wall = (time.perf_counter() - t0) * 1000
    acc = defaultdict(float); counts = defaultdict(int)
    for label, s, e in events:
        acc[label] += rt.event_elapsed_time_ms(s, e)
        counts[label] += 1
    print(f"model: {model.split('/')[-1]}")
    print(f"prefill wall (512 rows): {wall:.1f} ms")
    total = sum(acc.values())
    print(f"timed linear: {total:.1f} ms ({100*total/wall:.1f}% of wall)")
    print("by owner (ms, % of wall):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:18]:
        print(f"  {k:<46} {v:8.1f} ms ({100*v/wall:4.1f}%)  x{counts[k]}")
