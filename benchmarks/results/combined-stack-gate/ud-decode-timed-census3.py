import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from collections import defaultdict
from hipengine.core.hip import get_hip_runtime
from hipengine.runtime import gguf_linear as gl

rt = get_hip_runtime()
events = []  # (label, start_ev, stop_ev)
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
            rt.event_record(e)
            DEPTH['n'] -= 1
    else:
        try:
            return fn(*a, **kw)
        finally:
            DEPTH['n'] -= 1

def linear(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw):
    return timed(f"single:{weight.spec.quant_key}", _orig_linear,
                 weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)

def pair(a, b, x_ptr, oa, ob, rows, in_features, out_features, **kw):
    return timed(f"pair:{a.spec.quant_key}+{b.spec.quant_key}", _orig_pair,
                 a, b, x_ptr, oa, ob, rows, in_features, out_features, **kw)

def pair_silu(a, b, x_ptr, out_ptr, rows, in_features, out_features, **kw):
    return timed(f"pair_silu:{a.spec.quant_key}+{b.spec.quant_key}", _orig_pair_silu,
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
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
with Qwen35GGUFResidentSession(
    model, compiler_version=open('/tmp/ud-hipcc-version.txt').read(),
    max_sequence_length=64, use_wmma_prefill=True, use_gemv_decode=True,
) as session:
    cur = session.prefill([9707] * 8, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)
    cur = session.step(int(cur.token_id), return_logits=True)
    # Capture the decode graph with the event-instrumented wrappers active:
    # the event records become graph nodes, so each replay re-records them and
    # the elapsed times reflect steady-state graph-replay execution (the eager
    # path is CPU-bound and downclocks the GPU, inflating kernel times).
    events.clear()
    graph = session.capture_decode_graph(position=9, steps_per_replay=n_steps,
                                         max_replay_steps=2 * n_steps)
    graph.replay(n_steps)   # warm; events now hold these durations
    rt.device_synchronize()
    import time as _t
    t0 = _t.perf_counter()
    graph.replay(n_steps)   # measured; events overwritten with steady-state
    rt.device_synchronize()
    wall = (_t.perf_counter() - t0) / n_steps * 1000
    acc = defaultdict(float); counts = defaultdict(int)
    for label, s, e in events:
        acc[label] += rt.event_elapsed_time_ms(s, e)
        counts[label] += 1
    print(f"model: {model.split('/')[-1]}  steps: {n_steps}")
    print(f"wall per step: {wall:.2f} ms")
    total = sum(acc.values()) / n_steps
    print(f"timed linear (outermost): {total:.2f} ms/token ({100*total/wall:.1f}% of wall)")
    print("by owner family (ms/token, % of wall):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        v /= n_steps
        print(f"  {k:<44} {v:7.2f} ms  ({100*v/wall:4.1f}%)  x{counts[k]//n_steps}/step")
