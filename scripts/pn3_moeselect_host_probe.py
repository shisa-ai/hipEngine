"""PN3-MOESELECT host-vs-GPU: accumulated pure host wall in the selected-expert
dispatch wrappers vs the complete-wall drop from no-op'ing them.

Wraps the selected-expert launch helpers and times the pure Python host wall
(launch is async; the host wall excludes GPU execution). If the accumulated
host wall ~equals the no-op complete-wall drop, the selected-expert slice is
host-bound and a dispatch fast-path is the mechanism.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

WRAPPED = ("_launch_selected_raw_gguf_moe_pair_silu",
           "_launch_selected_raw_gguf_moe_pair",
           "_launch_selected_raw_gguf_moe_linear")
_host_us = [0.0]
_counts = [0]

def make_wrapper(name, orig):
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _host_us[0] += (time.perf_counter() - t0) * 1e6
            _counts[0] += 1
    w.__name__ = name
    return w

def run(s, n=30, warmup=6):
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    _host_us[0] = 0.0; _counts[0] = 0
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(walls)

saved = {name: getattr(rm, name) for name in WRAPPED}
for name in WRAPPED:
    setattr(rm, name, make_wrapper(name, saved[name]))
try:
    with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
            max_sequence_length=900, backend='hip_gfx1151') as s:
        s.prefill([9707]*512, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        wall = run(s)
        print(f"complete wall={wall:.2f} ms/tok")
        print(f"selected-expert host wall={_host_us[0]/30:.2f} us/tok over {_counts[0]//30} launches/step")
        print(f"host fraction of complete wall={_host_us[0]/30/1000/wall*100:.0f}%")
        print(f"per-launch host wall={_host_us[0]/max(_counts[0],1):.1f} us")
finally:
    for name, orig in saved.items():
        setattr(rm, name, orig)
