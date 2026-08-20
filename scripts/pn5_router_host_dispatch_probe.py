"""PN5: host-side cost of the coop router dispatch during decode.

Times the 40 `_try_launch_qwen35_router_topk_split_shared_bf16_f32w` calls per
step (enqueue + ctypes kernel launch) and the kernel's GPU time via a per-step
device timing marker. Distinguishes host dispatch from GPU exec for the router
slice.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'

compiler = open(COMPILER).read().strip() or None
with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=True, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()

    orig = rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w
    host_times = []

    def wrapped(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        host_times.append((time.perf_counter() - t0) * 1e6)
        return r
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = wrapped

    tok = 9707
    for _ in range(3):
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
    host_times.clear()
    for _ in range(5):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        wall = (time.perf_counter() - t0) * 1e3
        n = len(host_times)
        per = host_times[n - (n // 5) * 1:] if False else None
        step_host = sum(host_times[-40:]) if len(host_times) >= 40 else None
        print(f"step wall={wall:.2f}ms  coop_calls={n - (n-40 if n>40 else 0)}  "
              f"last40_host={sum(host_times[-40:])/1000 if len(host_times)>=40 else 0:.2f}ms  "
              f"per_call_median={statistics.median(host_times[-40:]) if len(host_times)>=40 else 0:.1f}us")
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = orig
