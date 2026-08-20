"""PN3-MOEHOIST probe: per-call MoE group_scatter build cost + A/B wall.

Measures the per-call `build_qwen35_moe_group_scatter(load=True)` host cost and
the full-step wall A/B (per-call build vs module-cached handle) on the
35B-A3B GGUF c1 decode. Mirrors PN5's router-lib probe.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine-main')
ctypes.CDLL('libamdhip64.so')

import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.kernels.hip_gfx1100.moe import group_scatter as gs

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'
compiler = open(COMPILER).read().strip() or None

# cached handle (the "new" path)
_CACHED = None

def cached_build(*a, **k):
    global _CACHED
    if _CACHED is None:
        _CACHED = gs.build_qwen35_moe_group_scatter(load=True)
    return _CACHED

def per_call_cost(n=50):
    # time the per-call build (default path) with the compiler pinned
    t0 = time.perf_counter()
    for _ in range(n):
        gs.build_qwen35_moe_group_scatter(load=True)
    dt = (time.perf_counter() - t0) / n
    print(f"per-call build_qwen35_moe_group_scatter(load=True): {dt*1e6:.1f} us/call")

def run(s, mode, n=30, warmup=6, start=9707):
    if mode == "old":
        orig = gs.build_qwen35_moe_group_scatter
        gs.build_qwen35_moe_group_scatter = orig  # default per-call
    else:
        gs.build_qwen35_moe_group_scatter = cached_build
    tok = start
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls, traj = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        traj.append(tok)
    gs.build_qwen35_moe_group_scatter = orig
    return statistics.median(walls), traj

orig = gs.build_qwen35_moe_group_scatter
with rm.Qwen35GGUFResidentSession(MODEL, max_sequence_length=900,
        compiler_version=compiler, require_cached_build=True,
        backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    per_call_cost()
    w_old1, t_old1 = run(s, "old")
    w_new, t_new = run(s, "new", start=t_old1[-1])
    w_old2, t_old2 = run(s, "old", start=t_new[-1])
    print(f"old (per-call build) #1: {w_old1:.2f} ms/tok")
    print(f"new (cached handle)    : {w_new:.2f} ms/tok")
    print(f"old (per-call build) #2: {w_old2:.2f} ms/tok")
    print(f"tokens identical old1 vs new: {t_old1 == t_new}")
    print(f"tokens identical new vs old2: {t_new == t_old2}")
    mean_old = (w_old1 + w_old2) / 2
    drop = mean_old - w_new
    print(f"mean old={mean_old:.2f}  new={w_new:.2f}  drop={drop*1000:+.0f} us/tok ({mean_old/w_new*100 - 100:+.1f}%)")
