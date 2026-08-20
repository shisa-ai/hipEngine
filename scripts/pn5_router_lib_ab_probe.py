"""PN5: exact + wall A/B for the hoisted router library cache.

- "old": monkeypatch _router_library to rebuild per call (reproduces the
  pre-fix `build_qwen35_router(load=True)` per-call path).
- "new": default module-level cached library.
Token trajectories must be identical; wall should drop by the router host slice.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.kernels.hip_gfx1100.moe import router as router_mod
from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
ctypes.CDLL('libamdhip64.so')

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'
compiler = open(COMPILER).read().strip() or None

def run(s, mode, n=30, warmup=6, start=9707):
    if mode == "old":
        old = router_mod._router_library
        def rebuild(*a, **k):
            return build_qwen35_router(load=True)
        router_mod._router_library = rebuild
    tok = start
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    traj = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        traj.append(tok)
    if mode == "old":
        router_mod._router_library = old
    return statistics.median(walls), traj

with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=True, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    # counter-rotate: old first, then new, then old again to control drift
    w_old1, t_old1 = run(s, "old")
    w_new, t_new = run(s, "new", start=t_old1[-1])
    w_old2, t_old2 = run(s, "old", start=t_new[-1])
    print(f"old (per-call build)  #1: {w_old1:.2f} ms/tok")
    print(f"new (cached library)     : {w_new:.2f} ms/tok")
    print(f"old (per-call build)  #2: {w_old2:.2f} ms/tok")
    print(f"tokens identical old1 vs new: {t_old1 == t_new}")
    print(f"tokens identical new vs old2: {t_new == t_old2}")
    drop = (w_old1 + w_old2) / 2 - w_new
    print(f"mean old={(w_old1+w_old2)/2:.2f}  new={w_new:.2f}  drop={drop*1000:+.0f} us/tok ({(w_old1+w_old2)/2/w_new*100 - 100:+.1f}%)")
