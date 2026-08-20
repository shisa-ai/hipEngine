"""PN6 audit: per-family build_X(load=True) cost + launch_gguf_linear host slice.

Classifies the GEMV launch families host-vs-GPU bound for the Qwen3.6-35B eager
decode on the ZBook (gfx1151), mirroring the PN5 router audit:

1. per-call host time of `launch_gguf_linear` during decode;
2. per-family `build_X(load=True)` cost with a pinned session compiler version
   (the per-call launch path passes compiler_version=None -> loaded-library
   cache miss -> full build/load machinery);
3. no-op A/B: recoverable wall for the whole launch_gguf_linear slice.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'
compiler = open(COMPILER).read().strip() or None

def noop(*a, **k): return None

with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=True, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()

    # --- per-family build_X(load=True) cost in-session (pinned version) ---
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_gemv as m1
    from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_t16_gemv as m2
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_gemv as m3
    from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as m4
    for name, mod, build in (
        ("q8_0_t16_gemv", m1, m1.build_gguf_q8_0_t16_gemv),
        ("q6_k_t16_gemv", m2, m2.build_gguf_q6_k_t16_gemv),
        ("q4_k_gemv", m3, m3.build_gguf_q4_k_gemv),
        ("t16_selected_gemv", m4, m4.build_gguf_t16_selected_gemv),
    ):
        try:
            build(load=True)  # warm
        except Exception as e:
            print(f"{name}: warm failed {e}")
            continue
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            build(load=True)
            ts.append((time.perf_counter() - t0) * 1e6)
        print(f"build {name:20s} (load=True, version=None): n={len(ts)} median={statistics.median(ts):.1f}us p90={sorted(ts)[180]:.1f}us")

    # --- per-call host time of launch_gguf_linear over 2 steps ---
    orig_lgl = rm.launch_gguf_linear
    host = []
    def wrap_lgl(*a, **k):
        t0 = time.perf_counter()
        r = orig_lgl(*a, **k)
        host.append((time.perf_counter() - t0) * 1e6)
        return r
    rm.launch_gguf_linear = wrap_lgl
    tok = 9707
    for _ in range(2):
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
    n = len(host)
    rm.launch_gguf_linear = orig_lgl
    print(f"launch_gguf_linear: {n} calls/2 steps = {n//2}/step  median={statistics.median(host):.1f}us  total/step={sum(host[-n//2:])/1000:.2f}ms")

    # --- no-op A/B: recoverable wall for the slice ---
    def run_steps(n=30, warmup=6, noops=()):
        saved = {}
        for name in noops:
            saved[name] = getattr(rm, name)
            setattr(rm, name, noop)
        t = 9707
        for _ in range(warmup):
            t = int(s.step(t, return_logits=False).token_id)
        walls = []
        for _ in range(n):
            t0 = time.perf_counter()
            t = int(s.step(t, return_logits=False).token_id)
            s.runtime.device_synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
        for name, o in saved.items():
            setattr(rm, name, o)
        return statistics.median(walls)
    base = run_steps(noops=())
    no_lgl = run_steps(noops=("launch_gguf_linear",))
    print(f"baseline ={base:.2f} ms/tok   no launch_gguf_linear ={no_lgl:.2f}  drop={(base-no_lgl)*1000:+.0f} us/tok")
