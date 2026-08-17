"""Decisive MoE-expert / router GPU-cost probe for the Qwen3.6-35B eager decode.

No-ops the MoE expert GEMV launches (via launch_gguf_linear) and separately the
pair/triple launchers to measure the real complete-wall contribution of the
largest GPU slices (marker walls overlap and overstate). Wall-timing only
(results are WRONG after each no-op).
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
import hipengine.runtime.gguf_linear as gl
ctypes.CDLL('libamdhip64.so')

def noop(*a, **k):
    return None

def run_steps(s, n=30, warmup=6, noops=()):
    # apply noops
    saved = {}
    for name in noops:
        saved[name] = getattr(rm, name)
        setattr(rm, name, noop)
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    for name, orig in saved.items():
        setattr(rm, name, orig)
    return statistics.median(walls)

compiler = open('/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt').read().strip() or None
with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=False, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    base = run_steps(s, noops=())
    no_lgl = run_steps(s, noops=("launch_gguf_linear",))
    no_lgl_pair = run_steps(s, noops=("launch_gguf_linear", "launch_gguf_linear_pair"))
    no_lgl_pair_triple = run_steps(s, noops=("launch_gguf_linear", "launch_gguf_linear_pair", "launch_gguf_linear_triple"))
    print(f"baseline          ={base:.2f} ms/tok")
    print(f"no launch_gguf    ={no_lgl:.2f}  drop {(base-no_lgl)*1000:+.1f} us/tok")
    print(f"no + pair         ={no_lgl_pair:.2f}  drop {(base-no_lgl_pair)*1000:+.1f} us/tok")
    print(f"no + pair + triple={no_lgl_pair_triple:.2f}  drop {(base-no_lgl_pair_triple)*1000:+.1f} us/tok")
