"""PN5: measure the MoE router slice of Qwen3.6-35B eager decode on the ZBook.

Phase 1 (census): count which router functions actually fire per decode step
and whether the cooperative fused router (router_topk_split_shared) is active
on gfx1151 vs the separate logits/shared/select chain.

Phase 2 (no-op A/B): complete-wall cost of the router slice by no-opping the
router launches (results are WRONG after each no-op; wall timing only).
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
import hipengine.runtime.gguf_linear as gl
ctypes.CDLL('libamdhip64.so')

def noop(*a, **k):
    return None

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'

def run_steps(s, n=30, warmup=6, noops=()):
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

compiler = open(COMPILER).read().strip() or None
with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=False, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()

    # ---- Phase 1: launch census over one decode step ----
    calls = {"logits_bf16": 0, "logits_f32": 0, "coop_try": 0,
             "coop_ok": 0, "select": 0, "shared": 0}
    orig_logits_b = rm._launch_qwen35_router_logits_bf16_hidden
    orig_logits_f = rm._launch_qwen35_router_logits_f32_hidden
    orig_coop = rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w
    orig_select = rm.qwen35_router_select

    def wrap_logits_b(*a, **k):
        calls["logits_bf16"] += 1
        return orig_logits_b(*a, **k)
    def wrap_logits_f(*a, **k):
        calls["logits_f32"] += 1
        return orig_logits_f(*a, **k)
    def wrap_coop(*a, **k):
        calls["coop_try"] += 1
        r = orig_coop(*a, **k)
        if r:
            calls["coop_ok"] += 1
        return r
    def wrap_select(*a, **k):
        calls["select"] += 1
        return orig_select(*a, **k)

    rm._launch_qwen35_router_logits_bf16_hidden = wrap_logits_b
    rm._launch_qwen35_router_logits_f32_hidden = wrap_logits_f
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = wrap_coop
    rm.qwen35_router_select = wrap_select
    tok = 9707
    for _ in range(3):
        tok = int(s.step(tok, return_logits=False).token_id)
    rm._launch_qwen35_router_logits_bf16_hidden = orig_logits_b
    rm._launch_qwen35_router_logits_f32_hidden = orig_logits_f
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = orig_coop
    rm.qwen35_router_select = orig_select
    s.runtime.device_synchronize()
    print("=== router launch census (per decode step, 3 steps averaged) ===")
    for k, v in calls.items():
        print(f"  {k:12s} {v//3}")

    # ---- Phase 2: complete-wall router slice ----
    base = run_steps(s, noops=())
    no_router = run_steps(s, noops=(
        "_launch_qwen35_router_logits_bf16_hidden",
        "_launch_qwen35_router_logits_f32_hidden",
        "_try_launch_qwen35_router_topk_split_shared_bf16_f32w",
        "qwen35_router_select",
    ))
    print(f"baseline             ={base:.2f} ms/tok")
    print(f"no router slice      ={no_router:.2f}  drop {(base-no_router)*1000:+.1f} us/tok")
