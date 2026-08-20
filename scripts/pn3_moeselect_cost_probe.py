"""PN3-MOESELECT probe: isolate the selected-expert MoE GEMV complete-wall cost.

No-ops the selected-expert launch helpers (the moe_linear registry path, which
is separate from launch_gguf_linear) and the shared-expert path to measure the
real post-PN6 complete-wall contribution of the selected-expert GEMV slice.
Wall-timing only (outputs are WRONG after each no-op).
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
import hipengine.runtime.gguf_linear as gl
ctypes.CDLL('libamdhip64.so')

def noop(*a, **k):
    return None

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

SEL = ("_launch_selected_raw_gguf_moe_pair_silu",
       "_launch_selected_raw_gguf_moe_pair",
       "_launch_selected_raw_gguf_moe_linear")

with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=900, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    base = run_steps(s)
    no_sel = run_steps(s, noops=SEL)
    no_sel_shared = run_steps(s, noops=SEL + ("launch_gguf_linear_pair", "launch_gguf_linear_pair_concat", "_try_launch_shared_gate_up_from_f32_post_norm"))
    print(f"baseline                 ={base:.2f} ms/tok")
    print(f"no selected-expert gemv  ={no_sel:.2f}  drop {(base-no_sel)*1000:+.1f} us/tok")
    print(f"no selected + shared     ={no_sel_shared:.2f}  drop {(base-no_sel_shared)*1000:+.1f} us/tok")
