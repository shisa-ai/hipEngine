"""Robustness check for the 512-row chunk override: 512 vs 1024 chunks at
4096 tokens on 35B-A3B, and at 2048 on the 27B dense (H5120 geometry).
"""
import sys, time, ctypes, statistics
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

def pref(model, linear, moe, L):
    cfg = PrefillConfig(linear_chunk_size=linear, moe_chunk_size=moe)
    with rm.Qwen35GGUFResidentSession(model,
            max_sequence_length=L, backend='hip_gfx1151', prefill_config=cfg) as s:
        toks = [9707]*L
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        t0 = time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        return (time.perf_counter() - t0) * 1e3

M35 = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
M27 = '/models/gguf/Qwen3.6-27B-Q4_K_M.gguf'

print("== 35B-A3B @ 4096 tokens ==", flush=True)
for i in range(2):
    d = pref(M35, 1024, 1024, 4096); g = pref(M35, 512, 512, 4096)
    print(f"iter{i}: 1024={d:.0f}ms 512={g:.0f}ms delta={(d-g):+.0f}ms", flush=True)

print("== 27B dense @ 2048 tokens ==", flush=True)
for i in range(2):
    d = pref(M27, 1024, 1024, 2048); g = pref(M27, 512, 512, 2048)
    print(f"iter{i}: 1024={d:.0f}ms 512={g:.0f}ms delta={(d-g):+.0f}ms", flush=True)
print("DONE")
