"""gfx1151 chunk-profile wiring A/B: default linear/moe 1024 chunks vs the
intended gfx1151 profile (256) on a >1K prefill. The gfx1151
_ARCH_CHUNK_PROFILES (linear=256, moe=256) exists but the GGUF runner never
passes target_arch to resolve_prefill_config_for_sequence, so it is not
applied on the GGUF path. Counter-rotated whole-prefill measurement at 2048
and 4096 tokens.
"""
import sys, time, ctypes, statistics
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

def pref(linear, moe, L):
    cfg = PrefillConfig(linear_chunk_size=linear, moe_chunk_size=moe)
    with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
            max_sequence_length=L, backend='hip_gfx1151', prefill_config=cfg) as s:
        toks = [9707]*L
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()  # warm
        t0 = time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        return (time.perf_counter() - t0) * 1e3

for L in (2048,):
    res = {"default": [], "gfx1151_profile": []}
    for i in range(2):
        res["default"].append(pref(0, 0, L))              # auto -> 1024/4096
        res["gfx1151_profile"].append(pref(256, 256, L))  # intended gfx1151
        print(f"L={L} iter{i}: default={res['default'][-1]:.0f}ms gfx1151(256)={res['gfx1151_profile'][-1]:.0f}ms", flush=True)
    d = statistics.median(res["default"]); g = statistics.median(res["gfx1151_profile"])
    print(f"L={L}: default={d:.0f}ms gfx1151-profile={g:.0f}ms delta={d-g:+.0f}ms ({(d/g-1)*100:+.1f}%)")
