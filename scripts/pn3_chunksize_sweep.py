"""Chunk-size sweep for gfx1151 GGUF prefill (linear/moe) at 2048 tokens, plus
a correctness check (logit KL + top-1) between chunk policies to confirm the
chunk boundary change is correctness-neutral within the prefill noise floor.
"""
import sys, time, ctypes, statistics, numpy as np, math
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

def pref(linear, moe, L, ret_logits=False):
    cfg = PrefillConfig(linear_chunk_size=linear, moe_chunk_size=moe)
    with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
            max_sequence_length=L, backend='hip_gfx1151', prefill_config=cfg) as s:
        toks = [9707]*L
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        t0 = time.perf_counter()
        r = s.prefill(toks, use_bulk=True, return_logits=ret_logits)
        s.runtime.device_synchronize()
        w = (time.perf_counter() - t0) * 1e3
        if ret_logits:
            return w, np.asarray(r.logits if hasattr(r,'logits') else r, dtype=np.float32).reshape(-1)
        return w, None

L = 2048
res = {}
for sz in (128, 256, 512, 1024):
    times = [pref(sz, sz, L)[0] for _ in range(2)]
    res[sz] = statistics.median(times)
    print(f"chunk {sz:4d}: {res[sz]:.0f}ms", flush=True)

best = min(res, key=res.get)
print(f"BEST chunk={best} ({res[best]:.0f}ms) vs 1024 ({res[1024]:.0f}ms): {(res[1024]/res[best]-1)*100:+.1f}%")

# correctness: chunk 256 vs 1024 logits at 2048 (within run-noise floor?)
_, l256 = pref(256, 256, L, ret_logits=True)
_, l1024 = pref(1024, 1024, L, ret_logits=True)
def softmax(x):
    z = x - x.max(); e = np.exp(z); return e/e.sum()
def kl(p,q): return float((p*(np.log(p+1e-12)-np.log(q+1e-12))).sum())
print(f"KL(l1024||l256)={kl(softmax(l1024),softmax(l256)):.5f} top1 {np.argmax(l1024)} vs {np.argmax(l256)} (floor ~0.034)")
