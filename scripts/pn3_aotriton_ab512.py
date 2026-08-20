"""Counter-rotated A/B: whole 512-token prefill, aotriton (default threshold)
vs native-forced, on gfx1151 35B-A3B. Confirms the native advantage is real
(not session/order noise) and quantifies the end-to-end prefill win.
"""
import sys, time, ctypes, statistics
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

TOKS = [9707]*512

def pref(threshold):
    cfg = PrefillConfig(attn_aotriton_min_tokens=threshold)
    with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
            max_sequence_length=1024, backend='hip_gfx1151', prefill_config=cfg) as s:
        s.prefill(TOKS, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()  # warm
        t0 = time.perf_counter()
        s.prefill(TOKS, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        return (time.perf_counter() - t0) * 1e3

res = {}
for label, thr in (("aotriton", 1), ("native", 2**30)):
    res[label] = []
for i in range(3):
    for label, thr in (("aotriton", 1), ("native", 2**30)):
        res[label].append(pref(thr))
    print(f"iter {i}: aotriton={res['aotriton'][-1]:.0f}ms native={res['native'][-1]:.0f}ms", flush=True)
for label in res:
    res[label] = statistics.median(res[label])
print(f"median aotriton={res['aotriton']:.0f}ms  native={res['native']:.0f}ms  "
      f"native saves {res['aotriton']-res['native']:.0f}ms ({(res['aotriton']/res['native']-1)*100:+.1f}%)")
