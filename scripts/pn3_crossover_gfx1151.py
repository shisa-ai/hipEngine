"""Measure native vs aotriton full-attention prefill crossover on gfx1151.

Forces aotriton (threshold=1) vs native causal_gqa_gate_bf16 (threshold=2**30)
at prompt lengths 64/128/256/512/1024/2048 via the real bulk prefill, with
in-situ instrumentation of _run_full_attention_prefill_layer_aotriton
(device_sync before/after => serialized upper-bound per-layer wall, 10
full-attention layers). Reports whole-prefill wall and full-attn share.

Run as background task: 2 sessions x ~5 lengths x full prefill each.
"""
import sys, time, ctypes, json
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

LENGTHS = [64, 128, 256, 512, 1024, 2048]

def measure(threshold, lengths, out):
    _ACC = [0.0, 0]
    _orig = rm.Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton
    def wrapped(self, *a, **k):
        self.runtime.device_synchronize()
        t0 = time.perf_counter()
        try:
            return _orig(self, *a, **k)
        finally:
            self.runtime.device_synchronize()
            _ACC[0] += (time.perf_counter() - t0) * 1e6
            _ACC[1] += 1
    rm.Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton = wrapped
    try:
        cfg = PrefillConfig(attn_aotriton_min_tokens=threshold)
        with rm.Qwen35GGUFResidentSession(
                '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
                max_sequence_length=2048, backend='hip_gfx1151',
                prefill_config=cfg) as s:
            for L in lengths:
                _ACC[0] = 0.0; _ACC[1] = 0
                t0 = time.perf_counter()
                s.prefill([9707]*L, use_bulk=True, return_logits=False)
                s.runtime.device_synchronize()
                whole_ms = (time.perf_counter() - t0) * 1e3
                att_ms = _ACC[0]/1e3
                out.append({
                    "threshold": threshold, "length": L,
                    "whole_prefill_ms": round(whole_ms, 1),
                    "full_attn_serialized_ms": round(att_ms, 1),
                    "full_attn_per_layer_ms": round(att_ms/_ACC[1], 2) if _ACC[1] else None,
                    "full_attn_calls": _ACC[1],
                    "full_attn_pct": round(att_ms/whole_ms*100, 1),
                })
                print(json.dumps(out[-1]), flush=True)
    finally:
        rm.Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton = _orig

out = []
measure(1, LENGTHS, out)      # aotriton-forced
measure(2**30, LENGTHS, out)  # native-forced
with open('/tmp/pn3_crossover_gfx1151.json', 'w') as f:
    json.dump(out, f, indent=1)
print("DONE")
