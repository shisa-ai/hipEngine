"""Isolate aotriton full-attention prefill wall in situ (35B-A3B, 512 tok).

Wraps _run_full_attention_prefill_layer_aotriton with device_sync before/after
to accumulate the serialized attention wall across the 10 full-attention
layers within the real bulk prefill. Compare against whole-prefill wall to get
the attention share of prefill. Serializing inflates the absolute number
(removes overlap with native layers) so it is an UPPER bound on the attention
share; the true GPU kernel time is bounded below by a rocprofv3 trace.
"""
import sys, time, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

_ACC = [0.0, 0]  # us, calls
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
    with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
            max_sequence_length=1024, backend='hip_gfx1151') as s:
        toks = [9707] * 512
        t0 = time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        total_ms = (time.perf_counter() - t0) * 1e3
        att_ms = _ACC[0] / 1e3
        print(f"whole prefill 512: {total_ms:.0f} ms")
        print(f"aotriton full-attn (10 layers, serialized): {att_ms:.0f} ms "
              f"({att_ms/_ACC[1]:.1f} ms/layer, {_ACC[1]} calls)")
        print(f"aotriton share of prefill (upper bound): {att_ms/total_ms*100:.0f}%")
        print(f"non-attention prefill (lower bound): {total_ms-att_ms:.0f} ms")
finally:
    rm.Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton = _orig
