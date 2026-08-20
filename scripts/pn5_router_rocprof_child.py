"""PN5 profiled child: short sync'd eager decode under rocprofv3 (cache-only).

Runs prefill + warmup + ~12 decode steps, then exits. External rocprofv3
captures kernel durations. Must NOT spawn hipcc: require_cached_build=True and
a pinned compiler-version file.
"""
import os, sys, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'

compiler = open(COMPILER).read().strip() or None
with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=True, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    tok = 9707
    for _ in range(3):  # warmup
        tok = int(s.step(tok, return_logits=False).token_id)
    for _ in range(12):  # profiled steps
        tok = int(s.step(tok, return_logits=False).token_id)
    s.runtime.device_synchronize()
print("done", flush=True)
