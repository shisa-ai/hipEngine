"""TP1 control kernel-attribution child for rocprofv3.

Single-window mode with a deliberate 0.5 s GPU idle gap before the measured
window, so the window is the trailing kernel burst after the last long idle
interval in the device timeline. Single-GPU sessions profile cleanly on this
stack, so the replicated attention/GDN kernels and the full-width MLP GEMVs
of the matched TP1 control get per-kernel attribution here; comparing the
TP1 layer floor against the TP2 graphed schedule's ~350 us/layer bounds the
MLP-shard and exchange share without a two-GPU profile.

Usage (prebuild the caches once without the profiler, then)::

    rocprofv3 --kernel-trace --output-format csv -d OUT_DIR -o NAME -- \
        python scripts/tp1_control_rocprof_child.py MODEL.gguf [STEPS]
"""
import pathlib

import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

model = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16

session = MlpTP2GenerationSession(model, devices=(0,), mode="tp1", max_sequence_length=256)
runtime = session.runtime
rng = np.random.default_rng(11)
prompt = [int(i) for i in rng.integers(1000, 50000, size=64)]
# Seed the window from a fresh forward at the last prompt position, sampled
# from the returned logits row (no teacher-forced blocking readbacks).
result = session.generate(prompt, max_new_tokens=4, eos_token_id=None)
runtime.device_synchronize()
time.sleep(0.5)  # deliberate GPU idle gap before the measured window

step_logits, _trace = session._forward_token(prompt[-1], len(prompt) - 1, kind="prefill")
runtime.device_synchronize()
time.sleep(0.5)  # second idle gap: the window is the trailing burst
token = int(np.argmax(step_logits))
position = len(prompt)

t0 = time.perf_counter()
for offset in range(steps):
    logits, _trace = session._forward_token(token, position + offset, kind="prefill")
    token = int(np.argmax(logits))
runtime.device_synchronize()
wall = time.perf_counter() - t0
print(f"tp1-control-rocprof-child: {steps} steps, {wall * 1e3:.2f} ms total, "
      f"{wall / steps * 1e3:.3f} ms/step", flush=True)
session.close()
