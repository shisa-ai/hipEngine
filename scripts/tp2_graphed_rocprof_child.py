"""TP2 graphed-schedule kernel-attribution child for rocprofv3.

Single-window mode with a deliberate 0.5 s GPU idle gap before the measured
window, so the window is the trailing kernel burst after the last long idle
interval in the device timeline - no marker trace needed to slice.

Usage (prebuild the caches once without the profiler, then)::

    rocprofv3 --kernel-trace --output-format csv -d OUT_DIR -o NAME -- \
        python scripts/tp2_graphed_rocprof_child.py MODEL.gguf [STEPS]

Builds the TP2 graphed session, runs one prompt, then STEPS measured decode
steps through the captured per-layer graphs (the production default path).
The wall print is the child's own timing; the trace's trailing window is the
per-kernel attribution.
"""
import sys
import time

sys.path.insert(0, "/home/lhl/hipEngine-main")
import numpy as np

from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

model = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24

session = MlpTP2GenerationSession(
    model,
    devices=(0, 1),
    mode="tp2",
    max_sequence_length=256,
    schedule="graphed",
)
rng = np.random.default_rng(11)
prompt = [int(i) for i in rng.integers(1000, 50000, size=64)]
result = session.generate(prompt, max_new_tokens=4, eos_token_id=None)
runtime = session.runtime
runtime.device_synchronize()
time.sleep(0.5)  # deliberate GPU idle gap before the measured window

# Seed the window from the generation's own tokens (no host readbacks under
# the profiler: a teacher-forced seed's blocking full-vocab reads dominate a
# profiled run).
token = int(result.token_ids[-1])
position = len(prompt) + 3
t0 = time.perf_counter()
for offset in range(steps):
    logits, _trace = session._forward_token(token, position + offset, kind="prefill")
    token = int(np.argmax(logits))
runtime.device_synchronize()
wall = time.perf_counter() - t0
print(f"tp2-rocprof-child: {steps} steps, {wall * 1e3:.2f} ms total, "
      f"{wall / steps * 1e3:.3f} ms/step", flush=True)
session.close()
