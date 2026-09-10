"""Decode kernel-attribution driver for rocprofv3 kernel traces.

Single-window modes with a deliberate 0.5 s GPU idle gap before the measured
window, so the window is the trailing kernel burst after the last long idle
interval in the device timeline - no marker trace is needed to slice.

Usage (prebuild the JIT cache once without the profiler, then)::

    rocprofv3 --kernel-trace --output-format csv -d OUT_DIR -o NAME -- \
        python scripts/gguf_decode_graph_rocprof_driver.py MODEL.gguf {graph,eager}

mode=graph captures the production decode graph and replays 32 measured
steps (the published decode protocol); mode=eager runs 32 eager steps.
Per-kernel pure durations come from the trailing window of the kernel
trace; see benchmarks/results/2026-09-11-decode-graph-attribution/ for the
derived four-arm attribution.

Decode kernel-profile driver, single-window modes.

mode=graph: prefill, warm, capture, warm-replay, [0.5s GPU idle gap],
32-step measured graph replay, exit.
mode=eager: prefill, warm, [0.5s GPU idle gap], 32 eager steps, exit.

The gap makes the measured window the trailing kernel burst after the last
long GPU idle interval, so no markers are needed to slice the trace.
"""
import sys, time, json
sys.path.insert(0, '/home/lhl/hipEngine-ud')
import numpy as np

model = sys.argv[1]
mode = sys.argv[2]
assert mode in ('graph', 'eager')
STEPS = 32

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

with Qwen35GGUFResidentSession(model,
        compiler_version=open('/tmp/ud-hipcc-version.txt').read().strip(),
        require_cached_build=False, max_sequence_length=1024,
        use_wmma_prefill=True, use_gemv_decode=True) as s:
    rng = np.random.default_rng(7)
    ids = list(rng.integers(1000, 50000, 512))
    cur = s.prefill(ids, use_bulk=True, bulk_attention_mode='bulk')
    for _ in range(4):
        cur = s.step(int(cur.token_id))
    graph = None
    if mode == 'graph':
        graph = s.capture_decode_graph(position=s.position,
                                        steps_per_replay=1,
                                        max_replay_steps=STEPS + 8,
                                        record_steps=0)
        graph.replay(8)
    s.runner.runtime.stream_synchronize(0)
    time.sleep(0.5)          # deliberate GPU idle gap before the window
    t0 = time.perf_counter()
    if mode == 'graph':
        graph.replay(STEPS)
        s.runner.runtime.stream_synchronize(0)
        tok = None
    else:
        tok = int(cur.token_id)
        for _ in range(STEPS):
            cur = s.step(tok, return_logits=False)
            tok = int(cur.token_id)
        s.runner.runtime.stream_synchronize(0)
    wall = time.perf_counter() - t0
    if graph is not None:
        graph.close()
print(json.dumps({"model": model.split('/')[-1], "mode": mode,
                  "steps": STEPS, "wall_s": wall,
                  "tok_s": STEPS / wall}, indent=2))
