"""Measure the MT-1 rows<=8 body-pass cost against serial c1 decode.

MT-1 needs "a body pass whose cost grows sub-linearly with rows while keeping
c1 arithmetic". The arithmetic half is asserted by
``tests/test_gpu_gguf_k_gemv.py::test_q8_0_coltile8_rowbatch4_is_width_invariant_to_serial_c1``;
this measures the cost half on the same kernel pair.

Both arms are launch sequences (the serial arm issues one launch per row), so the
ratio includes launch overhead - which is what a serial verifier genuinely pays,
but it is not a pure throughput ratio.

Run::

    python3 scripts/qwen4exp_mt1_rowbatch_cost.py --output <path.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv, gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    gguf_q8_0_pack8_gemv_f32_f32_out,
)
from test_gpu_gguf_k_gemv import make_q8_0_weight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--in-features", type=int, default=2560)
parser.add_argument("--out-features", type=int, default=10240)
parser.add_argument("--reps", type=int, default=30)
parser.add_argument("--warmups", type=int, default=5)
args = parser.parse_args()

rt = get_hip_runtime(); lib = build_gguf_k_gemv(load=True)
in_features, out_features = int(args.in_features), int(args.out_features)
weight = np.ascontiguousarray(make_q8_0_weight(out_features, in_features))
rng = np.random.default_rng(20260929)
x = np.ascontiguousarray(rng.normal(0, 0.2, size=(8, in_features)).astype(np.float32))
bufs = []
w_dev = malloc(weight.nbytes, runtime=rt); copy_host_to_device(w_dev, host_array_ptr(weight), runtime=rt); bufs.append(w_dev)
x_dev = malloc(x.nbytes, runtime=rt); copy_host_to_device(x_dev, host_array_ptr(x), runtime=rt); bufs.append(x_dev)
out8 = malloc(8*out_features*4, runtime=rt); bufs.append(out8)
out1 = malloc(out_features*4, runtime=rt); bufs.append(out1)

def timeit(fn, reps=None, warm=None):
    reps = int(args.reps) if reps is None else reps
    warm = int(args.warmups) if warm is None else warm
    for _ in range(warm): fn()
    rt.device_synchronize()
    s = []
    for _ in range(reps):
        rt.device_synchronize(); t0=time.perf_counter_ns(); fn(); rt.device_synchronize()
        s.append((time.perf_counter_ns()-t0)/1e3)
    return statistics.median(s)

rows_out = []
for rows in (1,2,4,8):
    batched = timeit(lambda r=rows: gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
        x_dev.ptr, w_dev.ptr, out8.ptr, r, in_features, out_features, library=lib, runtime=rt))
    serial = timeit(lambda r=rows: [gguf_q8_0_pack8_gemv_f32_f32_out(
        x_dev.ptr + i*in_features*4, w_dev.ptr, out1.ptr, 1, in_features, out_features,
        library=lib, runtime=rt) for i in range(r)])
    rows_out.append({"rows": rows, "batched_us": batched, "serial_c1_us": serial,
                     "speedup": serial/batched, "per_row_batched_us": batched/rows,
                     "per_row_serial_us": serial/rows})
    print(f"rows={rows}  batched={batched:9.1f} us  serial_c1={serial:9.1f} us  speedup={serial/batched:5.2f}x  "
          f"per-row batched={batched/rows:8.1f} serial={serial/rows:8.1f}")
art = {"schema":1, "kind":"qwen4exp_mt1_rowbatch_cost", "performance_claim": True,
       "created_at": datetime.now(timezone.utc).isoformat(),
       "boundary":"q8_0 GEMV kernel pair only (DE-3 linear family); excludes MoE DE-1, attention, and end-to-end verifier",
       "shape":{"in_features":in_features,"out_features":out_features}, "reps":int(args.reps), "warmups":int(args.warmups), "rows":rows_out}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(art, indent=2) + "\n")
for b in reversed(bufs): free(b, runtime=rt)
