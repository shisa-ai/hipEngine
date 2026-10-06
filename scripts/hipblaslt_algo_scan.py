#!/usr/bin/env python3
"""Time every zero-workspace hipBLASLt algorithm for a shape and report the spread.

The runtimes pick an algorithm by heuristic index (`yue2_ar` takes the first
zero-workspace candidate, the Laguna routes take a configured index), which is a
guess about this host. This measures the alternatives on the real device so the
choice can be made from data instead of from the heuristic's order.

Usage:
    python3 scripts/hipblaslt_algo_scan.py --rows 1299
    python3 scripts/hipblaslt_algo_scan.py --shape 1299,2048,6144 --repeat 12
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The YuE2 3B production shapes: (rows, in_features, out_features).
DEFAULT_SHAPES = ((1299, 2048, 6144), (1299, 6144, 2048), (1299, 2048, 2048), (1299, 2048, 1024))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", default=None,
                        help="rows,in,out (repeatable); defaults to the YuE2 NAR set")
    parser.add_argument("--rows", type=int, default=None,
                        help="override the row count of every default shape")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        raise SystemExit("ROCm/HIP runtime is not available")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.hipblaslt import HIP_R_16F, HipblasLt
    from hipengine.core.memory import free, malloc

    runtime = get_hip_runtime()
    library = HipblasLt()

    shapes = DEFAULT_SHAPES
    if args.shape:
        shapes = tuple(tuple(int(v) for v in spec.split(",")) for spec in args.shape)
    if args.rows is not None:
        shapes = tuple((args.rows, i, o) for _, i, o in shapes)

    for rows, inputs, outputs in shapes:
        problem = library.problem(rows, inputs, outputs, 0, ab_type=HIP_R_16F)
        candidates = [a for a in problem.algorithms(16) if a.workspace_size == 0]
        a_buf = malloc(rows * inputs * 2)
        b_buf = malloc(inputs * outputs * 2)
        c_buf = malloc(rows * outputs * 4)
        flops = 2.0 * rows * inputs * outputs
        print(f"shape rows={rows} in={inputs} out={outputs}: "
              f"{len(candidates)} zero-workspace algorithms")
        timings = []
        for index, algorithm in enumerate(candidates):
            for _ in range(args.warmup):
                problem.launch(algorithm, a_buf.ptr, b_buf.ptr, c_buf.ptr, stream=0)
            runtime.device_synchronize()
            samples = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                problem.launch(algorithm, a_buf.ptr, b_buf.ptr, c_buf.ptr, stream=0)
                runtime.device_synchronize()
                samples.append(time.perf_counter() - started)
            median = statistics.median(samples)
            timings.append((median, index))
            print(f"  [{index:2d}] {median * 1e3:8.3f} ms  {flops / median / 1e12:6.2f} TFLOP/s")
        timings.sort()
        best, best_index = timings[0]
        first = next(t for t, i in timings if i == 0)
        print(f"  best index {best_index} at {best * 1e3:.3f} ms; "
              f"index 0 at {first * 1e3:.3f} ms -> {first / best:.2f}x")
        for buffer in (a_buf, b_buf, c_buf):
            free(buffer)
    library.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
