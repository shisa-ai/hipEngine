#!/usr/bin/env python3
"""D8 screen: does hipMemcpyAsync from PAGEABLE host memory drain the queue?

The D8 attribution showed the six per-step staging uploads and the token-ids
upload are sync ``hipMemcpy`` calls on stream 0 -- mid-forward they drain the
already-enqueued kernels, which is why KB-sized copies cost ~91 us of host
time. Wiring them async only removes the drain if the async call does not
itself sync the stream for pageable sources (CUDA documents that pageable
async performs a stream sync first; ROCm's actual behavior is what matters).

Method: enqueue seconds-scale device work on stream 0, then time three H2D
uploads of a KB-scale pageable/pinned buffer:
  A  sync hipMemcpy            (expected: includes the drain)
  B  hipMemcpyAsync, pageable  (the wiring candidate -- must NOT drain)
  C  hipMemcpyAsync, pinned    (registered once; the safe fallback)
For each, also verify the bytes landed (write known pattern, drain, read back).

Run: .venv/bin/python scratch/d8_async_h2d_screen.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    MemcpyKind,
    malloc,
    free,
    host_array_ptr,
)

N = 4096  # bytes per upload (pageable and pinned copies are both KB-scale)


def fill_queue(runtime, stream=0, mb=1024):
    """Enqueue ~mb MB of memset work so the queue is deep."""
    buf = runtime.malloc(mb * 1024 * 1024)
    for _ in range(4):
        runtime.memset_async(buf, 0, mb * 1024 * 1024, stream)
    return buf


def main() -> int:
    rt = get_hip_runtime()
    dev = rt.malloc(N)
    src = np.arange(N, dtype=np.uint8)
    readback = np.empty(N, dtype=np.uint8)

    # Warm every symbol once.
    rt.memcpy(dev, host_array_ptr(src), N, MemcpyKind.HOST_TO_DEVICE)
    rt.memcpy_async(dev, host_array_ptr(src), N, MemcpyKind.HOST_TO_DEVICE, 0)
    rt.stream_synchronize(0)
    fill_work = fill_queue(rt)

    out = {}

    # A: sync copy with a deep queue -- the current production path.
    work_a = fill_queue(rt)
    t0 = time.perf_counter_ns()
    rt.memcpy(dev, host_array_ptr(src), N, MemcpyKind.HOST_TO_DEVICE)
    out["A_sync_pageable_host_ns"] = time.perf_counter_ns() - t0
    rt.stream_synchronize(0)

    # B: async copy, PAGEABLE source, deep queue -- timing is the enqueue.
    work_b = fill_queue(rt)
    t0 = time.perf_counter_ns()
    rt.memcpy_async(dev, host_array_ptr(src), N, MemcpyKind.HOST_TO_DEVICE, 0)
    out["B_async_pageable_host_ns"] = time.perf_counter_ns() - t0
    rt.stream_synchronize(0)
    rt.memcpy(host_array_ptr(readback), dev, N, MemcpyKind.DEVICE_TO_HOST)
    out["B_bytes_ok"] = bool(np.array_equal(readback, src))

    # C: async copy, PINNED source, deep queue.
    pinned = np.empty(N, dtype=np.uint8)
    rt.host_register(host_array_ptr(pinned), pinned.nbytes, flags=1)
    pinned[:] = src
    work_c = fill_queue(rt)
    t0 = time.perf_counter_ns()
    rt.memcpy_async(dev, host_array_ptr(pinned), N, MemcpyKind.HOST_TO_DEVICE, 0)
    out["C_async_pinned_host_ns"] = time.perf_counter_ns() - t0
    rt.stream_synchronize(0)
    rt.memcpy(host_array_ptr(readback), dev, N, MemcpyKind.DEVICE_TO_HOST)
    out["C_bytes_ok"] = bool(np.array_equal(readback, src))

    # Reference: the same async pageable call with an EMPTY queue.
    rt.stream_synchronize(0)
    t0 = time.perf_counter_ns()
    rt.memcpy_async(dev, host_array_ptr(src), N, MemcpyKind.HOST_TO_DEVICE, 0)
    out["D_async_pageable_empty_queue_ns"] = time.perf_counter_ns() - t0
    rt.stream_synchronize(0)

    for k, v in out.items():
        if k.endswith("_ns"):
            print(f"{k}: {v / 1e3:.1f} us")
        else:
            print(f"{k}: {v}")
    for w in (fill_work, work_a, work_b, work_c):
        rt.free(w)
    rt.free(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())