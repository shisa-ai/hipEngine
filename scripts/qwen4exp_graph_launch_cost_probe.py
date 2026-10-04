"""Measure the device-side wall cost of one HIP graph launch.

Tests the hypothesis in worklog 20261004T104131: the decode step issues one graph
launch per layer (48 per step), and if each carries a fixed device-side gap it
reproduces DE-11's corrected ~10.1 ms/step remainder at p512.

Three arms, each launched N times back-to-back with ONE sync at the end:
  A. empty graph            -> pure per-launch overhead
  B. one tiny kernel graph  -> overhead + one kernel
  C. the same kernel eager  -> the kernel's own cost, no graph
"""

from __future__ import annotations

import ctypes
import statistics
import sys
import time

sys.path.insert(0, "/home/lhl/hipEngine")

from hipengine.core.hip import HipRuntime  # noqa: E402

N = 2000
REPS = 5


def main() -> int:
    rt = HipRuntime.load()
    stream = rt.stream_create(nonblocking=True)

    # --- a minimal device buffer and an empty-ish kernel (memset) -----------
    nbytes = 4096
    buf = rt.malloc(nbytes)
    try:
        # A. empty graph: capture begin/end with nothing in between
        rt.stream_begin_capture(stream, 2)
        graph = rt.stream_end_capture(stream)
        exec_empty = rt.graph_instantiate(graph)

        # B. one-op graph: a single device memset captured into a graph
        rt.stream_begin_capture(stream, 2)
        rt.memset_async(buf, 0, nbytes, stream)
        graph1 = rt.stream_end_capture(stream)
        exec_one = rt.graph_instantiate(graph1)

        def time_loop(fn, n: int) -> float:
            rt.stream_synchronize(stream)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            rt.stream_synchronize(stream)
            return (time.perf_counter() - t0) * 1e3 / n

        # warmup
        for _ in range(200):
            rt.graph_launch(exec_empty, stream)
            rt.graph_launch(exec_one, stream)
            rt.memset_async(buf, 0, nbytes, stream)
        rt.stream_synchronize(stream)

        a = [time_loop(lambda: rt.graph_launch(exec_empty, stream), N) for _ in range(REPS)]
        b = [time_loop(lambda: rt.graph_launch(exec_one, stream), N) for _ in range(REPS)]
        c = [time_loop(lambda: rt.memset_async(buf, 0, nbytes, stream), N) for _ in range(REPS)]

        def show(label: str, v: list[float]) -> float:
            m = statistics.median(v)
            print(f"  {label:34s} median {m:8.4f} ms   runs {[round(x, 4) for x in v]}")
            return m

        print(f"graph-launch cost, N={N} per run, {REPS} runs, one sync per run")
        print(f"  stream = nonblocking, buffer {nbytes} B")
        m_a = show("A. empty graph launch", a)
        m_b = show("B. one-memset graph launch", b)
        m_c = show("C. one-memset eager (no graph)", c)

        print()
        print(f"  pure per-launch overhead (A)          {m_a:8.4f} ms")
        print(f"  graph wrapper cost over eager (B - C) {m_b - m_c:8.4f} ms")
        print()
        print("  IMPLIED decode-step cost at 48 launches per step:")
        print(f"    48 x A = {48 * m_a:8.3f} ms/step")
        print(f"    48 x (B - C) = {48 * (m_b - m_c):8.3f} ms/step")
        print()
        print("  reference: DE-11 corrected remainder ~10.1 ms/step at p512,")
        print("             ~16.7-19.2 ms/step at p4096; wall 47.7 / 55.2-57.7.")
        rt.graph_exec_destroy(exec_empty)
        rt.graph_destroy(graph)
        rt.graph_exec_destroy(exec_one)
        rt.graph_destroy(graph1)
    finally:
        rt.free(buf)
        rt.stream_destroy(stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
