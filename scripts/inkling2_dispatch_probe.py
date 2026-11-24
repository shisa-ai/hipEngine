#!/usr/bin/env python3
"""Inkling 2 dispatch and bandwidth ceiling probe (gfx1100-class parts).

Measures three things on the current machine. Nothing here runs the model:

1. Dependent no-op kernel launch cost from Python (ctypes), for chains of
   48, 1,300 and 2,000 launches -- the per-visit and per-token launch budgets
   called out in the GGUF execution plan. Also measures the same chain
   submitted as one captured HIP graph.
2. hipEventRecord + hipEventSynchronize and hipStreamSynchronize cost on an
   otherwise idle stream.
3. Q8_0 GEMV bandwidth at decode shapes (rows=1) using the in-tree
   ``gguf_q8_0_gemv_f32_f32_out`` kernel: weight bytes read divided by
   steady-state time. This is a weight-streaming ceiling, not a model
   measurement.

Output is a JSON document on stdout (or --output PATH). Numbers are machine
and clock specific; record the host in the output, which this script does.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PROBE_SOURCE = r"""
#include <hip/hip_runtime.h>
__global__ void hipengine_probe_noop_kernel(float* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *value += 1.0f;
    }
}
extern "C" void hipengine_probe_noop(float* value, void* stream) {
    hipengine_probe_noop_kernel<<<1, 32, 0, static_cast<hipStream_t>(stream)>>>(value);
}
"""

_DEFAULT_N = (48, 1300, 2000)
_Q8_BLOCK_BYTES = 34  # 2-byte fp16 scale + 32 int8 values
_Q8_BLOCK_VALUES = 32


def _probe_library(directory: Path) -> Path:
    source = directory / "probe.hip"
    source.write_text(_PROBE_SOURCE)
    output = directory / "probe.so"
    arch = os.environ.get("HIPENGINE_PROBE_ARCH", "gfx1100")
    subprocess.check_call(
        ["hipcc", "-shared", "-fPIC", "-O2", f"--offload-arch={arch}", str(source), "-o", str(output)],
        stdout=subprocess.DEVNULL,
    )
    return output


def _q8_weight_nbytes(out_features: int, in_features: int) -> int:
    return out_features * in_features // _Q8_BLOCK_VALUES * _Q8_BLOCK_BYTES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(_DEFAULT_N),
                        help="dependent-chain lengths to measure")
    parser.add_argument("--chain-reps", type=int, default=5, help="repetitions per chain length (median reported)")
    parser.add_argument("--graph-reps", type=int, default=20, help="graph replay repetitions")
    parser.add_argument("--event-reps", type=int, default=1000, help="event/synchronize samples")
    parser.add_argument("--gemv-reps", type=int, default=50, help="GEMV iterations per shape")
    parser.add_argument("--gemv-warmup", type=int, default=30, help="GEMV warmup iterations per shape (GPU clock ramp)")
    parser.add_argument("--skip-gemv", action="store_true", help="skip the Q8_0 GEMV bandwidth probe")
    parser.add_argument("--output", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    results: dict[str, object] = {
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "machine": platform_info(),
    }

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc

    runtime = get_hip_runtime()
    with tempfile.TemporaryDirectory(prefix="inkling2-probe-") as tmp:
        lib_path = _probe_library(Path(tmp))
        import ctypes

        probe = ctypes.CDLL(str(lib_path))
        noop = probe.hipengine_probe_noop
        noop.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        noop.restype = None

        counter = np.zeros(1, dtype=np.float32)
        counter_buf = malloc(counter.nbytes)
        copy_host_to_device(counter_buf, host_array_ptr(counter), counter.nbytes)

        def run_chain(n: int, stream: int) -> float:
            start = time.perf_counter()
            for _ in range(n):
                noop(counter_buf.ptr, None if stream == 0 else ctypes.c_void_p(stream))
            runtime.stream_synchronize(stream)
            return time.perf_counter() - start

        for _ in range(3):
            run_chain(48, 0)
        chain = {}
        for n in args.sizes:
            samples = sorted(run_chain(n, 0) for _ in range(args.chain_reps))
            chain[n] = {
                "total_seconds": samples[len(samples) // 2],
                "per_launch_us": 1e6 * samples[len(samples) // 2] / n,
                "min_total_seconds": samples[0],
            }
        results["launch_chain_default_stream"] = chain

        graph_results = {}
        for n in args.sizes:
            stream = runtime.stream_create(nonblocking=True)
            try:
                runtime.stream_begin_capture(stream)
                for _ in range(n):
                    noop(counter_buf.ptr, ctypes.c_void_p(stream))
                graph = runtime.stream_end_capture(stream)
                instance = runtime.graph_instantiate(graph)
                start = time.perf_counter()
                for _ in range(args.graph_reps):
                    runtime.graph_launch(instance, stream)
                runtime.stream_synchronize(stream)
                replay = (time.perf_counter() - start) / args.graph_reps
                graph_results[n] = {
                    "graph_replay_seconds": replay,
                    "per_launch_us": 1e6 * replay / n,
                    "speedup_over_eager": chain[n]["total_seconds"] / replay,
                }
                runtime.graph_exec_destroy(instance)
                runtime.graph_destroy(graph)
            finally:
                runtime.stream_destroy(stream)
        results["graph_chain"] = graph_results

        event = ctypes.c_void_p()
        check(runtime, runtime.library.hipEventCreateWithFlags(ctypes.byref(event), ctypes.c_uint(0x2)))
        samples = []
        for _ in range(args.event_reps):
            start = time.perf_counter()
            runtime.library.hipEventRecord(event, ctypes.c_void_p(0))
            runtime.library.hipEventSynchronize(event)
            samples.append(time.perf_counter() - start)
        samples.sort()
        results["event_record_synchronize_seconds"] = {
            "median": samples[len(samples) // 2],
            "p10": samples[len(samples) // 10],
        }
        runtime.library.hipEventDestroy(event)

        samples = []
        for _ in range(args.event_reps):
            start = time.perf_counter()
            runtime.stream_synchronize(0)
            samples.append(time.perf_counter() - start)
        samples.sort()
        results["stream_synchronize_seconds"] = {
            "median": samples[len(samples) // 2],
            "p10": samples[len(samples) // 10],
        }
        free(counter_buf)

    if not args.skip_gemv:
        results["q8_0_gemv"] = q8_0_gemv_probe(args.gemv_reps, args.gemv_warmup)

    text = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def platform_info() -> dict[str, str]:
    info = {"system": sys.platform}
    try:
        info["machine"] = os.uname().machine
    except AttributeError:
        pass
    return info


def check(runtime, err):
    code = int(err) if err is not None else 0
    if code != 0:
        raise RuntimeError(f"hip error {code}")
    return



def q8_0_gemv_probe(reps: int, warmup: int = 30) -> dict[str, object]:
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import gguf_q8_0_gemv_f32_f32_out

    shapes = [(2304, 2304), (3072, 3072), (2304, 200704)]
    report = {}
    for in_features, out_features in shapes:
        nbytes = _q8_weight_nbytes(out_features, in_features)
        rng = np.random.default_rng(0)
        # Incompressible-enough random bytes stand in for real Q8_0 blocks.
        weights = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
        x = rng.normal(0, 1, size=in_features).astype(np.float32)
        out = np.empty(out_features, dtype=np.float32)
        w_buf = malloc(nbytes)
        x_buf = malloc(x.nbytes)
        o_buf = malloc(out.nbytes)
        try:
            copy_host_to_device(w_buf, host_array_ptr(weights), nbytes)
            copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
            # Warm up enough to ramp GPU clocks out of idle.
            for _ in range(warmup):
                gguf_q8_0_gemv_f32_f32_out(x_buf.ptr, w_buf.ptr, o_buf.ptr, 1, in_features, out_features)
            from hipengine.core.hip import get_hip_runtime

            runtime = get_hip_runtime()
            runtime.stream_synchronize(0)
            start = time.perf_counter()
            for _ in range(reps):
                gguf_q8_0_gemv_f32_f32_out(x_buf.ptr, w_buf.ptr, o_buf.ptr, 1, in_features, out_features)
            runtime.stream_synchronize(0)
            elapsed = time.perf_counter() - start
        finally:
            free(w_buf)
            free(x_buf)
            free(o_buf)
        seconds = elapsed / reps
        report[f"{in_features}x{out_features}"] = {
            "weight_bytes": nbytes,
            "seconds_per_call": seconds,
            "gigabytes_per_second": nbytes / seconds / 1e9,
        }
    return report


if __name__ == "__main__":
    sys.exit(main())
