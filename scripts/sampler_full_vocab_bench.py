#!/usr/bin/env python3
"""Fixed-input, model-free sampler A/B and cache-only profiler child.

Synthetic timings are selector measurements, never model-throughput claims.
Build once outside rocprof with the same HIPENGINE_COMPILER_VERSION_FILE, then
set HIPENGINE_REQUIRE_CACHED_BUILD=1 for the profiler child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.sampling.sampler import (
    build_sampler, fast_sampler_scratch_bytes, sample_sorted_f32_rows_i32,
    sample_top_p_temperature_f32_rows_i32,
)


class SamplerCase:
    """Explicitly owned GPU fixture, shared by focused tests and profiler."""
    def __init__(self, logits, temperatures, top_ps, min_ps, seeds, *, top_logprobs=8):
        self.runtime = get_hip_runtime()
        self.library = build_sampler()
        self.buffers = []
        self.logits = np.ascontiguousarray(logits, dtype=np.float32)
        self.rows, self.vocab = self.logits.shape
        self.top = top_logprobs
        self.inputs = [self.upload(self.logits)]
        for array, dtype in [(temperatures, np.float32), (top_ps, np.float32),
                             (min_ps, np.float32), (seeds, np.uint64)]:
            self.inputs.append(self.upload(np.asarray(array, dtype=dtype)))
        self.outputs = [self.alloc(self.rows * width * np.dtype(dtype).itemsize)
                        for width, dtype in [(1, np.int32), (1, np.float32),
                            (1, np.int32), (self.top, np.int32), (self.top, np.float32),
                            (1, np.int64), (1, np.float32)]]
        self.scratch = self.alloc(fast_sampler_scratch_bytes(self.rows, self.vocab))

    def alloc(self, nbytes):
        buf = malloc(max(8, nbytes), runtime=self.runtime)
        self.buffers.append(buf)
        return buf

    def upload(self, array):
        array = np.ascontiguousarray(array)
        buf = self.alloc(array.nbytes)
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=self.runtime)
        return buf

    def read(self, buf, dtype, shape):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(out), buf, out.nbytes, runtime=self.runtime)
        return out

    def launch(self, algorithm="sorted", *, step=13, stream=0):
        fn = sample_sorted_f32_rows_i32 if algorithm == "sorted" else sample_top_p_temperature_f32_rows_i32
        kwargs = {"scratch_ptr": self.scratch.ptr, "scratch_bytes": self.scratch.nbytes} if algorithm == "sorted" else {}
        fn(*(buf.ptr for buf in self.inputs), *(buf.ptr for buf in self.outputs[:3]),
           self.rows, self.vocab,
           out_top_indices_i32_ptr=self.outputs[3].ptr,
           out_top_logprobs_f32_ptr=self.outputs[4].ptr,
           out_indices_i64_ptr=self.outputs[5].ptr, out_values_f32_ptr=self.outputs[6].ptr,
           top_logprobs=self.top, step_index=step, stream=stream,
           library=self.library, runtime=self.runtime, **kwargs)

    def result(self):
        return tuple(self.read(buf, dtype, shape) for buf, dtype, shape in zip(
            self.outputs,
            (np.int32, np.float32, np.int32, np.int32, np.float32, np.int64, np.float32),
            ((self.rows,), (self.rows,), (self.rows,), (self.rows, self.top),
             (self.rows, self.top), (self.rows,), (self.rows,)), strict=True))

    def sorted_weights(self):
        """Debug-only full readback of scan scratch for numerical gates."""
        from hipengine.core.memory import DeviceBuffer
        view = DeviceBuffer(self.scratch.ptr + 16 * self.rows * self.vocab,
                            8 * self.rows * self.vocab)
        prefix = self.read(view, np.float64, (self.rows, self.vocab))
        previous = np.roll(prefix, 1, axis=1)
        previous[:, ::256] = 0
        return prefix - previous

    def close(self):
        for buf in reversed(self.buffers):
            free(buf, runtime=self.runtime)
        self.buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--algorithm", choices=("sorted", "strict", "both"), default="both")
    parser.add_argument("--shape", choices=("peaked", "broad", "uniform"), default="peaked")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    scale = {"peaked": 8.0, "broad": 1.0, "uniform": 0.0}[args.shape]
    logits = (np.random.default_rng(20260920).normal(size=(args.rows, args.vocab)) * scale).astype(np.float32)
    algorithms = ("strict", "sorted") if args.algorithm == "both" else (args.algorithm,)
    source = Path(__file__).resolve().parents[1] / "hipengine/kernels/hip_gfx1100/sampling/sampler.hip"
    report = {"host": platform.node(), "model": "synthetic FP32 logits (no model/quant)",
              "command": [sys.executable, *sys.argv],
              "environment": {key: os.environ.get(key) for key in (
                  "HIPENGINE_HIP_ARCH", "HIP_VISIBLE_DEVICES", "HIPENGINE_COMPILER_VERSION_FILE",
                  "HIPENGINE_REQUIRE_CACHED_BUILD")},
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "arithmetic": "sorted: T2 FP64 centered scores, FP32 exp, FP64 scan; strict: original serial selector",
              "correctness_gate": "tests/test_gpu_sampler_full_vocab.py + tests/test_gpu_sampler_kernel.py",
              "fixture_seed": 20260920, "shape": args.shape, "rows": args.rows,
              "vocab": args.vocab, "temperature": 0.7, "top_p": 0.95, "min_p": 0.0,
              "step": 13, "results": {}}
    with SamplerCase(logits, np.full(args.rows, .7), np.full(args.rows, .95),
                     np.zeros(args.rows), np.arange(args.rows, dtype=np.uint64) + 17) as case:
        report["library_path"] = str(case.library._name)
        for algorithm in algorithms:
            case.launch(algorithm)
            case.runtime.device_synchronize()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                case.launch(algorithm)
                case.runtime.device_synchronize()
                times.append((time.perf_counter_ns() - start) / 1e6)
            result = case.result()
            report["results"][algorithm] = {"median_ms": float(np.median(times)), "times_ms": times,
                "selected": result[0].tolist(), "logprobs": result[1].tolist(),
                "retained": result[2].tolist()}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
