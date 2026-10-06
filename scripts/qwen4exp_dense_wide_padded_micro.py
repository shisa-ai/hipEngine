#!/usr/bin/env python3
"""Counterbalanced padded-tail GEMM screen; synthetic inputs, real GGUF weights.

Only cached direct leaves run. HIP events time a complete GEMM after warmup;
activation preparation and downloads are outside timing. This ranks the kernel
sibling, not model throughput or production qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from scripts.qwen4exp_row4_state_gate import _git_metadata, _host_metadata


def load_weight(root: Path, name: str) -> tuple[np.ndarray, tuple[int, int]]:
    for file in discover_gguf_files(root):
        index = load_gguf_index(file)
        for tensor in index.tensors:
            if tensor.name == name:
                if tensor.ggml_type != 8 or len(tensor.shape) != 2:
                    raise ValueError("requires a Q8_0 matrix")
                with file.open("rb") as source:
                    source.seek(index.tensor_data_offset + tensor.offset)
                    raw = np.frombuffer(source.read(tensor.nbytes), np.uint8).copy()
                if raw.nbytes != tensor.nbytes:
                    raise ValueError("truncated Q8_0 matrix")
                # GGUFTensorInfo.shape is row-major (outputs, inputs), not the
                # GGML storage dimension order. Return the launch ABI (K, N).
                n, k = tensor.shape
                return raw, (int(k), int(n))
    raise ValueError(f"weight not found: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--weight", default="blk.16.hc_attn_down.weight")
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--f32in", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.repeats < 2 or args.batch < 1:
        parser.error("positive shapes/batches and >=2 pairs required")
    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True, require_cached=True)
    weight, (k, n) = load_weight(args.model_root, args.weight)
    suffix = "f32_f32_out" if args.f32in else "f16in_f32_f32_out"
    variants = [f"dense_wide256_{suffix}", f"dense_wide256_padded_{suffix}"]
    start, stop = runtime.event_create(), runtime.event_create()
    report = dict(source=_git_metadata(ROOT), host=_host_metadata(), command=sys.argv,
                  model="Qwen3.8-Flash-Next", quant="UD-Q4_K_XL",
                  model_root=str(args.model_root), weight=args.weight,
                  weight_sha256=hashlib.sha256(weight).hexdigest(), variants=variants,
                  timing="HIP event per complete GEMM;1 warm pair then counterbalanced batched pairs",
                  performance_claim=False, status="running", cases=[])
    try:
        for rows in args.rows:
            rng = np.random.default_rng(70931)
            x = rng.normal(0, 0.1, (rows, k)).astype(
                np.float32 if args.f32in else np.float16)
            out = np.empty((rows, n), np.float32)
            buffers = [malloc(a.nbytes, runtime=runtime) for a in (x, weight, out)]
            xd, wd, od = buffers
            try:
                copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
                copy_host_to_device(wd, host_array_ptr(weight), runtime=runtime)
                hashes = [set(), set()]
                samples = [[], []]
                for rep in range(args.repeats + 1):
                    for arm in ((0, 1) if rep % 2 == 0 else (1, 0)):
                        fn = dw._WRAPPERS[variants[arm]]
                        runtime.event_record(start)
                        for _ in range(args.batch):
                            fn(xd.ptr, wd.ptr, od.ptr, rows, k, n,
                               library=library, runtime=runtime)
                        runtime.event_record(stop)
                        runtime.event_synchronize(stop)
                        if rep:
                            samples[arm].append(runtime.event_elapsed_time_ms(start, stop) / args.batch)
                        copy_device_to_host(host_array_ptr(out), od, runtime=runtime)
                        if not np.isfinite(out).all():
                            raise AssertionError("nonfinite output")
                        hashes[arm].add(hashlib.sha256(out).hexdigest())
                exact = len(hashes[0] | hashes[1]) == 1
                if not exact:
                    raise AssertionError("output parity/repeat failure; numerical review required")
                medians = [statistics.median(s) for s in samples]
                record = dict(rows=rows, in_features=k, out_features=n,
                              samples_ms=samples, medians_ms=medians,
                              speedup=medians[0] / medians[1], output_exact=exact,
                              output_sha256=next(iter(hashes[0])))
                report["cases"].append(record)
                print(json.dumps(record), flush=True)
            finally:
                for dev in buffers:
                    free(dev, runtime=runtime)
        report["status"] = "passed"
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
