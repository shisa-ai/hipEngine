#!/usr/bin/env python3
"""Counterbalance norm+cast+HC-down versus dual norm publication+HC-down.

Synthetic BF16 residuals, current real Q8_0 HC-down weights, parent F32 norm
and the current padded F16-input GEMM in both arms. No runtime admission change.
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
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc, memory_stats
from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_gr as gr
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
from hipengine.loading.materialize import float_array_to_bf16_bits
from scripts.qwen4exp_dense_wide_padded_micro import load_weight
from scripts.qwen4exp_row4_state_gate import _git_metadata, _host_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from tests.test_gpu_qwen4_exp_gr_hip import _download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    if min(args.rows) <= 0 or args.repeats < 2 or args.batch < 1:
        parser.error("positive rows/batch and >=2 repeats required")
    runtime = get_hip_runtime()
    norm_lib = gr.build_qwen4_exp_gr(load=True)
    wide_lib = wide.build_gguf_q8_0_dense_wide(load=True, require_cached=True)
    weight, (k, n) = load_weight(args.model_root, "blk.16.hc_attn_down.weight")
    assert (k, n) == (10240, 320)
    start, stop = runtime.event_create(), runtime.event_create()
    report = dict(source=_git_metadata(ROOT), host=_host_metadata(), command=sys.argv,
                  model_identity=model_identity(args.model_root), model="Qwen3.8-Flash-Next",
                  quant="UD-Q4_K_XL", cases=[], performance_claim=False, status="running",
                  weight_sha256=hashlib.sha256(weight).hexdigest(),
                  boundary="HIP events:complete norm+cast or dual norm;and same padded HC-down GEMM")
    try:
        for rows in args.rows:
            rng = np.random.default_rng(18031)
            x = float_array_to_bf16_bits(rng.normal(0, .3, (rows, 4, 2560)).astype(np.float32))
            gamma = rng.normal(1, .05, (4, 2560)).astype(np.float32)
            buffers = [malloc(a.nbytes, runtime=runtime) for a in (x, gamma, weight)]
            xd, gd, wd = buffers
            nd = malloc(rows*k*4, runtime=runtime)
            hd = malloc(rows*k*2, runtime=runtime)
            od = malloc(rows*n*4, runtime=runtime)
            buffers.extend([nd, hd, od])
            try:
                for dev, host in zip(buffers, (x, gamma, weight)):
                    copy_host_to_device(dev, host_array_ptr(host), runtime=runtime)
                def operation(arm, gemm):
                    if arm:
                        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
                            xd.ptr, gd.ptr, nd.ptr, hd.ptr, rows, 4, 2560, library=norm_lib, runtime=runtime)
                    else:
                        gr.qwen4_exp_grouped_rmsnorm_bf16_f32(
                            xd.ptr, gd.ptr, nd.ptr, rows, 4, 2560, library=norm_lib, runtime=runtime)
                        wide.f32_to_f16(nd.ptr, hd.ptr, rows*k, library=wide_lib, runtime=runtime)
                    if gemm:
                        wide._WRAPPERS["dense_wide256_padded_f16in_f32_f32_out"](
                            hd.ptr, wd.ptr, od.ptr, rows, k, n, library=wide_lib, runtime=runtime)
                record = dict(rows=rows, k=k, n=n, boundaries={})
                for gemm in (False, True):
                    samples, hashes = [[], []], [set(), set()]
                    for rep in range(args.repeats + 1):
                        for arm in ((0, 1) if rep % 2 == 0 else (1, 0)):
                            runtime.event_record(start)
                            for _ in range(args.batch):
                                operation(arm, gemm)
                            runtime.event_record(stop)
                            runtime.event_synchronize(stop)
                            if rep:
                                samples[arm].append(runtime.event_elapsed_time_ms(start, stop) / args.batch)
                            parts = [_download(nd, (rows, k), np.float32, runtime),
                                     _download(hd, (rows, k), np.float16, runtime)]
                            if gemm:
                                parts.append(_download(od, (rows, n), np.float32, runtime))
                            if not all(np.isfinite(part).all() for part in parts):
                                raise AssertionError("nonfinite synthetic operation")
                            hashes[arm].add(hashlib.sha256(b"".join(p.tobytes() for p in parts)).hexdigest())
                    if len(hashes[0] | hashes[1]) != 1:
                        raise AssertionError("operation parity/repeat mismatch;numerical review required")
                    medians = [statistics.median(s) for s in samples]
                    record["boundaries"]["norm_gemm" if gemm else "norm"] = dict(
                        samples_ms=samples, medians_ms=medians, speedup=medians[0]/medians[1],
                        output_exact=True, output_sha256=next(iter(hashes[0])))
                report["cases"].append(record)
                print(json.dumps(record), flush=True)
            finally:
                for dev in reversed(buffers):
                    free(dev, runtime=runtime)
        report["status"] = "passed"
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
