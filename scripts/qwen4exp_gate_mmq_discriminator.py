#!/usr/bin/env python3
"""Discriminate the GDN-gate MMQ error source: quantization vs f32 order.

The (2560, 6144) attn_gate (GDN gate) shape is the largest exact-coltile
owner in the non-GR linear family. Its T2 MMQ policy row failed the
production bars only marginally (p95 5.26e-3 vs 5e-3, top-1 98.67% vs 99%).
This probe measures, on actual layer weights and F32 inputs:

  * operation-complete speed of the exact wave-scale coltile parent versus
    the 3-residual-plane MMQ chain (quantize + guarded tile, no repair),
  * the F32 bit-flip fraction between the two chains,
  * the error of each chain against a float64 truth, decomposing whether
    the MMQ chain's excess error is activation-quantization dominated
    (a 4th residual plane could pass the T2 bars) or f32-accumulation-order
    dominated (no plane count helps; the chain is at its numerical floor).

Diagnostic only; no production code path is changed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as mmq
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def dequant_q8_0(raw: np.ndarray, n: int, k: int) -> np.ndarray:
    blocks = raw.reshape(n, k // 32, 34)
    d = (
        blocks[:, :, 0:2]
        .copy()
        .view(np.float16)
        .astype(np.float64)
        .reshape(n, k // 32)
    )
    q = blocks[:, :, 2:].astype(np.int8).astype(np.float64)
    return (d[:, :, None] * q).reshape(n, k)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--tensor", default="blk.0.attn_gate.weight")
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--truth-rows", type=int, default=512,
                   help="row count for the float64 truth decomposition (0 disables)")
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    if a.pairs < 1 or any(r < 1 for r in a.rows):
        p.error("positive rows and pairs required")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    runtime = get_hip_runtime()
    gemv_lib = q8.build_gguf_k_gemv(load=True, require_cached=bool(a.require_cached_build))
    mmq_lib = mmq.build_gguf_q8_0_mmq_prefill(load=True)

    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    reader = next(r for r in readers if any(t.name == a.tensor for t in r.info.tensors))
    info = reader.tensor_info(a.tensor)
    assert info.ggml_type_name == "Q8_0", info.ggml_type_name
    n, k = info.shape
    raw = reader.tensor_data(a.tensor)

    report = {
        "schema": 1,
        "kind": "qwen4exp_gate_mmq_discriminator",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "tensor": a.tensor,
        "tensor_shape": [int(n), int(k)],
        "tensor_sha256": hashlib.sha256(raw).hexdigest(),
        "arithmetic_class": "diagnostic",
        "runtime_default_changed": False,
        "boundary": ("projection only: exact wave-scale coltile parent versus "
                      "3-plane quantize+guarded MMQ without repair; no GDN "
                      "recurrence or downstream consumers"),
        "parent_variant": PARENT,
        "candidate_variants": [
            "gguf_q8_0_mmq128_quantize_f32_d4x3",
            "gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
        ],
        "cases": [],
    }

    allocations: list = []
    try:
        dw = _upload(raw, runtime, allocations)
        for rows in a.rows:
            mark = len(allocations)
            x = (
                np.random.default_rng(5486 + rows)
                .normal(0.0, 1.0, (rows, k))
                .astype(np.float32)
            )
            dx = _upload(x, runtime, allocations)
            d4 = _alloc(mmq.q8_mmq_d4x3_nbytes(rows, k), np.uint8, runtime, allocations)
            out_parent = _alloc(rows * n, np.float32, runtime, allocations)
            out_mmq = _alloc(rows * n, np.float32, runtime, allocations)
            risk_count = _alloc(1, np.int32, runtime, allocations)
            risk_cap = rows * n
            risk_indices = _alloc(risk_cap, np.int32, runtime, allocations)
            zero = np.zeros(1, dtype=np.int32)
            from hipengine.core.memory import copy_host_to_device, host_array_ptr

            def run_parent() -> None:
                getattr(q8, PARENT)(
                    dx.ptr, dw.ptr, out_parent.ptr, rows, k, n,
                    runtime=runtime, library=gemv_lib,
                )
                runtime.device_synchronize()

            def run_mmq() -> None:
                copy_host_to_device(risk_count, host_array_ptr(zero), 4,
                                    runtime=runtime)
                mmq.gguf_q8_0_mmq128_quantize_f32_d4x3(
                    dx.ptr, d4.ptr, rows, k,
                    library=mmq_lib, runtime=runtime,
                )
                mmq.gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out(
                    d4.ptr, dw.ptr, out_mmq.ptr,
                    risk_count.ptr, risk_indices.ptr, risk_cap, 0.0,
                    rows, k, n,
                    library=mmq_lib, runtime=runtime,
                )
                runtime.device_synchronize()

            run_parent()
            run_mmq()
            parent = _download(out_parent, (rows, n), np.float32, runtime)
            cand = _download(out_mmq, (rows, n), np.float32, runtime)
            flips = int(np.count_nonzero(
                parent.view(np.uint32) != cand.view(np.uint32)))
            abs_delta = np.abs(cand.astype(np.float64) - parent.astype(np.float64))
            rel_delta = abs_delta / np.maximum(np.abs(parent), 1e-30)

            times = {"parent": [], "mmq3": []}
            for pair in range(a.pairs):
                order = ((run_parent, "parent"), (run_mmq, "mmq3"))
                if pair % 2 == 1:
                    order = order[::-1]
                for fn, name in order:
                    start = time.perf_counter()
                    fn()
                    times[name].append(time.perf_counter() - start)

            case = {
                "rows": rows,
                "f32_bit_flips": flips,
                "f32_flip_fraction": flips / (rows * n),
                "abs_delta_max": float(abs_delta.max()),
                "abs_delta_median": float(np.median(abs_delta)),
                "rel_delta_p95": float(np.percentile(rel_delta, 95)),
                "parent_median_ms": statistics.median(times["parent"]) * 1e3,
                "mmq3_median_ms": statistics.median(times["mmq3"]) * 1e3,
                "speedup": (statistics.median(times["parent"]) /
                            statistics.median(times["mmq3"])),
                "seconds": times,
            }

            if a.truth_rows and rows <= a.truth_rows:
                w64 = dequant_q8_0(raw, n, k)
                truth = x.astype(np.float64) @ w64.T
                e_parent = np.abs(parent.astype(np.float64) - truth)
                e_mmq = np.abs(cand.astype(np.float64) - truth)
                scale = np.maximum(np.abs(truth), 1e-30)
                case["truth"] = {
                    "e_parent_median": float(np.median(e_parent)),
                    "e_parent_p95": float(np.percentile(e_parent, 95)),
                    "e_mmq3_median": float(np.median(e_mmq)),
                    "e_mmq3_p95": float(np.percentile(e_mmq, 95)),
                    "e_mmq3_over_e_parent_median": float(
                        np.median(e_mmq) / max(np.median(e_parent), 1e-300)),
                    "e_mmq3_over_e_parent_p95": float(
                        np.percentile(e_mmq, 95) / max(np.percentile(e_parent, 95), 1e-300)),
                    "rel_e_parent_median": float(np.median(e_parent / scale)),
                    "rel_e_mmq3_median": float(np.median(e_mmq / scale)),
                    "rel_e_mmq3_p95": float(np.percentile(e_mmq / scale, 95)),
                    "ulp_of_truth_p95": float(np.percentile(
                        np.abs(truth) * 2 ** -23, 95)),
                }
            report["cases"].append(case)
            print(json.dumps(case))
            for ptr in reversed(allocations[mark:]):
                free(ptr, runtime=runtime)
            del allocations[mark:]
    finally:
        for ptr in reversed(allocations):
            free(ptr, runtime=runtime)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
