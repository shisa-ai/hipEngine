#!/usr/bin/env python3
"""R5 feasibility probe: fused GR up parent vs iu8-WMMA split chain.

On actual hc_attn_up weights (Q8_0, K320->N10240) with realistic F32
activations, measures operation-complete speed and the drift distribution
of the candidate chain (iu8-WMMA projection + sigmoid + gated mean)
against the exact fused parent. T1 production-correctness screen: drift
is reported, not repaired. Diagnostic only; no runtime default changes.
"""

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
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32,
    gguf_q8_0_iu8_wmma_prefill_f32_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    build_qwen4_exp_gr,
    qwen4_exp_gated_mean_f32,
    qwen4_exp_sigmoid_f32,
)
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import (
    _upload, _alloc, _download,
)


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--which", choices=("attn", "ffn"), default="attn")
    p.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    runtime = get_hip_runtime()
    gemv_lib = build_gguf_k_gemv(load=True)
    gr_lib = build_qwen4_exp_gr(load=True)

    name = f"blk.{a.layer}.hc_{a.which}_up.weight"
    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
    info = reader.tensor_info(name)
    out_features, in_features = info.shape  # (10240, 320)
    raw = reader.tensor_data(name)
    branches, hidden = 4, out_features // 4
    assert (branches, hidden, in_features) == (4, 2560, 320), info

    report = {
        "schema": 1,
        "kind": "qwen4exp_gr_iu8_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "weights": {
            "tensor": name, "path": str(reader.path), "shape": info.shape,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "arithmetic_class": "T1_diagnostic",
        "runtime_default_changed": False,
        "boundary": ("GR up projection only: exact fused parent (coltile2 "
                     "branch4 rowbatch4, wave-scale in production) versus "
                     "iu8-WMMA projection + sigmoid + gated-mean split; the "
                     "drift is reported, not repaired (production envelope "
                     "gates any admission)"),
        "cases": [],
    }

    allocations = []
    try:
        dw = _upload(raw, runtime, allocations)
        for rows in a.rows:
            mark = len(allocations)
            rng = np.random.default_rng(6100 + rows)
            # low_rank activations after scaled SiLU: roughly N(0, 0.1)
            x = rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
            normalized = rng.normal(0.0, 1.0, size=(rows, branches * hidden)).astype(np.float32)
            dx = _upload(x, runtime, allocations)
            dn = _upload(normalized, runtime, allocations)
            gate_p = _alloc(rows * branches * hidden, np.float32, runtime, allocations)
            mixed_p = _alloc(rows * hidden, np.float32, runtime, allocations)
            gate_c = _alloc(rows * branches * hidden, np.float32, runtime, allocations)
            mixed_c = _alloc(rows * hidden, np.float32, runtime, allocations)

            def run_parent() -> None:
                gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
                    dx.ptr, dw.ptr, dn.ptr, gate_p.ptr, mixed_p.ptr,
                    rows, in_features, branches, hidden,
                    library=gemv_lib, runtime=runtime)
                runtime.device_synchronize()

            def run_candidate() -> None:
                gguf_q8_0_iu8_wmma_prefill_f32_f32(
                    dx.ptr, dw.ptr, gate_c.ptr, rows, in_features,
                    branches * hidden, library=gemv_lib, runtime=runtime)
                qwen4_exp_sigmoid_f32(
                    gate_c.ptr, gate_c.ptr, rows * branches * hidden,
                    library=gr_lib, runtime=runtime)
                qwen4_exp_gated_mean_f32(
                    dn.ptr, gate_c.ptr, mixed_c.ptr, rows, branches, hidden,
                    library=gr_lib, runtime=runtime)
                runtime.device_synchronize()

            run_parent()
            run_candidate()
            gp = _download(gate_p, (rows, branches * hidden), np.float32, runtime)
            gc = _download(gate_c, (rows, branches * hidden), np.float32, runtime)
            mp = _download(mixed_p, (rows, hidden), np.float32, runtime)
            mc = _download(mixed_c, (rows, hidden), np.float32, runtime)

            def rel_stats(cand, ref):
                diff = np.abs(cand - ref)
                scale = np.maximum(np.abs(ref), 1e-30)
                rel = diff / scale
                return {
                    "max_abs_diff": float(diff.max()),
                    "median_abs_diff": float(np.median(diff)),
                    "rel_p50": float(np.percentile(rel, 50)),
                    "rel_p99": float(np.percentile(rel, 99)),
                    "rel_max": float(rel.max()),
                }

            times = {"parent": [], "candidate": []}
            for pair in range(a.pairs):
                if pair % 2 == 0:
                    order = (("parent", run_parent), ("candidate", run_candidate))
                else:
                    order = (("candidate", run_candidate), ("parent", run_parent))
                for label, fn in order:
                    start = time.perf_counter()
                    fn()
                    times[label].append(time.perf_counter() - start)

            case = {
                "rows": rows,
                "gate_drift": rel_stats(gc, gp),
                "mixed_drift": rel_stats(mc, mp),
                "parent_median_ms": statistics.median(times["parent"]) * 1e3,
                "candidate_median_ms": statistics.median(times["candidate"]) * 1e3,
                "speedup": (statistics.median(times["parent"]) /
                            statistics.median(times["candidate"])),
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
