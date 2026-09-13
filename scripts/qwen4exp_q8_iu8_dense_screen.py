#!/usr/bin/env python3
"""#13 screen: Q8_0 F32/F32 dense linear, iu8-WMMA vs wave-scale coltile.

On actual weights for the two production shapes that route through the
retained F32 coltile family (attention-gate 2560->6144 and the QSA mixer
hidden projection 640->2560), measures the iu8-WMMA chain's speed and
drift versus the wave-scale coltile parent. T1 production-correctness
screen: drift is reported, not repaired. Diagnostic only.
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
    gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out,
    gguf_q8_0_iu8_wmma_prefill_f32_f32,
)
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import (
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
    library = build_gguf_k_gemv(load=True)

    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]

    def tensor(name):
        reader = next(
            r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        return reader.tensor_data(name), info

    # Discover the production shapes that route through the retained F32
    # coltile family: attention-gate (6144, 2560) and shared-expert down
    # (2560, 640). The 2560->10240 attention-qkv stays on the raw MMQ and
    # is included for reference when present.
    shapes = {}
    for reader in readers:
        for t in reader.info.tensors:
            if not t.name.startswith("blk.0.") or t.ggml_type_name != "Q8_0":
                continue
            out_f, in_f = t.shape
            if (in_f, out_f) == (2560, 6144):
                shapes.setdefault("attn_gate", (t.name, in_f, out_f))
            elif (in_f, out_f) == (640, 2560):
                shapes.setdefault("shexp_down", (t.name, in_f, out_f))
            elif (in_f, out_f) == (2560, 10240):
                shapes.setdefault("attn_qkv", (t.name, in_f, out_f))

    report = {
        "schema": 1,
        "kind": "qwen4exp_q8_iu8_dense_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "arithmetic_class": "T1_diagnostic",
        "runtime_default_changed": False,
        "boundary": ("Q8_0 F32/F32 dense linear only: wave-scale coltile "
                     "parent versus iu8-WMMA three-plane chain; drift is "
                     "reported, not repaired"),
        "cases": [],
    }

    allocations = []
    try:
        for label, (name, in_features, out_features) in shapes.items():
            raw, info = tensor(name)
            dw = _upload(raw, runtime, allocations)
            for rows in a.rows:
                mark = len(allocations)
                rng = np.random.default_rng(9100 + rows)
                x = rng.normal(0.0, 1.0, size=(rows, in_features)).astype(np.float32)
                dx = _upload(x, runtime, allocations)
                out_p = _alloc(rows * out_features, np.float32, runtime, allocations)
                out_c = _alloc(rows * out_features, np.float32, runtime, allocations)

                def run_parent() -> None:
                    gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out(
                        dx.ptr, dw.ptr, out_p.ptr, rows, in_features,
                        out_features, library=library, runtime=runtime)
                    runtime.device_synchronize()

                def run_candidate() -> None:
                    gguf_q8_0_iu8_wmma_prefill_f32_f32(
                        dx.ptr, dw.ptr, out_c.ptr, rows, in_features,
                        out_features, library=library, runtime=runtime)
                    runtime.device_synchronize()

                run_parent()
                run_candidate()
                gp = _download(out_p, (rows, out_features), np.float32, runtime)
                gc = _download(out_c, (rows, out_features), np.float32, runtime)
                diff = np.abs(gc - gp)
                rel = diff / np.maximum(np.abs(gp), 1e-30)

                times = {"parent": [], "candidate": []}
                for pair in range(a.pairs):
                    if pair % 2 == 0:
                        order = (("parent", run_parent), ("candidate", run_candidate))
                    else:
                        order = (("candidate", run_candidate), ("parent", run_parent))
                    for who, fn in order:
                        start = time.perf_counter()
                        fn()
                        times[who].append(time.perf_counter() - start)

                case = {
                    "label": label,
                    "tensor": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rows": rows,
                    "drift": {
                        "max_abs": float(diff.max()),
                        "median_abs": float(np.median(diff)),
                        "rel_p50": float(np.percentile(rel, 50)),
                        "rel_p99": float(np.percentile(rel, 99)),
                        "rel_max": float(rel.max()),
                    },
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
