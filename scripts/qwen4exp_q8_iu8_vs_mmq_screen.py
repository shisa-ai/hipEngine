#!/usr/bin/env python3
"""#23 screen: Q8_0 F32/F32 dense linear, iu8-WMMA vs the production MMQ chain.

On actual weights for the three MMQ-policy shapes (attention-qkv 2560->10240,
QSA attn_q 2560->12288, GDN ssm_out 6144->2560), measures the iu8-WMMA
kernel's speed and drift versus the guarded raw-Q8 MMQ chain (quantize +
guarded prefill + sparse exact repair) that production currently selects at
prefill row counts. T1 production-correctness screen: drift is reported, not
repaired. Diagnostic only.
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
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as mmq
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_iu8_wmma_prefill_f32_f32,
)
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import (
    _upload, _alloc, _download,
)
from tests.test_qwen4exp_mmq_prepack import pack_reference

MMQ_VARIANT = "gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
VEC4_VARIANT = "gguf_q8_0_mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out"


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
    mmq_library = mmq.build_gguf_q8_0_mmq_prefill(load=True)

    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]

    def tensor(name):
        reader = next(
            r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        return reader.tensor_data(name), info

    # The three production MMQ-policy shapes from the R2d owner ledger.
    # attn_q lives on the QSA layers (blk.3), not blk.0.
    shapes = {}
    for reader in readers:
        for t in reader.info.tensors:
            if t.ggml_type_name != "Q8_0":
                continue
            if not (t.name.startswith("blk.0.") or t.name.startswith("blk.3.")):
                continue
            out_f, in_f = t.shape
            if (in_f, out_f) == (2560, 10240):
                shapes.setdefault("attn_qkv", (t.name, in_f, out_f))
            elif (in_f, out_f) == (2560, 12288):
                shapes.setdefault("attn_q", (t.name, in_f, out_f))
            elif (in_f, out_f) == (6144, 2560):
                shapes.setdefault("ssm_out", (t.name, in_f, out_f))

    report = {
        "schema": 1,
        "kind": "qwen4exp_q8_iu8_vs_mmq_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "arithmetic_class": "T1_diagnostic",
        "runtime_default_changed": False,
        "boundary": ("Q8_0 F32/F32 dense linear only: guarded raw-Q8 MMQ chain "
                     "(quantize + guarded prefill + sparse exact repair, "
                     "production variant, threshold 0) versus iu8-WMMA "
                     "three-plane kernel; drift reported, not repaired"),
        "mmq_variant": MMQ_VARIANT,
        "cases": [],
    }

    allocations = []
    try:
        for label, (name, in_features, out_features) in shapes.items():
            raw, info = tensor(name)
            dw = _upload(raw, runtime, allocations)
            for rows in a.rows:
                mark = len(allocations)
                rng = np.random.default_rng(7100 + rows)
                x = rng.normal(0.0, 1.0, size=(rows, in_features)).astype(np.float32)
                dx = _upload(x, runtime, allocations)
                out_m = _alloc(rows * out_features, np.float32, runtime, allocations)
                out_i = _alloc(rows * out_features, np.float32, runtime, allocations)
                d4 = _alloc(
                    (mmq.q8_mmq_d4x3_nbytes(rows, in_features),),
                    np.uint8, runtime, allocations)
                count = _alloc((1,), np.int32, runtime, allocations)
                indices = _alloc(
                    (rows * out_features,), np.int32, runtime, allocations)
                # Production arm: K-major prepacked weights + vec4 variant.
                packed = pack_reference(raw, out_features, in_features)
                dp = _upload(packed, runtime, allocations)
                out_v = _alloc(rows * out_features, np.float32, runtime, allocations)

                def run_mmq() -> None:
                    mmq.gguf_q8_0_mmq128_quantize_f32_d4x3(
                        dx.ptr, d4.ptr, rows, in_features,
                        library=mmq_library, runtime=runtime)
                    runtime.memset(count.ptr, 0, 4)
                    getattr(mmq, MMQ_VARIANT)(
                        d4.ptr, dw.ptr, out_m.ptr, count.ptr, indices.ptr,
                        rows * out_features, 0.0, rows, in_features, out_features,
                        library=mmq_library, runtime=runtime)
                    mmq.gguf_q8_0_mmq128_sparse_exact_correct_f32(
                        dx.ptr, dw.ptr, out_m.ptr, count.ptr, indices.ptr,
                        rows * out_features, rows, in_features, out_features,
                        library=mmq_library, runtime=runtime)
                    runtime.device_synchronize()

                def run_vec4() -> None:
                    mmq.gguf_q8_0_mmq128_quantize_f32_d4x3(
                        dx.ptr, d4.ptr, rows, in_features,
                        library=mmq_library, runtime=runtime)
                    runtime.memset(count.ptr, 0, 4)
                    getattr(mmq, VEC4_VARIANT)(
                        d4.ptr, dp.ptr, out_v.ptr, count.ptr, indices.ptr,
                        rows * out_features, 0.0, rows, in_features, out_features,
                        library=mmq_library, runtime=runtime)
                    mmq.gguf_q8_0_mmq128_sparse_exact_correct_f32(
                        dx.ptr, dw.ptr, out_v.ptr, count.ptr, indices.ptr,
                        rows * out_features, rows, in_features, out_features,
                        library=mmq_library, runtime=runtime)
                    runtime.device_synchronize()

                def run_iu8() -> None:
                    gguf_q8_0_iu8_wmma_prefill_f32_f32(
                        dx.ptr, dw.ptr, out_i.ptr, rows, in_features,
                        out_features, library=library, runtime=runtime)
                    runtime.device_synchronize()

                run_mmq()
                run_vec4()
                run_iu8()
                gm = _download(out_m, (rows, out_features), np.float32, runtime)
                gi = _download(out_i, (rows, out_features), np.float32, runtime)
                diff = np.abs(gi - gm)
                rel = diff / np.maximum(np.abs(gm), 1e-30)

                times = {"mmq": [], "vec4": [], "iu8": []}
                for pair in range(a.pairs):
                    if pair % 3 == 0:
                        order = (("mmq", run_mmq), ("vec4", run_vec4), ("iu8", run_iu8))
                    elif pair % 3 == 1:
                        order = (("vec4", run_vec4), ("iu8", run_iu8), ("mmq", run_mmq))
                    else:
                        order = (("iu8", run_iu8), ("mmq", run_mmq), ("vec4", run_vec4))
                    for who, fn in order:
                        start = time.perf_counter()
                        fn()
                        times[who].append(time.perf_counter() - start)

                mmq_ms = statistics.median(times["mmq"]) * 1e3
                vec4_ms = statistics.median(times["vec4"]) * 1e3
                iu8_ms = statistics.median(times["iu8"]) * 1e3
                case = {
                    "label": label,
                    "tensor": name,
                    "shape": [rows, in_features, out_features],
                    "weight_sha256": hashlib.sha256(raw).hexdigest(),
                    "mmq_ms": mmq_ms,
                    "vec4_ms": vec4_ms,
                    "iu8_ms": iu8_ms,
                    "speedup_vec4_over_iu8": vec4_ms / iu8_ms if iu8_ms else None,
                    "drift_abs_max": float(diff.max()),
                    "drift_abs_p999": float(np.quantile(diff, 0.999)),
                    "drift_rel_max": float(rel.max()),
                    "drift_rel_p999": float(np.quantile(rel, 0.999)),
                }
                report["cases"].append(case)
                print(
                    f"{label:10s} rows={rows} k={in_features} n={out_features}: "
                    f"MMQ {mmq_ms:8.2f} ms  vec4 {vec4_ms:8.2f} ms  "
                    f"iu8 {iu8_ms:8.2f} ms  vec4/iu8 {vec4_ms / iu8_ms:5.2f}x  "
                    f"drift rel max {rel.max():.3e}"
                )
                # free the per-rows working set
                while len(allocations) > mark:
                    free(allocations.pop())
        report["status"] = "passed"
    finally:
        while allocations:
            free(allocations.pop())

    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
