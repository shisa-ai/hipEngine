#!/usr/bin/env python3
"""#28 R12 dense Q8 MMQ tile-geometry screen at the exact prefill legs.

Benchmarks the retained raw-vec4 d4x3 guarded MMQ at the four dense-leg
geometries (attn_qkv 2560->10240, attn_q 2560->6144, ssm_out 6144->2560,
plus attn_gate 2560->6144 on the same MMQ family) against the alternate
tile instantiations (64x64, 64x128, 128x64). Every tile publishes
bit-identical outputs to the retained 128x128 kernel (verified per
geometry), so this is a pure geometry/occupancy measurement:

  128x128: 57.9 KiB LDS/block -> 1 block/CU on gfx1151
  64x64:    28.9 KiB LDS/block -> 2 blocks/CU
  64x128:   38.0 KiB LDS/block
  128x64:   48.4 KiB LDS/block

Also records the f32->d4x3 quantize cost per geometry (the activation
plane producer) so tile gains are not misattributed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime


def hip_available() -> bool:
    import ctypes
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    build_gguf_q8_0_mmq_prefill,
    gguf_q8_0_mmq128_quantize_f32_d4x1,
    gguf_q8_0_mmq128_quantize_f32_d4x2,
    gguf_q8_0_mmq128_quantize_f32_d4x3,
    gguf_q8_0_mmq128_raw_vec4_q8_1_d4x1_guarded_f32_f32_out,
    gguf_q8_0_mmq128_raw_vec4_q8_1_d4x2_guarded_f32_f32_out,
    gguf_q8_0_mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out,
    gguf_q8_0_mmq128_tile_raw_vec4_q8_1_d4x3_guarded_f32_f32_out,
)

TILES = ("128x128", "64x64", "64x128", "128x64")
# plane-count family (128x128 tile, raw vec4): 3 = retained production
PLANES = (
    ("d4x3", gguf_q8_0_mmq128_quantize_f32_d4x3,
     gguf_q8_0_mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out, 3),
    ("d4x2", gguf_q8_0_mmq128_quantize_f32_d4x2,
     gguf_q8_0_mmq128_raw_vec4_q8_1_d4x2_guarded_f32_f32_out, 2),
    ("d4x1", gguf_q8_0_mmq128_quantize_f32_d4x1,
     gguf_q8_0_mmq128_raw_vec4_q8_1_d4x1_guarded_f32_f32_out, 1),
)
GEOMETRIES = (
    ("attn_qkv", 2560, 10240),
    ("attn_q", 2560, 6144),
    ("ssm_out", 6144, 2560),
    ("attn_gate", 2560, 6144),
)
Q8_QK = 32
Q8_BLOCK_BYTES = 34
D4_STRIDE_BYTES = 36 * 4  # per 256-K block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, action="append", default=[1024])
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    runtime = get_hip_runtime()
    library = build_gguf_q8_0_mmq_prefill(load=True)
    rng = np.random.default_rng(a.seed)
    stream = 0

    report = {
        "schema": 1, "kind": "qwen4exp_q8_mmq_tile_screen",
        "pairs": a.pairs, "seed": a.seed, "rows_sweep": a.rows,
        "tiles": list(TILES), "cases": [],
        "note": "bit-identical outputs verified per geometry vs the 128x128 tile",
    }
    allocations = []

    def _free_all():
        while allocations:
            free(allocations.pop(), runtime=runtime)

    try:
        for name, k, n in GEOMETRIES:
            for rows in a.rows:
                x = (rng.standard_normal((rows, k)) * 0.5).astype(np.float32)
                blocks = k // Q8_QK
                qs = rng.integers(-127, 128, size=(n, blocks, Q8_QK), dtype=np.int8)
                scales = (rng.uniform(0.002, 0.02, size=(n, blocks))).astype(np.float16)
                # ggml block_q8_0: fp16 scale first, then 32 int8 qs
                w = np.zeros((n, blocks, Q8_BLOCK_BYTES), dtype=np.uint8)
                w[:, :, :2] = scales.view(np.uint8).reshape(n, blocks, 2)
                w[:, :, 2:2 + Q8_QK] = qs.view(np.uint8)

                x_host = np.ascontiguousarray(x.reshape(-1))
                w_host = np.ascontiguousarray(w.reshape(-1))
                x_buf = malloc(x_host.nbytes, runtime=runtime); allocations.append(x_buf)
                w_buf = malloc(w_host.nbytes, runtime=runtime); allocations.append(w_buf)
                copy_host_to_device(x_buf, host_array_ptr(x_host), runtime=runtime)
                copy_host_to_device(w_buf, host_array_ptr(w_host), runtime=runtime)

                # three d4x3 residual planes, each (k/128) x rows blocks of
                # 36 words (144 B): plane stride is (hidden/128)*rows blocks
                d4_bytes = 3 * (k // 128) * rows * D4_STRIDE_BYTES
                d4_buf = malloc(d4_bytes, runtime=runtime); allocations.append(d4_buf)
                out_buf = malloc(rows * n * 4, runtime=runtime); allocations.append(out_buf)
                count_buf = malloc(4, runtime=runtime); allocations.append(count_buf)
                indices_buf = malloc(16, runtime=runtime); allocations.append(indices_buf)
                zero = np.zeros(1, dtype=np.int32)
                copy_host_to_device(count_buf, host_array_ptr(zero), runtime=runtime)

                gguf_q8_0_mmq128_quantize_f32_d4x3(
                    x_buf.ptr, d4_buf.ptr, rows, k, stream=stream,
                    library=library, runtime=runtime)
                runtime.device_synchronize()
                qt = []
                for _ in range(a.pairs):
                    t0 = time.perf_counter()
                    gguf_q8_0_mmq128_quantize_f32_d4x3(
                        x_buf.ptr, d4_buf.ptr, rows, k, stream=stream,
                        library=library, runtime=runtime)
                    runtime.device_synchronize()
                    qt.append(time.perf_counter() - t0)

                case = {"geometry": name, "k": k, "n": n, "rows": rows,
                        "quantize_us": statistics.median(qt) * 1e6,
                        "tiles": {}}
                ref_out = None
                for tile in TILES:
                    if tile == "128x128":
                        fn = gguf_q8_0_mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out
                    else:
                        def fn(*args, _t=tile, **kw):
                            return gguf_q8_0_mmq128_tile_raw_vec4_q8_1_d4x3_guarded_f32_f32_out(
                                *args, _tile=_t, **kw)
                    fn(d4_buf.ptr, w_buf.ptr, out_buf.ptr, count_buf.ptr,
                       indices_buf.ptr, 4, 1e30, rows, k, n,
                       stream=stream, library=library, runtime=runtime)
                    runtime.device_synchronize()
                    got = np.empty(rows * n, dtype=np.float32)
                    copy_device_to_host(host_array_ptr(got), out_buf, runtime=runtime)
                    if ref_out is None:
                        ref_out = got
                        exact = True
                    else:
                        exact = bool(np.array_equal(got, ref_out))
                    times = []
                    for _ in range(a.pairs):
                        t0 = time.perf_counter()
                        fn(d4_buf.ptr, w_buf.ptr, out_buf.ptr, count_buf.ptr,
                           indices_buf.ptr, 4, 1e30, rows, k, n,
                           stream=stream, library=library, runtime=runtime)
                        runtime.device_synchronize()
                        times.append(time.perf_counter() - t0)
                    med = statistics.median(times) * 1e6
                    gflop = 2.0 * rows * k * n / 1e9
                    case["tiles"][tile] = {
                        "median_us": med,
                        "tf_s": gflop / (med / 1e6) / 1e3,
                        "bit_exact_vs_128x128": exact,
                    }
                # plane-count sweep at the retained 128x128 tile
                plane_case = {"geometry": name, "k": k, "n": n, "rows": rows,
                              "planes": {}}
                for pname, qfn, mfn, passes in PLANES:
                    pd4_bytes = passes * (k // 128) * rows * D4_STRIDE_BYTES
                    pd4_buf = malloc(pd4_bytes, runtime=runtime)
                    allocations.append(pd4_buf)
                    qfn(x_buf.ptr, pd4_buf.ptr, rows, k, stream=stream,
                        library=library, runtime=runtime)
                    runtime.device_synchronize()
                    mfn(pd4_buf.ptr, w_buf.ptr, out_buf.ptr, count_buf.ptr,
                        indices_buf.ptr, 4, 1e30, rows, k, n,
                        stream=stream, library=library, runtime=runtime)
                    runtime.device_synchronize()
                    got = np.empty(rows * n, dtype=np.float32)
                    copy_device_to_host(host_array_ptr(got), out_buf, runtime=runtime)
                    times = []
                    for _ in range(a.pairs):
                        t0 = time.perf_counter()
                        mfn(pd4_buf.ptr, w_buf.ptr, out_buf.ptr, count_buf.ptr,
                            indices_buf.ptr, 4, 1e30, rows, k, n,
                            stream=stream, library=library, runtime=runtime)
                        runtime.device_synchronize()
                        times.append(time.perf_counter() - t0)
                    med = statistics.median(times) * 1e6
                    diff = np.abs(got - ref_out)
                    rel = diff / np.maximum(np.abs(ref_out), 1e-6)
                    plane_case["planes"][pname] = {
                        "median_us": med,
                        "max_abs_diff_vs_d4x3": float(diff.max()),
                        "p999_abs_diff": float(np.percentile(diff, 99.9)),
                        "rel_p50": float(np.percentile(rel, 50)),
                        "rel_p999": float(np.percentile(rel, 99.9)),
                    }
                    free(pd4_buf, runtime=runtime)
                    allocations.remove(pd4_buf)
                pbase = plane_case["planes"]["d4x3"]["median_us"]
                for pname, _, _, _ in PLANES:
                    plane_case["planes"][pname]["speedup_vs_d4x3"] = (
                        pbase / plane_case["planes"][pname]["median_us"])
                report["cases"].append(plane_case)
                print("  planes %s K=%d N=%d rows=%d: " % (name, k, n, rows) +
                      " ".join("%s %.0fus (%.2fx, rel_p999 %.1e)" % (
                          pn, plane_case["planes"][pn]["median_us"],
                          plane_case["planes"][pn]["speedup_vs_d4x3"],
                          plane_case["planes"][pn]["rel_p999"])
                          for pn, _, _, _ in PLANES), flush=True)
                report["cases"].append(case)
                base = case["tiles"]["128x128"]["median_us"]
                for tile in TILES:
                    case["tiles"][tile]["speedup_vs_128x128"] = (
                        base / case["tiles"][tile]["median_us"])
                print("%s K=%d N=%d rows=%d: " % (name, k, n, rows) +
                      " ".join("%s %.0fus (%.2fx)%s" % (
                          t, case["tiles"][t]["median_us"],
                          case["tiles"][t]["speedup_vs_128x128"],
                          "" if case["tiles"][t]["bit_exact_vs_128x128"] else " MISMATCH")
                          for t in TILES), flush=True)
                _free_all()
    finally:
        _free_all()
    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
