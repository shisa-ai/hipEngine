#!/usr/bin/env python3
"""Correctness smoke for the fused DFlash query-side Q/K/V projection.

The fused kernel must match the unfused GPU fallback
``dense_bf16_to_f32(Q)``, ``dense_bf16_to_f32(K)``, ``dense_bf16_to_bf16(V)``
and stay close to a NumPy BF16-input oracle.  This script is intentionally small
so it can be used as task evidence before retaining an E2E benchmark artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    build_dflash_drafter,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_qkv_proj_bf16_mixed,
)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32).copy()
    lsb = (bits >> 16) & 1
    bits += np.uint32(0x7FFF) + lsb
    return (bits >> 16).astype(np.uint16)


def _bf16_bits_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _cpu_dense_bf16(x_bits: np.ndarray, w_bits: np.ndarray) -> np.ndarray:
    x = _bf16_bits_to_f32(x_bits).astype(np.float32)
    w = _bf16_bits_to_f32(w_bits).astype(np.float32)
    rows, in_features = x.shape
    out_features = w.shape[0]
    out = np.empty((rows, out_features), dtype=np.float32)
    for row in range(rows):
        for col in range(out_features):
            acc = np.float32(0.0)
            for dim in range(in_features):
                acc = np.float32(acc + np.float32(x[row, dim] * w[col, dim]))
            out[row, col] = acc
    return out


def _copy_in(array: np.ndarray, *, runtime) -> DeviceBuffer:
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


def _read(ptr: int, shape: tuple[int, ...], dtype: np.dtype, *, runtime) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), DeviceBuffer(ptr, out.nbytes), runtime=runtime)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.compiler_version_file:
        compiler_version = Path(args.compiler_version_file).read_text(encoding="utf-8")
    else:
        compiler_version = None
    library = build_dflash_drafter(
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
        load=True,
    )
    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)
    rows = int(args.rows)
    in_features = int(args.in_features)
    q_features = int(args.q_features)
    kv_features = int(args.kv_features)
    x = _f32_to_bf16_bits(rng.normal(0.0, 0.35, size=(rows, in_features)).astype(np.float32))
    q_w = _f32_to_bf16_bits(rng.normal(0.0, 0.25, size=(q_features, in_features)).astype(np.float32))
    k_w = _f32_to_bf16_bits(rng.normal(0.0, 0.25, size=(kv_features, in_features)).astype(np.float32))
    v_w = _f32_to_bf16_bits(rng.normal(0.0, 0.25, size=(kv_features, in_features)).astype(np.float32))
    buffers: list[DeviceBuffer] = []
    try:
        x_b = _copy_in(x, runtime=runtime); buffers.append(x_b)
        q_b = _copy_in(q_w, runtime=runtime); buffers.append(q_b)
        k_b = _copy_in(k_w, runtime=runtime); buffers.append(k_b)
        v_b = _copy_in(v_w, runtime=runtime); buffers.append(v_b)
        q_ref_b = malloc(rows * q_features * DType.FP32.itemsize, runtime=runtime); buffers.append(q_ref_b)
        k_ref_b = malloc(rows * kv_features * DType.FP32.itemsize, runtime=runtime); buffers.append(k_ref_b)
        v_ref_b = malloc(rows * kv_features * DType.BF16.itemsize, runtime=runtime); buffers.append(v_ref_b)
        q_fused_b = malloc(rows * q_features * DType.FP32.itemsize, runtime=runtime); buffers.append(q_fused_b)
        k_fused_b = malloc(rows * kv_features * DType.FP32.itemsize, runtime=runtime); buffers.append(k_fused_b)
        v_fused_b = malloc(rows * kv_features * DType.BF16.itemsize, runtime=runtime); buffers.append(v_fused_b)
        dflash_dense_bf16_to_f32(x_b.ptr, q_b.ptr, q_ref_b.ptr, rows, in_features, q_features, threads=args.threads, library=library, runtime=runtime)
        dflash_dense_bf16_to_f32(x_b.ptr, k_b.ptr, k_ref_b.ptr, rows, in_features, kv_features, threads=args.threads, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(x_b.ptr, v_b.ptr, v_ref_b.ptr, rows, in_features, kv_features, threads=args.threads, library=library, runtime=runtime)
        dflash_qkv_proj_bf16_mixed(
            x_b.ptr,
            q_b.ptr,
            k_b.ptr,
            v_b.ptr,
            q_fused_b.ptr,
            k_fused_b.ptr,
            v_fused_b.ptr,
            rows,
            in_features,
            q_features,
            kv_features,
            threads=args.threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        q_ref = _read(q_ref_b.ptr, (rows, q_features), np.float32, runtime=runtime)
        k_ref = _read(k_ref_b.ptr, (rows, kv_features), np.float32, runtime=runtime)
        v_ref = _read(v_ref_b.ptr, (rows, kv_features), np.uint16, runtime=runtime)
        q_fused = _read(q_fused_b.ptr, (rows, q_features), np.float32, runtime=runtime)
        k_fused = _read(k_fused_b.ptr, (rows, kv_features), np.float32, runtime=runtime)
        v_fused = _read(v_fused_b.ptr, (rows, kv_features), np.uint16, runtime=runtime)
        q_cpu = _cpu_dense_bf16(x, q_w)
        k_cpu = _cpu_dense_bf16(x, k_w)
        v_cpu = _f32_to_bf16_bits(_cpu_dense_bf16(x, v_w))
        q_equal = bool(np.array_equal(q_ref.view(np.uint32), q_fused.view(np.uint32)))
        k_equal = bool(np.array_equal(k_ref.view(np.uint32), k_fused.view(np.uint32)))
        v_equal = bool(np.array_equal(v_ref, v_fused))
        q_cpu_err = float(np.max(np.abs(q_fused - q_cpu)))
        k_cpu_err = float(np.max(np.abs(k_fused - k_cpu)))
        v_cpu_equal = bool(np.array_equal(v_fused, v_cpu))
        passed = q_equal and k_equal and v_equal and q_cpu_err <= args.cpu_atol and k_cpu_err <= args.cpu_atol and v_cpu_equal
        return {
            "passed": passed,
            "shape": {"rows": rows, "in_features": in_features, "q_features": q_features, "kv_features": kv_features},
            "gpu_unfused_match": {"q_f32_bit_equal": q_equal, "k_f32_bit_equal": k_equal, "v_bf16_bit_equal": v_equal},
            "cpu_oracle": {"q_max_abs": q_cpu_err, "k_max_abs": k_cpu_err, "v_bf16_bit_equal": v_cpu_equal, "atol": float(args.cpu_atol)},
        }
    finally:
        for buf in reversed(buffers):
            free(buf, runtime=runtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=32)
    parser.add_argument("--q-features", type=int, default=24)
    parser.add_argument("--kv-features", type=int, default=8)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--cpu-atol", type=float, default=1.0e-5)
    parser.add_argument("--compiler-version-file", default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
