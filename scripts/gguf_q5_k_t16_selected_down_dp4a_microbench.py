"""Diagnostic A/B for GGUF Q5_K T16 selected-down q8_1 + sudot4."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref = logsm(ref64)
    log_cand = logsm(cand64)
    row_kl = np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)
    return float(np.mean(row_kl)), float(np.max(row_kl))


def _top1(ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.mean(ref.argmax(axis=-1) == cand.argmax(axis=-1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--in-features", type=int, default=512)
    ap.add_argument("--out-features", type=int, default=2048)
    ap.add_argument("--input-scale", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.in_features % 256 != 0:
        raise ValueError("--in-features must be divisible by 256")
    if args.out_features % 16 != 0:
        raise ValueError("--out-features must be divisible by 16")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
        gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
    from tests._gguf_synthetic_weights import make_q5_k_weight

    rt = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)
    t16_library = build_gguf_t16_selected_gemv(load=True, require_cached=args.require_cached_build)

    rng = np.random.default_rng(629)
    x = _f32_to_bf16_bits(
        (rng.standard_normal((args.rows, args.in_features)) * args.input_scale).astype(np.float32)
    )
    selected = np.ascontiguousarray((np.arange(args.rows) % args.experts).astype(np.int64))
    base = make_q5_k_weight(args.out_features, args.in_features)
    qweight = np.ascontiguousarray(
        np.stack([np.roll(base, shift=e + 61, axis=0) for e in range(args.experts)], axis=0)
    )
    tiles = repack_gguf_q5_k_tile16(qweight).tiles

    out_ref = np.zeros((args.rows, args.out_features), np.uint16)
    out_dp4a = np.zeros_like(out_ref)
    bufs = []

    def dev(arr: np.ndarray):
        buf = malloc(arr.nbytes, runtime=rt)
        copy_host_to_device(buf, host_array_ptr(arr), runtime=rt)
        bufs.append(buf)
        return buf

    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        tile_buf = dev(tiles)
        ref_buf = malloc(out_ref.nbytes, runtime=rt)
        dp4a_buf = malloc(out_dp4a.nbytes, runtime=rt)
        xq_buf = malloc(args.rows * (args.in_features // 32) * 36, runtime=rt)
        bufs.extend((ref_buf, dp4a_buf, xq_buf))

        def direct() -> None:
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                tile_buf.ptr,
                ref_buf.ptr,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=t16_library,
                runtime=rt,
            )

        def quant() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr,
                xq_buf.ptr,
                args.rows,
                args.in_features,
                library=q4_library,
                runtime=rt,
            )

        def dot() -> None:
            gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_buf.ptr,
                selected_buf.ptr,
                tile_buf.ptr,
                dp4a_buf.ptr,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=t16_library,
                runtime=rt,
            )

        def quant_dot() -> None:
            quant()
            dot()

        def bench(fn) -> float:
            for _ in range(args.warmup):
                fn()
            rt.device_synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            rt.device_synchronize()
            return (time.perf_counter() - start) * 1000.0 / args.iters

        quant()
        rt.device_synchronize()
        direct_ms = bench(direct)
        quant_ms = bench(quant)
        dot_ms = bench(dot)
        quant_dot_ms = bench(quant_dot)

        direct()
        quant_dot()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(out_ref), ref_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_dp4a), dp4a_buf, runtime=rt)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=rt)

    ref = _bf16_bits_to_f32(out_ref)
    dp4a = _bf16_bits_to_f32(out_dp4a)
    kl_mean, kl_max = _softmax_kl(ref, dp4a)

    env_prefix = []
    for name in (
        "PYTHONPATH",
        "HIPENGINE_HIP_ARCH",
        "HIPENGINE_COMPILER_VERSION_FILE",
        "HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS",
    ):
        value = os.environ.get(name)
        if value:
            env_prefix.append(f"{name}={value}")

    result = {
        "schema": "hipengine.gguf_q5_k_t16_selected_down_dp4a_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "selected_dp4a_threads": os.environ.get("HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS", "64"),
        "shape": {
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "input_scale": args.input_scale,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "timing_ms": {
            "t16_selected_down": direct_ms,
            "q8_1_quantize": quant_ms,
            "t16_q5_dp4a_dot_prequantized": dot_ms,
            "t16_q5_dp4a_quantize_plus_dot": quant_dot_ms,
        },
        "speedup": {
            "t16_down_over_dp4a_dot": direct_ms / dot_ms,
            "t16_down_over_dp4a_quantize_plus_dot": direct_ms / quant_dot_ms,
        },
        "correctness_vs_t16_float": {
            "max_abs": float(np.max(np.abs(ref - dp4a))),
            "mean_abs": float(np.mean(np.abs(ref - dp4a))),
            "kl_mean": kl_mean,
            "kl_max": kl_max,
            "top1": _top1(ref, dp4a),
        },
        "command": " ".join(env_prefix + [Path(sys.executable).name] + sys.argv),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
