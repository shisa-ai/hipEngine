"""Diagnostic A/B for GGUF selected-down X8 q8_1 + sudot4."""

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
    ap.add_argument("--quant", choices=("q5", "q6", "both"), default="both")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--in-features", type=int, default=512)
    ap.add_argument("--out-features", type=int, default=2048)
    ap.add_argument("--input-scale", type=float, default=0.1)
    ap.add_argument("--raw-threads", type=int, choices=(64, 128, 256), default=128)
    ap.add_argument("--x8-threads", type=int, choices=(64, 128, 256), default=64)
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
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        build_gguf_k_gemv,
        gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
        gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
        gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
        gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
        build_gguf_x8_selected_gemv,
        gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
        gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16, repack_gguf_q6_k_tile16
    from hipengine.quant.gguf_x8 import repack_gguf_q5_k_x8, repack_gguf_q6_k_x8
    from tests._gguf_synthetic_weights import make_q5_k_weight, make_q6_k_weight

    rt = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)
    k_library = build_gguf_k_gemv(load=True, require_cached=args.require_cached_build)
    t16_library = build_gguf_t16_selected_gemv(load=True, require_cached=args.require_cached_build)
    x8_library = build_gguf_x8_selected_gemv(load=True, require_cached=args.require_cached_build)

    rng = np.random.default_rng(1061)
    x = _f32_to_bf16_bits(
        (rng.standard_normal((args.rows, args.in_features)) * args.input_scale).astype(np.float32)
    )
    selected = np.ascontiguousarray((np.arange(args.rows) % args.experts).astype(np.int64))

    quant_names = ("q5", "q6") if args.quant == "both" else (args.quant,)
    results = []
    for quant in quant_names:
        if quant == "q5":
            make_weight = make_q5_k_weight
            t16_tiles = lambda raw: repack_gguf_q5_k_tile16(raw).tiles
            x8_tiles = lambda raw: repack_gguf_q5_k_x8(raw).tiles
            t16_wrapper = gguf_q5_k_t16_selected_gemv_bf16_bf16_out
            raw_float_wrapper = gguf_q5_k_selected_pack8_gemv_bf16_bf16_out
            raw_dp4a_wrapper = gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
            x8_dp4a_wrapper = gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
            roll = 37
        else:
            make_weight = make_q6_k_weight
            t16_tiles = lambda raw: repack_gguf_q6_k_tile16(raw).tiles
            x8_tiles = lambda raw: repack_gguf_q6_k_x8(raw).tiles
            t16_wrapper = gguf_q6_k_t16_selected_gemv_bf16_bf16_out
            raw_float_wrapper = gguf_q6_k_selected_pack8_gemv_bf16_bf16_out
            raw_dp4a_wrapper = gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
            x8_dp4a_wrapper = gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
            roll = 53

        base = make_weight(args.out_features, args.in_features)
        qweight = np.ascontiguousarray(
            np.stack([np.roll(base, shift=e + roll, axis=0) for e in range(args.experts)], axis=0)
        )
        t16 = t16_tiles(qweight)
        x8 = x8_tiles(qweight)

        out_t16 = np.zeros((args.rows, args.out_features), np.uint16)
        out_raw_float = np.zeros_like(out_t16)
        out_raw_dp4a = np.zeros_like(out_t16)
        out_x8_dp4a = np.zeros_like(out_t16)
        bufs = []

        def dev(arr: np.ndarray):
            buf = malloc(arr.nbytes, runtime=rt)
            copy_host_to_device(buf, host_array_ptr(arr), runtime=rt)
            bufs.append(buf)
            return buf

        try:
            x_buf = dev(x)
            selected_buf = dev(selected)
            qweight_buf = dev(qweight)
            t16_buf = dev(t16)
            x8_buf = dev(x8)
            t16_out_buf = malloc(out_t16.nbytes, runtime=rt)
            raw_float_out_buf = malloc(out_raw_float.nbytes, runtime=rt)
            raw_dp4a_out_buf = malloc(out_raw_dp4a.nbytes, runtime=rt)
            x8_dp4a_out_buf = malloc(out_x8_dp4a.nbytes, runtime=rt)
            xq_buf = malloc(args.rows * (args.in_features // 32) * 36, runtime=rt)
            bufs.extend((t16_out_buf, raw_float_out_buf, raw_dp4a_out_buf, x8_dp4a_out_buf, xq_buf))

            def t16_float() -> None:
                t16_wrapper(
                    x_buf.ptr,
                    selected_buf.ptr,
                    t16_buf.ptr,
                    t16_out_buf.ptr,
                    args.rows,
                    args.rows,
                    args.experts,
                    args.in_features,
                    args.out_features,
                    library=t16_library,
                    runtime=rt,
                )

            def raw_float() -> None:
                raw_float_wrapper(
                    x_buf.ptr,
                    selected_buf.ptr,
                    qweight_buf.ptr,
                    raw_float_out_buf.ptr,
                    args.rows,
                    args.rows,
                    args.experts,
                    args.in_features,
                    args.out_features,
                    threads=args.raw_threads,
                    library=k_library,
                    runtime=rt,
                )

            def quantize() -> None:
                gguf_q4_k_quantize_bf16_q8_1(
                    x_buf.ptr,
                    xq_buf.ptr,
                    args.rows,
                    args.in_features,
                    library=q4_library,
                    runtime=rt,
                )

            def raw_dot() -> None:
                raw_dp4a_wrapper(
                    xq_buf.ptr,
                    selected_buf.ptr,
                    qweight_buf.ptr,
                    raw_dp4a_out_buf.ptr,
                    args.rows,
                    args.rows,
                    args.experts,
                    args.in_features,
                    args.out_features,
                    threads=args.raw_threads,
                    library=k_library,
                    runtime=rt,
                )

            def x8_dot() -> None:
                x8_dp4a_wrapper(
                    xq_buf.ptr,
                    selected_buf.ptr,
                    x8_buf.ptr,
                    x8_dp4a_out_buf.ptr,
                    args.rows,
                    args.rows,
                    args.experts,
                    args.in_features,
                    args.out_features,
                    threads=args.x8_threads,
                    library=x8_library,
                    runtime=rt,
                )

            def raw_quant_dot() -> None:
                quantize()
                raw_dot()

            def x8_quant_dot() -> None:
                quantize()
                x8_dot()

            def bench(fn) -> float:
                for _ in range(args.warmup):
                    fn()
                rt.device_synchronize()
                start = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                rt.device_synchronize()
                return (time.perf_counter() - start) * 1000.0 / args.iters

            quantize()
            rt.device_synchronize()
            t16_ms = bench(t16_float)
            raw_float_ms = bench(raw_float)
            quant_ms = bench(quantize)
            raw_dot_ms = bench(raw_dot)
            raw_quant_dot_ms = bench(raw_quant_dot)
            x8_dot_ms = bench(x8_dot)
            x8_quant_dot_ms = bench(x8_quant_dot)

            t16_float()
            raw_float()
            raw_quant_dot()
            x8_quant_dot()
            rt.device_synchronize()
            copy_device_to_host(host_array_ptr(out_t16), t16_out_buf, runtime=rt)
            copy_device_to_host(host_array_ptr(out_raw_float), raw_float_out_buf, runtime=rt)
            copy_device_to_host(host_array_ptr(out_raw_dp4a), raw_dp4a_out_buf, runtime=rt)
            copy_device_to_host(host_array_ptr(out_x8_dp4a), x8_dp4a_out_buf, runtime=rt)
        finally:
            for buf in reversed(bufs):
                free(buf, runtime=rt)

        t16_ref = _bf16_bits_to_f32(out_t16)
        raw_float_ref = _bf16_bits_to_f32(out_raw_float)
        raw_dp4a = _bf16_bits_to_f32(out_raw_dp4a)
        x8_dp4a = _bf16_bits_to_f32(out_x8_dp4a)
        kl_t16_mean, kl_t16_max = _softmax_kl(t16_ref, x8_dp4a)
        kl_raw_mean, kl_raw_max = _softmax_kl(raw_float_ref, x8_dp4a)
        results.append(
            {
                "quant": quant,
                "timing_ms": {
                    "production_t16_float": t16_ms,
                    "raw_selected_pack8_float": raw_float_ms,
                    "q8_1_quantize": quant_ms,
                    "raw_selected_pack8_dp4a_dot_prequantized": raw_dot_ms,
                    "raw_selected_pack8_dp4a_quantize_plus_dot": raw_quant_dot_ms,
                    "x8_selected_dp4a_dot_prequantized": x8_dot_ms,
                    "x8_selected_dp4a_quantize_plus_dot": x8_quant_dot_ms,
                },
                "speedup": {
                    "production_t16_over_x8_dp4a_dot": t16_ms / x8_dot_ms,
                    "production_t16_over_x8_dp4a_quantize_plus_dot": t16_ms / x8_quant_dot_ms,
                    "raw_float_over_x8_dp4a_quantize_plus_dot": raw_float_ms / x8_quant_dot_ms,
                    "x8_dp4a_dot_over_raw_dp4a_dot": raw_dot_ms / x8_dot_ms,
                    "x8_dp4a_quant_dot_over_raw_dp4a_quant_dot": raw_quant_dot_ms / x8_quant_dot_ms,
                },
                "correctness_vs_production_t16_float": {
                    "max_abs": float(np.max(np.abs(t16_ref - x8_dp4a))),
                    "mean_abs": float(np.mean(np.abs(t16_ref - x8_dp4a))),
                    "kl_mean": kl_t16_mean,
                    "kl_max": kl_t16_max,
                    "top1": _top1(t16_ref, x8_dp4a),
                },
                "correctness_vs_raw_float": {
                    "max_abs": float(np.max(np.abs(raw_float_ref - x8_dp4a))),
                    "mean_abs": float(np.mean(np.abs(raw_float_ref - x8_dp4a))),
                    "kl_mean": kl_raw_mean,
                    "kl_max": kl_raw_max,
                    "top1": _top1(raw_float_ref, x8_dp4a),
                },
                "x8_vs_raw_dp4a": {
                    "max_abs": float(np.max(np.abs(raw_dp4a - x8_dp4a))),
                    "mean_abs": float(np.mean(np.abs(raw_dp4a - x8_dp4a))),
                    "top1": _top1(raw_dp4a, x8_dp4a),
                },
            }
        )

    env_prefix = []
    for name in ("PYTHONPATH", "HIPENGINE_HIP_ARCH", "HIPENGINE_COMPILER_VERSION_FILE"):
        value = os.environ.get(name)
        if value:
            env_prefix.append(f"{name}={value}")

    result = {
        "schema": "hipengine.gguf_x8_selected_down_dp4a_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "shape": {
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "input_scale": args.input_scale,
            "raw_threads": args.raw_threads,
            "x8_threads": args.x8_threads,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
        "command": " ".join(env_prefix + [Path(sys.executable).name] + sys.argv),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
