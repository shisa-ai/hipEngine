"""Diagnostic A/B for GGUF Q4_K T16 selected-dual q8_1 + sudot4.

This measures the production decode-repack T16 selected-MoE gate/up kernels
only. It does not promote the dp4a route to the runtime default.
"""

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
    ap.add_argument("--x-rows", type=int, default=4)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--in-features", type=int, default=2048)
    ap.add_argument("--out-features", type=int, default=512)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.rows % args.x_rows != 0:
        raise ValueError("--rows must be divisible by --x-rows")
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
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
        gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from tests._gguf_synthetic_weights import make_q4_k_weight

    rt = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)
    t16_library = build_gguf_t16_selected_gemv(load=True, require_cached=args.require_cached_build)

    rng = np.random.default_rng(627)
    x = _f32_to_bf16_bits(
        (rng.standard_normal((args.x_rows, args.in_features)) * 0.1).astype(np.float32)
    )
    selected = np.ascontiguousarray((np.arange(args.rows) % args.experts).astype(np.int64))
    base = make_q4_k_weight(args.out_features, args.in_features)
    qa = np.ascontiguousarray(
        np.stack([np.roll(base, e % args.out_features, axis=0) for e in range(args.experts)], axis=0)
    )
    qb = np.ascontiguousarray(
        np.stack([np.roll(base, (e * 3) % args.out_features, axis=0) for e in range(args.experts)], axis=0)
    )
    ta = repack_gguf_q4_k_tile16(qa).tiles
    tb = repack_gguf_q4_k_tile16(qb).tiles

    out_ref_a = np.zeros((args.rows, args.out_features), np.uint16)
    out_ref_b = np.zeros_like(out_ref_a)
    out_dp4a_a = np.zeros_like(out_ref_a)
    out_dp4a_b = np.zeros_like(out_ref_a)
    out_silu_ref = np.zeros_like(out_ref_a)
    out_silu_dp4a = np.zeros_like(out_ref_a)
    bufs = []

    def dev(arr: np.ndarray):
        buf = malloc(arr.nbytes, runtime=rt)
        copy_host_to_device(buf, host_array_ptr(arr), runtime=rt)
        bufs.append(buf)
        return buf

    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        ta_buf = dev(ta)
        tb_buf = dev(tb)
        ref_a_buf = malloc(out_ref_a.nbytes, runtime=rt)
        ref_b_buf = malloc(out_ref_b.nbytes, runtime=rt)
        dp4a_a_buf = malloc(out_dp4a_a.nbytes, runtime=rt)
        dp4a_b_buf = malloc(out_dp4a_b.nbytes, runtime=rt)
        silu_ref_buf = malloc(out_silu_ref.nbytes, runtime=rt)
        silu_dp4a_buf = malloc(out_silu_dp4a.nbytes, runtime=rt)
        xq_buf = malloc(args.x_rows * (args.in_features // 32) * 36, runtime=rt)
        bufs.extend((ref_a_buf, ref_b_buf, dp4a_a_buf, dp4a_b_buf, silu_ref_buf, silu_dp4a_buf, xq_buf))

        def direct() -> None:
            gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                ta_buf.ptr,
                tb_buf.ptr,
                ref_a_buf.ptr,
                ref_b_buf.ptr,
                args.x_rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=t16_library,
                runtime=rt,
            )

        def silu_direct() -> None:
            gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                ta_buf.ptr,
                tb_buf.ptr,
                silu_ref_buf.ptr,
                args.x_rows,
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
                args.x_rows,
                args.in_features,
                library=q4_library,
                runtime=rt,
            )

        def dot() -> None:
            gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_buf.ptr,
                selected_buf.ptr,
                ta_buf.ptr,
                tb_buf.ptr,
                dp4a_a_buf.ptr,
                dp4a_b_buf.ptr,
                args.x_rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=t16_library,
                runtime=rt,
            )

        def silu_dot() -> None:
            gguf_q4_k_t16_selected_dual_silu_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_buf.ptr,
                selected_buf.ptr,
                ta_buf.ptr,
                tb_buf.ptr,
                silu_dp4a_buf.ptr,
                args.x_rows,
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

        def quant_silu_dot() -> None:
            quant()
            silu_dot()

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
        silu_direct_ms = bench(silu_direct)
        quant_ms = bench(quant)
        dot_ms = bench(dot)
        quant_dot_ms = bench(quant_dot)
        silu_dot_ms = bench(silu_dot)
        quant_silu_dot_ms = bench(quant_silu_dot)

        direct()
        silu_direct()
        quant_dot()
        silu_dot()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(out_ref_a), ref_a_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_ref_b), ref_b_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_dp4a_a), dp4a_a_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_dp4a_b), dp4a_b_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_silu_ref), silu_ref_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_silu_dp4a), silu_dp4a_buf, runtime=rt)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=rt)

    ref_a = _bf16_bits_to_f32(out_ref_a)
    ref_b = _bf16_bits_to_f32(out_ref_b)
    dp4a_a = _bf16_bits_to_f32(out_dp4a_a)
    dp4a_b = _bf16_bits_to_f32(out_dp4a_b)
    silu_ref = _bf16_bits_to_f32(out_silu_ref)
    silu_dp4a = _bf16_bits_to_f32(out_silu_dp4a)
    kl_a_mean, kl_a_max = _softmax_kl(ref_a, dp4a_a)
    kl_b_mean, kl_b_max = _softmax_kl(ref_b, dp4a_b)
    kl_silu_mean, kl_silu_max = _softmax_kl(silu_ref, silu_dp4a)

    env_prefix = []
    for name in ("PYTHONPATH", "HIPENGINE_HIP_ARCH", "HIPENGINE_COMPILER_VERSION_FILE"):
        value = os.environ.get(name)
        if value:
            env_prefix.append(f"{name}={value}")

    result = {
        "schema": "hipengine.gguf_q4_k_t16_selected_dual_dp4a_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "shape": {
            "x_rows": args.x_rows,
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "timing_ms": {
            "t16_selected_dual": direct_ms,
            "t16_selected_dual_silu": silu_direct_ms,
            "q8_1_quantize": quant_ms,
            "t16_dp4a_dot_prequantized": dot_ms,
            "t16_dp4a_quantize_plus_dot": quant_dot_ms,
            "t16_silu_dp4a_dot_prequantized": silu_dot_ms,
            "t16_silu_dp4a_quantize_plus_dot": quant_silu_dot_ms,
        },
        "speedup": {
            "t16_dual_over_dp4a_dot": direct_ms / dot_ms,
            "t16_dual_over_dp4a_quantize_plus_dot": direct_ms / quant_dot_ms,
            "t16_silu_over_dp4a_dot": silu_direct_ms / silu_dot_ms,
            "t16_silu_over_dp4a_quantize_plus_dot": silu_direct_ms / quant_silu_dot_ms,
        },
        "correctness_vs_t16_float": {
            "gate": {
                "max_abs": float(np.max(np.abs(ref_a - dp4a_a))),
                "mean_abs": float(np.mean(np.abs(ref_a - dp4a_a))),
                "kl_mean": kl_a_mean,
                "kl_max": kl_a_max,
                "top1": _top1(ref_a, dp4a_a),
            },
            "up": {
                "max_abs": float(np.max(np.abs(ref_b - dp4a_b))),
                "mean_abs": float(np.mean(np.abs(ref_b - dp4a_b))),
                "kl_mean": kl_b_mean,
                "kl_max": kl_b_max,
                "top1": _top1(ref_b, dp4a_b),
            },
            "silu": {
                "max_abs": float(np.max(np.abs(silu_ref - silu_dp4a))),
                "mean_abs": float(np.mean(np.abs(silu_ref - silu_dp4a))),
                "kl_mean": kl_silu_mean,
                "kl_max": kl_silu_max,
                "top1": _top1(silu_ref, silu_dp4a),
            },
        },
        "command": " ".join(env_prefix + [Path(sys.executable).name] + sys.argv),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
