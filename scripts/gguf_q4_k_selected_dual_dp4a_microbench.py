"""Diagnostic A/B for GGUF Q4_K selected-dual q8_1 + sudot4 POC.

This measures the verifier-shaped MoE gate/up kernel only. It does not route the
runtime default through the POC path.
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
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
        gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
        gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight

    if args.rows % args.x_rows != 0:
        raise ValueError("--rows must be divisible by --x-rows")
    if args.in_features % 256 != 0:
        raise ValueError("--in-features must be divisible by 256")
    if args.in_features % 32 != 0:
        raise ValueError("--in-features must be divisible by 32")

    rt = get_hip_runtime()
    library = build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)

    rng = np.random.default_rng(27)
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

    out_raw_a = np.zeros((args.rows, args.out_features), np.uint16)
    out_raw_b = np.zeros_like(out_raw_a)
    out_dp4a_a = np.zeros_like(out_raw_a)
    out_dp4a_b = np.zeros_like(out_raw_a)
    bufs = []

    def dev(arr: np.ndarray):
        buf = malloc(arr.nbytes, runtime=rt)
        copy_host_to_device(buf, host_array_ptr(arr), runtime=rt)
        bufs.append(buf)
        return buf

    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        qa_buf = dev(qa)
        qb_buf = dev(qb)
        raw_a_buf = malloc(out_raw_a.nbytes, runtime=rt)
        raw_b_buf = malloc(out_raw_b.nbytes, runtime=rt)
        dp4a_a_buf = malloc(out_dp4a_a.nbytes, runtime=rt)
        dp4a_b_buf = malloc(out_dp4a_b.nbytes, runtime=rt)
        xq_buf = malloc(args.x_rows * (args.in_features // 32) * 36, runtime=rt)
        bufs.extend((raw_a_buf, raw_b_buf, dp4a_a_buf, dp4a_b_buf, xq_buf))

        def raw() -> None:
            gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                qa_buf.ptr,
                qb_buf.ptr,
                raw_a_buf.ptr,
                raw_b_buf.ptr,
                args.x_rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                threads=args.threads,
                library=library,
                runtime=rt,
            )

        def quant() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr,
                xq_buf.ptr,
                args.x_rows,
                args.in_features,
                library=library,
                runtime=rt,
            )

        def dot() -> None:
            gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_buf.ptr,
                selected_buf.ptr,
                qa_buf.ptr,
                qb_buf.ptr,
                dp4a_a_buf.ptr,
                dp4a_b_buf.ptr,
                args.x_rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                threads=args.threads,
                library=library,
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
        raw_ms = bench(raw)
        quant_ms = bench(quant)
        dot_ms = bench(dot)
        quant_dot_ms = bench(quant_dot)

        raw()
        quant_dot()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(out_raw_a), raw_a_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_raw_b), raw_b_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_dp4a_a), dp4a_a_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_dp4a_b), dp4a_b_buf, runtime=rt)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=rt)

    raw_a = _bf16_bits_to_f32(out_raw_a)
    raw_b = _bf16_bits_to_f32(out_raw_b)
    dp4a_a = _bf16_bits_to_f32(out_dp4a_a)
    dp4a_b = _bf16_bits_to_f32(out_dp4a_b)
    kl_a_mean, kl_a_max = _softmax_kl(raw_a, dp4a_a)
    kl_b_mean, kl_b_max = _softmax_kl(raw_b, dp4a_b)

    env_prefix = []
    for name in ("PYTHONPATH", "HIPENGINE_HIP_ARCH", "HIPENGINE_COMPILER_VERSION_FILE"):
        value = os.environ.get(name)
        if value:
            env_prefix.append(f"{name}={value}")

    result = {
        "schema": "hipengine.gguf_q4_k_selected_dual_dp4a_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "shape": {
            "x_rows": args.x_rows,
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "threads": args.threads,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "timing_ms": {
            "raw_selected_dual": raw_ms,
            "q8_1_quantize": quant_ms,
            "dp4a_dot_prequantized": dot_ms,
            "dp4a_quantize_plus_dot": quant_dot_ms,
        },
        "speedup": {
            "raw_over_dp4a_dot": raw_ms / dot_ms,
            "raw_over_dp4a_quantize_plus_dot": raw_ms / quant_dot_ms,
        },
        "correctness_vs_raw": {
            "gate": {
                "max_abs": float(np.max(np.abs(raw_a - dp4a_a))),
                "mean_abs": float(np.mean(np.abs(raw_a - dp4a_a))),
                "kl_mean": kl_a_mean,
                "kl_max": kl_a_max,
                "top1": _top1(raw_a, dp4a_a),
            },
            "up": {
                "max_abs": float(np.max(np.abs(raw_b - dp4a_b))),
                "mean_abs": float(np.mean(np.abs(raw_b - dp4a_b))),
                "kl_mean": kl_b_mean,
                "kl_max": kl_b_max,
                "top1": _top1(raw_b, dp4a_b),
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
