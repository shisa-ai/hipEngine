"""Diagnostic A/B for exact GGUF Q6_K T16 selected-down expert pairing."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--in-features", type=int, default=512)
    ap.add_argument("--out-features", type=int, default=2048)
    ap.add_argument("--input-scale", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--selection-pattern",
        choices=("unique", "random", "paired"),
        default="unique",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.in_features % 256 != 0:
        raise ValueError("--in-features must be divisible by 256")
    if args.out_features % 16 != 0:
        raise ValueError("--out-features must be divisible by 16")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
        gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16
    from tests._gguf_synthetic_weights import make_q6_k_weight

    rt = get_hip_runtime()
    library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    rng = np.random.default_rng(631)
    x = _f32_to_bf16_bits(
        (rng.standard_normal((args.rows, args.in_features)) * args.input_scale).astype(np.float32)
    )
    if args.selection_pattern == "paired":
        if args.rows % 2 != 0:
            raise ValueError("--rows must be even for --selection-pattern paired")
        half = np.arange(args.rows // 2, dtype=np.int64) % args.experts
        selected = np.ascontiguousarray(np.concatenate((half, half)))
    elif args.selection_pattern == "random":
        selected = np.ascontiguousarray(
            rng.integers(0, args.experts, size=args.rows, dtype=np.int64)
        )
    else:
        selected = np.ascontiguousarray(np.arange(args.rows, dtype=np.int64) % args.experts)

    base = make_q6_k_weight(args.out_features, args.in_features)
    qweight = np.ascontiguousarray(
        np.stack([np.roll(base, shift=e + 79, axis=0) for e in range(args.experts)], axis=0)
    )
    tiles = repack_gguf_q6_k_tile16(qweight).tiles
    out_ref = np.zeros((args.rows, args.out_features), np.uint16)
    out_pairreuse = np.zeros_like(out_ref)
    buffers = []

    def dev(arr: np.ndarray):
        buf = malloc(arr.nbytes, runtime=rt)
        copy_host_to_device(buf, host_array_ptr(arr), runtime=rt)
        buffers.append(buf)
        return buf

    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        tile_buf = dev(tiles)
        ref_buf = malloc(out_ref.nbytes, runtime=rt)
        pairreuse_buf = malloc(out_pairreuse.nbytes, runtime=rt)
        buffers.extend((ref_buf, pairreuse_buf))

        def direct() -> None:
            gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                tile_buf.ptr,
                ref_buf.ptr,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=library,
                runtime=rt,
            )

        def pairreuse() -> None:
            gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                tile_buf.ptr,
                pairreuse_buf.ptr,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                library=library,
                runtime=rt,
            )

        def bench(fn) -> float:
            for _ in range(args.warmup):
                fn()
            rt.device_synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            rt.device_synchronize()
            return (time.perf_counter() - start) * 1000.0 / args.iters

        direct_ms = bench(direct)
        pairreuse_ms = bench(pairreuse)
        direct()
        pairreuse()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(out_ref), ref_buf, runtime=rt)
        copy_device_to_host(host_array_ptr(out_pairreuse), pairreuse_buf, runtime=rt)
    finally:
        for buf in reversed(buffers):
            free(buf, runtime=rt)

    env_prefix = []
    for name in ("PYTHONPATH", "HIPENGINE_HIP_ARCH", "HIPENGINE_COMPILER_VERSION_FILE"):
        value = os.environ.get(name)
        if value:
            env_prefix.append(f"{name}={value}")
    result = {
        "schema": "hipengine.gguf_q6_k_t16_selected_down_pairreuse_microbench.v1",
        "host": platform.node(),
        "hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "shape": {
            "selection_pattern": args.selection_pattern,
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
            "t16_selected_down_pairreuse": pairreuse_ms,
        },
        "speedup": direct_ms / pairreuse_ms,
        "pairreuse_exact": bool(np.array_equal(out_ref, out_pairreuse)),
        "command": " ".join(env_prefix + [Path(sys.executable).name] + sys.argv),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
