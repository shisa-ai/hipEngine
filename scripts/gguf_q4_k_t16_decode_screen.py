"""D1 screen: t16 decode GEMV vs the production selected GEMV at Gemma gate_up.

Iteration 38's scoped measurement: the loader-side t16 repack must not start on
an assumed number, so the t16 decode kernel is measured at Gemma 4 MoE gate_up
decode shapes (in 2816, out 1408, top_k rows, 128 resident experts) against the
production raw-layout selected GEMV in the same process, on the same bytes.

Both arms read identical Q4_K weight bytes (the t16 arm is a bit-lossless
repack of the same raw stack) for the same 8 selected rows of one token, and
their outputs are compared; a fixture that does not reproduce agreement fails
loudly. Reported per arm: ms/call and achieved GB/s over the selected bytes
(rows x out x bytes_per_row). The reference to beat is the production form's
measured ~104 GB/s (iterations 31/32; the D1 row's 95 GB/s at trace shape).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

QK_K = 256
Q4_K_BLOCK_BYTES = 144


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    cand = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        s = x - x.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))

    p = np.exp(logsm(ref))
    return float(np.mean(np.sum(p * (logsm(ref) - logsm(cand)), axis=-1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--in-features", type=int, default=2816)
    ap.add_argument("--out-features", type=int, default=1408)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_selected_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from tests._gguf_synthetic_weights import make_q4_k_weight

    H, O, E, R = args.in_features, args.out_features, args.experts, args.rows
    if H % QK_K:
        raise SystemExit(f"in_features {H} must be a multiple of {QK_K}")
    if O % 16:
        raise SystemExit(f"out_features {O} must be a multiple of 16 (T16 tile)")
    build_gguf_q4_k_gemv(load=True, require_cached=args.require_cached_build)
    rt = get_hip_runtime()

    bytes_per_row = H // QK_K * Q4_K_BLOCK_BYTES
    selected_bytes = R * O * bytes_per_row

    # One raw Q4_K stack, rolled per expert exactly like the campaign's MoE
    # microbench, so both arms consume byte-identical weights.
    base = make_q4_k_weight(O, H)
    raw = np.ascontiguousarray(
        np.stack([np.roll(base, e + 1, 0) for e in range(E)])
    )
    repack_started = time.perf_counter()
    tiles = repack_gguf_q4_k_tile16(raw).tiles
    repack_seconds = time.perf_counter() - repack_started

    rng = np.random.default_rng(31)
    x = _f32_to_bf16_u16((rng.standard_normal((1, H)) * 1e-3).astype(np.float32))
    selected = np.ascontiguousarray(np.arange(R, dtype=np.int64) % E)

    bufs: list = []

    def dev(arr: np.ndarray) -> int:
        b = malloc(arr.nbytes)
        copy_host_to_device(b, host_array_ptr(arr), arr.nbytes)
        bufs.append(b)
        return b

    try:
        xb, sb, rb, tb = dev(x), dev(selected), dev(raw), dev(tiles)
        out_prod = np.zeros((R, O), np.uint16)
        out_t16 = np.zeros((R, O), np.uint16)
        opb, otb = dev(out_prod), dev(out_t16)

        def run_prod() -> None:
            gguf_q4_k_selected_gemv_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                rb.ptr,
                opb.ptr,
                1,
                R,
                E,
                H,
                O,
                threads=128,
            )

        def run_t16() -> None:
            gguf_q4_k_t16_selected_gemv_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                tb.ptr,
                otb.ptr,
                1,
                R,
                E,
                H,
                O,
            )

        def bench(fn) -> float:
            for _ in range(args.warmup):
                fn()
            rt.device_synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            rt.device_synchronize()
            return (time.perf_counter() - t0) / args.iters * 1e3  # ms/call

        # First make sure both arms agree before timing anything: a broken
        # fixture must not produce a plausible bandwidth number.
        run_prod()
        run_t16()
        rt.device_synchronize()
        copy_device_to_host(host_array_ptr(out_prod), opb, out_prod.nbytes)
        copy_device_to_host(host_array_ptr(out_t16), otb, out_t16.nbytes)
        p = _bf16_u16_to_f32(out_prod)
        t = _bf16_u16_to_f32(out_t16)
        kl = _softmax_kl(p, t)
        max_abs = float(np.max(np.abs(p - t)))
        mean_abs = float(np.mean(np.abs(p - t)))
        scale = float(np.max(np.abs(p)))
        rel = max_abs / scale if scale else 0.0
        if not np.isfinite(kl) or rel > 2e-2:
            raise SystemExit(
                f"arms disagree: kl={kl:.3e} max_abs={max_abs:.4e} rel={rel:.3e}"
            )

        prod_ms = bench(run_prod)
        t16_ms = bench(run_t16)

        def gbs(ms: float) -> float:
            return selected_bytes / (ms * 1e-3) / 1e9

        result = {
            "kind": "gemma4-t16-decode-gemv-screen",
            "provenance": {
                "git": _git_head(),
                "gpu_env": {k: os.environ[k] for k in ("ROCR_VISIBLE_DEVICES",) if k in os.environ},
                "iters": args.iters,
                "warmup": args.warmup,
                "command_args": {
                    k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                },
                "protocol": "same-run same-bytes A/B; both arms warmed before timing (agreement phase runs both first)",
            },
            "shape": {
                "in_features": H,
                "out_features": O,
                "experts": E,
                "rows": R,
                "bytes_per_row": bytes_per_row,
                "selected_bytes": selected_bytes,
            },
            "production": {"ms_per_call": prod_ms, "gbps": gbs(prod_ms)},
            "t16": {
                "ms_per_call": t16_ms,
                "gbps": gbs(t16_ms),
                "repack_seconds_host": repack_seconds,
            },
            "speedup_prod_over_t16": prod_ms / t16_ms,
            "agreement": {
                "kl_prod_vs_t16": kl,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "rel_to_peak": rel,
            },
            "reference": {
                "production_measured_gbps": 104.0,
                "note": "iterations 31/32 measured both launch structures at ~104 GB/s; the D1 row's trace shape gives 95 GB/s at 187 us",
            },
        }
        print(
            f"shape: in={H} out={O} E={E} rows={R} selected_bytes={selected_bytes/1e6:.1f} MB"
        )
        print(f"production : {prod_ms:.4f} ms/call  {gbs(prod_ms):6.1f} GB/s")
        print(f"t16        : {t16_ms:.4f} ms/call  {gbs(t16_ms):6.1f} GB/s")
        print(f"speedup (prod/t16): {prod_ms / t16_ms:.3f}x")
        print(
            f"agreement  : kl={kl:.3e} max_abs={max_abs:.4e} rel={rel:.3e} (same bytes, different layout)"
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            print(f"wrote {args.out}")
    finally:
        for b in bufs:
            free(b)


if __name__ == "__main__":
    main()