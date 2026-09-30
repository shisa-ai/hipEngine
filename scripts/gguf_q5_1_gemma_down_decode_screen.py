#!/usr/bin/env python3
"""Gemma 4 down-decode screen: production Q5_1 GEMV vs in-tree candidates.

Punchlist D3 row 2 names ``q5_1_selected_gemv_bf16_bf16_kernel`` as the decode
down producer: 22528 workgroups x 256 threads, ~97 us/call in the census trace
(2.81 ms / 28 launches), about 122 GB/s against a 933 GB/s roofline estimate.
This screen measures, at the exact production geometry (in_features=704,
out_features=2816, 128 experts, compact decode rows), same-run same-bytes:

  * production   qwen4_exp_q5_1_selected_gemv_bf16_bf16_out (incumbent)
  * logical_t64  qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out
  * logical_t128 qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out
  * pack8        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out
                 (128-thread pack-of-8 blocks, 4-wave32 reduction, in_features
                 % 32 == 0 — 704 qualifies)

All four arms read the same raw Q5_1 byte stack on device; the agreement phase
runs every arm against the incumbent before any timing. Diagnostic only; no
runtime default changes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime

_QK_Q5_1 = 32
_Q5_1_BLOCK_BYTES = 24


def _git_head() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f = np.ascontiguousarray(arr, dtype=np.float32)
    bits = f.view(np.uint32)
    lsb = (bits >> 16) & 1
    bits = bits + np.uint32(0x7FFF + lsb)
    return (bits >> 16).astype(np.uint16)


def _bf16_u16_to_f32(bits: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
    return (b << 16).view(np.float32)


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
    def logsm(x: np.ndarray) -> np.ndarray:
        x = x.ravel().astype(np.float64)
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    p = logsm(ref)
    q = logsm(cand)
    return float((np.exp(p) * (p - q)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-features", type=int, default=704)
    ap.add_argument("--out-features", type=int, default=2816)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_selected_pack8_gemv import (
        build_gguf_q5_1_selected_pack8_gemv,
        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q5_1_weight

    H, O, E, R = args.in_features, args.out_features, args.experts, args.rows
    if H % _QK_Q5_1:
        raise SystemExit(f"in_features {H} must be a multiple of {_QK_Q5_1}")
    if O % 8:
        raise SystemExit(f"out_features {O} must be a multiple of 8 (pack8)")
    build_qwen4_exp_q5_1(load=True)
    build_gguf_q5_1_selected_pack8_gemv(load=True)
    rt = get_hip_runtime()

    bytes_per_row = H // _QK_Q5_1 * _Q5_1_BLOCK_BYTES
    selected_bytes = R * O * bytes_per_row

    # One raw Q5_1 stack per expert from the shared fixture; every arm reads
    # the same device bytes (roll differs only per expert, like the campaign).
    base = make_q5_1_weight(O, H)
    raw = np.ascontiguousarray(
        np.stack([np.roll(base, e + 1, 0) for e in range(E)])
    )

    rng = np.random.default_rng(20260930)
    x = _f32_to_bf16_u16(
        (rng.standard_normal((R, H)) * 1e-3).astype(np.float32)
    )
    # Distinct ascending experts: the pack8 compact ABI recovers the row's
    # expert by scanning an expert_start prefix, which requires the rows to be
    # grouped by expert in ascending id order.
    selected = np.ascontiguousarray(np.arange(R, dtype=np.int64) % E)
    expert_start = np.arange(R + 1, dtype=np.int64)

    bufs: list = []

    def dev(arr: np.ndarray) -> int:
        b = malloc(arr.nbytes)
        copy_host_to_device(b, host_array_ptr(arr), arr.nbytes)
        bufs.append(b)
        return b

    try:
        xb, sb, rb, esb = dev(x), dev(selected), dev(raw), dev(expert_start)
        outs = {
            name: np.zeros((R, O), np.uint16)
            for name in ("production", "wave64", "logical_t64", "logical_t128", "pack8")
        }
        obufs = {name: dev(buf) for name, buf in outs.items()}

        def run_production() -> None:
            qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                rb.ptr,
                obufs["production"].ptr,
                R,
                R,
                E,
                H,
                O,
            )

        def run_logical_t64() -> None:
            qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                rb.ptr,
                obufs["logical_t64"].ptr,
                R,
                R,
                E,
                H,
                O,
            )

        def run_logical_t128() -> None:
            qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                rb.ptr,
                obufs["logical_t128"].ptr,
                R,
                R,
                E,
                H,
                O,
            )

        def run_wave64() -> None:
            qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out(
                xb.ptr,
                sb.ptr,
                rb.ptr,
                obufs["wave64"].ptr,
                R,
                R,
                E,
                H,
                O,
            )

        def run_pack8() -> None:
            gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(
                xb.ptr,
                esb.ptr,
                rb.ptr,
                obufs["pack8"].ptr,
                R,
                H,
                O,
                E,
            )

        arms = [
            ("production", run_production),
            ("wave64", run_wave64),
            ("logical_t64", run_logical_t64),
            ("logical_t128", run_logical_t128),
            ("pack8", run_pack8),
        ]

        def bench(fn) -> float:
            for _ in range(args.warmup):
                fn()
            rt.device_synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            rt.device_synchronize()
            return (time.perf_counter() - t0) / args.iters * 1e3  # ms/call

        # Agreement first: a broken fixture must not yield a bandwidth number.
        agreement = {}
        for name, fn in arms:
            fn()
        rt.device_synchronize()
        for name, buf in obufs.items():
            copy_device_to_host(host_array_ptr(outs[name]), buf, outs[name].nbytes)
        ref = _bf16_u16_to_f32(outs["production"])
        ref_scale = float(np.max(np.abs(ref)))
        if not np.isfinite(ref_scale) or ref_scale == 0.0:
            raise SystemExit(
                f"production arm produced no usable output (max_abs={ref_scale})"
            )
        for name in outs:
            if name == "production":
                continue
            cand = _bf16_u16_to_f32(outs[name])
            kl = _softmax_kl(ref, cand)
            max_abs = float(np.max(np.abs(ref - cand)))
            scale = float(np.max(np.abs(ref)))
            rel = max_abs / scale if scale else 0.0
            bit_equal = float(np.mean(outs["production"] == outs[name]))
            if not np.isfinite(kl) or rel > 2e-2:
                raise SystemExit(
                    f"arm {name} disagrees: kl={kl:.3e} rel={rel:.3e}"
                )
            agreement[name] = {
                "kl_vs_production": kl,
                "max_abs": max_abs,
                "rel_to_peak": rel,
                "bit_equal_fraction": bit_equal,
            }

        timing = {name: bench(fn) for name, fn in arms}

        def gbs(ms: float) -> float:
            return selected_bytes / (ms * 1e-3) / 1e9

        result = {
            "kind": "gemma4-q5_1-down-decode-screen",
            "provenance": {
                "git": _git_head(),
                "gpu_env": {
                    k: os.environ[k]
                    for k in ("ROCR_VISIBLE_DEVICES",)
                    if k in os.environ
                },
                "iters": args.iters,
                "warmup": args.warmup,
                "command_args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "protocol": (
                    "same-run same-bytes A/B; all arms warmed before timing "
                    "(agreement phase runs every arm against production first)"
                ),
            },
            "shape": {
                "in_features": H,
                "out_features": O,
                "experts": E,
                "rows": R,
                "bytes_per_row": bytes_per_row,
                "selected_bytes": selected_bytes,
            },
            "timing": {
                name: {"ms_per_call": ms, "gbps": gbs(ms)}
                for name, ms in timing.items()
            },
            "speedup_production_over": {
                name: timing["production"] / ms for name, ms in timing.items()
            },
            "agreement": {
                "production_peak_abs": ref_scale,
                **agreement,
            },
            "reference": {
                "census_situ_us": 96.85,
                "census_situ_gbps": 123.0,
                "note": (
                    "in-situ census: 2.81 ms over 28 launches for the down "
                    "family; 122-123 GB/s at rows=8 against a 933 GB/s "
                    "roofline weight-fetch floor of 12.8 us (D3 row 2)"
                ),
            },
        }
        print(
            f"shape: in={H} out={O} E={E} rows={R} "
            f"selected_bytes={selected_bytes / 1e6:.2f} MB"
        )
        for name, ms in timing.items():
            print(
                f"{name:<12}: {ms:.4f} ms/call  {gbs(ms):6.1f} GB/s  "
                f"({timing['production'] / ms:.3f}x vs production)"
            )
        for name, agr in agreement.items():
            print(
                f"{name:<12}: kl={agr['kl_vs_production']:.3e} "
                f"rel={agr['rel_to_peak']:.3e} "
                f"bit_equal={agr['bit_equal_fraction']:.4f}"
            )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, indent=1, sort_keys=True) + "\n"
            )
            print(f"wrote {args.out}")
    finally:
        for b in bufs:
            free(b)


if __name__ == "__main__":
    main()