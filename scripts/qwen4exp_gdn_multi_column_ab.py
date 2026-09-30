#!/usr/bin/env python3
"""A/B the Qwen4Exp GDN multi-column prefill route against its DPP incumbent.

The multi-column route (`HIPENGINE_QWEN4_GDN_MULTI_COLUMN`, registered as
`qwen4exp_gdn_tiled16_multi_prefill`) widens the per-warp column tile from 1 to 8, so one
k-load feeds eight columns instead of one. It is the same kernel template with different
`warps_per_block`/`cols` arguments, and both configurations cover all 128 value columns
exactly once with the same per-lane k-reduction.

Both arms are built from the same source with `-DHIPENGINE_QWEN4_GDN_DPP=1`, so the only
difference between them is the multi-column flag.

Usage:
    python3 scripts/qwen4exp_gdn_multi_column_ab.py --tokens 1024 --reps 20
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (  # noqa: E402
    build_qwen4_exp_gdn_dpp,
    qwen4_exp_gdn_prefill_tiled16_dpp_f32,
    qwen4_exp_gdn_prefill_tiled16_multi_f32,
)

H_K = 16
H_V = 48
D_K = 128
D_V = 128


def _upload(array, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _time(fn, runtime, reps, warmup=3):
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        runtime.device_synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4038)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    tokens = args.tokens
    runtime = get_hip_runtime()
    rng = np.random.default_rng(args.seed)

    qkv_width = 2 * H_K * D_K + H_V * D_V
    core_width = H_V * D_V
    conv = rng.normal(0.0, 0.05, size=(tokens, qkv_width)).astype(np.float32)
    gate = rng.normal(0.0, 0.5, size=(tokens, core_width)).astype(np.float32)
    alpha = rng.normal(-0.2, 0.1, size=(tokens, H_V)).astype(np.float32)
    beta = rng.normal(0.0, 0.2, size=(tokens, H_V)).astype(np.float32)
    dt_bias = rng.normal(-1.0, 0.1, size=H_V).astype(np.float32)
    a = -np.exp(rng.normal(-0.5, 0.1, size=H_V)).astype(np.float32)
    norm = rng.normal(1.0, 0.05, size=D_V).astype(np.float32)
    initial_state = rng.normal(0.0, 0.01, size=(H_V, D_K, D_V)).astype(np.float32)

    print("building incumbent (DPP) ...", flush=True)
    lib_incumbent = build_qwen4_exp_gdn_dpp(multi_column=False, load=True)
    print("building candidate (DPP + multi-column) ...", flush=True)
    lib_candidate = build_qwen4_exp_gdn_dpp(multi_column=True, load=True)

    allocations = []
    try:
        d_conv = _upload(conv, runtime, allocations)
        d_gate = _upload(gate, runtime, allocations)
        d_alpha = _upload(alpha, runtime, allocations)
        d_beta = _upload(beta, runtime, allocations)
        d_dt = _upload(dt_bias, runtime, allocations)
        d_a = _upload(a, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        d_state_inc = _upload(initial_state, runtime, allocations)
        d_state_can = _upload(initial_state, runtime, allocations)
        d_out_inc = _alloc((tokens, core_width), np.float32, runtime, allocations)
        d_out_can = _alloc((tokens, core_width), np.float32, runtime, allocations)

        common = (d_conv.ptr, d_gate.ptr, d_alpha.ptr, d_beta.ptr, d_dt.ptr, d_a.ptr,
                  d_norm.ptr)

        def run_incumbent():
            qwen4_exp_gdn_prefill_tiled16_dpp_f32(
                *common, d_state_inc.ptr, d_out_inc.ptr,
                tokens, H_K, H_V, D_K, D_V,
                library=lib_incumbent, runtime=runtime)

        def run_candidate():
            qwen4_exp_gdn_prefill_tiled16_multi_f32(
                *common, d_state_can.ptr, d_out_can.ptr,
                tokens, H_K, H_V, D_K, D_V,
                library=lib_candidate, runtime=runtime)

        run_incumbent()
        run_candidate()
        runtime.device_synchronize()
        out_inc = _download(d_out_inc, (tokens, core_width), np.float32, runtime)
        out_can = _download(d_out_can, (tokens, core_width), np.float32, runtime)
        st_inc = _download(d_state_inc, initial_state.shape, np.float32, runtime)
        st_can = _download(d_state_can, initial_state.shape, np.float32, runtime)

        # --- numerics ---
        def drift(ref, cand, label):
            diff = np.asarray(cand, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
            denom = np.linalg.norm(np.asarray(ref, dtype=np.float64))
            return {
                "field": label,
                "bit_exact": bool(np.array_equal(ref, cand)),
                "max_abs": float(np.abs(diff).max()),
                "rms_abs": float(np.sqrt((diff ** 2).mean())),
                "nmse": float((diff ** 2).sum() / max((np.asarray(ref, dtype=np.float64) ** 2).sum(), 1e-300)),
                "rel_fro": float(np.linalg.norm(diff) / denom) if denom else float("nan"),
            }

        num = [drift(out_inc, out_can, "core_out"), drift(st_inc, st_can, "recurrent_state")]

        inc = _time(run_incumbent, runtime, args.reps)
        can = _time(run_candidate, runtime, args.reps)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    inc_med = statistics.median(inc)
    can_med = statistics.median(can)
    speedup = inc_med / can_med if can_med else float("nan")

    print(f"\n=== GDN prefill A/B, {tokens} tokens, Hk={H_K} Hv={H_V} Dk=Dv={D_K} ===")
    print(f"  incumbent (DPP)            : {inc_med:8.3f} ms  (min {min(inc):.3f})")
    print(f"  candidate (DPP+multicol)   : {can_med:8.3f} ms  (min {min(can):.3f})")
    print(f"  speedup (incumbent/candidate): {speedup:.4f}x")
    for row in num:
        print(f"  {row['field']:16} bit_exact={row['bit_exact']}  "
              f"max_abs={row['max_abs']:.3e}  nmse={row['nmse']:.3e}")

    if args.out:
        artifact = {
            "schema": 1,
            "kind": "qwen4exp-gdn-multi-column-ab",
            "performance_claim": True,
            "host": {"hostname": pathlib.Path("/etc/hostname").read_text().strip()
                     if pathlib.Path("/etc/hostname").exists() else "unknown",
                     "gpu": "AMD Radeon 8060S (gfx1151)"},
            "geometry": {"tokens": tokens, "num_k_heads": H_K, "num_v_heads": H_V,
                         "head_k_dim": D_K, "head_v_dim": D_V},
            "arms": {
                "incumbent": {"key": "qwen4exp_gdn_tiled16_dpp_prefill",
                              "flags": ["-DHIPENGINE_QWEN4_GDN_DPP=1"],
                              "median_ms": inc_med, "min_ms": min(inc),
                              "samples_ms": inc},
                "candidate": {"key": "qwen4exp_gdn_tiled16_multi_prefill",
                              "flags": ["-DHIPENGINE_QWEN4_GDN_DPP=1",
                                        "-DHIPENGINE_QWEN4_GDN_MULTI_COLUMN=1"],
                              "median_ms": can_med, "min_ms": min(can),
                              "samples_ms": can},
            },
            "speedup": speedup,
            "numerics": num,
            "reps": args.reps,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"\nartifact: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
