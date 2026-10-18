#!/usr/bin/env python3
"""D7 GEGLU-fold screen: does consuming gate_up_out inside the down GEMV pay?

The routed decode path runs the activation as its own stage --
`gate_up_out -> gemma4_gelu_tanh_mul_bf16 -> activated -> t64 selected GEMV` --
at 2.0 us/launch x 30 layers = 0.060 ms/token on top of the down GEMV itself.
The fused arm replaces the three kernel visits with one:
`gate_up_out -> t64-geglu selected GEMV`, whose input load computes
`bf16(gelu_tanh(gate) * up)` -- the exact store/load rounding of the chain --
and then the same dot product.

The arithmetic side is already pinned by
`test_q5_1_selected_geglu_down_matches_the_chain_bitwise` (bitwise at rows 1
and 8); this screen measures the time and re-checks the bits at the production
geometry (in 704, out 2816, E 128). The gate is ABSOLUTE (>= 1.0 us/call
saved): the D5 pathology showed a new launch shape can carry fixed cost that
an idle microbenchmark hides, so a ratio against a ~50 us denominator would
flatter a marginal result.

Run: python3 scripts/gemma4_geglu_down_fold_screen.py --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_QK_Q5_1 = 32
_Q5_1_BLOCK_BYTES = 24


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _f32_to_bf16_u16(arr) -> "np.ndarray":
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--in-features", type=int, default=704)
    parser.add_argument("--out-features", type=int, default=2816)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "2026-09-30-gemma4-geglu-fold-screen.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.gpu))
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        build_gemma4_moe,
        gemma4_gelu_tanh_mul_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_gemv_geglu_logical256_t64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q5_1_weight

    H, O, E = args.in_features, args.out_features, args.experts
    if H % _QK_Q5_1:
        raise SystemExit(f"in_features {H} must be a multiple of {_QK_Q5_1}")
    build_qwen4_exp_q5_1(load=True)
    build_gemma4_moe(load=True)
    rt = get_hip_runtime()

    base = make_q5_1_weight(O, H)
    raw = np.ascontiguousarray(np.stack([np.roll(base, e + 1, 0) for e in range(E)]))

    rng = np.random.default_rng(20260930)
    bufs: list = []

    def dev(arr: np.ndarray) -> int:
        b = malloc(arr.nbytes)
        copy_host_to_device(b, host_array_ptr(arr), arr.nbytes)
        bufs.append(b)
        return b

    results: dict[str, float] = {}
    bit_equal: dict[str, bool] = {}

    try:
        rb = dev(raw)
        for rows in (1, 8):
            gate_up = _f32_to_bf16_u16(
                (rng.standard_normal((rows, 2 * H)) * 0.3).astype(np.float32)
            )
            selected = np.ascontiguousarray(
                (np.arange(rows, dtype=np.int64) * 7) % E
            )
            gb, sb = dev(gate_up), dev(selected)
            act_buf = dev(np.zeros((rows, H), np.uint16))
            out_chain = dev(np.zeros((rows, O), np.uint16))
            out_fold = dev(np.zeros((rows, O), np.uint16))

            def run_chain(gb=gb, sb=sb, act=act_buf, ob=out_chain, rows=rows) -> None:
                gemma4_gelu_tanh_mul_bf16(gb.ptr, act.ptr, rows, H)
                qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out(
                    act.ptr, sb.ptr, rb.ptr, ob.ptr, rows, rows, E, H, O
                )

            def run_fold(gb=gb, sb=sb, ob=out_fold, rows=rows) -> None:
                qwen4_exp_q5_1_selected_gemv_geglu_logical256_t64_bf16_bf16_out(
                    gb.ptr, sb.ptr, rb.ptr, ob.ptr, rows, rows, E, H, O
                )

            # Agreement first: bit-equal or we do not time.
            run_chain()
            run_fold()
            chain_bits = np.empty((rows, O), np.uint16)
            fold_bits = np.empty((rows, O), np.uint16)
            copy_device_to_host(host_array_ptr(chain_bits), out_chain, chain_bits.nbytes)
            copy_device_to_host(host_array_ptr(fold_bits), out_fold, fold_bits.nbytes)
            equal = bool(np.array_equal(chain_bits, fold_bits))
            bit_equal[str(rows)] = equal
            if not equal:
                print(f"rows={rows}: fold differs bitwise from chain -- not timing", file=sys.stderr)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({
                    "schema": "gemma4-geglu-fold-screen/v1",
                    "bit_equal": bit_equal,
                    "verdict": "reject",
                }, indent=2) + "\n")
                return 3

            def bench(fn) -> float:
                for _ in range(args.warmup):
                    fn()
                rt.device_synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                rt.device_synchronize()
                return (time.perf_counter() - t0) / args.iters * 1e6

            results[f"chain_r{rows}"] = round(bench(run_chain), 3)
            results[f"fold_r{rows}"] = round(bench(run_fold), 3)
    finally:
        for b in bufs:
            free(b)

    delta_r1 = results["chain_r1"] - results["fold_r1"]
    delta_r8 = results["chain_r8"] - results["fold_r8"]
    wins = delta_r1 >= 1.0 and delta_r8 >= 1.0
    verdict = "wire" if (wins and all(bit_equal.values())) else "record"

    artifact = {
        "schema": "gemma4-geglu-fold-screen/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": (
            f"python3 scripts/gemma4_geglu_down_fold_screen.py --gpu {args.gpu} "
            f"--in-features {H} --out-features {O} --experts {E} --iters {args.iters}"
        ),
        "git_commit": _git_head(),
        "arch": platform.machine(),
        "gpu_env": {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "")},
        "geometry": {"in_features": H, "out_features": O, "experts": E,
                     "iters": args.iters, "warmup": args.warmup},
        "arms_us_per_call": results,
        "delta_us": {"rows1": round(delta_r1, 3), "rows8": round(delta_r8, 3)},
        "bit_equal": bit_equal,
        "gate_us": 1.0,
        "verdict": verdict,
        "evidence_note": (
            "chain = gelu launch + plain t64 selected GEMV; fold = the fused "
            "t64 kernel consuming gate_up_out. Bitwise equality is asserted in "
            "test_q5_1_selected_geglu_down_matches_the_chain_bitwise at rows "
            "1/8 with sentinels; re-checked in-process here. The shipped-path "
            "claim is the same-lane census A/B, not this projection."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    return 0 if verdict == "wire" else 2


if __name__ == "__main__":
    sys.exit(main())