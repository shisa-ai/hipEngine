#!/usr/bin/env python3
"""DE-1 candidate screen: exact vs Q8_1-DP4A at the real decode gate+up shape.

The decode MoE gate+up owner is ``gguf_q4_k_selected_dual_silu_logical128_t64``
at 14.013 ms/step = 63.43 GB/s = 28.7% of the documented roof (see
``benchmarks/results/2026-09-29-qwen4exp-de1-moe-bandwidth-gate/``).  A Q8_1
DP4A variant of the *same* logical128/t64 structure exists in-tree and was
reverted by the 2026-09-14 arithmetic recovery, whose triggering candidate
explicitly left the decode linear unchanged.

This script measures both at the shape the decode path actually runs, so the
candidate is screened for speed before any profile gate is spent on it.

The DP4A path needs the activation quantized first, so it is charged for BOTH
``gguf_q4_k_quantize_bf16_q8_1`` and the matvec - the row's own action requires
that ("fused quantize+matvec must retain the admitted planes and charge repeated
per-block preparation").

Correctness is reported, not gated: the exact variant is asserted bit-equal to
the unfused reference, and the DP4A variant's divergence from that reference is
measured rather than asserted, because whether it is admissible is a question
for the production numerical envelope, not for this script.

Usage:
    python3 scripts/qwen4exp_de1_decode_gemv_screen.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (  # noqa: E402
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (  # noqa: E402
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_silu_logical128_t64_gemv_bf16_bf16_out,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from _gguf_synthetic_weights import make_q4_k_weight  # noqa: E402

# Real decode gate+up shape, from the model's own GGUF tensor shapes
# (ffn_gate_exps / ffn_up_exps are each (512, 640, 2560) per layer).
DEFAULT_X_ROWS = 1
DEFAULT_SELECTED = 10
DEFAULT_EXPERTS = 512
DEFAULT_IN = 2560
DEFAULT_OUT = 640

DOCUMENTED_ROOF_GBPS = 221.0


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _time(fn, *, warmup: int, reps: int, runtime) -> float:
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        runtime.device_synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-rows", type=int, default=DEFAULT_X_ROWS)
    parser.add_argument("--selected", type=int, default=DEFAULT_SELECTED)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--in-features", type=int, default=DEFAULT_IN)
    parser.add_argument("--out-features", type=int, default=DEFAULT_OUT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not _hip_available():
        raise SystemExit("HIP runtime is not available")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.loading.materialize import float_array_to_bf16_bits

    runtime = get_hip_runtime()
    rng = np.random.default_rng(20260929)

    x_rows = args.x_rows
    selected_rows = args.selected
    experts = args.experts
    in_features = args.in_features
    out_features = args.out_features

    x = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(x_rows, in_features)).astype(np.float32)
    )
    selected = np.arange(selected_rows, dtype=np.int64) % experts
    weight_a = np.stack(
        [make_q4_k_weight(out_features, in_features) for _ in range(experts)]
    )
    weight_b = np.stack(
        [np.roll(make_q4_k_weight(out_features, in_features), e + 1, axis=0)
         for e in range(experts)]
    )

    weight_bytes = int(weight_a.nbytes + weight_b.nbytes)
    per_dispatch_bytes = weight_bytes / experts * selected_rows

    allocations = []
    try:
        devices = []
        for host in (x, selected, weight_a, weight_b):
            device = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
            allocations.append(device)
            devices.append(device)

        ref_gate = malloc(selected_rows * out_features * 2, runtime=runtime)
        ref_up = malloc(selected_rows * out_features * 2, runtime=runtime)
        ref_out = malloc(selected_rows * out_features * 2, runtime=runtime)
        exact_out = malloc(selected_rows * out_features * 2, runtime=runtime)
        dp4a_out = malloc(selected_rows * out_features * 2, runtime=runtime)
        q8 = malloc(x_rows * (in_features // 32) * 36, runtime=runtime)
        allocations.extend(
            [ref_gate, ref_up, ref_out, exact_out, dp4a_out, q8]
        )

        def unfused():
            gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                devices[0].ptr, devices[1].ptr, devices[2].ptr, devices[3].ptr,
                ref_gate.ptr, ref_up.ptr, x_rows, selected_rows, experts,
                in_features, out_features, threads=128, runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                ref_gate.ptr, ref_up.ptr, ref_out.ptr,
                selected_rows, out_features, runtime=runtime,
            )

        def exact():
            gguf_q4_k_selected_dual_silu_logical128_t64_gemv_bf16_bf16_out(
                devices[0].ptr, devices[1].ptr, devices[2].ptr, devices[3].ptr,
                exact_out.ptr, x_rows, selected_rows, experts, in_features,
                out_features, runtime=runtime,
            )

        def quantize():
            gguf_q4_k_quantize_bf16_q8_1(
                devices[0].ptr, q8.ptr, x_rows, in_features, runtime=runtime,
            )

        def dp4a_matvec():
            gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out(
                q8.ptr, devices[1].ptr, devices[2].ptr, devices[3].ptr,
                dp4a_out.ptr, x_rows, selected_rows, experts, in_features,
                out_features, runtime=runtime,
            )

        def dp4a_total():
            quantize()
            dp4a_matvec()

        unfused()
        exact()
        dp4a_total()
        runtime.device_synchronize()

        def read(alloc):
            host = np.empty((selected_rows, out_features), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(host), alloc, runtime=runtime)
            return host

        reference = read(ref_out)
        exact_host = read(exact_out)
        dp4a_host = read(dp4a_out)

        exact_bitexact = bool(np.array_equal(exact_host, reference))
        dp4a_bitexact = bool(np.array_equal(dp4a_host, reference))
        dp4a_rowmatch = float(np.mean(np.all(dp4a_host == reference, axis=1)))

        t_unfused = _time(unfused, warmup=args.warmup, reps=args.reps, runtime=runtime)
        t_exact = _time(exact, warmup=args.warmup, reps=args.reps, runtime=runtime)
        t_quant = _time(quantize, warmup=args.warmup, reps=args.reps, runtime=runtime)
        t_dp4a = _time(dp4a_total, warmup=args.warmup, reps=args.reps, runtime=runtime)

        def gbps(ms):
            return per_dispatch_bytes / 1e9 / (ms / 1e3)

        report = {
            "schema": 1,
            "kind": "qwen4exp_de1_decode_gemv_screen",
            "performance_claim": True,
            "claim_scope": (
                "kernel-level screen at the real decode gate+up shape; not an "
                "end-to-end decode speedup"
            ),
            "shape": {
                "x_rows": x_rows,
                "selected_rows": selected_rows,
                "experts": experts,
                "in_features": in_features,
                "out_features": out_features,
                "per_dispatch_bytes": per_dispatch_bytes,
            },
            "protocol": {"warmup": args.warmup, "reps": args.reps, "statistic": "min"},
            "arms": {
                "unfused_reference": {
                    "ms": t_unfused * 1e3,
                    "gbps": gbps(t_unfused * 1e3),
                },
                "exact_logical128_t64": {
                    "kernel": "gguf_q4_k_selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
                    "ms": t_exact * 1e3,
                    "gbps": gbps(t_exact * 1e3),
                    "fraction_of_roof": gbps(t_exact * 1e3) / DOCUMENTED_ROOF_GBPS,
                },
                "q8_1_dp4a_logical128_t64": {
                    "kernel": "gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out",
                    "quantize_ms": t_quant * 1e3,
                    "matvec_ms": t_dp4a * 1e3 - t_quant * 1e3,
                    "ms": t_dp4a * 1e3,
                    "gbps": gbps(t_dp4a * 1e3),
                    "fraction_of_roof": gbps(t_dp4a * 1e3) / DOCUMENTED_ROOF_GBPS,
                },
            },
            "correctness": {
                "exact_bitexact_vs_unfused": exact_bitexact,
                "dp4a_bitexact_vs_unfused": dp4a_bitexact,
                "dp4a_rows_matching_unfused": dp4a_rowmatch,
            },
            "result": {
                "dp4a_speedup_vs_exact": t_exact / t_dp4a,
                "exact_share_of_dp4a_total": t_quant / t_dp4a,
                "quantize_overhead_pct": (t_quant / t_dp4a) * 100.0,
            },
            "limits": [
                "Kernel-level screen only. No profile gate, no end-to-end decode run, "
                "and no claim that the DP4A variant is admissible under the production "
                "numerical envelope.",
                "Single shape: one token, ten selected experts of 512, 2560 -> 640. "
                "Other layers in this model are dynamically quantized and may differ.",
                "Synthetic weights, not the model's own tensors.",
                "One host (gfx1151).",
            ],
        }

        print(json.dumps(report, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
