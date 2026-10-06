#!/usr/bin/env python3
"""Measure every registered ``dense_wide`` tile family at the packet shape.

The kernel's own comment says of the tile choice: *"The narrower layouts give up
some weight reuse (traffic is M * K * rows / BN) for headroom; which one wins is
a measurement, not an argument."*  No such measurement was recorded: the ledger's
tile sweep (`128x64`, `64x64`, `64x128`, "128x128 optimal in family") is for the
**MMQ** family, and every reference to the ``dense_wide`` tile variants in the
tree is a correctness test, not a throughput comparison.

This closes that gap.  It is a microbenchmark of one GEMM shape, so it ranks
tilings and nothing else - it is not an end-to-end claim and it promotes nothing.

The default packet shape is ``rows 1024, K 2560, N 10240``, which is the shape
``QWEN4EXP-OPTIMIZATION-LEDGER.md`` §4A quotes at 2.573 ms / 20.9 TFLOP/s.

Usage:
    uv run python scripts/qwen4exp_dense_wide_tile_sweep.py
    uv run python scripts/qwen4exp_dense_wide_tile_sweep.py --rows 2048 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw

from tests.test_gpu_gguf_k_gemv import make_q8_0_weight

# The production default, and the family it is compared against.
DEFAULT_VARIANT = "dense_wide256_f32_f32_out"

# (BM, BN, WTM, WTN) per export, from the HIPENGINE_DENSE_WIDE_EXPORT table.
TILE_FAMILY: dict[str, tuple[int, int, int, int]] = {
    "dense_split64x128_f32_f32_out": (64, 128, 32, 32),
    "dense_wide256_f32_f32_out": (128, 256, 64, 64),
    "dense_wide64x256_f32_f32_out": (64, 256, 32, 64),
    "dense_wide128x128_f32_f32_out": (128, 128, 64, 32),
    "dense_wide64x128_f32_f32_out": (64, 128, 32, 32),
    "dense_wide256_padded_f32_f32_out": (128, 256, 64, 64),
    "dense_wide256_padded_f16in_f32_f32_out": (128, 256, 64, 64),
    "dense_wide256_f16in_f32_f32_out": (128, 256, 64, 64),
    "dense_wide128x128_f16in_f32_f32_out": (128, 128, 64, 32),
    "dense_wide256_bf16in_f32_f32_out": (128, 256, 64, 64),
    "dense_wide256_split3_f32_f32_out": (128, 256, 64, 64),
}

# The BF16 arm uses the same 128x256 tile, but its raw activation ABI is
# uint16_t BF16 rather than f16. It remains a timing/qualification probe only.
TIMING_ONLY = frozenset({"dense_wide256_split3_f32_f32_out"})


def _time_variant(
    variant: str,
    wrapper,
    library,
    runtime,
    qweight: np.ndarray,
    x_f32: np.ndarray,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    """Return per-repeat milliseconds and the variant's output."""

    f16in = "_f16in_" in variant
    bf16in = "_bf16in_" in variant
    if f16in and bf16in:
        raise ValueError(f"variant cannot be both f16in and bf16in: {variant}")
    if f16in or bf16in:
        x_dev = malloc(rows * in_features * 2)
        if f16in:
            dw.f32_to_f16(
                host_array_ptr(np.ascontiguousarray(x_f32)),
                x_dev.ptr,
                rows * in_features,
                runtime=runtime,
            )
        else:
            bf16_host = np.ascontiguousarray(
                ((x_f32.view(np.uint32) + 0x7FFF + ((x_f32.view(np.uint32) >> 16) & 1)) >> 16),
                dtype=np.uint16,
            )
            copy_host_to_device(x_dev, host_array_ptr(bf16_host), runtime=runtime)
        runtime.device_synchronize()
    else:
        x_dev = malloc(x_f32.nbytes)


    w_dev = malloc(qweight.nbytes)
    out = np.zeros((rows, out_features), dtype=np.float32)
    out_dev = malloc(out.nbytes)
    try:
        if not f16in and not bf16in:
            copy_host_to_device(x_dev, host_array_ptr(np.ascontiguousarray(x_f32)), runtime=runtime)
        copy_host_to_device(w_dev, host_array_ptr(qweight), runtime=runtime)

        def once() -> None:
            wrapper(
                x_dev.ptr,
                w_dev.ptr,
                out_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        for _ in range(warmup):
            once()
        runtime.device_synchronize()

        samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            once()
            runtime.device_synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)

        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return samples, out
    finally:
        for buffer in (x_dev, w_dev, out_dev):
            free(buffer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--out-features", type=int, default=10240)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows, k, n = args.rows, args.in_features, args.out_features
    if k % 64 != 0:
        print(f"in_features {k} must be a multiple of 64", file=sys.stderr)
        return 2

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)

    rng = np.random.default_rng(0xD3E5E)
    qweight = make_q8_0_weight(n, k)
    x = rng.standard_normal((rows, k), dtype=np.float32)

    flops = 2.0 * rows * k * n

    print(
        f"shape rows={rows} K={k} N={n}  "
        f"({flops / 1e9:.2f} GFLOP, {qweight.nbytes / 1e6:.1f} MB Q8_0 weights)\n"
    )

    reference: np.ndarray | None = None
    records = []
    for variant in dw._VARIANTS:
        wrapper = dw._WRAPPERS[variant]
        bm, bn, wtm, wtn = TILE_FAMILY[variant]
        try:
            samples, out = _time_variant(
                variant,
                wrapper,
                library,
                runtime,
                qweight,
                x,
                rows,
                k,
                n,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except Exception as exc:  # noqa: BLE001 - a tile may reject the shape
            print(f"  {variant:38s} SKIPPED: {type(exc).__name__}: {exc}")
            continue

        best = min(samples)
        tflops = flops / (best / 1e3) / 1e12
        timing_only = variant in TIMING_ONLY
        if variant == DEFAULT_VARIANT:
            reference = out
        max_diff = None
        if reference is not None and not timing_only:
            max_diff = float(np.max(np.abs(out - reference)))

        records.append(
            {
                "variant": variant,
                "tile": {"BM": bm, "BN": bn, "WTM": wtm, "WTN": wtn},
                "best_ms": best,
                "median_ms": statistics.median(samples),
                "spread_pct": 100.0 * (max(samples) - min(samples)) / min(samples),
                "tflops": tflops,
                "timing_only": timing_only,
                "max_diff_vs_default": max_diff,
            }
        )

    base = next(
        (r for r in records if r["variant"] == DEFAULT_VARIANT), None
    )
    print(
        f"{'variant':38s} {'BM':>4} {'BN':>4} {'ms':>8} {'TFLOP/s':>8} "
        f"{'vs default':>11} {'max|d|':>10}"
    )
    for r in sorted(records, key=lambda r: -r["tflops"]):
        rel = ""
        if base and r["variant"] != DEFAULT_VARIANT:
            rel = f"{100.0 * (r['best_ms'] / base['best_ms'] - 1):+10.2f}%"
        diff = "" if r["max_diff_vs_default"] is None else f"{r['max_diff_vs_default']:.3e}"
        if r["variant"] == DEFAULT_VARIANT:
            rel = f"{'baseline':>11}"
        print(
            f"{r['variant']:38s} {r['tile']['BM']:4d} {r['tile']['BN']:4d} "
            f"{r['best_ms']:8.4f} {r['tflops']:8.2f} {rel:>11} {diff:>10}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "shape": {"rows": rows, "in_features": k, "out_features": n},
                    "flops": flops,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "default_variant": DEFAULT_VARIANT,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
