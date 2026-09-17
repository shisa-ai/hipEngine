#!/usr/bin/env python3
"""Does hoisting the wide kernel's activation conversion out of the K loop pay?

The wide-row Q8_0 prefill GEMM converts its activation from f32 to f16 during
LDS staging, inside the K loop. That conversion is repeated once per column
block, so an activation element is converted ``ceil(N / BN)`` times, and the f32
activation is re-read from global memory the same number of times. The kernel
file's ``activation_path_probe`` measured the *load path* alone with
reinterpreted (invalid) values: 2.541 ms converting in the loop against 1.303 ms
reading a pre-converted f16 activation. That is a hypothesis about the load
path, not a route: it excludes the conversion pass, its extra memory traffic,
its launch and its scratch buffer.

This measures the real thing per admitted shape class:

    f32in       the promoted route: one launch, conversion inside the K loop
    cast        the conversion pass alone, ``rows * K`` f32 -> f16
    f16in       the same tile with a pre-converted activation, no conversion
    cast+f16in  the composite a shipped route would pay: conversion pass, then
                the f16-input matmul, with a reused scratch buffer

``cast+f16in`` is the number to compare against ``f32in``. It is a sequence of
two launches on one stream, so the second launch's cost is included and the
scratch is allocated once and reused across repetitions (a shipped route must not
allocate per call).

The composite also depends on how many consumers read the same activation. The
admitted shapes include pairs that share one activation tensor - ``attn_qkv``
with ``attn_gate``, and ``shared_gate`` with ``shared_up`` - so the harness
reports the composite at reuse 1 and at reuse 2 for every shape.

Correctness is checked, not assumed: the f16-input variant must produce
bit-identical output to the f32-input variant, because both put the same f16
bytes in LDS for the same values.

    python3 scripts/qwen4exp_dense_wide_f16_activation.py \\
        --model-root <gguf> --output benchmarks/results/<dir>/artifact.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import malloc, free, copy_host_to_device  # noqa: E402
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide import (  # noqa: E402
    build_gguf_q8_0_dense_wide,
    f32_to_f16,
)
from scripts.qwen4exp_dense_projection_ab import (  # noqa: E402
    load_layer_weights,
    launch_variant,
)

# The dense Q8_0 linears the shipped default admits at layers 16-47, read from
# the launch census of the promoted route
# (benchmarks/results/2026-09-17-q8-dense-default-path-census/census.json):
# (role, the layer whose weights the packet reads, in_features, out_features,
# how many layers of the window carry the role, the Q8_0 tensor with that
# geometry). The geometry, the layer and the count all come from the census, and
# the tensor is resolved from the committed model-shape table, so the packet
# reads real Q8_0 blocks of the right size at a layer where the route ran. The
# full-attention roles sit on the every-fourth-layer schedule, so their first
# layer is 19 rather than 16. The table is asserted against the census in
# tests/test_unit_qwen4exp_dense_wide_f16_activation.py.
ADMITTED_SHAPES: tuple[tuple[str, int, int, int, int, str], ...] = (
    ("attn_qkv", 16, 2560, 10240, 24, "attn_qkv.weight"),
    ("attn_q", 19, 2560, 12288, 8, "attn_q.weight"),
    ("attn_k", 19, 2560, 512, 8, "attn_k.weight"),
    ("attn_v", 19, 2560, 512, 8, "attn_v.weight"),
    ("attn_gate", 16, 2560, 6144, 24, "attn_gate.weight"),
    ("attn_output", 19, 6144, 2560, 8, "attn_output.weight"),
    ("ssm_out", 16, 6144, 2560, 24, "ssm_out.weight"),
    ("shared_gate", 16, 2560, 640, 32, "ffn_gate_shexp.weight"),
    ("shared_up", 16, 2560, 640, 32, "ffn_up_shexp.weight"),
    ("shared_down", 16, 640, 2560, 32, "ffn_down_shexp.weight"),
    ("hc_attn_down", 16, 10240, 320, 32, "hc_attn_down.weight"),
    ("hc_ffn_down", 16, 10240, 320, 32, "hc_ffn_down.weight"),
)

MODEL_SHAPES = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/2026-09-17-qwen4exp-per-role-cost/model-shapes.json"
)

Q8_0_ROW_DIVISOR = 32
Q8_0_ROW_BYTES = 34

# Roles whose input is one live activation tensor produced once, so a hoisted
# cast is paid once for the pair. Everything else pays its own cast. This is the
# only sharing the packet assumes; it does not assume cross-layer or cross-chunk
# reuse.
SHARED_INPUT_PAIRS: tuple[tuple[str, str], ...] = (
    ("attn_qkv", "attn_gate"),
    ("shared_gate", "shared_up"),
)

# The canonical prefill runs one 1024-token chunk at a time, so every admitted
# shape is launched once per layer per chunk.
CHUNK_ROWS = 1024
CHUNKS_PER_PREFILL = 4


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the packet implies for one 4096-token prefill.

    The per-launch times are the measured ones at ``CHUNK_ROWS``; the launch
    count is the census's per-layer count times the chunk count. A shape whose
    input is shared with another admitted role pays half a cast. This is an
    estimate from measured per-launch times, not an engine measurement: it holds
    only if a launch at this shape costs the same inside the engine, which the
    per-shape numbers alone cannot establish.
    """

    paired = {role for pair in SHARED_INPUT_PAIRS for role in pair}
    by_role = {row["role"]: row for row in rows if row["rows"] == CHUNK_ROWS}
    missing = [role for role, *_ in ADMITTED_SHAPES if role not in by_role]
    if missing:
        raise SystemExit(f"aggregate needs rows={CHUNK_ROWS} for {missing}")

    total_f32 = total_composite = 0.0
    per_role: list[dict[str, Any]] = []
    for role, _layer, _k, _n, layers, _tensor in ADMITTED_SHAPES:
        row = by_role[role]
        launches = layers * CHUNKS_PER_PREFILL
        share = 0.5 if role in paired else 1.0
        composite = row["f16in_ms"] + share * row["cast_ms"]
        total_f32 += launches * row["f32in_ms"]
        total_composite += launches * composite
        per_role.append(
            {
                "role": role,
                "launches": launches,
                "cast_share": share,
                "f32in_ms_total": launches * row["f32in_ms"],
                "composite_ms_total": launches * composite,
            }
        )
    saving = total_f32 - total_composite
    return {
        "rows": CHUNK_ROWS,
        "chunks": CHUNKS_PER_PREFILL,
        "shared_input_pairs": [list(pair) for pair in SHARED_INPUT_PAIRS],
        "f32in_ms_total": total_f32,
        "composite_ms_total": total_composite,
        "saving_ms": saving,
        "saving_pct_of_dense_route": 100.0 * saving / total_f32,
        "per_role": per_role,
    }


def _load_weights(model_root: str, tensor: str, layer: int, in_features: int, out_features: int) -> np.ndarray:
    """Raw Q8_0 bytes for one tensor, with the geometry checked against them."""

    loaded = load_layer_weights(model_root, tensor, [layer])
    raw = loaded[0][1]
    expected = out_features * (in_features // Q8_0_ROW_DIVISOR) * Q8_0_ROW_BYTES
    if raw.nbytes != expected:
        raise SystemExit(
            f"{tensor}: {raw.nbytes} bytes for K={in_features} N={out_features}; "
            f"expected {expected}"
        )
    return raw


def _check_geometry(tensor: str, in_features: int, out_features: int) -> None:
    """The committed shape table must agree with the census geometry."""

    table = json.loads(MODEL_SHAPES.read_text())["tensors"]
    info = table.get(tensor)
    if info is None:
        raise SystemExit(f"{tensor} is not in {MODEL_SHAPES}")
    if (info["in_features"], info["out_features"]) != (in_features, out_features):
        raise SystemExit(
            f"{tensor}: shape table says {info['in_features']}x{info['out_features']}, "
            f"the census says {in_features}x{out_features}"
        )
    if info.get("quant") != "Q8_0":
        raise SystemExit(f"{tensor} is {info.get('quant')}, not Q8_0")


def _time_sequence(
    make_launches: Any,
    runtime: Any,
    repetitions: int,
) -> float:
    """Median ms per repetition for a launch sequence.

    ``make_launches(rep)`` returns the ``(callable, args)`` pairs for one
    repetition, so a caller can rotate the weight buffer it reads and keep the
    reads out of cache.
    """

    samples: list[float] = []
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        for rep in range(repetitions):
            runtime.event_record(start)
            for fn, args in make_launches(rep):
                fn(*args)
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            samples.append(runtime.event_elapsed_time_ms(start, stop))
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
    return statistics.median(samples)


def _max_abs_diff(left_buffer: Any, right_buffer: Any, count: int, runtime: Any) -> float:
    """Max |a - b| between two device buffers, read back through the host."""

    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    left = np.zeros(count, dtype=np.float32)
    right = np.zeros(count, dtype=np.float32)
    copy_device_to_host(host_array_ptr(left), left_buffer, count * 4)
    copy_device_to_host(host_array_ptr(right), right_buffer, count * 4)
    runtime.device_synchronize()
    return float(np.max(np.abs(left - right))) if count else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="512,1024")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--weight-buffers",
        type=int,
        default=8,
        help="Distinct copies of the weight tensor to rotate across repetitions. "
        "One copy is re-read from L2 every repetition and measures a cached "
        "kernel; the engine reads weights that other kernels have evicted, so "
        "the default rotates enough copies to exceed the GPU's cache.",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    rows_list = [int(part) for part in args.rows.split(",") if part.strip()]
    runtime = get_hip_runtime()
    library = build_gguf_q8_0_dense_wide(load=True)

    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide

    f32in = wide._WRAPPERS["dense_wide256_f32_f32_out"]
    f16in = wide._WRAPPERS["dense_wide256_f16in_f32_f32_out"]

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []

    for rows in rows_list:
        for role, layer, in_features, out_features, layers, tensor in ADMITTED_SHAPES:
            _check_geometry(tensor, in_features, out_features)
            raw = _load_weights(args.model_root, tensor, layer, in_features, out_features)
            weight_buffers = [malloc(raw.nbytes) for _ in range(args.weight_buffers)]
            for buffer in weight_buffers:
                copy_host_to_device(buffer, raw.ctypes.data, raw.nbytes)

            x = rng.standard_normal((rows, in_features), dtype=np.float32)
            x_ptr = malloc(x.nbytes)
            copy_host_to_device(x_ptr, x.ctypes.data, x.nbytes)

            n_act = rows * in_features
            # +128 bytes of slack: the f16 tile reads 16-byte vectors and the
            # converter writes whole vectors, so a page-committed buffer with
            # padding is what a shipped route would carry.
            scratch = malloc(n_act * 2 + 128)
            out_a = malloc(rows * out_features * 4)
            out_b = malloc(rows * out_features * 4)

            args_f32 = lambda w: (x_ptr.ptr, w.ptr, out_a.ptr, rows, in_features, out_features)
            args_f16 = lambda w: (scratch.ptr, w.ptr, out_b.ptr, rows, in_features, out_features)

            cast_args = (x_ptr.ptr, scratch.ptr, n_act)
            t_cast = _time_sequence(
                lambda rep: [(lambda *a: f32_to_f16(*a, stream=0), cast_args)],
                runtime,
                args.repetitions,
            )
            t_f32in = _time_sequence(
                lambda rep: [
                    (lambda *a: launch_variant(f32in, *a),
                     args_f32(weight_buffers[rep % len(weight_buffers)]))
                ],
                runtime,
                args.repetitions,
            )
            t_f16in = _time_sequence(
                lambda rep: [
                    (lambda *a: launch_variant(f16in, *a),
                     args_f16(weight_buffers[rep % len(weight_buffers)]))
                ],
                runtime,
                args.repetitions,
            )
            t_composite = _time_sequence(
                lambda rep: [
                    (lambda *a: f32_to_f16(*a, stream=0), cast_args),
                    (lambda *a: launch_variant(f16in, *a),
                     args_f16(weight_buffers[rep % len(weight_buffers)])),
                ],
                runtime,
                args.repetitions,
            )

            # Correctness: the two variants must agree bit for bit. Buffer 0 is
            # the one the f32 arm also reads, so the check is not a rotation
            # artifact.
            launch_variant(f32in, *args_f32(weight_buffers[0]))
            f32_to_f16(*cast_args, stream=0)
            launch_variant(f16in, *args_f16(weight_buffers[0]))
            runtime.device_synchronize()
            max_abs = _max_abs_diff(out_a, out_b, rows * out_features, runtime)

            flops = 2.0 * rows * in_features * out_features
            results.append(
                {
                    "role": role,
                    "tensor": tensor,
                    "weight_layer": layer,
                    "rows": rows,
                    "in_features": in_features,
                    "out_features": out_features,
                    "in_window_layers": layers,
                    "flops": flops,
                    "f32in_ms": t_f32in,
                    "cast_ms": t_cast,
                    "f16in_ms": t_f16in,
                    "composite_ms": t_composite,
                    "composite_reuse2_ms": t_cast / 2.0 + t_f16in,
                    "delta_reuse1_ms": t_f32in - t_composite,
                    "delta_reuse1_pct": 100.0 * (t_f32in - t_composite) / t_f32in,
                    "delta_reuse2_pct": 100.0
                    * (t_f32in - (t_cast / 2.0 + t_f16in))
                    / t_f32in,
                    "f32in_tflops": flops / (t_f32in * 1e9),
                    "f16in_tflops": flops / (t_f16in * 1e9),
                    "weight_bytes": int(raw.nbytes),
                    "weight_buffers": len(weight_buffers),
                    "weight_working_set_bytes": int(raw.nbytes) * len(weight_buffers),
                    "max_abs_diff_f32in_vs_f16in": max_abs,
                    "bit_identical": max_abs == 0.0,
                }
            )

            free(x_ptr)
            free(scratch)
            free(out_a)
            free(out_b)
            for buffer in weight_buffers:
                free(buffer)

    payload = {
        "schema": 1,
        "kind": "qwen4exp_dense_wide_f16_activation",
        "performance_claim": True,
        "question": (
            "For each dense Q8_0 prefill shape the shipped default admits at "
            "layers 16-47, does a conversion pass plus an f16-input tile beat "
            "the tile that converts during LDS staging?"
        ),
        "protocol": {
            "route": "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE, layers 16-47",
            "tile": "dense_wide256 (128 columns x 256 rows, BK 64)",
            "repetitions": args.repetitions,
            "timing": "median of per-repetition CUDA/HIP event pairs; the "
                      "composite is two launches on one stream",
            "scratch": "allocated once per shape and reused across repetitions",
            "weight_buffers": args.weight_buffers,
            "weight_buffers_note": (
                "each repetition reads a different copy of the weight tensor, so "
                "the weight traffic comes from DRAM instead of a cached buffer. "
                "At 1 the packet measures an L2-resident kernel, which is about "
                "four times faster per launch than the same shape in the engine."
            ),
            "weights": "Q8_0 bytes of the tensor carrying each geometry at the "
                       "layer named per shape, resolved from the committed "
                       "model-shape table; geometry set explicitly",
            "rows": rows_list,
        },
        "shapes": results,
        "aggregate": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'role':14s} {'K':>6s} {'N':>6s} {'f32in':>8s} {'cast':>7s} "
          f"{'f16in':>8s} {'comp':>8s} {'d%':>7s} {'d2%':>7s} {'bit':>4s}")
    for r in results:
        print(
            f"{r['role']:14s} {r['in_features']:6d} {r['out_features']:6d} "
            f"{r['f32in_ms']:8.3f} {r['cast_ms']:7.3f} {r['f16in_ms']:8.3f} "
            f"{r['composite_ms']:8.3f} {r['delta_reuse1_pct']:7.1f} "
            f"{r['delta_reuse2_pct']:7.1f} {'ok' if r['bit_identical'] else 'DIFF':>4s}"
        )
    agg = payload["aggregate"]
    print(
        f"\none {CHUNKS_PER_PREFILL}-chunk prefill at rows={CHUNK_ROWS}: "
        f"{agg['f32in_ms_total']:.0f} ms -> {agg['composite_ms_total']:.0f} ms "
        f"({agg['saving_ms']:.0f} ms, {agg['saving_pct_of_dense_route']:.1f}% of the route)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
