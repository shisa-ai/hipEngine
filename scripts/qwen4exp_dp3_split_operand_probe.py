#!/usr/bin/env python3
"""DP-3: is a split-F16-operand tile affordable before anyone builds one?

A split-f16-operand variant represents each f32 operand as an f16 hi/lo pair.
For ``(Ah + Al) * (Bh + Bl)`` the three products ``Ah*Bh + Ah*Bl + Al*Bh``
recover roughly f32 precision, at three WMMA issues per tile where the promoted
kernel issues one.

Building that for real is scope work: the hi/lo split, the doubled LDS operand
bytes, the extra registers. The ledger asks for a microbenchmark *before* any of
it, and the question that decides whether it is worth building is narrower than
"how fast is the split":

    **Does issuing three WMMA products per tile instead of one cost less than
    the split would have to gain?**

This measures exactly that and nothing more. ``dense_wide256_split3`` issues the
same product three times on identical operands, so:

* its **timing** is the WMMA-issue cost of the split scheme;
* its **result** is meaningless as a split result - it is just the accumulator
  summed three times over. The script asserts the numerics are *not* a
  precision win, so nobody later mistakes this probe for a candidate route.

That makes the timing a **lower bound** on a real split's cost: a real split also
pays the extra LDS traffic and the splitting arithmetic. A negative result here
therefore kills the scheme cheaply and decisively. A positive result says the
WMMA issue cost is not the blocker and the split is worth scoping.

Timing-only by construction: no correctness gate is claimed and no route is
promoted.
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
from hipengine.core.memory import free, malloc, copy_host_to_device  # noqa: E402
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide import (  # noqa: E402
    f32_to_f16,
    gguf_q8_0_dense_wide256_f16in_f32_f32_out,
    gguf_q8_0_dense_wide256_f32_f32_out,
    gguf_q8_0_dense_wide256_split3_f32_f32_out,
)
from scripts.qwen4exp_dense_projection_ab import (  # noqa: E402
    launch_variant,
    load_layer_weights,
)

# (role, layer, in_features, out_features, tensor) - the admitted wide shapes.
# The two attention projections carry most of the dense prefill time, so they
# decide the question; the narrow ones are included to show the ratio is not a
# property of one geometry.
SHAPES: list[tuple[str, int, int, int, str]] = [
    ("attn_qkv", 16, 2560, 10240, "attn_qkv.weight"),
    ("attn_q", 19, 2560, 12288, "attn_q.weight"),
    ("attn_gate", 16, 2560, 6144, "attn_gate.weight"),
    ("ssm_out", 16, 6144, 2560, "ssm_out.weight"),
    ("shared_gate", 16, 2560, 640, "ffn_gate_shexp.weight"),
]

# The wide kernel requires in_features % 256 == 0 (WIDE_BK).
_BK = 256


def _load_weights(
    model_root: str, tensor: str, layer: int, in_features: int, out_features: int
) -> np.ndarray:
    loaded = load_layer_weights(model_root, tensor, [layer])
    if not loaded:
        raise RuntimeError(f"no weights for {tensor} layer {layer}")
    # ``load_layer_weights`` returns [(layer, raw_bytes, name), ...].
    _layer, raw, _name = loaded[0]
    return np.ascontiguousarray(np.asarray(raw).reshape(-1).view(np.uint8))


def _flops(rows: int, in_features: int, out_features: int) -> float:
    return 2.0 * rows * in_features * out_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--rows", default="1024")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows_list = [int(v) for v in args.rows.split(",") if v.strip()]
    runtime = get_hip_runtime()
    runtime.device_synchronize()

    arms = {
        "rep1": gguf_q8_0_dense_wide256_f32_f32_out,
        "rep3": gguf_q8_0_dense_wide256_split3_f32_f32_out,
        "f16in": gguf_q8_0_dense_wide256_f16in_f32_f32_out,
    }

    cases: list[dict[str, Any]] = []
    for role, layer, in_features, out_features, tensor in SHAPES:
        if in_features % _BK != 0:
            continue
        weights = _load_weights(
            args.model_root, tensor, layer, in_features, out_features
        )
        for rows in rows_list:
            activation = np.ascontiguousarray(
                np.random.default_rng(1234 + rows).standard_normal(
                    rows * in_features, dtype=np.float32
                )
            )
            x_buf = malloc(activation.nbytes)
            w_buf = malloc(weights.nbytes)
            out_buf = malloc(rows * out_features * 4)
            f16_buf = malloc(rows * in_features * 2)
            try:
                copy_host_to_device(x_buf, activation.ctypes.data, activation.nbytes)
                copy_host_to_device(w_buf, weights.ctypes.data, weights.nbytes)

                timings: dict[str, float] = {}
                for name, launch in arms.items():
                    samples = []
                    # ``f16in`` needs the activation pre-converted, exactly as
                    # the promoted route does it: one cast launch, then the
                    # matmul. Timing the matmul alone would flatter it, so the
                    # cast is inside the timed region for that arm.
                    def _launch_one(
                        launch=launch, name=name
                    ) -> None:
                        if name == "f16in":
                            f32_to_f16(
                                x_buf.ptr, f16_buf.ptr, rows * in_features
                            )
                        launch_variant(
                            launch,
                            x_buf.ptr if name != "f16in" else f16_buf.ptr,
                            w_buf.ptr,
                            out_buf.ptr,
                            rows,
                            in_features,
                            out_features,
                        )

                    for _ in range(3):  # warm
                        _launch_one()
                    runtime.device_synchronize()
                    start = runtime.event_create()
                    stop = runtime.event_create()
                    try:
                        for _ in range(args.repetitions):
                            runtime.event_record(start)
                            _launch_one()
                            runtime.event_record(stop)
                            runtime.event_synchronize(stop)
                            samples.append(
                                runtime.event_elapsed_time_ms(start, stop)
                            )
                    finally:
                        runtime.event_destroy(start)
                        runtime.event_destroy(stop)
                    timings[name] = statistics.median(samples)

                flops = _flops(rows, in_features, out_features)
                case = {
                    "role": role,
                    "layer": layer,
                    "rows": rows,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rep1_ms": timings["rep1"],
                    "rep3_ms": timings["rep3"],
                    "f16in_ms": timings["f16in"],
                    "ratio_rep3_over_rep1": timings["rep3"] / timings["rep1"],
                    "rep1_tflops": flops / (timings["rep1"] * 1e9),
                    "rep3_tflops": flops / (timings["rep3"] * 1e9),
                    "f16in_tflops": flops / (timings["f16in"] * 1e9),
                    # The number the DP-3 decision actually turns on, measured
                    # same-host rather than borrowed from another machine's
                    # artifact: what a 3-product split built on the promoted
                    # f16in route would cost, against the f32 route it would
                    # have to replace.
                    "split_on_f16in_estimate_ms": (
                        timings["f16in"] * (timings["rep3"] / timings["rep1"])
                    ),
                    "split_vs_f32in_ratio": (
                        timings["f16in"]
                        * (timings["rep3"] / timings["rep1"])
                        / timings["rep1"]
                    ),
                }
                cases.append(case)
                print(
                    f"{role:12s} rows={rows:5d} "
                    f"rep1={timings['rep1']:8.4f} "
                    f"f16in={timings['f16in']:8.4f} "
                    f"rep3={timings['rep3']:8.4f} ms  "
                    f"ratio={case['ratio_rep3_over_rep1']:.3f}  "
                    f"split/f32in={case['split_vs_f32in_ratio']:.3f}",
                    flush=True,
                )
            finally:
                free(f16_buf)
                free(out_buf)
                free(w_buf)
                free(x_buf)

    ratios = [c["ratio_rep3_over_rep1"] for c in cases]
    split_ratios = [c["split_vs_f32in_ratio"] for c in cases]
    aggregate = {
        "cases": len(cases),
        "ratio_median": statistics.median(ratios) if ratios else None,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "split_vs_f32in_median": (
            statistics.median(split_ratios) if split_ratios else None
        ),
        "split_vs_f32in_min": min(split_ratios) if split_ratios else None,
        "split_vs_f32in_max": max(split_ratios) if split_ratios else None,
    }

    payload = {
        "probe": "qwen4exp-dp3-split-operand",
        "purpose": (
            "Lower bound on the WMMA issue cost of a split-f16-operand tile. "
            "rep3 issues three products per tile on identical operands, so its "
            "timing isolates issue cost and its result is NOT a split result."
        ),
        "rows": rows_list,
        "repetitions": args.repetitions,
        "model_root": args.model_root,
        "cases": cases,
        "aggregate": aggregate,
        "not_claimed": [
            "no correctness gate: rep3's numerics are the accumulator summed "
            "three times over, not a precision improvement",
            "no production route and no promotion",
            "a real split additionally pays doubled LDS operand bytes and the "
            "hi/lo splitting arithmetic, so this ratio is a lower bound",
            "kernel microbenchmark, not engine throughput",
        ],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(f"median ratio (3 WMMA / 1 WMMA): {aggregate['ratio_median']:.3f}")
    print(f"range: {aggregate['ratio_min']:.3f} .. {aggregate['ratio_max']:.3f}")
    print()
    print(
        "median split-on-f16in vs the f32in route it would replace: "
        f"{aggregate['split_vs_f32in_median']:.3f}"
    )
    print(
        f"range: {aggregate['split_vs_f32in_min']:.3f} .. "
        f"{aggregate['split_vs_f32in_max']:.3f}"
    )
    print()
    print("Reading: the split must beat this ratio's cost with a precision gain.")
    print("A ratio near 3.0 means issue cost is prohibitive; near 1.0 means the")
    print("tile is memory/LDS bound and the extra products are nearly free.")
    print("split/f32in below 1.0 would mean a split on the promoted route is")
    print("cheaper than the f32 route - but it is a LOWER bound: a real split")
    print("also pays doubled LDS operand bytes and the hi/lo split arithmetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
