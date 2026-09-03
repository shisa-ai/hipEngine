#!/usr/bin/env python3
"""Leaf A/B: direct-load dp4a Q5_K MMQ32 versus the raw mmq32 owner.

C8-P2 residual screen (iteration 38). The iter37 inspection found both
retained Q5 owners at ~30 GB/s effective (schedule-bound, not a DRAM
roofline); this screen times the new direct-load dp4a kernel (no LDS
staging, no per-k barriers) against the retained raw mmq32 d4s4 owner on
the actual 48 ssm_out weights at the production R24/R32 shapes, in
alternating order. Both arms include the q8_1 quantize launch
(operation-complete convention). The arms share the same integer-dp4a
arithmetic class and the min-correction contract, so outputs should agree
to fp32 accumulation-order noise; mismatches are reported, not gated.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import numpy as np


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rows", type=int, nargs="+", default=(24, 32))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from hipengine.benchmark.provenance import detect_device_name
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.backends import detect_hip_target_arches
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
        build_gguf_k_mmq_prefill,
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_f32_out,
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
        gguf_q8_1_d4s4_f32_quantize_bf16,
    )
    from hipengine.loading.gguf import GGUFReader

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    names = [
        tensor.name
        for tensor in reader.info.tensors
        if tensor.name.endswith(".ssm_out.weight")
    ]
    if not names:
        raise ValueError("no ssm_out tensors found in the model")
    cases = []
    for name in names:
        info = reader.tensor_info(name)
        if info.ggml_type_name != "Q5_K":
            continue
        cases.append((name, int(info.shape[0]), int(info.shape[1])))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no Q5_K ssm_out tensors found")

    library = build_gguf_k_mmq_prefill(load=True)

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    def event_ms(function) -> float:
        start = runtime.event_create()
        stop = runtime.event_create()
        try:
            runtime.event_record(start)
            for _ in range(args.burst):
                function()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            return float(runtime.event_elapsed_time_ms(start, stop)) / args.burst
        finally:
            runtime.event_destroy(stop)
            runtime.event_destroy(start)

    device_name = detect_device_name()
    arches = detect_hip_target_arches()
    all_results = []
    for rows in args.rows:
        raw_wins = 0
        direct_wins = 0
        per_case = []
        for index, (name, out_features, in_features) in enumerate(cases):
            weight = np.ascontiguousarray(reader.tensor_data(name))
            weight_device = upload(weight)
            buffers = [weight_device]
            try:
                rng = np.random.default_rng(
                    2_026_090_500 + index * 100 + rows
                )
                x_bits = (
                    rng.normal(0.0, 0.2, size=(rows, in_features))
                    .astype(np.float32)
                    .view(np.uint32)
                )
                x_bits = ((x_bits + np.uint32(0x7FFF)) +
                          ((x_bits >> np.uint32(16)) & np.uint32(1)))
                x = np.ascontiguousarray(
                    (x_bits >> np.uint32(16)).astype(np.uint16)
                )
                x_device = upload(x)
                xq_device = malloc(
                    rows * (in_features // 128) * 160, runtime=runtime
                )
                raw_device = malloc(rows * out_features * 4, runtime=runtime)
                direct_device = malloc(rows * out_features * 4, runtime=runtime)
                buffers.extend((x_device, xq_device, raw_device, direct_device))

                def quantize() -> None:
                    gguf_q8_1_d4s4_f32_quantize_bf16(
                        x_device.ptr,
                        xq_device.ptr,
                        rows,
                        in_features,
                        library=library,
                        runtime=runtime,
                    )

                def raw_arm() -> None:
                    quantize()
                    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(
                        xq_device.ptr,
                        weight_device.ptr,
                        raw_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                def direct_arm() -> None:
                    quantize()
                    gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_f32_out(
                        xq_device.ptr,
                        weight_device.ptr,
                        direct_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )

                raw_arm()
                direct_arm()
                runtime.device_synchronize()
                raw_out = np.empty((rows, out_features), dtype=np.float32)
                direct_out = np.empty_like(raw_out)
                copy_device_to_host(
                    host_array_ptr(raw_out), raw_device, runtime=runtime
                )
                copy_device_to_host(
                    host_array_ptr(direct_out), direct_device, runtime=runtime
                )
                denom = np.maximum(np.abs(raw_out), 1e-6)
                max_rel = float(
                    np.max(np.abs(direct_out - raw_out) / denom)
                )
                finite = bool(np.isfinite(direct_out).all())

                raw_times = []
                direct_times = []
                order = [("raw", raw_arm), ("direct", direct_arm)] * args.samples
                if index % 2 == 1:
                    order = list(reversed(order))
                for arm_name, arm in order:
                    for _ in range(args.warmups):
                        arm()
                    runtime.device_synchronize()
                    ms = event_ms(arm)
                    (raw_times if arm_name == "raw" else direct_times).append(ms)
                raw_med = statistics.median(raw_times)
                direct_med = statistics.median(direct_times)
                winner = "direct" if direct_med < raw_med else "raw"
                if winner == "direct":
                    direct_wins += 1
                else:
                    raw_wins += 1
                per_case.append(
                    {
                        "tensor": name,
                        "rows": rows,
                        "out_features": out_features,
                        "in_features": in_features,
                        "raw_ms": raw_med,
                        "direct_ms": direct_med,
                        "ratio_raw_over_direct": raw_med / direct_med,
                        "winner": winner,
                        "max_rel_vs_raw": max_rel,
                        "finite": finite,
                    }
                )
                print(
                    f"[{index:2d}] {name:28s} R{rows} raw {raw_med:.4f} ms  "
                    f"direct {direct_med:.4f} ms  ratio {raw_med/direct_med:.3f}  "
                    f"max_rel {max_rel:.2e}  {winner}"
                )
            finally:
                for buffer in reversed(buffers):
                    free(buffer, runtime=runtime)
        all_results.append(
            {
                "rows": rows,
                "cases": per_case,
                "raw_wins": raw_wins,
                "direct_wins": direct_wins,
                "geomean_ratio": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [c["ratio_raw_over_direct"] for c in per_case]
                            )
                        )
                    )
                ),
            }
        )
        print(
            f"rows={rows}: direct wins {direct_wins}/{len(cases)}, "
            f"geomean ratio {all_results[-1]['geomean_ratio']:.3f}"
        )

    payload = {
        "schema": 1,
        "kind": "w7900_qwen38_q4km_c8_p2_q5_direct_dp4a_leaf_screen",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "status": "completed",
        "hardware": {
            "physical_host": "epyc",
            "gpu": device_name,
            "target_arch": (arches[0] if arches else "gfx1100"),
        },
        "model": args.model,
        "protocol": {
            "arms": "raw mmq32 d4s4 (retained owner) vs direct-load dp4a candidate; both include the q8_1 quantize launch (operation-complete)",
            "samples": args.samples,
            "burst": args.burst,
            "warmups": args.warmups,
            "order": "alternating by weight index",
            "weights": "actual model ssm_out Q5_K tensors",
        },
        "results": all_results,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
