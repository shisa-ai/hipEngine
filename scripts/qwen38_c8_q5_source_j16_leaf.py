#!/usr/bin/env python3
"""Leaf A/B: forced-J16 minitile versus the J32 owner on all 48 ssm_out weights.

C8-P2 Q5 candidate. Both arms share the same d4s4_f32_kmajor producer, route,
and per-(row, output) accumulation order, so outputs must be bf16 bit-identical;
the only variable is the launch geometry (grid.y minitiles at rows=32).
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import numpy as np


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _kl(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = _bf16_f32(reference).astype(np.float64)
    act = _bf16_f32(actual).astype(np.float64)
    ref = ref - ref.max(axis=1, keepdims=True)
    act = act - act.max(axis=1, keepdims=True)
    ref_p = np.exp(ref)
    ref_p /= ref_p.sum(axis=1, keepdims=True)
    act_p = np.exp(act)
    act_p /= act_p.sum(axis=1, keepdims=True)
    return float(np.sum(ref_p * (np.log(ref_p) - np.log(act_p))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.benchmark.provenance import detect_device_name
    from hipengine.kernels.backends import detect_hip_target_arches
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
        build_gguf_q5_k_source_mmq_prefill,
        gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out,
        gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out,
        gguf_q8_1_d4s4_f32_quantize_bf16_kmajor,
        q8_1_d4s4_f32_kmajor_nbytes,
    )

    rows = args.rows
    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    names = tuple(
        tensor.name
        for tensor in reader.info.tensors
        if tensor.name.endswith(".ssm_out.weight")
    )
    if len(names) != 48:
        raise ValueError(f"expected 48 ssm_out weights, found {len(names)}")

    library = build_gguf_q5_k_source_mmq_prefill(load=True)

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    def download(buffer, shape):
        host = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), buffer, runtime=runtime)
        return host

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

    results = []
    j32_wins = 0
    j16_wins = 0
    for weight_index, name in enumerate(names):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        raw_device = upload(raw)
        buffers = [raw_device]
        try:
            rng = np.random.default_rng(2_026_090_400 + weight_index * 100 + rows)
            x = _bf16_bits(
                rng.normal(0.0, 0.2, size=(rows, 6_144)).astype(np.float32)
            )
            x_device = upload(x)
            q8_device = malloc(
                q8_1_d4s4_f32_kmajor_nbytes(rows, 6_144), runtime=runtime
            )
            j32_device = malloc(rows * 5_120 * 2, runtime=runtime)
            j16_device = malloc(rows * 5_120 * 2, runtime=runtime)
            buffers.extend((x_device, q8_device, j32_device, j16_device))

            gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(
                x_device.ptr,
                q8_device.ptr,
                rows,
                6_144,
                library=library,
                runtime=runtime,
            )

            def j32() -> None:
                gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
                    q8_device.ptr,
                    raw_device.ptr,
                    j32_device.ptr,
                    rows,
                    6_144,
                    5_120,
                    library=library,
                    runtime=runtime,
                )

            def j16() -> None:
                gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
                    q8_device.ptr,
                    raw_device.ptr,
                    j16_device.ptr,
                    rows,
                    6_144,
                    5_120,
                    library=library,
                    runtime=runtime,
                )

            # Correctness first: bit-exactness of the forced-J16 output.
            j32()
            j16()
            j32_out = download(j32_device, (rows, 5_120))
            j16_out = download(j16_device, (rows, 5_120))
            mismatches = int(np.count_nonzero(j32_out != j16_out))
            exact = mismatches == 0

            j32_samples: list[float] = []
            j16_samples: list[float] = []
            for sample in range(args.samples):
                # Counterbalance arm order per weight and per sample.
                if (weight_index + sample) % 2 == 0:
                    j32_samples.append(event_ms(j32))
                    j16_samples.append(event_ms(j16))
                else:
                    j16_samples.append(event_ms(j16))
                    j32_samples.append(event_ms(j32))
            for _ in range(args.warmups):
                j32()
                j16()

            j32_med = statistics.median(j32_samples)
            j16_med = statistics.median(j16_samples)
            if j16_med < j32_med:
                j16_wins += 1
            elif j32_med < j16_med:
                j32_wins += 1
            results.append(
                {
                    "weight": name,
                    "rows": rows,
                    "bf16_bit_exact": exact,
                    "bf16_mismatches": mismatches,
                    "kl_vs_j32": _kl(j32_out, j16_out) if not exact else 0.0,
                    "j32_ms": j32_med,
                    "j16_ms": j16_med,
                    "j32_samples_ms": j32_samples,
                    "j16_samples_ms": j16_samples,
                    "j16_vs_j32_ratio": j16_med / j32_med,
                    "order_counterbalanced": True,
                }
            )
        finally:
            for buffer in buffers:
                try:
                    free(buffer, runtime=runtime)
                except Exception:
                    pass

    exact_all = all(r["bf16_bit_exact"] for r in results)
    j32_sum = sum(r["j32_ms"] for r in results)
    j16_sum = sum(r["j16_ms"] for r in results)
    payload = {
        "schema": 1,
        "kind": "w7900_qwen38_q4km_c8_q5_source_tile_leaf",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "candidate",
        "hardware": {
            "device": detect_device_name() or "unknown",
            "arch": ",".join(detect_hip_target_arches()) or "unknown",
        },
        "model": args.model,
        "protocol": {
            "rows": rows,
            "weights": 48,
            "samples": args.samples,
            "burst": args.burst,
            "warmups": args.warmups,
            "timing": "HIP events, consumer-only bursts (shared producer excluded)",
            "arm_order": "counterbalanced per weight and sample",
            "producer": "q8_1_d4s4_f32_quantize_bf16_kmajor (both arms)",
        },
        "correctness": {
            "all_bf16_bit_exact": exact_all,
            "declared_contract": "forced-J16 bit-identical to J32 owner (same accumulation order)",
        },
        "summary": {
            "j32_sum_ms": j32_sum,
            "j16_sum_ms": j16_sum,
            "j16_vs_j32_ratio": j16_sum / j32_sum,
            "j16_wins": j16_wins,
            "j32_wins": j32_wins,
            "ties": 48 - j16_wins - j32_wins,
        },
        "weights": results,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(
        f"exact={exact_all} j32_sum={j32_sum:.3f} j16_sum={j16_sum:.3f} "
        f"ratio={j16_sum / j32_sum:.4f} j16_wins={j16_wins}/48"
    )
    return 0 if exact_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
