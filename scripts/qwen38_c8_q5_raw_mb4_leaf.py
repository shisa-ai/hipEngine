#!/usr/bin/env python3
"""Leaf A/B: raw MMQ32 candidates versus the registered mb2 owner at R24.

C8-P2 Q5 latency probe (iterations 27-28). Arms: forced minblocks=4
residency hint (iteration 27, measured neutral) and the software-pipelined
variant (iteration 28: prefetch next K32 tile into registers, one barrier
per iteration). All arms share the same kernel math and per-(row, output)
accumulation order, so outputs must be bf16 bit-identical.
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
    parser.add_argument("--rows", type=int, default=24)
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
        build_gguf_k_mmq_prefill,
        gguf_q5_k_mmq32_mb4_q8_1_d4s4_f32_bf16_bf16_out,
        gguf_q5_k_mmq32_pipe_q8_1_d4s4_f32_bf16_bf16_out,
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
        gguf_q8_1_d4s4_f32_quantize_bf16,
        q8_1_d4s4_f32_nbytes,
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

    library = build_gguf_k_mmq_prefill(load=True)

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
    mb2_wins = 0
    mb4_wins = 0
    pipe_wins = 0
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
            q8_device = malloc(q8_1_d4s4_f32_nbytes(rows, 6_144), runtime=runtime)
            mb2_device = malloc(rows * 5_120 * 2, runtime=runtime)
            mb4_device = malloc(rows * 5_120 * 2, runtime=runtime)
            pipe_device = malloc(rows * 5_120 * 2, runtime=runtime)
            buffers.extend(
                (x_device, q8_device, mb2_device, mb4_device, pipe_device)
            )

            gguf_q8_1_d4s4_f32_quantize_bf16(
                x_device.ptr,
                q8_device.ptr,
                rows,
                6_144,
                library=library,
                runtime=runtime,
            )

            def mb2() -> None:
                gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(
                    q8_device.ptr,
                    raw_device.ptr,
                    mb2_device.ptr,
                    rows,
                    6_144,
                    5_120,
                    library=library,
                    runtime=runtime,
                )

            def mb4() -> None:
                gguf_q5_k_mmq32_mb4_q8_1_d4s4_f32_bf16_bf16_out(
                    q8_device.ptr,
                    raw_device.ptr,
                    mb4_device.ptr,
                    rows,
                    6_144,
                    5_120,
                    library=library,
                    runtime=runtime,
                )

            def pipe() -> None:
                gguf_q5_k_mmq32_pipe_q8_1_d4s4_f32_bf16_bf16_out(
                    q8_device.ptr,
                    raw_device.ptr,
                    pipe_device.ptr,
                    rows,
                    6_144,
                    5_120,
                    library=library,
                    runtime=runtime,
                )

            # Correctness first: bit-exactness of both candidate outputs.
            mb2()
            mb4()
            pipe()
            mb2_out = download(mb2_device, (rows, 5_120))
            mb4_out = download(mb4_device, (rows, 5_120))
            pipe_out = download(pipe_device, (rows, 5_120))
            mb4_mismatches = int(np.count_nonzero(mb2_out != mb4_out))
            pipe_mismatches = int(np.count_nonzero(mb2_out != pipe_out))
            exact = mb4_mismatches == 0 and pipe_mismatches == 0
            mismatches = mb4_mismatches + pipe_mismatches

            mb2_samples: list[float] = []
            mb4_samples: list[float] = []
            pipe_samples: list[float] = []
            for sample in range(args.samples):
                # Counterbalance arm order per weight and per sample.
                if (weight_index + sample) % 2 == 0:
                    mb2_samples.append(event_ms(mb2))
                    mb4_samples.append(event_ms(mb4))
                    pipe_samples.append(event_ms(pipe))
                else:
                    pipe_samples.append(event_ms(pipe))
                    mb4_samples.append(event_ms(mb4))
                    mb2_samples.append(event_ms(mb2))
            for _ in range(args.warmups):
                mb2()
                mb4()
                pipe()

            mb2_med = statistics.median(mb2_samples)
            mb4_med = statistics.median(mb4_samples)
            pipe_med = statistics.median(pipe_samples)
            if pipe_med < mb2_med:
                pipe_wins += 1
            elif mb2_med < pipe_med:
                mb2_wins += 1
            results.append(
                {
                    "weight": name,
                    "rows": rows,
                    "bf16_bit_exact": exact,
                    "bf16_mismatches": mismatches,
                    "kl_vs_mb2": _kl(mb2_out, pipe_out)
                    if pipe_mismatches
                    else 0.0,
                    "mb2_ms": mb2_med,
                    "mb4_ms": mb4_med,
                    "pipe_ms": pipe_med,
                }
            )
            print(
                f"[{weight_index + 1}/48] {name}: mb2 {mb2_med:.4f} ms "
                f"mb4 {mb4_med:.4f} ms pipe {pipe_med:.4f} ms exact={exact}",
                flush=True,
            )
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)

    mb2_sum = sum(row["mb2_ms"] for row in results)
    mb4_sum = sum(row["mb4_ms"] for row in results)
    pipe_sum = sum(row["pipe_ms"] for row in results)
    payload = {
        "schema": 1,
        "kind": "w7900_qwen38_q4km_k3_c8_q5_raw_mmq32_minblocks_pipe_leaf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "target_arches": detect_hip_target_arches(),
        "model": args.model,
        "rows": rows,
        "weights": len(results),
        "arms": {
            "mb2": "gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out (registered raw owner)",
            "mb4": "gguf_q5_k_mmq32_mb4_q8_1_d4s4_f32_bf16_bf16_out (forced minblocks=4)",
            "pipe": "gguf_q5_k_mmq32_pipe_q8_1_d4s4_f32_bf16_bf16_out (software-pipelined)",
        },
        "mb2_sum_ms": mb2_sum,
        "mb4_sum_ms": mb4_sum,
        "pipe_sum_ms": pipe_sum,
        "mb4_ratio": mb4_sum / mb2_sum,
        "pipe_ratio": pipe_sum / mb2_sum,
        "mb2_wins": mb2_wins,
        "mb4_wins": mb4_wins,
        "pipe_wins": pipe_wins,
        "all_bit_exact": all(row["bf16_bit_exact"] for row in results),
        "results": results,
        "timing": "HIP events, consumer-only bursts (shared producer excluded)",
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(
        f"leaf complete: mb2 {mb2_sum:.3f} ms mb4 {mb4_sum:.3f} ms "
        f"pipe {pipe_sum:.3f} ms | pipe/mb2 {pipe_sum / mb2_sum:.4f} "
        f"(pipe_wins {pipe_wins}/48) | mb4/mb2 {mb4_sum / mb2_sum:.4f} "
        f"all_exact={payload['all_bit_exact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
