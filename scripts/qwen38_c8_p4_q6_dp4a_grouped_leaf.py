#!/usr/bin/env python3
"""Leaf A/B: grouped q8_1 DP4A versus the grouped-R8 BF16 owner at R8-R32.

C8-P4 reduced-dequantization realization screen (iteration 33). The new
grouped dp4a kernel (rows 8-64) is timed against the registered grouped-R8
BF16 owner on actual Qwen3.8 weights at the production row counts. The dp4a
arm includes the q8_1 quantize launch. Arms differ in arithmetic (integer
dp4a decode vs BF16 decode); outputs are compared informationally (mismatch,
KL), not gated on exactness. Kernel-level oracle/floor/determinism contracts
live in tests/test_gguf_q6_k_t16_planar_q8_1_grouped_gemv.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

import numpy as np


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


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
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 24, 32])
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
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out,
        gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    cases = []
    for role, name in (
        ("ffn_down", "blk.0.ffn_down.weight"),
        ("recurrent_qkv", "blk.0.attn_qkv.weight"),
        ("attention_v", "blk.3.attn_v.weight"),
    ):
        info = reader.tensor_info(name)
        cases.append((role, name, int(info.shape[0]), int(info.shape[1])))

    q6_library = build_gguf_q6_k_t16_gemv(load=True)
    q4_library = build_gguf_q4_k_gemv(load=True)

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
    control_wins = 0
    dp4a_wins = 0
    for index, (role, name, out_features, in_features) in enumerate(cases):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        planar = np.ascontiguousarray(
            repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles
        )
        planar_device = upload(planar)
        buffers = [planar_device]
        try:
            for rows in args.rows:
                rng = np.random.default_rng(
                    2_026_090_410 + index * 1000 + rows
                )
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, in_features)).astype(
                        np.float32
                    )
                )
                x_device = upload(x)
                xq_device = malloc(
                    rows * (in_features // 32) * 36, runtime=runtime
                )
                control_device = malloc(rows * out_features * 2, runtime=runtime)
                dp4a_device = malloc(rows * out_features * 2, runtime=runtime)
                buffers.extend((x_device, xq_device, control_device, dp4a_device))

                control_fn = (
                    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out
                    if rows >= 16
                    else gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out
                )

                def control() -> None:
                    control_fn(
                        x_device.ptr,
                        planar_device.ptr,
                        control_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=q6_library,
                        runtime=runtime,
                    )

                def dp4a() -> None:
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_device.ptr,
                        xq_device.ptr,
                        rows,
                        in_features,
                        library=q4_library,
                        runtime=runtime,
                    )
                    gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out(
                        xq_device.ptr,
                        planar_device.ptr,
                        dp4a_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=q6_library,
                        runtime=runtime,
                    )

                control()
                dp4a()
                control_out = download(control_device, (rows, out_features))
                dp4a_out = download(dp4a_device, (rows, out_features))
                mismatches = int(np.count_nonzero(control_out != dp4a_out))
                finite = bool(
                    np.isfinite(_bf16_f32(control_out)).all()
                    and np.isfinite(_bf16_f32(dp4a_out)).all()
                )

                control_samples: list[float] = []
                dp4a_samples: list[float] = []
                for sample in range(args.samples):
                    if (index + sample) % 2 == 0:
                        control_samples.append(event_ms(control))
                        dp4a_samples.append(event_ms(dp4a))
                    else:
                        dp4a_samples.append(event_ms(dp4a))
                        control_samples.append(event_ms(control))
                for _ in range(args.warmups):
                    control()
                    dp4a()

                control_med = statistics.median(control_samples)
                dp4a_med = statistics.median(dp4a_samples)
                if dp4a_med < control_med:
                    dp4a_wins += 1
                elif control_med < dp4a_med:
                    control_wins += 1
                results.append(
                    {
                        "role": role,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "control_ms": control_med,
                        "dp4a_incl_quantize_ms": dp4a_med,
                        "dp4a_ratio": dp4a_med / control_med,
                        "mismatches_vs_control": mismatches,
                        "kl_vs_control": _kl(control_out, dp4a_out),
                        "finite": finite,
                    }
                )
                print(
                    f"[{index + 1}/{len(cases)}] {role} rows={rows}: control "
                    f"{control_med:.4f} ms dp4a+quant {dp4a_med:.4f} ms ratio "
                    f"{dp4a_med / control_med:.4f} mismatches {mismatches}",
                    flush=True,
                )
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)

    payload = {
        "schema": 1,
        "kind": "w7900_qwen38_q4km_k3_c8_p4_q6_dp4a_grouped_leaf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "target_arches": detect_hip_target_arches(),
        "model": args.model,
        "arms": {
            "control": "grouped_rows8 BF16 owner at rows>=16, rowtile_col8 BF16 owner at rows=8 (the dispatched owners)",
            "dp4a": "gguf_q4_k_quantize_bf16_q8_1 + gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_bf16_bf16_out (new grouped integer decode, quantize included)",
        },
        "control_wins": control_wins,
        "dp4a_wins": dp4a_wins,
        "results": results,
        "timing": "HIP events, counterbalanced bursts; dp4a arm includes the q8_1 quantize launch",
        "note": "changed-arithmetic screen (informational KL, no exactness gate); kernel-level oracle/floor/determinism contracts in tests/test_gguf_q6_k_t16_planar_q8_1_grouped_gemv.py",
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    ratios = [r["dp4a_ratio"] for r in results]
    print(
        f"leaf complete: dp4a ratios {['%.4f' % r for r in ratios]} "
        f"dp4a_wins {dp4a_wins}/{len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
