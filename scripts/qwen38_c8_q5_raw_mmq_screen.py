#!/usr/bin/env python3
"""Screen operation-complete raw-Q5 MMQ on all Qwen3.8 ssm_out weights."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _quality(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    reference_f64 = _bf16_f32(reference).astype(np.float64)
    actual_f64 = _bf16_f32(actual).astype(np.float64)
    reference_shifted = reference_f64 - reference_f64.max(axis=1, keepdims=True)
    actual_shifted = actual_f64 - actual_f64.max(axis=1, keepdims=True)
    reference_probs = np.exp(reference_shifted)
    reference_probs /= reference_probs.sum(axis=1, keepdims=True)
    actual_probs = np.exp(actual_shifted)
    actual_probs /= actual_probs.sum(axis=1, keepdims=True)
    kl = np.sum(
        reference_probs * (np.log(reference_probs) - np.log(actual_probs)),
        axis=1,
    )
    return {
        "mismatch": int(np.count_nonzero(reference != actual)),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "top1": float(
            np.mean(reference_f64.argmax(axis=1) == actual_f64.argmax(axis=1))
        ),
        "max_abs": float(np.max(np.abs(reference_f64 - actual_f64))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"),
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--source-c8-candidate",
        action="store_true",
        help="Compare FP32-metadata K-major I64/J16-J32 against retained raw D4S4",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.burst <= 0 or args.warmups < 0:
        raise ValueError("samples/burst must be positive and warmups nonnegative")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
        build_gguf_k_mmq_prefill,
        build_gguf_q5_k_source_mmq_prefill,
        gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out,
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
        gguf_q8_1_d4s4_f32_quantize_bf16,
        gguf_q8_1_d4s4_f32_quantize_bf16_kmajor,
        q8_1_d4s4_f32_kmajor_nbytes,
        q8_1_d4s4_f32_nbytes,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out,
        gguf_q5_k_t16_gemv_rowtile_grouped_rows8_bf16_bf16_out,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text().strip()

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    names = tuple(
        tensor.name
        for tensor in reader.info.tensors
        if tensor.name.endswith(".ssm_out.weight")
    )
    if len(names) != 48:
        raise ValueError(f"expected 48 ssm_out weights, found {len(names)}")
    control_library = None
    if not args.source_c8_candidate:
        control_library = build_gguf_t16_selected_gemv(
            load=True,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )
    candidate_library = build_gguf_k_mmq_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    source_library = None
    if args.source_c8_candidate:
        source_library = build_gguf_q5_k_source_mmq_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=args.require_cached_build,
        )

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    def download(buffer, shape: tuple[int, int]) -> np.ndarray:
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

    results: list[dict[str, object]] = []
    for weight_index, name in enumerate(names):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        tiles = None
        if not args.source_c8_candidate:
            tiles = np.ascontiguousarray(
                repack_gguf_q5_k_tile16(raw[None, ...]).tiles
            )
        buffers = []
        try:
            raw_device = upload(raw)
            tiles_device = upload(tiles) if tiles is not None else None
            buffers.append(raw_device)
            if tiles_device is not None:
                buffers.append(tiles_device)
            for rows in (24, 32):
                rng = np.random.default_rng(2_026_090_200 + weight_index * 100 + rows)
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, 6_144)).astype(np.float32)
                )
                x_device = upload(x)
                q8_device = malloc(
                    q8_1_d4s4_f32_nbytes(rows, 6_144), runtime=runtime
                )
                source_q8_device = None
                if args.source_c8_candidate:
                    source_q8_device = malloc(
                        q8_1_d4s4_f32_kmajor_nbytes(rows, 6_144), runtime=runtime
                    )
                control_device = malloc(rows * 5_120 * 2, runtime=runtime)
                candidate_device = malloc(rows * 5_120 * 2, runtime=runtime)
                buffers.extend((x_device, q8_device))
                if source_q8_device is not None:
                    buffers.append(source_q8_device)
                buffers.extend((control_device, candidate_device))
                grouped = (
                    gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out
                    if rows == 24
                    else gguf_q5_k_t16_gemv_rowtile_grouped_rows8_bf16_bf16_out
                )

                def grouped_control() -> None:
                    assert tiles_device is not None
                    assert control_library is not None
                    grouped(
                        x_device.ptr,
                        tiles_device.ptr,
                        control_device.ptr,
                        rows,
                        6_144,
                        5_120,
                        library=control_library,
                        runtime=runtime,
                    )

                def retained_raw(output_device) -> None:
                    gguf_q8_1_d4s4_f32_quantize_bf16(
                        x_device.ptr,
                        q8_device.ptr,
                        rows,
                        6_144,
                        library=candidate_library,
                        runtime=runtime,
                    )
                    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(
                        q8_device.ptr,
                        raw_device.ptr,
                        output_device.ptr,
                        rows,
                        6_144,
                        5_120,
                        library=candidate_library,
                        runtime=runtime,
                    )

                def source_c8_candidate() -> None:
                    assert source_q8_device is not None
                    assert source_library is not None
                    gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(
                        x_device.ptr,
                        source_q8_device.ptr,
                        rows,
                        6_144,
                        library=candidate_library,
                        runtime=runtime,
                    )
                    gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
                        source_q8_device.ptr,
                        raw_device.ptr,
                        candidate_device.ptr,
                        rows,
                        6_144,
                        5_120,
                        library=source_library,
                        runtime=runtime,
                    )

                def retained_control() -> None:
                    retained_raw(control_device)

                def retained_candidate() -> None:
                    retained_raw(candidate_device)

                if args.source_c8_candidate:
                    control = retained_control
                    candidate = source_c8_candidate
                else:
                    control = grouped_control
                    candidate = retained_candidate

                for _ in range(args.warmups):
                    control()
                    candidate()
                runtime.device_synchronize()
                control()
                candidate()
                runtime.device_synchronize()
                quality = _quality(
                    download(control_device, (rows, 5_120)),
                    download(candidate_device, (rows, 5_120)),
                )
                samples = {"control": [], "candidate": []}
                for sample in range(args.samples):
                    order = (
                        ("control", "candidate")
                        if (weight_index + sample) % 2 == 0
                        else ("candidate", "control")
                    )
                    for variant in order:
                        samples[variant].append(
                            event_ms(control if variant == "control" else candidate)
                        )
                control_median = statistics.median(samples["control"])
                candidate_median = statistics.median(samples["candidate"])
                result = {
                    "weight": name,
                    "rows": rows,
                    "control_ms": control_median,
                    "candidate_ms": candidate_median,
                    "speedup": control_median / candidate_median,
                    "saving_ms": control_median - candidate_median,
                    "quality": quality,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)

    summary = {}
    for rows in (24, 32):
        cells = [result for result in results if result["rows"] == rows]
        control_sum = sum(float(result["control_ms"]) for result in cells)
        candidate_sum = sum(float(result["candidate_ms"]) for result in cells)
        summary[str(rows)] = {
            "control_sum_ms": control_sum,
            "candidate_sum_ms": candidate_sum,
            "speedup": control_sum / candidate_sum,
            "saving_ms": control_sum - candidate_sum,
            "winning_weights": sum(float(result["speedup"]) > 1.0 for result in cells),
            "mean_kl": statistics.mean(
                float(result["quality"]["mean_kl"]) for result in cells
            ),
            "max_kl": max(float(result["quality"]["max_kl"]) for result in cells),
            "top1": statistics.mean(
                float(result["quality"]["top1"]) for result in cells
            ),
        }
    payload = {
        "schema": 1,
        "kind": (
            "qwen38_c8_q5_all48_source_mmq_screen"
            if args.source_c8_candidate
            else "qwen38_c8_q5_all48_raw_mmq_screen"
        ),
        "host": platform.node(),
        "hardware": "AMD Radeon Pro W7900",
        "gpu": "GPU0",
        "model": str(args.model),
        "quant": "Q4_K_M / ssm_out Q5_K",
        "shapes": {
            "weights": len(names),
            "in_features": 6_144,
            "out_features": 5_120,
            "rows": [24, 32],
        },
        "protocol": {
            "timing": "counterbalanced HIP events",
            "samples": args.samples,
            "burst": args.burst,
            "warmups": args.warmups,
            "candidate": (
                "Q8_1 D4S4-FP32 K-major producer + adaptive raw Q5 "
                "I64/J16-J32 integer-WMMA BF16 output"
                if args.source_c8_candidate
                else "Q8_1 D4S4 FP32 producer + raw Q5 MMQ32 BF16 output"
            ),
            "control": (
                "retained Q8_1 D4S4 FP32 producer + raw Q5 MMQ32 BF16 output"
                if args.source_c8_candidate
                else "T16 exact grouped rows6/rows8 BF16 output"
            ),
        },
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
