#!/usr/bin/env python3
"""Actual-weight operation-complete Qwen3.8 gate/up IU4 research screen.

The candidate re-quantizes one Q4_K_S gate/up pair to a clean per-output S4
sidecar, dynamically packs BF16 rows to asymmetric U4, executes native packed
IU4 WMMA, applies I32 zero-point correction/scales, publishes gate/up through
BF16, and emits BF16 SiLU*up.  The control is the current exact qmicro Q8_1x2
rowtile8 owner, including activation packing and runtime-equivalent M>8 chunks.

Diagnostic only: this does not create a model-wide sidecar or runtime route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1x2,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1151.quant.iu4_s4_sidecar import (
    build_iu4_s4_sidecar,
    iu4_s4_dual_silu_bf16_out,
    iu4_u4_quantize_bf16,
    iu4_u4_wmma_nbytes,
    plan_iu4_s4_sidecar_build,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16_qmicro
from hipengine.quant.iu4_s4 import (
    S4Sidecar,
    bf16_bits_to_f32,
    f32_to_bf16_bits,
    pack_s4_wmma_tiles,
    quantize_s4_per_output,
    unpack_s4,
)
from hipengine.runtime.gguf_linear import _rowtile8_row_chunks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
DEFAULT_MODEL_SHA256 = "22200efcd98a7aeeaf83f59b0f1400b055d9e0437900e26b930ef2d42a3eb3f9"
DEFAULT_ROWS = (2, 3, 4, 5, 8, 16, 32, 64, 96, 128)
IU4_ARITHMETIC_ROOF_TOPS = float(
    os.environ.get("HIPENGINE_IU4_ARITHMETIC_ROOF_TOPS", "109.715")
)
IU4_DOT8_ARITHMETIC_ROOF_TOPS = float(
    os.environ.get("HIPENGINE_IU4_DOT8_ROOF_TOPS", "56.830")
)
MEMORY_ROOF_GBPS = float(os.environ.get("HIPENGINE_MEMORY_ROOF_GBPS", "221.0"))
WMMA_VGPR_ANOMALY_THRESHOLD = 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", default=",".join(str(value) for value in DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0x1A4B3)
    parser.add_argument("--dequant-chunk-rows", type=int, default=128)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--target-arch", default=os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"))
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _tracked_status() -> list[str]:
    result = subprocess.run(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()


def _upload(runtime, values: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read_bf16(runtime, buffer: DeviceBuffer, shape: tuple[int, ...]) -> np.ndarray:
    output = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(output), buffer, runtime=runtime)
    return output


def _event_ms(runtime, launch: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launch()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _timing_summary(values: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": _percentile(values, 0.95),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _implementation_quality_metrics(
    *,
    rows: int,
    core_executed_tops: float,
    candidate_effective_weight_gbps: float,
    control_effective_weight_gbps: float,
    arithmetic_roof_tops: float = IU4_ARITHMETIC_ROOF_TOPS,
) -> dict[str, float | str]:
    """Derive the mandatory roof-comparison fields for one IU4 shape."""

    useful_weight_memory_roof_tops = 4.0 * rows * MEMORY_ROOF_GBPS / 1000.0
    memory_binds = useful_weight_memory_roof_tops <= arithmetic_roof_tops
    candidate_memory_fraction = candidate_effective_weight_gbps / MEMORY_ROOF_GBPS
    candidate_arithmetic_fraction = core_executed_tops / arithmetic_roof_tops
    binding_fraction = (
        candidate_memory_fraction if memory_binds else candidate_arithmetic_fraction
    )
    return {
        "binding_roof": "memory" if memory_binds else "arithmetic",
        "binding_roof_value": (
            MEMORY_ROOF_GBPS if memory_binds else arithmetic_roof_tops
        ),
        "binding_roof_unit": "GB/s" if memory_binds else "TOPS",
        "weight_only_useful_memory_roof_tops": useful_weight_memory_roof_tops,
        "arithmetic_roof_tops": arithmetic_roof_tops,
        "memory_roof_gbps": MEMORY_ROOF_GBPS,
        "candidate_effective_weight_gbps": candidate_effective_weight_gbps,
        "control_effective_weight_gbps": control_effective_weight_gbps,
        "candidate_fraction_of_arithmetic_roof": candidate_arithmetic_fraction,
        "candidate_percent_of_arithmetic_roof": candidate_arithmetic_fraction * 100.0,
        "candidate_fraction_of_memory_roof": candidate_memory_fraction,
        "candidate_percent_of_memory_roof": candidate_memory_fraction * 100.0,
        "candidate_fraction_of_binding_roof": binding_fraction,
        "candidate_percent_of_binding_roof": binding_fraction * 100.0,
    }


def _kernel_resource_assessment(
    *,
    accumulators_per_wave: int,
    vgpr_count: int,
) -> dict[str, int | bool | str]:
    """Flag the low-VGPR/single-chain signature of an unblocked WMMA body."""

    anomaly = vgpr_count < WMMA_VGPR_ANOMALY_THRESHOLD
    return {
        "accumulators_per_wave": accumulators_per_wave,
        "vgpr_count": vgpr_count,
        "vgpr_anomaly_threshold": WMMA_VGPR_ANOMALY_THRESHOLD,
        "vgpr_anomaly": anomaly,
        "vgpr_anomaly_reason": (
            "WMMA kernel below 64 VGPR is likely register-unblocked; inspect independent accumulator chains"
            if anomaly
            else "VGPR count is consistent with a register-blocked WMMA candidate"
        ),
    }


def _softmax_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    delta = cand - ref
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)
    ref_shift = ref64 - ref64.max(axis=1, keepdims=True)
    cand_shift = cand64 - cand64.max(axis=1, keepdims=True)
    ref_logp = ref_shift - np.log(np.exp(ref_shift).sum(axis=1, keepdims=True))
    cand_logp = cand_shift - np.log(np.exp(cand_shift).sum(axis=1, keepdims=True))
    row_kl = np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=1)
    ref_norm = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64))))
    rmse = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
    return {
        "finite": bool(np.isfinite(cand).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "normalized_rmse": rmse / ref_norm if ref_norm > 0.0 else 0.0,
        "mean_kl": float(np.mean(row_kl)),
        "max_kl": float(np.max(row_kl)),
        "top1_matches": int(np.count_nonzero(ref.argmax(axis=1) == cand.argmax(axis=1))),
        "top1_total": int(ref.shape[0]),
        "top1_agreement": float(np.mean(ref.argmax(axis=1) == cand.argmax(axis=1))),
    }


def _build_s4_from_q4_k(
    raw: np.ndarray,
    *,
    out_features: int,
    in_features: int,
    chunk_rows: int,
) -> tuple[S4Sidecar, dict[str, float | int]]:
    packed = np.empty((out_features, in_features // 2), dtype=np.uint8)
    scales = np.empty(out_features, dtype=np.float32)
    sums = np.empty(out_features, dtype=np.int32)
    squared_error = 0.0
    squared_source = 0.0
    max_abs = 0.0
    started = time.perf_counter()
    for start in range(0, out_features, chunk_rows):
        end = min(out_features, start + chunk_rows)
        source = dequantize_gguf_data(
            raw[start:end],
            GGMLQuantizationType.Q4_K,
        )
        sidecar = quantize_s4_per_output(source)
        packed[start:end] = sidecar.packed
        scales[start:end] = sidecar.scales
        sums[start:end] = sidecar.sums
        reconstructed = unpack_s4(sidecar.packed).astype(np.float32)
        reconstructed *= sidecar.scales[:, None]
        delta = reconstructed - source
        squared_error += float(np.square(delta, dtype=np.float64).sum())
        squared_source += float(np.square(source, dtype=np.float64).sum())
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
    elapsed = time.perf_counter() - started
    sidecar = S4Sidecar(packed=packed, scales=scales, sums=sums)
    values = out_features * in_features
    rmse = math.sqrt(squared_error / values)
    source_rms = math.sqrt(squared_source / values)
    return sidecar, {
        "source": "dequantized authoritative Q4_K_S tensor (not original BF16/F16)",
        "quantizer": "per-output symmetric S4; scale=max(-min/8,max/7); RNE+clip[-8,7]",
        "values": values,
        "build_seconds": elapsed,
        "max_abs": max_abs,
        "rmse": rmse,
        "source_rms": source_rms,
        "normalized_rmse": rmse / source_rms if source_rms > 0.0 else 0.0,
    }


def _screen_rows(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    runtime,
    q4_library,
    t16_library,
    iu4_library,
    gate_qmicro: DeviceBuffer,
    up_qmicro: DeviceBuffer,
    gate_s4: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    up_s4: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    warmups: int,
    samples: int,
    burst: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + rows)
    x = f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    )
    chunks = _rowtile8_row_chunks(rows)
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        q8_dev = malloc(2 * 8 * (hidden // 32) * 36, runtime=runtime)
        u4_dev = malloc(iu4_u4_wmma_nbytes(rows, hidden), runtime=runtime)
        u4_scale_dev = malloc(rows * 4, runtime=runtime)
        u4_zero_dev = malloc(rows * 4, runtime=runtime)
        control_out_dev = malloc(rows * out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(rows * out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                q8_dev,
                u4_dev,
                u4_scale_dev,
                u4_zero_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            for chunk_rows, row_base in chunks:
                x_ptr = x_dev.ptr + row_base * hidden * 2
                out_ptr = control_out_dev.ptr + row_base * out_features * 2
                gguf_q4_k_quantize_bf16_q8_1x2(
                    x_ptr,
                    q8_dev.ptr,
                    chunk_rows,
                    hidden,
                    library=q4_library,
                    runtime=runtime,
                )
                gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_bf16_bf16_out(
                    q8_dev.ptr,
                    gate_qmicro.ptr,
                    up_qmicro.ptr,
                    out_ptr,
                    chunk_rows,
                    hidden,
                    out_features,
                    library=t16_library,
                    runtime=runtime,
                )

        def candidate_quant() -> None:
            iu4_u4_quantize_bf16(
                x_dev.ptr,
                u4_dev.ptr,
                u4_scale_dev.ptr,
                u4_zero_dev.ptr,
                rows,
                hidden,
                library=iu4_library,
                runtime=runtime,
            )

        def candidate_core() -> None:
            iu4_s4_dual_silu_bf16_out(
                u4_dev.ptr,
                u4_scale_dev.ptr,
                u4_zero_dev.ptr,
                gate_s4[0].ptr,
                gate_s4[1].ptr,
                gate_s4[2].ptr,
                up_s4[0].ptr,
                up_s4[1].ptr,
                up_s4[2].ptr,
                candidate_out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=iu4_library,
                runtime=runtime,
            )

        def candidate() -> None:
            candidate_quant()
            candidate_core()

        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()

        timings = {"current_exact": [], "iu4_inclusive": []}
        candidate_wins = 0
        for sample in range(samples):
            order: list[tuple[str, Callable[[], None]]] = [
                ("current_exact", control),
                ("iu4_inclusive", candidate),
            ]
            if sample & 1:
                order.reverse()
            pair: dict[str, float] = {}
            for name, launch in order:
                pair[name] = _event_ms(runtime, launch, burst=burst)
                timings[name].append(pair[name])
            candidate_wins += int(pair["iu4_inclusive"] < pair["current_exact"])

        candidate_quant()
        runtime.device_synchronize()
        quant_samples = [
            _event_ms(runtime, candidate_quant, burst=burst) for _ in range(samples)
        ]
        core_samples = [
            _event_ms(runtime, candidate_core, burst=burst) for _ in range(samples)
        ]

        control()
        candidate()
        runtime.device_synchronize()
        control_bits = _read_bf16(runtime, control_out_dev, (rows, out_features))
        candidate_bits = _read_bf16(runtime, candidate_out_dev, (rows, out_features))
        candidate()
        runtime.device_synchronize()
        candidate_repeat_bits = _read_bf16(
            runtime, candidate_out_dev, (rows, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    current_summary = _timing_summary(timings["current_exact"])
    candidate_summary = _timing_summary(timings["iu4_inclusive"])
    quant_summary = _timing_summary(quant_samples)
    core_summary = _timing_summary(core_samples)
    current_ms = float(current_summary["median_ms"])
    candidate_ms = float(candidate_summary["median_ms"])
    core_ms = float(core_summary["median_ms"])
    useful_ops = 4 * rows * hidden * out_features
    if rows <= 16:
        candidate_route = "u4s4_dot8_rowtile"
        padded_rows = 4 if rows <= 4 else (8 if rows <= 8 else 16)
        candidate_weight_sweeps = 1
        candidate_arithmetic_roof_tops = IU4_DOT8_ARITHMETIC_ROOF_TOPS
        candidate_accumulators_per_wave = padded_rows * 4
    else:
        candidate_route = "iu4_wmma_bulk_m256"
        padded_rows = ((rows + 255) // 256) * 256
        candidate_weight_sweeps = (rows + 255) // 256
        candidate_arithmetic_roof_tops = IU4_ARITHMETIC_ROOF_TOPS
        candidate_accumulators_per_wave = 16
    executed_ops = 4 * padded_rows * hidden * out_features
    sidecar_pair_bytes = 2 * (out_features * hidden // 2 + out_features * 8)
    qmicro_pair_bytes = 100_270_080
    candidate_effective_weight_gbps = (
        sidecar_pair_bytes * candidate_weight_sweeps / core_ms / 1e6
    )
    control_effective_weight_gbps = (
        qmicro_pair_bytes * len(chunks) / current_ms / 1e6
    )
    implementation_quality = _implementation_quality_metrics(
        rows=rows,
        core_executed_tops=executed_ops / (core_ms / 1000.0) / 1e12,
        candidate_effective_weight_gbps=candidate_effective_weight_gbps,
        control_effective_weight_gbps=control_effective_weight_gbps,
        arithmetic_roof_tops=candidate_arithmetic_roof_tops,
    )
    return {
        "rows": rows,
        "current_chunks": [
            {"rows": chunk_rows, "row_base": row_base}
            for chunk_rows, row_base in chunks
        ],
        "timing": {
            "current_exact_inclusive": current_summary,
            "iu4_inclusive": candidate_summary,
            "iu4_activation_pack": quant_summary,
            "iu4_core_correction_bf16_silu": core_summary,
            "candidate_wins": candidate_wins,
            "pair_count": samples,
        },
        "performance": {
            "inclusive_speedup": current_ms / candidate_ms,
            "inclusive_candidate_over_control": candidate_ms / current_ms,
            "inclusive_delta_percent": (candidate_ms / current_ms - 1.0) * 100.0,
            "useful_tops": useful_ops / (candidate_ms / 1000.0) / 1e12,
            "core_useful_tops": useful_ops / (core_ms / 1000.0) / 1e12,
            "core_executed_tops": executed_ops / (core_ms / 1000.0) / 1e12,
            "candidate_route": candidate_route,
            "candidate_accumulators_per_wave": candidate_accumulators_per_wave,
            "candidate_row_utilization": rows / padded_rows,
            "candidate_weight_sweeps": candidate_weight_sweeps,
            "candidate_arithmetic_roof_tops": candidate_arithmetic_roof_tops,
            "wmma_row_utilization": rows / padded_rows,
            "iu4_weight_bytes_per_core": sidecar_pair_bytes * candidate_weight_sweeps,
            "iu4_effective_weight_gbps": candidate_effective_weight_gbps,
            "current_qmicro_weight_bytes_per_inclusive_call": (
                qmicro_pair_bytes * len(chunks)
            ),
            "current_effective_weight_gbps": control_effective_weight_gbps,
            "implementation_quality": implementation_quality,
        },
        "correctness_vs_current_exact": _softmax_metrics(
            bf16_bits_to_f32(control_bits),
            bf16_bits_to_f32(candidate_bits),
        ),
        "candidate_deterministic_bits": bool(
            np.array_equal(candidate_bits, candidate_repeat_bits)
        ),
        "candidate_output_sha256": _sha256_array(candidate_bits),
        "current_output_sha256": _sha256_array(control_bits),
    }


def main() -> int:
    args = _parse_args()
    rows_values = tuple(int(value) for value in args.rows.split(",") if value)
    if rows_values != DEFAULT_ROWS:
        raise ValueError(f"rows must be exactly {DEFAULT_ROWS} for the R2 gate")
    if args.layer < 0 or args.warmups < 0 or min(args.samples, args.burst, args.dequant_chunk_rows) <= 0:
        raise ValueError("layer/warmups must be non-negative; samples/burst/chunk positive")
    compiler_version = None
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()

    reader = GGUFReader(args.model)
    names = {
        "gate": f"blk.{args.layer}.ffn_gate.weight",
        "up": f"blk.{args.layer}.ffn_up.weight",
    }
    infos = {role: reader.tensor_info(name) for role, name in names.items()}
    if any(info.ggml_type_name != "Q4_K" or len(info.shape) != 2 for info in infos.values()):
        raise ValueError("gate/up must be rank-2 Q4_K tensors")
    if infos["gate"].shape != infos["up"].shape:
        raise ValueError("gate/up tensor shapes differ")
    out_features, hidden = (int(value) for value in infos["gate"].shape)
    source_pair_bytes = sum(int(info.nbytes) for info in infos.values())
    if source_pair_bytes <= 64 * 1024 * 1024:
        raise ValueError("actual gate/up source pair must exceed the 64-MiB cold-pool gate")

    raw_gate = np.asarray(reader.tensor_data(names["gate"]))
    raw_up = np.asarray(reader.tensor_data(names["up"]))
    gate_s4_host, gate_quant = _build_s4_from_q4_k(
        raw_gate,
        out_features=out_features,
        in_features=hidden,
        chunk_rows=args.dequant_chunk_rows,
    )
    up_s4_host, up_quant = _build_s4_from_q4_k(
        raw_up,
        out_features=out_features,
        in_features=hidden,
        chunk_rows=args.dequant_chunk_rows,
    )
    gate_qmicro_host = repack_gguf_q4_k_tile16_qmicro(raw_gate[None, ...]).tiles
    up_qmicro_host = repack_gguf_q4_k_tile16_qmicro(raw_up[None, ...]).tiles

    runtime = get_hip_runtime()
    before_bytes = memory_stats()["current_allocated_bytes"]
    require_cached = bool(args.require_cached_build)
    q4_library = build_gguf_q4_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    t16_library = build_gguf_t16_selected_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    iu4_library = build_iu4_s4_sidecar(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
        target_arch=args.target_arch,
    )
    build_plan = plan_iu4_s4_sidecar_build(
        compiler_version=compiler_version, target_arch=args.target_arch
    )
    persistent: list[DeviceBuffer] = []
    try:
        gate_qmicro = _upload(runtime, gate_qmicro_host)
        up_qmicro = _upload(runtime, up_qmicro_host)
        gate_s4 = (
            _upload(runtime, pack_s4_wmma_tiles(gate_s4_host)),
            _upload(runtime, gate_s4_host.scales),
            _upload(runtime, gate_s4_host.sums),
        )
        up_s4 = (
            _upload(runtime, pack_s4_wmma_tiles(up_s4_host)),
            _upload(runtime, up_s4_host.scales),
            _upload(runtime, up_s4_host.sums),
        )
        persistent.extend((gate_qmicro, up_qmicro, *gate_s4, *up_s4))
        results = [
            _screen_rows(
                rows=rows,
                hidden=hidden,
                out_features=out_features,
                runtime=runtime,
                q4_library=q4_library,
                t16_library=t16_library,
                iu4_library=iu4_library,
                gate_qmicro=gate_qmicro,
                up_qmicro=up_qmicro,
                gate_s4=gate_s4,
                up_s4=up_s4,
                warmups=args.warmups,
                samples=args.samples,
                burst=args.burst,
                seed=args.seed,
            )
            for rows in rows_values
        ]
    finally:
        for buffer in reversed(persistent):
            free(buffer, runtime=runtime)
    runtime.device_synchronize()
    after_bytes = memory_stats()["current_allocated_bytes"]

    tiny = [row for row in results if int(row["rows"]) <= 5]
    tiny_all_faster = all(
        float(row["performance"]["inclusive_speedup"]) > 1.0 for row in tiny
    )
    b3_scope_all_faster = all(
        float(row["performance"]["inclusive_speedup"]) > 1.0
        for row in results
        if int(row["rows"]) <= 4
    )
    wide_m_all_faster = all(
        float(row["performance"]["inclusive_speedup"]) > 1.0
        for row in results
        if int(row["rows"]) >= 16
    )
    all_finite = all(bool(row["correctness_vs_current_exact"]["finite"]) for row in results)
    all_deterministic = all(bool(row["candidate_deterministic_bits"]) for row in results)
    teardown = before_bytes == after_bytes
    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_iu4_s4_gate_up_leaf",
        "status": (
            "tiny_m_gate_passed"
            if tiny_all_faster and all_finite and all_deterministic and teardown
            else (
                "tiny_m_rejected_wide_m_candidate"
                if wide_m_all_faster and all_finite and all_deterministic and teardown
                else "diagnostic_rejected"
            )
        ),
        "performance_claim": False,
        "scope": "one-layer actual-weight operation-complete research leaf; no runtime route or model-wide sidecar",
        "hardware": {
            "hostname": platform.node(),
            "cpu": os.environ.get("HIPENGINE_HW_CPU", "AMD Ryzen AI MAX+ 395"),
            "gpu": os.environ.get("HIPENGINE_HW_GPU", "AMD Radeon 8060S Graphics"),
            "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        "software": {
            "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "tracked_dirty_paths": _tracked_status(),
            "require_cached_build": require_cached,
            "compiler_version_file": (
                str(args.compiler_version_file) if args.compiler_version_file else None
            ),
            "iu4_build": {
                "cache_key": build_plan.cache_key,
                "cache_dir": str(build_plan.cache_dir),
                "output": str(build_plan.output_path),
                "loaded_output": str(getattr(iu4_library, "_name", "")),
                "command": list(build_plan.command),
            },
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": str(args.model_sha256),
            "size_bytes": args.model.stat().st_size,
            "quant": "Q4_K_S",
            "layer": args.layer,
            "gate_tensor": names["gate"],
            "up_tensor": names["up"],
            "shape": [out_features, hidden],
            "source_pair_bytes": source_pair_bytes,
        },
        "representations": {
            "control": {
                "variant": "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
                "layout": "gguf_q4_k_qmicro_t16_v1",
                "pair_bytes": int(gate_qmicro_host.nbytes + up_qmicro_host.nbytes),
                "strict_fallback": True,
            },
            "candidate": {
                "variant": "dot8_m2_m16_wmma_bulk_m17_m1024_bf16_out",
                "layout": "iu4_s4_sidecar_v1",
                "arithmetic_class": "T3",
                "source": "re-quantized dequantized Q4_K_S (not original BF16/F16)",
                "gate_bytes": gate_s4_host.nbytes,
                "up_bytes": up_s4_host.nbytes,
                "pair_bytes": gate_s4_host.nbytes + up_s4_host.nbytes,
                "gate_payload_sha256": _sha256_array(gate_s4_host.packed),
                "up_payload_sha256": _sha256_array(up_s4_host.packed),
                "gate_quantization": gate_quant,
                "up_quantization": up_quant,
            },
        },
        "protocol": {
            "rows": list(rows_values),
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP events; inclusive pack+core+correction+BF16+SiLU",
            "control_chunking": "runtime _rowtile8_row_chunks for M>8",
            "cold_pool": "actual gate/up source pair and both execution views exceed 64 MiB",
            "seed": args.seed,
            "implementation_quality_roofs": {
                "iu4_arithmetic_tops": IU4_ARITHMETIC_ROOF_TOPS,
                "practical_memory_gbps": MEMORY_ROOF_GBPS,
                "binding_rule": "min(weight-only 4*M*memory roof, IU4 arithmetic roof)",
            },
        },
        "candidate_kernel_routes": {
            "m2_m16": {
                "route": "u4s4_dot8_rowtile",
                "accumulators_per_wave": "4 * padded rows (16/32/64 for row caps 4/8/16)",
            },
            "m17_m1024": {
                "route": "iu4_wmma_bulk_m256",
                "accumulators_per_wave": 16,
            },
        },
        "results": results,
        "gates": {
            "tiny_m_inclusive_faster_all_m2_m5": tiny_all_faster,
            "native_b1_b3_inclusive_faster_all_m2_m4": b3_scope_all_faster,
            "wide_m_inclusive_faster_all_m16_m128": wide_m_all_faster,
            "intermediate_error_diagnostic_only": True,
            "model_logit_quality_gate_run": False,
            "model_logit_quality_qualified": False,
            "quality_note": (
                "Gate/up channel top-1 and softmax KL are localization diagnostics, not the "
                "project's full-vocabulary outer gate. The re-quantized Q4_K_S S4 view is "
                "T3 and remains unqualified without the complete model gate."
            ),
            "all_finite": all_finite,
            "all_deterministic": all_deterministic,
            "teardown_exact": teardown,
        },
        "memory": {
            "tracked_current_before_bytes": before_bytes,
            "tracked_current_after_bytes": after_bytes,
            "teardown_exact": teardown,
        },
        "command": " ".join(
            [
                f"HIPENGINE_HIP_ARCH={os.environ.get('HIPENGINE_HIP_ARCH', '')}",
                Path(os.sys.executable).name,
                *os.sys.argv,
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": artifact["status"],
        "pair_bytes": artifact["representations"]["candidate"]["pair_bytes"],
        "rows": [
            {
                "m": row["rows"],
                "speedup": row["performance"]["inclusive_speedup"],
                "pack_ms": row["timing"]["iu4_activation_pack"]["median_ms"],
                "core_ms": row["timing"]["iu4_core_correction_bf16_silu"]["median_ms"],
                "top1": row["correctness_vs_current_exact"]["top1_agreement"],
                "max_kl": row["correctness_vs_current_exact"]["max_kl"],
            }
            for row in results
        ],
    }, indent=2))
    return 0 if all_finite and all_deterministic and teardown else 1


if __name__ == "__main__":
    raise SystemExit(main())
