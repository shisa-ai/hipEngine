#!/usr/bin/env python3
"""Measure the current bulk-prefill gate/up+SiLU owner for the IU4 R7 control.

This B3 diagnostic uses actual Qwen3.8-27B Q4_K_S layer weights, losslessly
repacks them into the current pack8 view, and times only the operation-complete
BF16-input gate/up WMMA+SiLU kernel.  It does not launch an IU4 candidate or
create a runtime route.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    build_gguf_q4_k_prefill,
    gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out,
    plan_gguf_q4_k_prefill_build,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
DEFAULT_MODEL_SHA256 = "22200efcd98a7aeeaf83f59b0f1400b055d9e0437900e26b930ef2d42a3eb3f9"
DEFAULT_ROWS = (64, 128, 256, 512, 1024)
F16_WMMA_ROOF_TFLOPS = float(
    os.environ.get("HIPENGINE_F16_WMMA_ROOF_TFLOPS", "55.066")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", default=",".join(str(value) for value in DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0xB300)
    parser.add_argument("--compiler-version-file", type=Path)
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


def _event_ms(runtime, launch: Callable[[], None]) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        launch()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end))
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


def _timing_summary(values: list[float]) -> dict[str, object]:
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


def _prefill_control_metrics(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    median_ms: float,
    pair_bytes: int,
) -> dict[str, float | int]:
    """Return operation-complete payload bandwidth and WMMA arithmetic rate."""

    rows_per_block = 256
    row_blocks = (rows + rows_per_block - 1) // rows_per_block
    executed_rows = row_blocks * rows_per_block
    useful_operations = 4 * rows * hidden * out_features
    executed_operations = 4 * executed_rows * hidden * out_features
    executed_payload_bytes = pair_bytes * row_blocks
    executed_tflops = executed_operations / (median_ms / 1000.0) / 1e12
    roof_fraction = executed_tflops / F16_WMMA_ROOF_TFLOPS
    return {
        "wmma_useful_ops": useful_operations,
        "wmma_executed_ops": executed_operations,
        "wmma_row_blocks": row_blocks,
        "wmma_executed_rows": executed_rows,
        "wmma_row_utilization": rows / executed_rows,
        "pair_payload_bytes": pair_bytes,
        "executed_payload_bytes": executed_payload_bytes,
        "effective_payload_gbps": executed_payload_bytes / median_ms / 1e6,
        "executed_tflops": executed_tflops,
        "f16_wmma_roof_tflops": F16_WMMA_ROOF_TFLOPS,
        "fraction_of_f16_wmma_roof": roof_fraction,
        "percent_of_f16_wmma_roof": roof_fraction * 100.0,
    }


def _screen_rows(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    pair_bytes: int,
    runtime,
    library,
    gate,
    up,
    warmups: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng_a = np.random.default_rng(seed + rows)
    rng_b = np.random.default_rng(seed + 0x10000 + rows)
    x_a = f32_to_bf16_bits(
        rng_a.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    )
    x_b = f32_to_bf16_bits(
        rng_b.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    )
    buffers: list[DeviceBuffer] = []
    try:
        x_a_dev = _upload(runtime, x_a)
        x_b_dev = _upload(runtime, x_b)
        out_a_dev = malloc(rows * out_features * 2, runtime=runtime)
        out_b_dev = malloc(rows * out_features * 2, runtime=runtime)
        buffers.extend((x_a_dev, x_b_dev, out_a_dev, out_b_dev))

        def launch_a() -> None:
            gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
                x_a_dev.ptr,
                gate[0].ptr,
                gate[1].ptr,
                gate[2].ptr,
                up[0].ptr,
                up[1].ptr,
                up[2].ptr,
                out_a_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )

        def launch_b() -> None:
            gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
                x_b_dev.ptr,
                gate[0].ptr,
                gate[1].ptr,
                gate[2].ptr,
                up[0].ptr,
                up[1].ptr,
                up[2].ptr,
                out_b_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )

        for _ in range(warmups):
            launch_a()
            launch_b()
        runtime.device_synchronize()

        timings: dict[str, list[float]] = {"activation_a": [], "activation_b": []}
        for sample in range(samples):
            order = [("activation_a", launch_a), ("activation_b", launch_b)]
            if sample & 1:
                order.reverse()
            for name, launch in order:
                timings[name].append(_event_ms(runtime, launch))

        launch_a()
        launch_b()
        runtime.device_synchronize()
        output_a = _read_bf16(runtime, out_a_dev, (rows, out_features))
        output_b = _read_bf16(runtime, out_b_dev, (rows, out_features))
        launch_a()
        runtime.device_synchronize()
        output_a_repeat = _read_bf16(runtime, out_a_dev, (rows, out_features))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    pooled = timings["activation_a"] + timings["activation_b"]
    pooled_summary = _timing_summary(pooled)
    median_ms = float(pooled_summary["median_ms"])
    finite = bool(
        np.isfinite(bf16_bits_to_f32(output_a)).all()
        and np.isfinite(bf16_bits_to_f32(output_b)).all()
    )
    return {
        "rows": rows,
        "timing": {
            "activation_a": _timing_summary(timings["activation_a"]),
            "activation_b": _timing_summary(timings["activation_b"]),
            "pooled": pooled_summary,
            "counterbalanced_pairs": samples,
        },
        "performance": _prefill_control_metrics(
            rows=rows,
            hidden=hidden,
            out_features=out_features,
            median_ms=median_ms,
            pair_bytes=pair_bytes,
        ),
        "correctness": {
            "finite": finite,
            "activation_sensitive": _sha256_array(output_a) != _sha256_array(output_b),
            "deterministic_bits": bool(np.array_equal(output_a, output_a_repeat)),
            "activation_a_output_sha256": _sha256_array(output_a),
            "activation_b_output_sha256": _sha256_array(output_b),
        },
    }


def main() -> int:
    args = _parse_args()
    rows_values = tuple(int(value) for value in args.rows.split(",") if value)
    if rows_values != DEFAULT_ROWS:
        raise ValueError(f"rows must be exactly {DEFAULT_ROWS} for the B3 control")
    if args.layer < 0 or args.warmups < 0 or args.samples <= 0:
        raise ValueError("layer/warmups must be non-negative and samples positive")
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

    repack_started = time.perf_counter()
    gate_host = repack_gguf_q4_k_pack8(np.asarray(reader.tensor_data(names["gate"])))
    up_host = repack_gguf_q4_k_pack8(np.asarray(reader.tensor_data(names["up"])))
    repack_seconds = time.perf_counter() - repack_started
    pair_bytes = sum(
        array.nbytes
        for packed in (gate_host, up_host)
        for array in (packed.qweight, packed.scales, packed.mins)
    )
    if pair_bytes <= 64 * 1024 * 1024:
        raise ValueError("pack8 pair must exceed the 64-MiB cold-pool gate")

    runtime = get_hip_runtime()
    before_bytes = memory_stats()["current_allocated_bytes"]
    require_cached = bool(args.require_cached_build)
    library = build_gguf_q4_k_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    build_plan = plan_gguf_q4_k_prefill_build(compiler_version=compiler_version)
    persistent: list[DeviceBuffer] = []
    try:
        gate = tuple(
            _upload(runtime, array)
            for array in (gate_host.qweight, gate_host.scales, gate_host.mins)
        )
        up = tuple(
            _upload(runtime, array)
            for array in (up_host.qweight, up_host.scales, up_host.mins)
        )
        persistent.extend((*gate, *up))
        results = [
            _screen_rows(
                rows=rows,
                hidden=hidden,
                out_features=out_features,
                pair_bytes=pair_bytes,
                runtime=runtime,
                library=library,
                gate=gate,
                up=up,
                warmups=args.warmups,
                samples=args.samples,
                seed=args.seed,
            )
            for rows in rows_values
        ]
    finally:
        for buffer in reversed(persistent):
            free(buffer, runtime=runtime)
    runtime.device_synchronize()
    after_bytes = memory_stats()["current_allocated_bytes"]

    all_finite = all(bool(row["correctness"]["finite"]) for row in results)
    all_deterministic = all(
        bool(row["correctness"]["deterministic_bits"]) for row in results
    )
    all_sensitive = all(
        bool(row["correctness"]["activation_sensitive"]) for row in results
    )
    teardown = before_bytes == after_bytes
    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_pack8_gate_up_prefill_control",
        "status": (
            "accepted_diagnostic_control"
            if all_finite and all_deterministic and all_sensitive and teardown
            else "rejected_correctness"
        ),
        "performance_claim": False,
        "scope": "B3 one-layer current-owner control for future IU4 R7; no IU4 candidate",
        "hardware": {
            "hostname": platform.node(),
            "cpu": "AMD Ryzen AI MAX+ 395",
            "gpu": "AMD Radeon 8060S Graphics",
            "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        "software": {
            "commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "tracked_dirty_paths": _tracked_status(),
            "require_cached_build": require_cached,
            "compiler_version_file": (
                str(args.compiler_version_file) if args.compiler_version_file else None
            ),
            "build": {
                "cache_key": build_plan.cache_key,
                "cache_dir": str(build_plan.cache_dir),
                "output": str(build_plan.output_path),
                "loaded_output": str(getattr(library, "_name", "")),
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
        "control": {
            "variant": "pack8_dual_wmma_prefill_bf16_bf16_out",
            "layout": "gguf_q4_k_pack8",
            "arithmetic": "Q4_K reconstruction to FP16; F16 WMMA F32 accumulate; BF16 gate/up boundary; SiLU",
            "pair_payload_bytes": pair_bytes,
            "repack_seconds": repack_seconds,
        },
        "protocol": {
            "rows": list(rows_values),
            "warmups": args.warmups,
            "counterbalanced_pairs": args.samples,
            "timed_calls_per_shape": 2 * args.samples,
            "timing": "counterbalanced A/B deterministic activation matrices; HIP events; pooled median",
            "cold_pool": "actual pack8 gate/up pair exceeds 64 MiB",
            "seed": args.seed,
            "f16_wmma_roof_tflops": F16_WMMA_ROOF_TFLOPS,
        },
        "results": results,
        "gates": {
            "all_finite": all_finite,
            "all_deterministic": all_deterministic,
            "all_activation_sensitive": all_sensitive,
            "teardown_exact": teardown,
            "iu4_candidate_run": False,
            "model_logit_quality_gate_required": False,
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
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "pair_payload_bytes": pair_bytes,
                "rows": [
                    {
                        "m": row["rows"],
                        "median_ms": row["timing"]["pooled"]["median_ms"],
                        "effective_gbps": row["performance"]["effective_payload_gbps"],
                        "executed_tflops": row["performance"]["executed_tflops"],
                        "percent_of_f16_roof": row["performance"]["percent_of_f16_wmma_roof"],
                    }
                    for row in results
                ],
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "accepted_diagnostic_control" else 1


if __name__ == "__main__":
    raise SystemExit(main())
