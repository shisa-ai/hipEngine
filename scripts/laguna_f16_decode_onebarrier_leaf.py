#!/usr/bin/env python3
"""Screen exact one-barrier source-F16 GEMV on Laguna decode shapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    build_laguna_f16_projection,
    laguna_f16w_gemv_bf16_bf16_out,
    laguna_f16w_gemv_bf16_f32_out,
    laguna_f16w_onebarrier_gemv_bf16_bf16_out,
    laguna_f16w_onebarrier_gemv_bf16_f32_out,
    laguna_f16w_triple_gemv_bf16_f32_out,
    laguna_f16w_triple_onebarrier_gemv_bf16_f32_out,
)
from hipengine.loading.materialize import float_array_to_bf16_bits


_SHAPES = (
    ("full_qkv", 3072, (6144, 1024, 1024), "triple_f32", 12),
    ("swa_qkv", 3072, (9216, 1024, 1024), "triple_f32", 36),
    ("full_gate", 3072, (48,), "single_f32", 12),
    ("swa_gate", 3072, (72,), "single_f32", 36),
    ("full_output", 6144, (3072,), "single_bf16", 12),
    ("swa_output", 9216, (3072,), "single_bf16", 36),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _upload(runtime, host: np.ndarray):
    device = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(
        device,
        host_array_ptr(host),
        host.nbytes,
        runtime=runtime,
    )
    return device


def _download(runtime, device, width: int, dtype) -> np.ndarray:
    host = np.empty((1, width), dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        device,
        host.nbytes,
        runtime=runtime,
    )
    return host


def _time_ms(runtime, launch: Callable[[], None], burst: int) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop)) / burst
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.samples <= 0 or args.warmups < 0 or args.burst <= 0:
        raise ValueError("samples/burst must be positive and warmups non-negative")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    runtime = get_hip_runtime()
    library = build_laguna_f16_projection(
        load=True,
        require_cached=args.require_cached_build,
    )
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, object]] = []
    for name, in_features, widths, kind, calls_per_token in _SHAPES:
        allocations = []
        try:
            x = float_array_to_bf16_bits(
                rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
            )
            weights = tuple(
                rng.normal(0.0, 0.1, size=(width, in_features)).astype(
                    np.float16
                )
                for width in widths
            )
            dx = _upload(runtime, x)
            dweights = tuple(_upload(runtime, weight) for weight in weights)
            allocations.extend((dx, *dweights))
            output_dtype = np.uint16 if kind == "single_bf16" else np.float32
            itemsize = np.dtype(output_dtype).itemsize
            baseline_outputs = tuple(
                malloc(width * itemsize, runtime=runtime) for width in widths
            )
            candidate_outputs = tuple(
                malloc(width * itemsize, runtime=runtime) for width in widths
            )
            allocations.extend((*baseline_outputs, *candidate_outputs))

            if kind == "triple_f32":
                baseline = lambda: laguna_f16w_triple_gemv_bf16_f32_out(
                    dx.ptr,
                    *(weight.ptr for weight in dweights),
                    *(out.ptr for out in baseline_outputs),
                    1,
                    in_features,
                    *widths,
                    library=library,
                    runtime=runtime,
                )
                candidate = (
                    lambda: laguna_f16w_triple_onebarrier_gemv_bf16_f32_out(
                        dx.ptr,
                        *(weight.ptr for weight in dweights),
                        *(out.ptr for out in candidate_outputs),
                        1,
                        in_features,
                        *widths,
                        library=library,
                        runtime=runtime,
                    )
                )
            else:
                baseline_fn = (
                    laguna_f16w_gemv_bf16_bf16_out
                    if kind == "single_bf16"
                    else laguna_f16w_gemv_bf16_f32_out
                )
                candidate_fn = (
                    laguna_f16w_onebarrier_gemv_bf16_bf16_out
                    if kind == "single_bf16"
                    else laguna_f16w_onebarrier_gemv_bf16_f32_out
                )
                baseline = lambda: baseline_fn(
                    dx.ptr,
                    dweights[0].ptr,
                    baseline_outputs[0].ptr,
                    1,
                    in_features,
                    widths[0],
                    library=library,
                    runtime=runtime,
                )
                candidate = lambda: candidate_fn(
                    dx.ptr,
                    dweights[0].ptr,
                    candidate_outputs[0].ptr,
                    1,
                    in_features,
                    widths[0],
                    library=library,
                    runtime=runtime,
                )

            baseline()
            candidate()
            runtime.device_synchronize()
            digests = []
            for baseline_out, candidate_out, width in zip(
                baseline_outputs, candidate_outputs, widths, strict=True
            ):
                baseline_host = _download(
                    runtime, baseline_out, width, output_dtype
                )
                candidate_host = _download(
                    runtime, candidate_out, width, output_dtype
                )
                np.testing.assert_array_equal(candidate_host, baseline_host)
                digests.append(hashlib.sha256(candidate_host.tobytes()).hexdigest())

            for _ in range(args.warmups):
                baseline()
                candidate()
            runtime.device_synchronize()
            samples = {"baseline": [], "onebarrier": []}
            for repeat in range(args.samples):
                order = (
                    (("baseline", baseline), ("onebarrier", candidate))
                    if repeat % 2 == 0
                    else (("onebarrier", candidate), ("baseline", baseline))
                )
                for mode, launch in order:
                    samples[mode].append(_time_ms(runtime, launch, args.burst))
            baseline_ms = statistics.median(samples["baseline"])
            candidate_ms = statistics.median(samples["onebarrier"])
            weight_bytes = sum(weight.nbytes for weight in weights)
            results.append(
                {
                    "name": name,
                    "kind": kind,
                    "in_features": in_features,
                    "out_features": list(widths),
                    "calls_per_token": calls_per_token,
                    "weight_bytes": weight_bytes,
                    "byte_exact": True,
                    "output_sha256": digests,
                    "baseline": _summary(samples["baseline"]),
                    "onebarrier": _summary(samples["onebarrier"]),
                    "latency_change_percent": (
                        candidate_ms / baseline_ms - 1.0
                    )
                    * 100.0,
                    "speedup": baseline_ms / candidate_ms,
                    "baseline_effective_weight_gb_s": weight_bytes
                    / (baseline_ms * 1.0e6),
                    "onebarrier_effective_weight_gb_s": weight_bytes
                    / (candidate_ms * 1.0e6),
                }
            )
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)

    baseline_family_ms = sum(
        row["baseline"]["median_ms"] * row["calls_per_token"] for row in results
    )
    candidate_family_ms = sum(
        row["onebarrier"]["median_ms"] * row["calls_per_token"]
        for row in results
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_f16_decode_onebarrier_leaf",
        "status": "candidate",
        "hardware": {
            "backend": "hip_gfx1151",
            "arch": os.environ.get("HIPENGINE_HIP_ARCH", ""),
            "queue_policy": os.environ.get("GPU_MAX_HW_QUEUES", ""),
        },
        "source": {
            "revision": _revision(),
            "kernel": (
                "hipengine/kernels/hip_gfx1100/linear/"
                "laguna_f16_projection.hip"
            ),
        },
        "protocol": {
            "rows": 1,
            "samples": args.samples,
            "warmups_per_mode": args.warmups,
            "burst": args.burst,
            "order": "counterbalanced by repetition",
            "data": "deterministic nonzero BF16 activations and FP16 weights",
        },
        "results": results,
        "modeled_family": {
            "baseline_ms_per_token": baseline_family_ms,
            "onebarrier_ms_per_token": candidate_family_ms,
            "latency_change_percent": (
                candidate_family_ms / baseline_family_ms - 1.0
            )
            * 100.0,
            "modeled_saving_ms_per_token": baseline_family_ms
            - candidate_family_ms,
        },
        "gate": {
            "all_shapes_byte_exact": True,
            "all_shapes_faster": all(
                row["latency_change_percent"] < 0.0 for row in results
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    payload = json.dumps(run(args), indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
