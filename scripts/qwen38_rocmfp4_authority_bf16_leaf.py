#!/usr/bin/env python3
"""Benchmark the lossless expanded-BF16 ROCmFP4 authority prefill owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    build_dense_gemv,
    dense_prefill_gemm_out_bf16,
    dense_prefill_wmma_out_bf16,
)
from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits
from scripts.qwen38_iu4_prefill_gate_up_leaf import (
    _event_ms,
    _read_bf16,
    _timing_summary,
    _upload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (512, 1024, 2048)
DEFAULT_SHAPES = (
    (5120, 1024),
    (5120, 6144),
    (5120, 10240),
    (5120, 12288),
    (5120, 17408),
    (6144, 5120),
    (17408, 5120),
)
SOURCE_COMMIT = "ciru-ai/ROCmFPX@e1da26bb8237fb5642488a2387efc793b141aae5"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    parser.add_argument(
        "--shapes",
        default=",".join(f"{k}x{n}" for k, n in DEFAULT_SHAPES),
        help="comma-separated KxN matrix shapes",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0xE4D)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or min(values) <= 0:
        raise ValueError("rows must be positive")
    return values


def _parse_shapes(raw: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        left, separator, right = item.strip().lower().partition("x")
        if not separator:
            raise ValueError(f"shape must use KxN syntax: {item!r}")
        shape = (int(left), int(right))
        if min(shape) <= 0 or shape[0] % 32 or shape[1] % 128:
            raise ValueError(f"shape is not dense-WMMA aligned: {shape}")
        shapes.append(shape)
    if not shapes:
        raise ValueError("at least one shape is required")
    return tuple(shapes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _source_state() -> dict[str, object]:
    paths = (
        "hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip",
        "hipengine/kernels/hip_gfx1151/__init__.py",
        "hipengine/loading/qwen35_gguf_materialize.py",
        "hipengine/runtime/gguf_linear.py",
        "scripts/qwen38_rocmfp4_authority_bf16_leaf.py",
    )
    return {
        "base_commit": _git_head(),
        "files": {
            path: _sha256(ROOT / path)
            for path in paths
        },
        "scoped_status": subprocess.check_output(
            ["git", "status", "--short", "--", *paths],
            cwd=ROOT,
            text=True,
        ).splitlines(),
    }


def _screen(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    runtime,
    library,
    rng: np.random.Generator,
    weight: DeviceBuffer,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    x = f32_to_bf16_bits(
        rng.normal(0.0, 0.02, size=(rows, in_features)).astype(np.float32)
    )
    buffers: list[DeviceBuffer] = []
    try:
        x_device = _upload(runtime, x)
        control_out = malloc(rows * out_features * 2, runtime=runtime)
        candidate_out = malloc(rows * out_features * 2, runtime=runtime)
        buffers.extend((x_device, control_out, candidate_out))

        def control() -> None:
            dense_prefill_gemm_out_bf16(
                x_device.ptr,
                weight.ptr,
                control_out.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        def candidate() -> None:
            dense_prefill_wmma_out_bf16(
                x_device.ptr,
                weight.ptr,
                candidate_out.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()

        timings: dict[str, list[float]] = {"control": [], "candidate": []}
        candidate_wins = 0
        for sample in range(samples):
            launches: list[tuple[str, Callable[[], None]]] = [
                ("control", control),
                ("candidate", candidate),
            ]
            if sample & 1:
                launches.reverse()
            paired: dict[str, float] = {}
            for name, launch in launches:
                paired[name] = _event_ms(runtime, launch)
                timings[name].append(paired[name])
            candidate_wins += int(paired["candidate"] < paired["control"])

        control()
        candidate()
        runtime.device_synchronize()
        control_f32 = bf16_bits_to_f32(
            _read_bf16(runtime, control_out, (rows, out_features))
        )
        candidate_f32 = bf16_bits_to_f32(
            _read_bf16(runtime, candidate_out, (rows, out_features))
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    difference = np.abs(control_f32 - candidate_f32)
    control_norm = max(float(np.linalg.norm(control_f32)), 1.0e-12)
    control_summary = _timing_summary(timings["control"])
    candidate_summary = _timing_summary(timings["candidate"])
    control_ms = float(control_summary["median_ms"])
    candidate_ms = float(candidate_summary["median_ms"])
    return {
        "rows": rows,
        "in_features": in_features,
        "out_features": out_features,
        "control": control_summary,
        "candidate": candidate_summary,
        "candidate_wins": candidate_wins,
        "pair_count": samples,
        "speedup": control_ms / candidate_ms,
        "latency_reduction_percent": (1.0 - candidate_ms / control_ms) * 100.0,
        "correctness": {
            "finite": bool(np.isfinite(candidate_f32).all()),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "relative_l2": float(
                np.linalg.norm(control_f32 - candidate_f32) / control_norm
            ),
            "row_top1_agreement": float(
                np.mean(control_f32.argmax(axis=1) == candidate_f32.argmax(axis=1))
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    rows = _parse_csv_ints(args.rows)
    shapes = _parse_shapes(args.shapes)
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("warmups must be non-negative and samples positive")

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    runtime = get_hip_runtime()
    library = build_dense_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=bool(args.require_cached_build),
    )
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, object]] = []
    for in_features, out_features in shapes:
        weight_bits = f32_to_bf16_bits(
            rng.normal(
                0.0,
                0.01,
                size=(out_features, in_features),
            ).astype(np.float32)
        )
        weight = _upload(runtime, weight_bits)
        del weight_bits
        try:
            for row_count in rows:
                result = _screen(
                    rows=row_count,
                    in_features=in_features,
                    out_features=out_features,
                    runtime=runtime,
                    library=library,
                    rng=rng,
                    weight=weight,
                    warmups=args.warmups,
                    samples=args.samples,
                )
                print(json.dumps(result), flush=True)
                results.append(result)
        finally:
            free(weight, runtime=runtime)

    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "gfx1151_qwen38_rocmfp4_authority_dense_bf16_leaf",
        "status": "selected_explicit_authority_path",
        "performance_claim": True,
        "source_lineage": SOURCE_COMMIT,
        "source_state": _source_state(),
        "hardware": {
            "host": platform.node(),
            "machine": platform.machine(),
            "gpu": "AMD Radeon 8060S",
            "arch": "gfx1151",
        },
        "protocol": {
            "rows": list(rows),
            "shapes_kxn": [list(shape) for shape in shapes],
            "warmups": args.warmups,
            "samples": args.samples,
            "seed": args.seed,
            "control": "dense_prefill_gemm_out_bf16",
            "candidate": "dense_prefill_wmma_out_bf16",
            "compiler_version_file": (
                None
                if args.compiler_version_file is None
                else str(args.compiler_version_file)
            ),
            "compiler_version": compiler_version,
            "require_cached_build": bool(args.require_cached_build),
        },
        "results": results,
        "acceptance": {
            "all_finite": all(row["correctness"]["finite"] for row in results),
            "all_faster": all(float(row["speedup"]) > 1.0 for row in results),
            "all_paired_wins": all(
                int(row["candidate_wins"]) == int(row["pair_count"])
                for row in results
            ),
            "max_relative_l2": max(
                float(row["correctness"]["relative_l2"]) for row in results
            ),
            "min_row_top1_agreement": min(
                float(row["correctness"]["row_top1_agreement"])
                for row in results
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
