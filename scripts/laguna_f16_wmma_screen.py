#!/usr/bin/env python3
"""Screen the direct-resident Laguna source-F16 WMMA leaf at production shapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    build_laguna_f16_projection_prefill,
    laguna_f16w_tiled_bf16_bf16_out,
    laguna_f16w_tiled_bf16_f32_out,
    laguna_f16w_wmma_bf16_bf16_out,
    laguna_f16w_wmma_bf16_f32_out,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from scripts.laguna_f16_library_ceiling import (
    DEFAULT_ROWS,
    _FAMILIES,
    _SHAPES,
    _allocate_buffers,
    _event_sample,
    _offset,
    _parse_rows,
)
from scripts.laguna_target_ar_bench import _compiler_version, _repo_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_CEILING = (
    ROOT / "benchmarks/results/2026-07-23-gfx1151-laguna-f16-library-ceiling.json"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-screen.json"
)
_MODES = ("exact", "wmma")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--library-ceiling", type=Path, default=DEFAULT_LIBRARY_CEILING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _family_launches(
    family: str,
    rows: int,
    runtime: HipRuntime,
    buffers,
    library,
) -> dict[str, Callable[[], None]]:
    q_name = f"{family}_q"
    gate_name = f"{family}_gate"
    o_name = f"{family}_o"
    q_width = _SHAPES[q_name][1]
    o_k = _SHAPES[o_name][0]
    kv_width = _SHAPES["kv"][1]
    q_out = buffers.out_f32.ptr
    k_out = _offset(q_out, rows, q_width)
    v_out = _offset(k_out, rows, kv_width)
    gate_out = _offset(v_out, rows, kv_width)

    def launch(single_f32, single_bf16) -> None:
        for name, out_ptr in (
            (q_name, q_out),
            ("kv", k_out),
            ("kv", v_out),
            (gate_name, gate_out),
        ):
            in_features, out_features = _SHAPES[name]
            single_f32(
                buffers.x_bf16.ptr,
                buffers.weight_fp16.ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        single_bf16(
            buffers.x_bf16.ptr,
            buffers.weight_fp16.ptr,
            buffers.out_bf16.ptr,
            rows,
            o_k,
            3072,
            library=library,
            runtime=runtime,
        )

    return {
        "exact": lambda: launch(
            laguna_f16w_tiled_bf16_f32_out,
            laguna_f16w_tiled_bf16_bf16_out,
        ),
        "wmma": lambda: launch(
            laguna_f16w_wmma_bf16_f32_out,
            laguna_f16w_wmma_bf16_bf16_out,
        ),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _math_smoke(
    runtime: HipRuntime,
    buffers,
    library,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows, in_features, out_features = 16, 64, 35
    x = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    weight = rng.normal(
        0.0, 0.05, size=(out_features, in_features)
    ).astype(np.float16)
    copy_host_to_device(buffers.x_bf16, host_array_ptr(x), x.nbytes, runtime=runtime)
    copy_host_to_device(
        buffers.weight_fp16,
        host_array_ptr(weight),
        weight.nbytes,
        runtime=runtime,
    )
    laguna_f16w_tiled_bf16_f32_out(
        buffers.x_bf16.ptr,
        buffers.weight_fp16.ptr,
        buffers.out_f32.ptr,
        rows,
        in_features,
        out_features,
        library=library,
        runtime=runtime,
    )
    runtime.device_synchronize()
    exact = np.empty((rows, out_features), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(exact), buffers.out_f32, exact.nbytes, runtime=runtime
    )
    laguna_f16w_wmma_bf16_f32_out(
        buffers.x_bf16.ptr,
        buffers.weight_fp16.ptr,
        buffers.out_f32.ptr,
        rows,
        in_features,
        out_features,
        library=library,
        runtime=runtime,
    )
    runtime.device_synchronize()
    wmma = np.empty_like(exact)
    copy_device_to_host(
        host_array_ptr(wmma), buffers.out_f32, wmma.nbytes, runtime=runtime
    )
    p = _softmax(exact.astype(np.float64))
    q = _softmax(wmma.astype(np.float64))
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    top1 = float(np.mean(np.argmax(exact, axis=1) == np.argmax(wmma, axis=1)))
    max_kl = float(np.max(kl))
    max_abs = float(np.max(np.abs(wmma - exact)))
    return {
        "pass": bool(
            np.all(np.isfinite(wmma))
            and max_kl <= 0.05
            and top1 >= 0.9
        ),
        "shape_mkn": [rows, in_features, out_features],
        "max_abs": max_abs,
        "max_kl": max_kl,
        "top1_agreement": top1,
    }


def _load_library_ceiling(path: Path, rows: Sequence[int]) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "hipengine_laguna_f16_library_ceiling" or not payload.get(
        "pass"
    ):
        raise ValueError("library ceiling must be a passing Laguna F16 ceiling artifact")
    shapes = payload["summary"]["shapes"]
    return {
        row: {
            family: float(
                shapes[str(row)]["families"][family]["hipblaslt_inclusive"][
                    "gpu_ms_median"
                ]
            )
            for family in _FAMILIES
        }
        for row in rows
    }


def _summarize(
    rows: Sequence[int],
    samples: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    wall_samples: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    library: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    failed: list[str] = []
    weighted_exact = 0.0
    weighted_wmma = 0.0
    for row in rows:
        families: dict[str, Any] = {}
        for family in _FAMILIES:
            exact_gpu = tuple(float(value) for value in samples[row][family]["exact"])
            wmma_gpu = tuple(float(value) for value in samples[row][family]["wmma"])
            exact_wall = tuple(
                float(value) for value in wall_samples[row][family]["exact"]
            )
            wmma_wall = tuple(
                float(value) for value in wall_samples[row][family]["wmma"]
            )
            if not exact_gpu or len(exact_gpu) != len(wmma_gpu):
                raise ValueError(f"rows={row}/{family} requires equal non-empty samples")
            exact = statistics.median(exact_gpu)
            wmma = statistics.median(wmma_gpu)
            ratio = exact / wmma
            if ratio <= 1.0:
                failed.append(f"rows_{row}_{family}_wmma_not_faster")
            library_ms = float(library[row][family])
            families[family] = {
                "exact_gpu_ms_samples": list(exact_gpu),
                "exact_gpu_ms_median": exact,
                "wmma_gpu_ms_samples": list(wmma_gpu),
                "wmma_gpu_ms_median": wmma,
                "exact_wall_ms_samples": list(exact_wall),
                "exact_wall_ms_median": statistics.median(exact_wall),
                "wmma_wall_ms_samples": list(wmma_wall),
                "wmma_wall_ms_median": statistics.median(wmma_wall),
                "wmma_speedup_vs_exact": ratio,
                "library_inclusive_gpu_ms_median": library_ms,
                "wmma_speedup_vs_library_inclusive": library_ms / wmma,
            }
            if row == 128:
                multiplier = 12 if family == "full" else 36
                weighted_exact += multiplier * exact
                weighted_wmma += multiplier * wmma
        shapes[str(row)] = {"rows": row, "families": families}
    weighted_speedup = weighted_exact / weighted_wmma
    if weighted_speedup < 2.0:
        failed.append("m128_weighted_speedup_below_2x")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "shapes": shapes,
        "m128_weighted_projection_sum": {
            "exact_ms": weighted_exact,
            "wmma_ms": weighted_wmma,
            "speedup": weighted_speedup,
        },
        "policy": (
            "quality-lane screen requires every production row/family faster "
            "than exact and the 12-full/36-SWA M128 sum at least 2x faster"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != DEFAULT_ROWS:
        raise ValueError(f"retained Laguna F16 WMMA screen requires rows {DEFAULT_ROWS}")
    if args.backend != "hip_gfx1151":
        raise ValueError("retained Laguna F16 WMMA screen is qualified only for hip_gfx1151")
    if args.iterations <= 0 or args.repetitions < 3 or args.warmups < 0:
        raise ValueError("iterations must be positive, repetitions >=3, and warmups non-negative")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna F16 WMMA screen requires a clean tracked worktree")

    compiler_version = _compiler_version(args.compiler_version_file)
    library = build_laguna_f16_projection_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    buffers = _allocate_buffers(runtime, rows, 1)
    samples = {
        row: {family: {mode: [] for mode in _MODES} for family in _FAMILIES}
        for row in rows
    }
    wall_samples = {
        row: {family: {mode: [] for mode in _MODES} for family in _FAMILIES}
        for row in rows
    }
    smoke: dict[str, Any] = {"pass": False}
    try:
        smoke = _math_smoke(runtime, buffers, library, args.seed)
        if not smoke["pass"]:
            raise RuntimeError("Laguna source-F16 WMMA math smoke failed")
        for buffer in buffers.all():
            runtime.memset(buffer.ptr, 0, buffer.nbytes)
        for row_index, row in enumerate(rows):
            for family_index, family in enumerate(_FAMILIES):
                launches = _family_launches(family, row, runtime, buffers, library)
                for _ in range(args.warmups):
                    for mode in _MODES:
                        launches[mode]()
                runtime.device_synchronize()
                for repetition in range(args.repetitions):
                    forward = (repetition + row_index + family_index) % 2 == 0
                    order = _MODES if forward else tuple(reversed(_MODES))
                    for mode in order:
                        gpu_ms, wall_ms = _event_sample(
                            runtime, launches[mode], args.iterations
                        )
                        samples[row][family][mode].append(gpu_ms)
                        wall_samples[row][family][mode].append(wall_ms)
    finally:
        runtime.device_synchronize()
        for buffer in reversed(buffers.all()):
            free(buffer, runtime=runtime)

    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna F16 WMMA screen")
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    library_ceiling = _load_library_ceiling(args.library_ceiling, rows)
    summary = _summarize(rows, samples, wall_samples, library_ceiling)
    if not smoke["pass"]:
        summary["failed_checks"].append("math_smoke_failed")
    if not recovered:
        summary["failed_checks"].append("tracked_ownership_not_recovered")
    summary["pass"] = not summary["failed_checks"]
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch="gfx1151",
        quant="source_f16_projection_geometry",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_f16_wmma_screen",
        timing_protocol="hip_event_counterbalanced_exact_vs_wmma_family_sequences",
        warmups=args.warmups,
        repetitions=args.repetitions,
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_f16_wmma_screen",
        "status": "quality_lane_admitted" if summary["pass"] else "rejected",
        "pass": bool(summary["pass"]),
        "performance_claim": False,
        "performance_claim_scope": (
            "zero-data production geometry screen only; full model quality, "
            "trajectory, and E2E admission remains required"
        ),
        "provenance": provenance,
        "repo": repo,
        "platform": {
            "backend": args.backend,
            "target_arch": "gfx1151",
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "rows": list(rows),
            "projection_shapes_kn": {
                name: list(shape) for name, shape in _SHAPES.items()
            },
            "families": {
                "full": ["full_q", "kv", "kv", "full_gate", "full_o"],
                "swa": ["swa_q", "kv", "kv", "swa_gate", "swa_o"],
            },
            "iterations_per_sample": args.iterations,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "timed_order": "counterbalanced exact/WMMA by row/family/repetition",
            "data": "zero timing buffers plus a seeded nonzero M16/K64/N35 math smoke",
            "library_ceiling": str(args.library_ceiling),
        },
        "summary": summary,
        "correctness": {
            "pass": smoke["pass"] and recovered,
            "math_smoke": smoke,
            "tracked_returned_to_baseline": recovered,
            "full_model_quality_gate_required": True,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "The candidate reads the existing row-major resident F16 allocation directly.",
            (
                "BF16 activations convert to F16 in registers; there is no "
                "persistent sidecar or inference-time repack."
            ),
            (
                "Rows==1 and runtime defaults remain on the exact GEMV/tiled "
                "routes during this screen."
            ),
            (
                "hipBLASLt values are imported from the retained clean "
                "library-ceiling artifact and are not remeasured."
            ),
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
