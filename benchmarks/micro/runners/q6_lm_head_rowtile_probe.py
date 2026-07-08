#!/usr/bin/env python3
"""Run HIP Q6_K T16 rowtile versus Vulkan Q6_K X8 lm-head-shaped probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
VULKAN_RUNNER = MICRO_ROOT / "runners" / "q6_x8_real_slice.py"
HIP_Q6_T16_SOURCE = REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q6_k_t16_gemv.hip"
VULKAN_Q6_X8_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q6_x8_selected_down.comp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
BENCH_NAME = "q6_lm_head_rowtile_probe"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q6-lm-head-rowtile")
QK_K = 256
Q6_T16_COLS = 16
Q6_T16_BLOCK_BYTES = 3360

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_collect_env_module():
    spec = importlib.util.spec_from_file_location("micro_collect_env", COLLECT_ENV)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load environment collector: {COLLECT_ENV}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_collect_env_module()
    return collector.collect_environment(
        repo_root=REPO_ROOT,
        include_device_probes=not args.skip_device_probes,
        timeout_s=args.env_timeout_s,
        max_output_chars=args.env_max_output_chars,
    )


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _run_command(command: list[str], *, cwd: Path, echo: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if echo and completed.stdout:
        sys.stdout.write(completed.stdout)
    if echo and completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed


def _source_record(environment: dict[str, Any], source_hash: str) -> dict[str, Any]:
    repo = environment.get("repo")
    if not isinstance(repo, dict):
        repo = {}
    return {
        "repo": str(repo.get("root") or REPO_ROOT),
        "branch": str(repo.get("branch") or ""),
        "commit": str(repo.get("commit") or ""),
        "dirty": bool(repo.get("dirty")),
        "source_hash": source_hash,
    }


def _parse_csv_u32(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text}")
    return values


def _parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes = []
    for item in text.split(","):
        if not item:
            continue
        left, right = item.lower().split("x", 1)
        in_features = int(left)
        out_features = int(right)
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"invalid shape: {item}")
        if in_features % QK_K != 0 or out_features % Q6_T16_COLS != 0:
            raise ValueError(f"shape must be divisible by 256x16: {item}")
        shapes.append((in_features, out_features))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _make_x_bf16(rows: int, in_features: int, input_scale: float) -> np.ndarray:
    rng = np.random.default_rng(0xA510 + rows * 17 + in_features)
    x = rng.standard_normal((rows, in_features), dtype=np.float32) * np.float32(input_scale)
    return np.ascontiguousarray(_f32_to_bf16_bits(x))


def _make_q6_t16_tiles(in_features: int, out_features: int) -> np.ndarray:
    blocks = in_features // QK_K
    out_tiles = out_features // Q6_T16_COLS
    rng = np.random.default_rng(0xA606)
    tiles = rng.integers(
        0,
        256,
        size=(out_tiles, blocks, Q6_T16_BLOCK_BYTES),
        dtype=np.uint8,
    )
    d = (
        np.full((out_tiles, blocks, Q6_T16_COLS), np.float16(0.0078125), dtype=np.float16)
        .view(np.uint8)
        .reshape(out_tiles, blocks, Q6_T16_COLS * 2)
    )
    tiles[:, :, : Q6_T16_COLS * 2] = d
    return np.ascontiguousarray(tiles)


def _rowtile_chunks(rows: int, max_chunk: int = 6) -> tuple[int, ...]:
    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows <= max_chunk:
        return (rows,)
    chunks: list[int] = []
    remaining = rows
    while remaining > 0:
        if remaining <= max_chunk:
            if remaining == 1 and chunks:
                chunks[-1] -= 1
                chunks.append(2)
            else:
                chunks.append(remaining)
            break
        take = max_chunk
        if remaining - take == 1:
            take -= 1
        chunks.append(take)
        remaining -= take
    return tuple(chunks)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    t = pos - lo
    return sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t


def _stats(samples_us: list[float]) -> dict[str, Any]:
    return {
        "median_us": statistics.median(samples_us),
        "median_ms": statistics.median(samples_us) / 1000.0,
        "p05_us": _percentile(samples_us, 0.05),
        "p95_us": _percentile(samples_us, 0.95),
        "min_us": min(samples_us) if samples_us else 0.0,
        "max_us": max(samples_us) if samples_us else 0.0,
        "samples_us": samples_us,
    }


def _time_hip(runtime: Any, fn: Callable[[], None], *, reps: int, warmup: int, samples: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    runtime.device_synchronize()
    start = runtime.event_create()
    stop = runtime.event_create()
    samples_us: list[float] = []
    try:
        for _ in range(samples):
            runtime.event_record(start)
            for _ in range(reps):
                fn()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            elapsed_us = runtime.event_elapsed_time_ms(start, stop) * 1000.0 / reps
            if elapsed_us <= 0.0:
                raise RuntimeError(f"non-positive HIP event elapsed time: {elapsed_us}")
            samples_us.append(elapsed_us)
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
    return _stats(samples_us)


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_gemv_decode_bf16_f32_out,
        gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
    )

    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), HIP_Q6_T16_SOURCE])
    runtime = get_hip_runtime()
    library = build_gguf_q6_k_t16_gemv(load=True)
    rows_out: list[dict[str, Any]] = []

    for in_features, out_features in _parse_shapes(args.shapes):
        tiles = _make_q6_t16_tiles(in_features, out_features)
        tile_buf = malloc(tiles.nbytes, runtime=runtime)
        runtime.memcpy(tile_buf.ptr, host_array_ptr(tiles), tiles.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        try:
            for rows in _parse_csv_u32(args.rows_list):
                x_bits = _make_x_bf16(rows, in_features, args.input_scale)
                x_buf = malloc(x_bits.nbytes, runtime=runtime)
                out_buf = malloc(rows * out_features * 4, runtime=runtime)
                ref_buf = malloc(rows * out_features * 4, runtime=runtime)
                runtime.memcpy(x_buf.ptr, host_array_ptr(x_bits), x_bits.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
                try:
                    chunks = _rowtile_chunks(rows)

                    def per_row_decode() -> None:
                        for row in range(rows):
                            gguf_q6_k_t16_gemv_decode_bf16_f32_out(
                                x_buf.ptr + row * in_features * 2,
                                tile_buf.ptr,
                                ref_buf.ptr + row * out_features * 4,
                                1,
                                in_features,
                                out_features,
                                library=library,
                                runtime=runtime,
                            )

                    def rowtile_chunked() -> None:
                        row_offset = 0
                        for chunk_rows in chunks:
                            fn = (
                                gguf_q6_k_t16_gemv_decode_bf16_f32_out
                                if chunk_rows == 1
                                else gguf_q6_k_t16_gemv_rowtile_bf16_f32_out
                            )
                            fn(
                                x_buf.ptr + row_offset * in_features * 2,
                                tile_buf.ptr,
                                out_buf.ptr + row_offset * out_features * 4,
                                chunk_rows,
                                in_features,
                                out_features,
                                library=library,
                                runtime=runtime,
                            )
                            row_offset += chunk_rows

                    per_row_decode()
                    rowtile_chunked()
                    runtime.device_synchronize()
                    ref = np.empty(rows * out_features, dtype=np.float32)
                    got = np.empty(rows * out_features, dtype=np.float32)
                    runtime.memcpy(host_array_ptr(ref), ref_buf.ptr, ref.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                    runtime.memcpy(host_array_ptr(got), out_buf.ptr, got.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                    max_abs = float(np.max(np.abs(ref - got)))
                    correctness = {
                        "oracle": "HIP Q6_K T16 rowtile chunked output versus per-row decode output",
                        "max_abs": max_abs,
                        "pass": max_abs == 0.0,
                    }
                    if not correctness["pass"]:
                        raise RuntimeError(
                            f"HIP Q6 T16 rowtile correctness failed for "
                            f"{in_features}x{out_features} rows={rows}: {correctness}"
                        )
                    timing = _time_hip(
                        runtime,
                        rowtile_chunked,
                        reps=args.reps,
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    rows_out.append(
                        {
                            "backend": "hip",
                            "variant": "q6_t16_rowtile_chunked_f32",
                            "rows": rows,
                            "in_features": in_features,
                            "out_features": out_features,
                            "chunks": list(chunks),
                            "q6_lm_head_median_us": timing["median_us"],
                            "timing": {"q6_t16_rowtile_chunked_f32": timing},
                            "correctness": correctness,
                            "correctness_pass": correctness["pass"],
                        }
                    )
                finally:
                    free(ref_buf, runtime=runtime)
                    free(out_buf, runtime=runtime)
                    free(x_buf, runtime=runtime)
        finally:
            free(tile_buf, runtime=runtime)

    return _json_safe(
        {
            "schema_version": 1,
            "kind": "hipengine_micro_result",
            "bench": BENCH_NAME,
            "backend": "hip",
            "classification": "real_slice_probe",
            "source": _source_record(environment, source_hash),
            "hardware": {
                "gpu": args.hardware_gpu,
                "gfx_arch": args.gfx_arch or os.environ.get("HIPENGINE_HIP_ARCH"),
            },
            "environment": None if args.environment_ref else environment,
            "environment_ref": str(args.environment_ref) if args.environment_ref else None,
            "artifact_ref": str(args.out) if args.out else None,
            "wrapper": {
                "schema": "hipengine.micro.q6_lm_head_rowtile_probe.v1",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "method": "HIP events around production-style Q6_K T16 rowtile chunks; transfer and build excluded",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "all_pass": all(bool(row.get("correctness_pass")) for row in rows_out),
                "rows": len(rows_out),
            },
            "notes": (
                "HIP production-style Q6_K T16 lm-head rowtile baseline. "
                "Rows larger than six use the same small-B chunking policy as the runtime."
            ),
        }
    )


def _run_vulkan(args: argparse.Namespace) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), VULKAN_RUNNER, VULKAN_Q6_X8_SHADER, VULKAN_QUANT_SHADER])
    args.build_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for local_size in _parse_csv_u32(args.local_sizes):
        for in_features, out_features in _parse_shapes(args.shapes):
            for rows in _parse_csv_u32(args.rows_list):
                run_dir = args.build_dir / f"vulkan_ls{local_size}_{in_features}x{out_features}_r{rows}"
                raw_out = run_dir / "q6_x8.json"
                command = [
                    sys.executable,
                    str(VULKAN_RUNNER),
                    "--out",
                    str(raw_out),
                    "--rows",
                    str(rows),
                    "--experts",
                    "1",
                    "--in-features",
                    str(in_features),
                    "--out-features",
                    str(out_features),
                    "--input-scale",
                    str(args.input_scale),
                    "--local-size",
                    str(local_size),
                    "--reps",
                    str(args.reps),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--device-index",
                    str(args.device_index),
                    "--build-dir",
                    str(run_dir),
                ]
                completed = _run_command(command, cwd=REPO_ROOT)
                commands.append(
                    {
                        "kind": "run_vulkan_q6_x8_runner",
                        "local_size": local_size,
                        "shape": f"{in_features}x{out_features}",
                        "rows": rows,
                        "command": command,
                        "returncode": completed.returncode,
                    }
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"Vulkan Q6 lm-head-shaped run failed: {' '.join(command)}")
                raw = json.loads(raw_out.read_text(encoding="utf-8"))
                timing = raw.get("timing", {})
                correctness = raw.get("correctness_vs_cpu", {})
                row = {
                    "backend": "vulkan",
                    "variant": "q6_x8_q8_1_dp4a_full_output_bf16",
                    "rows": rows,
                    "in_features": in_features,
                    "out_features": out_features,
                    "local_size": local_size,
                    "q8_1_quantize_median_us": timing["q8_1_quantize"]["median_us"],
                    "q6_x8_dot_median_us": timing["x8_selected_dp4a_dot_prequantized"]["median_us"],
                    "q6_x8_quantize_plus_dot_median_us": timing["x8_selected_dp4a_quantize_plus_dot"]["median_us"],
                    "timing": timing,
                    "correctness": correctness,
                    "correctness_pass": bool(correctness.get("pass")),
                    "hardware": raw.get("hardware", {}),
                }
                rows_out.append(row)

    return _json_safe(
        {
            "schema_version": 1,
            "kind": "hipengine_micro_result",
            "bench": BENCH_NAME,
            "backend": "vulkan",
            "classification": "real_slice_probe",
            "source": _source_record(environment, source_hash),
            "hardware": {
                "gpu": args.hardware_gpu,
                "gfx_arch": args.gfx_arch,
                "device": rows_out[0].get("hardware", {}) if rows_out else {},
            },
            "environment": None if args.environment_ref else environment,
            "environment_ref": str(args.environment_ref) if args.environment_ref else None,
            "artifact_ref": str(args.out) if args.out else None,
            "wrapper": {
                "schema": "hipengine.micro.q6_lm_head_rowtile_probe.v1",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
                "commands": commands,
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "method": "pre-recorded Vulkan command buffer via q6_x8_real_slice.py; transfer and pipeline creation excluded",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "all_pass": all(bool(row.get("correctness_pass")) for row in rows_out),
                "rows": len(rows_out),
            },
            "notes": (
                "Vulkan Q6_K X8 q8_1+dp4a full-output lm-head-shaped diagnostic. "
                "This is not bit-identical to the HIP BF16 x Q6_K T16 rowtile path; "
                "it tests whether the existing Vulkan Q6 X8 dot shape is a plausible "
                "backend target for large lm-head-sized output."
            ),
        }
    )


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["in_features"]), int(row["out_features"]), int(row["rows"]))


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    hip_rows = {
        _row_key(row): row
        for row in hip_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    matched = []
    for row in vulkan_result.get("measurements", {}).get("rows", []):
        if not isinstance(row, dict):
            continue
        key = _row_key(row)
        hip = hip_rows.get(key)
        if hip is None:
            continue
        hip_us = float(hip["q6_lm_head_median_us"])
        vk_dot_us = float(row["q6_x8_dot_median_us"])
        vk_combined_us = float(row["q6_x8_quantize_plus_dot_median_us"])
        matched.append(
            {
                "in_features": key[0],
                "out_features": key[1],
                "rows": key[2],
                "vulkan_local_size": row.get("local_size"),
                "hip_variant": hip.get("variant"),
                "vulkan_variant": row.get("variant"),
                "hip_q6_t16_rowtile_median_us": hip_us,
                "vulkan_q6_x8_dot_median_us": vk_dot_us,
                "vulkan_q6_x8_quantize_plus_dot_median_us": vk_combined_us,
                "vulkan_vs_hip_dot_speedup": hip_us / vk_dot_us if vk_dot_us > 0 else None,
                "vulkan_vs_hip_quantize_plus_dot_speedup": hip_us / vk_combined_us
                if vk_combined_us > 0
                else None,
                "hip_correctness_pass": hip.get("correctness_pass"),
                "vulkan_correctness_pass": row.get("correctness_pass"),
            }
        )
    combined_speedups = [float(row["vulkan_vs_hip_quantize_plus_dot_speedup"]) for row in matched]
    dot_speedups = [float(row["vulkan_vs_hip_dot_speedup"]) for row in matched]
    return _json_safe(
        {
            "schema_version": 1,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "real_slice_probe",
            "source": hip_result.get("source", {}),
            "command": command,
            "hardware": {
                "hip": hip_result.get("hardware", {}),
                "vulkan": vulkan_result.get("hardware", {}),
            },
            "inputs": {
                "hip_result": hip_result.get("artifact_ref"),
                "vulkan_result": vulkan_result.get("artifact_ref"),
                "out": out_ref,
            },
            "correctness": {
                "hip": hip_result.get("correctness", {}),
                "vulkan": vulkan_result.get("correctness", {}),
            },
            "matched_rows": matched,
            "summary": {
                "matched_rows": len(matched),
                "combined_speedup_min": min(combined_speedups) if combined_speedups else None,
                "combined_speedup_max": max(combined_speedups) if combined_speedups else None,
                "dot_speedup_min": min(dot_speedups) if dot_speedups else None,
                "dot_speedup_max": max(dot_speedups) if dot_speedups else None,
            },
            "interpretation": (
                "Large Q6_K lm-head-shaped diagnostic. HIP uses production-style BF16 x Q6_K "
                "T16 rowtile chunking; Vulkan uses the existing q8_1+dp4a Q6_K X8 full-output "
                "shader. Correctness is gated within each backend, not cross-backend bit identity."
            ),
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hip", "vulkan"], help="Backend to run")
    parser.add_argument("--compare", nargs=2, metavar=("HIP_RESULT", "VULKAN_RESULT"), type=Path)
    parser.add_argument("--out", type=Path, help="Write normalized/comparison JSON")
    parser.add_argument("--environment-json", type=Path, help="Use an existing environment artifact")
    parser.add_argument("--environment-ref", type=Path, help="Reference this environment path instead of embedding")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gfx-arch", help="Override gfx arch and HIP offload arch")
    parser.add_argument("--hardware-gpu", help="Override GPU name in normalized output")
    parser.add_argument("--shapes", default="2048x32768")
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--local-sizes", default="64,128,256")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=10.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gfx_arch:
        os.environ.setdefault("HIPENGINE_HIP_ARCH", args.gfx_arch)
    if args.compare:
        hip_result = json.loads(args.compare[0].read_text(encoding="utf-8"))
        vulkan_result = json.loads(args.compare[1].read_text(encoding="utf-8"))
        result = build_comparison(
            hip_result,
            vulkan_result,
            command=[Path(sys.executable).name, *sys.argv],
            out_ref=str(args.out) if args.out else None,
        )
    elif args.backend == "hip":
        result = _run_hip(args)
    elif args.backend == "vulkan":
        result = _run_vulkan(args)
    else:
        raise SystemExit("--backend or --compare is required")

    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
