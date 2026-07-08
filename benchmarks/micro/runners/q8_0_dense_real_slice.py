#!/usr/bin/env python3
"""Run or compare HIP/Vulkan raw Q8_0 dense q8_1+dp4a real-slice probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_q8_0_dense.cpp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
VULKAN_DOT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_0_dense.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
HIP_Q8_SOURCE = REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q8_0_dp4a_gemv.hip"
BENCH_NAME = "q8_0_dense_real_slice"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q8-0-dense-real-slice")
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34
Q8_1_BLOCK_BYTES = 36

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


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    echo: bool = True,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=run_env,
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


def _vulkan_cflags_libs() -> list[str]:
    pkg_config = shutil.which("pkg-config")
    if not pkg_config:
        return ["-lvulkan"]
    completed = subprocess.run(
        [pkg_config, "--cflags", "--libs", "vulkan"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return ["-lvulkan"]
    return shlex.split(completed.stdout.strip()) or ["-lvulkan"]


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
        if in_features <= 0 or out_features <= 0 or in_features % Q8_0_BLOCK:
            raise ValueError(f"invalid Q8_0 dense shape: {item}")
        shapes.append((in_features, out_features))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _hash_u32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _make_x_bf16(rows: int, in_features: int, input_scale: float) -> np.ndarray:
    x = np.empty((rows, in_features), dtype=np.float32)
    for row in range(rows):
        for k in range(in_features):
            bits = _hash_u32(row * 1315423911 + k * 2654435761 + 0x9E3779B9)
            centered = (int(bits & 0xFFFF) - 32768) / 32768.0
            x[row, k] = centered * input_scale
    return _f32_to_bf16_bits(x)


def _make_q8_0_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % Q8_0_BLOCK:
        raise ValueError("in_features must be a multiple of 32")
    blocks = in_features // Q8_0_BLOCK
    data = np.empty((out_features, blocks * Q8_0_BLOCK_BYTES), dtype=np.uint8)
    lanes = np.arange(Q8_0_BLOCK, dtype=np.int64)
    for out_idx in range(out_features):
        d = np.asarray([np.float16(0.03125 * (1 + (out_idx % 5)))], dtype=np.float16).view(np.uint8)
        for block_idx in range(blocks):
            start = block_idx * Q8_0_BLOCK_BYTES
            q = (((lanes + out_idx * 7 + block_idx * 3) % 31) - 15).astype(np.int8)
            data[out_idx, start : start + 2] = d
            data[out_idx, start + 2 : start + Q8_0_BLOCK_BYTES] = q.view(np.uint8)
    return data


def _quantize_q8_1_cpu(x_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _bf16_bits_to_f32(x_bf16)
    rows, in_features = x.shape
    blocks = in_features // Q8_0_BLOCK
    q = np.empty((rows, blocks, Q8_0_BLOCK), dtype=np.int8)
    d = np.empty((rows, blocks), dtype=np.float32)
    for row in range(rows):
        for block in range(blocks):
            chunk = x[row, block * Q8_0_BLOCK : (block + 1) * Q8_0_BLOCK].astype(np.float32)
            amax = float(np.max(np.abs(chunk)))
            scale = 0.0 if amax == 0.0 else amax / 127.0
            d[row, block] = np.float16(scale).astype(np.float32)
            q[row, block] = 0 if amax == 0.0 else np.clip(np.round(chunk / scale), -128, 127).astype(np.int8)
    return q, d


def _q8_0_q8_1_oracle(x_bf16: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, in_features = x_bf16.shape
    out_features = weight.shape[0]
    blocks = in_features // Q8_0_BLOCK
    q8, d8 = _quantize_q8_1_cpu(x_bf16)
    out = np.zeros((rows, out_features), dtype=np.float32)
    for col in range(out_features):
        row_bytes = weight[col]
        for block in range(blocks):
            blk = row_bytes[block * Q8_0_BLOCK_BYTES : (block + 1) * Q8_0_BLOCK_BYTES]
            wd = blk[0:2].view(np.float16).astype(np.float32)[0]
            wq = blk[2:34].view(np.int8).astype(np.int32)
            dots = (q8[:, block, :].astype(np.int32) * wq[None, :]).sum(axis=1)
            out[:, col] += d8[:, block] * float(wd) * dots.astype(np.float32)
    return out


def _compare_bf16_output(expected_f32: np.ndarray, actual_bits: np.ndarray) -> dict[str, Any]:
    actual_f32 = _bf16_bits_to_f32(actual_bits.reshape(-1)).reshape(expected_f32.shape)
    diff = np.abs(expected_f32.astype(np.float32) - actual_f32)
    top1 = float(np.mean(np.argmax(expected_f32, axis=-1) == np.argmax(actual_f32, axis=-1)))
    return {
        "oracle": "CPU q8_1 quantize plus raw GGUF Q8_0 dense dp4a, bf16 output",
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "top1": top1,
        "pass": bool(float(np.max(diff)) <= 1.0 and top1 == 1.0),
    }


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
        "median_us": _percentile(samples_us, 0.5),
        "median_ms": _percentile(samples_us, 0.5) / 1000.0,
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
            samples_us.append(runtime.event_elapsed_time_ms(start, stop) * 1000.0 / reps)
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
    return _stats(samples_us)


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
        build_gguf_q8_0_dp4a_gemv,
        gguf_q8_0_dp4a_gemv_bf16_bf16_out,
        gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out,
    )

    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), HIP_Q8_SOURCE])
    runtime = get_hip_runtime()
    q4_lib = build_gguf_q4_k_gemv(load=True)
    q8_lib = build_gguf_q8_0_dp4a_gemv(load=True)
    rows_out: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for in_features, out_features in _parse_shapes(args.shapes):
        for rows in _parse_csv_u32(args.rows_list):
            x_bits = np.ascontiguousarray(_make_x_bf16(rows, in_features, args.input_scale))
            weight = np.ascontiguousarray(_make_q8_0_weight(out_features, in_features))
            expected = _q8_0_q8_1_oracle(x_bits, weight)
            blocks = in_features // Q8_0_BLOCK
            buffers = []

            def _dev(arr: np.ndarray):
                buf = malloc(arr.nbytes, runtime=runtime)
                buffers.append(buf)
                copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes, runtime=runtime)
                return buf

            try:
                x_buf = _dev(x_bits)
                weight_buf = _dev(weight)
                xq_buf = malloc(rows * blocks * Q8_1_BLOCK_BYTES, runtime=runtime)
                out_buf = malloc(rows * out_features * 2, runtime=runtime)
                buffers.extend([xq_buf, out_buf])

                def quantize() -> None:
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_buf.ptr,
                        xq_buf.ptr,
                        rows,
                        in_features,
                        library=q4_lib,
                        runtime=runtime,
                    )

                quant_stats = _time_hip(runtime, quantize, reps=args.reps, warmup=args.warmup, samples=args.samples)
                quantize()
                runtime.device_synchronize()

                variants = [
                    ("single", 1, gguf_q8_0_dp4a_gemv_bf16_bf16_out),
                    ("rowtile4", 4, gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out),
                ]
                for variant, row_tile, dot_fn in variants:
                    if row_tile not in _parse_csv_u32(args.row_tiles):
                        continue

                    def dot() -> None:
                        dot_fn(
                            xq_buf.ptr,
                            weight_buf.ptr,
                            out_buf.ptr,
                            rows,
                            in_features,
                            out_features,
                            library=q8_lib,
                            runtime=runtime,
                        )

                    def combined() -> None:
                        quantize()
                        dot()

                    dot()
                    runtime.device_synchronize()
                    out_bits = np.empty(rows * out_features, dtype=np.uint16)
                    copy_device_to_host(host_array_ptr(out_bits), out_buf, out_bits.nbytes, runtime=runtime)
                    correctness = _compare_bf16_output(expected, out_bits.reshape(rows, out_features))
                    if not correctness["pass"]:
                        raise RuntimeError(
                            f"HIP Q8_0 dense correctness failed for {in_features}x{out_features} "
                            f"rows={rows} variant={variant}: {correctness}"
                        )
                    dot_stats = _time_hip(runtime, dot, reps=args.reps, warmup=args.warmup, samples=args.samples)
                    combined_stats = _time_hip(runtime, combined, reps=args.reps, warmup=args.warmup, samples=args.samples)
                    rows_out.append(
                        {
                            "backend": "hip",
                            "variant": variant,
                            "row_tile": row_tile,
                            "rows": rows,
                            "in_features": in_features,
                            "out_features": out_features,
                            "local_size": 32,
                            "q8_blocks_per_row": blocks,
                            "q8_0_weight_bytes": int(weight.nbytes),
                            "q8_1_quantize_median_us": quant_stats["median_us"],
                            "q8_0_dense_dp4a_dot_median_us": dot_stats["median_us"],
                            "q8_0_dense_dp4a_quantize_plus_dot_median_us": combined_stats["median_us"],
                            "timing": {
                                "q8_1_quantize": quant_stats,
                                "q8_0_dense_dp4a_dot_prequantized": dot_stats,
                                "q8_0_dense_dp4a_quantize_plus_dot": combined_stats,
                            },
                            "correctness": correctness,
                            "correctness_pass": correctness["pass"],
                        }
                    )
            finally:
                for buffer in reversed(buffers):
                    free(buffer, runtime=runtime)

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
                "schema": "hipengine.micro.q8_0_dense_real_slice_runner.v1",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
                "commands": commands,
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "method": "HIP events around repeated launches; transfer and build excluded",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "all_pass": all(bool(row.get("correctness_pass")) for row in rows_out),
                "rows": len(rows_out),
            },
            "notes": (
                "HIP raw GGUF Q8_0 dense q8_1+dp4a production-shaped probe using "
                "single-row and rowtile4 kernels."
            ),
        }
    )


def _compile_shader(shader: Path, spirv: Path, defines: list[str]) -> list[str]:
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    if glslc:
        command = [glslc, "--target-env=vulkan1.1", "-O", *defines, str(shader), "-o", str(spirv)]
    elif glslang:
        command = [glslang, "-V", "--target-env", "vulkan1.1", *defines, str(shader), "-o", str(spirv)]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"shader build failed: {shader}")
    return command


def _compile_harness(exe: Path) -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(VULKAN_HARNESS),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan Q8_0 dense harness build failed")
    return command


def _run_vulkan(args: argparse.Namespace) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), VULKAN_HARNESS, VULKAN_QUANT_SHADER, VULKAN_DOT_SHADER])
    args.build_dir.mkdir(parents=True, exist_ok=True)
    exe = args.build_dir / "vulkan_q8_0_dense"
    quant_spv = args.build_dir / "q8_1_quantize.spv"
    commands: list[dict[str, Any]] = []
    commands.append({"kind": "compile_shader", "command": _compile_shader(VULKAN_QUANT_SHADER, quant_spv, [])})
    commands.append({"kind": "compile_harness", "command": _compile_harness(exe)})
    rows_out: list[dict[str, Any]] = []

    for local_size in _parse_csv_u32(args.local_sizes):
        for row_tile in _parse_csv_u32(args.row_tiles):
            variant = "single" if row_tile == 1 else f"rowtile{row_tile}"
            variant_dir = args.build_dir / "vulkan" / f"ls{local_size}_rt{row_tile}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            dot_spv = variant_dir / "q8_0_dense.spv"
            shader_command = _compile_shader(
                VULKAN_DOT_SHADER,
                dot_spv,
                [f"-DHIPENGINE_LOCAL_SIZE_X={local_size}", f"-DHIPENGINE_ROW_TILE={row_tile}"],
            )
            commands.append(
                {
                    "kind": "compile_shader",
                    "variant": variant,
                    "local_size": local_size,
                    "row_tile": row_tile,
                    "command": shader_command,
                }
            )
            for in_features, out_features in _parse_shapes(args.shapes):
                for rows in _parse_csv_u32(args.rows_list):
                    raw_out = variant_dir / f"raw_{in_features}x{out_features}_rows{rows}.json"
                    run_command = [
                        str(exe),
                        "--quantize-spirv",
                        str(quant_spv),
                        "--dot-spirv",
                        str(dot_spv),
                        "--json",
                        str(raw_out),
                        "--rows",
                        str(rows),
                        "--in-features",
                        str(in_features),
                        "--out-features",
                        str(out_features),
                        "--input-scale",
                        str(args.input_scale),
                        "--local-size",
                        str(local_size),
                        "--row-tile",
                        str(row_tile),
                        "--reps",
                        str(args.reps),
                        "--warmup",
                        str(args.warmup),
                        "--samples",
                        str(args.samples),
                        "--device-index",
                        str(args.device_index),
                    ]
                    completed = _run_command(run_command, cwd=REPO_ROOT)
                    commands.append(
                        {
                            "kind": "run_harness",
                            "variant": variant,
                            "local_size": local_size,
                            "row_tile": row_tile,
                            "shape": f"{in_features}x{out_features}",
                            "rows": rows,
                            "command": run_command,
                            "returncode": completed.returncode,
                            "raw_json_retained": False,
                        }
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(f"Vulkan Q8_0 dense run failed: {' '.join(run_command)}")
                    raw = json.loads(raw_out.read_text(encoding="utf-8"))
                    timing = raw.get("timing", {})
                    correctness = raw.get("correctness_vs_cpu", {})
                    row = {
                        "backend": "vulkan",
                        "variant": variant,
                        "row_tile": row_tile,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "local_size": local_size,
                        "q8_blocks_per_row": in_features // Q8_0_BLOCK,
                        "q8_0_weight_bytes": out_features * (in_features // Q8_0_BLOCK) * Q8_0_BLOCK_BYTES,
                        "q8_1_quantize_median_us": timing["q8_1_quantize"]["median_us"],
                        "q8_0_dense_dp4a_dot_median_us": timing["q8_0_dense_dp4a_dot_prequantized"]["median_us"],
                        "q8_0_dense_dp4a_quantize_plus_dot_median_us": timing[
                            "q8_0_dense_dp4a_quantize_plus_dot"
                        ]["median_us"],
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
                "schema": "hipengine.micro.q8_0_dense_real_slice_runner.v1",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
                "commands": commands,
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "method": "pre-recorded Vulkan command buffer, host wall divided by reps; transfer and pipeline creation excluded",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "all_pass": all(bool(row.get("correctness_pass")) for row in rows_out),
                "rows": len(rows_out),
            },
            "notes": (
                "Vulkan raw GGUF Q8_0 dense q8_1+dp4a production-shaped probe with "
                "single-row and rowtile4 shader variants."
            ),
        }
    )


def _row_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["in_features"]),
        int(row["out_features"]),
        int(row["rows"]),
        int(row["row_tile"]),
    )


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
        hip_dot = float(hip["q8_0_dense_dp4a_dot_median_us"])
        vk_dot = float(row["q8_0_dense_dp4a_dot_median_us"])
        hip_combined = float(hip["q8_0_dense_dp4a_quantize_plus_dot_median_us"])
        vk_combined = float(row["q8_0_dense_dp4a_quantize_plus_dot_median_us"])
        matched.append(
            {
                "in_features": key[0],
                "out_features": key[1],
                "rows": key[2],
                "row_tile": key[3],
                "variant": hip.get("variant"),
                "vulkan_local_size": row.get("local_size"),
                "hip_dot_median_us": hip_dot,
                "vulkan_dot_median_us": vk_dot,
                "vulkan_vs_hip_dot_speedup": hip_dot / vk_dot if vk_dot > 0 else None,
                "hip_quantize_plus_dot_median_us": hip_combined,
                "vulkan_quantize_plus_dot_median_us": vk_combined,
                "vulkan_vs_hip_quantize_plus_dot_speedup": hip_combined / vk_combined if vk_combined > 0 else None,
                "hip_q8_1_quantize_median_us": hip.get("q8_1_quantize_median_us"),
                "vulkan_q8_1_quantize_median_us": row.get("q8_1_quantize_median_us"),
                "hip_correctness_pass": hip.get("correctness_pass"),
                "vulkan_correctness_pass": row.get("correctness_pass"),
            }
        )
    speedups = [float(row["vulkan_vs_hip_quantize_plus_dot_speedup"]) for row in matched]
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
                "combined_speedup_min": min(speedups) if speedups else None,
                "combined_speedup_max": max(speedups) if speedups else None,
                "dot_speedup_min": min(dot_speedups) if dot_speedups else None,
                "dot_speedup_max": max(dot_speedups) if dot_speedups else None,
            },
            "interpretation": (
                "Matched raw Q8_0 dense q8_1+dp4a production-shaped probe. "
                "This row covers the prior q8_0 dense GEMV / dense attention projection "
                "matrix gap for the tested synthetic shapes; ISA extraction is a separate follow-up."
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
    parser.add_argument("--shapes", default="2048x2048,2048x6144,768x2048")
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--local-sizes", default="64,128,256")
    parser.add_argument("--row-tiles", default="1,4")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=10.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare and args.backend:
        raise SystemExit("--compare and --backend are mutually exclusive")
    if not args.compare and not args.backend:
        raise SystemExit("one of --backend or --compare is required")
    if args.compare:
        hip_path, vulkan_path = args.compare
        result = build_comparison(
            json.loads(hip_path.read_text(encoding="utf-8")),
            json.loads(vulkan_path.read_text(encoding="utf-8")),
            command=[Path(sys.executable).name, *sys.argv],
            out_ref=str(args.out) if args.out else None,
        )
    elif args.backend == "hip":
        result = _run_hip(args)
    else:
        result = _run_vulkan(args)

    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
