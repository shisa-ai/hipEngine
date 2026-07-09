#!/usr/bin/env python3
"""Run or compare HIP/Vulkan raw Q8_0 dense q8_1+dp4a real-slice probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_q8_0_dense.cpp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
VULKAN_DOT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_0_dense.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
HIP_TIMING = MICRO_ROOT / "hip_timing.py"
HIP_Q8_SOURCE = REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q8_0_dp4a_gemv.hip"
HIP_Q8_QUANTIZE_SOURCE = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q4_k_gemv.hip"
)
HIP_Q8_WRAPPER = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q8_0_dp4a_gemv.py"
)
HIP_Q8_QUANTIZE_WRAPPER = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q4_k_gemv.py"
)
BENCH_NAME = "q8_0_dense_real_slice"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q8-0-dense-real-slice")
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34
Q8_1_BLOCK_BYTES = 36

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing_contract = _load_module(TIMING_CONTRACT, "micro_q8_timing_contract")
hip_timing = _load_module(HIP_TIMING, "micro_q8_hip_timing")


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
        include_privileged=False,
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


def _infer_gpu_name(
    environment: dict[str, Any],
    override: str | None,
    fallback: str | None = None,
) -> str:
    if override:
        return override
    if fallback:
        return fallback
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for key in ("vulkan_summary_lines", "lspci_display_lines", "rocm_smi_lines"):
            lines = devices.get(key)
            if not isinstance(lines, list):
                continue
            for line in lines:
                text = str(line)
                if any(marker in text for marker in ("AMD", "ATI", "Radeon", "Instinct")):
                    return text
    return "unknown"


def _infer_gfx_arch(environment: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    env_arch = os.environ.get("HIPENGINE_HIP_ARCH")
    if env_arch:
        return env_arch
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for value in devices.values():
            text = "\n".join(str(item) for item in value) if isinstance(value, list) else str(value)
            match = re.search(r"\bgfx[0-9a-fA-F]+\b", text)
            if match:
                return match.group(0)
    return "unknown"


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


def _make_x_bf16(
    rows: int,
    in_features: int,
    input_scale: float,
    *,
    repetition: int = 0,
) -> np.ndarray:
    x = np.empty((rows, in_features), dtype=np.float32)
    for row in range(rows):
        for k in range(in_features):
            bits = _hash_u32(
                row * 1315423911
                + k * 2654435761
                + repetition * 2246822519
                + 0x9E3779B9
            )
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


def _aggregate_correctness(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "oracle": "CPU q8_1 quantize plus raw GGUF Q8_0 dense dp4a, bf16 output",
        "outputs_checked": len(items),
        "max_abs": max((float(item["max_abs"]) for item in items), default=0.0),
        "mean_abs": max((float(item["mean_abs"]) for item in items), default=0.0),
        "top1": min((float(item["top1"]) for item in items), default=1.0),
        "pass": bool(items) and all(bool(item["pass"]) for item in items),
    }


def _make_operation_row(
    *,
    backend: str,
    timing_mode: str,
    operation: str,
    repetitions: int,
    dispatches_per_iteration: int,
    stream_count: int,
    single_samples: Any,
    burst_samples: Any,
    single_correctness: dict[str, Any],
    burst_correctness: dict[str, Any],
    barrier_count: int,
    shape_fields: dict[str, Any],
    gpu_timing_supported: bool = True,
) -> dict[str, Any]:
    correctness_pass = bool(single_correctness["pass"] and burst_correctness["pass"])
    correctness = timing_contract.make_correctness(
        status="pass" if correctness_pass else "fail",
        oracle="CPU q8_1 quantize plus raw GGUF Q8_0 dense dp4a, bf16 output",
        logical_iterations=repetitions,
        coverage=(
            "all_dispatches"
            if timing_mode == "independent_throughput"
            else "chained_final_state"
        ),
        synchronization_method=(
            "disjoint_xq_and_output_slices"
            if timing_mode == "independent_throughput"
            else ("hip_stream_order" if backend == "hip" else "vulkan_compute_barrier")
        ),
        barrier_count=barrier_count,
    )
    gpu_clock = "hip_event" if backend == "hip" else "vulkan_timestamp"
    contract = timing_contract.make_timed_row_contract(
        timing_mode=timing_mode,
        backend=backend,
        repetitions=repetitions,
        dispatches_per_iteration=dispatches_per_iteration,
        dependency_validation_status="pass" if correctness_pass else "fail",
        submission=timing_contract.make_submission(
            strategy=(
                "multi_stream"
                if backend == "hip" and timing_mode == "independent_throughput"
                else ("direct" if backend == "hip" else "vulkan_command_buffer")
            ),
            queue_or_stream_count=stream_count if backend == "hip" else 1,
            recording_in_timed_region=False,
        ),
        single_timing=timing_contract.make_timing_control(
            logical_iterations=1,
            dispatches_per_iteration=dispatches_per_iteration,
            gpu_samples_us=single_samples.gpu_sequence_us if gpu_timing_supported else None,
            host_samples_us=single_samples.host_sequence_us,
            gpu_clock=gpu_clock,
            gpu_status="ok" if gpu_timing_supported else "unsupported",
        ),
        burst_timing=timing_contract.make_timing_control(
            logical_iterations=repetitions,
            dispatches_per_iteration=dispatches_per_iteration,
            gpu_samples_us=burst_samples.gpu_sequence_us if gpu_timing_supported else None,
            host_samples_us=burst_samples.host_sequence_us,
            gpu_clock=gpu_clock,
            gpu_status="ok" if gpu_timing_supported else "unsupported",
        ),
        correctness=correctness,
    )
    return {
        "backend": backend,
        "operation": operation,
        **shape_fields,
        **contract,
        "numeric_correctness": {
            "single": single_correctness,
            "burst": burst_correctness,
        },
        "sequence_validation": {
            "input_repetition_pattern": "distinct_deterministic_salt",
            "single_expected_repetitions": single_correctness.get("expected_repetitions", [0]),
            "burst_expected_repetitions": burst_correctness.get("expected_repetitions", []),
            "xq_partitioning": "disjoint" if timing_mode == "independent_throughput" else "shared",
            "output_partitioning": "disjoint" if timing_mode == "independent_throughput" else "shared",
        },
        "correctness_pass": correctness_pass,
        "median_us": (
            contract["timing"]["burst"]["gpu_elapsed"]["per_iteration_us"]["median"]
            if gpu_timing_supported
            else contract["timing"]["burst"]["host_wall"]["per_iteration_us"]["median"]
        ),
    }


def _measure_hip_operation(
    timer: Any,
    launch: Any,
    *,
    repetitions: int,
    warmup: int,
    samples: int,
) -> tuple[Any, Any]:
    if warmup:
        timer.run_and_wait(warmup, launch)
    return (
        timer.measure(1, samples, launch),
        timer.measure(repetitions, samples, launch),
    )


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
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
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            HIP_Q8_SOURCE,
            HIP_Q8_QUANTIZE_SOURCE,
            HIP_Q8_WRAPPER,
            HIP_Q8_QUANTIZE_WRAPPER,
            TIMING_CONTRACT,
            HIP_TIMING,
        ]
    )
    runtime = get_hip_runtime()
    q4_lib = build_gguf_q4_k_gemv(load=True)
    q8_lib = build_gguf_q8_0_dp4a_gemv(load=True)
    rows_out: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    retained_quantize: set[tuple[int, int, int]] = set()

    for in_features, out_features in _parse_shapes(args.shapes):
        for rows in _parse_csv_u32(args.rows_list):
            work_repetitions = max(args.reps, args.warmup, 1)
            x_slices = [
                np.ascontiguousarray(
                    _make_x_bf16(
                        rows,
                        in_features,
                        args.input_scale,
                        repetition=rep,
                    )
                )
                for rep in range(work_repetitions)
            ]
            x_bits = np.ascontiguousarray(np.stack(x_slices))
            weight = np.ascontiguousarray(_make_q8_0_weight(out_features, in_features))
            expected = [_q8_0_q8_1_oracle(x_slice, weight) for x_slice in x_slices]
            blocks = in_features // Q8_0_BLOCK
            x_slice_bytes = rows * in_features * 2
            xq_slice_bytes = rows * blocks * Q8_1_BLOCK_BYTES
            out_slice_bytes = rows * out_features * 2
            work_slices = (
                work_repetitions
                if args.timing_mode == "independent_throughput"
                else 1
            )
            buffers = []

            def _dev(arr: np.ndarray):
                buf = malloc(arr.nbytes, runtime=runtime)
                buffers.append(buf)
                copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes, runtime=runtime)
                return buf

            try:
                x_buf = _dev(x_bits)
                weight_buf = _dev(weight)
                xq_pre_buf = malloc(work_repetitions * xq_slice_bytes, runtime=runtime)
                xq_work_buf = malloc(work_slices * xq_slice_bytes, runtime=runtime)
                out_buf = malloc(work_slices * out_slice_bytes, runtime=runtime)
                buffers.extend([xq_pre_buf, xq_work_buf, out_buf])

                for rep in range(work_repetitions):
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_buf.ptr + rep * x_slice_bytes,
                        xq_pre_buf.ptr + rep * xq_slice_bytes,
                        rows,
                        in_features,
                        library=q4_lib,
                        runtime=runtime,
                    )
                runtime.device_synchronize()

                variants = [
                    ("single", 1, gguf_q8_0_dp4a_gemv_bf16_bf16_out),
                    ("rowtile4", 4, gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out),
                ]
                for variant, row_tile, dot_fn in variants:
                    if row_tile not in _parse_csv_u32(args.row_tiles):
                        continue

                    def quantize(rep: int, stream: int) -> None:
                        output_slice = rep if args.timing_mode == "independent_throughput" else 0
                        gguf_q4_k_quantize_bf16_q8_1(
                            x_buf.ptr + rep * x_slice_bytes,
                            xq_work_buf.ptr + output_slice * xq_slice_bytes,
                            rows,
                            in_features,
                            stream=stream,
                            library=q4_lib,
                            runtime=runtime,
                        )

                    def dot(rep: int, stream: int) -> None:
                        output_slice = rep if args.timing_mode == "independent_throughput" else 0
                        dot_fn(
                            xq_pre_buf.ptr + rep * xq_slice_bytes,
                            weight_buf.ptr,
                            out_buf.ptr + output_slice * out_slice_bytes,
                            rows,
                            in_features,
                            out_features,
                            stream=stream,
                            library=q8_lib,
                            runtime=runtime,
                        )

                    def combined(rep: int, stream: int) -> None:
                        output_slice = rep if args.timing_mode == "independent_throughput" else 0
                        quantize(rep, stream)
                        dot_fn(
                            xq_work_buf.ptr + output_slice * xq_slice_bytes,
                            weight_buf.ptr,
                            out_buf.ptr + output_slice * out_slice_bytes,
                            rows,
                            in_features,
                            out_features,
                            stream=stream,
                            library=q8_lib,
                            runtime=runtime,
                        )

                    def dot_work(rep: int, stream: int) -> None:
                        output_slice = rep if args.timing_mode == "independent_throughput" else 0
                        dot_fn(
                            xq_work_buf.ptr + output_slice * xq_slice_bytes,
                            weight_buf.ptr,
                            out_buf.ptr + output_slice * out_slice_bytes,
                            rows,
                            in_features,
                            out_features,
                            stream=stream,
                            library=q8_lib,
                            runtime=runtime,
                        )

                    def validate_output(expected_indices: list[int]) -> dict[str, Any]:
                        out_bits = np.empty(work_slices * rows * out_features, dtype=np.uint16)
                        copy_device_to_host(
                            host_array_ptr(out_bits),
                            DeviceBuffer(out_buf.ptr, out_bits.nbytes),
                            out_bits.nbytes,
                            runtime=runtime,
                        )
                        shaped = out_bits.reshape(work_slices, rows, out_features)
                        slices = expected_indices if args.timing_mode == "independent_throughput" else [0]
                        return _aggregate_correctness(
                            [
                                _compare_bf16_output(expected[expected_idx], shaped[output_idx])
                                for output_idx, expected_idx in zip(slices, expected_indices, strict=True)
                            ]
                        )

                    shape_fields = {
                        "variant": variant,
                        "row_tile": row_tile,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "local_size": 32,
                        "workgroup_match": "exact_hip_wave32",
                        "q8_blocks_per_row": blocks,
                        "q8_0_weight_bytes": int(weight.nbytes),
                    }
                    with hip_timing.HipSequenceTimer(
                        runtime,
                        args.timing_mode,
                        args.independent_streams,
                    ) as timer:
                        operations = (
                            ("q8_1_quantize", quantize, 1),
                            ("q8_0_dense_dp4a_dot_prequantized", dot, 1),
                            ("q8_0_dense_dp4a_quantize_plus_dot", combined, 2),
                        )
                        for operation, launch, dispatches in operations:
                            operation_shape_fields = dict(shape_fields)
                            if operation == "q8_1_quantize":
                                quantize_key = (in_features, out_features, rows)
                                if quantize_key in retained_quantize:
                                    continue
                                retained_quantize.add(quantize_key)
                                operation_shape_fields.update(
                                    {"variant": "quantize", "row_tile": 0}
                                )
                            single_samples, burst_samples = _measure_hip_operation(
                                timer,
                                launch,
                                repetitions=args.reps,
                                warmup=args.warmup,
                                samples=args.samples,
                            )
                            timer.run_and_wait(1, launch)
                            if operation == "q8_1_quantize":
                                timer.run_and_wait(1, dot_work)
                            single_correctness = validate_output([0])
                            single_correctness["expected_repetitions"] = [0]
                            timer.run_and_wait(args.reps, launch)
                            expected_indices = (
                                list(range(args.reps))
                                if args.timing_mode == "independent_throughput"
                                else [args.reps - 1]
                            )
                            if operation == "q8_1_quantize":
                                timer.run_and_wait(
                                    args.reps if args.timing_mode == "independent_throughput" else 1,
                                    dot_work,
                                )
                            burst_correctness = validate_output(expected_indices)
                            burst_correctness["expected_repetitions"] = expected_indices
                            if not single_correctness["pass"] or not burst_correctness["pass"]:
                                raise RuntimeError(
                                    f"HIP Q8_0 dense {operation} correctness failed for "
                                    f"{in_features}x{out_features} rows={rows} variant={variant}"
                                )
                            rows_out.append(
                                _make_operation_row(
                                    backend="hip",
                                    timing_mode=args.timing_mode,
                                    operation=operation,
                                    repetitions=args.reps,
                                    dispatches_per_iteration=dispatches,
                                    stream_count=timer.stream_count,
                                    single_samples=single_samples,
                                    burst_samples=burst_samples,
                                    single_correctness=single_correctness,
                                    burst_correctness=burst_correctness,
                                    barrier_count=0,
                                    shape_fields=operation_shape_fields,
                                )
                            )
            finally:
                for buffer in reversed(buffers):
                    free(buffer, runtime=runtime)

    correctness_pass = bool(rows_out) and all(
        bool(row.get("correctness_pass")) for row in rows_out
    )
    result = {
            "schema_version": 2,
            "kind": "hipengine_micro_result",
            "bench": BENCH_NAME,
            "backend": "hip",
            "classification": "real_slice_probe",
            "source": _source_record(environment, source_hash),
            "hardware": {
                "gpu_name": _infer_gpu_name(environment, args.hardware_gpu),
                "gfx_arch": _infer_gfx_arch(environment, args.gfx_arch),
            },
            "command": [Path(sys.executable).name, *sys.argv],
            "parameters": {
                "shapes": args.shapes,
                "rows_list": args.rows_list,
                "row_tiles": args.row_tiles,
                "timing_mode": args.timing_mode,
                "input_scale": args.input_scale,
                "repetitions": args.reps,
                "warmup_sequences": args.warmup,
                "samples": args.samples,
                "independent_streams": args.independent_streams,
            },
            "artifact_ref": str(args.out) if args.out else None,
            "wrapper": {
                "schema": "hipengine.micro.q8_0_dense_real_slice_runner.v2",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
                "commands": commands,
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "timing_mode": args.timing_mode,
                "independent_streams": args.independent_streams,
                "method": "HIP events plus host wall for one-dispatch and exact repeated sequences",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "status": "pass" if correctness_pass else "fail",
                "oracle": "CPU q8_1 quantize plus raw GGUF Q8_0 dense dp4a",
                "all_pass": correctness_pass,
                "rows": len(rows_out),
            },
            "notes": (
                "HIP raw GGUF Q8_0 dense q8_1+dp4a production-shaped probe using "
                "single-row and rowtile4 kernels."
            ),
        }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


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
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            VULKAN_HARNESS,
            VULKAN_QUANT_SHADER,
            VULKAN_DOT_SHADER,
            TIMING_CONTRACT,
            MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp",
        ]
    )
    args.build_dir.mkdir(parents=True, exist_ok=True)
    exe = args.build_dir / "vulkan_q8_0_dense"
    quant_spv = args.build_dir / "q8_1_quantize.spv"
    commands: list[dict[str, Any]] = []
    commands.append({"kind": "compile_shader", "command": _compile_shader(VULKAN_QUANT_SHADER, quant_spv, [])})
    commands.append({"kind": "compile_harness", "command": _compile_harness(exe)})
    rows_out: list[dict[str, Any]] = []
    retained_quantize: set[tuple[int, int, int]] = set()

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
                        "--timing-mode",
                        args.timing_mode,
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
                    correctness = raw.get("correctness", {})
                    operations = (
                        ("q8_1_quantize", 1),
                        ("q8_0_dense_dp4a_dot_prequantized", 1),
                        ("q8_0_dense_dp4a_quantize_plus_dot", 2),
                    )
                    for operation, dispatches in operations:
                        quantize_key = (in_features, out_features, rows)
                        if operation == "q8_1_quantize":
                            if quantize_key in retained_quantize:
                                continue
                            retained_quantize.add(quantize_key)
                            operation_variant = "quantize"
                            operation_row_tile = 0
                            operation_local_size = 32
                            workgroup_match = "exact_hip_wave32"
                        else:
                            operation_variant = variant
                            operation_row_tile = row_tile
                            operation_local_size = local_size
                            workgroup_match = (
                                "exact_hip_wave32" if local_size == 32 else "vulkan_diagnostic_unmatched"
                            )
                        operation_timing = timing[operation]
                        operation_correctness = correctness[operation]
                        single_correctness = dict(operation_correctness["single"])
                        burst_correctness = dict(operation_correctness["burst"])
                        single_correctness["expected_repetitions"] = [0]
                        burst_correctness["expected_repetitions"] = (
                            list(range(args.reps))
                            if args.timing_mode == "independent_throughput"
                            else [args.reps - 1]
                        )
                        single = operation_timing["single"]
                        burst = operation_timing["burst"]
                        single_samples = hip_timing.HipTimingSamples(
                            list(single["gpu_samples_us"]),
                            list(single["host_samples_us"]),
                        )
                        burst_samples = hip_timing.HipTimingSamples(
                            list(burst["gpu_samples_us"]),
                            list(burst["host_samples_us"]),
                        )
                        barrier_count = 0
                        if args.timing_mode == "serial_latency":
                            barrier_count = (
                                args.reps - 1
                                if dispatches == 1
                                else 3 * args.reps - 2
                            )
                        elif dispatches == 2:
                            barrier_count = args.reps
                        row = _make_operation_row(
                            backend="vulkan",
                            timing_mode=args.timing_mode,
                            operation=operation,
                            repetitions=args.reps,
                            dispatches_per_iteration=dispatches,
                            stream_count=1,
                            single_samples=single_samples,
                            burst_samples=burst_samples,
                            single_correctness=single_correctness,
                            burst_correctness=burst_correctness,
                            barrier_count=barrier_count,
                            gpu_timing_supported=bool(raw.get("gpu_timestamps_supported")),
                            shape_fields={
                                "variant": operation_variant,
                                "row_tile": operation_row_tile,
                                "rows": rows,
                                "in_features": in_features,
                                "out_features": out_features,
                                "local_size": operation_local_size,
                                "workgroup_match": workgroup_match,
                                "q8_blocks_per_row": in_features // Q8_0_BLOCK,
                                "q8_0_weight_bytes": out_features
                                * (in_features // Q8_0_BLOCK)
                                * Q8_0_BLOCK_BYTES,
                                "hardware": raw.get("hardware", {}),
                            },
                        )
                        rows_out.append(row)

    correctness_pass = bool(rows_out) and all(
        bool(row.get("correctness_pass")) for row in rows_out
    )
    device = rows_out[0].get("hardware", {}) if rows_out else {}
    device_name = device.get("device_name") if isinstance(device, dict) else None
    result = {
            "schema_version": 2,
            "kind": "hipengine_micro_result",
            "bench": BENCH_NAME,
            "backend": "vulkan",
            "classification": "real_slice_probe",
            "source": _source_record(environment, source_hash),
            "hardware": {
                "gpu_name": _infer_gpu_name(environment, args.hardware_gpu, device_name),
                "gfx_arch": _infer_gfx_arch(environment, args.gfx_arch),
                "device": device,
            },
            "command": [Path(sys.executable).name, *sys.argv],
            "parameters": {
                "shapes": args.shapes,
                "rows_list": args.rows_list,
                "local_sizes": args.local_sizes,
                "row_tiles": args.row_tiles,
                "timing_mode": args.timing_mode,
                "input_scale": args.input_scale,
                "repetitions": args.reps,
                "warmup_sequences": args.warmup,
                "samples": args.samples,
            },
            "artifact_ref": str(args.out) if args.out else None,
            "wrapper": {
                "schema": "hipengine.micro.q8_0_dense_real_slice_runner.v2",
                "command": [Path(sys.executable).name, *sys.argv],
                "cwd": str(REPO_ROOT),
                "build_dir": str(args.build_dir),
                "commands": commands,
            },
            "timing_config": {
                "reps": args.reps,
                "warmup": args.warmup,
                "samples": args.samples,
                "timing_mode": args.timing_mode,
                "method": "Vulkan timestamps plus submit/fence host wall for one-dispatch and exact repeated command buffers",
            },
            "measurements": {"rows": rows_out},
            "correctness": {
                "status": "pass" if correctness_pass else "fail",
                "oracle": "CPU q8_1 quantize plus raw GGUF Q8_0 dense dp4a",
                "all_pass": correctness_pass,
                "rows": len(rows_out),
            },
            "notes": (
                "Vulkan raw GGUF Q8_0 dense q8_1+dp4a production-shaped probe with "
                "single-row and rowtile4 shader variants."
            ),
        }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int, int, int]:
    return (
        str(row["operation"]),
        int(row["in_features"]),
        int(row["out_features"]),
        int(row["rows"]),
        int(row["row_tile"]),
        int(row["local_size"]),
    )


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    if hip_result.get("schema_version") != 2 or vulkan_result.get("schema_version") != 2:
        raise ValueError("Q8 comparison requires v2 timing-contract results")
    if hip_result.get("backend") != "hip" or vulkan_result.get("backend") != "vulkan":
        raise ValueError("Q8 comparison inputs must be HIP then Vulkan")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan Q8 gfx architectures do not match")
    hip_parameters = hip_result.get("parameters", {})
    vulkan_parameters = vulkan_result.get("parameters", {})
    for field in (
        "shapes",
        "rows_list",
        "row_tiles",
        "timing_mode",
        "input_scale",
        "repetitions",
        "warmup_sequences",
        "samples",
    ):
        if hip_parameters.get(field) != vulkan_parameters.get(field):
            raise ValueError(f"HIP and Vulkan Q8 {field} values do not match")
    hip_modes = {
        str(row.get("timing_mode"))
        for row in hip_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    vulkan_modes = {
        str(row.get("timing_mode"))
        for row in vulkan_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    if len(hip_modes) != 1 or hip_modes != vulkan_modes:
        raise ValueError("HIP and Vulkan Q8 results must use the same single timing mode")
    hip_comparable = [
        row
        for row in hip_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict) and row.get("workgroup_match") == "exact_hip_wave32"
    ]
    vulkan_comparable = [
        row
        for row in vulkan_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict) and row.get("workgroup_match") == "exact_hip_wave32"
    ]
    hip_rows = {
        _row_key(row): row
        for row in hip_comparable
    }
    vulkan_rows = {_row_key(row): row for row in vulkan_comparable}
    if len(hip_rows) != len(hip_comparable) or len(vulkan_rows) != len(vulkan_comparable):
        raise ValueError("Q8 comparison inputs contain duplicate comparable rows")
    if not hip_rows or set(hip_rows) != set(vulkan_rows):
        raise ValueError("HIP and Vulkan Q8 comparable row sets do not match")
    matched = []
    for key in sorted(hip_rows):
        hip = hip_rows[key]
        row = vulkan_rows[key]
        timing_contract.dependency_signature(hip)
        timing_contract.dependency_signature(row)
        ratios: dict[str, Any] = {}
        for control in timing_contract.TIMING_CONTROLS:
            try:
                gpu_ratio: dict[str, Any] = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip,
                        row,
                        control=control,
                        domain="gpu_elapsed",
                    ),
                }
            except ValueError as exc:
                gpu_ratio = {"status": "not_comparable", "reason": str(exc)}
            try:
                host_ratio: dict[str, Any] = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip,
                        row,
                        control=control,
                        domain="host_wall",
                    ),
                }
            except ValueError as exc:
                host_ratio = {
                    "status": "not_comparable_submission_contract",
                    "reason": str(exc),
                    "hip_submission": hip["submission"]["strategy"],
                    "vulkan_submission": row["submission"]["strategy"],
                }
            ratios[control] = {
                "gpu_elapsed": gpu_ratio,
                "host_wall": host_ratio,
            }
        matched.append(
            {
                "operation": key[0],
                "in_features": key[1],
                "out_features": key[2],
                "rows": key[3],
                "row_tile": key[4],
                "local_size": key[5],
                "variant": hip.get("variant"),
                "workgroup_match": row.get("workgroup_match"),
                "ratios": ratios,
                "hip_correctness_pass": hip.get("correctness_pass"),
                "vulkan_correctness_pass": row.get("correctness_pass"),
            }
        )
    speedups = [
        float(row["ratios"]["burst"]["gpu_elapsed"]["vulkan_vs_hip_speedup"])
        for row in matched
        if row["ratios"]["burst"]["gpu_elapsed"]["status"] == "ok"
    ]
    comparisons = [
        {
            "operation": row["operation"],
            "in_features": row["in_features"],
            "out_features": row["out_features"],
            "rows": row["rows"],
            "row_tile": row["row_tile"],
            "local_size": row["local_size"],
            "timing_mode": next(iter(hip_modes)),
            "control": control,
            "gpu_elapsed": row["ratios"][control]["gpu_elapsed"],
            "host_wall": row["ratios"][control]["host_wall"],
        }
        for row in matched
        for control in timing_contract.TIMING_CONTROLS
    ]
    return _json_safe(
        {
            "schema_version": 2,
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
            "comparisons": comparisons,
            "summary": {
                "matched_rows": len(matched),
                "timing_mode": next(iter(hip_modes)) if hip_modes else None,
                "burst_gpu_speedup_min": min(speedups) if speedups else None,
                "burst_gpu_speedup_max": max(speedups) if speedups else None,
                "host_wall_status": "not_comparable_direct_vs_command_buffer",
            },
            "interpretation": (
                "Matched wave32/rowtile raw Q8_0 dense q8_1+dp4a probe. GPU timestamp "
                "ratios compare equal dependency contracts. HIP direct/multi-stream host wall "
                "is intentionally not compared with Vulkan pre-recorded command-buffer wall."
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
    parser.add_argument("--local-sizes", default="32,64,128,256")
    parser.add_argument("--row-tiles", default="1,4")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--timing-mode",
        choices=timing_contract.TIMING_MODES,
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=10.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.reps <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("--reps and --samples must be positive; --warmup must be non-negative")
    if args.independent_streams <= 0:
        parser.error("--independent-streams must be positive")
    return args


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
