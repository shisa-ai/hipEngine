#!/usr/bin/env python3
"""Run or compare paired Q6_K X8 selected-down q8_1+dp4a real-slice probes."""

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
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_q6_x8_selected_down.cpp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
VULKAN_DOT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q6_x8_selected_down.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
HIP_TIMING = MICRO_ROOT / "hip_timing.py"
HIP_Q8_QUANTIZE_SOURCE = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q4_k_gemv.hip"
)
HIP_Q6_RAW_SOURCE = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_k_gemv.hip"
)
HIP_Q6_X8_SOURCE = (
    REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_x8_selected_gemv.hip"
)
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q6-x8-real-slice-build")
BENCH_NAME = "q6_x8_selected_down_real_slice"
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


timing_contract = _load_module(TIMING_CONTRACT, "micro_q6_x8_timing_contract")
hip_timing = _load_module(HIP_TIMING, "micro_q6_x8_hip_timing")


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_module(COLLECT_ENV, "micro_q6_x8_collect_env")
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
        raise RuntimeError("Vulkan Q6 X8 real-slice harness build failed")
    return command


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
    repetition: int,
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


def _compare_bf16_output(expected_bits: np.ndarray, actual_bits: np.ndarray) -> dict[str, Any]:
    expected = _bf16_bits_to_f32(expected_bits.reshape(-1)).reshape(expected_bits.shape)
    actual = _bf16_bits_to_f32(actual_bits.reshape(-1)).reshape(actual_bits.shape)
    diff = np.abs(expected - actual)
    top1 = float(np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1)))
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    expected_shift = expected64 - np.max(expected64, axis=-1, keepdims=True)
    actual_shift = actual64 - np.max(actual64, axis=-1, keepdims=True)
    expected_log_z = np.log(np.sum(np.exp(expected_shift), axis=-1, keepdims=True))
    actual_log_z = np.log(np.sum(np.exp(actual_shift), axis=-1, keepdims=True))
    expected_log_p = expected_shift - expected_log_z
    actual_log_p = actual_shift - actual_log_z
    expected_p = np.exp(expected_log_p)
    kl_divergence = float(
        np.max(np.sum(expected_p * (expected_log_p - actual_log_p), axis=-1))
    )
    return {
        "oracle": "raw Q6_K selected q8_1+dp4a BF16 logits with KL/top-1 gate",
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "kl_divergence": kl_divergence,
        "top1": top1,
        "exact_bf16_mismatches": int(np.count_nonzero(expected_bits != actual_bits)),
        "pass": bool(kl_divergence <= 0.05 and top1 >= 0.90),
    }


def _aggregate_correctness(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "oracle": "raw Q6_K selected q8_1+dp4a BF16 logits with KL/top-1 gate",
        "outputs_checked": len(items),
        "max_abs": max((float(item["max_abs"]) for item in items), default=0.0),
        "mean_abs": max((float(item["mean_abs"]) for item in items), default=0.0),
        "kl_divergence": max(
            (float(item["kl_divergence"]) for item in items), default=0.0
        ),
        "top1": min((float(item["top1"]) for item in items), default=1.0),
        "exact_bf16_mismatches": sum(
            (int(item.get("exact_bf16_mismatches", 0)) for item in items), 0
        ),
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
    calibrated_gpu_timing: bool = False,
) -> dict[str, Any]:
    correctness_pass = bool(single_correctness["pass"] and burst_correctness["pass"])
    correctness = timing_contract.make_correctness(
        status="pass" if correctness_pass else "fail",
        oracle="raw Q6_K selected q8_1+dp4a BF16 logits with KL <= 0.05 and top-1 >= 90%",
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
    gpu_clock = (
        "hip_event"
        if backend == "hip"
        else (
            "vulkan_calibrated_cross_queue_timestamp"
            if calibrated_gpu_timing
            else "vulkan_timestamp"
        )
    )
    if backend == "hip":
        submission_strategy = (
            "multi_stream"
            if timing_mode == "independent_throughput" and stream_count > 1
            else "direct"
        )
    else:
        submission_strategy = (
            "vulkan_multi_queue"
            if timing_mode == "independent_throughput"
            and operation == "x8_selected_dp4a_quantize_plus_dot"
            and stream_count > 1
            else "vulkan_command_buffer"
        )
    contract = timing_contract.make_timed_row_contract(
        timing_mode=timing_mode,
        backend=backend,
        repetitions=repetitions,
        dispatches_per_iteration=dispatches_per_iteration,
        dependency_validation_status="pass" if correctness_pass else "fail",
        submission=timing_contract.make_submission(
            strategy=submission_strategy,
            queue_or_stream_count=stream_count,
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


def _effective_independent_lanes(args: argparse.Namespace) -> int:
    if args.timing_mode != "independent_throughput":
        return 1
    return min(args.independent_streams, args.reps)


def _common_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quant": "q6_k",
        "activation_quant": "q8_1",
        "weight_layout": "q6_k_x8",
        "selected_dtype": "int64",
        "output_dtype": "bf16",
        "kv_type": "not_applicable",
        "model_fingerprint": "synthetic_q6_x8_selected_down_v1",
        "rows": args.rows,
        "experts": args.experts,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "input_scale": args.input_scale,
        "local_size": args.local_size,
        "row_tile": 1,
        "timing_mode": args.timing_mode,
        "independent_lanes": _effective_independent_lanes(args),
        "repetitions": args.reps,
        "warmup_iterations": args.warmup,
        "samples": args.samples,
    }


def _shape_fields(args: argparse.Namespace, *, operation: str) -> dict[str, Any]:
    quantize = operation == "q8_1_quantize"
    return {
        "variant": "quantize" if quantize else "q6_x8_selected_down",
        "quant": "q6_k",
        "activation_quant": "q8_1",
        "weight_layout": "q6_k_x8",
        "selected_dtype": "int64",
        "output_dtype": "bf16",
        "kv_type": "not_applicable",
        "model_fingerprint": "synthetic_q6_x8_selected_down_v1",
        "row_tile": 0 if quantize else 1,
        "rows": args.rows,
        "experts": args.experts,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "local_size": 32 if quantize else args.local_size,
        "workgroup_match": "exact",
        "q8_blocks_per_row": args.in_features // 32,
        "blocks_per_row": args.in_features // 256,
        "out_packed": args.out_features // 8,
    }


def _legacy_timing_aliases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for row in rows:
        metric = row["timing"]["burst"]["gpu_elapsed"]
        if metric["status"] != "ok":
            metric = row["timing"]["burst"]["host_wall"]
        stats = metric["per_iteration_us"]
        aliases[str(row["operation"])] = {
            "median_us": stats["median"],
            "median_ms": stats["median"] / 1000.0,
            "p05_us": stats["p05"],
            "p95_us": stats["p95"],
            "min_us": stats["min"],
            "max_us": stats["max"],
        }
    return aliases


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        build_gguf_k_gemv,
        gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
        build_gguf_x8_selected_gemv,
        gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_x8 import repack_gguf_q6_k_x8
    from tests._gguf_synthetic_weights import make_q6_k_weight

    if args.compiler_version_file:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            HIP_Q8_QUANTIZE_SOURCE,
            HIP_Q6_RAW_SOURCE,
            HIP_Q6_X8_SOURCE,
            REPO_ROOT / "hipengine" / "quant" / "gguf_x8.py",
            REPO_ROOT / "tests" / "_gguf_synthetic_weights.py",
            TIMING_CONTRACT,
            HIP_TIMING,
        ]
    )
    runtime = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(
        load=True, require_cached=args.require_cached_build
    )
    raw_library = build_gguf_k_gemv(
        load=True, require_cached=args.require_cached_build
    )
    x8_library = build_gguf_x8_selected_gemv(
        load=True, require_cached=args.require_cached_build
    )
    work_repetitions = max(args.reps, args.warmup, 1)
    x_slices = [
        np.ascontiguousarray(
            _make_x_bf16(args.rows, args.in_features, args.input_scale, rep)
        )
        for rep in range(work_repetitions)
    ]
    x = np.ascontiguousarray(np.stack(x_slices))
    selected = np.ascontiguousarray(
        (np.arange(args.rows, dtype=np.int64) % args.experts).astype(np.int64)
    )
    base = make_q6_k_weight(args.out_features, args.in_features)
    qweight = np.ascontiguousarray(
        np.stack(
            [
                np.roll(base, shift=expert + 53, axis=0)
                for expert in range(args.experts)
            ],
            axis=0,
        )
    )
    x8_tiles = np.ascontiguousarray(repack_gguf_q6_k_x8(qweight).tiles)
    x_slice_bytes = args.rows * args.in_features * 2
    xq_slice_bytes = args.rows * (args.in_features // 32) * Q8_1_BLOCK_BYTES
    out_slice_bytes = args.rows * args.out_features * 2
    work_slices = work_repetitions if args.timing_mode == "independent_throughput" else 1
    buffers = []

    def dev(arr: np.ndarray):
        contiguous = np.ascontiguousarray(arr)
        buf = malloc(contiguous.nbytes, runtime=runtime)
        copy_host_to_device(buf, host_array_ptr(contiguous), contiguous.nbytes, runtime=runtime)
        buffers.append(buf)
        return buf

    rows_out: list[dict[str, Any]] = []
    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        qweight_buf = dev(qweight)
        x8_buf = dev(x8_tiles)
        xq_pre_buf = malloc(work_repetitions * xq_slice_bytes, runtime=runtime)
        xq_work_buf = malloc(work_slices * xq_slice_bytes, runtime=runtime)
        ref_out_buf = malloc(work_repetitions * out_slice_bytes, runtime=runtime)
        out_buf = malloc(work_slices * out_slice_bytes, runtime=runtime)
        buffers.extend((xq_pre_buf, xq_work_buf, ref_out_buf, out_buf))

        for rep in range(work_repetitions):
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr + rep * x_slice_bytes,
                xq_pre_buf.ptr + rep * xq_slice_bytes,
                args.rows,
                args.in_features,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_pre_buf.ptr + rep * xq_slice_bytes,
                selected_buf.ptr,
                qweight_buf.ptr,
                ref_out_buf.ptr + rep * out_slice_bytes,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                threads=args.local_size,
                library=raw_library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        reference_bits = np.empty(
            (work_repetitions, args.rows, args.out_features), dtype=np.uint16
        )
        copy_device_to_host(
            host_array_ptr(reference_bits),
            DeviceBuffer(ref_out_buf.ptr, reference_bits.nbytes),
            reference_bits.nbytes,
            runtime=runtime,
        )

        def quantize(rep: int, stream: int) -> None:
            output_slice = rep if args.timing_mode == "independent_throughput" else 0
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr + rep * x_slice_bytes,
                xq_work_buf.ptr + output_slice * xq_slice_bytes,
                args.rows,
                args.in_features,
                stream=stream,
                library=q4_library,
                runtime=runtime,
            )

        def launch_x8(xq_ptr: int, out_ptr: int, stream: int) -> None:
            gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
                xq_ptr,
                selected_buf.ptr,
                x8_buf.ptr,
                out_ptr,
                args.rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                threads=args.local_size,
                stream=stream,
                library=x8_library,
                runtime=runtime,
            )

        def dot(rep: int, stream: int) -> None:
            output_slice = rep if args.timing_mode == "independent_throughput" else 0
            launch_x8(
                xq_pre_buf.ptr + rep * xq_slice_bytes,
                out_buf.ptr + output_slice * out_slice_bytes,
                stream,
            )

        def combined(rep: int, stream: int) -> None:
            output_slice = rep if args.timing_mode == "independent_throughput" else 0
            quantize(rep, stream)
            launch_x8(
                xq_work_buf.ptr + output_slice * xq_slice_bytes,
                out_buf.ptr + output_slice * out_slice_bytes,
                stream,
            )

        def dot_work(rep: int, stream: int) -> None:
            output_slice = rep if args.timing_mode == "independent_throughput" else 0
            launch_x8(
                xq_work_buf.ptr + output_slice * xq_slice_bytes,
                out_buf.ptr + output_slice * out_slice_bytes,
                stream,
            )

        def validate_output(expected_indices: list[int]) -> dict[str, Any]:
            actual_bits = np.empty(
                (work_slices, args.rows, args.out_features), dtype=np.uint16
            )
            copy_device_to_host(
                host_array_ptr(actual_bits),
                DeviceBuffer(out_buf.ptr, actual_bits.nbytes),
                actual_bits.nbytes,
                runtime=runtime,
            )
            output_indices = (
                expected_indices if args.timing_mode == "independent_throughput" else [0]
            )
            return _aggregate_correctness(
                [
                    _compare_bf16_output(
                        reference_bits[expected_idx], actual_bits[output_idx]
                    )
                    for output_idx, expected_idx in zip(
                        output_indices, expected_indices, strict=True
                    )
                ]
            )

        operations = (
            ("q8_1_quantize", quantize, 1),
            ("x8_selected_dp4a_dot_prequantized", dot, 1),
            ("x8_selected_dp4a_quantize_plus_dot", combined, 2),
        )
        with hip_timing.HipSequenceTimer(
            runtime, args.timing_mode, _effective_independent_lanes(args)
        ) as timer:
            for operation, launch, dispatches in operations:
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
                        f"HIP Q6 X8 {operation} correctness failed: "
                        f"single={single_correctness} burst={burst_correctness}"
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
                        shape_fields=_shape_fields(args, operation=operation),
                    )
                )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    correctness_pass = bool(rows_out) and all(
        bool(row.get("correctness_pass")) for row in rows_out
    )
    result: dict[str, Any] = {
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
        "parameters": _common_parameters(args),
        "artifact_ref": str(args.out),
        "measurements": {"rows": rows_out},
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "raw Q6_K selected q8_1+dp4a BF16 output",
            "all_pass": correctness_pass,
            "rows": len(rows_out),
        },
        "timing": _legacy_timing_aliases(rows_out),
        "correctness_vs_cpu": {
            "pass": correctness_pass,
            "oracle": "raw Q6_K selected q8_1+dp4a GPU reference",
        },
        "notes": "Production HIP Q6_K X8 selected-down q8_1+dp4a slice under timing contract v2.",
    }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


def _run_vulkan(args: argparse.Namespace) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            VULKAN_HARNESS,
            VULKAN_QUANT_SHADER,
            VULKAN_DOT_SHADER,
            MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp",
            TIMING_CONTRACT,
        ]
    )
    args.build_dir.mkdir(parents=True, exist_ok=True)
    quant_spv = args.build_dir / "q8_1_quantize.spv"
    dot_spv = args.build_dir / "q6_x8_selected_down.spv"
    exe = args.build_dir / "vulkan_q6_x8_selected_down"
    commands: list[dict[str, Any]] = []
    commands.append(
        {
            "kind": "compile_shader",
            "command": _compile_shader(VULKAN_QUANT_SHADER, quant_spv, []),
        }
    )
    commands.append(
        {
            "kind": "compile_shader",
            "command": _compile_shader(
                VULKAN_DOT_SHADER,
                dot_spv,
                [f"-DHIPENGINE_LOCAL_SIZE_X={args.local_size}"],
            ),
        }
    )
    commands.append({"kind": "compile_harness", "command": _compile_harness(exe)})
    raw_out = args.build_dir / "vulkan-q6-x8-selected-down-raw.json"
    run_command = [
        str(exe),
        "--quantize-spirv",
        str(quant_spv),
        "--dot-spirv",
        str(dot_spv),
        "--json",
        str(raw_out),
        "--rows",
        str(args.rows),
        "--experts",
        str(args.experts),
        "--in-features",
        str(args.in_features),
        "--out-features",
        str(args.out_features),
        "--input-scale",
        str(args.input_scale),
        "--local-size",
        str(args.local_size),
        "--reps",
        str(args.reps),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--timing-mode",
        args.timing_mode,
        "--independent-lanes",
        str(_effective_independent_lanes(args)),
        "--device-index",
        str(args.device_index),
    ]
    completed = _run_command(run_command, cwd=REPO_ROOT)
    commands.append(
        {"kind": "run_harness", "command": run_command, "returncode": completed.returncode}
    )
    if completed.returncode != 0:
        raise RuntimeError("Vulkan Q6 X8 real-slice run failed")
    raw = json.loads(raw_out.read_text(encoding="utf-8"))
    raw_timing = raw["timing"]
    raw_correctness = raw["correctness"]
    raw_hardware = raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {}
    active_queue_count = int(raw_hardware.get("active_queue_count", 1))
    expected_queue_count = _effective_independent_lanes(args)
    if active_queue_count != expected_queue_count:
        raise RuntimeError(
            "Vulkan Q6 X8 active queue count does not match the requested "
            f"independent lane contract: {active_queue_count} != {expected_queue_count}"
        )
    calibrated_extension = raw_hardware.get("calibrated_timestamps_extension")
    if expected_queue_count > 1 and (
        not calibrated_extension
        or not bool(raw_hardware.get("cross_queue_gpu_timing_calibrated"))
    ):
        raise RuntimeError(
            "Vulkan Q6 X8 multi-queue result lacks calibrated timestamp metadata"
        )
    rows_out: list[dict[str, Any]] = []
    operations = (
        ("q8_1_quantize", 1),
        ("x8_selected_dp4a_dot_prequantized", 1),
        ("x8_selected_dp4a_quantize_plus_dot", 2),
    )
    for operation, dispatches in operations:
        operation_timing = raw_timing[operation]
        single = operation_timing["single"]
        burst = operation_timing["burst"]
        single_samples = hip_timing.HipTimingSamples(
            list(single["gpu_samples_us"]), list(single["host_samples_us"])
        )
        burst_samples = hip_timing.HipTimingSamples(
            list(burst["gpu_samples_us"]), list(burst["host_samples_us"])
        )
        single_correctness = dict(raw_correctness[operation]["single"])
        burst_correctness = dict(raw_correctness[operation]["burst"])
        single_correctness["expected_repetitions"] = [0]
        burst_correctness["expected_repetitions"] = (
            list(range(args.reps))
            if args.timing_mode == "independent_throughput"
            else [args.reps - 1]
        )
        barrier_count = 0
        if args.timing_mode == "serial_latency":
            barrier_count = args.reps - 1 if dispatches == 1 else 3 * args.reps - 2
        elif dispatches == 2:
            barrier_count = args.reps
        calibrated_gpu_timing = bool(
            args.timing_mode == "independent_throughput"
            and dispatches == 2
            and active_queue_count > 1
        )
        if calibrated_gpu_timing and (
            not bool(single.get("calibrated_timestamp_domain"))
            or not bool(burst.get("calibrated_timestamp_domain"))
        ):
            raise RuntimeError(
                "Vulkan Q6 X8 combined independent timing is not in a calibrated domain"
            )
        row = _make_operation_row(
            backend="vulkan",
            timing_mode=args.timing_mode,
            operation=operation,
            repetitions=args.reps,
            dispatches_per_iteration=dispatches,
            stream_count=active_queue_count if calibrated_gpu_timing else 1,
            single_samples=single_samples,
            burst_samples=burst_samples,
            single_correctness=single_correctness,
            burst_correctness=burst_correctness,
            barrier_count=barrier_count,
            shape_fields=_shape_fields(args, operation=operation),
            gpu_timing_supported=bool(raw.get("gpu_timestamps_supported")),
            calibrated_gpu_timing=calibrated_gpu_timing,
        )
        if calibrated_gpu_timing:
            row["vulkan_multi_queue_timing"] = {
                "active_queue_count": active_queue_count,
                "calibrated_timestamps_extension": calibrated_extension,
                "single_lane_gpu_samples_us": single.get("lane_gpu_samples_us", []),
                "burst_lane_gpu_samples_us": burst.get("lane_gpu_samples_us", []),
            }
        rows_out.append(row)
    correctness_pass = bool(rows_out) and all(
        bool(row.get("correctness_pass")) for row in rows_out
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": "vulkan",
        "classification": "real_slice_probe",
        "source": _source_record(environment, source_hash),
        "hardware": {
            "gpu_name": _infer_gpu_name(
                environment, args.hardware_gpu, raw_hardware.get("device_name")
            ),
            "gfx_arch": _infer_gfx_arch(environment, args.gfx_arch),
            "device": raw_hardware,
        },
        "command": [Path(sys.executable).name, *sys.argv],
        "parameters": _common_parameters(args),
        "artifact_ref": str(args.out),
        "wrapper": {
            "schema": "hipengine.micro.q6_x8_real_slice_runner.v2",
            "cwd": str(REPO_ROOT),
            "build_dir": str(args.build_dir),
            "commands": _json_safe(commands),
        },
        "measurements": {"rows": rows_out},
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "CPU Q6_K X8 selected-down q8_1+dp4a BF16 output",
            "all_pass": correctness_pass,
            "rows": len(rows_out),
            "q8_dot_isolation": raw.get("isolation", {}),
        },
        "timing": _legacy_timing_aliases(rows_out),
        "correctness_vs_cpu": {
            "pass": correctness_pass,
            "oracle": "CPU Q6_K X8 selected-down q8_1+dp4a",
        },
        "notes": "Vulkan Q6_K X8 selected-down q8_1+dp4a slice under timing contract v2.",
    }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["operation"]),
        str(row["quant"]),
        str(row["activation_quant"]),
        str(row["weight_layout"]),
        str(row["selected_dtype"]),
        str(row["output_dtype"]),
        str(row["kv_type"]),
        str(row["model_fingerprint"]),
        int(row["rows"]),
        int(row["experts"]),
        int(row["in_features"]),
        int(row["out_features"]),
        int(row["row_tile"]),
        int(row["local_size"]),
        str(row["timing_mode"]),
    )


def _device_fingerprint(name: Any) -> str:
    text = str(name).lower()
    match = re.search(r"radeon\s+(\d+s)", text)
    if match:
        return f"radeon_{match.group(1)}"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _expected_row_keys(parameters: dict[str, Any]) -> set[tuple[Any, ...]]:
    common = (
        str(parameters["quant"]),
        str(parameters["activation_quant"]),
        str(parameters["weight_layout"]),
        str(parameters["selected_dtype"]),
        str(parameters["output_dtype"]),
        str(parameters["kv_type"]),
        str(parameters["model_fingerprint"]),
        int(parameters["rows"]),
        int(parameters["experts"]),
        int(parameters["in_features"]),
        int(parameters["out_features"]),
    )
    timing_mode = str(parameters["timing_mode"])
    return {
        ("q8_1_quantize", *common, 0, 32, timing_mode),
        (
            "x8_selected_dp4a_dot_prequantized",
            *common,
            int(parameters["row_tile"]),
            int(parameters["local_size"]),
            timing_mode,
        ),
        (
            "x8_selected_dp4a_quantize_plus_dot",
            *common,
            int(parameters["row_tile"]),
            int(parameters["local_size"]),
            timing_mode,
        ),
    }


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    for result, backend in ((hip_result, "hip"), (vulkan_result, "vulkan")):
        if result.get("schema_version") != 2:
            raise ValueError("Q6 X8 comparison requires v2 timing-contract results")
        if result.get("kind") != "hipengine_micro_result":
            raise ValueError("Q6 X8 comparison inputs must be micro results")
        if result.get("bench") != BENCH_NAME or result.get("backend") != backend:
            raise ValueError("Q6 X8 comparison inputs must be matching HIP then Vulkan results")
        if result.get("classification") != "real_slice_probe":
            raise ValueError("Q6 X8 comparison inputs must be real-slice probes")
        correctness = result.get("correctness", {})
        if (
            correctness.get("status") != "pass"
            or not correctness.get("all_pass")
            or correctness.get("rows") != 3
        ):
            raise ValueError(f"{backend} Q6 X8 correctness gate did not pass")
        source = result.get("source", {})
        for field in ("repo", "commit", "source_hash"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise ValueError(f"{backend} Q6 X8 source {field} is missing")
        if not isinstance(source.get("dirty"), bool):
            raise ValueError(f"{backend} Q6 X8 source dirty is missing")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan Q6 X8 gfx architectures do not match")
    hip_device = _device_fingerprint(hip_result.get("hardware", {}).get("gpu_name", ""))
    vulkan_device = _device_fingerprint(
        vulkan_result.get("hardware", {}).get("gpu_name", "")
    )
    if (
        not hip_device
        or hip_device == "unknown"
        or hip_device != vulkan_device
    ):
        raise ValueError("HIP and Vulkan Q6 X8 device identities do not match")
    for field in ("repo", "branch", "commit", "dirty"):
        if hip_result.get("source", {}).get(field) != vulkan_result.get("source", {}).get(field):
            raise ValueError(f"HIP and Vulkan Q6 X8 source {field} values do not match")
    if hip_result.get("parameters") != vulkan_result.get("parameters"):
        raise ValueError("HIP and Vulkan Q6 X8 parameters do not match")
    parameters = hip_result["parameters"]
    required_parameters = {
        "quant",
        "activation_quant",
        "weight_layout",
        "selected_dtype",
        "output_dtype",
        "kv_type",
        "model_fingerprint",
        "rows",
        "experts",
        "in_features",
        "out_features",
        "input_scale",
        "local_size",
        "row_tile",
        "timing_mode",
        "independent_lanes",
        "repetitions",
        "warmup_iterations",
        "samples",
    }
    if set(parameters) != required_parameters:
        raise ValueError("Q6 X8 comparison parameter schema is incomplete or unexpected")
    hip_rows_list = hip_result.get("measurements", {}).get("rows", [])
    vulkan_rows_list = vulkan_result.get("measurements", {}).get("rows", [])
    if (
        not isinstance(hip_rows_list, list)
        or not isinstance(vulkan_rows_list, list)
        or not all(isinstance(row, dict) for row in hip_rows_list)
        or not all(isinstance(row, dict) for row in vulkan_rows_list)
    ):
        raise ValueError("Q6 X8 comparison inputs contain invalid rows")
    hip_rows = {_row_key(row): row for row in hip_rows_list}
    vulkan_rows = {_row_key(row): row for row in vulkan_rows_list}
    if len(hip_rows) != len(hip_rows_list) or len(vulkan_rows) != len(vulkan_rows_list):
        raise ValueError("Q6 X8 comparison inputs contain duplicate or invalid rows")
    expected_rows = _expected_row_keys(parameters)
    if set(hip_rows) != expected_rows or set(vulkan_rows) != expected_rows:
        raise ValueError(
            "Q6 X8 results must contain the exact quantize/dot/combined row triplet"
        )
    expected_dispatches = {
        "q8_1_quantize": 1,
        "x8_selected_dp4a_dot_prequantized": 1,
        "x8_selected_dp4a_quantize_plus_dot": 2,
    }
    expected_variant = {
        "q8_1_quantize": "quantize",
        "x8_selected_dp4a_dot_prequantized": "q6_x8_selected_down",
        "x8_selected_dp4a_quantize_plus_dot": "q6_x8_selected_down",
    }
    expected_shape_metadata = {
        "q8_blocks_per_row": int(parameters["in_features"]) // 32,
        "blocks_per_row": int(parameters["in_features"]) // 256,
        "out_packed": int(parameters["out_features"]) // 8,
    }
    for backend, rows in (("hip", hip_rows), ("vulkan", vulkan_rows)):
        for key, row in rows.items():
            operation = key[0]
            if row.get("backend") != backend:
                raise ValueError(f"{backend} Q6 X8 row backend metadata does not match")
            if row.get("workgroup_match") != "exact":
                raise ValueError(f"{backend} Q6 X8 row workgroup metadata does not match")
            if row.get("variant") != expected_variant[operation]:
                raise ValueError(f"{backend} Q6 X8 row variant metadata does not match")
            if not row.get("correctness_pass"):
                raise ValueError(f"{backend} Q6 X8 row correctness did not pass")
            for field, value in expected_shape_metadata.items():
                if row.get(field) != value:
                    raise ValueError(
                        f"{backend} Q6 X8 row {field} metadata does not match"
                    )
            for control in timing_contract.TIMING_CONTROLS:
                if (
                    row.get("timing", {}).get(control, {}).get("dispatches_per_iteration")
                    != expected_dispatches[operation]
                ):
                    raise ValueError(
                        f"{backend} Q6 X8 row dispatch metadata does not match"
                    )
    expected_lanes = int(parameters["independent_lanes"])
    combined_keys = [
        key
        for key in hip_rows
        if key[0] == "x8_selected_dp4a_quantize_plus_dot"
    ]
    if len(combined_keys) != 1:
        raise ValueError("Q6 X8 comparison requires exactly one combined row")
    combined_key = combined_keys[0]
    hip_combined_lanes = int(
        hip_rows[combined_key]["submission"]["queue_or_stream_count"]
    )
    vulkan_combined_lanes = int(
        vulkan_rows[combined_key]["submission"]["queue_or_stream_count"]
    )
    if (
        hip_combined_lanes != expected_lanes
        or vulkan_combined_lanes != expected_lanes
    ):
        raise ValueError(
            "HIP and Vulkan Q6 X8 combined lane counts do not match the parameter contract"
        )
    if parameters["timing_mode"] == "independent_throughput" and expected_lanes > 1:
        if hip_rows[combined_key].get("submission", {}).get("strategy") != "multi_stream":
            raise ValueError("HIP Q6 X8 combined throughput requires multi-stream submission")
        if (
            vulkan_rows[combined_key].get("submission", {}).get("strategy")
            != "vulkan_multi_queue"
        ):
            raise ValueError("Vulkan Q6 X8 combined throughput requires multi-queue submission")
        vulkan_device_metadata = vulkan_result.get("hardware", {}).get("device", {})
        if (
            int(vulkan_device_metadata.get("active_queue_count", 0)) != expected_lanes
            or not vulkan_device_metadata.get("calibrated_timestamps_extension")
            or not bool(
                vulkan_device_metadata.get("cross_queue_gpu_timing_calibrated")
            )
        ):
            raise ValueError(
                "Vulkan Q6 X8 combined lane metadata is missing or uncalibrated"
            )
        for control in timing_contract.TIMING_CONTROLS:
            clock = vulkan_rows[combined_key]["timing"][control]["gpu_elapsed"].get(
                "clock"
            )
            if clock != "vulkan_calibrated_cross_queue_timestamp":
                raise ValueError(
                    "Vulkan Q6 X8 combined GPU timing is not calibrated cross-queue"
                )
    repetitions = int(parameters["repetitions"])
    matched: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for key in sorted(hip_rows):
        hip = hip_rows[key]
        vulkan = vulkan_rows[key]
        timing_contract.validate_timed_row(hip, expected_repetitions=repetitions)
        timing_contract.validate_timed_row(vulkan, expected_repetitions=repetitions)
        ratios: dict[str, Any] = {}
        for control in timing_contract.TIMING_CONTROLS:
            try:
                gpu_ratio: dict[str, Any] = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip, vulkan, control=control, domain="gpu_elapsed"
                    ),
                }
            except ValueError as exc:
                gpu_ratio = {"status": "not_comparable", "reason": str(exc)}
            try:
                host_ratio: dict[str, Any] = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip, vulkan, control=control, domain="host_wall"
                    ),
                }
            except ValueError as exc:
                host_ratio = {
                    "status": "not_comparable_submission_contract",
                    "reason": str(exc),
                    "hip_submission": hip["submission"]["strategy"],
                    "vulkan_submission": vulkan["submission"]["strategy"],
                }
            ratios[control] = {
                "gpu_elapsed": gpu_ratio,
                "host_wall": host_ratio,
            }
            comparisons.append(
                {
                    "operation": hip["operation"],
                    "rows": hip["rows"],
                    "experts": hip["experts"],
                    "in_features": hip["in_features"],
                    "out_features": hip["out_features"],
                    "row_tile": hip["row_tile"],
                    "local_size": hip["local_size"],
                    "timing_mode": hip["timing_mode"],
                    "control": control,
                    "gpu_elapsed": gpu_ratio,
                    "host_wall": host_ratio,
                }
            )
        matched.append(
            {
                "operation": hip["operation"],
                "rows": hip["rows"],
                "experts": hip["experts"],
                "in_features": hip["in_features"],
                "out_features": hip["out_features"],
                "row_tile": hip["row_tile"],
                "local_size": hip["local_size"],
                "ratios": ratios,
                "hip_correctness_pass": hip["correctness_pass"],
                "vulkan_correctness_pass": vulkan["correctness_pass"],
            }
        )
    speedups = [
        float(row["ratios"]["burst"]["gpu_elapsed"]["vulkan_vs_hip_speedup"])
        for row in matched
        if row["ratios"]["burst"]["gpu_elapsed"]["status"] == "ok"
    ]
    hip_source = hip_result["source"]
    vulkan_source = vulkan_result["source"]
    commit_match = (
        bool(hip_source["commit"])
        and hip_source["commit"] == vulkan_source["commit"]
    )
    dirty = bool(hip_source["dirty"]) or bool(vulkan_source["dirty"])
    device_match = bool(hip_device) and hip_device == vulkan_device
    correctness_pass = (
        hip_result["correctness"].get("status") == "pass"
        and bool(hip_result["correctness"].get("all_pass"))
        and vulkan_result["correctness"].get("status") == "pass"
        and bool(vulkan_result["correctness"].get("all_pass"))
    )
    performance_claim = commit_match and not dirty and device_match and correctness_pass
    blocking_reasons = []
    if not commit_match:
        blocking_reasons.append("commit_mismatch_or_missing")
    if dirty:
        blocking_reasons.append("dirty_source")
    if not device_match:
        blocking_reasons.append("device_identity_mismatch_or_missing")
    if not correctness_pass:
        blocking_reasons.append("correctness_not_passed")
    return _json_safe(
        {
            "schema_version": 2,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "real_slice_probe",
            "performance_claim": performance_claim,
            "source": hip_source,
            "sources": {
                "hip": hip_source,
                "vulkan": vulkan_source,
            },
            "command": command,
            "hardware": {
                "hip": hip_result["hardware"],
                "vulkan": vulkan_result["hardware"],
            },
            "inputs": {
                "hip_result": hip_result.get("artifact_ref"),
                "vulkan_result": vulkan_result.get("artifact_ref"),
                "out": out_ref,
            },
            "correctness": {
                "hip": hip_result["correctness"],
                "vulkan": vulkan_result["correctness"],
            },
            "provenance": {
                "commit_match": commit_match,
                "dirty": dirty,
                "device_match": device_match,
                "hip_device_fingerprint": hip_device,
                "vulkan_device_fingerprint": vulkan_device,
                "hip_source_hash": hip_source["source_hash"],
                "vulkan_source_hash": vulkan_source["source_hash"],
                "correctness_pass": correctness_pass,
                "performance_claim": performance_claim,
                "blocking_reasons": blocking_reasons,
            },
            "comparisons": comparisons,
            "matched_rows": matched,
            "summary": {
                "matched_rows": len(matched),
                "timing_mode": parameters["timing_mode"],
                "burst_gpu_speedup_min": min(speedups) if speedups else None,
                "burst_gpu_speedup_max": max(speedups) if speedups else None,
                "host_wall_status": "not_comparable_submission_contract",
                "performance_claim": performance_claim,
            },
            "interpretation": (
                "Strictly paired Q6_K X8 selected-down q8_1+dp4a rows. GPU ratios "
                "compare identical dependency/shape/workgroup contracts; HIP direct or "
                "multi-stream host wall is rejected against Vulkan command-buffer wall."
            ),
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hip", "vulkan"))
    parser.add_argument("--compare", nargs=2, metavar=("HIP_RESULT", "VULKAN_RESULT"), type=Path)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--local-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--reps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--timing-mode",
        choices=timing_contract.TIMING_MODES,
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gfx-arch")
    parser.add_argument("--hardware-gpu")
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--environment-ref", type=Path)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=10.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.compare and args.backend:
        parser.error("--compare and --backend are mutually exclusive")
    if args.rows <= 0 or args.experts <= 0:
        parser.error("--rows and --experts must be positive")
    if args.in_features <= 0 or args.in_features % 256:
        parser.error("--in-features must be positive and divisible by 256")
    if args.out_features <= 0 or args.out_features % 8:
        parser.error("--out-features must be positive and divisible by 8")
    if args.reps <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("--reps and --samples must be positive; --warmup must be non-negative")
    if args.independent_streams <= 0:
        parser.error("--independent-streams must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare:
        hip_path, vulkan_path = args.compare
        result = build_comparison(
            json.loads(hip_path.read_text(encoding="utf-8")),
            json.loads(vulkan_path.read_text(encoding="utf-8")),
            command=[Path(sys.executable).name, *sys.argv],
            out_ref=str(args.out),
        )
    elif (args.backend or "vulkan") == "hip":
        result = _run_hip(args)
    else:
        result = _run_vulkan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
