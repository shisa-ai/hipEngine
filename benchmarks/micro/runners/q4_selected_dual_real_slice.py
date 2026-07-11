#!/usr/bin/env python3
"""Run or compare HIP/Vulkan Q4_K selected-dual q8_1+dp4a real slices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_q4_selected_dual.cpp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
VULKAN_DOT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q4_selected_dual.comp"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
HIP_TIMING = MICRO_ROOT / "hip_timing.py"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
HIP_Q4_SOURCE = (
    REPO_ROOT
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_q4_k_gemv.hip"
)
HIP_Q4_WRAPPER = HIP_Q4_SOURCE.with_suffix(".py")
SYNTHETIC_WEIGHTS = REPO_ROOT / "tests" / "_gguf_synthetic_weights.py"
BENCH_NAME = "q4_selected_dual_real_slice"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q4-selected-dual-real-slice-build")
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


timing_contract = _load_module(TIMING_CONTRACT, "micro_q4_timing_contract")
hip_timing = _load_module(HIP_TIMING, "micro_q4_hip_timing")


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_module(COLLECT_ENV, "micro_collect_env_for_q4")
    return collector.collect_environment(
        repo_root=REPO_ROOT,
        include_device_probes=not args.skip_device_probes,
        include_privileged=False,
        timeout_s=args.env_timeout_s,
        max_output_chars=args.env_max_output_chars,
    )


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_record(environment: dict[str, Any], source_hash: str) -> dict[str, Any]:
    def git_value(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = git_value("status", "--porcelain")
    return {
        "repo": str(REPO_ROOT),
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "dirty": bool(status),
        "source_hash": source_hash,
    }


def _environment_source_record(environment: dict[str, Any]) -> dict[str, Any]:
    repo = environment.get("repo")
    return dict(repo) if isinstance(repo, dict) else {}


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
        for key in ("rocm_smi_lines", "vulkan_summary_lines"):
            values = devices.get(key, [])
            if not isinstance(values, list):
                continue
            for item in values:
                text = str(item)
                if "Card Series:" in text:
                    return text.split("Card Series:", 1)[1].strip()
                if "deviceName" in text and "=" in text:
                    return text.split("=", 1)[1].strip()
        for value in devices.values():
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item)
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_workgroups(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value not in (64, 128, 256) for value in values):
        raise ValueError("workgroups must be a comma-separated subset of 64,128,256")
    return list(dict.fromkeys(values))


def _effective_independent_lanes(args: argparse.Namespace) -> int:
    if args.timing_mode != "independent_throughput":
        return 1
    return min(args.independent_streams, args.reps, 4)


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(bits, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32).reshape(values.shape)


def _make_x_slices(
    repetitions: int,
    x_rows: int,
    in_features: int,
    input_scale: float,
) -> np.ndarray:
    slices = []
    for rep in range(repetitions):
        rng = np.random.default_rng(27 + rep * 104729)
        values = (rng.standard_normal((x_rows, in_features)) * input_scale).astype(np.float32)
        slices.append(_f32_to_bf16_bits(values))
    return np.ascontiguousarray(np.stack(slices))


def _selected_ids(rows: int, experts: int) -> np.ndarray:
    return np.ascontiguousarray((np.arange(rows) % experts).astype(np.int64))


def _top1(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(expected.argmax(axis=-1) == actual.argmax(axis=-1)))


def _max_kl_divergence(expected: np.ndarray, actual: np.ndarray) -> float:
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    expected_shift = expected64 - np.max(expected64, axis=-1, keepdims=True)
    actual_shift = actual64 - np.max(actual64, axis=-1, keepdims=True)
    expected_log_p = expected_shift - np.log(
        np.sum(np.exp(expected_shift), axis=-1, keepdims=True)
    )
    actual_log_p = actual_shift - np.log(
        np.sum(np.exp(actual_shift), axis=-1, keepdims=True)
    )
    expected_p = np.exp(expected_log_p)
    return float(
        np.max(np.sum(expected_p * (expected_log_p - actual_log_p), axis=-1))
    )


def _compare_dual_slices(
    expected_a: np.ndarray,
    expected_b: np.ndarray,
    actual_a: np.ndarray,
    actual_b: np.ndarray,
    output_slices: list[int],
    expected_slices: list[int],
    *,
    require_exact: bool,
) -> dict[str, Any]:
    if not output_slices or len(output_slices) != len(expected_slices):
        raise ValueError("output and expected slice lists must be non-empty and matched")
    max_abs = 0.0
    mean_abs = 0.0
    kl_divergence = 0.0
    top1 = 1.0
    exact_mismatches = 0
    for output_slice, expected_slice in zip(output_slices, expected_slices, strict=True):
        for expected, actual in (
            (expected_a[expected_slice], actual_a[output_slice]),
            (expected_b[expected_slice], actual_b[output_slice]),
        ):
            exact_mismatches += int(np.count_nonzero(expected != actual))
            expected_f32 = _bf16_bits_to_f32(expected)
            actual_f32 = _bf16_bits_to_f32(actual)
            diff = np.abs(expected_f32 - actual_f32)
            max_abs = max(max_abs, float(np.max(diff)))
            mean_abs = max(mean_abs, float(np.mean(diff)))
            kl_divergence = max(
                kl_divergence, _max_kl_divergence(expected_f32, actual_f32)
            )
            top1 = min(top1, _top1(expected_f32, actual_f32))
    passed = (
        exact_mismatches == 0
        if require_exact
        else kl_divergence <= 0.05 and top1 >= 0.90
    )
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "kl_divergence": kl_divergence,
        "top1": top1,
        "exact_bf16_mismatches": exact_mismatches,
        "outputs_checked": len(output_slices),
        "pass": passed,
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
    submission_strategy: str | None = None,
    gpu_clock_override: str | None = None,
) -> dict[str, Any]:
    correctness_pass = bool(single_correctness["pass"] and burst_correctness["pass"])
    contract = timing_contract.make_timed_row_contract(
        timing_mode=timing_mode,
        backend=backend,
        repetitions=repetitions,
        dispatches_per_iteration=dispatches_per_iteration,
        dependency_validation_status="pass" if correctness_pass else "fail",
        submission=timing_contract.make_submission(
            strategy=submission_strategy
            or (
                "multi_stream"
                if backend == "hip" and timing_mode == "independent_throughput"
                else "direct"
                if backend == "hip"
                else "vulkan_command_buffer"
            ),
            queue_or_stream_count=stream_count,
            recording_in_timed_region=False,
        ),
        single_timing=timing_contract.make_timing_control(
            logical_iterations=1,
            dispatches_per_iteration=dispatches_per_iteration,
            gpu_samples_us=single_samples.gpu_sequence_us if gpu_timing_supported else None,
            host_samples_us=single_samples.host_sequence_us,
            gpu_clock=gpu_clock_override
            or ("hip_event" if backend == "hip" else "vulkan_timestamp"),
            gpu_status="ok" if gpu_timing_supported else "unsupported",
        ),
        burst_timing=timing_contract.make_timing_control(
            logical_iterations=repetitions,
            dispatches_per_iteration=dispatches_per_iteration,
            gpu_samples_us=burst_samples.gpu_sequence_us if gpu_timing_supported else None,
            host_samples_us=burst_samples.host_sequence_us,
            gpu_clock=gpu_clock_override
            or ("hip_event" if backend == "hip" else "vulkan_timestamp"),
            gpu_status="ok" if gpu_timing_supported else "unsupported",
        ),
        correctness=timing_contract.make_correctness(
            status="pass" if correctness_pass else "fail",
            oracle=(
                "downstream Q4_K selected-dual BF16 equivalence after q8_1 quantize"
                if operation == "q8_1_quantize"
                else "exact BF16 output versus isolated HIP q8_1+Q4_K dp4a launches"
                if backend == "hip"
                else "full CPU q8_1 plus Q4_K selected-dual BF16 logits with KL <= 0.05 and top-1 >= 90%"
            ),
            logical_iterations=repetitions,
            coverage=(
                "all_dispatches"
                if timing_mode == "independent_throughput"
                else "chained_final_state"
            ),
            synchronization_method=(
                "disjoint_xq_and_dual_outputs"
                if timing_mode == "independent_throughput"
                else "hip_stream_order"
                if backend == "hip"
                else "vulkan_compute_barriers"
            ),
            barrier_count=barrier_count,
        ),
    )
    median_domain = "gpu_elapsed" if gpu_timing_supported else "host_wall"
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
            "xq_partitioning": "disjoint" if timing_mode == "independent_throughput" else "shared",
            "output_partitioning": "disjoint_dual" if timing_mode == "independent_throughput" else "shared_dual",
        },
        "correctness_pass": correctness_pass,
        "median_us": contract["timing"]["burst"][median_domain]["per_iteration_us"]["median"],
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
    return timer.measure(1, samples, launch), timer.measure(repetitions, samples, launch)


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
        gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
        gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    )
    from tests._gguf_synthetic_weights import make_q4_k_weight

    if args.gfx_arch:
        os.environ["HIPENGINE_HIP_ARCH"] = args.gfx_arch
    if args.compiler_version_file:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            HIP_Q4_SOURCE,
            HIP_Q4_WRAPPER,
            SYNTHETIC_WEIGHTS,
            TIMING_CONTRACT,
            HIP_TIMING,
        ]
    )
    runtime = get_hip_runtime()
    library = build_gguf_q4_k_gemv(
        load=True, require_cached=args.require_cached_build
    )
    local_sizes = _parse_workgroups(args.workgroups)
    independent_lanes = _effective_independent_lanes(args)
    work_repetitions = max(args.reps, args.warmup, 1)
    x = _make_x_slices(
        work_repetitions,
        args.x_rows,
        args.in_features,
        args.input_scale,
    )
    selected = _selected_ids(args.rows, args.experts)
    base = make_q4_k_weight(args.out_features, args.in_features)
    qa = np.ascontiguousarray(
        np.stack(
            [np.roll(base, expert % args.out_features, axis=0) for expert in range(args.experts)]
        )
    )
    qb = np.ascontiguousarray(
        np.stack(
            [
                np.roll(base, (expert * 3) % args.out_features, axis=0)
                for expert in range(args.experts)
            ]
        )
    )
    x_slice_bytes = args.x_rows * args.in_features * 2
    xq_slice_bytes = args.x_rows * (args.in_features // 32) * Q8_1_BLOCK_BYTES
    out_slice_bytes = args.rows * args.out_features * 2
    output_slices = work_repetitions if args.timing_mode == "independent_throughput" else 1
    buffers = []

    def dev(array: np.ndarray):
        buffer = malloc(array.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
        buffers.append(buffer)
        return buffer

    rows_out: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    try:
        x_buf = dev(x)
        selected_buf = dev(selected)
        qa_buf = dev(qa)
        qb_buf = dev(qb)
        xq_pre_buf = malloc(work_repetitions * xq_slice_bytes, runtime=runtime)
        xq_work_buf = malloc(output_slices * xq_slice_bytes, runtime=runtime)
        out_a_buf = malloc(output_slices * out_slice_bytes, runtime=runtime)
        out_b_buf = malloc(output_slices * out_slice_bytes, runtime=runtime)
        buffers.extend((xq_pre_buf, xq_work_buf, out_a_buf, out_b_buf))

        for rep in range(work_repetitions):
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr + rep * x_slice_bytes,
                xq_pre_buf.ptr + rep * xq_slice_bytes,
                args.x_rows,
                args.in_features,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()

        for local_index, local_size in enumerate(local_sizes):
            oracle_a_buf = malloc(args.reps * out_slice_bytes, runtime=runtime)
            oracle_b_buf = malloc(args.reps * out_slice_bytes, runtime=runtime)
            raw_a_buf = malloc(out_slice_bytes, runtime=runtime)
            raw_b_buf = malloc(out_slice_bytes, runtime=runtime)
            buffers.extend((oracle_a_buf, oracle_b_buf, raw_a_buf, raw_b_buf))
            for rep in range(args.reps):
                gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                    xq_pre_buf.ptr + rep * xq_slice_bytes,
                    selected_buf.ptr,
                    qa_buf.ptr,
                    qb_buf.ptr,
                    oracle_a_buf.ptr + rep * out_slice_bytes,
                    oracle_b_buf.ptr + rep * out_slice_bytes,
                    args.x_rows,
                    args.rows,
                    args.experts,
                    args.in_features,
                    args.out_features,
                    threads=local_size,
                    library=library,
                    runtime=runtime,
                )
            gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                x_buf.ptr,
                selected_buf.ptr,
                qa_buf.ptr,
                qb_buf.ptr,
                raw_a_buf.ptr,
                raw_b_buf.ptr,
                args.x_rows,
                args.rows,
                args.experts,
                args.in_features,
                args.out_features,
                threads=local_size,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            oracle_a = np.empty(
                (args.reps, args.rows, args.out_features), dtype=np.uint16
            )
            oracle_b = np.empty_like(oracle_a)
            raw_a = np.empty((args.rows, args.out_features), dtype=np.uint16)
            raw_b = np.empty_like(raw_a)
            copy_device_to_host(host_array_ptr(oracle_a), oracle_a_buf, runtime=runtime)
            copy_device_to_host(host_array_ptr(oracle_b), oracle_b_buf, runtime=runtime)
            copy_device_to_host(host_array_ptr(raw_a), raw_a_buf, runtime=runtime)
            copy_device_to_host(host_array_ptr(raw_b), raw_b_buf, runtime=runtime)
            quality = _compare_dual_slices(
                raw_a[None, ...],
                raw_b[None, ...],
                oracle_a[:1],
                oracle_b[:1],
                [0],
                [0],
                require_exact=False,
            )
            quality_rows.append({"local_size": local_size, **quality})

            with hip_timing.HipSequenceTimer(
                runtime, args.timing_mode, independent_lanes
            ) as timer:
                def quantize(rep: int, stream: int) -> None:
                    target = rep if args.timing_mode == "independent_throughput" else 0
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_buf.ptr + rep * x_slice_bytes,
                        xq_work_buf.ptr + target * xq_slice_bytes,
                        args.x_rows,
                        args.in_features,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

                def dot(rep: int, stream: int) -> None:
                    target = rep if args.timing_mode == "independent_throughput" else 0
                    gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                        xq_pre_buf.ptr + rep * xq_slice_bytes,
                        selected_buf.ptr,
                        qa_buf.ptr,
                        qb_buf.ptr,
                        out_a_buf.ptr + target * out_slice_bytes,
                        out_b_buf.ptr + target * out_slice_bytes,
                        args.x_rows,
                        args.rows,
                        args.experts,
                        args.in_features,
                        args.out_features,
                        threads=local_size,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

                def combined(rep: int, stream: int) -> None:
                    target = rep if args.timing_mode == "independent_throughput" else 0
                    quantize(rep, stream)
                    gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                        xq_work_buf.ptr + target * xq_slice_bytes,
                        selected_buf.ptr,
                        qa_buf.ptr,
                        qb_buf.ptr,
                        out_a_buf.ptr + target * out_slice_bytes,
                        out_b_buf.ptr + target * out_slice_bytes,
                        args.x_rows,
                        args.rows,
                        args.experts,
                        args.in_features,
                        args.out_features,
                        threads=local_size,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

                def validate(operation: str, logical_iterations: int) -> dict[str, Any]:
                    launch = {"quantize": quantize, "dot": dot, "combined": combined}[operation]
                    timer.run_and_wait(logical_iterations, launch)
                    if operation == "quantize":
                        if args.timing_mode == "independent_throughput":
                            for rep in range(logical_iterations):
                                gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                                    xq_work_buf.ptr + rep * xq_slice_bytes,
                                    selected_buf.ptr,
                                    qa_buf.ptr,
                                    qb_buf.ptr,
                                    out_a_buf.ptr + rep * out_slice_bytes,
                                    out_b_buf.ptr + rep * out_slice_bytes,
                                    args.x_rows,
                                    args.rows,
                                    args.experts,
                                    args.in_features,
                                    args.out_features,
                                    threads=local_size,
                                    library=library,
                                    runtime=runtime,
                                )
                        else:
                            gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                                xq_work_buf.ptr,
                                selected_buf.ptr,
                                qa_buf.ptr,
                                qb_buf.ptr,
                                out_a_buf.ptr,
                                out_b_buf.ptr,
                                args.x_rows,
                                args.rows,
                                args.experts,
                                args.in_features,
                                args.out_features,
                                threads=local_size,
                                library=library,
                                runtime=runtime,
                            )
                    runtime.device_synchronize()
                    actual_a = np.empty(
                        (output_slices, args.rows, args.out_features), dtype=np.uint16
                    )
                    actual_b = np.empty_like(actual_a)
                    copy_device_to_host(host_array_ptr(actual_a), out_a_buf, runtime=runtime)
                    copy_device_to_host(host_array_ptr(actual_b), out_b_buf, runtime=runtime)
                    if args.timing_mode == "independent_throughput":
                        output_indices = list(range(logical_iterations))
                        expected_indices = list(range(logical_iterations))
                    else:
                        output_indices = [0]
                        expected_indices = [logical_iterations - 1]
                    return _compare_dual_slices(
                        oracle_a,
                        oracle_b,
                        actual_a,
                        actual_b,
                        output_indices,
                        expected_indices,
                        require_exact=True,
                    )

                operations = [
                    ("selected_dual_dp4a_dot_prequantized", "dot", dot, 1),
                    ("selected_dual_dp4a_quantize_plus_dot", "combined", combined, 2),
                ]
                if local_index == 0:
                    operations.insert(0, ("q8_1_quantize", "quantize", quantize, 1))
                for operation, validation_name, launch, dispatches in operations:
                    single_samples, burst_samples = _measure_hip_operation(
                        timer,
                        launch,
                        repetitions=args.reps,
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    single_correctness = validate(validation_name, 1)
                    burst_correctness = validate(validation_name, args.reps)
                    if not single_correctness["pass"] or not burst_correctness["pass"]:
                        raise RuntimeError(
                            f"HIP Q4 selected-dual {operation} sequence validation failed"
                        )
                    operation_local_size = 32 if operation == "q8_1_quantize" else local_size
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
                            shape_fields={
                                "quant": "q4_k",
                                "buffer_abi": "hip_raw_device_pointer_q8_1_q4_k",
                                "input_scale": args.input_scale,
                                "x_rows": args.x_rows,
                                "rows": args.rows,
                                "experts": args.experts,
                                "in_features": args.in_features,
                                "out_features": args.out_features,
                                "local_size": operation_local_size,
                                "workgroup_match": "exact",
                            },
                        )
                    )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    correctness_pass = bool(rows_out) and all(row["correctness_pass"] for row in rows_out)
    quality_pass = bool(quality_rows) and all(row["pass"] for row in quality_rows)
    result = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": "hip",
        "classification": "real_slice_probe",
        "source": _source_record(environment, source_hash),
        "environment_source": _environment_source_record(environment),
        "hardware": {
            "gpu_name": _infer_gpu_name(environment, args.hardware_gpu),
            "gfx_arch": _infer_gfx_arch(environment, args.gfx_arch),
        },
        "command": [Path(sys.executable).name, *sys.argv],
        "parameters": {
            "input_scale": args.input_scale,
            "x_rows": args.x_rows,
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "workgroups": local_sizes,
            "timing_mode": args.timing_mode,
            "repetitions": args.reps,
            "warmup_logical_iterations": args.warmup,
            "work_slices": work_repetitions,
            "samples": args.samples,
            "requested_independent_lanes": args.independent_streams,
            "actual_independent_lanes": independent_lanes,
            "buffer_abi": "hip_raw_device_pointer_q8_1_q4_k",
        },
        "artifact_ref": str(args.out) if args.out else None,
        "measurements": {"rows": rows_out},
        "correctness": {
            "status": "pass" if correctness_pass and quality_pass else "fail",
            "timed_sequence_pass": correctness_pass,
            "quality_vs_raw_pass": quality_pass,
            "quality_vs_raw": quality_rows,
        },
    }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


def _run_command(command: list[str], *, echo: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
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
    return shlex.split(completed.stdout.strip()) if completed.returncode == 0 else ["-lvulkan"]


def _compile_shader(shader: Path, output: Path, defines: list[str]) -> list[str]:
    compiler = shutil.which("glslc")
    if compiler:
        command = [
            compiler,
            "--target-env=vulkan1.2",
            "-O",
            *defines,
            str(shader),
            "-o",
            str(output),
        ]
    else:
        compiler = shutil.which("glslangValidator")
        if not compiler:
            raise RuntimeError("neither glslc nor glslangValidator is available")
        command = [
            compiler,
            "-V",
            "--target-env",
            "vulkan1.2",
            *defines,
            str(shader),
            "-o",
            str(output),
        ]
    completed = _run_command(command)
    if completed.returncode != 0:
        raise RuntimeError(f"shader build failed: {shader}")
    return command


def _compile_harness(output: Path) -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found")
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(VULKAN_HARNESS),
        "-o",
        str(output),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(command)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan Q4 selected-dual harness build failed")
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
    local_sizes = _parse_workgroups(args.workgroups)
    requested_lanes = _effective_independent_lanes(args)
    work_repetitions = max(args.reps, args.warmup, 1)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    quant_spv = args.build_dir / "q8_1_quantize.spv"
    executable = args.build_dir / "vulkan_q4_selected_dual"
    x_path = args.build_dir / "q4_selected_dual_x_bf16.bin"
    x = _make_x_slices(
        work_repetitions,
        args.x_rows,
        args.in_features,
        args.input_scale,
    )
    x.tofile(x_path)
    commands: list[dict[str, Any]] = []
    commands.append(
        {"kind": "compile_shader", "command": _compile_shader(VULKAN_QUANT_SHADER, quant_spv, [])}
    )
    commands.append({"kind": "compile_harness", "command": _compile_harness(executable)})
    rows_out: list[dict[str, Any]] = []
    quantize_retained = False
    device: dict[str, Any] = {}
    actual_lane_count: int | None = None
    isolation_by_workgroup: dict[str, Any] = {}
    for local_size in local_sizes:
        dot_spv = args.build_dir / f"q4_selected_dual_wg{local_size}.spv"
        commands.append(
            {
                "kind": "compile_shader",
                "command": _compile_shader(
                    VULKAN_DOT_SHADER,
                    dot_spv,
                    [f"-DHIPENGINE_LOCAL_SIZE_X={local_size}"],
                ),
            }
        )
        raw_path = args.build_dir / f"vulkan-q4-selected-dual-wg{local_size}-raw.json"
        run_command = [
            str(executable),
            "--quantize-spirv",
            str(quant_spv),
            "--dot-spirv",
            str(dot_spv),
            "--json",
            str(raw_path),
            "--x-bf16",
            str(x_path),
            "--x-rows",
            str(args.x_rows),
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
            str(local_size),
            "--reps",
            str(args.reps),
            "--warmup",
            str(args.warmup),
            "--samples",
            str(args.samples),
            "--independent-queues",
            str(requested_lanes),
            "--timing-mode",
            args.timing_mode,
            "--device-index",
            str(args.device_index),
        ]
        completed = _run_command(run_command)
        commands.append(
            {"kind": "run_harness", "command": run_command, "returncode": completed.returncode}
        )
        if completed.returncode != 0:
            raise RuntimeError("Vulkan Q4 selected-dual run failed")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        isolation_by_workgroup[str(local_size)] = raw.get("isolation", {})
        device = raw.get("hardware", {})
        timing_config = raw.get("timing_config", {})
        raw_lane_count = int(timing_config.get("actual_lane_count", 1))
        if actual_lane_count is not None and actual_lane_count != raw_lane_count:
            raise RuntimeError("Vulkan Q4 workgroups selected different queue lane counts")
        actual_lane_count = raw_lane_count
        operations = (
            ("q8_1_quantize", 1),
            ("selected_dual_dp4a_dot_prequantized", 1),
            ("selected_dual_dp4a_quantize_plus_dot", 2),
        )
        for operation, dispatches in operations:
            if operation == "q8_1_quantize" and quantize_retained:
                continue
            if operation == "q8_1_quantize":
                quantize_retained = True
            operation_timing = raw["timing"][operation]
            operation_correctness = raw["correctness"][operation]
            single_samples = hip_timing.HipTimingSamples(
                list(operation_timing["single"]["gpu_samples_us"]),
                list(operation_timing["single"]["host_samples_us"]),
            )
            burst_samples = hip_timing.HipTimingSamples(
                list(operation_timing["burst"]["gpu_samples_us"]),
                list(operation_timing["burst"]["host_samples_us"]),
            )
            if args.timing_mode == "serial_latency":
                barrier_count = args.reps - 1
                if dispatches == 2:
                    barrier_count = 3 * args.reps - 2
            else:
                barrier_count = args.reps if dispatches == 2 else 0
            operation_local_size = 32 if operation == "q8_1_quantize" else local_size
            operation_stream_count = (
                actual_lane_count
                if operation == "selected_dual_dp4a_quantize_plus_dot"
                and args.timing_mode == "independent_throughput"
                else 1
            )
            row = _make_operation_row(
                backend="vulkan",
                timing_mode=args.timing_mode,
                operation=operation,
                repetitions=args.reps,
                dispatches_per_iteration=dispatches,
                stream_count=operation_stream_count,
                single_samples=single_samples,
                burst_samples=burst_samples,
                single_correctness=dict(operation_correctness["single"]),
                burst_correctness=dict(operation_correctness["burst"]),
                barrier_count=barrier_count,
                gpu_timing_supported=bool(raw.get("gpu_timestamps_supported")),
                submission_strategy=(
                    "vulkan_multi_queue"
                    if operation == "selected_dual_dp4a_quantize_plus_dot"
                    and args.timing_mode == "independent_throughput"
                    and actual_lane_count >= 2
                    else "vulkan_command_buffer"
                ),
                gpu_clock_override=(
                    "vulkan_calibrated_cross_queue_timestamp"
                    if operation == "selected_dual_dp4a_quantize_plus_dot"
                    and args.timing_mode == "independent_throughput"
                    and actual_lane_count >= 2
                    else None
                ),
                shape_fields={
                    "quant": "q4_k",
                    "buffer_abi": "vulkan_storage_buffer_q8_1_q4_k",
                    "input_scale": args.input_scale,
                    "x_rows": args.x_rows,
                    "rows": args.rows,
                    "experts": args.experts,
                    "in_features": args.in_features,
                    "out_features": args.out_features,
                    "local_size": operation_local_size,
                    "workgroup_match": "exact",
                    "hardware": device,
                },
            )
            lane_samples = {
                control: operation_timing[control].get("lane_gpu_samples_us", [])
                for control in timing_contract.TIMING_CONTROLS
            }
            if any(lane_samples.values()):
                row["lane_gpu_samples_us"] = lane_samples
                row["calibrated_timestamp_domain"] = bool(
                    timing_config.get("calibrated_timestamp_domain")
                )
                row["calibrated_timestamps_extension"] = timing_config.get(
                    "calibrated_timestamps_extension"
                )
            rows_out.append(row)
    correctness_pass = bool(rows_out) and all(row["correctness_pass"] for row in rows_out)
    result = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": "vulkan",
        "classification": "real_slice_probe",
        "source": _source_record(environment, source_hash),
        "environment_source": _environment_source_record(environment),
        "hardware": {
            "gpu_name": _infer_gpu_name(environment, args.hardware_gpu, device.get("device_name")),
            "gfx_arch": _infer_gfx_arch(environment, args.gfx_arch),
            "device": device,
        },
        "command": [Path(sys.executable).name, *sys.argv],
        "parameters": {
            "input_scale": args.input_scale,
            "x_rows": args.x_rows,
            "rows": args.rows,
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "workgroups": local_sizes,
            "timing_mode": args.timing_mode,
            "repetitions": args.reps,
            "warmup_logical_iterations": args.warmup,
            "work_slices": work_repetitions,
            "samples": args.samples,
            "requested_independent_lanes": args.independent_streams,
            "actual_independent_lanes": actual_lane_count or 1,
            "buffer_abi": "vulkan_storage_buffer_q8_1_q4_k",
        },
        "artifact_ref": str(args.out) if args.out else None,
        "wrapper": {
            "schema": "hipengine.micro.q4_selected_dual_real_slice_runner.v2",
            "commands": commands,
            "build_dir": str(args.build_dir),
        },
        "measurements": {"rows": rows_out},
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "operation-specific downstream Q4_K selected-dual BF16 equivalence",
            "rows": len(rows_out),
            "q8_dot_isolation_by_workgroup": isolation_by_workgroup,
        },
    }
    if args.environment_ref:
        result["environment_ref"] = str(args.environment_ref)
    else:
        result["environment"] = environment
    return _json_safe(result)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["operation"]),
        int(row["x_rows"]),
        int(row["rows"]),
        int(row["experts"]),
        int(row["in_features"]),
        int(row["out_features"]),
        int(row["local_size"]),
    )


def _device_fingerprint(name: Any) -> str:
    text = str(name).lower()
    match = re.search(r"radeon\s+(\d+s)", text)
    if match:
        return f"radeon_{match.group(1)}"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _expected_row_keys(parameters: dict[str, Any]) -> set[tuple[Any, ...]]:
    shape = (
        int(parameters["x_rows"]),
        int(parameters["rows"]),
        int(parameters["experts"]),
        int(parameters["in_features"]),
        int(parameters["out_features"]),
    )
    keys = {("q8_1_quantize", *shape, 32)}
    for workgroup in parameters["workgroups"]:
        keys.add(("selected_dual_dp4a_dot_prequantized", *shape, int(workgroup)))
        keys.add(("selected_dual_dp4a_quantize_plus_dot", *shape, int(workgroup)))
    return keys


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    if hip_result.get("schema_version") != 2 or vulkan_result.get("schema_version") != 2:
        raise ValueError("Q4 comparison requires v2 timing-contract results")
    if hip_result.get("backend") != "hip" or vulkan_result.get("backend") != "vulkan":
        raise ValueError("Q4 comparison inputs must be HIP then Vulkan")
    for backend, result in (("HIP", hip_result), ("Vulkan", vulkan_result)):
        if result.get("kind") != "hipengine_micro_result":
            raise ValueError(f"{backend} Q4 result kind is not hipengine_micro_result")
        if result.get("bench") != BENCH_NAME:
            raise ValueError(f"{backend} Q4 result bench does not match {BENCH_NAME}")
        if result.get("classification") != "real_slice_probe":
            raise ValueError(f"{backend} Q4 result classification is not real_slice_probe")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan Q4 gfx architectures do not match")
    hip_parameters = hip_result.get("parameters", {})
    vulkan_parameters = vulkan_result.get("parameters", {})
    for field in (
        "input_scale",
        "x_rows",
        "rows",
        "experts",
        "in_features",
        "out_features",
        "workgroups",
        "timing_mode",
        "repetitions",
        "warmup_logical_iterations",
        "samples",
        "requested_independent_lanes",
        "actual_independent_lanes",
    ):
        if field not in hip_parameters or field not in vulkan_parameters:
            raise ValueError(f"HIP and Vulkan Q4 results must both declare {field}")
        if hip_parameters[field] != vulkan_parameters[field]:
            raise ValueError(f"HIP and Vulkan Q4 {field} values do not match")
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
        raise ValueError("HIP and Vulkan Q4 results must use the same single timing mode")
    hip_comparable = hip_result.get("measurements", {}).get("rows", [])
    vulkan_comparable = vulkan_result.get("measurements", {}).get("rows", [])
    if (
        not isinstance(hip_comparable, list)
        or not isinstance(vulkan_comparable, list)
        or not hip_comparable
        or not vulkan_comparable
        or not all(isinstance(row, dict) for row in hip_comparable)
        or not all(isinstance(row, dict) for row in vulkan_comparable)
    ):
        raise ValueError("Q4 comparison requires non-empty exact measurement rows")
    if any(row.get("backend") != "hip" for row in hip_comparable) or any(
        row.get("backend") != "vulkan" for row in vulkan_comparable
    ):
        raise ValueError("Q4 measurement row backends do not match their result backends")
    for backend, rows, parameters in (
        ("HIP", hip_comparable, hip_parameters),
        ("Vulkan", vulkan_comparable, vulkan_parameters),
    ):
        for row in rows:
            if row.get("quant") != "q4_k":
                raise ValueError(f"{backend} Q4 row quant must be q4_k")
            if row.get("workgroup_match") != "exact":
                raise ValueError(f"{backend} Q4 row workgroup_match must be exact")
            if row.get("input_scale") != parameters["input_scale"]:
                raise ValueError(f"{backend} Q4 row input_scale does not match parameters")
            if row.get("timing_mode") != parameters["timing_mode"]:
                raise ValueError(f"{backend} Q4 row timing_mode does not match parameters")
    hip_rows = {_row_key(row): row for row in hip_comparable}
    vulkan_rows = {_row_key(row): row for row in vulkan_comparable}
    if len(hip_rows) != len(hip_comparable) or len(vulkan_rows) != len(vulkan_comparable):
        raise ValueError("Q4 comparison inputs contain duplicate exact rows")
    expected_rows = _expected_row_keys(hip_parameters)
    if set(hip_rows) != expected_rows or set(vulkan_rows) != expected_rows:
        expected_count = 1 + 2 * len(hip_parameters["workgroups"])
        raise ValueError(
            f"Q4 results must contain the exact expected {expected_count}-row set"
        )
    if set(hip_rows) != set(vulkan_rows):
        raise ValueError("HIP and Vulkan Q4 exact row sets do not match")
    if hip_parameters["timing_mode"] == "independent_throughput":
        expected_lanes = int(hip_parameters["actual_independent_lanes"])
        for key, hip_row in hip_rows.items():
            if key[0] != "selected_dual_dp4a_quantize_plus_dot":
                continue
            vulkan_row = vulkan_rows[key]
            if (
                int(hip_row.get("submission", {}).get("queue_or_stream_count", 0))
                != expected_lanes
                or int(
                    vulkan_row.get("submission", {}).get("queue_or_stream_count", 0)
                )
                != expected_lanes
            ):
                raise ValueError("Q4 combined HIP/Vulkan lane counts do not match")
            if vulkan_row.get("submission", {}).get("strategy") != "vulkan_multi_queue":
                raise ValueError("Q4 combined Vulkan throughput requires multi-queue submission")
            if not vulkan_row.get("calibrated_timestamp_domain"):
                raise ValueError("Q4 combined Vulkan throughput requires calibrated timestamps")
            for control in timing_contract.TIMING_CONTROLS:
                if (
                    vulkan_row["timing"][control]["gpu_elapsed"].get("clock")
                    != "vulkan_calibrated_cross_queue_timestamp"
                ):
                    raise ValueError("Q4 combined Vulkan GPU timing is not cross-queue calibrated")
    matched = []
    for key in sorted(hip_rows):
        hip_row = hip_rows[key]
        vulkan_row = vulkan_rows[key]
        timing_contract.dependency_signature(hip_row)
        timing_contract.dependency_signature(vulkan_row)
        ratios: dict[str, Any] = {}
        for control in timing_contract.TIMING_CONTROLS:
            try:
                gpu = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip_row,
                        vulkan_row,
                        control=control,
                        domain="gpu_elapsed",
                    ),
                }
            except ValueError as exc:
                gpu = {"status": "not_comparable", "reason": str(exc)}
            try:
                host = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip_row,
                        vulkan_row,
                        control=control,
                        domain="host_wall",
                    ),
                }
            except ValueError as exc:
                host = {
                    "status": "not_comparable_submission_contract",
                    "reason": str(exc),
                    "hip_submission": hip_row["submission"]["strategy"],
                    "vulkan_submission": vulkan_row["submission"]["strategy"],
                }
            ratios[control] = {"gpu_elapsed": gpu, "host_wall": host}
        matched.append(
            {
                "operation": key[0],
                "x_rows": key[1],
                "rows": key[2],
                "experts": key[3],
                "in_features": key[4],
                "out_features": key[5],
                "local_size": key[6],
                "ratios": ratios,
                "hip_correctness_pass": hip_row.get("correctness_pass"),
                "vulkan_correctness_pass": vulkan_row.get("correctness_pass"),
            }
        )
    comparisons = [
        {
            "operation": row["operation"],
            "x_rows": row["x_rows"],
            "rows": row["rows"],
            "experts": row["experts"],
            "in_features": row["in_features"],
            "out_features": row["out_features"],
            "local_size": row["local_size"],
            "timing_mode": next(iter(hip_modes)),
            "control": control,
            "gpu_elapsed": row["ratios"][control]["gpu_elapsed"],
            "host_wall": row["ratios"][control]["host_wall"],
        }
        for row in matched
        for control in timing_contract.TIMING_CONTROLS
    ]
    hip_source = hip_result.get("source", {})
    vulkan_source = vulkan_result.get("source", {})
    hip_commit = str(hip_source.get("commit", ""))
    vulkan_commit = str(vulkan_source.get("commit", ""))
    commit_match = bool(hip_commit) and hip_commit == vulkan_commit
    dirty = bool(hip_source.get("dirty")) or bool(vulkan_source.get("dirty"))
    hip_device = _device_fingerprint(hip_result.get("hardware", {}).get("gpu_name", ""))
    vulkan_device = _device_fingerprint(
        vulkan_result.get("hardware", {}).get("gpu_name", "")
    )
    device_match = bool(hip_device) and hip_device == vulkan_device
    correctness_pass = (
        hip_result.get("correctness", {}).get("status") == "pass"
        and vulkan_result.get("correctness", {}).get("status") == "pass"
    )
    performance_claim = commit_match and not dirty and device_match and correctness_pass
    provenance_reasons = []
    if not commit_match:
        provenance_reasons.append("commit_mismatch_or_missing")
    if dirty:
        provenance_reasons.append("dirty_source")
    if not device_match:
        provenance_reasons.append("device_identity_mismatch_or_missing")
    if not correctness_pass:
        provenance_reasons.append("correctness_not_passed")
    return _json_safe(
        {
            "schema_version": 2,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "real_slice_probe",
            "performance_claim": performance_claim,
            "source": hip_source,
            "sources": {"hip": hip_source, "vulkan": vulkan_source},
            "environment_source": {
                "hip": hip_result.get("environment_source", {}),
                "vulkan": vulkan_result.get("environment_source", {}),
            },
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
            "provenance": {
                "commit_match": commit_match,
                "dirty": dirty,
                "device_match": device_match,
                "hip_device_fingerprint": hip_device,
                "vulkan_device_fingerprint": vulkan_device,
                "performance_claim": performance_claim,
                "blocking_reasons": provenance_reasons,
            },
            "matched_rows": matched,
            "comparisons": comparisons,
            "summary": {
                "matched_rows": len(matched),
                "timing_mode": next(iter(hip_modes)),
                "host_wall_status": "not_comparable_backend_submission_contract",
            },
            "interpretation": (
                "Matched Q4_K selected-dual q8_1+dp4a GPU timing. Host wall is not "
                "compared across HIP direct/multi-stream launches and Vulkan pre-recorded "
                "single/multi-queue submission."
            ),
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hip", "vulkan"))
    parser.add_argument("--compare", nargs=2, metavar=("HIP_RESULT", "VULKAN_RESULT"), type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--environment-ref", type=Path)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gfx-arch")
    parser.add_argument("--hardware-gpu")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--x-rows", type=int, default=4)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--workgroups", default="64,128,256")
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
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=10.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.reps <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("--reps and --samples must be positive; --warmup must be non-negative")
    if args.independent_streams <= 0:
        parser.error("--independent-streams must be positive")
    if args.x_rows <= 0 or args.rows <= 0 or args.rows % args.x_rows:
        parser.error("--rows must be positive and divisible by --x-rows")
    if args.experts <= 0 or args.in_features <= 0 or args.in_features % 256:
        parser.error("--experts must be positive and --in-features divisible by 256")
    if args.out_features <= 0:
        parser.error("--out-features must be positive")
    try:
        _parse_workgroups(args.workgroups)
    except ValueError as exc:
        parser.error(str(exc))
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
