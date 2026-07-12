#!/usr/bin/env python3
"""Run paired HIP/Vulkan sampler top-k/argmax microbenchmarks."""

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


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
HIP_HARNESS = MICRO_ROOT / "runners" / "hip_sampler_argmax.hip"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_sampler_argmax.cpp"
VULKAN_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "sampler_argmax.comp"
HIP_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_hip.hpp"
VULKAN_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
ISA_STATS = MICRO_ROOT / "runners" / "isa_stats.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
BENCH_NAME = "sampler_argmax_topk"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-sampler-topk")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing_contract = _load_module(TIMING_CONTRACT, "micro_sampler_timing_contract")


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_module(COLLECT_ENV, "micro_collect_env")
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


def _read_text_command(command: list[str], *, cwd: Path) -> str:
    completed = _run_command(command, cwd=cwd, echo=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed.stdout + completed.stderr


def _find_single(paths: list[Path], suffix: str) -> Path:
    matches = [path for path in paths if path.name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"could not find build artifact ending with {suffix}")
    return sorted(matches)[0]


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


def _compile_defines(workgroup: int, top_k: int) -> list[str]:
    return [f"-DHIPENGINE_ARGMAX_WG={workgroup}", f"-DHIPENGINE_ARGMAX_TOPK={top_k}"]


def _compile_hip(workgroup: int, top_k: int, args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    build_dir = args.build_dir / "hip" / f"wg{workgroup}_k{top_k}"
    build_dir.mkdir(parents=True, exist_ok=True)
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    command = [hipcc]
    if args.gfx_arch:
        command.append(f"--offload-arch={args.gfx_arch}")
    command.extend(
        [
            "-O3",
            "-std=c++17",
            "--save-temps",
            *_compile_defines(workgroup, top_k),
            str(HIP_HARNESS),
            "-o",
            "hip_sampler_argmax",
        ]
    )
    completed = _run_command(command, cwd=build_dir)
    if completed.returncode != 0:
        raise RuntimeError(f"HIP sampler argmax build failed for wg{workgroup} top_k{top_k}")
    artifacts = [path for path in build_dir.iterdir() if path.is_file()]
    obj = (
        _find_single(artifacts, f"{args.gfx_arch}.o")
        if args.gfx_arch
        else _find_single(artifacts, ".o")
    )
    return build_dir / "hip_sampler_argmax", obj, command


def _compile_vulkan(workgroup: int, top_k: int, args: argparse.Namespace) -> tuple[Path, Path, list[str], list[str]]:
    build_dir = args.build_dir / "vulkan" / f"wg{workgroup}_k{top_k}"
    build_dir.mkdir(parents=True, exist_ok=True)
    spirv = build_dir / "sampler_argmax.spv"
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    if glslc:
        shader_command = [
            glslc,
            "-O",
            *_compile_defines(workgroup, top_k),
            str(VULKAN_SHADER),
            "-o",
            str(spirv),
        ]
    elif glslang:
        shader_command = [
            glslang,
            "-V",
            *_compile_defines(workgroup, top_k),
            str(VULKAN_SHADER),
            "-o",
            str(spirv),
        ]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(shader_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan sampler argmax shader build failed for wg{workgroup} top_k{top_k}")

    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    exe = build_dir / "vulkan_sampler_argmax"
    build_command = [
        compiler,
        "-O2",
        "-std=c++17",
        *_compile_defines(workgroup, top_k),
        str(VULKAN_HARNESS),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(build_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan sampler argmax harness build failed for wg{workgroup} top_k{top_k}")
    return spirv, exe, shader_command, build_command


def _harness_args(
    args: argparse.Namespace,
    raw_path: Path,
    rows: int,
    *,
    backend: str,
) -> list[str]:
    command = [
        "--json",
        str(raw_path),
        "--rows",
        str(rows),
        "--vocab",
        str(args.vocab),
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
    if backend == "hip":
        command.extend(["--independent-streams", str(args.independent_streams)])
    return command


def _row_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    if not rows:
        raise RuntimeError("raw sampler argmax JSON has no rows")
    return dict(rows[0])


def _hip_isa(obj: Path) -> dict[str, Any]:
    isa = _load_module(ISA_STATS, "micro_isa_stats_for_sampler_argmax")
    notes = _read_text_command([shutil.which("llvm-readobj") or "llvm-readobj", "--notes", str(obj)], cwd=REPO_ROOT)
    disasm = _read_text_command(
        [shutil.which("llvm-objdump") or "llvm-objdump", "-d", "--no-show-raw-insn", str(obj)],
        cwd=REPO_ROOT,
    )
    metadata = isa.parse_hip_metadata(notes)
    stats = isa.parse_disassembly_stats(disasm)
    return {**metadata, **stats, "stats_status": "actual_hip_code_object_metadata_plus_objdump"}


def _vulkan_isa(
    exe: Path,
    spirv: Path,
    args: argparse.Namespace,
    workgroup: int,
    top_k: int,
) -> tuple[dict[str, Any], list[str], int]:
    isa = _load_module(ISA_STATS, "micro_isa_stats_for_sampler_argmax")
    raw_path = args.build_dir / "vulkan" / f"wg{workgroup}_k{top_k}" / "debug_raw.json"
    command = [
        str(exe),
        "--spirv",
        str(spirv),
        "--json",
        str(raw_path),
        "--rows",
        "1",
        "--vocab",
        str(args.debug_vocab),
        "--reps",
        "1",
        "--warmup",
        "0",
        "--samples",
        "1",
        "--device-index",
        str(args.device_index),
    ]
    completed = _run_command(
        command,
        cwd=REPO_ROOT,
        env={"RADV_DEBUG": "shaders,shaderstats"},
        echo=not args.quiet_shader_dump,
    )
    if completed.returncode != 0:
        raise RuntimeError("Vulkan RADV_DEBUG=shaders,shaderstats sampler argmax run failed")
    dump = completed.stdout + completed.stderr
    rows = isa.parse_radv_shader_dump(dump)
    if not rows:
        raise RuntimeError("RADV shader dump did not contain a compute shader")
    row = rows[-1]
    row.update(
        {
            "stats_status": "radv_debug_shaders_shaderstats_final_disassembly",
            "raw_probe_retained": False,
            "shader_dump_retained": False,
        }
    )
    return row, command, len(dump.encode("utf-8"))


def _infer_gfx_arch(environment: dict[str, Any], raw: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    hardware = raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {}
    arch = hardware.get("gcn_arch_name")
    if isinstance(arch, str) and arch:
        return arch
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for key in ("rocminfo_name_gfx_lines", "vulkan_summary_lines", "lspci_display_lines"):
            lines = devices.get(key)
            if isinstance(lines, list):
                import re

                match = re.search(r"\bgfx[0-9a-fA-F]+\b", "\n".join(str(line) for line in lines))
                if match:
                    return match.group(0)
    return "unknown"


def _infer_gpu_name(raw: dict[str, Any], environment: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    hardware = raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {}
    if hardware.get("device_name"):
        return str(hardware["device_name"])
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


def _load_count(isa: dict[str, Any]) -> int:
    return int(isa.get("global_load_count") or 0) + int(isa.get("buffer_load_count") or 0)


def _normalize_result(
    *,
    backend: str,
    raw_rows: list[dict[str, Any]],
    isa_by_variant: dict[tuple[int, int], dict[str, Any]],
    environment: dict[str, Any],
    source_hash: str,
    wrapper_command: list[str],
    commands: list[dict[str, Any]],
    hardware_gpu: str | None,
    gfx_arch: str | None,
    environment_ref: str | None,
) -> dict[str, Any]:
    if not raw_rows or len(raw_rows) != len(commands):
        raise ValueError("sampler result rows and requested commands must align")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        isa = isa_by_variant[(int(raw["workgroup_size"]), int(raw.get("top_k", 1)))]
        config = raw.get("raw_config") if isinstance(raw.get("raw_config"), dict) else {}
        for field in ("rows", "vocab", "workgroup_size", "top_k", "timing_mode"):
            if config.get(field) != raw.get(field):
                raise ValueError(f"sampler raw row disagrees with config field {field}")
        repetitions = int(config.get("reps", 0))
        timing_mode = timing_contract.parse_timing_mode(str(raw.get("timing_mode")))
        gpu_supported = backend == "hip" or bool(raw.get("gpu_timestamps_supported"))
        gpu_clock = "hip_event" if backend == "hip" else "vulkan_timestamp"

        def timing_control(name: str, logical_iterations: int) -> dict[str, Any]:
            gpu_samples = raw.get(f"{name}_gpu_samples_us") if gpu_supported else None
            return timing_contract.make_timing_control(
                logical_iterations=logical_iterations,
                dispatches_per_iteration=1,
                gpu_samples_us=gpu_samples,
                host_samples_us=raw.get(f"{name}_host_samples_us"),
                gpu_clock=gpu_clock,
                gpu_status="ok" if gpu_supported else "unsupported",
            )

        correctness_pass = bool(raw.get("correctness_pass"))
        correctness = timing_contract.make_correctness(
            status="pass" if correctness_pass else "fail",
            oracle="CPU top-k over deterministic logits with stable value/index ordering",
            logical_iterations=repetitions,
            coverage=(
                "all_dispatches"
                if timing_mode == "independent_throughput"
                else "chained_final_state"
            ),
            synchronization_method=(
                "disjoint_output_slices"
                if timing_mode == "independent_throughput"
                else ("hip_stream_order" if backend == "hip" else "vulkan_compute_barrier")
            ),
            barrier_count=(
                repetitions - 1
                if backend == "vulkan" and timing_mode == "serial_latency"
                else 0
            ),
        )
        submission = timing_contract.make_submission(
            strategy=(
                "multi_stream"
                if backend == "hip" and timing_mode == "independent_throughput"
                else (
                    "direct"
                    if backend == "hip"
                    else "vulkan_command_buffer"
                )
            ),
            queue_or_stream_count=int(raw.get("stream_count", 1)) if backend == "hip" else 1,
            recording_in_timed_region=False,
        )
        contract = timing_contract.make_timed_row_contract(
            timing_mode=timing_mode,
            backend=backend,
            repetitions=repetitions,
            dispatches_per_iteration=1,
            dependency_validation_status="pass" if correctness_pass else "fail",
            submission=submission,
            single_timing=timing_control("single", 1),
            burst_timing=timing_control("burst", repetitions),
            correctness=correctness,
        )
        raw_fields = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "raw_config",
                "single_gpu_samples_us",
                "single_host_samples_us",
                "burst_gpu_samples_us",
                "burst_host_samples_us",
            }
        }
        row = {**raw_fields, **contract, **{f"isa_{k}": v for k, v in isa.items()}}
        row["instruction_count"] = isa.get("instruction_count")
        row["waitcnt_count"] = isa.get("waitcnt_count")
        row["waitcnt_depctr_count"] = isa.get("waitcnt_depctr_count")
        row["vopd_count"] = isa.get("vopd_count")
        row["dot4_count"] = isa.get("dot4_count")
        row["load_instruction_count"] = _load_count(isa)
        row["waitcnt_per_load_instruction"] = (
            float(row["waitcnt_count"]) / row["load_instruction_count"]
            if row["load_instruction_count"]
            else None
        )
        row["wave_size"] = isa.get("wave_size") or isa.get("api_subgroup_size")
        for key in (
            "vgpr",
            "sgpr",
            "scratch_bytes",
            "sgpr_spill_count",
            "vgpr_spill_count",
            "subgroups_per_simd",
            "code_size_bytes",
            "register_count_status",
            "shaderstats_status",
        ):
            if isa.get(key) is not None:
                row[key] = isa.get(key)
        burst_gpu = row["timing"]["burst"]["gpu_elapsed"]
        burst_host = row["timing"]["burst"]["host_wall"]
        primary_metric = burst_gpu if burst_gpu["status"] == "ok" else burst_host
        row["median_us"] = primary_metric["per_iteration_us"]["median"]
        row["bandwidth_gbps"] = float(row["bytes_per_dispatch"]) / row["median_us"] / 1000.0
        row["gcomparisons_per_s"] = (
            float(row["comparisons_per_dispatch"]) / row["median_us"] / 1000.0
        )
        rows.append(row)

    rows_list = sorted({int(row["rows"]) for row in rows})
    workgroups = sorted({int(row["workgroup_size"]) for row in rows})
    top_k_list = sorted({int(row.get("top_k", 1)) for row in rows})
    vocabs = {int(row["vocab"]) for row in rows}
    timing_modes = {str(row["timing_mode"]) for row in rows}
    repetitions = {int(raw["raw_config"]["reps"]) for raw in raw_rows}
    warmups = {int(raw["raw_config"]["warmup"]) for raw in raw_rows}
    sample_counts = {int(raw["raw_config"]["samples"]) for raw in raw_rows}
    if any(len(values) != 1 for values in (vocabs, timing_modes, repetitions, warmups, sample_counts)):
        raise ValueError("sampler requested rows do not share one workload contract")
    vocab = next(iter(vocabs))
    timing_mode = next(iter(timing_modes))
    expected_keys = {
        (timing_mode, top_k, row_count, workgroup, vocab)
        for top_k in top_k_list
        for row_count in rows_list
        for workgroup in workgroups
    }
    row_keys = [_row_key(row) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("sampler result contains duplicate requested rows")
    if set(row_keys) != expected_keys:
        raise ValueError(
            "sampler result does not contain the complete requested matrix: "
            f"expected {len(expected_keys)}, got {len(row_keys)}"
        )
    primary = min(rows, key=lambda row: float(row["median_us"])) if rows else {}
    correctness_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    raw0 = raw_rows[0] if raw_rows else {}
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": backend,
        "hardware": {
            "gpu_name": _infer_gpu_name(raw0, environment, hardware_gpu),
            "gfx_arch": _infer_gfx_arch(environment, raw0, gfx_arch),
        },
        "source": _source_record(environment, source_hash),
        "command": wrapper_command,
        "cwd": str(REPO_ROOT),
        "parameters": {
            "benchmark_family": "sampler_argmax",
            "rows_list": rows_list,
            "workgroups": workgroups,
            "top_k_list": top_k_list,
            "vocab": vocab,
            "timing_mode": timing_mode,
            "repetitions": next(iter(repetitions)),
            "warmup_logical_iterations": next(iter(warmups)),
            "samples": next(iter(sample_counts)),
            "expected_row_count": len(expected_keys),
            "commands": commands,
        },
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "CPU top-k over deterministic logits with stable value/index ordering",
            "max_abs": max((float(row.get("max_abs", 0.0)) for row in rows), default=0.0),
            "mismatches": sum((int(row.get("mismatches", 0)) for row in rows), 0),
        },
        "isa": primary,
        "classification": "diagnostic_unclassified",
        "measurements": {"rows": rows},
        "notes": (
            "Sampler/top-k argmax diagnostic with explicit serial-latency or independent-throughput semantics. "
            "One workgroup reduces one logits row; "
            "the row tests reduction, memory scan, tie-break, LDS/shared-memory, and scheduler behavior. "
            "This is deterministic top-k, not stochastic sampling."
        ),
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return _json_safe(result)


def _run_hip(
    args: argparse.Namespace,
    rows_list: list[int],
    workgroups: list[int],
    top_k_list: list[int],
) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [Path(__file__).resolve(), HIP_HARNESS, HIP_TIMING_HEADER, TIMING_CONTRACT]
    )
    raw_rows: list[dict[str, Any]] = []
    isa_by_variant: dict[tuple[int, int], dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    for top_k in top_k_list:
        for workgroup in workgroups:
            exe, obj, build_command = _compile_hip(workgroup, top_k, args)
            isa_by_variant[(workgroup, top_k)] = _hip_isa(obj)
            for rows in rows_list:
                raw_path = args.build_dir / "hip" / f"wg{workgroup}_k{top_k}" / f"rows{rows}.json"
                harness_command = [
                    str(exe),
                    *_harness_args(args, raw_path, rows, backend="hip"),
                ]
                completed = _run_command(harness_command, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"HIP sampler argmax run failed for wg{workgroup} top_k{top_k} rows{rows}"
                    )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_row = _row_from_raw(raw)
                raw_row["hardware"] = raw.get("hardware", {})
                raw_row["raw_config"] = raw.get("config", {})
                raw_rows.append(raw_row)
                commands.append(
                    {
                        "workgroup_size": workgroup,
                        "top_k": top_k,
                        "rows": rows,
                        "build_command": build_command,
                        "harness_command": harness_command,
                        "object_path": str(obj),
                        "raw_json_retained": False,
                    }
                )
    return _normalize_result(
        backend="hip",
        raw_rows=raw_rows,
        isa_by_variant=isa_by_variant,
        environment=environment,
        source_hash=source_hash,
        wrapper_command=sys.argv.copy(),
        commands=commands,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
    )


def _run_vulkan(
    args: argparse.Namespace,
    rows_list: list[int],
    workgroups: list[int],
    top_k_list: list[int],
) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [
            Path(__file__).resolve(),
            VULKAN_HARNESS,
            VULKAN_SHADER,
            VULKAN_TIMING_HEADER,
            TIMING_CONTRACT,
        ]
    )
    raw_rows: list[dict[str, Any]] = []
    isa_by_variant: dict[tuple[int, int], dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    for top_k in top_k_list:
        for workgroup in workgroups:
            spirv, exe, shader_command, build_command = _compile_vulkan(workgroup, top_k, args)
            isa_row, debug_command, shader_dump_bytes = _vulkan_isa(exe, spirv, args, workgroup, top_k)
            isa_by_variant[(workgroup, top_k)] = isa_row
            for rows in rows_list:
                raw_path = args.build_dir / "vulkan" / f"wg{workgroup}_k{top_k}" / f"rows{rows}.json"
                harness_command = [
                    str(exe),
                    "--spirv",
                    str(spirv),
                    *_harness_args(args, raw_path, rows, backend="vulkan"),
                ]
                completed = _run_command(harness_command, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Vulkan sampler argmax run failed for wg{workgroup} top_k{top_k} rows{rows}"
                    )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_row = _row_from_raw(raw)
                raw_row["hardware"] = raw.get("hardware", {})
                raw_row["raw_config"] = raw.get("config", {})
                raw_rows.append(raw_row)
                commands.append(
                    {
                        "workgroup_size": workgroup,
                        "top_k": top_k,
                        "rows": rows,
                        "shader_command": shader_command,
                        "build_command": build_command,
                        "harness_command": harness_command,
                        "debug_command": debug_command,
                        "debug_env": {"RADV_DEBUG": "shaders,shaderstats"},
                        "shader_dump_bytes": shader_dump_bytes,
                        "raw_json_retained": False,
                        "shader_dump_retained": False,
                    }
                )
    return _normalize_result(
        backend="vulkan",
        raw_rows=raw_rows,
        isa_by_variant=isa_by_variant,
        environment=environment,
        source_hash=source_hash,
        wrapper_command=sys.argv.copy(),
        commands=commands,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row.get("timing_mode")),
        int(row.get("top_k", 1)),
        int(row["rows"]),
        int(row["workgroup_size"]),
        int(row["vocab"]),
    )


def _device_fingerprint(name: Any) -> str:
    text = str(name).lower()
    match = re.search(r"radeon\s+(\d+s)", text)
    if match:
        return f"radeon_{match.group(1)}"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _validate_comparison_inputs(
    hip_result: dict[str, Any], vulkan_result: dict[str, Any]
) -> None:
    for label, result, backend in (
        ("HIP", hip_result, "hip"),
        ("Vulkan", vulkan_result, "vulkan"),
    ):
        if result.get("schema_version") != 2:
            raise ValueError("sampler comparison requires v2 timing-contract results")
        if result.get("kind") != "hipengine_micro_result":
            raise ValueError(f"{label} sampler input must be a micro result")
        if result.get("bench") != BENCH_NAME:
            raise ValueError(f"{label} sampler input bench does not match {BENCH_NAME}")
        if result.get("backend") != backend:
            raise ValueError("sampler comparison inputs must be HIP then Vulkan")
        if result.get("classification") != "diagnostic_unclassified":
            raise ValueError(f"{label} sampler input classification is invalid")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan sampler gfx architectures do not match")
    hip_device = _device_fingerprint(hip_result.get("hardware", {}).get("gpu_name", ""))
    vulkan_device = _device_fingerprint(
        vulkan_result.get("hardware", {}).get("gpu_name", "")
    )
    if not hip_device or hip_device == "unknown" or hip_device != vulkan_device:
        raise ValueError("HIP and Vulkan sampler device identities do not match")
    hip_source = hip_result.get("source", {})
    vulkan_source = vulkan_result.get("source", {})
    if not isinstance(hip_source, dict) or not isinstance(vulkan_source, dict):
        raise ValueError("HIP and Vulkan sampler source provenance must be objects")
    for field in ("repo", "branch", "commit", "dirty"):
        if field not in hip_source or field not in vulkan_source:
            raise ValueError(f"HIP and Vulkan sampler source {field} is required")
        if hip_source[field] != vulkan_source[field]:
            raise ValueError(f"HIP and Vulkan sampler source {field} values do not match")
    if not str(hip_source.get("repo", "")) or not str(hip_source.get("commit", "")):
        raise ValueError("HIP and Vulkan sampler source repo/commit must not be empty")
    if not isinstance(hip_source.get("dirty"), bool):
        raise ValueError("HIP and Vulkan sampler source dirty must be boolean")
    for label, source in (("HIP", hip_source), ("Vulkan", vulkan_source)):
        if not str(source.get("source_hash", "")):
            raise ValueError(f"{label} sampler source hash must not be empty")


def _expected_row_keys(parameters: dict[str, Any]) -> set[tuple[str, int, int, int, int]]:
    rows_list = parameters.get("rows_list")
    workgroups = parameters.get("workgroups")
    top_k_list = parameters.get("top_k_list")
    if (
        not isinstance(rows_list, list)
        or not rows_list
        or not isinstance(workgroups, list)
        or not workgroups
        or not isinstance(top_k_list, list)
        or not top_k_list
    ):
        raise ValueError("sampler parameters do not describe the requested matrix")
    try:
        row_values = [int(rows) for rows in rows_list]
        workgroup_values = [int(workgroup) for workgroup in workgroups]
        top_k_values = [int(top_k) for top_k in top_k_list]
        vocab = int(parameters["vocab"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sampler parameters contain an invalid requested matrix") from exc
    if any(
        len(set(values)) != len(values)
        for values in (row_values, workgroup_values, top_k_values)
    ):
        raise ValueError("sampler parameters contain duplicate requested rows")
    return {
        (
            str(parameters["timing_mode"]),
            int(top_k),
            int(rows),
            int(workgroup),
            vocab,
        )
        for top_k in top_k_values
        for rows in row_values
        for workgroup in workgroup_values
    }


def _domain_comparison(
    hip: dict[str, Any],
    vulkan: dict[str, Any],
    *,
    control: str,
    domain: str,
) -> dict[str, Any]:
    hip_status = hip["timing"][control][domain]["status"]
    vulkan_status = vulkan["timing"][control][domain]["status"]
    if hip_status != "ok" or vulkan_status != "ok":
        return {
            "status": "unsupported",
            "hip_status": hip_status,
            "vulkan_status": vulkan_status,
        }
    try:
        ratio = timing_contract.comparison_ratio(
            hip,
            vulkan,
            control=control,
            domain=domain,
        )
    except ValueError as exc:
        if domain != "host_wall":
            raise
        return {
            "status": "not_comparable_submission_contract",
            "reason": str(exc),
        }
    return {"status": "ok", **ratio}


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    hip_ref: str | None = None,
    vulkan_ref: str | None = None,
    out_ref: str | None = None,
) -> dict[str, Any]:
    _validate_comparison_inputs(hip_result, vulkan_result)
    hip_parameters = hip_result.get("parameters", {})
    vulkan_parameters = vulkan_result.get("parameters", {})
    for field in (
        "rows_list",
        "workgroups",
        "top_k_list",
        "vocab",
        "timing_mode",
        "repetitions",
        "warmup_logical_iterations",
        "samples",
        "expected_row_count",
    ):
        if field not in hip_parameters or field not in vulkan_parameters:
            raise ValueError(f"HIP and Vulkan sampler parameter {field} is required")
        if hip_parameters[field] != vulkan_parameters[field]:
            if field == "timing_mode":
                raise ValueError("HIP and Vulkan sampler timing modes do not match (timing_mode)")
            raise ValueError(f"HIP and Vulkan sampler parameter {field} values do not match")
    hip_input_rows = hip_result.get("measurements", {}).get("rows", [])
    vulkan_input_rows = vulkan_result.get("measurements", {}).get("rows", [])
    if (
        not isinstance(hip_input_rows, list)
        or not isinstance(vulkan_input_rows, list)
        or not hip_input_rows
        or not vulkan_input_rows
        or not all(isinstance(row, dict) for row in hip_input_rows)
        or not all(isinstance(row, dict) for row in vulkan_input_rows)
    ):
        raise ValueError("sampler comparison requires non-empty object rows")
    hip_rows = {_row_key(row): row for row in hip_input_rows}
    vulkan_rows = {_row_key(row): row for row in vulkan_input_rows}
    if len(hip_rows) != len(hip_input_rows) or len(vulkan_rows) != len(vulkan_input_rows):
        raise ValueError("sampler comparison inputs contain duplicate rows")
    hip_modes = {key[0] for key in hip_rows}
    vulkan_modes = {key[0] for key in vulkan_rows}
    if hip_modes != vulkan_modes:
        raise ValueError("HIP and Vulkan timing modes do not match")
    expected_rows = _expected_row_keys(hip_parameters)
    if int(hip_parameters["expected_row_count"]) != len(expected_rows):
        raise ValueError("sampler expected row count does not match its matrix")
    if set(hip_rows) != expected_rows or set(vulkan_rows) != expected_rows:
        raise ValueError(
            "HIP and Vulkan sampler results must contain the exact requested "
            f"{len(expected_rows)}-row matrix"
        )
    comparisons: list[dict[str, Any]] = []
    for key in sorted(hip_rows):
        hip = hip_rows[key]
        vulkan = vulkan_rows[key]
        timing_contract.validate_timed_row(
            hip, expected_repetitions=int(hip_parameters["repetitions"])
        )
        timing_contract.validate_timed_row(
            vulkan, expected_repetitions=int(hip_parameters["repetitions"])
        )
        for control in timing_contract.TIMING_CONTROLS:
            comparisons.append(
                {
                    "timing_mode": key[0],
                    "control": control,
                    "top_k": key[1],
                    "rows": key[2],
                    "workgroup_size": key[3],
                    "vocab": key[4],
                    "gpu_elapsed": _domain_comparison(
                        hip, vulkan, control=control, domain="gpu_elapsed"
                    ),
                    "host_wall": _domain_comparison(
                        hip, vulkan, control=control, domain="host_wall"
                    ),
                    "isa": {
                        "hip_instruction_count": hip.get("instruction_count"),
                        "vulkan_instruction_count": vulkan.get("instruction_count"),
                        "hip_waitcnt_count": hip.get("waitcnt_count"),
                        "vulkan_waitcnt_count": vulkan.get("waitcnt_count"),
                        "hip_wave_size": hip.get("wave_size"),
                        "vulkan_wave_size": vulkan.get("wave_size"),
                    },
                }
            )
    burst_gpu = [
        row
        for row in comparisons
        if row["control"] == "burst" and row["gpu_elapsed"].get("status") == "ok"
    ]
    best = max(
        burst_gpu,
        key=lambda row: float(row["gpu_elapsed"]["vulkan_vs_hip_speedup"]),
        default={},
    )
    dirty = bool(hip_result["source"]["dirty"])
    correctness_pass = (
        hip_result.get("correctness", {}).get("status") == "pass"
        and vulkan_result.get("correctness", {}).get("status") == "pass"
        and all(bool(row.get("correctness_pass")) for row in hip_input_rows)
        and all(bool(row.get("correctness_pass")) for row in vulkan_input_rows)
    )
    performance_claim = not dirty and correctness_pass
    blocking_reasons = []
    if dirty:
        blocking_reasons.append("dirty_source")
    if not correctness_pass:
        blocking_reasons.append("correctness_not_passed")
    return _json_safe(
        {
            "schema_version": 2,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "diagnostic_unclassified",
            "performance_claim": performance_claim,
            "source": hip_result.get("source", {}),
            "sources": {
                "hip": hip_result.get("source", {}),
                "vulkan": vulkan_result.get("source", {}),
            },
            "command": command,
            "hardware": {
                "hip": hip_result.get("hardware", {}),
                "vulkan": vulkan_result.get("hardware", {}),
            },
            "inputs": {"hip_result": hip_ref, "vulkan_result": vulkan_ref, "out": out_ref},
            "correctness": {
                "hip": hip_result.get("correctness", {}),
                "vulkan": vulkan_result.get("correctness", {}),
            },
            "provenance": {
                "commit_match": True,
                "dirty": dirty,
                "gfx_arch_match": True,
                "device_match": True,
                "source_hashes_present": True,
                "performance_claim": performance_claim,
                "blocking_reasons": blocking_reasons,
            },
            "comparisons": comparisons,
            "summary": {
                "best_vulkan_vs_hip_gpu_burst_speedup": (
                    best.get("gpu_elapsed", {}).get("vulkan_vs_hip_speedup")
                    if best
                    else None
                ),
                "best_row": best,
                "all_correct": bool(comparisons)
                and hip_result.get("correctness", {}).get("status") == "pass"
                and vulkan_result.get("correctness", {}).get("status") == "pass",
            },
            "interpretation": (
                "Sampler/top-k argmax diagnostic. This covers the sampler/top-k/argmax "
                "matrix bucket for reduction, scan, LDS/shared-memory, register, VOPD, "
                "and waitcnt evidence for deterministic top-k. It is not a full stochastic sampler or production "
                "lm-head fusion result."
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
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--workgroups", default="64,128,256")
    parser.add_argument("--top-k-list", default="1")
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--timing-mode",
        choices=timing_contract.TIMING_MODES,
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--debug-vocab", type=int, default=1024)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=8.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--quiet-shader-dump", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.compare and args.backend:
        parser.error("--compare and --backend are mutually exclusive")
    if not args.compare and not args.backend:
        parser.error("one of --backend or --compare is required")
    if (
        args.vocab <= 0
        or args.reps <= 0
        or args.samples <= 0
        or args.debug_vocab <= 0
        or args.independent_streams <= 0
    ):
        parser.error(
            "--vocab, --reps, --samples, --debug-vocab, and --independent-streams must be positive"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        hip_path, vulkan_path = args.compare
        hip_result = json.loads(hip_path.read_text(encoding="utf-8"))
        vulkan_result = json.loads(vulkan_path.read_text(encoding="utf-8"))
        result = build_comparison(
            hip_result,
            vulkan_result,
            command=sys.argv.copy(),
            hip_ref=str(hip_path),
            vulkan_ref=str(vulkan_path),
            out_ref=str(args.out) if args.out else None,
        )
    else:
        rows_list = _parse_csv_u32(args.rows_list)
        workgroups = _parse_csv_u32(args.workgroups)
        top_k_list = _parse_csv_u32(args.top_k_list)
        if any(top_k > args.vocab for top_k in top_k_list):
            raise ValueError(f"top-k must be <= vocab: {top_k_list} > {args.vocab}")
        for workgroup in workgroups:
            if workgroup & (workgroup - 1):
                raise ValueError(f"workgroup must be a power of two: {workgroup}")
            if any(top_k > workgroup for top_k in top_k_list):
                raise ValueError(f"top-k must be <= workgroup size for wg{workgroup}: {top_k_list}")
        if args.backend == "hip":
            result = _run_hip(args, rows_list, workgroups, top_k_list)
        else:
            result = _run_vulkan(args, rows_list, workgroups, top_k_list)
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
