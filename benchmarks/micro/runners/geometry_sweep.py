#!/usr/bin/env python3
"""Run or compare matched HIP/Vulkan f32 GEMV geometry microbenchmarks."""

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
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HIP_HARNESS = REPO_ROOT / "benchmarks" / "micro" / "runners" / "hip_geometry_sweep.hip"
VULKAN_HARNESS = REPO_ROOT / "benchmarks" / "micro" / "runners" / "vulkan_geometry_sweep.cpp"
VULKAN_SHADER = REPO_ROOT / "benchmarks" / "micro" / "kernels" / "vulkan" / "geometry_sweep.comp"
HIP_TIMING_HEADER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "micro_timing_hip.hpp"
VULKAN_TIMING_HEADER = REPO_ROOT / "benchmarks" / "micro" / "runners" / "micro_timing_vulkan.hpp"
COLLECT_ENV = Path(__file__).resolve().parents[1] / "collect_env.py"
TIMING_CONTRACT = Path(__file__).resolve().parents[1] / "timing_contract.py"
BENCH_NAME = "f32_gemv_geometry_sweep"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-geometry-build")


def _load_collect_env_module():
    spec = importlib.util.spec_from_file_location("micro_collect_env", COLLECT_ENV)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load environment collector: {COLLECT_ENV}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_timing_contract_module():
    spec = importlib.util.spec_from_file_location("micro_timing_contract", TIMING_CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load timing contract: {TIMING_CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed


def _compile_hip_harness(
    build_dir: Path,
    gfx_arch: str | None,
    *,
    fixed_workgroup_size: int | None = None,
    hip_wavefront_size: str = "default",
) -> tuple[Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / "hip_geometry_sweep"
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    command = [hipcc]
    arch = gfx_arch or os.environ.get("HIPENGINE_HIP_ARCH")
    if arch:
        command.append(f"--offload-arch={arch}")
    if hip_wavefront_size == "64":
        command.append("-mwavefrontsize64")
    elif hip_wavefront_size == "32":
        command.append("-mno-wavefrontsize64")
    command.extend(
        [
            "-O3",
            "-std=c++17",
            *(
                [f"-DHIPENGINE_FIXED_WORKGROUP_SIZE={fixed_workgroup_size}"]
                if fixed_workgroup_size
                else []
            ),
            str(HIP_HARNESS),
            "-o",
            str(exe_path),
        ]
    )
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("HIP geometry harness build failed")
    return exe_path, command


def _compile_vulkan_shader(build_dir: Path) -> tuple[Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    spirv_path = build_dir / "geometry_sweep.spv"
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    if glslc:
        command = [glslc, "-O", str(VULKAN_SHADER), "-o", str(spirv_path)]
    elif glslang:
        command = [glslang, "-V", str(VULKAN_SHADER), "-o", str(spirv_path)]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan geometry shader compilation failed")
    return spirv_path, command


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


def _compile_vulkan_harness(build_dir: Path) -> tuple[Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / "vulkan_geometry_sweep"
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(VULKAN_HARNESS),
        "-o",
        str(exe_path),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan geometry harness build failed")
    return exe_path, command


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


def _find_gfx_arch(text: str) -> str | None:
    match = re.search(r"\bgfx[0-9a-fA-F]+\b", text)
    return match.group(0) if match else None


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
                found = _find_gfx_arch("\n".join(str(line) for line in lines))
                if found:
                    return found
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


def _reference_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    for row in rows:
        if row.get("k") == 2048 and row.get("rows") == 1 and row.get("workgroup_size") == 64:
            return row
    return rows[0]


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    best = min(rows, key=lambda row: float(row.get("median_us", float("inf"))))
    worst = max(rows, key=lambda row: float(row.get("median_us", 0.0)))
    return {
        "best": best,
        "worst": worst,
        "best_worst_ratio": (
            float(worst.get("median_us", 0.0)) / float(best.get("median_us", 1.0))
            if float(best.get("median_us", 0.0)) > 0.0
            else None
        ),
    }


def normalize_raw_result(
    raw: dict[str, Any],
    *,
    backend: str,
    environment: dict[str, Any],
    wrapper_command: list[str],
    harness_command: list[str] | None,
    build_command: list[str] | None,
    shader_command: list[str] | None,
    source_hash: str,
    hardware_gpu: str | None = None,
    gfx_arch: str | None = None,
    environment_ref: str | None = None,
) -> dict[str, Any]:
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    timing_contract = _load_timing_contract_module()
    timing_mode = timing_contract.parse_timing_mode(str(config.get("timing_mode")))
    repetitions = int(config.get("reps", 0))
    if repetitions <= 0:
        raise ValueError("raw geometry result has invalid repetitions")
    if backend == "hip" and config.get("hip_workgroup_specialization") != "fixed":
        raise ValueError("v2 HIP geometry rows require fixed workgroup specialization")
    for row in rows:
        if row.get("timing_mode") != timing_mode:
            raise ValueError("raw geometry row timing mode disagrees with config")
        timing_contract.validate_timed_row(row, expected_repetitions=repetitions)
        row["workgroup_specialization"] = (
            "fixed" if backend == "hip" else "specialization_constant"
        )
    reference = _reference_row(rows)
    max_abs = max((float(row.get("max_abs", 0.0)) for row in rows), default=0.0)
    max_rel = max((float(row.get("max_rel", 0.0)) for row in rows), default=0.0)
    correctness_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    hardware = raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {}
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": backend,
        "hardware": {
            "gpu_name": _infer_gpu_name(raw, environment, hardware_gpu),
            "gfx_arch": _infer_gfx_arch(environment, raw, gfx_arch),
        },
        "source": _source_record(environment, source_hash),
        "command": wrapper_command,
        "cwd": str(REPO_ROOT),
        "parameters": _json_safe(
            {
                "benchmark_family": "geometry_sweep",
                "algorithm": "repeat_shifted_f32_gemv_row_shared_tree_reduce",
                "hip_workgroup_specialization": config.get("hip_workgroup_specialization")
                if backend == "hip"
                else None,
                "hip_wavefront_size_request": config.get("hip_wavefront_size_request")
                if backend == "hip"
                else None,
                "hip_fixed_workgroup_sizes": config.get("hip_fixed_workgroup_sizes")
                if backend == "hip"
                else None,
                "raw_config": config,
                "timing_mode": timing_mode,
                "independent_streams": config.get("independent_streams")
                if backend == "hip"
                else None,
                "harness_command": harness_command,
                "build_command": build_command,
                "shader_command": shader_command,
                "hardware": hardware,
                "timing_method": "v2 single+burst GPU elapsed and host wall",
            }
        ),
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "CPU f32 reference using the same per-workgroup reduction order",
            "max_abs": max_abs,
            "max_rel": max_rel,
        },
        "timing": _json_safe(
            {
                "unit": "us_per_dispatch",
                "primary_domain": "gpu_elapsed",
                "median": reference.get("median_us"),
                "p05": reference.get("p05_us"),
                "p95": reference.get("p95_us"),
                "warmup_iters": config.get("warmup"),
                "measured_iters": config.get("reps"),
                "samples": config.get("samples"),
                "primary": reference,
                "summary": _summarize_rows(rows),
            }
        ),
        "isa": _json_safe(
            {
                "workgroup_size": reference.get("workgroup_size"),
                "lds_bytes": (
                    int(reference["workgroup_size"]) * 4
                    if isinstance(reference.get("workgroup_size"), int)
                    else None
                ),
                "stats_status": "not_collected",
                "stats_note": (
                    "This family isolates workgroup/reduction geometry first. "
                    "Register, scratch, waitcnt, and VOPD extraction are handled "
                    "by the later ISA-stat microbench families."
                ),
            }
        ),
        "classification": "geometry",
        "measurements": {
            "rows": rows,
        },
        "notes": (
            "Matched repeat-shifted f32 GEMV/reduction geometry diagnostic. Uses "
            "the same data, K/rows/workgroup/body-repeat shape, and CPU oracle on both backends. "
            "Do not use this row alone as compiler_aco evidence until ISA stats "
            "are collected."
        ),
    }
    if hardware.get("device_id") is not None:
        try:
            result["hardware"]["device_id"] = f"0x{int(hardware['device_id']):x}"
        except (TypeError, ValueError):
            result["hardware"]["device_id"] = str(hardware["device_id"])
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return result


def _shape_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["k"]),
        int(row["rows"]),
        int(row["workgroup_size"]),
        int(row["body_repeats"]),
        str(row["timing_mode"]),
    )


def _metric_median(row: dict[str, Any], control: str, domain: str) -> float | None:
    metric = row.get("timing", {}).get(control, {}).get(domain, {})
    if metric.get("status") != "ok":
        return None
    value = metric.get("per_iteration_us", {}).get("median")
    return float(value) if value is not None else None


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    if hip_result.get("schema_version") != 2 or vulkan_result.get("schema_version") != 2:
        raise ValueError("geometry comparisons require v2 result artifacts")
    hip_rows = hip_result.get("measurements", {}).get("rows", [])
    vulkan_rows = vulkan_result.get("measurements", {}).get("rows", [])
    if any(row.get("workgroup_specialization") != "fixed" for row in hip_rows):
        raise ValueError("HIP geometry comparison rows must use fixed workgroups")
    hip_modes = {row.get("timing_mode") for row in hip_rows}
    vulkan_modes = {row.get("timing_mode") for row in vulkan_rows}
    if len(hip_modes) != 1 or hip_modes != vulkan_modes:
        raise ValueError("HIP and Vulkan geometry timing modes do not match")
    timing_contract = _load_timing_contract_module()
    hip_by_key = {_shape_key(row): row for row in hip_rows}
    vulkan_by_key = {_shape_key(row): row for row in vulkan_rows}
    matched = []
    comparisons = []
    for key in sorted(set(hip_by_key) & set(vulkan_by_key)):
        hip_row = hip_by_key[key]
        vulkan_row = vulkan_by_key[key]
        timing_contract.validate_timed_row(hip_row)
        timing_contract.validate_timed_row(vulkan_row)
        hip_us = _metric_median(hip_row, "burst", "gpu_elapsed")
        vulkan_us = _metric_median(vulkan_row, "burst", "gpu_elapsed")
        matched.append(
            {
                "k": key[0],
                "rows": key[1],
                "workgroup_size": key[2],
                "body_repeats": key[3],
                "timing_mode": key[4],
                "hip_gpu_burst_median_us": hip_us,
                "vulkan_gpu_burst_median_us": vulkan_us,
                "vulkan_vs_hip_gpu_burst_speedup": (
                    hip_us / vulkan_us
                    if hip_us is not None and vulkan_us is not None and vulkan_us > 0
                    else None
                ),
                "hip_gflops": hip_row.get("gflops"),
                "vulkan_gflops": vulkan_row.get("gflops"),
                "hip_correctness_pass": hip_row.get("correctness_pass"),
                "vulkan_correctness_pass": vulkan_row.get("correctness_pass"),
            }
        )
        for control in ("single", "burst"):
            domain_results: dict[str, Any] = {}
            for domain in ("gpu_elapsed", "host_wall"):
                hip_metric = hip_row["timing"][control][domain]
                vulkan_metric = vulkan_row["timing"][control][domain]
                if hip_metric.get("status") != "ok" or vulkan_metric.get("status") != "ok":
                    domain_results[domain] = {
                        "status": "unavailable",
                        "hip_status": hip_metric.get("status"),
                        "vulkan_status": vulkan_metric.get("status"),
                    }
                else:
                    try:
                        ratio = timing_contract.comparison_ratio(
                            hip_row,
                            vulkan_row,
                            control=control,
                            domain=domain,
                        )
                    except ValueError as exc:
                        domain_results[domain] = {
                            "status": "not_comparable_submission_contract",
                            "reason": str(exc),
                            "hip_metric": hip_metric,
                            "vulkan_metric": vulkan_metric,
                        }
                        continue
                    domain_results[domain] = {
                        "status": "ok",
                        **ratio,
                    }
            comparisons.append(
                {
                    "k": key[0],
                    "rows": key[1],
                    "workgroup_size": key[2],
                    "body_repeats": key[3],
                    "timing_mode": key[4],
                    "control": control,
                    **domain_results,
                }
            )

    by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in matched:
        by_shape.setdefault((int(row["k"]), int(row["rows"])), []).append(row)
    shape_summary = []
    for (k, rows), shape_rows in sorted(by_shape.items()):
        comparable = [
            row
            for row in shape_rows
            if row["hip_gpu_burst_median_us"] is not None
            and row["vulkan_gpu_burst_median_us"] is not None
        ]
        if not comparable:
            continue
        best_hip = min(comparable, key=lambda row: float(row["hip_gpu_burst_median_us"]))
        best_vulkan = min(
            comparable, key=lambda row: float(row["vulkan_gpu_burst_median_us"])
        )
        best_speedup = max(
            comparable,
            key=lambda row: float(row["vulkan_vs_hip_gpu_burst_speedup"] or 0.0),
        )
        shape_summary.append(
            {
                "k": k,
                "rows": rows,
                "best_hip_workgroup": best_hip["workgroup_size"],
                "best_hip_gpu_burst_median_us": best_hip["hip_gpu_burst_median_us"],
                "best_vulkan_workgroup": best_vulkan["workgroup_size"],
                "best_vulkan_gpu_burst_median_us": best_vulkan[
                    "vulkan_gpu_burst_median_us"
                ],
                "best_native_vulkan_vs_hip_gpu_burst_speedup": (
                    best_hip["hip_gpu_burst_median_us"]
                    / best_vulkan["vulkan_gpu_burst_median_us"]
                    if best_vulkan["vulkan_gpu_burst_median_us"] > 0
                    else None
                ),
                "largest_matched_vulkan_vs_hip_gpu_burst_speedup": best_speedup[
                    "vulkan_vs_hip_gpu_burst_speedup"
                ],
                "largest_matched_speedup_workgroup": best_speedup["workgroup_size"],
            }
        )

    source = hip_result.get("source", {})
    comparison = {
        "schema_version": 2,
        "kind": "hipengine_micro_comparison",
        "bench": BENCH_NAME,
        "classification": "diagnostic_unclassified",
        "source": source,
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
        "shape_summary": shape_summary,
        "interpretation": (
            "Matched f32 GEMV/reduction workgroup sweep. This artifact can "
            "test whether workgroup geometry explains a HIP/Vulkan gap. If "
            "Vulkan still wins at identical shape, the result remains "
            "diagnostic_unclassified until paired ISA/stat extraction shows "
            "register, scratch, waitcnt, VOPD, or instruction-count differences."
        ),
    }
    return _json_safe(comparison)


def _list_arg(values: str) -> list[str]:
    return [item for item in values.split(",") if item]


def _int_list_arg(values: str) -> list[int]:
    return [int(item) for item in _list_arg(values)]


def _merge_fixed_hip_raw_results(
    raw_results: list[dict[str, Any]],
    *,
    workgroups: list[int],
) -> dict[str, Any]:
    if not raw_results:
        raise RuntimeError("no fixed-workgroup HIP raw results to merge")
    merged = dict(raw_results[0])
    config = dict(merged.get("config") if isinstance(merged.get("config"), dict) else {})
    config["workgroups"] = workgroups
    config["hip_workgroup_specialization"] = "fixed"
    config["hip_fixed_workgroup_sizes"] = workgroups
    rows: list[dict[str, Any]] = []
    for raw in raw_results:
        raw_rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
        rows.extend(row for row in raw_rows if isinstance(row, dict))
    merged["config"] = config
    merged["rows"] = rows
    return merged


def _run_backend(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.build_dir / args.backend
    environment = _collect_environment(args)
    wrapper_command = sys.argv.copy()
    raw_path: Path
    temp_raw: tempfile.NamedTemporaryFile[str] | None = None
    if args.raw_json:
        raw_path = args.raw_json
    else:
        temp_raw = tempfile.NamedTemporaryFile(prefix="hipengine-geometry-", suffix=".json", delete=False)
        temp_raw.close()
        raw_path = Path(temp_raw.name)

    if args.backend == "hip":
        if args.hip_workgroup_specialization == "fixed":
            raw_results: list[dict[str, Any]] = []
            build_commands: list[list[str]] = []
            harness_commands: list[list[str]] = []
            workgroups = _int_list_arg(args.workgroups)
            for workgroup in workgroups:
                fixed_dir = build_dir / f"fixed_wg{workgroup}"
                exe, build_command_one = _compile_hip_harness(
                    fixed_dir,
                    args.gfx_arch,
                    fixed_workgroup_size=workgroup,
                    hip_wavefront_size=args.hip_wavefront_size,
                )
                raw_one = fixed_dir / "raw.json"
                harness_command_one = [
                    str(exe),
                    "--json",
                    str(raw_one),
                    "--k-list",
                    args.k_list,
                    "--rows-list",
                    args.rows_list,
                    "--workgroups",
                    str(workgroup),
                    "--body-repeats",
                    str(args.body_repeats),
                    "--reps",
                    str(args.reps),
                    "--warmup",
                    str(args.warmup),
                    "--samples",
                    str(args.samples),
                    "--timing-mode",
                    args.timing_mode,
                    "--independent-streams",
                    str(args.independent_streams),
                    "--device-index",
                    str(args.device_index),
                ]
                completed = _run_command(harness_command_one, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    raise RuntimeError(f"HIP fixed-workgroup geometry run failed for wg={workgroup}")
                raw_results.append(json.loads(raw_one.read_text(encoding="utf-8")))
                build_commands.append(build_command_one)
                harness_commands.append(harness_command_one)
            raw = _merge_fixed_hip_raw_results(raw_results, workgroups=workgroups)
            raw.setdefault("config", {})["hip_wavefront_size_request"] = args.hip_wavefront_size
            raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            build_command = build_commands
            harness_command = harness_commands
        else:
            exe, build_command = _compile_hip_harness(
                build_dir,
                args.gfx_arch,
                hip_wavefront_size=args.hip_wavefront_size,
            )
            harness_command = [
                str(exe),
                "--json",
                str(raw_path),
                "--k-list",
                args.k_list,
                "--rows-list",
                args.rows_list,
                "--workgroups",
                args.workgroups,
                "--body-repeats",
                str(args.body_repeats),
                "--reps",
                str(args.reps),
                "--warmup",
                str(args.warmup),
                "--samples",
                str(args.samples),
                "--timing-mode",
                args.timing_mode,
                "--independent-streams",
                str(args.independent_streams),
                "--device-index",
                str(args.device_index),
            ]
            completed = _run_command(harness_command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                raise RuntimeError(f"{args.backend} geometry harness run failed")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw.setdefault("config", {})["hip_wavefront_size_request"] = args.hip_wavefront_size
        shader_command = None
        source_hash = _hash_files(
            [Path(__file__).resolve(), HIP_HARNESS, HIP_TIMING_HEADER, TIMING_CONTRACT]
        )
    else:
        spirv, shader_command = _compile_vulkan_shader(build_dir)
        exe, build_command = _compile_vulkan_harness(build_dir)
        harness_command = [
            str(exe),
            "--spirv",
            str(spirv),
            "--json",
            str(raw_path),
            "--k-list",
            args.k_list,
            "--rows-list",
            args.rows_list,
            "--workgroups",
            args.workgroups,
            "--body-repeats",
            str(args.body_repeats),
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
        source_hash = _hash_files(
            [
                Path(__file__).resolve(),
                VULKAN_HARNESS,
                VULKAN_SHADER,
                VULKAN_TIMING_HEADER,
                TIMING_CONTRACT,
            ]
        )

        completed = _run_command(harness_command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"{args.backend} geometry harness run failed")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    result = normalize_raw_result(
        raw,
        backend=args.backend,
        environment=environment,
        wrapper_command=wrapper_command,
        harness_command=harness_command,
        build_command=build_command,
        shader_command=shader_command,
        source_hash=source_hash,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
    )
    if args.raw_json:
        result["raw_artifact_ref"] = str(args.raw_json)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hip", "vulkan"], help="Backend to run")
    parser.add_argument("--compare", nargs=2, metavar=("HIP_RESULT", "VULKAN_RESULT"), type=Path)
    parser.add_argument("--out", type=Path, help="Write normalized/comparison JSON")
    parser.add_argument("--raw-json", type=Path, help="Keep backend raw harness JSON at this path")
    parser.add_argument("--environment-json", type=Path, help="Use an existing environment artifact")
    parser.add_argument("--environment-ref", type=Path, help="Reference this environment path instead of embedding")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gfx-arch", help="Override gfx arch and HIP offload arch")
    parser.add_argument("--hardware-gpu", help="Override GPU name in normalized output")
    parser.add_argument(
        "--hip-workgroup-specialization",
        choices=["runtime", "fixed"],
        default="fixed",
        help="For HIP, compile runtime blockDim code or one fixed-workgroup binary per requested workgroup",
    )
    parser.add_argument("--hip-wavefront-size", choices=["default", "32", "64"], default="default")
    parser.add_argument("--k-list", default="512,2048,8192")
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--workgroups", default="32,64,128,256")
    parser.add_argument("--body-repeats", type=int, default=128)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument(
        "--timing-mode",
        choices=["serial_latency", "independent_throughput"],
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=8.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.compare and args.backend:
        parser.error("--compare and --backend are mutually exclusive")
    if not args.compare and not args.backend:
        parser.error("one of --backend or --compare is required")
    if args.backend == "vulkan" and args.hip_workgroup_specialization != "fixed":
        parser.error("corrected Vulkan comparisons require fixed HIP workgroup artifacts")
    if args.backend is None and args.hip_workgroup_specialization != "fixed":
        parser.error("comparison mode does not accept runtime HIP workgroup specialization")
    if args.backend != "hip" and args.hip_wavefront_size != "default":
        parser.error("--hip-wavefront-size only applies to --backend hip")
    for name in ("k_list", "rows_list", "workgroups"):
        values = _list_arg(getattr(args, name))
        if not values or any(not value.isdigit() or int(value) <= 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must be a comma-separated positive integer list")
    if args.independent_streams <= 0:
        parser.error("--independent-streams must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        hip_path, vulkan_path = args.compare
        hip_result = json.loads(hip_path.read_text(encoding="utf-8"))
        vulkan_result = json.loads(vulkan_path.read_text(encoding="utf-8"))
        hip_result["artifact_ref"] = str(hip_path)
        vulkan_result["artifact_ref"] = str(vulkan_path)
        result = build_comparison(
            hip_result,
            vulkan_result,
            command=sys.argv.copy(),
            out_ref=str(args.out) if args.out else None,
        )
    else:
        result = _run_backend(args)

    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
