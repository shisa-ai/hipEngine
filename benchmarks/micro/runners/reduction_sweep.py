#!/usr/bin/env python3
"""Run HIP/Vulkan reduction-shape microbenchmarks."""

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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
HIP_HARNESS = MICRO_ROOT / "runners" / "hip_geometry_sweep.hip"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_geometry_sweep.cpp"
VULKAN_SHADERS = {
    "lds_tree": MICRO_ROOT / "kernels" / "vulkan" / "geometry_sweep.comp",
    "extra_barrier": MICRO_ROOT / "kernels" / "vulkan" / "reduction_extra_barrier.comp",
    "subgroup": MICRO_ROOT / "kernels" / "vulkan" / "reduction_subgroup.comp",
    "multi_accum": MICRO_ROOT / "kernels" / "vulkan" / "reduction_multi_accum.comp",
}
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-reduction-sweep")

HIP_VARIANTS = {
    "lds_tree": 0,
    "extra_barrier": 1,
    "wave_shuffle": 2,
    "multi_accum4": 3,
    "multi_accum8": 4,
    "multi_accum16": 5,
}
VULKAN_VARIANTS = {
    "lds_tree": {"shader": "lds_tree", "defines": []},
    "extra_barrier": {"shader": "extra_barrier", "defines": []},
    "subgroup": {"shader": "subgroup", "defines": []},
    "multi_accum4": {"shader": "multi_accum", "defines": ["-DHIPENGINE_ACCUM_COUNT=4"]},
    "multi_accum8": {"shader": "multi_accum", "defines": ["-DHIPENGINE_ACCUM_COUNT=8"]},
    "multi_accum16": {"shader": "multi_accum", "defines": ["-DHIPENGINE_ACCUM_COUNT=16"]},
}


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
    if isinstance(value, Path):
        return str(value)
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


def _parse_csv_u32(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text}")
    return values


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


def _compile_hip(
    args: argparse.Namespace,
    variant: str,
    variant_id: int,
    workgroup: int,
) -> tuple[Path, list[str]]:
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    exe = args.build_dir / f"hip_reduction_{variant}_wg{workgroup}"
    command = [hipcc]
    if args.gfx_arch:
        command.append(f"--offload-arch={args.gfx_arch}")
    command.extend(
        [
            "-O3",
            "-std=c++17",
            f"-DHIPENGINE_REDUCTION_VARIANT={variant_id}",
            f"-DHIPENGINE_FIXED_WORKGROUP_SIZE={workgroup}",
            str(HIP_HARNESS),
            "-o",
            str(exe),
        ]
    )
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"HIP reduction build failed for {variant}")
    return exe, command


def _compile_vulkan_shader(args: argparse.Namespace, variant: str) -> tuple[Path, list[str]]:
    variant_config = VULKAN_VARIANTS[variant]
    shader = VULKAN_SHADERS[str(variant_config["shader"])]
    defines = list(variant_config["defines"])
    spirv = args.build_dir / f"vulkan_reduction_{variant}.spv"
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
        raise RuntimeError(f"Vulkan reduction shader build failed for {variant}")
    return spirv, command


def _compile_vulkan_harness(args: argparse.Namespace) -> tuple[Path, list[str]]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    exe = args.build_dir / "vulkan_reduction_sweep"
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
        raise RuntimeError("Vulkan reduction harness build failed")
    return exe, command


def _run_raw(
    exe: Path,
    raw_json: Path,
    args: argparse.Namespace,
    *,
    spirv: Path | None = None,
    workgroups: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    command = [str(exe)]
    if spirv is not None:
        command.extend(["--spirv", str(spirv)])
    command.extend(
        [
            "--json",
            str(raw_json),
            "--k-list",
            args.k_list,
            "--rows-list",
            args.rows_list,
            "--workgroups",
            workgroups or args.workgroups,
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
            *(
                ["--independent-streams", str(args.independent_streams)]
                if spirv is None
                else []
            ),
            "--device-index",
            str(args.device_index),
        ]
    )
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"reduction run failed: {' '.join(command)}")
    return json.loads(raw_json.read_text(encoding="utf-8")), command


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


def _validate_raw_config(
    raw: dict[str, Any],
    args: argparse.Namespace,
    *,
    backend: str,
) -> None:
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    expected = {
        "k_list": _parse_csv_u32(args.k_list),
        "rows_list": _parse_csv_u32(args.rows_list),
        "body_repeats": args.body_repeats,
        "reps": args.reps,
        "warmup": args.warmup,
        "samples": args.samples,
        "timing_mode": args.timing_mode,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"{backend} reduction raw config {field} does not match invocation"
            )


def _annotate_rows(raw: dict[str, Any], *, backend: str, variant: str) -> list[dict[str, Any]]:
    timing_contract = _load_timing_contract_module()
    repetitions = int(raw.get("config", {}).get("reps", 0))
    out = []
    for row in raw.get("rows", []):
        item = dict(row)
        timing_contract.validate_timed_row(item, expected_repetitions=repetitions)
        item["backend"] = backend
        item["variant"] = variant
        item["workgroup_specialization"] = (
            "fixed" if backend == "hip" else "specialization_constant"
        )
        item["row_key"] = {
            "k": item.get("k"),
            "rows": item.get("rows"),
            "workgroup_size": item.get("workgroup_size"),
        }
        out.append(item)
    return out


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any, str, str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (
            row.get("k"),
            row.get("rows"),
            row.get("workgroup_size"),
            row.get("backend"),
            row.get("variant"),
            row.get("timing_mode"),
        )
        if key in indexed:
            raise ValueError(f"duplicate reduction result row: {key}")
        indexed[key] = row
    return indexed


def _ratio(a: Any, b: Any) -> float | None:
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return None
    if bf <= 0.0:
        return None
    return af / bf


def _timing_median(row: dict[str, Any], control: str, domain: str) -> float | None:
    metric = row.get("timing", {}).get(control, {}).get(domain, {})
    if metric.get("status") != "ok":
        return None
    value = metric.get("per_iteration_us", {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _backend_timing_ratios(
    hip: dict[str, Any],
    vulkan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    timing_contract = _load_timing_contract_module()
    ratios: dict[str, dict[str, Any]] = {}
    for control in timing_contract.TIMING_CONTROLS:
        domains: dict[str, Any] = {}
        for domain in timing_contract.TIMING_DOMAINS:
            try:
                domains[domain] = {
                    "status": "ok",
                    **timing_contract.comparison_ratio(
                        hip,
                        vulkan,
                        control=control,
                        domain=domain,
                    ),
                }
            except ValueError as exc:
                domains[domain] = {
                    "status": (
                        "not_comparable_submission_contract"
                        if domain == "host_wall"
                        else "not_comparable"
                    ),
                    "reason": str(exc),
                }
        ratios[control] = domains
    return ratios


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    timing_contract = _load_timing_contract_module()
    index = _row_index(rows)
    keys = sorted({
        (row.get("k"), row.get("rows"), row.get("workgroup_size"), row.get("timing_mode"))
        for row in rows
    })
    backend_pairs = [
        ("lds_tree", "hip", "vulkan", "vulkan_lds_vs_hip_lds"),
        ("extra_barrier", "hip", "vulkan", "vulkan_extra_barrier_vs_hip_extra_barrier"),
        ("multi_accum4", "hip", "vulkan", "vulkan_multi_accum4_vs_hip_multi_accum4"),
        ("multi_accum8", "hip", "vulkan", "vulkan_multi_accum8_vs_hip_multi_accum8"),
        ("multi_accum16", "hip", "vulkan", "vulkan_multi_accum16_vs_hip_multi_accum16"),
    ]
    variant_pairs = [
        ("hip", "extra_barrier", "lds_tree", "hip_extra_barrier_vs_lds_tree"),
        ("vulkan", "extra_barrier", "lds_tree", "vulkan_extra_barrier_vs_lds_tree"),
        ("hip", "wave_shuffle", "lds_tree", "hip_wave_shuffle_vs_lds_tree"),
        ("vulkan", "subgroup", "lds_tree", "vulkan_subgroup_vs_lds_tree"),
        ("hip", "multi_accum4", "lds_tree", "hip_multi_accum4_vs_lds_tree"),
        ("hip", "multi_accum8", "lds_tree", "hip_multi_accum8_vs_lds_tree"),
        ("hip", "multi_accum16", "lds_tree", "hip_multi_accum16_vs_lds_tree"),
        ("vulkan", "multi_accum4", "lds_tree", "vulkan_multi_accum4_vs_lds_tree"),
        ("vulkan", "multi_accum8", "lds_tree", "vulkan_multi_accum8_vs_lds_tree"),
        ("vulkan", "multi_accum16", "lds_tree", "vulkan_multi_accum16_vs_lds_tree"),
        ("mixed", "vulkan_subgroup", "hip_wave_shuffle", "vulkan_subgroup_vs_hip_wave_shuffle"),
    ]
    out: dict[str, list[dict[str, Any]]] = {"backend": [], "variant": []}
    for k, rows_count, wg, timing_mode in keys:
        for variant, lhs_backend, rhs_backend, label in backend_pairs:
            lhs = index.get((k, rows_count, wg, lhs_backend, variant, timing_mode))
            rhs = index.get((k, rows_count, wg, rhs_backend, variant, timing_mode))
            if lhs and rhs:
                if timing_contract.dependency_signature(lhs) != timing_contract.dependency_signature(rhs):
                    raise ValueError("reduction backend dependency contracts do not match")
                ratios = _backend_timing_ratios(lhs, rhs)
                burst_gpu = ratios["burst"]["gpu_elapsed"]
                out["backend"].append(
                    {
                        "comparison": label,
                        "k": k,
                        "rows": rows_count,
                        "workgroup_size": wg,
                        "lhs_backend": lhs_backend,
                        "rhs_backend": rhs_backend,
                        "variant": variant,
                        "timing_mode": timing_mode,
                        "ratios": ratios,
                        "hip_gpu_burst_median_us": burst_gpu.get(
                            "hip_us_per_iteration"
                        ),
                        "vulkan_gpu_burst_median_us": burst_gpu.get(
                            "vulkan_us_per_iteration"
                        ),
                        "vulkan_vs_hip_gpu_burst_speedup": burst_gpu.get(
                            "vulkan_vs_hip_speedup"
                        ),
                        "lhs_correctness_pass": lhs.get("correctness_pass"),
                        "rhs_correctness_pass": rhs.get("correctness_pass"),
                    }
                )
        for backend, lhs_variant, rhs_variant, label in variant_pairs:
            if backend == "mixed":
                lhs = index.get((k, rows_count, wg, "vulkan", "subgroup", timing_mode))
                rhs = index.get((k, rows_count, wg, "hip", "wave_shuffle", timing_mode))
            else:
                lhs = index.get((k, rows_count, wg, backend, lhs_variant, timing_mode))
                rhs = index.get((k, rows_count, wg, backend, rhs_variant, timing_mode))
            if lhs and rhs:
                if timing_contract.dependency_signature(lhs) != timing_contract.dependency_signature(rhs):
                    raise ValueError("reduction variant dependency contracts do not match")
                lhs_gpu = _timing_median(lhs, "burst", "gpu_elapsed")
                rhs_gpu = _timing_median(rhs, "burst", "gpu_elapsed")
                out["variant"].append(
                    {
                        "comparison": label,
                        "k": k,
                        "rows": rows_count,
                        "workgroup_size": wg,
                        "lhs_backend": lhs.get("backend"),
                        "lhs_variant": lhs.get("variant"),
                        "rhs_backend": rhs.get("backend"),
                        "rhs_variant": rhs.get("variant"),
                        "timing_mode": timing_mode,
                        "timing_domain": "gpu_elapsed",
                        "control": "burst",
                        "lhs_gpu_burst_median_us": lhs_gpu,
                        "rhs_gpu_burst_median_us": rhs_gpu,
                        "rhs_over_lhs_time_ratio": _ratio(rhs_gpu, lhs_gpu),
                        "lhs_correctness_pass": lhs.get("correctness_pass"),
                        "rhs_correctness_pass": rhs.get("correctness_pass"),
                    }
                )
    return out


def _summarize_ratios(items: list[dict[str, Any]], ratio_key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        ratio = item.get(ratio_key)
        if isinstance(ratio, (int, float)) and math.isfinite(float(ratio)):
            grouped.setdefault(str(item.get("comparison")), []).append(float(ratio))
    return {
        label: {
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2],
        }
        for label, values in grouped.items()
        if values
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("both", "hip", "vulkan"), default="both")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--environment-ref")
    parser.add_argument("--skip-device-probes", action="store_true")
    parser.add_argument("--env-timeout-s", type=float, default=5.0)
    parser.add_argument("--env-max-output-chars", type=int, default=16000)
    parser.add_argument("--gfx-arch", default=os.environ.get("HIPENGINE_HIP_ARCH") or "")
    parser.add_argument("--hardware-gpu", default="")
    parser.add_argument("--k-list", default="512,2048,8192")
    parser.add_argument("--rows-list", default="1")
    parser.add_argument("--workgroups", default="64,256")
    parser.add_argument("--body-repeats", type=int, default=128)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--timing-mode",
        choices=("serial_latency", "independent_throughput"),
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.independent_streams <= 0:
        parser.error("--independent-streams must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _parse_csv_u32(args.k_list)
    _parse_csv_u32(args.rows_list)
    _parse_csv_u32(args.workgroups)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    environment = _collect_environment(args)
    commands: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    if args.backend in ("both", "hip"):
        for variant, variant_id in HIP_VARIANTS.items():
            for workgroup in _parse_csv_u32(args.workgroups):
                exe, build_command = _compile_hip(args, variant, variant_id, workgroup)
                raw_json = args.build_dir / f"hip-{variant}-wg{workgroup}.json"
                raw, run_command = _run_raw(
                    exe,
                    raw_json,
                    args,
                    workgroups=str(workgroup),
                )
                _validate_raw_config(raw, args, backend="hip")
                raw_results[f"hip:{variant}:wg{workgroup}"] = raw
                rows.extend(_annotate_rows(raw, backend="hip", variant=variant))
                commands.append(
                    {
                        "kind": "hip",
                        "variant": variant,
                        "workgroup": workgroup,
                        "build_command": build_command,
                        "run_command": run_command,
                    }
                )

    if args.backend in ("both", "vulkan"):
        vulkan_exe, harness_command = _compile_vulkan_harness(args)
        commands.append({"kind": "vulkan_harness", "build_command": harness_command})
        for variant in VULKAN_VARIANTS:
            spirv, shader_command = _compile_vulkan_shader(args, variant)
            raw_json = args.build_dir / f"vulkan-{variant}.json"
            raw, run_command = _run_raw(vulkan_exe, raw_json, args, spirv=spirv)
            _validate_raw_config(raw, args, backend="vulkan")
            raw_results[f"vulkan:{variant}"] = raw
            rows.extend(_annotate_rows(raw, backend="vulkan", variant=variant))
            commands.append(
                {
                    "kind": "vulkan",
                    "variant": variant,
                    "shader_command": shader_command,
                    "run_command": run_command,
                }
            )

    comparisons = _comparisons(rows)
    shape_count = (
        len(_parse_csv_u32(args.k_list))
        * len(_parse_csv_u32(args.rows_list))
        * len(_parse_csv_u32(args.workgroups))
    )
    if args.backend == "both":
        expected_rows = shape_count * (len(HIP_VARIANTS) + len(VULKAN_VARIANTS))
        if len(rows) != expected_rows:
            raise ValueError(
                "HIP and Vulkan reduction row sets do not match the requested matrix: "
                f"expected {expected_rows}, got {len(rows)}"
            )
        expected_backend_pairs = shape_count * 5
        if len(comparisons["backend"]) != expected_backend_pairs:
            raise ValueError(
                "HIP and Vulkan matched reduction row sets are incomplete: "
                f"expected {expected_backend_pairs}, got {len(comparisons['backend'])}"
            )
    source_paths = [
        Path(__file__).resolve(),
        HIP_HARNESS,
        VULKAN_HARNESS,
        *VULKAN_SHADERS.values(),
        MICRO_ROOT / "runners" / "micro_timing_hip.hpp",
        MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp",
        TIMING_CONTRACT,
    ]
    source_hash = _hash_files(source_paths)
    result = {
        "schema": "hipengine.micro.reduction_sweep.v2",
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": "reduction_sweep",
        "classification": "diagnostic_unclassified",
        "hardware": {
            "gfx_arch": args.gfx_arch or "unknown",
            "gpu_name": args.hardware_gpu or "unknown",
        },
        "config": {
            "backend": args.backend,
            "k_list": _parse_csv_u32(args.k_list),
            "rows_list": _parse_csv_u32(args.rows_list),
            "workgroups": _parse_csv_u32(args.workgroups),
            "body_repeats": args.body_repeats,
            "reps": args.reps,
            "warmup": args.warmup,
            "samples": args.samples,
            "timing_mode": args.timing_mode,
            "independent_streams": args.independent_streams,
            "variants": {
                "hip": list(HIP_VARIANTS),
                "vulkan": list(VULKAN_VARIANTS),
            },
        },
        "environment": {
            "ref": args.environment_ref,
            "captured": environment if not args.environment_ref else None,
        },
        "source": _source_record(environment, source_hash),
        "commands": _json_safe(commands),
        "rows": rows,
        "comparisons": comparisons,
        "summary": {
            "backend_ratio_summary": _summarize_ratios(
                comparisons["backend"], "vulkan_vs_hip_gpu_burst_speedup"
            ),
            "variant_ratio_summary": _summarize_ratios(
                comparisons["variant"], "rhs_over_lhs_time_ratio"
            ),
            "all_correctness_pass": all(bool(row.get("correctness_pass")) for row in rows),
            "primary_domain": "gpu_elapsed",
            "host_wall_status": "not_comparable_direct_vs_command_buffer",
        },
        "artifact_ref": str(args.out),
        "interpretation": (
            "Reduction-shape control for LDS tree, extra barrier, HIP wave-shuffle, "
            "Vulkan subgroup reductions, and 4/8/16-way lane-local accumulators. "
            "serial_latency orders shared-output iterations; independent_throughput "
            "uses disjoint output slices. Cross-backend GPU ratios use equal dependency "
            "contracts, while direct/multi-stream HIP and Vulkan command-buffer host "
            "walls are intentionally not ratioed."
        ),
        "wrapper": {
            "command": [Path(sys.executable).name, *sys.argv],
            "cwd": str(REPO_ROOT),
            "build_dir": str(args.build_dir),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    args.out.write_text(json.dumps(_json_safe(result), indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
