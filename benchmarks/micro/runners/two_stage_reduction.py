#!/usr/bin/env python3
"""Run matched HIP/Vulkan two-stage reduction microbenchmarks."""

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
HIP_SOURCE = MICRO_ROOT / "runners" / "hip_two_stage_reduction.hip"
VULKAN_SOURCE = MICRO_ROOT / "runners" / "vulkan_two_stage_reduction.cpp"
VULKAN_PARTIAL_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "reduction_two_stage_partial.comp"
VULKAN_FINAL_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "reduction_two_stage_final.comp"
HIP_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_hip.hpp"
VULKAN_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
COMPARISON_CLAIM = MICRO_ROOT / "comparison_claim.py"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-two-stage-reduction")


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


def _load_comparison_claim_module():
    spec = importlib.util.spec_from_file_location(
        "micro_comparison_claim", COMPARISON_CLAIM
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load comparison claim helper: {COMPARISON_CLAIM}")
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


def _compile_hip(args: argparse.Namespace, workgroup: int) -> tuple[Path, list[str]]:
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    exe = args.build_dir / f"hip_two_stage_reduction_wg{workgroup}"
    command = [hipcc]
    if args.gfx_arch:
        command.append(f"--offload-arch={args.gfx_arch}")
    command.extend(
        [
            "-O3",
            "-std=c++17",
            f"-DHIPENGINE_FIXED_WORKGROUP_SIZE={workgroup}",
            str(HIP_SOURCE),
            "-o",
            str(exe),
        ]
    )
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("HIP two-stage reduction build failed")
    return exe, command


def _compile_vulkan_shader(args: argparse.Namespace, shader: Path, name: str) -> tuple[Path, list[str]]:
    spirv = args.build_dir / f"{name}.spv"
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    if glslc:
        command = [glslc, "--target-env=vulkan1.1", "-O", str(shader), "-o", str(spirv)]
    elif glslang:
        command = [glslang, "-V", "--target-env", "vulkan1.1", str(shader), "-o", str(spirv)]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan shader build failed: {shader}")
    return spirv, command


def _compile_vulkan_harness(args: argparse.Namespace) -> tuple[Path, list[str]]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    exe = args.build_dir / "vulkan_two_stage_reduction"
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(VULKAN_SOURCE),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan two-stage reduction harness build failed")
    return exe, command


def _base_run_args(
    args: argparse.Namespace,
    raw_json: Path,
    *,
    workgroups: str | None = None,
) -> list[str]:
    return [
        "--json",
        str(raw_json),
        "--k-list",
        args.k_list,
        "--rows-list",
        args.rows_list,
        "--workgroups",
        workgroups or args.workgroups,
        "--split-counts",
        args.split_counts,
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


def _run_hip(
    exe: Path,
    args: argparse.Namespace,
    workgroup: int,
) -> tuple[dict[str, Any], list[str]]:
    raw_json = args.build_dir / f"hip-two-stage-wg{workgroup}.json"
    command = [
        str(exe),
        *_base_run_args(args, raw_json, workgroups=str(workgroup)),
        "--independent-streams",
        str(args.independent_streams),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("HIP two-stage reduction run failed")
    return json.loads(raw_json.read_text(encoding="utf-8")), command


def _run_vulkan(
    exe: Path,
    partial_spirv: Path,
    final_spirv: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    raw_json = args.build_dir / "vulkan-two-stage.json"
    command = [
        str(exe),
        "--partial-spirv",
        str(partial_spirv),
        "--final-spirv",
        str(final_spirv),
        *_base_run_args(args, raw_json),
        "--independent-queues",
        str(args.independent_streams),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan two-stage reduction run failed")
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


def _source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        HIP_SOURCE,
        VULKAN_SOURCE,
        VULKAN_PARTIAL_SHADER,
        VULKAN_FINAL_SHADER,
        HIP_TIMING_HEADER,
        VULKAN_TIMING_HEADER,
        TIMING_CONTRACT,
        COMPARISON_CLAIM,
    ]


def _matrix_evidence(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    raw_results: dict[str, Any],
) -> dict[str, Any]:
    shapes = {
        (k, row_count, workgroup, split_count)
        for k in _parse_csv_u32(args.k_list)
        for row_count in _parse_csv_u32(args.rows_list)
        for workgroup in _parse_csv_u32(args.workgroups)
        for split_count in _parse_csv_u32(args.split_counts)
    }
    expected_row_keys = {
        (k, row_count, workgroup, split_count, args.timing_mode, backend)
        for k, row_count, workgroup, split_count in shapes
        for backend in ("hip", "vulkan")
    }
    indexed = _row_index(rows)
    row_metadata_pass = all(
        row.get("variant") == "two_stage"
        and row.get("workgroup_specialization")
        == ("fixed" if row.get("backend") == "hip" else "specialization_constant")
        and row.get("body_repeats") == args.body_repeats
        for row in rows
    )
    expected_raw_keys = {
        f"hip:wg{workgroup}" for workgroup in _parse_csv_u32(args.workgroups)
    } | {"vulkan"}
    return {
        "backend_pair_requested": args.backend == "both",
        "exact_row_set": set(indexed) == expected_row_keys,
        "row_metadata_pass": row_metadata_pass,
        "exact_raw_result_set": set(raw_results) == expected_raw_keys,
        "expected_rows": len(expected_row_keys),
        "actual_rows": len(rows),
        "expected_matched_rows": len(shapes),
        "actual_matched_rows": len(matched),
        "comparison_set_complete": len(matched) == len(shapes),
    }


def _comparison_provenance(
    *,
    args: argparse.Namespace,
    environment: dict[str, Any],
    source_hash: str,
    raw_results: dict[str, Any],
    rows: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    correctness: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _source_record(environment, source_hash)
    matrix = _matrix_evidence(args, rows, matched, raw_results)
    correctness_pass = correctness.get("status") == "pass" and bool(
        correctness.get("all_rows_pass")
    )
    matrix_complete = all(
        bool(matrix[field])
        for field in (
            "backend_pair_requested",
            "exact_row_set",
            "row_metadata_pass",
            "exact_raw_result_set",
            "comparison_set_complete",
        )
    )
    claim = _load_comparison_claim_module()
    hardware, source_coverage, claim_gate = claim.build_joint_claim_evidence(
        source=source,
        source_paths=_source_paths(),
        repo_root=REPO_ROOT,
        raw_results=raw_results,
        expected_run_tags={
            "hip": "hip-two-stage-reduction",
            "vulkan": "vulkan-two-stage-reduction",
        },
        configured_arch=str(args.gfx_arch or ""),
        fallback_gpu=str(args.hardware_gpu or ""),
        correctness_pass=correctness_pass,
        matrix=matrix,
        matrix_complete=matrix_complete,
    )
    return source, hardware, source_coverage, claim_gate


def _annotate_rows(raw: dict[str, Any], *, backend: str) -> list[dict[str, Any]]:
    timing_contract = _load_timing_contract_module()
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    repetitions = int(config.get("reps", 0))
    timing_mode = timing_contract.parse_timing_mode(str(config.get("timing_mode")))
    if repetitions <= 0:
        raise ValueError(f"{backend} two-stage result has invalid repetitions")
    rows = []
    for row in raw.get("rows", []):
        item = dict(row)
        if item.get("timing_mode") != timing_mode:
            raise ValueError(f"{backend} two-stage row timing mode disagrees with config")
        timing_contract.validate_timed_row(item, expected_repetitions=repetitions)
        item["backend"] = backend
        item["variant"] = "two_stage"
        item["workgroup_specialization"] = (
            "fixed" if backend == "hip" else "specialization_constant"
        )
        rows.append(item)
    return rows


def _row_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, Any, Any, Any, Any, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (
            row.get("k"),
            row.get("rows"),
            row.get("workgroup_size"),
            row.get("split_count"),
            row.get("timing_mode"),
            row.get("backend"),
        )
        if key in indexed:
            raise ValueError(f"duplicate two-stage result row: {key}")
        indexed[key] = row
    return indexed


def _matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timing_contract = _load_timing_contract_module()
    index = _row_index(rows)
    shape_keys = sorted({key[:5] for key in index})
    matched = []
    for k, row_count, workgroup_size, split_count, timing_mode in shape_keys:
        hip = index.get((k, row_count, workgroup_size, split_count, timing_mode, "hip"))
        vulkan = index.get(
            (k, row_count, workgroup_size, split_count, timing_mode, "vulkan")
        )
        if not hip or not vulkan:
            continue
        hip_lanes = int(hip.get("submission", {}).get("queue_or_stream_count", 0))
        vulkan_lanes = int(
            vulkan.get("submission", {}).get("queue_or_stream_count", 0)
        )
        if hip_lanes <= 0 or hip_lanes != vulkan_lanes:
            raise ValueError(
                "HIP and Vulkan two-stage worker lane counts do not match"
            )
        ratios: dict[str, Any] = {}
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
        burst_gpu = ratios["burst"]["gpu_elapsed"]
        matched.append(
            {
                "k": k,
                "rows": row_count,
                "workgroup_size": workgroup_size,
                "split_count": split_count,
                "timing_mode": timing_mode,
                "worker_lanes": hip_lanes,
                "ratios": ratios,
                "hip_gpu_burst_median_us": burst_gpu.get("hip_us_per_iteration"),
                "vulkan_gpu_burst_median_us": burst_gpu.get(
                    "vulkan_us_per_iteration"
                ),
                "vulkan_vs_hip_gpu_burst_speedup": burst_gpu.get(
                    "vulkan_vs_hip_speedup"
                ),
                "hip_correctness_pass": bool(hip.get("correctness_pass")),
                "vulkan_correctness_pass": bool(vulkan.get("correctness_pass")),
            }
        )
    return matched


def _summary(matched: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [
        float(item["vulkan_vs_hip_gpu_burst_speedup"])
        for item in matched
        if isinstance(item.get("vulkan_vs_hip_gpu_burst_speedup"), (int, float))
    ]
    return {
        "matched_rows": len(matched),
        "speedup_min": min(speedups) if speedups else None,
        "speedup_max": max(speedups) if speedups else None,
        "speedup_median": sorted(speedups)[len(speedups) // 2] if speedups else None,
        "all_correctness_pass": all(
            bool(item.get("hip_correctness_pass")) and bool(item.get("vulkan_correctness_pass"))
            for item in matched
        ),
        "primary_domain": "gpu_elapsed",
        "host_wall_status": "not_comparable_direct_vs_command_buffer",
    }


def _schema_comparisons(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timing_contract = _load_timing_contract_module()
    out: list[dict[str, Any]] = []
    for item in matched:
        ratios = item.get("ratios")
        if not isinstance(ratios, dict):
            raise ValueError("two-stage matched row is missing timing ratios")
        for control in timing_contract.TIMING_CONTROLS:
            domains = ratios.get(control)
            if not isinstance(domains, dict):
                raise ValueError(f"two-stage matched row is missing {control} ratios")
            gpu_elapsed = domains.get("gpu_elapsed")
            host_wall = domains.get("host_wall")
            if not isinstance(gpu_elapsed, dict) or not isinstance(host_wall, dict):
                raise ValueError(
                    f"two-stage matched row has incomplete {control} domains"
                )
            out.append(
                {
                    "k": item.get("k"),
                    "rows": item.get("rows"),
                    "workgroup_size": item.get("workgroup_size"),
                    "split_count": item.get("split_count"),
                    "worker_lanes": item.get("worker_lanes"),
                    "timing_mode": item.get("timing_mode"),
                    "control": control,
                    "gpu_elapsed": gpu_elapsed,
                    "host_wall": host_wall,
                    "hip_correctness_pass": item.get("hip_correctness_pass"),
                    "vulkan_correctness_pass": item.get("vulkan_correctness_pass"),
                }
            )
    return out


def _correctness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def backend_summary(backend: str) -> dict[str, Any]:
        backend_rows = [row for row in rows if row.get("backend") == backend]
        if not backend_rows:
            return {"status": "not_run", "row_count": 0}
        all_pass = all(bool(row.get("correctness_pass")) for row in backend_rows)
        return {
            "status": "pass" if all_pass else "fail",
            "row_count": len(backend_rows),
            "all_rows_pass": all_pass,
        }

    all_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    return {
        "status": "pass" if all_pass else "fail" if rows else "not_run",
        "row_count": len(rows),
        "all_rows_pass": all_pass,
        "hip": backend_summary("hip"),
        "vulkan": backend_summary("vulkan"),
    }


def _build_comparison_artifact(
    *,
    args: argparse.Namespace,
    environment: dict[str, Any],
    source_hash: str,
    commands: list[dict[str, Any]],
    raw_results: dict[str, Any],
    rows: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    wrapper_command: list[str],
) -> dict[str, Any]:
    config = {
        "backend": args.backend,
        "k_list": _parse_csv_u32(args.k_list),
        "rows_list": _parse_csv_u32(args.rows_list),
        "workgroups": _parse_csv_u32(args.workgroups),
        "split_counts": _parse_csv_u32(args.split_counts),
        "body_repeats": args.body_repeats,
        "reps": args.reps,
        "warmup": args.warmup,
        "samples": args.samples,
        "timing_mode": args.timing_mode,
        "independent_streams": args.independent_streams,
        "independent_queues": args.independent_streams,
    }
    correctness = _correctness_summary(rows)
    source, hardware, source_coverage, claim_gate = _comparison_provenance(
        args=args,
        environment=environment,
        source_hash=source_hash,
        raw_results=raw_results,
        rows=rows,
        matched=matched,
        correctness=correctness,
    )
    performance_claim = bool(claim_gate["performance_claim"])
    return {
        "schema": "hipengine.micro.two_stage_reduction.v2",
        "schema_version": 2,
        "kind": "hipengine_micro_comparison",
        "bench": "two_stage_reduction",
        "classification": "diagnostic_unclassified",
        "performance_claim": performance_claim,
        "hardware": hardware,
        "source": source,
        "sources": {"shared": source},
        "source_coverage": source_coverage,
        "claim_gate": claim_gate,
        "command": wrapper_command,
        "inputs": config,
        "correctness": correctness,
        "comparisons": _schema_comparisons(matched),
        "environment": {
            "ref": args.environment_ref,
            "captured": environment if not args.environment_ref else None,
        },
        "config": config,
        "commands": _json_safe(commands),
        "raw_results": raw_results,
        "rows": rows,
        "matched_rows": matched,
        "summary": {**_summary(matched), "performance_claim": performance_claim},
        "artifact_ref": str(args.out),
        "interpretation": (
            "True two-stage f32 reduction control. serial_latency orders each partial "
            "and final dispatch and reuses shared state. independent_throughput uses "
            "disjoint slices with one intra-operation partial-to-final dependency per "
            "logical operation, distributed over matched HIP stream and Vulkan queue "
            "lanes. Vulkan cross-queue GPU span uses calibrated timestamps. GPU ratios "
            "are comparable; unlike host submission classes are intentionally not "
            "compared."
        ),
        "wrapper": {
            "command": wrapper_command,
            "cwd": str(REPO_ROOT),
            "build_dir": str(args.build_dir),
        },
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
        "split_counts": _parse_csv_u32(args.split_counts),
        "body_repeats": args.body_repeats,
        "reps": args.reps,
        "warmup": args.warmup,
        "samples": args.samples,
        "timing_mode": args.timing_mode,
    }
    expected[
        "independent_streams" if backend == "hip" else "independent_queues"
    ] = args.independent_streams
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"{backend} two-stage raw config {field} does not match invocation"
            )


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
    parser.add_argument("--k-list", default="8192,32768")
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--workgroups", default="128,256")
    parser.add_argument("--split-counts", default="2,4,8")
    parser.add_argument("--body-repeats", type=int, default=32)
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
    _parse_csv_u32(args.split_counts)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    environment = _collect_environment(args)
    commands: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    if args.backend in ("both", "hip"):
        for workgroup in _parse_csv_u32(args.workgroups):
            hip_exe, build_command = _compile_hip(args, workgroup)
            raw, run_command = _run_hip(hip_exe, args, workgroup)
            _validate_raw_config(raw, args, backend="hip")
            raw_results[f"hip:wg{workgroup}"] = raw
            rows.extend(_annotate_rows(raw, backend="hip"))
            commands.append(
                {
                    "kind": "hip",
                    "workgroup": workgroup,
                    "build_command": build_command,
                    "run_command": run_command,
                }
            )

    if args.backend in ("both", "vulkan"):
        vulkan_exe, harness_command = _compile_vulkan_harness(args)
        partial_spirv, partial_command = _compile_vulkan_shader(
            args, VULKAN_PARTIAL_SHADER, "reduction_two_stage_partial"
        )
        final_spirv, final_command = _compile_vulkan_shader(
            args, VULKAN_FINAL_SHADER, "reduction_two_stage_final"
        )
        raw, run_command = _run_vulkan(vulkan_exe, partial_spirv, final_spirv, args)
        _validate_raw_config(raw, args, backend="vulkan")
        raw_results["vulkan"] = raw
        rows.extend(_annotate_rows(raw, backend="vulkan"))
        commands.append(
            {
                "kind": "vulkan",
                "harness_build_command": harness_command,
                "partial_shader_command": partial_command,
                "final_shader_command": final_command,
                "run_command": run_command,
            }
        )

    matched = _matched_rows(rows)
    if args.backend == "both":
        expected_matches = (
            len(_parse_csv_u32(args.k_list))
            * len(_parse_csv_u32(args.rows_list))
            * len(_parse_csv_u32(args.workgroups))
            * len(_parse_csv_u32(args.split_counts))
        )
        if len(matched) != expected_matches:
            raise ValueError(
                "HIP and Vulkan two-stage row sets do not match exactly: "
                f"expected {expected_matches}, got {len(matched)}"
            )
    source_hash = _hash_files(_source_paths())
    result = _build_comparison_artifact(
        args=args,
        environment=environment,
        source_hash=source_hash,
        commands=commands,
        raw_results=raw_results,
        rows=rows,
        matched=matched,
        wrapper_command=[Path(sys.executable).name, *sys.argv],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    args.out.write_text(json.dumps(_json_safe(result), indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
