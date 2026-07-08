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
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-two-stage-reduction")


def _load_collect_env_module():
    spec = importlib.util.spec_from_file_location("micro_collect_env", COLLECT_ENV)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load environment collector: {COLLECT_ENV}")
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


def _compile_hip(args: argparse.Namespace) -> tuple[Path, list[str]]:
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    exe = args.build_dir / "hip_two_stage_reduction"
    command = [hipcc]
    if args.gfx_arch:
        command.append(f"--offload-arch={args.gfx_arch}")
    command.extend(["-O3", "-std=c++17", str(HIP_SOURCE), "-o", str(exe)])
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


def _base_run_args(args: argparse.Namespace, raw_json: Path) -> list[str]:
    return [
        "--json",
        str(raw_json),
        "--k-list",
        args.k_list,
        "--rows-list",
        args.rows_list,
        "--workgroups",
        args.workgroups,
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
        "--device-index",
        str(args.device_index),
    ]


def _run_hip(exe: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    raw_json = args.build_dir / "hip-two-stage.json"
    command = [str(exe), *_base_run_args(args, raw_json)]
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
        timeout_s=args.env_timeout_s,
        max_output_chars=args.env_max_output_chars,
    )


def _annotate_rows(raw: dict[str, Any], *, backend: str) -> list[dict[str, Any]]:
    rows = []
    for row in raw.get("rows", []):
        item = dict(row)
        item["backend"] = backend
        item["variant"] = "two_stage"
        rows.append(item)
    return rows


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any, Any, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = (
            row.get("k"),
            row.get("rows"),
            row.get("workgroup_size"),
            row.get("split_count"),
            row.get("backend"),
        )
        indexed[key] = row
    return indexed


def _matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = _row_index(rows)
    shape_keys = sorted({key[:4] for key in index})
    matched = []
    for k, row_count, workgroup_size, split_count in shape_keys:
        hip = index.get((k, row_count, workgroup_size, split_count, "hip"))
        vulkan = index.get((k, row_count, workgroup_size, split_count, "vulkan"))
        if not hip or not vulkan:
            continue
        hip_us = float(hip["median_us"])
        vulkan_us = float(vulkan["median_us"])
        matched.append(
            {
                "k": k,
                "rows": row_count,
                "workgroup_size": workgroup_size,
                "split_count": split_count,
                "hip_median_us": hip_us,
                "vulkan_median_us": vulkan_us,
                "vulkan_vs_hip_speedup": hip_us / vulkan_us if vulkan_us > 0.0 else None,
                "hip_correctness_pass": bool(hip.get("correctness_pass")),
                "vulkan_correctness_pass": bool(vulkan.get("correctness_pass")),
            }
        )
    return matched


def _summary(matched: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [
        float(item["vulkan_vs_hip_speedup"])
        for item in matched
        if isinstance(item.get("vulkan_vs_hip_speedup"), (int, float))
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
    parser.add_argument("--k-list", default="8192,32768")
    parser.add_argument("--rows-list", default="1,4,8")
    parser.add_argument("--workgroups", default="128,256")
    parser.add_argument("--split-counts", default="2,4,8")
    parser.add_argument("--body-repeats", type=int, default=32)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


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
        hip_exe, build_command = _compile_hip(args)
        raw, run_command = _run_hip(hip_exe, args)
        raw_results["hip"] = raw
        rows.extend(_annotate_rows(raw, backend="hip"))
        commands.append(
            {"kind": "hip", "build_command": build_command, "run_command": run_command}
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
    source_paths = [
        Path(__file__).resolve(),
        HIP_SOURCE,
        VULKAN_SOURCE,
        VULKAN_PARTIAL_SHADER,
        VULKAN_FINAL_SHADER,
    ]
    result = {
        "schema": "hipengine.micro.two_stage_reduction.v1",
        "kind": "hipengine_micro_result",
        "bench": "two_stage_reduction",
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
            "split_counts": _parse_csv_u32(args.split_counts),
            "body_repeats": args.body_repeats,
            "reps": args.reps,
            "warmup": args.warmup,
            "samples": args.samples,
        },
        "environment": {
            "ref": args.environment_ref,
            "captured": environment if not args.environment_ref else None,
        },
        "source": {
            "repo": str(REPO_ROOT),
            "source_hash": _hash_files(source_paths),
        },
        "commands": _json_safe(commands),
        "raw_results": raw_results,
        "rows": rows,
        "matched_rows": matched,
        "summary": _summary(matched),
        "interpretation": (
            "True two-stage f32 reduction control: each timed repetition records a "
            "block-partial dispatch followed by a final-reduce dispatch. This closes "
            "the previous block-partial plus final-reduce matrix gap for the tested "
            "large-K shapes."
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
