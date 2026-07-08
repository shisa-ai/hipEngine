#!/usr/bin/env python3
"""Run paired HIP/Vulkan sampler top-k/argmax microbenchmarks."""

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
HIP_HARNESS = MICRO_ROOT / "runners" / "hip_sampler_argmax.hip"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_sampler_argmax.cpp"
VULKAN_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "sampler_argmax.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
ISA_STATS = MICRO_ROOT / "runners" / "isa_stats.py"
BENCH_NAME = "sampler_argmax_topk"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-sampler-topk")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_json:
        return json.loads(args.environment_json.read_text(encoding="utf-8"))
    collector = _load_module(COLLECT_ENV, "micro_collect_env")
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


def _harness_args(args: argparse.Namespace, raw_path: Path, rows: int) -> list[str]:
    return [
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
        "--device-index",
        str(args.device_index),
    ]


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
    rows = []
    for raw in raw_rows:
        isa = isa_by_variant[(int(raw["workgroup_size"]), int(raw.get("top_k", 1)))]
        row = {**raw, **{f"isa_{k}": v for k, v in isa.items()}}
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
        rows.append(row)

    primary = min(rows, key=lambda row: float(row["median_us"])) if rows else {}
    correctness_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    raw0 = raw_rows[0] if raw_rows else {}
    result: dict[str, Any] = {
        "schema_version": 1,
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
            "rows_list": sorted({int(row["rows"]) for row in rows}),
            "workgroups": sorted({int(row["workgroup_size"]) for row in rows}),
            "top_k_list": sorted({int(row.get("top_k", 1)) for row in rows}),
            "vocab": primary.get("vocab"),
            "commands": commands,
        },
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "CPU top-k over deterministic logits with stable value/index ordering",
            "max_abs": max((float(row.get("max_abs", 0.0)) for row in rows), default=0.0),
            "mismatches": sum((int(row.get("mismatches", 0)) for row in rows), 0),
        },
        "timing": {
            "unit": "us_per_dispatch",
            "median": primary.get("median_us"),
            "primary": primary,
            "summary": {
                "best_bandwidth_gbps": max(
                    (float(row.get("bandwidth_gbps", 0.0)) for row in rows),
                    default=0.0,
                ),
                "row_count": len(rows),
            },
        },
        "isa": primary,
        "classification": "diagnostic_unclassified",
        "measurements": {"rows": rows},
        "notes": (
            "Sampler/top-k argmax diagnostic. One workgroup repeatedly reduces one logits row; "
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
    source_hash = _hash_files([Path(__file__).resolve(), HIP_HARNESS])
    raw_rows: list[dict[str, Any]] = []
    isa_by_variant: dict[tuple[int, int], dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    for top_k in top_k_list:
        for workgroup in workgroups:
            exe, obj, build_command = _compile_hip(workgroup, top_k, args)
            isa_by_variant[(workgroup, top_k)] = _hip_isa(obj)
            for rows in rows_list:
                raw_path = args.build_dir / "hip" / f"wg{workgroup}_k{top_k}" / f"rows{rows}.json"
                harness_command = [str(exe), *_harness_args(args, raw_path, rows)]
                completed = _run_command(harness_command, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"HIP sampler argmax run failed for wg{workgroup} top_k{top_k} rows{rows}"
                    )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_row = _row_from_raw(raw)
                raw_row["hardware"] = raw.get("hardware", {})
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
    source_hash = _hash_files([Path(__file__).resolve(), VULKAN_HARNESS, VULKAN_SHADER])
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
                harness_command = [str(exe), "--spirv", str(spirv), *_harness_args(args, raw_path, rows)]
                completed = _run_command(harness_command, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Vulkan sampler argmax run failed for wg{workgroup} top_k{top_k} rows{rows}"
                    )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_row = _row_from_raw(raw)
                raw_row["hardware"] = raw.get("hardware", {})
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


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row.get("top_k", 1)), int(row["rows"]), int(row["workgroup_size"])


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    hip_ref: str | None = None,
    vulkan_ref: str | None = None,
    out_ref: str | None = None,
) -> dict[str, Any]:
    hip_rows = {
        _row_key(row): row
        for row in hip_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    vulkan_rows = {
        _row_key(row): row
        for row in vulkan_result.get("measurements", {}).get("rows", [])
        if isinstance(row, dict)
    }
    matched = []
    for key in sorted(set(hip_rows) & set(vulkan_rows)):
        hip = hip_rows[key]
        vulkan = vulkan_rows[key]
        hip_us = float(hip["median_us"])
        vulkan_us = float(vulkan["median_us"])
        matched.append(
            {
                "top_k": key[0],
                "rows": key[1],
                "workgroup_size": key[2],
                "vocab": hip.get("vocab"),
                "hip_median_us": hip_us,
                "vulkan_median_us": vulkan_us,
                "vulkan_vs_hip_speedup": hip_us / vulkan_us if vulkan_us > 0 else None,
                "hip_bandwidth_gbps": hip.get("bandwidth_gbps"),
                "vulkan_bandwidth_gbps": vulkan.get("bandwidth_gbps"),
                "hip_correctness_pass": hip.get("correctness_pass"),
                "vulkan_correctness_pass": vulkan.get("correctness_pass"),
                "hip_instruction_count": hip.get("instruction_count"),
                "vulkan_instruction_count": vulkan.get("instruction_count"),
                "hip_waitcnt_count": hip.get("waitcnt_count"),
                "vulkan_waitcnt_count": vulkan.get("waitcnt_count"),
                "hip_load_instruction_count": hip.get("load_instruction_count"),
                "vulkan_load_instruction_count": vulkan.get("load_instruction_count"),
                "hip_wave_size": hip.get("wave_size"),
                "vulkan_wave_size": vulkan.get("wave_size"),
                "hip_vgpr": hip.get("vgpr"),
                "hip_sgpr": hip.get("sgpr"),
                "hip_scratch_bytes": hip.get("scratch_bytes"),
                "vulkan_vgpr": vulkan.get("vgpr"),
                "vulkan_sgpr": vulkan.get("sgpr"),
                "vulkan_scratch_bytes": vulkan.get("scratch_bytes"),
                "vulkan_sgpr_spill_count": vulkan.get("sgpr_spill_count"),
                "vulkan_vgpr_spill_count": vulkan.get("vgpr_spill_count"),
                "vulkan_subgroups_per_simd": vulkan.get("subgroups_per_simd"),
                "vulkan_code_size_bytes": vulkan.get("code_size_bytes"),
                "hip_vopd_count": hip.get("vopd_count"),
                "vulkan_vopd_count": vulkan.get("vopd_count"),
                "hip_barrier_count": hip.get("isa_barrier_count"),
                "vulkan_barrier_count": vulkan.get("isa_barrier_count"),
            }
        )
    best = max(matched, key=lambda row: float(row["vulkan_vs_hip_speedup"])) if matched else {}
    return _json_safe(
        {
            "schema_version": 1,
            "kind": "hipengine_micro_comparison",
            "bench": BENCH_NAME,
            "classification": "diagnostic_unclassified",
            "source": hip_result.get("source", {}),
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
            "matched_rows": matched,
            "summary": {
                "best_vulkan_vs_hip_speedup": best.get("vulkan_vs_hip_speedup"),
                "best_row": best,
                "all_correct": bool(matched)
                and all(row.get("hip_correctness_pass") and row.get("vulkan_correctness_pass") for row in matched),
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
    if args.vocab <= 0 or args.reps <= 0 or args.samples <= 0 or args.debug_vocab <= 0:
        parser.error("--vocab, --reps, --samples, and --debug-vocab must be positive")
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
