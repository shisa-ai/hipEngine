#!/usr/bin/env python3
"""Run or compare paired HIP/Vulkan VOPD scheduling microbenchmarks."""

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
HIP_HARNESS = MICRO_ROOT / "runners" / "hip_vopd_sweep.hip"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_vopd_sweep.cpp"
VULKAN_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "vopd_sweep.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
ISA_STATS = MICRO_ROOT / "runners" / "isa_stats.py"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
HIP_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_hip.hpp"
VULKAN_TIMING_HEADER = MICRO_ROOT / "runners" / "micro_timing_vulkan.hpp"
BENCH_NAME = "f32_vopd_scheduling"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-vopd-build")
DEFAULT_VARIANTS = (
    "independent_fma:2,independent_fma:4,independent_fma:8,"
    "dependent_fma:4,mixed_int_float:4,dequant_like:4"
)

MODE_IDS = {
    "independent_fma": 0,
    "dependent_fma": 1,
    "mixed_int_float": 2,
    "dequant_like": 3,
}


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


def parse_variants(text: str) -> list[dict[str, Any]]:
    variants = []
    for item in text.split(","):
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"variant must be mode:accums: {item}")
        mode, accums_text = item.split(":", 1)
        if mode not in MODE_IDS:
            raise ValueError(f"unknown mode: {mode}")
        accums = int(accums_text)
        if accums <= 0 or accums > 8:
            raise ValueError(f"accums must be in [1, 8]: {item}")
        if mode in ("mixed_int_float", "dequant_like") and accums > 4:
            raise ValueError(f"{mode} currently supports accums <= 4")
        variants.append({"mode": mode, "mode_id": MODE_IDS[mode], "accums": accums})
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


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


def _variant_name(variant: dict[str, Any]) -> str:
    return f"{variant['mode']}_a{variant['accums']}"


def _compile_defines(variant: dict[str, Any], block_size: int) -> list[str]:
    return [
        f"-DHIPENGINE_VOPD_MODE={variant['mode_id']}",
        f"-DHIPENGINE_VOPD_ACCUMS={variant['accums']}",
        f"-DHIPENGINE_BLOCK_SIZE={block_size}",
    ]


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


def _find_single(paths: list[Path], suffix: str) -> Path:
    matches = [path for path in paths if path.name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"could not find build artifact ending with {suffix}")
    return sorted(matches)[0]


def _row_from_raw(raw: dict[str, Any], *, backend: str) -> dict[str, Any]:
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    if not rows:
        raise RuntimeError("raw VOPD harness JSON has no rows")
    row = dict(rows[0])
    timing = _load_module(TIMING_CONTRACT, "micro_timing_contract_for_vopd")
    raw_timing = row.pop("timing_raw")
    gpu_supported = bool(row.pop("gpu_timestamps_supported", True))

    def control(name: str) -> dict[str, Any]:
        values = raw_timing[name]
        return timing.make_timing_control(
            logical_iterations=int(values["logical_iterations"]),
            dispatches_per_iteration=int(values["dispatches_per_iteration"]),
            gpu_samples_us=values["gpu_samples_us"] if gpu_supported else None,
            host_samples_us=values["host_samples_us"],
            gpu_clock="hip_event" if backend == "hip" else "vulkan_timestamp",
            gpu_status="ok" if gpu_supported else "unsupported",
        )

    mode = str(row["timing_mode"])
    passed = bool(row.get("correctness_pass")) and bool(
        row.pop("timed_sequence_correctness_pass")
    ) and bool(row.pop("synchronization_pass"))
    repetitions = int(raw_timing["burst"]["logical_iterations"])
    row.update(
        timing.make_timed_row_contract(
            timing_mode=mode,
            backend=backend,
            repetitions=repetitions,
            dispatches_per_iteration=1,
            dependency_validation_status="pass" if passed else "fail",
            submission=timing.make_submission(
                strategy=(
                    "multi_stream"
                    if backend == "hip" and mode == "independent_throughput"
                    else "direct"
                    if backend == "hip"
                    else "vulkan_command_buffer"
                ),
                queue_or_stream_count=int(row.pop("queue_or_stream_count")),
                recording_in_timed_region=False,
            ),
            single_timing=control("single"),
            burst_timing=control("burst"),
            correctness=timing.make_correctness(
                status="pass" if passed else "fail",
                oracle="sampled CPU reference for the timed sequence",
                logical_iterations=repetitions,
                coverage=(
                    "all_dispatches" if mode == "independent_throughput" else "chained_final_state"
                ),
                synchronization_method=(
                    "disjoint_outputs"
                    if mode == "independent_throughput"
                    else "ordered_stream"
                    if backend == "hip"
                    else "compute_barriers"
                ),
                barrier_count=int(row.pop("barrier_count")),
            ),
        )
    )
    return row


def _compile_hip_variant(
    build_dir: Path,
    variant: dict[str, Any],
    gfx_arch: str | None,
    block_size: int,
) -> tuple[Path, Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc is not available")
    command = [hipcc]
    if gfx_arch:
        command.append(f"--offload-arch={gfx_arch}")
    command.extend(
        [
            "-O3",
            "-std=c++17",
            "--save-temps",
            *_compile_defines(variant, block_size),
            str(HIP_HARNESS),
            "-o",
            "hip_vopd_sweep",
        ]
    )
    completed = _run_command(command, cwd=build_dir)
    if completed.returncode != 0:
        raise RuntimeError(f"HIP VOPD build failed for {_variant_name(variant)}")
    artifacts = [path for path in build_dir.iterdir() if path.is_file()]
    arch_tag = gfx_arch or "gfx"
    obj = _find_single(artifacts, f"{arch_tag}.o") if gfx_arch else _find_single(artifacts, ".o")
    return build_dir / "hip_vopd_sweep", obj, command


def _compile_vulkan_variant(
    build_dir: Path,
    variant: dict[str, Any],
    block_size: int,
) -> tuple[Path, Path, list[str], list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    spirv = build_dir / "vopd_sweep.spv"
    if glslc:
        shader_command = [
            glslc,
            "-O",
            *_compile_defines(variant, block_size),
            str(VULKAN_SHADER),
            "-o",
            str(spirv),
        ]
    elif glslang:
        shader_command = [
            glslang,
            "-V",
            *_compile_defines(variant, block_size),
            str(VULKAN_SHADER),
            "-o",
            str(spirv),
        ]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(shader_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan VOPD shader build failed for {_variant_name(variant)}")

    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    exe = build_dir / "vulkan_vopd_sweep"
    build_command = [
        compiler,
        "-O2",
        "-std=c++17",
        *_compile_defines(variant, block_size),
        str(VULKAN_HARNESS),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(build_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan VOPD harness build failed for {_variant_name(variant)}")
    return spirv, exe, shader_command, build_command


def _harness_args(args: argparse.Namespace, raw_path: Path, *, backend: str) -> list[str]:
    command = [
        "--json",
        str(raw_path),
        "--n",
        str(args.n),
        "--body-iters",
        str(args.body_iters),
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


def _hip_isa(obj: Path) -> dict[str, Any]:
    isa = _load_module(ISA_STATS, "micro_isa_stats_for_vopd")
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
) -> tuple[dict[str, Any], list[str], int]:
    isa = _load_module(ISA_STATS, "micro_isa_stats_for_vopd")
    raw_path = args.build_dir / "vulkan" / "debug_raw.json"
    command = [
        str(exe),
        "--spirv",
        str(spirv),
        "--json",
        str(raw_path),
        "--n",
        str(args.debug_n),
        "--body-iters",
        str(args.debug_body_iters),
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
        raise RuntimeError("Vulkan RADV_DEBUG=shaders,shaderstats VOPD run failed")
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


def _normalize_result(
    *,
    backend: str,
    raw_rows: list[dict[str, Any]],
    isa_rows: list[dict[str, Any]],
    environment: dict[str, Any],
    source_hash: str,
    wrapper_command: list[str],
    commands: list[dict[str, Any]],
    hardware_gpu: str | None,
    gfx_arch: str | None,
    environment_ref: str | None,
) -> dict[str, Any]:
    if not raw_rows or len(raw_rows) != len(isa_rows) or len(raw_rows) != len(commands):
        raise ValueError("VOPD result rows, ISA rows, and requested commands must align")
    rows = []
    for raw, isa in zip(raw_rows, isa_rows, strict=True):
        config = raw.get("raw_config")
        if not isinstance(config, dict):
            raise ValueError("VOPD raw row is missing its requested config")
        row = {
            **{key: value for key, value in raw.items() if key != "raw_config"},
            **{f"isa_{k}": v for k, v in isa.items()},
        }
        for field in ("mode", "accums", "n", "body_iters", "block_size", "timing_mode"):
            if config.get(field) != raw.get(field):
                raise ValueError(f"VOPD raw row disagrees with config field {field}")
        if int(config.get("reps", 0)) != int(row["timing"]["burst"]["logical_iterations"]):
            raise ValueError("VOPD raw row repetitions disagree with config")
        row["mode"] = raw.get("mode")
        row["accums"] = raw.get("accums")
        row["vopd_count"] = isa.get("vopd_count")
        row["vopd_op_count"] = isa.get("vopd_op_count")
        row["instruction_count"] = isa.get("instruction_count")
        row["waitcnt_count"] = isa.get("waitcnt_count")
        row["dot4_count"] = isa.get("dot4_count")
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
        if isa.get("estimated_vgpr_span") is not None:
            row["estimated_vgpr_span"] = isa.get("estimated_vgpr_span")
        if isa.get("estimated_sgpr_span") is not None:
            row["estimated_sgpr_span"] = isa.get("estimated_sgpr_span")
        rows.append(row)

    variants = sorted({(str(row["mode"]), int(row["accums"])) for row in rows})
    workgroups = sorted({int(row["block_size"]) for row in rows})
    n_values = {int(row["n"]) for row in rows}
    body_iters_values = {int(row["body_iters"]) for row in rows}
    timing_modes = {str(row["timing_mode"]) for row in rows}
    repetitions = {int(raw["raw_config"]["reps"]) for raw in raw_rows}
    warmups = {int(raw["raw_config"]["warmup"]) for raw in raw_rows}
    sample_counts = {int(raw["raw_config"]["samples"]) for raw in raw_rows}
    if any(len(values) != 1 for values in (n_values, body_iters_values, timing_modes, repetitions, warmups, sample_counts)):
        raise ValueError("VOPD requested rows do not share one workload contract")
    n = next(iter(n_values))
    body_iters = next(iter(body_iters_values))
    timing_mode = next(iter(timing_modes))
    expected_keys = {
        (mode, accums, workgroup, n, body_iters, timing_mode)
        for mode, accums in variants
        for workgroup in workgroups
    }
    row_keys = [_row_key(row) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("VOPD result contains duplicate requested rows")
    if set(row_keys) != expected_keys:
        raise ValueError(
            "VOPD result does not contain the complete requested matrix: "
            f"expected {len(expected_keys)}, got {len(row_keys)}"
        )
    primary = rows[0] if rows else {}
    correctness_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    raw0 = {"hardware": {}} if not raw_rows else {"hardware": raw_rows[0].get("hardware", {})}
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
            "benchmark_family": "vopd_scheduling",
            "variants": [
                {"mode": mode, "accums": accums} for mode, accums in variants
            ],
            "n": n,
            "body_iters": body_iters,
            "workgroup_sizes": workgroups,
            "timing_mode": timing_mode,
            "repetitions": next(iter(repetitions)),
            "warmup_logical_iterations": next(iter(warmups)),
            "samples": next(iter(sample_counts)),
            "expected_row_count": len(expected_keys),
            "commands": commands,
        },
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "sampled CPU reference for first 64 output elements",
            "max_abs": max((float(row.get("max_abs", 0.0)) for row in rows), default=0.0),
            "max_rel": max((float(row.get("max_rel", 0.0)) for row in rows), default=0.0),
        },
        "isa": primary,
        "classification": "diagnostic_unclassified",
        "measurements": {"rows": rows},
        "notes": (
            "Pure VALU VOPD scheduling diagnostic. Independent modes create "
            "dual-issue opportunities; dependent modes intentionally suppress them."
        ),
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return _json_safe(result)


def _run_hip(args: argparse.Namespace, variants: list[dict[str, Any]]) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files(
        [Path(__file__).resolve(), HIP_HARNESS, HIP_TIMING_HEADER, TIMING_CONTRACT]
    )
    raw_rows: list[dict[str, Any]] = []
    isa_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for variant in variants:
        for block_size in args.workgroup_sizes:
            variant_dir = args.build_dir / "hip" / _variant_name(variant) / f"wg{block_size}"
            exe, obj, build_command = _compile_hip_variant(
                variant_dir, variant, args.gfx_arch, block_size
            )
            raw_path = variant_dir / "raw.json"
            harness_command = [str(exe), *_harness_args(args, raw_path, backend="hip")]
            completed = _run_command(harness_command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                raise RuntimeError(f"HIP VOPD run failed for {_variant_name(variant)}")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_row = _row_from_raw(raw, backend="hip")
            raw_row["hardware"] = raw.get("hardware", {})
            raw_row["raw_config"] = raw.get("config", {})
            raw_rows.append(raw_row)
            isa_rows.append(_hip_isa(obj))
            commands.append(
                {
                    "variant": variant,
                    "workgroup_size": block_size,
                    "build_command": build_command,
                    "harness_command": harness_command,
                    "object_path": str(obj),
                    "raw_json_retained": False,
                }
            )
    return _normalize_result(
        backend="hip",
        raw_rows=raw_rows,
        isa_rows=isa_rows,
        environment=environment,
        source_hash=source_hash,
        wrapper_command=sys.argv.copy(),
        commands=commands,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
    )


def _run_vulkan(args: argparse.Namespace, variants: list[dict[str, Any]]) -> dict[str, Any]:
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
    isa_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for variant in variants:
        for block_size in args.workgroup_sizes:
            variant_dir = args.build_dir / "vulkan" / _variant_name(variant) / f"wg{block_size}"
            spirv, exe, shader_command, build_command = _compile_vulkan_variant(
                variant_dir, variant, block_size
            )
            raw_path = variant_dir / "raw.json"
            harness_command = [
                str(exe),
                "--spirv",
                str(spirv),
                *_harness_args(args, raw_path, backend="vulkan"),
            ]
            completed = _run_command(harness_command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                raise RuntimeError(f"Vulkan VOPD run failed for {_variant_name(variant)}")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_row = _row_from_raw(raw, backend="vulkan")
            raw_row["hardware"] = raw.get("hardware", {})
            raw_row["raw_config"] = raw.get("config", {})
            raw_rows.append(raw_row)
            isa_row, debug_command, shader_dump_bytes = _vulkan_isa(exe, spirv, args)
            isa_rows.append(isa_row)
            commands.append(
                {
                    "variant": variant,
                    "workgroup_size": block_size,
                    "shader_command": shader_command,
                    "build_command": build_command,
                    "harness_command": harness_command,
                    "debug_command": debug_command,
                    "debug_env": {"RADV_DEBUG": "shaders,shaderstats"},
                    "shader_dump_bytes": shader_dump_bytes,
                    "raw_json_retained": False,
                }
            )
    return _normalize_result(
        backend="vulkan",
        raw_rows=raw_rows,
        isa_rows=isa_rows,
        environment=environment,
        source_hash=source_hash,
        wrapper_command=sys.argv.copy(),
        commands=commands,
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, int, int, str]:
    return (
        str(row["mode"]),
        int(row["accums"]),
        int(row["block_size"]),
        int(row["n"]),
        int(row["body_iters"]),
        str(row["timing_mode"]),
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
            raise ValueError("VOPD comparison requires v2 timing-contract results")
        if result.get("kind") != "hipengine_micro_result":
            raise ValueError(f"{label} VOPD input must be a micro result")
        if result.get("bench") != BENCH_NAME:
            raise ValueError(f"{label} VOPD input bench does not match {BENCH_NAME}")
        if result.get("backend") != backend:
            raise ValueError("VOPD comparison inputs must be HIP then Vulkan")
        if result.get("classification") != "diagnostic_unclassified":
            raise ValueError(f"{label} VOPD input classification is invalid")
    hip_arch = str(hip_result.get("hardware", {}).get("gfx_arch", ""))
    vulkan_arch = str(vulkan_result.get("hardware", {}).get("gfx_arch", ""))
    if not hip_arch or hip_arch == "unknown" or hip_arch != vulkan_arch:
        raise ValueError("HIP and Vulkan VOPD gfx architectures do not match")
    hip_device = _device_fingerprint(hip_result.get("hardware", {}).get("gpu_name", ""))
    vulkan_device = _device_fingerprint(
        vulkan_result.get("hardware", {}).get("gpu_name", "")
    )
    if not hip_device or hip_device == "unknown" or hip_device != vulkan_device:
        raise ValueError("HIP and Vulkan VOPD device identities do not match")
    hip_source = hip_result.get("source", {})
    vulkan_source = vulkan_result.get("source", {})
    if not isinstance(hip_source, dict) or not isinstance(vulkan_source, dict):
        raise ValueError("HIP and Vulkan VOPD source provenance must be objects")
    for field in ("repo", "branch", "commit", "dirty"):
        if field not in hip_source or field not in vulkan_source:
            raise ValueError(f"HIP and Vulkan VOPD source {field} is required")
        if hip_source[field] != vulkan_source[field]:
            raise ValueError(f"HIP and Vulkan VOPD source {field} values do not match")
    if not str(hip_source.get("repo", "")) or not str(hip_source.get("commit", "")):
        raise ValueError("HIP and Vulkan VOPD source repo/commit must not be empty")
    if not isinstance(hip_source.get("dirty"), bool):
        raise ValueError("HIP and Vulkan VOPD source dirty must be boolean")
    for label, source in (("HIP", hip_source), ("Vulkan", vulkan_source)):
        if not str(source.get("source_hash", "")):
            raise ValueError(f"{label} VOPD source hash must not be empty")


def _expected_row_keys(parameters: dict[str, Any]) -> set[tuple[str, int, int, int, int, str]]:
    variants = parameters.get("variants")
    workgroups = parameters.get("workgroup_sizes")
    if not isinstance(variants, list) or not variants or not isinstance(workgroups, list) or not workgroups:
        raise ValueError("VOPD parameters do not describe the requested matrix")
    try:
        variant_keys = [(str(variant["mode"]), int(variant["accums"])) for variant in variants]
        workgroup_values = [int(workgroup) for workgroup in workgroups]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("VOPD parameters contain an invalid requested matrix") from exc
    if len(set(variant_keys)) != len(variant_keys) or len(set(workgroup_values)) != len(
        workgroup_values
    ):
        raise ValueError("VOPD parameters contain duplicate requested rows")
    return {
        (
            mode,
            accums,
            workgroup,
            int(parameters["n"]),
            int(parameters["body_iters"]),
            str(parameters["timing_mode"]),
        )
        for mode, accums in variant_keys
        for workgroup in workgroup_values
    }


def _comparison_domain(
    timing: Any,
    hip: dict[str, Any],
    vulkan: dict[str, Any],
    *,
    control: str,
    domain: str,
) -> dict[str, Any]:
    try:
        return {
            "status": "ok",
            **timing.comparison_ratio(hip, vulkan, control=control, domain=domain),
        }
    except ValueError as exc:
        status = (
            "not_comparable_submission_contract"
            if "submission contracts" in str(exc)
            else "not_comparable"
        )
        return {
            "status": status,
            "reason": str(exc),
            "hip": hip["timing"][control][domain],
            "vulkan": vulkan["timing"][control][domain],
        }


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    _validate_comparison_inputs(hip_result, vulkan_result)
    hip_parameters = hip_result.get("parameters", {})
    vulkan_parameters = vulkan_result.get("parameters", {})
    for field in (
        "variants",
        "n",
        "body_iters",
        "workgroup_sizes",
        "timing_mode",
        "repetitions",
        "warmup_logical_iterations",
        "samples",
        "expected_row_count",
    ):
        if field not in hip_parameters or field not in vulkan_parameters:
            raise ValueError(f"HIP and Vulkan VOPD parameter {field} is required")
        if hip_parameters[field] != vulkan_parameters[field]:
            if field == "timing_mode":
                raise ValueError("HIP and Vulkan VOPD timing modes do not match (timing_mode)")
            raise ValueError(f"HIP and Vulkan VOPD parameter {field} values do not match")
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
        raise ValueError("VOPD comparison requires non-empty object rows")
    hip_modes = {row.get("timing_mode") for row in hip_input_rows if isinstance(row, dict)}
    vulkan_modes = {row.get("timing_mode") for row in vulkan_input_rows if isinstance(row, dict)}
    if hip_modes != vulkan_modes or None in hip_modes:
        raise ValueError("HIP and Vulkan timing modes are missing or do not match")
    hip_rows = {_row_key(row): row for row in hip_input_rows}
    vulkan_rows = {_row_key(row): row for row in vulkan_input_rows}
    if len(hip_rows) != len(hip_input_rows) or len(vulkan_rows) != len(vulkan_input_rows):
        raise ValueError("VOPD comparison inputs contain duplicate rows")
    expected_rows = _expected_row_keys(hip_parameters)
    if int(hip_parameters["expected_row_count"]) != len(expected_rows):
        raise ValueError("VOPD expected row count does not match its matrix")
    if set(hip_rows) != expected_rows or set(vulkan_rows) != expected_rows:
        raise ValueError(
            "HIP and Vulkan VOPD results must contain the exact requested "
            f"{len(expected_rows)}-row matrix"
        )
    timing = _load_module(TIMING_CONTRACT, "micro_timing_contract_for_vopd_compare")
    matched = []
    for key in sorted(hip_rows):
        hip = hip_rows[key]
        vulkan = vulkan_rows[key]
        timing.validate_timed_row(hip, expected_repetitions=int(hip_parameters["repetitions"]))
        timing.validate_timed_row(vulkan, expected_repetitions=int(hip_parameters["repetitions"]))
        for control in ("single", "burst"):
            matched.append({
                "mode": key[0],
                "accums": key[1],
                "workgroup_size": key[2],
                "n": key[3],
                "body_iters": key[4],
                "timing_mode": key[5],
                "control": control,
                "gpu_elapsed": _comparison_domain(
                    timing, hip, vulkan, control=control, domain="gpu_elapsed"
                ),
                "host_wall": _comparison_domain(
                    timing, hip, vulkan, control=control, domain="host_wall"
                ),
                "hip_gops": hip.get("gops"),
                "vulkan_gops": vulkan.get("gops"),
                "hip_correctness_pass": hip.get("correctness_pass"),
                "vulkan_correctness_pass": vulkan.get("correctness_pass"),
                "hip_vopd_count": hip.get("vopd_count"),
                "hip_vopd_op_count": hip.get("vopd_op_count"),
                "vulkan_vopd_count": vulkan.get("vopd_count"),
                "vulkan_vopd_op_count": vulkan.get("vopd_op_count"),
                "hip_instruction_count": hip.get("instruction_count"),
                "vulkan_instruction_count": vulkan.get("instruction_count"),
                "hip_waitcnt_count": hip.get("waitcnt_count"),
                "vulkan_waitcnt_count": vulkan.get("waitcnt_count"),
                "hip_wave_size": hip.get("wave_size"),
                "vulkan_wave_size": vulkan.get("wave_size"),
                "hip_vgpr": hip.get("vgpr"),
                "hip_sgpr": hip.get("sgpr"),
                "vulkan_vgpr": vulkan.get("vgpr"),
                "vulkan_sgpr": vulkan.get("sgpr"),
                "vulkan_scratch_bytes": vulkan.get("scratch_bytes"),
                "vulkan_sgpr_spill_count": vulkan.get("sgpr_spill_count"),
                "vulkan_vgpr_spill_count": vulkan.get("vgpr_spill_count"),
                "vulkan_subgroups_per_simd": vulkan.get("subgroups_per_simd"),
                "vulkan_code_size_bytes": vulkan.get("code_size_bytes"),
                "vulkan_estimated_vgpr_span": vulkan.get("estimated_vgpr_span"),
                "vulkan_estimated_sgpr_span": vulkan.get("estimated_sgpr_span"),
            })
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
                "commit_match": True,
                "dirty": dirty,
                "gfx_arch_match": True,
                "device_match": True,
                "source_hashes_present": True,
                "performance_claim": performance_claim,
                "blocking_reasons": blocking_reasons,
            },
            "comparisons": matched,
            "matched_rows": matched,
            "interpretation": (
                "Targeted VOPD scheduling diagnostic. A dual-issue compiler claim "
                "requires the independent rows to show more useful VOPD or equivalent "
                "paired VALU scheduling at matched occupancy and no corresponding win "
                "on dependent rows."
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
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--n", type=int, default=65536)
    parser.add_argument("--body-iters", type=int, default=2048)
    parser.add_argument("--workgroups", default="64,128,256")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--timing-mode",
        choices=["serial_latency", "independent_throughput"],
        default="serial_latency",
    )
    parser.add_argument("--independent-streams", type=int, default=4)
    parser.add_argument("--debug-n", type=int, default=1024)
    parser.add_argument("--debug-body-iters", type=int, default=64)
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
    if min(
        args.n,
        args.body_iters,
        args.reps,
        args.warmup + 1,
        args.samples,
        args.independent_streams,
    ) <= 0:
        parser.error("--n, --body-iters, --reps, and --samples must be positive")
    if args.debug_n <= 0 or args.debug_body_iters <= 0:
        parser.error("--debug-n and --debug-body-iters must be positive")
    try:
        args.variant_specs = parse_variants(args.variants)
        args.workgroup_sizes = [int(value) for value in args.workgroups.split(",") if value]
        if not args.workgroup_sizes or any(value not in (64, 128, 256) for value in args.workgroup_sizes):
            raise ValueError("workgroups must be a comma-separated subset of 64,128,256")
    except ValueError as exc:
        parser.error(str(exc))
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
    elif args.backend == "hip":
        result = _run_hip(args, args.variant_specs)
    else:
        result = _run_vulkan(args, args.variant_specs)

    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
