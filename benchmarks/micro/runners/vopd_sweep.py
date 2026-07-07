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


def _compile_defines(variant: dict[str, Any]) -> list[str]:
    return [
        f"-DHIPENGINE_VOPD_MODE={variant['mode_id']}",
        f"-DHIPENGINE_VOPD_ACCUMS={variant['accums']}",
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


def _row_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    if not rows:
        raise RuntimeError("raw VOPD harness JSON has no rows")
    return dict(rows[0])


def _compile_hip_variant(
    build_dir: Path,
    variant: dict[str, Any],
    gfx_arch: str | None,
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
            *_compile_defines(variant),
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
) -> tuple[Path, Path, list[str], list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    spirv = build_dir / "vopd_sweep.spv"
    if glslc:
        shader_command = [
            glslc,
            "-O",
            *_compile_defines(variant),
            str(VULKAN_SHADER),
            "-o",
            str(spirv),
        ]
    elif glslang:
        shader_command = [
            glslang,
            "-V",
            *_compile_defines(variant),
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
        *_compile_defines(variant),
        str(VULKAN_HARNESS),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(build_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Vulkan VOPD harness build failed for {_variant_name(variant)}")
    return spirv, exe, shader_command, build_command


def _harness_args(args: argparse.Namespace, raw_path: Path) -> list[str]:
    return [
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
        "--device-index",
        str(args.device_index),
    ]


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
    rows = []
    for raw, isa in zip(raw_rows, isa_rows, strict=True):
        row = {**raw, **{f"isa_{k}": v for k, v in isa.items()}}
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
        ):
            if isa.get(key) is not None:
                row[key] = isa.get(key)
        if isa.get("estimated_vgpr_span") is not None:
            row["estimated_vgpr_span"] = isa.get("estimated_vgpr_span")
        if isa.get("estimated_sgpr_span") is not None:
            row["estimated_sgpr_span"] = isa.get("estimated_sgpr_span")
        rows.append(row)

    primary = rows[0] if rows else {}
    correctness_pass = bool(rows) and all(bool(row.get("correctness_pass")) for row in rows)
    raw0 = {"hardware": {}} if not raw_rows else {"hardware": raw_rows[0].get("hardware", {})}
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
            "benchmark_family": "vopd_scheduling",
            "variants": [
                {"mode": row.get("mode"), "accums": row.get("accums")} for row in rows
            ],
            "n": primary.get("n"),
            "body_iters": primary.get("body_iters"),
            "block_size": primary.get("block_size"),
            "commands": commands,
        },
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "sampled CPU reference for first 64 output elements",
            "max_abs": max((float(row.get("max_abs", 0.0)) for row in rows), default=0.0),
            "max_rel": max((float(row.get("max_rel", 0.0)) for row in rows), default=0.0),
        },
        "timing": {
            "unit": "us_per_dispatch",
            "median": primary.get("median_us"),
            "primary": primary,
            "summary": {
                "best_gops": max((float(row.get("gops", 0.0)) for row in rows), default=0.0),
                "row_count": len(rows),
            },
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
    source_hash = _hash_files([Path(__file__).resolve(), HIP_HARNESS])
    raw_rows: list[dict[str, Any]] = []
    isa_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = args.build_dir / "hip" / _variant_name(variant)
        exe, obj, build_command = _compile_hip_variant(variant_dir, variant, args.gfx_arch)
        raw_path = variant_dir / "raw.json"
        harness_command = [str(exe), *_harness_args(args, raw_path)]
        completed = _run_command(harness_command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"HIP VOPD run failed for {_variant_name(variant)}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_row = _row_from_raw(raw)
        raw_row["hardware"] = raw.get("hardware", {})
        raw_rows.append(raw_row)
        isa_rows.append(_hip_isa(obj))
        commands.append(
            {
                "variant": variant,
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
    source_hash = _hash_files([Path(__file__).resolve(), VULKAN_HARNESS, VULKAN_SHADER])
    raw_rows: list[dict[str, Any]] = []
    isa_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = args.build_dir / "vulkan" / _variant_name(variant)
        spirv, exe, shader_command, build_command = _compile_vulkan_variant(variant_dir, variant)
        raw_path = variant_dir / "raw.json"
        harness_command = [str(exe), "--spirv", str(spirv), *_harness_args(args, raw_path)]
        completed = _run_command(harness_command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"Vulkan VOPD run failed for {_variant_name(variant)}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_row = _row_from_raw(raw)
        raw_row["hardware"] = raw.get("hardware", {})
        raw_rows.append(raw_row)
        isa_row, debug_command, shader_dump_bytes = _vulkan_isa(exe, spirv, args)
        isa_rows.append(isa_row)
        commands.append(
            {
                "variant": variant,
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


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["mode"]), int(row["accums"])


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
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
                "mode": key[0],
                "accums": key[1],
                "hip_median_us": hip_us,
                "vulkan_median_us": vulkan_us,
                "vulkan_vs_hip_speedup": hip_us / vulkan_us if vulkan_us > 0 else None,
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
            }
        )
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
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
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
    if min(args.n, args.body_iters, args.reps, args.warmup + 1, args.samples) <= 0:
        parser.error("--n, --body-iters, --reps, and --samples must be positive")
    if args.debug_n <= 0 or args.debug_body_iters <= 0:
        parser.error("--debug-n and --debug-body-iters must be positive")
    try:
        args.variant_specs = parse_variants(args.variants)
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
