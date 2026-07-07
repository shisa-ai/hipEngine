#!/usr/bin/env python3
"""Extract HIP/LLVM and Vulkan/RADV ISA stats for geometry microbench kernels."""

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
HIP_HARNESS = MICRO_ROOT / "runners" / "hip_geometry_sweep.hip"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_geometry_sweep.cpp"
VULKAN_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "geometry_sweep.comp"
COLLECT_ENV = MICRO_ROOT / "collect_env.py"
BENCH_NAME = "f32_gemv_geometry_isa_stats"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-isa-stats-build")


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


def _infer_gfx_arch(environment: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    devices = environment.get("devices")
    if isinstance(devices, dict):
        for key in ("rocminfo_name_gfx_lines", "vulkan_summary_lines", "lspci_display_lines"):
            lines = devices.get(key)
            if isinstance(lines, list):
                found = _find_gfx_arch("\n".join(str(line) for line in lines))
                if found:
                    return found
    return "unknown"


def _infer_gpu_name(environment: dict[str, Any], override: str | None) -> str:
    if override:
        return override
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


def _parse_int_field(text: str, name: str) -> int | None:
    escaped = re.escape(name)
    match = re.search(rf"{escaped}\s*:\s*([0-9]+)", text)
    return int(match.group(1)) if match else None


def parse_hip_metadata(readobj_notes: str) -> dict[str, Any]:
    return {
        "kernel_name": (
            re.search(r"\.name:\s+(\S+)", readobj_notes).group(1)
            if re.search(r"\.name:\s+(\S+)", readobj_notes)
            else None
        ),
        "target": (
            re.search(r"amdhsa\.target:\s+(\S+)", readobj_notes).group(1)
            if re.search(r"amdhsa\.target:\s+(\S+)", readobj_notes)
            else None
        ),
        "sgpr": _parse_int_field(readobj_notes, ".sgpr_count"),
        "vgpr": _parse_int_field(readobj_notes, ".vgpr_count"),
        "scratch_bytes": _parse_int_field(readobj_notes, ".private_segment_fixed_size"),
        "lds_static_bytes": _parse_int_field(readobj_notes, ".group_segment_fixed_size"),
        "sgpr_spill_count": _parse_int_field(readobj_notes, ".sgpr_spill_count"),
        "vgpr_spill_count": _parse_int_field(readobj_notes, ".vgpr_spill_count"),
        "wave_size": _parse_int_field(readobj_notes, ".wavefront_size"),
    }


def _instruction_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.endswith(":") or stripped.startswith(("/", ";")):
        return None
    if stripped.startswith("."):
        return None
    if "//" in stripped:
        stripped = stripped.split("//", 1)[0].strip()
    if ";" in stripped:
        stripped = stripped.split(";", 1)[0].strip()
    if not stripped:
        return None
    token = stripped.split(None, 1)[0]
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
        return None
    return token


def _estimate_register_spans(disasm: str) -> dict[str, int | None]:
    max_v = -1
    max_s = -1
    for prefix, max_value in (("v", "max_v"), ("s", "max_s")):
        for match in re.finditer(rf"\b{prefix}\[(\d+):(\d+)\]", disasm):
            value = max(int(match.group(1)), int(match.group(2)))
            if max_value == "max_v":
                max_v = max(max_v, value)
            else:
                max_s = max(max_s, value)
        for match in re.finditer(rf"\b{prefix}(\d+)\b", disasm):
            value = int(match.group(1))
            if max_value == "max_v":
                max_v = max(max_v, value)
            else:
                max_s = max(max_s, value)
    return {
        "estimated_vgpr_span": max_v + 1 if max_v >= 0 else None,
        "estimated_sgpr_span": max_s + 1 if max_s >= 0 else None,
    }


def parse_disassembly_stats(disasm: str) -> dict[str, Any]:
    names = [_instruction_name(line) for line in disasm.splitlines()]
    instructions = [name for name in names if name]
    dual_ops = len(re.findall(r"\bv_dual_[A-Za-z0-9_]+", disasm))
    stats: dict[str, Any] = {
        "instruction_count": len(instructions),
        "salu_count": sum(1 for name in instructions if name.startswith("s_")),
        "valu_count": sum(1 for name in instructions if name.startswith("v_")),
        "waitcnt_count": sum(1 for name in instructions if name.startswith("s_waitcnt")),
        "waitcnt_depctr_count": sum(1 for name in instructions if name == "s_waitcnt_depctr"),
        "vopd_count": sum(1 for line in disasm.splitlines() if "v_dual_" in line),
        "vopd_op_count": dual_ops,
        "dot4_count": sum(1 for name in instructions if "dot4" in name),
        "fma_or_fmac_count": sum(
            1 for name in instructions if "fma" in name or "fmac" in name or "ffma" in name
        ),
        "global_load_count": sum(1 for name in instructions if name.startswith("global_load")),
        "global_store_count": sum(1 for name in instructions if name.startswith("global_store")),
        "buffer_load_count": sum(1 for name in instructions if name.startswith("buffer_load")),
        "buffer_store_count": sum(1 for name in instructions if name.startswith("buffer_store")),
        "ds_load_count": sum(1 for name in instructions if name.startswith("ds_load")),
        "ds_store_count": sum(1 for name in instructions if name.startswith("ds_store")),
        "barrier_count": sum(1 for name in instructions if name == "s_barrier"),
        "branch_count": sum(1 for name in instructions if "branch" in name),
        "delay_alu_count": sum(1 for name in instructions if name == "s_delay_alu"),
        "nop_count": sum(1 for name in instructions if name == "s_nop"),
    }
    stats.update(_estimate_register_spans(disasm))
    return stats


_RADV_SHADERSTATS_KEYS = {
    "Driver pipeline hash": "radv_pipeline_hash",
    "SGPRs": "sgpr",
    "VGPRs": "vgpr",
    "Spilled SGPRs": "sgpr_spill_count",
    "Spilled VGPRs": "vgpr_spill_count",
    "Code size": "code_size_bytes",
    "LDS size": "shaderstats_lds_size_bytes",
    "Scratch size": "scratch_bytes",
    "Subgroups per SIMD": "subgroups_per_simd",
    "Combined inputs": "combined_inputs",
    "Combined outputs": "combined_outputs",
    "Hash": "shader_hash",
    "Instructions": "shaderstats_instruction_count",
    "Copies": "copy_count",
    "Branches": "shaderstats_branch_count",
    "Latency": "aco_latency",
    "Inverse Throughput": "aco_inverse_throughput",
    "VMEM Clause": "vmem_clause_count",
    "SMEM Clause": "smem_clause_count",
    "Pre-Sched SGPRs": "presched_sgpr",
    "Pre-Sched VGPRs": "presched_vgpr",
    "VALU": "shaderstats_valu_count",
    "SALU": "shaderstats_salu_count",
    "VMEM": "shaderstats_vmem_count",
    "SMEM": "shaderstats_smem_count",
    "VOPD": "shaderstats_vopd_count",
}


def _parse_shaderstats_value(text: str) -> int | float | str:
    value = text.strip()
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"-?(?:[0-9]*\.)?[0-9]+", value):
        return float(value)
    return value


def parse_radv_shader_stats(dump: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"(?s)\*\*\* SHADER STATS \*\*\*\n(.*?)\n\*{20,}")
    for match in pattern.finditer(dump):
        row: dict[str, Any] = {
            "shaderstats_status": "radv_debug_shaderstats",
            "register_count_status": "official_radv_debug_shaderstats",
        }
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            mapped = _RADV_SHADERSTATS_KEYS.get(key.strip())
            if mapped:
                row[mapped] = _parse_shaderstats_value(value)
        rows.append(row)
    return rows


def split_radv_shader_sections(dump: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^shader:\s+MESA_SHADER_COMPUTE\b", dump)]
    if not starts:
        return []
    sections: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(dump)
        sections.append(dump[start:end])
    return sections


def _parse_header_int(section: str, name: str) -> int | None:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*([0-9]+)", section)
    return int(match.group(1)) if match else None


def _extract_final_disasm(section: str) -> str:
    match = re.search(r"(?s)Compute Shader\s+disasm:\s*\n(.*)", section)
    return match.group(1) if match else ""


def parse_radv_shader_dump(dump: str) -> list[dict[str, Any]]:
    parsed = []
    shaderstats_rows = parse_radv_shader_stats(dump)
    for index, section in enumerate(split_radv_shader_sections(dump)):
        disasm = _extract_final_disasm(section)
        stats = parse_disassembly_stats(disasm)
        workgroup = _parse_header_int(section, "workgroup_size")
        shaderstats = shaderstats_rows[index] if index < len(shaderstats_rows) else {}
        register_status = shaderstats.get(
            "register_count_status",
            (
                "estimated_from_final_disasm_physical_register_span; "
                "RADV_DEBUG=shaderstats did not print official allocation counts"
            ),
        )
        parsed.append(
            {
                "workgroup_size": workgroup,
                "shared_size": _parse_header_int(section, "shared_size"),
                "api_subgroup_size": _parse_header_int(section, "api_subgroup_size"),
                "min_subgroup_size": _parse_header_int(section, "min_subgroup_size"),
                "max_subgroup_size": _parse_header_int(section, "max_subgroup_size"),
                "has_after_ra": "After RA:" in section,
                "has_final_disasm": bool(disasm.strip()),
                **stats,
                **shaderstats,
                "register_count_status": register_status,
            }
        )
    return parsed


def _read_text_command(command: list[str], *, cwd: Path) -> str:
    completed = _run_command(command, cwd=cwd, echo=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed.stdout + completed.stderr


def _find_single(paths: list[Path], suffix: str) -> Path:
    matches = [path for path in paths if path.name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"could not find save-temps artifact ending with {suffix}")
    return sorted(matches)[0]


def _hip_wavefront_flags(wavefront_size: str) -> list[str]:
    if wavefront_size == "64":
        return ["-mwavefrontsize64"]
    if wavefront_size == "32":
        return ["-mno-wavefrontsize64"]
    return []


def _compile_hip_save_temps(
    build_dir: Path,
    gfx_arch: str | None,
    *,
    hip_wavefront_size: str = "default",
    fixed_workgroup_size: int | None = None,
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
            *_hip_wavefront_flags(hip_wavefront_size),
            "-O3",
            "-std=c++17",
            "--save-temps",
            *(
                [f"-DHIPENGINE_FIXED_WORKGROUP_SIZE={fixed_workgroup_size}"]
                if fixed_workgroup_size
                else []
            ),
            str(HIP_HARNESS),
            "-o",
            "hip_geometry_sweep",
        ]
    )
    completed = _run_command(command, cwd=build_dir)
    if completed.returncode != 0:
        raise RuntimeError("HIP save-temps build failed")
    artifacts = [path for path in build_dir.iterdir() if path.is_file()]
    arch_tag = gfx_arch or "gfx"
    obj = _find_single(artifacts, f"{arch_tag}.o") if gfx_arch else _find_single(artifacts, ".o")
    asm = _find_single(artifacts, f"{arch_tag}.s") if gfx_arch else _find_single(artifacts, ".s")
    return obj, asm, command


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


def _compile_vulkan(build_dir: Path) -> tuple[Path, Path, list[str], list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    spirv = build_dir / "geometry_sweep.spv"
    if glslc:
        shader_command = [glslc, "-O", str(VULKAN_SHADER), "-o", str(spirv)]
    elif glslang:
        shader_command = [glslang, "-V", str(VULKAN_SHADER), "-o", str(spirv)]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(shader_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan shader compilation failed")

    exe = build_dir / "vulkan_geometry_sweep"
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    build_command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(VULKAN_HARNESS),
        "-o",
        str(exe),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(build_command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan harness build failed")
    return spirv, exe, shader_command, build_command


def _geometry_correctness(result_path: Path | None) -> dict[str, Any]:
    if not result_path:
        return {"status": "not_applicable", "oracle": "ISA extraction only; no geometry result provided"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    correctness = result.get("correctness") if isinstance(result.get("correctness"), dict) else {}
    return {
        "status": correctness.get("status", "not_run"),
        "oracle": "Referenced retained geometry-sweep CPU oracle artifact",
        "artifact_ref": str(result_path),
        "source_correctness": correctness,
    }


def _base_result(
    *,
    backend: str,
    environment: dict[str, Any],
    source_hash: str,
    command: list[str],
    hardware_gpu: str | None,
    gfx_arch: str | None,
    environment_ref: str | None,
    geometry_result: Path | None,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": backend,
        "hardware": {
            "gpu_name": _infer_gpu_name(environment, hardware_gpu),
            "gfx_arch": _infer_gfx_arch(environment, gfx_arch),
        },
        "source": _source_record(environment, source_hash),
        "command": command,
        "cwd": str(REPO_ROOT),
        "parameters": {
            "benchmark_family": "geometry_isa_stats",
            "algorithm": "repeat_shifted_f32_gemv_row_shared_tree_reduce",
            "geometry_result_ref": str(geometry_result) if geometry_result else None,
        },
        "correctness": _geometry_correctness(geometry_result),
        "timing": {
            "unit": "not_applicable_isa_extraction",
            "note": "Use the referenced geometry-sweep artifact for timing; RADV_DEBUG=shaders distorts timing.",
        },
        "classification": "diagnostic_unclassified",
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return result


def _run_hip(args: argparse.Namespace) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), HIP_HARNESS])
    result = _base_result(
        backend="hip",
        environment=environment,
        source_hash=source_hash,
        command=sys.argv.copy(),
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
        geometry_result=args.geometry_result,
    )
    workgroups = [int(value) for value in args.workgroups.split(",") if value]
    rows = []
    build_dir = args.build_dir / "hip"
    build_commands: list[list[str]] = []
    readobj_commands: list[list[str]] = []
    objdump_commands: list[list[str]] = []
    object_paths: list[str] = []
    assembly_paths: list[str] = []
    for workgroup in workgroups:
        fixed_workgroup_size = (
            workgroup if args.hip_workgroup_specialization == "fixed" else None
        )
        compile_dir = (
            build_dir / f"fixed_wg{workgroup}"
            if fixed_workgroup_size is not None
            else build_dir / "runtime"
        )
        obj, asm, build_command_one = _compile_hip_save_temps(
            compile_dir,
            args.gfx_arch,
            hip_wavefront_size=args.hip_wavefront_size,
            fixed_workgroup_size=fixed_workgroup_size,
        )
        readobj_command = [shutil.which("llvm-readobj") or "llvm-readobj", "--notes", str(obj)]
        objdump_command = [
            shutil.which("llvm-objdump") or "llvm-objdump",
            "-d",
            "--no-show-raw-insn",
            str(obj),
        ]
        notes = _read_text_command(readobj_command, cwd=REPO_ROOT)
        disasm = _read_text_command(objdump_command, cwd=REPO_ROOT)
        metadata = parse_hip_metadata(notes)
        disasm_stats = parse_disassembly_stats(disasm)
        build_commands.append(build_command_one)
        readobj_commands.append(readobj_command)
        objdump_commands.append(objdump_command)
        object_paths.append(str(obj))
        assembly_paths.append(str(asm))
        rows.append(
            {
                "k": args.k,
                "rows": args.rows,
                "workgroup_size": workgroup,
                "body_repeats": args.body_repeats,
                "lds_bytes": workgroup * 4,
                "hip_workgroup_specialization": args.hip_workgroup_specialization,
                "hip_wavefront_size_request": args.hip_wavefront_size,
                **metadata,
                **disasm_stats,
                "stats_status": "actual_hip_code_object_metadata_plus_objdump_disassembly",
            }
        )
        if args.hip_workgroup_specialization != "fixed":
            break
    if args.hip_workgroup_specialization != "fixed" and rows:
        template = rows[0]
        rows = [
            {
                **template,
                "workgroup_size": workgroup,
                "lds_bytes": workgroup * 4,
            }
            for workgroup in workgroups
        ]
    primary = rows[-1] if rows else {}
    result["parameters"].update(
        {
            "hip_workgroup_specialization": args.hip_workgroup_specialization,
            "hip_wavefront_size_request": args.hip_wavefront_size,
            "build_command": build_commands,
            "readobj_command": readobj_commands,
            "objdump_command": objdump_commands,
            "save_temps_dir": str(build_dir),
            "object_path": object_paths,
            "assembly_path": assembly_paths,
        }
    )
    result["isa"] = _json_safe(primary)
    result["measurements"] = {"rows": _json_safe(rows)}
    result["notes"] = (
        "HIP ISA stats are compile-time code-object metadata plus llvm-objdump counts. "
        "Runtime-workgroup extraction uses one code object for all listed workgroups; "
        "fixed-workgroup extraction compiles one code object per workgroup. Dynamic LDS "
        "bytes are reported per workgroup."
    )
    return _json_safe(result)


def _run_vulkan(args: argparse.Namespace) -> dict[str, Any]:
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), VULKAN_HARNESS, VULKAN_SHADER])
    result = _base_result(
        backend="vulkan",
        environment=environment,
        source_hash=source_hash,
        command=sys.argv.copy(),
        hardware_gpu=args.hardware_gpu,
        gfx_arch=args.gfx_arch,
        environment_ref=str(args.environment_ref) if args.environment_ref else None,
        geometry_result=args.geometry_result,
    )
    build_dir = args.build_dir / "vulkan"
    spirv, exe, shader_command, build_command = _compile_vulkan(build_dir)
    raw_json = build_dir / "vulkan_isa_probe_raw.json"
    harness_command = [
        str(exe),
        "--spirv",
        str(spirv),
        "--json",
        str(raw_json),
        "--k-list",
        str(args.k),
        "--rows-list",
        str(args.rows),
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
        "--device-index",
        str(args.device_index),
    ]
    completed = _run_command(
        harness_command,
        cwd=REPO_ROOT,
        env={"RADV_DEBUG": "shaders,shaderstats"},
        echo=not args.quiet_shader_dump,
    )
    if completed.returncode != 0:
        raise RuntimeError("Vulkan RADV_DEBUG=shaders,shaderstats run failed")
    dump = completed.stdout + completed.stderr
    shader_rows = parse_radv_shader_dump(dump)
    raw = json.loads(raw_json.read_text(encoding="utf-8"))
    raw_rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    correctness_by_wg = {
        int(row["workgroup_size"]): bool(row.get("correctness_pass")) for row in raw_rows
    }
    rows = []
    for row in shader_rows:
        workgroup = row.get("workgroup_size")
        row.update(
            {
                "k": args.k,
                "rows": args.rows,
                "body_repeats": args.body_repeats,
                "lds_bytes": row.get("shared_size"),
                "wave_size": row.get("api_subgroup_size"),
                "correctness_pass": correctness_by_wg.get(int(workgroup), None)
                if workgroup is not None
                else None,
                "stats_status": "radv_debug_shaders_final_disassembly",
            }
        )
        rows.append(row)
    primary = rows[-1] if rows else {}
    all_pass = bool(correctness_by_wg) and all(correctness_by_wg.values())
    result["correctness"] = {
        "status": "pass" if all_pass else "fail",
        "oracle": "Vulkan geometry harness CPU reference during RADV_DEBUG=shaders run",
        "raw_probe_retained": False,
        "geometry_result_ref": str(args.geometry_result) if args.geometry_result else None,
        "per_workgroup": correctness_by_wg,
    }
    result["parameters"].update(
        {
            "shader_command": shader_command,
            "build_command": build_command,
            "harness_command": harness_command,
            "debug_env": {"RADV_DEBUG": "shaders,shaderstats"},
            "raw_probe_retained": False,
            "shader_dump_bytes": len(dump.encode("utf-8")),
            "shader_dump_retained": False,
        }
    )
    result["isa"] = _json_safe(primary)
    result["measurements"] = {"rows": _json_safe(rows)}
    result["notes"] = (
        "Vulkan ISA stats come from RADV_DEBUG=shaders,shaderstats. This exposes final "
        "disassembly, ACO after-RA text, and RADV shaderstats allocation counts when "
        "the Mesa build supports them. Estimated physical register spans are also kept "
        "for cross-checking."
    )
    return _json_safe(result)


def _row_by_workgroup(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = result.get("measurements", {}).get("rows", [])
    by_wg: dict[int, dict[str, Any]] = {}
    for row in (rows if isinstance(rows, list) else []):
        if isinstance(row, dict) and row.get("workgroup_size") is not None:
            by_wg[int(row["workgroup_size"])] = row
    return by_wg


def build_comparison(
    hip_result: dict[str, Any],
    vulkan_result: dict[str, Any],
    *,
    command: list[str],
    out_ref: str | None = None,
) -> dict[str, Any]:
    hip_rows = _row_by_workgroup(hip_result)
    vulkan_rows = _row_by_workgroup(vulkan_result)
    matched = []
    for workgroup in sorted(set(hip_rows) & set(vulkan_rows)):
        hip = hip_rows[workgroup]
        vulkan = vulkan_rows[workgroup]
        matched.append(
            {
                "workgroup_size": workgroup,
                "k": hip.get("k") or vulkan.get("k"),
                "rows": hip.get("rows") or vulkan.get("rows"),
                "body_repeats": hip.get("body_repeats") or vulkan.get("body_repeats"),
                "hip_vgpr": hip.get("vgpr"),
                "hip_sgpr": hip.get("sgpr"),
                "hip_scratch_bytes": hip.get("scratch_bytes"),
                "hip_vgpr_spill_count": hip.get("vgpr_spill_count"),
                "hip_sgpr_spill_count": hip.get("sgpr_spill_count"),
                "hip_wave_size": hip.get("wave_size"),
                "hip_instruction_count": hip.get("instruction_count"),
                "hip_waitcnt_count": hip.get("waitcnt_count"),
                "hip_vopd_count": hip.get("vopd_count"),
                "hip_vopd_op_count": hip.get("vopd_op_count"),
                "hip_dot4_count": hip.get("dot4_count"),
                "vulkan_official_register_counts": (
                    vulkan.get("register_count_status") == "official_radv_debug_shaderstats"
                ),
                "vulkan_vgpr": vulkan.get("vgpr"),
                "vulkan_sgpr": vulkan.get("sgpr"),
                "vulkan_scratch_bytes": vulkan.get("scratch_bytes"),
                "vulkan_vgpr_spill_count": vulkan.get("vgpr_spill_count"),
                "vulkan_sgpr_spill_count": vulkan.get("sgpr_spill_count"),
                "vulkan_subgroups_per_simd": vulkan.get("subgroups_per_simd"),
                "vulkan_code_size_bytes": vulkan.get("code_size_bytes"),
                "vulkan_estimated_vgpr_span": vulkan.get("estimated_vgpr_span"),
                "vulkan_estimated_sgpr_span": vulkan.get("estimated_sgpr_span"),
                "vulkan_wave_size": vulkan.get("wave_size"),
                "vulkan_shared_size": vulkan.get("shared_size"),
                "vulkan_instruction_count": vulkan.get("instruction_count"),
                "vulkan_waitcnt_count": vulkan.get("waitcnt_count"),
                "vulkan_waitcnt_depctr_count": vulkan.get("waitcnt_depctr_count"),
                "vulkan_vopd_count": vulkan.get("vopd_count"),
                "vulkan_vopd_op_count": vulkan.get("vopd_op_count"),
                "vulkan_dot4_count": vulkan.get("dot4_count"),
                "vulkan_correctness_pass": vulkan.get("correctness_pass"),
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
                "ISA/stat comparison for the retained f32 geometry family. HIP has "
                "actual code-object register/spill metadata; Vulkan has RADV final "
                "disassembly plus RADV shaderstats allocation counts when present. "
                "On this evidence alone, do not classify the geometry timing gap as "
                "compiler_aco."
            ),
        }
    )


def _positive_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError
        values.append(value)
    if not values:
        raise ValueError
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hip", "vulkan"], help="Backend to extract")
    parser.add_argument("--compare", nargs=2, metavar=("HIP_RESULT", "VULKAN_RESULT"), type=Path)
    parser.add_argument("--out", type=Path, help="Write normalized/comparison JSON")
    parser.add_argument("--environment-json", type=Path, help="Use an existing environment artifact")
    parser.add_argument("--environment-ref", type=Path, help="Reference this environment path instead of embedding")
    parser.add_argument("--geometry-result", type=Path, help="Reference retained geometry result for correctness")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gfx-arch", help="Override gfx arch and HIP offload arch")
    parser.add_argument("--hardware-gpu", help="Override GPU name in normalized output")
    parser.add_argument(
        "--hip-workgroup-specialization",
        choices=["runtime", "fixed"],
        default="runtime",
        help="For HIP, extract runtime blockDim code or one fixed-workgroup code object per workgroup",
    )
    parser.add_argument("--hip-wavefront-size", choices=["default", "32", "64"], default="default")
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--workgroups", default="64,256")
    parser.add_argument("--body-repeats", type=int, default=128)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
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
    if args.backend != "hip" and args.hip_workgroup_specialization != "runtime":
        parser.error("--hip-workgroup-specialization=fixed only applies to --backend hip")
    if args.backend != "hip" and args.hip_wavefront_size != "default":
        parser.error("--hip-wavefront-size only applies to --backend hip")
    if args.k <= 0 or args.rows <= 0 or args.body_repeats <= 0:
        parser.error("--k, --rows, and --body-repeats must be positive")
    try:
        _positive_int_list(args.workgroups)
    except ValueError:
        parser.error("--workgroups must be a comma-separated positive integer list")
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
        result = _run_hip(args)
    else:
        result = _run_vulkan(args)

    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
