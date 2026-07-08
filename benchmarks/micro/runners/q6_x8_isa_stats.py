#!/usr/bin/env python3
"""Compare HIP/LLVM and Vulkan/RADV ISA for the Q6_K X8 real slice."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
ISA_STATS = MICRO_ROOT / "runners" / "isa_stats.py"
HIP_SOURCE = REPO_ROOT / "hipengine" / "kernels" / "hip_gfx1100" / "quant" / "gguf_q6_k_pack8_gemv.hip"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q6-x8-isa-stats")

HIP_KERNEL = {
    "label": "q6_x8_dot",
    "symbol": "_ZN12_GLOBAL__N_146gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1_kernelEPKNS_15gguf_q8_1_blockEPKhPfPilllll",
}


def _load_isa_stats():
    spec = importlib.util.spec_from_file_location("micro_isa_stats_for_q6_x8", ISA_STATS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load ISA helpers: {ISA_STATS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


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


def _read_command(command: list[str], *, cwd: Path) -> str:
    completed = _run_command(command, cwd=cwd, echo=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed.stdout


def _compile_hip(args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc not found")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    output = args.build_dir / "gguf_q6_k_pack8_gemv.so"
    command = [
        hipcc,
        f"--offload-arch={args.gfx_arch}",
        "-shared",
        "-fPIC",
        "-O3",
        "-mllvm",
        "-amdgpu-unroll-threshold-local=600",
        "-mcumode",
        "--save-temps",
        str(HIP_SOURCE),
        "-o",
        str(output),
    ]
    completed = _run_command(command, cwd=args.build_dir)
    if completed.returncode != 0:
        raise RuntimeError("HIP Q6 X8 ISA build failed")
    objects = sorted(args.build_dir.glob("gguf_q6_k_pack8_gemv-hip-amdgcn-amd-amdhsa-*.o"))
    if not objects:
        raise RuntimeError(f"no device object emitted under {args.build_dir}")
    return output, objects[0], command


def _metadata_by_kernel(notes: str, isa: Any) -> dict[str, dict[str, Any]]:
    lines = notes.splitlines()
    target_match = re.search(r"amdhsa\.target:\s+(\S+)", notes)
    target = target_match.group(1) if target_match else None
    rows: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        match = re.search(r"\.name:\s+(\S+)", line)
        if not match:
            continue
        name = match.group(1)
        start = max(0, index - 32)
        end = min(len(lines), index + 80)
        row = isa.parse_hip_metadata("\n".join(lines[start:end]))
        row["kernel_name"] = name
        if row.get("target") is None:
            row["target"] = target
        rows[name] = row
    return rows


def _disasm_section(disasm: str, symbol: str) -> str:
    pattern = re.compile(
        rf"(?ms)^Disassembly of section \.text\.{re.escape(symbol)}:\n"
        rf"(.*?)(?=^Disassembly of section |\Z)"
    )
    match = pattern.search(disasm)
    if not match:
        raise RuntimeError(f"could not find disassembly section for {symbol}")
    return match.group(1)


def _hip_row(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    isa = _load_isa_stats()
    output, obj, build_command = _compile_hip(args)
    readobj_command = [shutil.which("llvm-readobj") or "llvm-readobj", "--notes", str(obj)]
    objdump_command = [
        shutil.which("llvm-objdump") or "llvm-objdump",
        "-d",
        "--no-show-raw-insn",
        str(obj),
    ]
    notes = _read_command(readobj_command, cwd=REPO_ROOT)
    disasm = _read_command(objdump_command, cwd=REPO_ROOT)
    metadata = _metadata_by_kernel(notes, isa)
    symbol = HIP_KERNEL["symbol"]
    section = _disasm_section(disasm, symbol)
    row = {
        "label": HIP_KERNEL["label"],
        "kernel_symbol": symbol,
        **metadata.get(symbol, {}),
        **isa.parse_disassembly_stats(section),
        "stats_status": "actual_hip_code_object_metadata_plus_objdump_disassembly",
    }
    params = {
        "build_command": build_command,
        "readobj_command": readobj_command,
        "objdump_command": objdump_command,
        "build_dir": args.build_dir,
        "output_path": output,
        "object_path": obj,
        "profile_flags": ["-mllvm", "-amdgpu-unroll-threshold-local=600", "-mcumode"],
    }
    return _json_safe(row), _json_safe(params)


def _vulkan_rows(vulkan_isa_result: Path) -> list[dict[str, Any]]:
    result = json.loads(vulkan_isa_result.read_text(encoding="utf-8"))
    rows = result.get("radv_final_shaders") or []
    labels = ["q8_1_quantize", "q6_x8_dot"]
    out = []
    for index, row in enumerate(rows):
        label = labels[index] if index < len(labels) else f"shader_{index}"
        out.append({"label": label, **row})
    return out


def _row_summary(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_wave_size": row.get("wave_size") or row.get("api_subgroup_size"),
        f"{prefix}_sgpr": row.get("sgpr"),
        f"{prefix}_vgpr": row.get("vgpr"),
        f"{prefix}_scratch_bytes": row.get("scratch_bytes"),
        f"{prefix}_sgpr_spill_count": row.get("sgpr_spill_count"),
        f"{prefix}_vgpr_spill_count": row.get("vgpr_spill_count"),
        f"{prefix}_instruction_count": row.get("instruction_count"),
        f"{prefix}_dot4_count": row.get("dot4_count"),
        f"{prefix}_vopd_count": row.get("vopd_count"),
        f"{prefix}_waitcnt_count": row.get("waitcnt_count"),
        f"{prefix}_buffer_load_count": row.get("buffer_load_count"),
        f"{prefix}_buffer_store_count": row.get("buffer_store_count"),
        f"{prefix}_global_load_count": row.get("global_load_count"),
        f"{prefix}_global_store_count": row.get("global_store_count"),
        f"{prefix}_ds_load_count": row.get("ds_load_count"),
        f"{prefix}_ds_store_count": row.get("ds_store_count"),
        f"{prefix}_estimated_sgpr_span": row.get("estimated_sgpr_span"),
        f"{prefix}_estimated_vgpr_span": row.get("estimated_vgpr_span"),
    }


def _timing_context(hip_result: dict[str, Any], vulkan_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "hip_timing_ms": vulkan_result.get("hip_timing_ms")
        or (hip_result.get("results") or [{}])[0].get("timing_ms"),
        "best_vulkan": vulkan_result.get("best_vulkan"),
        "vulkan_notes": vulkan_result.get("notes"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hip-result", type=Path, required=True)
    parser.add_argument("--vulkan-result", type=Path, required=True)
    parser.add_argument("--vulkan-isa-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--gfx-arch", default=os.environ.get("HIPENGINE_HIP_ARCH") or "gfx1151")
    parser.add_argument("--hardware-gpu", default="Radeon 8060S Graphics")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hip_result = json.loads(args.hip_result.read_text(encoding="utf-8"))
    vulkan_result = json.loads(args.vulkan_result.read_text(encoding="utf-8"))
    vulkan_isa = json.loads(args.vulkan_isa_result.read_text(encoding="utf-8"))
    hip_row, hip_params = _hip_row(args)
    vulkan_rows = _vulkan_rows(args.vulkan_isa_result)
    vulkan_by_label = {str(row.get("label")): row for row in vulkan_rows}
    vulkan_dot = vulkan_by_label.get("q6_x8_dot")
    if vulkan_dot is None:
        raise RuntimeError("retained Vulkan ISA artifact does not contain a q6_x8_dot row")
    matched = {
        "label": "q6_x8_dot",
        **_row_summary(hip_row, "hip"),
        **_row_summary(vulkan_dot, "vulkan"),
        "vulkan_official_register_counts": (
            vulkan_dot.get("register_count_status") == "official_radv_debug_shaderstats"
        ),
        "hip_stats_status": hip_row.get("stats_status"),
        "vulkan_stats_status": vulkan_dot.get("shaderstats_status") or vulkan_dot.get("stats_status"),
    }
    result = {
        "schema": "hipengine.micro.q6_x8_hip_vulkan_isa_comparison.v1",
        "kind": "hipengine_micro_comparison",
        "bench": "q6_x8_real_slice_isa_stats",
        "classification": "real_slice_probe",
        "hardware": {
            "gfx_arch": args.gfx_arch,
            "gpu_name": args.hardware_gpu,
        },
        "shape": vulkan_isa.get("shape") or vulkan_result.get("shape") or hip_result.get("shape"),
        "correctness": {
            "hip": {
                "artifact_ref": str(args.hip_result),
                "correctness_vs_raw_float": (hip_result.get("results") or [{}])[0].get("correctness_vs_raw_float"),
                "correctness_vs_production_t16_float": (hip_result.get("results") or [{}])[0].get(
                    "correctness_vs_production_t16_float"
                ),
            },
            "vulkan": {
                "artifact_ref": str(args.vulkan_result),
                "best_vulkan_correctness_pass": (vulkan_result.get("best_vulkan") or {}).get("correctness_pass"),
            },
        },
        "timing_context": _timing_context(hip_result, vulkan_result),
        "inputs": {
            "hip_result": str(args.hip_result),
            "vulkan_result": str(args.vulkan_result),
            "vulkan_isa_result": str(args.vulkan_isa_result),
        },
        "hip_parameters": hip_params,
        "hip_row": hip_row,
        "vulkan_rows": vulkan_rows,
        "matched_row": matched,
        "interpretation": (
            "Targeted ISA/stat comparison for the retained Q6_K X8 selected-down "
            "real-slice probe. This checks whether the synthetic memory/dot Vulkan "
            "wins transfer to the memory-heavy production-shaped Q6 X8 dot shader."
        ),
        "command": [Path(sys.executable).name, *sys.argv],
        "source": {
            "repo": str(REPO_ROOT),
            "source_hash": _hash_files([Path(__file__).resolve(), HIP_SOURCE, ISA_STATS]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
