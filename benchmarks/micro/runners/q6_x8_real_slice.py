#!/usr/bin/env python3
"""Run the Vulkan Q6_K X8 selected-down q8_1+dp4a real-slice probe."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
VULKAN_HARNESS = MICRO_ROOT / "runners" / "vulkan_q6_x8_selected_down.cpp"
VULKAN_QUANT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q8_1_quantize.comp"
VULKAN_DOT_SHADER = MICRO_ROOT / "kernels" / "vulkan" / "q6_x8_selected_down.comp"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-q6-x8-real-slice-build")


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


def _compile_shader(shader: Path, spirv: Path, defines: list[str]) -> list[str]:
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
        raise RuntimeError(f"shader build failed: {shader}")
    return command


def _compile_harness(exe: Path) -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
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
        raise RuntimeError("Vulkan Q6 X8 real-slice harness build failed")
    return command


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vulkan",), default="vulkan")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--local-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--reps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    quant_spv = args.build_dir / "q8_1_quantize.spv"
    dot_spv = args.build_dir / "q6_x8_selected_down.spv"
    exe = args.build_dir / "vulkan_q6_x8_selected_down"

    commands: list[dict[str, Any]] = []
    commands.append(
        {
            "kind": "compile_shader",
            "command": _compile_shader(VULKAN_QUANT_SHADER, quant_spv, []),
        }
    )
    commands.append(
        {
            "kind": "compile_shader",
            "command": _compile_shader(
                VULKAN_DOT_SHADER,
                dot_spv,
                [f"-DHIPENGINE_LOCAL_SIZE_X={args.local_size}"],
            ),
        }
    )
    commands.append({"kind": "compile_harness", "command": _compile_harness(exe)})

    raw_out = args.build_dir / "vulkan-q6-x8-selected-down-raw.json"
    run_command = [
        str(exe),
        "--quantize-spirv",
        str(quant_spv),
        "--dot-spirv",
        str(dot_spv),
        "--json",
        str(raw_out),
        "--rows",
        str(args.rows),
        "--experts",
        str(args.experts),
        "--in-features",
        str(args.in_features),
        "--out-features",
        str(args.out_features),
        "--input-scale",
        str(args.input_scale),
        "--local-size",
        str(args.local_size),
        "--reps",
        str(args.reps),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--device-index",
        str(args.device_index),
    ]
    completed = _run_command(run_command, cwd=REPO_ROOT)
    commands.append({"kind": "run_harness", "command": run_command, "returncode": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError("Vulkan Q6 X8 real-slice run failed")

    result = json.loads(raw_out.read_text(encoding="utf-8"))
    result["wrapper"] = {
        "schema": "hipengine.micro.q6_x8_real_slice_runner.v1",
        "command": [Path(sys.executable).name, *sys.argv],
        "cwd": str(REPO_ROOT),
        "build_dir": str(args.build_dir),
        "commands": _json_safe(commands),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
