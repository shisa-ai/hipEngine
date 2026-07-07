#!/usr/bin/env python3
"""Run or normalize the Vulkan dispatch/grid-floor microbenchmark."""

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
RUNNER_CPP = REPO_ROOT / "benchmarks" / "micro" / "runners" / "vulkan_dispatch_floor.cpp"
SHADER_GLSL = REPO_ROOT / "benchmarks" / "micro" / "kernels" / "vulkan" / "dispatch_floor.comp"
COLLECT_ENV = Path(__file__).resolve().parents[1] / "collect_env.py"
BENCH_NAME = "dispatch_grid_floor"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-micro-vulkan-build")


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


def _compile_shader(build_dir: Path) -> tuple[Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    spirv_path = build_dir / "dispatch_floor.spv"
    glslc = shutil.which("glslc")
    glslang = shutil.which("glslangValidator")
    if glslc:
        command = [glslc, "-O", str(SHADER_GLSL), "-o", str(spirv_path)]
    elif glslang:
        command = [glslang, "-V", str(SHADER_GLSL), "-o", str(spirv_path)]
    else:
        raise RuntimeError("neither glslc nor glslangValidator is available")
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("shader compilation failed")
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


def _compile_harness(build_dir: Path) -> tuple[Path, list[str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / "vulkan_dispatch_floor"
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("no C++ compiler found; set CXX or install c++/g++")
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        str(RUNNER_CPP),
        "-o",
        str(exe_path),
        *_vulkan_cflags_libs(),
    ]
    completed = _run_command(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError("Vulkan harness build failed")
    return exe_path, command


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
                arch = _find_gfx_arch("\n".join(str(line) for line in lines))
                if arch:
                    return arch
    return "unknown"


def _infer_gpu_name(
    legacy: dict[str, Any],
    environment: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override
    hardware = legacy.get("hardware")
    if isinstance(hardware, dict) and hardware.get("device_name"):
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
        if row.get("dispatch_count") == 941:
            return row
    return rows[-1]


def _timing_summary(legacy: dict[str, Any]) -> dict[str, Any]:
    config = legacy.get("config") if isinstance(legacy.get("config"), dict) else {}
    rows = legacy.get("rows") if isinstance(legacy.get("rows"), list) else []
    reference = _reference_row(rows)
    return _json_safe(
        {
            "unit": "us_per_dispatch",
            "median": reference.get("us_per_dispatch"),
            "warmup_iters": config.get("warmup"),
            "measured_iters": config.get("reps"),
            "primary": {
                "dispatch_count": reference.get("dispatch_count"),
                "grid_blocks": reference.get("grid_blocks"),
                "burst_us_median": reference.get("burst_us_median"),
                "burst_us_min": reference.get("burst_us_min"),
                "us_per_dispatch": reference.get("us_per_dispatch"),
            },
        }
    )


def normalize_legacy_dispatch_result(
    legacy: dict[str, Any],
    *,
    environment: dict[str, Any],
    wrapper_command: list[str],
    legacy_command: list[str] | None,
    shader_command: list[str] | None,
    harness_build_command: list[str] | None,
    source_hash: str,
    hardware_gpu: str | None = None,
    gfx_arch: str | None = None,
    environment_ref: str | None = None,
) -> dict[str, Any]:
    config = legacy.get("config") if isinstance(legacy.get("config"), dict) else {}
    hardware = legacy.get("hardware") if isinstance(legacy.get("hardware"), dict) else {}
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hipengine_micro_result",
        "bench": BENCH_NAME,
        "backend": "vulkan",
        "hardware": {
            "gpu_name": _infer_gpu_name(legacy, environment, hardware_gpu),
            "gfx_arch": _infer_gfx_arch(environment, gfx_arch),
            "device_id": (
                f"0x{int(hardware['device_id']):x}"
                if isinstance(hardware.get("device_id"), int)
                else str(hardware.get("device_id", ""))
            ),
        },
        "source": _source_record(environment, source_hash),
        "command": wrapper_command,
        "cwd": str(REPO_ROOT),
        "parameters": _json_safe(
            {
                "benchmark_family": "dispatch_and_grid_floor",
                "vulkan_runner": str(RUNNER_CPP.relative_to(REPO_ROOT)),
                "shader": str(SHADER_GLSL.relative_to(REPO_ROOT)),
                "legacy_run_tag": legacy.get("run_tag"),
                "legacy_status": legacy.get("status"),
                "shader_command": shader_command,
                "harness_build_command": harness_build_command,
                "legacy_command": legacy_command,
                "counts": config.get("counts"),
                "grid_sweep": config.get("grid_sweep"),
                "grid_sweep_count": config.get("grid_sweep_count"),
                "n_elements": config.get("n_elements"),
                "local_size_x": config.get("local_size_x"),
                "reps": config.get("reps"),
                "warmup": config.get("warmup"),
                "method": config.get("method"),
                "vulkan_hardware": hardware,
            }
        ),
        "correctness": {
            "status": "not_applicable",
            "oracle": "dispatch-floor diagnostic; shader writes only prevent empty dispatch removal",
        },
        "timing": _timing_summary(legacy),
        "classification": "runtime_dispatch",
        "measurements": _json_safe(
            {
                "rows": legacy.get("rows"),
                "grid_sweep_rows": legacy.get("grid_sweep_rows"),
            }
        ),
        "notes": (
            "Vulkan command buffers are pre-recorded outside the timed region. Timing "
            "is wall time around vkQueueSubmit+vkWaitForFences, so this isolates "
            "steady dispatch/replay cost rather than shader compilation, pipeline "
            "creation, or descriptor setup."
        ),
    }
    if environment_ref:
        result["environment_ref"] = environment_ref
    else:
        result["environment"] = environment
    return _json_safe(result)


def _legacy_command(args: argparse.Namespace, exe_path: Path, spirv_path: Path, legacy_json: Path) -> list[str]:
    command = [
        str(exe_path),
        "--spirv",
        str(spirv_path),
        "--counts",
        args.counts,
        "--n",
        str(args.n),
        "--reps",
        str(args.reps),
        "--warmup",
        str(args.warmup),
        "--grid-sweep-count",
        str(args.grid_sweep_count),
        "--device-index",
        str(args.device_index),
        "--json",
        str(legacy_json),
    ]
    if args.grid_sweep:
        command.extend(["--grid-sweep", args.grid_sweep])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-input", type=Path, help="Normalize an existing harness JSON")
    parser.add_argument("--legacy-json", type=Path, help="Keep the raw Vulkan harness JSON at this path")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--out", type=Path, help="Write normalized JSON to this path instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print normalized JSON")
    parser.add_argument("--environment-json", type=Path, help="Use a pre-collected environment JSON")
    parser.add_argument(
        "--environment-ref",
        help="Record this external environment artifact path instead of embedding it",
    )
    parser.add_argument(
        "--skip-device-probes",
        action="store_true",
        help="Skip rocminfo/rocm-smi/vulkaninfo/lspci collection",
    )
    parser.add_argument("--env-timeout-s", type=float, default=8.0)
    parser.add_argument("--env-max-output-chars", type=int, default=20000)
    parser.add_argument("--gfx-arch", default=None, help="gfx arch to record, if known")
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name override")
    parser.add_argument("--device-index", type=int, default=0)

    parser.add_argument("--counts", default="1,50,200,941")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--grid-sweep", default="")
    parser.add_argument("--grid-sweep-count", type=int, default=941)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment = _collect_environment(args)
    source_hash = _hash_files([Path(__file__).resolve(), RUNNER_CPP, SHADER_GLSL])
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        *(argv if argv is not None else sys.argv[1:]),
    ]

    shader_command: list[str] | None = None
    harness_build_command: list[str] | None = None
    legacy_command: list[str] | None = None
    temp_path: Path | None = None
    try:
        if args.legacy_input:
            legacy_path = args.legacy_input
        else:
            spirv_path, shader_command = _compile_shader(args.build_dir)
            exe_path, harness_build_command = _compile_harness(args.build_dir)
            if args.legacy_json:
                legacy_path = args.legacy_json
                legacy_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                tmp = tempfile.NamedTemporaryFile(
                    prefix="hipengine-vulkan-dispatch-floor-",
                    suffix=".json",
                    delete=False,
                )
                tmp.close()
                temp_path = Path(tmp.name)
                legacy_path = temp_path
            legacy_command = _legacy_command(args, exe_path, spirv_path, legacy_path)
            completed = _run_command(legacy_command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                return completed.returncode

        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        result = normalize_legacy_dispatch_result(
            legacy,
            environment=environment,
            wrapper_command=wrapper_command,
            legacy_command=legacy_command,
            shader_command=shader_command,
            harness_build_command=harness_build_command,
            source_hash=source_hash,
            hardware_gpu=args.hardware_gpu,
            gfx_arch=args.gfx_arch,
            environment_ref=args.environment_ref,
        )
        text = json.dumps(
            result,
            indent=2 if args.pretty else None,
            sort_keys=args.pretty,
            allow_nan=False,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
