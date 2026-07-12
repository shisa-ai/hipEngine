#!/usr/bin/env python3
"""Run the isolated llama.cpp build plan and emit a compact result manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_PLAN = Path("benchmarks/results/mtp-gguf-iter286-llamacpp-build-plan.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter287-llamacpp-build-result.json")
DEFAULT_LOG_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter287-build-logs")
BUILD_SENTINEL = ".hipengine_llamacpp_mtp_build_dir"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--replace-build-dir", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--iteration", type=int, default=287)
    args = parser.parse_args()

    result = run_build_plan(
        plan_path=args.plan,
        log_dir=args.log_dir,
        replace_build_dir=args.replace_build_dir,
        timeout_seconds=args.timeout_seconds,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "configure_rc": result["commands"]["configure"]["returncode"],
                "build_rc": result["commands"]["build"]["returncode"],
                "libraries": result["outputs"]["libraries"],
                "next_action": result["next_action"],
            },
            indent=2,
        )
    )


def run_build_plan(
    *,
    plan_path: Path,
    log_dir: Path,
    replace_build_dir: bool = False,
    timeout_seconds: int = 1800,
    env: Mapping[str, str] | None = None,
    iteration: int = 287,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    build_dir = Path(plan["build_dir"])
    source_dir = Path(plan["source_dir"])
    build_plan = plan["build_plan"]
    env_map = dict(os.environ if env is None else env)
    ensure_build_dir_ready(build_dir, replace=replace_build_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / BUILD_SENTINEL).write_text(
        json.dumps(
            {
                "kind": "hipengine_llamacpp_mtp_build_dir",
                "plan_path": str(plan_path),
                "source_dir": str(source_dir),
            },
            indent=2,
        )
        + "\n"
    )

    configure = run_logged_command(
        build_plan["configure_command"],
        cwd=Path.cwd(),
        env=env_map,
        log_prefix=log_dir / "configure",
        timeout_seconds=timeout_seconds,
    )
    if configure["returncode"] != 0:
        status = "configure_failed"
        build = skipped_command(build_plan["build_command"], "configure_failed")
    else:
        build = run_logged_command(
            build_plan["build_command"],
            cwd=Path.cwd(),
            env=env_map,
            log_prefix=log_dir / "build",
            timeout_seconds=timeout_seconds,
        )
        status = "built" if build["returncode"] == 0 else "build_failed"

    outputs = summarize_outputs(build_dir)
    diagnostics = diagnose_build_result(
        status=status,
        configure=configure,
        build=build,
        env=env_map,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_build_result",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "plan_path": str(plan_path),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "log_dir": str(log_dir),
        "source_manifest_summary": plan.get("source_manifest_summary"),
        "build_plan_summary": {
            "generator": build_plan.get("generator"),
            "arch": build_plan.get("arch"),
            "target": build_plan.get("target"),
        },
        "commands": {"configure": configure, "build": build},
        "outputs": outputs,
        "diagnostics": diagnostics,
        "external_checkout_modified": False,
        "next_action": next_action(status, outputs, diagnostics),
    }


def ensure_build_dir_ready(build_dir: Path, *, replace: bool) -> None:
    if not build_dir.exists():
        build_dir.mkdir(parents=True)
        return
    if not replace:
        raise FileExistsError(f"build directory already exists: {build_dir}")
    sentinel = build_dir / BUILD_SENTINEL
    if not sentinel.exists():
        raise FileExistsError(
            f"refusing to replace build directory without {BUILD_SENTINEL}: {build_dir}"
        )
    shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)


def run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_prefix: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = decode_timeout_stream(exc.stdout)
        stderr = decode_timeout_stream(exc.stderr)
        stderr += f"\nTIMEOUT after {timeout_seconds}s"
    elapsed = time.monotonic() - started
    stdout_path = log_prefix.with_suffix(".stdout.log")
    stderr_path = log_prefix.with_suffix(".stderr.log")
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "command": command,
        "command_shell": shell_join(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def skipped_command(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "command": command,
        "command_shell": shell_join(command),
        "returncode": None,
        "elapsed_seconds": 0.0,
        "stdout_log": None,
        "stderr_log": None,
        "stdout_tail": "",
        "stderr_tail": reason,
    }


def summarize_outputs(build_dir: Path) -> dict[str, Any]:
    libs = sorted(
        str(path)
        for pattern in ("**/libllama.so*", "**/libllama.dylib", "**/llama.dll")
        for path in build_dir.glob(pattern)
        if path.is_file()
    )
    cache = build_dir / "CMakeCache.txt"
    return {
        "cmake_cache_exists": cache.exists(),
        "build_sentinel_exists": (build_dir / BUILD_SENTINEL).exists(),
        "libraries": libs,
        "library_count": len(libs),
    }


def diagnose_build_result(
    *,
    status: str,
    configure: dict[str, Any],
    build: dict[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    stderr = str(configure.get("stderr_tail") or "")
    if status == "configure_failed" and "hipcc wrapper" in stderr:
        return {
            "kind": "cmake_hip_compiler_wrapper_rejected",
            "suggestion": "remove CMAKE_HIP_COMPILER=hipcc or set it to amdclang++/clang++",
            "candidate_hip_compiler": find_candidate_hip_compiler(env),
        }
    if bool(configure.get("timed_out")) or bool(build.get("timed_out")):
        return {"kind": "timeout", "suggestion": "increase timeout or inspect logs"}
    return {"kind": "none", "suggestion": None, "candidate_hip_compiler": None}


def find_candidate_hip_compiler(env: Mapping[str, str]) -> str | None:
    path = env.get("PATH")
    for name in ("amdclang++", "clang++"):
        candidate = shutil.which(name, path=path)
        if candidate:
            return candidate
    return None


def next_action(
    status: str,
    outputs: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> str:
    if status == "built" and outputs["library_count"] > 0:
        return "compile_hidden_in_capture_harness_against_built_libllama"
    if status == "built":
        return "locate_llama_library_or_adjust_build_target"
    if diagnostics and diagnostics.get("kind") == "cmake_hip_compiler_wrapper_rejected":
        return "regenerate_build_plan_with_amdclangpp_or_without_cmake_hip_compiler"
    if status == "configure_failed":
        return "inspect_configure_log_and_fix_build_configuration"
    return "inspect_build_log_and_fix_compile_or_link_failure"


def shell_join(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    main()
