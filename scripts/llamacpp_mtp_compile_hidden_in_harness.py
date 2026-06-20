#!/usr/bin/env python3
"""Compile a minimal llama.cpp hidden-in capture probe.

The probe is intentionally small: it includes llama.h and llama-ext.h, takes the
addresses of llama_set/get_embeddings_layer_inp so the extension symbols must
link, and prints a JSON-ish readiness line.  It does not load or run the model;
that is the next step after proving the patched libllama can be linked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BUILD_RESULT = Path(
    "benchmarks/results/mtp-gguf-iter289-llamacpp-build-result-amdclang.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter290-llamacpp-hidden-in-harness-compile.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter290-hidden-in-harness")
HARNESS_NAME = "llamacpp_hidden_in_probe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-result", type=Path, default=DEFAULT_BUILD_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--iteration", type=int, default=290)
    args = parser.parse_args()

    artifact = compile_hidden_in_harness(
        build_result_path=args.build_result,
        output_dir=args.output_dir,
        compiler=args.compiler,
        timeout_seconds=args.timeout_seconds,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "compiler": artifact["compiler"],
                "executable": artifact["outputs"]["executable"],
                "probe_rc": artifact["probe_run"]["returncode"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compile_hidden_in_harness(
    *,
    build_result_path: Path,
    output_dir: Path,
    compiler: str | None = None,
    timeout_seconds: int = 120,
    env: Mapping[str, str] | None = None,
    iteration: int = 290,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    build_result = json.loads(build_result_path.read_text())
    source_dir = Path(build_result["source_dir"])
    build_dir = Path(build_result["build_dir"])
    lib_dir = choose_lib_dir(build_result)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{HARNESS_NAME}.cpp"
    exe_path = output_dir / HARNESS_NAME
    source_path.write_text(hidden_in_harness_source())

    selected_compiler = compiler or choose_compiler(env_map)
    header_validation = validate_headers(source_dir)
    command = build_compile_command(
        compiler=selected_compiler,
        source_path=source_path,
        exe_path=exe_path,
        source_dir=source_dir,
        lib_dir=lib_dir,
    )
    compile_result = run_logged(
        command,
        cwd=Path.cwd(),
        env=env_map,
        stdout_path=output_dir / "compile.stdout.log",
        stderr_path=output_dir / "compile.stderr.log",
        timeout_seconds=timeout_seconds,
    )
    if compile_result["returncode"] == 0:
        probe_run = run_probe(exe_path, lib_dir=lib_dir, env=env_map, timeout_seconds=30)
    else:
        probe_run = skipped_command([str(exe_path)], "compile_failed")
    outputs = {
        "source": str(source_path),
        "executable": str(exe_path),
        "executable_exists": exe_path.exists(),
        "executable_bytes": exe_path.stat().st_size if exe_path.exists() else 0,
    }
    status = "compiled" if compile_result["returncode"] == 0 else "compile_failed"
    if status == "compiled" and probe_run["returncode"] != 0:
        status = "probe_failed"
    return {
        "schema": 1,
        "kind": "llamacpp_hidden_in_harness_compile",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "build_result_path": str(build_result_path),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "lib_dir": str(lib_dir),
        "compiler": selected_compiler,
        "header_validation": header_validation,
        "link_symbols": [
            "llama_set_embeddings_layer_inp",
            "llama_get_embeddings_layer_inp",
        ],
        "compile": compile_result,
        "probe_run": probe_run,
        "outputs": outputs,
        "external_checkout_modified": False,
        "next_action": next_action(status),
    }


def hidden_in_harness_source() -> str:
    return r'''#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdint>

int main() {
    auto set_layer_input = &llama_set_embeddings_layer_inp;
    auto get_layer_input = &llama_get_embeddings_layer_inp;
    if (set_layer_input == nullptr || get_layer_input == nullptr) {
        return 2;
    }
    std::printf("{\"linked_layer_input_api\":true,\"layer_id\":3}\n");
    return 0;
}
'''


def validate_headers(source_dir: Path) -> dict[str, Any]:
    return {
        "llama_h": {
            "path": str(source_dir / "include" / "llama.h"),
            "exists": (source_dir / "include" / "llama.h").exists(),
        },
        "llama_ext_h": {
            "path": str(source_dir / "src" / "llama-ext.h"),
            "exists": (source_dir / "src" / "llama-ext.h").exists(),
        },
        "ggml_include": {
            "path": str(source_dir / "ggml" / "include"),
            "exists": (source_dir / "ggml" / "include").exists(),
        },
    }


def choose_lib_dir(build_result: dict[str, Any]) -> Path:
    libraries = build_result.get("outputs", {}).get("libraries", [])
    for library in libraries:
        path = Path(library)
        if path.name == "libllama.so":
            return path.parent
    if libraries:
        return Path(libraries[0]).parent
    return Path(build_result["build_dir"]) / "bin"


def choose_compiler(env: Mapping[str, str]) -> str:
    path = env.get("PATH")
    for name in ("amdclang++", "clang++", "c++", "g++"):
        resolved = shutil.which(name, path=path)
        if resolved:
            return resolved
    return "c++"


def build_compile_command(
    *,
    compiler: str,
    source_path: Path,
    exe_path: Path,
    source_dir: Path,
    lib_dir: Path,
) -> list[str]:
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-I",
        str(source_dir / "include"),
        "-I",
        str(source_dir / "src"),
        "-I",
        str(source_dir / "ggml" / "include"),
        str(source_path),
        "-L",
        str(lib_dir),
        f"-Wl,-rpath,{lib_dir}",
        "-Wl,--enable-new-dtags",
        "-lllama",
        "-o",
        str(exe_path),
    ]


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
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
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = decode_timeout_stream(exc.stdout)
        stderr = decode_timeout_stream(exc.stderr)
        stderr += f"\nTIMEOUT after {timeout_seconds}s"
        timed_out = True
    elapsed = time.monotonic() - start
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "command": command,
        "command_shell": subprocess.list2cmdline(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def run_probe(
    exe_path: Path,
    *,
    lib_dir: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    run_env = dict(env)
    old_ld_path = run_env.get("LD_LIBRARY_PATH", "")
    run_env["LD_LIBRARY_PATH"] = str(lib_dir) + (os.pathsep + old_ld_path if old_ld_path else "")
    return run_logged(
        [str(exe_path)],
        cwd=Path.cwd(),
        env=run_env,
        stdout_path=exe_path.with_suffix(".stdout.log"),
        stderr_path=exe_path.with_suffix(".stderr.log"),
        timeout_seconds=timeout_seconds,
    )


def skipped_command(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "command": command,
        "command_shell": subprocess.list2cmdline(command),
        "returncode": None,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "stdout_log": None,
        "stderr_log": None,
        "stdout_tail": "",
        "stderr_tail": reason,
    }


def next_action(status: str) -> str:
    if status == "compiled":
        return "extend_harness_to_load_model_decode_prompt_and_dump_hidden_in"
    if status == "probe_failed":
        return "inspect_harness_runtime_linker_or_rocm_library_path"
    return "inspect_hidden_in_harness_compile_logs"


def decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    main()
