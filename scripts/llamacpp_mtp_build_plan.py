#!/usr/bin/env python3
"""Emit a reproducible build plan for the patched llama.cpp MTP source tree.

This is a preflight/planning helper only: it does not run CMake or mutate the
read-only llama.cpp checkout.  The output artifact records the isolated source,
chosen generator, HIP/CMake options, required tools, and exact configure/build
commands for the next numeric hidden-in capture step.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MANIFEST = Path(
    "benchmarks/results/mtp-gguf-iter285-llamacpp-patched-source-manifest.json"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter286-llamacpp-build-plan.json")
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter286-build")
TARGET_RELATIVE_PATH = Path("src/models/qwen35moe.cpp")
LAYER_INPUT_ASSIGNMENT = "        res->t_layer_inp[il] = inpL;"
DEFAULT_ARCH = "gfx1151"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--iteration", type=int, default=286)
    args = parser.parse_args()

    artifact = build_plan_artifact(
        manifest_path=args.manifest,
        build_dir=args.build_dir,
        arch=args.arch,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "generator": artifact["build_plan"]["generator"],
                "source_dir": artifact["source_dir"],
                "build_dir": artifact["build_dir"],
                "preflight_passed": artifact["preflight"]["passed"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_plan_artifact(
    *,
    manifest_path: Path,
    build_dir: Path,
    arch: str = DEFAULT_ARCH,
    env: Mapping[str, str] | None = None,
    iteration: int = 286,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    manifest = json.loads(manifest_path.read_text())
    source_dir = Path(manifest["output_dir"])
    source_validation = validate_source_tree(source_dir)
    tools = discover_tools(env_map)
    rocm = discover_rocm(env_map)
    generator = choose_generator(tools)
    configure = build_configure_command(
        source_dir=source_dir,
        build_dir=build_dir,
        arch=arch,
        generator=generator,
        tools=tools,
        rocm=rocm,
    )
    build = ["cmake", "--build", str(build_dir), "--target", "llama", "-j", "8"]
    preflight = summarize_preflight(
        manifest=manifest,
        source_validation=source_validation,
        tools=tools,
        rocm=rocm,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_build_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "ready" if preflight["passed"] else "blocked",
        "manifest_path": str(manifest_path),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "target_relative_path": str(TARGET_RELATIVE_PATH),
        "source_manifest_summary": summarize_manifest(manifest),
        "source_validation": source_validation,
        "tools": tools,
        "rocm": rocm,
        "build_plan": {
            "generator": generator,
            "arch": arch,
            "target": "llama",
            "cmake_hip_compiler": tools["cmake_hip_compiler"],
            "configure_command": configure,
            "configure_command_shell": shlex.join(configure),
            "build_command": build,
            "build_command_shell": shlex.join(build),
            "notes": [
                "Run from the hipEngine shell with the therock env active.",
                "This builds libllama with GGML_HIP; capture harness can include "
                "src/llama-ext.h.",
                "Keep /home/lhl/llama.cpp/llama.cpp-hip read-only; build only "
                "the isolated /tmp source.",
            ],
        },
        "preflight": preflight,
        "next_action": next_action(preflight),
    }


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    ref = manifest.get("reference_basis") or {}
    patch = manifest.get("patch") or {}
    validation = manifest.get("target_validation") or {}
    return {
        "status": manifest.get("status"),
        "observed_commit": ref.get("observed_commit"),
        "commit_matches_expected": ref.get("commit_matches_expected"),
        "external_checkout_modified": ref.get("external_checkout_modified"),
        "patch_sha256": patch.get("sha256"),
        "patch_applied": validation.get("patch_applied"),
        "assignment_line": validation.get("assignment_line"),
    }


def validate_source_tree(source_dir: Path) -> dict[str, Any]:
    target = source_dir / TARGET_RELATIVE_PATH
    target_text = target.read_text() if target.exists() else ""
    return {
        "source_dir_exists": source_dir.exists(),
        "cmakelists_exists": (source_dir / "CMakeLists.txt").exists(),
        "llama_ext_header_exists": (source_dir / "src" / "llama-ext.h").exists(),
        "target_exists": target.exists(),
        "layer_input_assignment_count": target_text.count(LAYER_INPUT_ASSIGNMENT),
        "layer_input_assignment_line": find_line(target_text, LAYER_INPUT_ASSIGNMENT),
        "sentinel_exists": (source_dir / ".hipengine_llamacpp_mtp_temp_source").exists(),
    }


def discover_tools(env: Mapping[str, str]) -> dict[str, Any]:
    path = env.get("PATH")
    names = ("cmake", "hipcc", "amdclang++", "clang++", "make", "ninja")
    tools = {name: tool_record(name, path) for name in names}
    tools["ninja"]["required"] = False
    tools["amdclang++"]["required"] = False
    tools["clang++"]["required"] = False
    tools["cmake_hip_compiler"] = choose_cmake_hip_compiler(tools)
    return tools


def tool_record(name: str, path: str | None) -> dict[str, Any]:
    resolved = shutil.which(name, path=path)
    return {"path": resolved, "present": resolved is not None, "required": True}


def discover_rocm(env: Mapping[str, str]) -> dict[str, Any]:
    rocm_path = env.get("ROCM_PATH") or env.get("HIP_PATH") or ""
    prefix_parts = [part for part in env.get("CMAKE_PREFIX_PATH", "").split(os.pathsep) if part]
    prefix_paths = [Path(part) for part in prefix_parts]
    if rocm_path and Path(rocm_path) not in prefix_paths:
        prefix_paths.insert(0, Path(rocm_path))
    rocblas_config = first_existing(prefix_paths, Path("lib/cmake/rocblas/rocblas-config.cmake"))
    hip_config = first_existing(prefix_paths, Path("lib/cmake/hip/hip-config.cmake"))
    return {
        "ROCM_PATH": rocm_path,
        "HIP_PATH": env.get("HIP_PATH") or rocm_path,
        "CMAKE_PREFIX_PATH": prefix_parts,
        "rocblas_config": None if rocblas_config is None else str(rocblas_config),
        "hip_config": None if hip_config is None else str(hip_config),
        "rocblas_config_present": rocblas_config is not None,
        "hip_config_present": hip_config is not None,
    }


def choose_generator(tools: dict[str, Any]) -> str:
    return "Ninja" if tools["ninja"]["present"] else "Unix Makefiles"


def choose_cmake_hip_compiler(tools: dict[str, Any]) -> dict[str, Any]:
    for name in ("amdclang++", "clang++"):
        record = tools.get(name) or {}
        if record.get("present"):
            return {"name": name, "path": record["path"], "source": "explicit_clang"}
    return {
        "name": None,
        "path": None,
        "source": "cmake_default",
        "reason": "no amdclang++/clang++ on PATH; do not pass hipcc wrapper",
    }


def build_configure_command(
    *,
    source_dir: Path,
    build_dir: Path,
    arch: str,
    generator: str,
    tools: dict[str, Any],
    rocm: dict[str, Any],
) -> list[str]:
    command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_HIP=ON",
        f"-DAMDGPU_TARGETS={arch}",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_CURL=OFF",
        "-DBUILD_SHARED_LIBS=ON",
    ]
    hip_compiler = tools.get("cmake_hip_compiler") or {}
    if hip_compiler.get("path"):
        command.append(f"-DCMAKE_HIP_COMPILER={hip_compiler['path']}")
    prefix = rocm.get("CMAKE_PREFIX_PATH") or []
    if prefix:
        command.append("-DCMAKE_PREFIX_PATH=" + ";".join(prefix))
    return command


def summarize_preflight(
    *,
    manifest: dict[str, Any],
    source_validation: dict[str, Any],
    tools: dict[str, Any],
    rocm: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "manifest_prepared": manifest.get("status") == "prepared",
        "manifest_patch_applied": (
            (manifest.get("target_validation") or {}).get("patch_applied") is True
        ),
        "manifest_external_clean": (manifest.get("reference_basis") or {}).get(
            "external_checkout_modified"
        )
        is False,
        "source_dir_exists": source_validation["source_dir_exists"],
        "cmakelists_exists": source_validation["cmakelists_exists"],
        "llama_ext_header_exists": source_validation["llama_ext_header_exists"],
        "target_has_single_layer_input_assignment": (
            source_validation["layer_input_assignment_count"] == 1
        ),
        "sentinel_exists": source_validation["sentinel_exists"],
        "cmake_present": tools["cmake"]["present"],
        "hipcc_present": tools["hipcc"]["present"],
        "cmake_hip_compiler_safe": not _is_hipcc_wrapper(
            (tools.get("cmake_hip_compiler") or {}).get("path")
        ),
        "make_present": tools["make"]["present"],
        "rocblas_config_present": rocm["rocblas_config_present"],
        "hip_config_present": rocm["hip_config_present"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _is_hipcc_wrapper(path: str | None) -> bool:
    return path is not None and Path(path).name == "hipcc"


def next_action(preflight: dict[str, Any]) -> str:
    if preflight["passed"]:
        return "run_configure_and_build_llama_target_then_compile_hidden_in_capture_harness"
    missing = [name for name, passed in preflight["checks"].items() if not passed]
    return "resolve_build_preflight_blockers:" + ",".join(missing)


def first_existing(prefixes: list[Path], suffix: Path) -> Path | None:
    for prefix in prefixes:
        candidate = prefix / suffix
        if candidate.exists():
            return candidate
    return None


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


if __name__ == "__main__":
    main()
