#!/usr/bin/env python3
"""Build a temporary llama.cpp input-embedding h_nextn capture harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_build_layer_boundary_harness import (  # noqa: E402
    build_layer_build_result,
)
from scripts.llamacpp_mtp_build_pre_output_norm_harness import (  # noqa: E402
    apply_post_output_preserve_patch,
    copy_source_tree,
    discover_libraries,
    read_json,
    replace_unique_block,
    rewrite_build_command,
    rewrite_configure_command,
    safe_rmtree,
    skipped_command,
    skipped_harness_compile,
    status_from_results,
)
from scripts.llamacpp_mtp_compile_hidden_in_harness import run_logged  # noqa: E402
from scripts.llamacpp_mtp_compile_hidden_seed_harness import (  # noqa: E402
    compile_hidden_seed_harness,
)

DEFAULT_PLAN = Path("benchmarks/results/mtp-gguf-iter317-initial-input-capture-plan.json")
DEFAULT_BASE_BUILD = Path("benchmarks/results/mtp-gguf-iter289-llamacpp-build-result-amdclang.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter318-input-embed-harness-build.json")
DEFAULT_PATCHED_BUILD = Path(
    "benchmarks/results/mtp-gguf-iter318-input-embed-llamacpp-build-result.json"
)
DEFAULT_HARNESS_COMPILE = Path(
    "benchmarks/results/mtp-gguf-iter318-input-embed-harness-compile.json"
)
DEFAULT_SOURCE_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter318-input-embed-src")
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter318-input-embed-build")
DEFAULT_LOG_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter318-input-embed-build-logs")
DEFAULT_HARNESS_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter318-input-embed-harness")
DEFAULT_COMPILER = "/home/lhl/miniforge3/envs/therock/bin/amdclang++"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--base-build", type=Path, default=DEFAULT_BASE_BUILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patched-build-result", type=Path, default=DEFAULT_PATCHED_BUILD)
    parser.add_argument("--harness-compile", type=Path, default=DEFAULT_HARNESS_COMPILE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--harness-dir", type=Path, default=DEFAULT_HARNESS_DIR)
    parser.add_argument("--compiler", default=DEFAULT_COMPILER)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--iteration", type=int, default=318)
    args = parser.parse_args()

    artifact = build_input_embed_harness(
        plan_path=args.plan,
        base_build_path=args.base_build,
        output_path=args.output,
        patched_build_result_path=args.patched_build_result,
        harness_compile_path=args.harness_compile,
        target_source_dir=args.source_dir,
        target_build_dir=args.build_dir,
        log_dir=args.log_dir,
        harness_dir=args.harness_dir,
        compiler=args.compiler,
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
        clean=bool(args.clean),
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "patch_applied": artifact["patch"]["applied"],
                "configure_rc": artifact["commands"]["configure"]["returncode"],
                "build_rc": artifact["commands"]["build"]["returncode"],
                "harness_status": artifact["harness_compile"]["status"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_input_embed_harness(
    *,
    plan_path: Path,
    base_build_path: Path,
    output_path: Path,
    patched_build_result_path: Path,
    harness_compile_path: Path,
    target_source_dir: Path,
    target_build_dir: Path,
    log_dir: Path,
    harness_dir: Path,
    compiler: str,
    jobs: int,
    timeout_seconds: int,
    clean: bool,
    env: Mapping[str, str] | None = None,
    iteration: int = 318,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    plan = read_json(plan_path)
    base_build = read_json(base_build_path)
    base_source_dir = Path(base_build["source_dir"])
    if clean:
        safe_rmtree(target_source_dir)
        safe_rmtree(target_build_dir)
        safe_rmtree(log_dir)
        safe_rmtree(harness_dir)
    copy_result = copy_source_tree(base_source_dir, target_source_dir)
    patch_result = apply_input_embed_patch(
        source_dir=target_source_dir,
        old_text=plan["llamacpp_input_patch"]["input_capture_old_text"],
        new_text=plan["llamacpp_input_patch"]["input_capture_new_text"],
    )
    preserve_result = apply_post_output_preserve_patch(source_dir=target_source_dir)
    patch_result["post_output_preserve"] = preserve_result
    patch_result["layer_capture_patch_count"] = count_patch_label(
        target_source_dir,
        "h_nextn_layer_out",
    )
    patch_result["final_pre_output_patch_count"] = count_patch_label(
        target_source_dir,
        "h_nextn_pre_output_norm",
    )
    patch_result["applied"] = bool(
        patch_result["applied"]
        and preserve_result["applied"]
        and patch_result["layer_capture_patch_count"] == 0
        and patch_result["final_pre_output_patch_count"] == 0
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_command = rewrite_configure_command(
        base_build["commands"]["configure"]["command"],
        source_dir=target_source_dir,
        build_dir=target_build_dir,
    )
    build_command = rewrite_build_command(
        base_build["commands"]["build"]["command"],
        build_dir=target_build_dir,
        jobs=jobs,
    )
    configure = run_logged(
        configure_command,
        cwd=Path.cwd(),
        env=env_map,
        stdout_path=log_dir / "configure.stdout.log",
        stderr_path=log_dir / "configure.stderr.log",
        timeout_seconds=timeout_seconds,
    )
    if configure["returncode"] == 0 and patch_result["applied"]:
        build = run_logged(
            build_command,
            cwd=Path.cwd(),
            env=env_map,
            stdout_path=log_dir / "build.stdout.log",
            stderr_path=log_dir / "build.stderr.log",
            timeout_seconds=timeout_seconds,
        )
    else:
        reason = "patch_failed" if not patch_result["applied"] else "configure_failed"
        build = skipped_command(build_command, reason)
    libraries = discover_libraries(target_build_dir)
    patched_build = build_input_build_result(
        base_build=base_build,
        plan_path=plan_path,
        source_dir=target_source_dir,
        build_dir=target_build_dir,
        log_dir=log_dir,
        patch_result=patch_result,
        configure=configure,
        build=build,
        libraries=libraries,
        iteration=iteration,
    )
    patched_build_result_path.parent.mkdir(parents=True, exist_ok=True)
    patched_build_result_path.write_text(json.dumps(patched_build, indent=2) + "\n")
    if build["returncode"] == 0 and libraries:
        harness_compile = compile_hidden_seed_harness(
            build_result_path=patched_build_result_path,
            output_dir=harness_dir,
            compiler=compiler,
            harness_kind="capture",
            timeout_seconds=120,
            env=env_map,
            iteration=iteration,
        )
    else:
        harness_compile = skipped_harness_compile(harness_dir, reason="build_failed")
    harness_compile_path.parent.mkdir(parents=True, exist_ok=True)
    harness_compile_path.write_text(json.dumps(harness_compile, indent=2) + "\n")
    status = status_from_results(configure, build, harness_compile, libraries)
    return {
        "schema": 1,
        "kind": "llamacpp_input_embed_harness_build",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "plan_path": str(plan_path),
        "base_build_path": str(base_build_path),
        "patched_build_result_path": str(patched_build_result_path),
        "harness_compile_path": str(harness_compile_path),
        "source_dir": str(target_source_dir),
        "build_dir": str(target_build_dir),
        "log_dir": str(log_dir),
        "copy": copy_result,
        "patch": patch_result,
        "commands": {"configure": configure, "build": build},
        "libraries": libraries,
        "harness_compile": harness_compile,
        "external_checkout_modified": False,
        "next_action": next_action(status),
    }


def apply_input_embed_patch(*, source_dir: Path, old_text: str, new_text: str) -> dict[str, Any]:
    result = replace_unique_block(
        source_dir=source_dir,
        old_text=old_text,
        new_text=new_text,
        patch_name="input_embed_capture",
    )
    result["capture_label"] = "h_nextn_input_embed"
    return result


def count_patch_label(source_dir: Path, label: str) -> int:
    path = source_dir / "src" / "models" / "qwen35moe.cpp"
    return path.read_text().count(label)


def build_input_build_result(
    *,
    base_build: Mapping[str, Any],
    plan_path: Path,
    source_dir: Path,
    build_dir: Path,
    log_dir: Path,
    patch_result: Mapping[str, Any],
    configure: Mapping[str, Any],
    build: Mapping[str, Any],
    libraries: list[str],
    iteration: int,
) -> dict[str, Any]:
    result = build_layer_build_result(
        base_build=base_build,
        plan_path=plan_path,
        source_dir=source_dir,
        build_dir=build_dir,
        log_dir=log_dir,
        layer_id=0,
        patch_result=patch_result,
        configure=configure,
        build=build,
        libraries=libraries,
        iteration=iteration,
    )
    result["kind"] = "llamacpp_mtp_input_embed_build_result"
    result.pop("layer_id", None)
    return result


def next_action(status: str) -> str:
    if status == "ready":
        return "run_input_embed_harness_and_compare_hipengine_hidden_in"
    if status == "configure_failed":
        return "inspect_input_embed_cmake_configure_logs"
    if status == "build_failed":
        return "inspect_input_embed_cmake_build_logs"
    return "inspect_input_embed_harness_compile_logs"


if __name__ == "__main__":
    main()
