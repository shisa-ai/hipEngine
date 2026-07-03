#!/usr/bin/env python3
"""Build a temporary llama.cpp pre-output_norm h_nextn capture harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_mtp_compile_hidden_in_harness import run_logged  # noqa: E402
from scripts.llamacpp_mtp_compile_hidden_seed_harness import (  # noqa: E402
    compile_hidden_seed_harness,
)

DEFAULT_PLAN = Path("benchmarks/results/mtp-gguf-iter306-pre-output-norm-capture-plan.json")
DEFAULT_BASE_BUILD = Path("benchmarks/results/mtp-gguf-iter289-llamacpp-build-result-amdclang.json")
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter307-pre-output-norm-harness-build.json"
)
DEFAULT_PATCHED_BUILD = Path(
    "benchmarks/results/mtp-gguf-iter307-pre-output-norm-llamacpp-build-result.json"
)
DEFAULT_HARNESS_COMPILE = Path(
    "benchmarks/results/mtp-gguf-iter307-pre-output-norm-harness-compile.json"
)
DEFAULT_SOURCE_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter307-pre-output-norm-src")
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter307-pre-output-norm-build")
DEFAULT_LOG_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter307-pre-output-norm-build-logs")
DEFAULT_HARNESS_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter307-pre-output-norm-harness")
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
    parser.add_argument("--iteration", type=int, default=307)
    args = parser.parse_args()

    artifact = build_pre_output_norm_harness(
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


def build_pre_output_norm_harness(
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
    iteration: int = 307,
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
    patch_result = apply_pre_output_patch(
        source_dir=target_source_dir,
        old_text=plan["llamacpp_pre_output_patch"]["old_text"],
        new_text=plan["llamacpp_pre_output_patch"]["new_text"],
    )
    preserve_result = apply_post_output_preserve_patch(source_dir=target_source_dir)
    patch_result["post_output_preserve"] = preserve_result
    patch_result["applied"] = bool(patch_result["applied"] and preserve_result["applied"])
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
    if configure["returncode"] == 0:
        build = run_logged(
            build_command,
            cwd=Path.cwd(),
            env=env_map,
            stdout_path=log_dir / "build.stdout.log",
            stderr_path=log_dir / "build.stderr.log",
            timeout_seconds=timeout_seconds,
        )
    else:
        build = skipped_command(build_command, "configure_failed")
    libraries = discover_libraries(target_build_dir)
    patched_build = build_patched_build_result(
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
        "kind": "llamacpp_pre_output_norm_harness_build",
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


def copy_source_tree(source_dir: Path, target_source_dir: Path) -> dict[str, Any]:
    if target_source_dir.exists():
        return {
            "status": "reused_existing",
            "source_dir": str(source_dir),
            "target_source_dir": str(target_source_dir),
        }
    target_source_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_source_dir, symlinks=True)
    return {
        "status": "copied",
        "source_dir": str(source_dir),
        "target_source_dir": str(target_source_dir),
    }


def apply_pre_output_patch(*, source_dir: Path, old_text: str, new_text: str) -> dict[str, Any]:
    return replace_unique_block(
        source_dir=source_dir,
        old_text=old_text,
        new_text=new_text,
        patch_name="pre_output_capture",
    )


def apply_post_output_preserve_patch(*, source_dir: Path) -> dict[str, Any]:
    old_text = (
        '    cb(cur, "h_nextn", -1);\n'
        "    res->t_h_nextn = cur;\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {"
    )
    new_text = (
        '    cb(cur, "h_nextn_post_output_norm", -1);\n'
        "    // PRE-output_norm diagnostic: keep res->t_h_nextn from before output_norm.\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {"
    )
    return replace_unique_block(
        source_dir=source_dir,
        old_text=old_text,
        new_text=new_text,
        patch_name="post_output_preserve_pre_capture",
    )


def replace_unique_block(
    *,
    source_dir: Path,
    old_text: str,
    new_text: str,
    patch_name: str,
) -> dict[str, Any]:
    path = source_dir / "src" / "models" / "qwen35moe.cpp"
    text = path.read_text()
    old_count = text.count(old_text)
    already_count = text.count(new_text)
    if old_count == 1:
        path.write_text(text.replace(old_text, new_text, 1))
        status = "applied"
        applied = True
    elif old_count == 0 and already_count == 1:
        status = "already_applied"
        applied = True
    else:
        status = "patch_anchor_error"
        applied = False
    final_text = path.read_text()
    return {
        "name": patch_name,
        "status": status,
        "applied": applied,
        "path": str(path),
        "old_count_before": old_count,
        "new_count_before": already_count,
        "old_count_after": final_text.count(old_text),
        "new_count_after": final_text.count(new_text),
    }


def rewrite_configure_command(
    command: list[str], *, source_dir: Path, build_dir: Path
) -> list[str]:
    rewritten = list(command)
    for flag, value in (("-S", source_dir), ("-B", build_dir)):
        if flag in rewritten:
            rewritten[rewritten.index(flag) + 1] = str(value)
    return rewritten


def rewrite_build_command(command: list[str], *, build_dir: Path, jobs: int) -> list[str]:
    rewritten = list(command)
    if "--build" in rewritten:
        rewritten[rewritten.index("--build") + 1] = str(build_dir)
    if "-j" in rewritten:
        rewritten[rewritten.index("-j") + 1] = str(int(jobs))
    else:
        rewritten.extend(["-j", str(int(jobs))])
    return rewritten


def discover_libraries(build_dir: Path) -> list[str]:
    bin_dir = build_dir / "bin"
    names = ["libllama.so", "libllama.so.0", "libllama.so.0.0.0"]
    return [str(bin_dir / name) for name in names if (bin_dir / name).exists()]


def build_patched_build_result(
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
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_pre_output_norm_build_result",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "built" if build.get("returncode") == 0 and libraries else "build_failed",
        "plan_path": str(plan_path),
        "base_build_source": base_build.get("source_dir"),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "log_dir": str(log_dir),
        "patch": dict(patch_result),
        "commands": {"configure": dict(configure), "build": dict(build)},
        "outputs": {
            "libraries": libraries,
            "library_count": len(libraries),
            "cmake_cache_exists": (build_dir / "CMakeCache.txt").exists(),
        },
        "external_checkout_modified": False,
    }


def status_from_results(
    configure: Mapping[str, Any],
    build: Mapping[str, Any],
    harness_compile: Mapping[str, Any],
    libraries: list[str],
) -> str:
    if configure.get("returncode") != 0:
        return "configure_failed"
    if build.get("returncode") != 0 or not libraries:
        return "build_failed"
    if harness_compile.get("status") != "compiled":
        return "harness_compile_failed"
    return "ready"


def next_action(status: str) -> str:
    if status == "ready":
        return "run_pre_output_norm_harness_and_compare_hipengine_serial_row"
    if status == "configure_failed":
        return "inspect_pre_output_norm_cmake_configure_logs"
    if status == "build_failed":
        return "inspect_pre_output_norm_cmake_build_logs"
    return "inspect_pre_output_norm_harness_compile_logs"


def safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    tmp_root = Path("/tmp").resolve()
    if tmp_root not in resolved.parents and resolved != tmp_root:
        raise ValueError(f"refusing to remove non-/tmp path: {path}")
    if not path.exists():
        return
    shutil.rmtree(resolved)


def skipped_command(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "command": command,
        "command_shell": " ".join(command),
        "returncode": None,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "stdout_log": None,
        "stderr_log": None,
        "stdout_tail": "",
        "stderr_tail": f"skipped: {reason}",
    }


def skipped_harness_compile(harness_dir: Path, *, reason: str) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "llamacpp_hidden_seed_harness_compile",
        "status": "skipped",
        "harness_kind": "capture",
        "outputs": {"executable": str(harness_dir / "llamacpp_hidden_seed_capture")},
        "probe_run": skipped_command([str(harness_dir)], reason),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
