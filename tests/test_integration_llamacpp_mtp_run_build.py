from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.llamacpp_mtp_run_build import (
    BUILD_SENTINEL,
    diagnose_build_result,
    ensure_build_dir_ready,
    run_build_plan,
    summarize_outputs,
)


def test_run_build_plan_records_success_with_fake_cmake(tmp_path: Path) -> None:
    source_dir = _write_source_tree(tmp_path)
    build_dir = tmp_path / "build"
    log_dir = tmp_path / "logs"
    tool_dir = _write_fake_cmake(tmp_path, fail_configure=False, fail_build=False)
    plan_path = _write_plan(tmp_path, source_dir=source_dir, build_dir=build_dir)

    result = run_build_plan(
        plan_path=plan_path,
        log_dir=log_dir,
        replace_build_dir=False,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tool_dir)},
    )

    assert result["status"] == "built"
    assert result["commands"]["configure"]["returncode"] == 0
    assert result["commands"]["build"]["returncode"] == 0
    assert result["outputs"]["cmake_cache_exists"] is True
    assert result["outputs"]["library_count"] == 1
    assert "compile_hidden_in_capture" in result["next_action"]
    assert (build_dir / BUILD_SENTINEL).exists()
    assert Path(result["commands"]["configure"]["stdout_log"]).exists()
    json.dumps(result)


def test_run_build_plan_records_configure_failure_and_skips_build(tmp_path: Path) -> None:
    source_dir = _write_source_tree(tmp_path)
    build_dir = tmp_path / "build"
    tool_dir = _write_fake_cmake(tmp_path, fail_configure=True, fail_build=False)
    plan_path = _write_plan(tmp_path, source_dir=source_dir, build_dir=build_dir)

    result = run_build_plan(
        plan_path=plan_path,
        log_dir=tmp_path / "logs",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tool_dir)},
    )

    assert result["status"] == "configure_failed"
    assert result["commands"]["configure"]["returncode"] == 7
    assert result["commands"]["build"]["returncode"] is None
    assert "configure_failed" in result["commands"]["build"]["stderr_tail"]


def test_diagnoses_cmake_hipcc_wrapper_rejection(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    compiler = tool_dir / "amdclang++"
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)

    diagnostics = diagnose_build_result(
        status="configure_failed",
        configure={"stderr_tail": "CMAKE_HIP_COMPILER is set to the hipcc wrapper"},
        build={"stderr_tail": ""},
        env={"PATH": str(tool_dir)},
    )

    assert diagnostics["kind"] == "cmake_hip_compiler_wrapper_rejected"
    assert diagnostics["candidate_hip_compiler"] == str(compiler)
    assert "amdclang++" in diagnostics["suggestion"]


def test_replace_build_dir_requires_sentinel(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "keep.txt").write_text("do not delete")

    with pytest.raises(FileExistsError, match="without"):
        ensure_build_dir_ready(build_dir, replace=True)


def test_replace_owned_build_dir(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / BUILD_SENTINEL).write_text("{}\n")
    (build_dir / "old.txt").write_text("remove")

    ensure_build_dir_ready(build_dir, replace=True)

    assert build_dir.exists()
    assert not (build_dir / "old.txt").exists()


def test_summarize_outputs_finds_libllama(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    lib_dir = build_dir / "src"
    lib_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text("cache\n")
    (build_dir / BUILD_SENTINEL).write_text("{}\n")
    (lib_dir / "libllama.so").write_text("fake\n")

    outputs = summarize_outputs(build_dir)

    assert outputs["cmake_cache_exists"] is True
    assert outputs["build_sentinel_exists"] is True
    assert outputs["library_count"] == 1


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")


def _write_source_tree(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    return source_dir


def _write_plan(tmp_path: Path, *, source_dir: Path, build_dir: Path) -> Path:
    plan = {
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "source_manifest_summary": {"status": "prepared"},
        "build_plan": {
            "generator": "Unix Makefiles",
            "arch": "gfx1151",
            "target": "llama",
            "configure_command": ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
            "build_command": ["cmake", "--build", str(build_dir), "--target", "llama"],
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path


def _write_fake_cmake(
    tmp_path: Path, *, fail_configure: bool, fail_build: bool
) -> Path:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    cmake = tool_dir / "cmake"
    configure_rc = 7 if fail_configure else 0
    build_rc = 9 if fail_build else 0
    cmake.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if '--build' in args:\n"
        f"    rc = {build_rc}\n"
        "    build_dir = pathlib.Path(args[args.index('--build') + 1])\n"
        "    if rc == 0:\n"
        "        (build_dir / 'src').mkdir(parents=True, exist_ok=True)\n"
        "        (build_dir / 'src' / 'libllama.so').write_text('fake lib')\n"
        "    print('fake build')\n"
        "    raise SystemExit(rc)\n"
        f"rc = {configure_rc}\n"
        "if rc == 0:\n"
        "    build_dir = pathlib.Path(args[args.index('-B') + 1])\n"
        "    build_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (build_dir / 'CMakeCache.txt').write_text('fake cache')\n"
        "print('fake configure')\n"
        "raise SystemExit(rc)\n"
    )
    cmake.chmod(0o755)
    return tool_dir
