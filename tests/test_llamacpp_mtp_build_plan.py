from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_build_plan import (
    build_plan_artifact,
    choose_generator,
    discover_rocm,
    discover_tools,
    validate_source_tree,
)
from scripts.llamacpp_mtp_layer_input_patch import LAYER_INPUT_ASSIGNMENT


def test_build_plan_ready_with_make_fallback(tmp_path: Path) -> None:
    manifest_path, source_dir = _write_manifest_and_source(tmp_path)
    tool_dir = _write_tools(tmp_path, names=("cmake", "hipcc", "make"))
    rocm = _write_rocm_prefix(tmp_path)
    env = {
        "PATH": str(tool_dir),
        "ROCM_PATH": str(rocm),
        "HIP_PATH": str(rocm),
        "CMAKE_PREFIX_PATH": f"{rocm}:{rocm / 'lib' / 'cmake'}",
    }

    artifact = build_plan_artifact(
        manifest_path=manifest_path,
        build_dir=tmp_path / "build",
        env=env,
    )

    assert artifact["status"] == "ready"
    assert artifact["source_dir"] == str(source_dir)
    assert artifact["build_plan"]["generator"] == "Unix Makefiles"
    assert "-DGGML_HIP=ON" in artifact["build_plan"]["configure_command"]
    assert "-DAMDGPU_TARGETS=gfx1151" in artifact["build_plan"]["configure_command"]
    assert artifact["preflight"]["passed"] is True
    assert artifact["tools"]["ninja"]["required"] is False
    assert "run_configure_and_build" in artifact["next_action"]
    json.dumps(artifact)


def test_build_plan_uses_ninja_when_available(tmp_path: Path) -> None:
    manifest_path, _ = _write_manifest_and_source(tmp_path)
    tool_dir = _write_tools(tmp_path, names=("cmake", "hipcc", "make", "ninja"))
    rocm = _write_rocm_prefix(tmp_path)
    env = {
        "PATH": str(tool_dir),
        "ROCM_PATH": str(rocm),
        "CMAKE_PREFIX_PATH": str(rocm),
    }

    artifact = build_plan_artifact(
        manifest_path=manifest_path,
        build_dir=tmp_path / "build",
        env=env,
    )

    assert choose_generator(artifact["tools"]) == "Ninja"
    assert artifact["build_plan"]["generator"] == "Ninja"


def test_build_plan_reports_missing_rocblas_config(tmp_path: Path) -> None:
    manifest_path, _ = _write_manifest_and_source(tmp_path)
    tool_dir = _write_tools(tmp_path, names=("cmake", "hipcc", "make"))
    rocm = tmp_path / "rocm"
    rocm.mkdir()
    env = {"PATH": str(tool_dir), "ROCM_PATH": str(rocm), "CMAKE_PREFIX_PATH": str(rocm)}

    artifact = build_plan_artifact(
        manifest_path=manifest_path,
        build_dir=tmp_path / "build",
        env=env,
    )

    assert artifact["status"] == "blocked"
    assert artifact["preflight"]["checks"]["rocblas_config_present"] is False
    assert artifact["next_action"].startswith("resolve_build_preflight_blockers")


def test_validate_source_tree_requires_single_layer_input_assignment(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = source / "src" / "models" / "qwen35moe.cpp"
    target.parent.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.21)\n")
    (source / "src" / "llama-ext.h").write_text("// ext\n")
    (source / ".hipengine_llamacpp_mtp_temp_source").write_text("{}\n")
    target.write_text("missing assignment\n")

    validation = validate_source_tree(source)

    assert validation["source_dir_exists"] is True
    assert validation["layer_input_assignment_count"] == 0
    assert validation["layer_input_assignment_line"] is None


def test_discover_tools_marks_ninja_optional(tmp_path: Path) -> None:
    tool_dir = _write_tools(tmp_path, names=("cmake", "hipcc", "make"))

    tools = discover_tools({"PATH": str(tool_dir)})

    assert tools["cmake"]["present"] is True
    assert tools["ninja"]["present"] is False
    assert tools["ninja"]["required"] is False


def test_discover_rocm_finds_nested_cmake_configs(tmp_path: Path) -> None:
    rocm = _write_rocm_prefix(tmp_path)

    result = discover_rocm({"ROCM_PATH": str(rocm), "CMAKE_PREFIX_PATH": str(rocm)})

    assert result["rocblas_config_present"] is True
    assert result["hip_config_present"] is True


def _write_manifest_and_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    target = source / "src" / "models" / "qwen35moe.cpp"
    target.parent.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.21)\n")
    (source / "src" / "llama-ext.h").write_text("// ext\n")
    (source / ".hipengine_llamacpp_mtp_temp_source").write_text("{}\n")
    target.write_text(
        "void f() {\n"
        "    for (int il = 0; il < n_layer; ++il) {\n"
        f"{LAYER_INPUT_ASSIGNMENT}\n"
        "        ggml_tensor * inpSA = inpL;\n"
        "    }\n"
        "}\n"
    )
    manifest = {
        "status": "prepared",
        "output_dir": str(source),
        "reference_basis": {
            "observed_commit": "abc123",
            "commit_matches_expected": True,
            "external_checkout_modified": False,
        },
        "patch": {"sha256": "patch-sha"},
        "target_validation": {"patch_applied": True, "assignment_line": 3},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, source


def _write_tools(tmp_path: Path, *, names: tuple[str, ...]) -> Path:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir(exist_ok=True)
    for name in names:
        tool = tool_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    return tool_dir


def _write_rocm_prefix(tmp_path: Path) -> Path:
    rocm = tmp_path / "rocm"
    (rocm / "lib" / "cmake" / "rocblas").mkdir(parents=True)
    (rocm / "lib" / "cmake" / "hip").mkdir(parents=True)
    (rocm / "lib" / "cmake" / "rocblas" / "rocblas-config.cmake").write_text("# rocblas\n")
    (rocm / "lib" / "cmake" / "hip" / "hip-config.cmake").write_text("# hip\n")
    return rocm
