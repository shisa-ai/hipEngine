from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_build_pre_output_norm_harness import (
    apply_pre_output_patch,
    build_pre_output_norm_harness,
    rewrite_build_command,
    rewrite_configure_command,
    safe_rmtree,
)

OLD_TEXT = (
    "    cur = inpL;\n\n"
    "    // post-norm hidden state feeds both the LM head and the MTP seed below\n"
    "    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);"
)
NEW_TEXT = (
    "    cur = inpL;\n\n"
    "    // PRE-output_norm diagnostic: expose final decoder output through h_nextn.\n"
    "    cb(cur, \"h_nextn_pre_output_norm\", -1);\n"
    "    res->t_h_nextn = cur;\n\n"
    "    // post-norm hidden state feeds both the LM head and the MTP seed below\n"
    "    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);"
)


def test_apply_pre_output_patch_is_idempotent(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path / "src")

    first = apply_pre_output_patch(source_dir=source, old_text=OLD_TEXT, new_text=NEW_TEXT)
    second = apply_pre_output_patch(source_dir=source, old_text=OLD_TEXT, new_text=NEW_TEXT)

    text = (source / "src" / "models" / "qwen35moe.cpp").read_text()
    assert first["status"] == "applied"
    assert second["status"] == "already_applied"
    assert text.count(NEW_TEXT) == 1


def test_rewrite_configure_and_build_commands() -> None:
    configure = rewrite_configure_command(
        ["cmake", "-S", "old-src", "-B", "old-build", "-DYES=1"],
        source_dir=Path("new-src"),
        build_dir=Path("new-build"),
    )
    build = rewrite_build_command(
        ["cmake", "--build", "old-build", "--target", "llama", "-j", "8"],
        build_dir=Path("new-build"),
        jobs=3,
    )

    assert configure[:5] == ["cmake", "-S", "new-src", "-B", "new-build"]
    assert build == ["cmake", "--build", "new-build", "--target", "llama", "-j", "3"]


def test_build_pre_output_norm_harness_with_fake_tools(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path / "base-src")
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_fake_cmake(tools / "cmake")
    compiler = _write_fake_compiler(tools / "fake-c++")
    base_build = _write_base_build(tmp_path, source)
    plan = _write_plan(tmp_path)

    artifact = build_pre_output_norm_harness(
        plan_path=plan,
        base_build_path=base_build,
        output_path=tmp_path / "summary.json",
        patched_build_result_path=tmp_path / "patched-build.json",
        harness_compile_path=tmp_path / "harness-compile.json",
        target_source_dir=tmp_path / "patched-src",
        target_build_dir=tmp_path / "patched-build",
        log_dir=tmp_path / "logs",
        harness_dir=tmp_path / "harness",
        compiler=str(compiler),
        jobs=2,
        timeout_seconds=30,
        clean=True,
        env={"PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")},
    )

    assert artifact["status"] == "ready"
    assert artifact["patch"]["applied"] is True
    assert artifact["commands"]["configure"]["returncode"] == 0
    assert artifact["commands"]["build"]["returncode"] == 0
    assert artifact["harness_compile"]["status"] == "compiled"
    assert artifact["libraries"][0].endswith("libllama.so")
    assert artifact["next_action"] == "run_pre_output_norm_harness_and_compare_hipengine_serial_row"
    assert (tmp_path / "patched-build.json").exists()
    assert (tmp_path / "harness-compile.json").exists()
    json.dumps(artifact)


def test_build_pre_output_norm_harness_reports_configure_failure(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path / "base-src")
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_fake_cmake(tools / "cmake", fail_configure=True)
    compiler = _write_fake_compiler(tools / "fake-c++")
    base_build = _write_base_build(tmp_path, source)
    plan = _write_plan(tmp_path)

    artifact = build_pre_output_norm_harness(
        plan_path=plan,
        base_build_path=base_build,
        output_path=tmp_path / "summary.json",
        patched_build_result_path=tmp_path / "patched-build.json",
        harness_compile_path=tmp_path / "harness-compile.json",
        target_source_dir=tmp_path / "patched-src",
        target_build_dir=tmp_path / "patched-build",
        log_dir=tmp_path / "logs",
        harness_dir=tmp_path / "harness",
        compiler=str(compiler),
        jobs=2,
        timeout_seconds=30,
        clean=True,
        env={"PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")},
    )

    assert artifact["status"] == "configure_failed"
    assert artifact["commands"]["configure"]["returncode"] == 19
    assert artifact["commands"]["build"]["returncode"] is None
    assert artifact["harness_compile"]["status"] == "skipped"


def test_safe_rmtree_rejects_non_tmp_path() -> None:
    try:
        safe_rmtree(Path("/home/lhl/not-a-temp-dir"))
    except ValueError as exc:
        assert "refusing" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def _write_source_tree(path: Path) -> Path:
    (path / "src" / "models").mkdir(parents=True)
    (path / "include").mkdir()
    (path / "ggml" / "include").mkdir(parents=True)
    (path / "include" / "llama.h").write_text("// llama\n")
    (path / "src" / "llama-ext.h").write_text("// ext\n")
    (path / "src" / "models" / "qwen35moe.cpp").write_text(OLD_TEXT + "\n")
    return path


def _write_base_build(tmp_path: Path, source: Path) -> Path:
    build_dir = tmp_path / "base-build"
    build_dir.mkdir()
    artifact = {
        "source_dir": str(source),
        "build_dir": str(build_dir),
        "commands": {
            "configure": {
                "command": ["cmake", "-S", str(source), "-B", str(build_dir), "-DFAKE=1"]
            },
            "build": {
                "command": ["cmake", "--build", str(build_dir), "--target", "llama", "-j", "8"]
            },
        },
    }
    path = tmp_path / "base-build.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_plan(tmp_path: Path) -> Path:
    artifact = {
        "llamacpp_pre_output_patch": {"old_text": OLD_TEXT, "new_text": NEW_TEXT}
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_fake_cmake(path: Path, *, fail_configure: bool = False) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        f"fail_configure = {str(fail_configure)}\n"
        "if '-S' in args:\n"
        "    if fail_configure:\n"
        "        print('fake configure failed', file=sys.stderr)\n"
        "        raise SystemExit(19)\n"
        "    build = pathlib.Path(args[args.index('-B') + 1])\n"
        "    (build / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "    (build / 'CMakeCache.txt').write_text('fake cache\\n')\n"
        "    print('fake configure ok')\n"
        "    raise SystemExit(0)\n"
        "if '--build' in args:\n"
        "    build = pathlib.Path(args[args.index('--build') + 1])\n"
        "    (build / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "    for name in ['libllama.so', 'libllama.so.0', 'libllama.so.0.0.0']:\n"
        "        (build / 'bin' / name).write_text('fake lib\\n')\n"
        "    print('fake build ok')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    path.chmod(0o755)
    return path


def _write_fake_compiler(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, stat, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(\"#!/bin/sh\\necho fake capture\\n\")\n"
        "out.chmod(out.stat().st_mode | stat.S_IXUSR)\n"
        "print('fake compile ok')\n"
    )
    path.chmod(0o755)
    return path
