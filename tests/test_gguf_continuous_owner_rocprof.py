"""Host-only tests for mechanical owner-profile cache preparation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "gguf_continuous_owner_rocprof.py"
    module_name = "_gguf_continuous_owner_rocprof_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_cache_snapshot_hashes_content_mode_and_mtime(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    artifact = cache / "kernel.so"
    artifact.write_bytes(b"first")

    before = SCRIPT.snapshot_cache_tree(cache)
    os.utime(artifact, ns=(artifact.stat().st_atime_ns, artifact.stat().st_mtime_ns + 1))
    after_mtime = SCRIPT.snapshot_cache_tree(cache)
    artifact.write_bytes(b"second")
    after_content = SCRIPT.snapshot_cache_tree(cache)

    assert before["tree_sha256"] != after_mtime["tree_sha256"]
    assert after_mtime["tree_sha256"] != after_content["tree_sha256"]
    assert before["file_count"] == 1


def test_cache_manifest_summary_records_profile_key_and_output_hash(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "family-key"
    cache.mkdir(parents=True)
    (cache / "manifest.txt").write_text(
        "family=family\nprofile=decode\ncache_key=abc123\n"
        "target_arch=gfx1151\ncompiler_version<<EOF\nversion\nEOF\n",
        encoding="utf-8",
    )
    (cache / "family.so").write_bytes(b"code-object")

    summary = SCRIPT.cache_build_manifest_summary(tmp_path / "cache")

    assert summary[0]["family"] == "family"
    assert summary[0]["profile"] == "decode"
    assert summary[0]["cache_key"] == "abc123"
    assert summary[0]["target_arch"] == "gfx1151"
    assert set(summary[0]["output_sha256"]) == {"family.so"}


def test_compiler_guard_fails_and_leaves_marker(tmp_path: Path) -> None:
    guard = SCRIPT.prepare_compiler_guard(tmp_path / "guard")

    result = subprocess.run(
        [str(guard["directory"] / "hipcc"), "--version"],
        check=False,
    )

    assert result.returncode == 97
    assert guard["marker"].is_file()


def test_owner_child_command_carries_isolated_cache_and_fail_closed_flag(
    tmp_path: Path,
) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--model",
            str(tmp_path / "model.gguf"),
            "--compiler-version-file",
            str(tmp_path / "hipcc-version.txt"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--run-root",
            str(tmp_path / "runs"),
            "--run-tag",
            "unit",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    command = SCRIPT._child_command(
        args,
        output=tmp_path / "child.json",
        profile=True,
        require_cached=True,
    )

    assert command[1] == "scripts/gguf_continuous_owner_profile_child.py"
    assert "--profile" in command
    assert "--require-cached-build" in command
    assert command[command.index("--cache-root") + 1] == str(tmp_path / "cache")
    assert command[command.index("--compiler-version-file") + 1] == str(
        tmp_path / "hipcc-version.txt"
    )


def test_profile_command_wraps_only_final_child(tmp_path: Path) -> None:
    child = [
        "/venv/bin/python",
        "scripts/gguf_continuous_owner_profile_child.py",
        "--profile",
        "--require-cached-build",
    ]

    command = SCRIPT.profile_command(
        rocprofv3="rocprofv3",
        trace_dir=tmp_path / "trace",
        child_command=child,
    )

    assert command[:9] == [
        "rocprofv3",
        "--kernel-trace",
        "--marker-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--output-format",
        "csv",
        "-d",
        str(tmp_path / "trace"),
    ]
    assert command[9:] == ["--", *child]
    assert command.count("scripts/gguf_continuous_owner_profile_child.py") == 1


def test_cache_only_validation_rejects_mutation_or_compiler_activity(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.so").write_bytes(b"same")
    before = SCRIPT.snapshot_cache_tree(cache)
    after = SCRIPT.snapshot_cache_tree(cache)
    marker = tmp_path / "compiler-invoked"

    SCRIPT.validate_cache_only_stage(
        before=before,
        after=after,
        compiler_guard_marker=marker,
        observed_compiler_processes=(),
    )

    marker.write_text("hipcc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="compiler guard"):
        SCRIPT.validate_cache_only_stage(
            before=before,
            after=after,
            compiler_guard_marker=marker,
            observed_compiler_processes=(),
        )
    marker.unlink()
    with pytest.raises(ValueError, match="cache mutated"):
        SCRIPT.validate_cache_only_stage(
            before=before,
            after={**after, "tree_sha256": "different"},
            compiler_guard_marker=marker,
            observed_compiler_processes=(),
        )
    with pytest.raises(ValueError, match="compiler subprocess"):
        SCRIPT.validate_cache_only_stage(
            before=before,
            after=after,
            compiler_guard_marker=marker,
            observed_compiler_processes=("hipcc -shared",),
        )
