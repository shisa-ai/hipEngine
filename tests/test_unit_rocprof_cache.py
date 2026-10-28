"""Shared profiling helpers keep cache-only checks outside workload scripts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hipengine.benchmark import rocprof_cache as cache
from scripts import gguf_continuous_owner_rocprof as owner
from scripts import qwen38_gfx1151_prefill_kernel_profile as prefill


def test_profile_entry_points_share_cache_helpers() -> None:
    for name in (
        "_default_roctx_sdk", "_prepare_roctx_override", "_run_monitored",
        "_write_json", "prepare_compiler_guard", "snapshot_cache_tree",
        "validate_cache_only_stage", "_cache_only_environment",
    ):
        assert getattr(owner, name) is getattr(cache, name)
        assert getattr(prefill, name) is getattr(cache, name)


def test_cache_only_environment_preserves_inputs_and_guard_precedence(tmp_path) -> None:
    base = {"PATH": "/original/bin", "HIPENGINE_REQUIRE_CACHED_BUILD": "0", "OTHER": "same"}
    guard = cache.prepare_compiler_guard(tmp_path / "guard")
    selected = cache._cache_only_environment(base, guard=guard)
    assert selected == {**base, "PATH": str(guard["directory"]) + os.pathsep + base["PATH"], "HIPENGINE_REQUIRE_CACHED_BUILD": "1"}
    assert base["HIPENGINE_REQUIRE_CACHED_BUILD"] == "0"
    assert base["PATH"] == "/original/bin"


def test_cache_snapshot_detects_mode_change(tmp_path) -> None:
    artifact = tmp_path / "kernel.so"
    artifact.write_bytes(b"same")
    artifact.chmod(0o600)
    before = cache.snapshot_cache_tree(tmp_path)
    artifact.chmod(0o700)
    after = cache.snapshot_cache_tree(tmp_path)
    assert before["tree_sha256"] != after["tree_sha256"]
    assert before["files"][0]["sha256"] == after["files"][0]["sha256"]


def test_roctx_override_rejects_missing_sdk_and_links_existing_sdk(tmp_path) -> None:
    sdk = tmp_path / "sdk.so"
    with pytest.raises(FileNotFoundError, match="SDK ROCTX library not found"):
        cache._prepare_roctx_override(sdk, tmp_path / "run")
    sdk.write_bytes(b"sdk")
    override, dependencies = cache._prepare_roctx_override(sdk, tmp_path / "run")
    assert (override / "libroctx64.so").resolve() == sdk
    assert sdk.parent in dependencies


def _run(tmp_path, command, timeout=5):
    return cache._run_monitored(
        command, cwd=tmp_path, environment=os.environ,
        stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr",
        timeout_seconds=timeout,
    )


def test_monitored_child_preserves_logs_and_exit_status(tmp_path) -> None:
    result = _run(tmp_path, [sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr);sys.exit(7)"])
    assert result["returncode"] == 7
    assert Path(result["stdout"]).read_text() == "out\n"
    assert Path(result["stderr"]).read_text() == "err\n"
    assert result["observed_compiler_processes"] == []


def test_monitored_child_detects_compiler_and_cache_gate_rejects_it(tmp_path) -> None:
    compiler = tmp_path / "hipcc"
    compiler.write_text("#!/bin/sh\nsleep 0.3\n")
    compiler.chmod(0o755)
    child = "import subprocess,sys; subprocess.run([sys.argv[1]],check=True)"
    result = _run(tmp_path, [sys.executable, "-c", child, str(compiler)])
    assert result["returncode"] == 0
    assert result["observed_compiler_processes"]
    with pytest.raises(ValueError, match="compiler subprocess"):
        cache.validate_cache_only_stage(
            before={"tree_sha256": "same"}, after={"tree_sha256": "same"},
            compiler_guard_marker=tmp_path / "marker",
            observed_compiler_processes=result["observed_compiler_processes"],
        )


def test_monitored_child_timeout_raises_bounded_error(tmp_path) -> None:
    with pytest.raises(cache.RocprofCacheError, match="command timed out"):
        _run(tmp_path, [sys.executable, "-c", "import time;time.sleep(60)"], timeout=0.1)


def test_guard_marker_cannot_be_reused(tmp_path) -> None:
    guard = cache.prepare_compiler_guard(tmp_path)
    subprocess.run([str(guard["hipcc"])], check=False)
    with pytest.raises(ValueError, match="marker already exists"):
        cache.prepare_compiler_guard(tmp_path)


@pytest.mark.parametrize("failure", [None, "mutation", "compiler"])
def test_owner_workflow_uses_shared_guards_across_real_child_processes(monkeypatch, tmp_path, failure) -> None:
    import json

    model, compiler_file, sdk = (tmp_path / name for name in ("model.gguf", "version.txt", "sdk.so"))
    for path in (model, compiler_file, sdk):
        path.write_text("fixture")
    args = owner.build_parser().parse_args([
        "--source-root", str(tmp_path), "--model", str(model),
        "--compiler-version-file", str(compiler_file), "--cache-root", str(tmp_path / "cache"),
        "--run-root", str(tmp_path / "runs"), "--run-tag", "fixture",
        "--out", str(tmp_path / "result.json"), "--rebuild", "--profile", "--roctx-sdk", str(sdk),
    ])
    monkeypatch.setattr(owner, "_git_clean", lambda root: True)
    monkeypatch.setattr(owner, "_git_value", lambda *args: "fixture-commit")
    monkeypatch.setattr(owner, "collect_model_identity", lambda path: {"path": str(path)})
    # Only the workload and profiler executables are synthetic. The parent runs
    # its real subprocess, cache snapshot, guard and stage validation code.
    child = """import json,os,subprocess,sys
from pathlib import Path
root=Path(os.environ['HIPENGINE_BUILD_CACHE_ROOT'])
cached=sys.argv[2]=='1'
profile=sys.argv[3]=='1'
failure=sys.argv[4]
if not cached:
    assert 'HIPENGINE_REQUIRE_CACHED_BUILD' not in os.environ
    root.mkdir(exist_ok=True)
    (root/'kernel.so').write_bytes(b'code')
    (root/'manifest.txt').write_text('family=fixture\\nprofile=fixture\\ncache_key=fixture\\n')
else:
    assert os.environ['HIPENGINE_REQUIRE_CACHED_BUILD']=='1'
    assert (Path(os.environ['PATH'].split(os.pathsep)[0])/'hipcc').is_file()
    if not profile and failure=='mutation': (root/'kernel.so').write_bytes(b'changed')
    if not profile and failure=='compiler': subprocess.run(['hipcc'],check=False)
Path(sys.argv[1]).write_text(json.dumps({'profile':profile,'require_cached_build':cached}))
"""

    def child_command(args, *, output, profile, require_cached):
        return [sys.executable, "-c", child, str(output), str(int(require_cached)), str(int(profile)), str(failure)]

    monkeypatch.setattr(owner, "_child_command", child_command)
    monkeypatch.setattr(owner, "profile_command", lambda **kwargs: kwargs["child_command"])
    if failure:
        with pytest.raises(ValueError, match="cache mutated" if failure == "mutation" else "compiler guard"):
            owner.run(args)
        assert not args.out.exists()
    else:
        result = owner.run(args)
        assert result["cache"]["immutable_cache_only"] is True
        assert result["compiler_guard"]["invoked"] is False
        assert set(result["stages"]) == {"build", "cache_warm", "profile"}
        assert all(stage["returncode"] == 0 for stage in result["stages"].values())
        assert json.loads(args.out.read_text()) == result
