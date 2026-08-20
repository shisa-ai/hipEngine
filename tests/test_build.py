from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core import build as build_module
from hipengine.core.build import build_hip, plan_hip_build


def write_source(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_plan_hip_build_hashes_source_flags_and_compiler_version(tmp_path: Path) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" __global__ void smoke() {}\n")

    artifact_a = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    artifact_b = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    artifact_c = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="prefill",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    artifact_d = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="different hipcc",
    )

    assert artifact_a.cache_key == artifact_b.cache_key
    assert artifact_a.cache_key != artifact_c.cache_key
    assert artifact_a.cache_key != artifact_d.cache_key
    assert artifact_a.cache_dir.name.startswith("smoke-")
    assert artifact_a.output_path.name == "smoke.so"
    assert artifact_a.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact_a.flags
    assert "-mwavefrontsize64" not in artifact_a.flags
    assert artifact_a.profile.wavefront == 32


def test_plan_hip_build_can_disable_unroll600_for_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" __global__ void smoke() {}\n")

    default = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    monkeypatch.setenv("HIPENGINE_DISABLE_UNROLL600", "1")
    no_unroll = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )

    assert default.cache_key != no_unroll.cache_key
    assert default.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mllvm" not in no_unroll.flags
    assert "-amdgpu-unroll-threshold-local=600" not in no_unroll.flags
    assert "-mcumode" in no_unroll.flags


def test_plan_hip_build_can_enable_prefill_mcumode_for_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" __global__ void smoke() {}\n")

    default = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="prefill",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    monkeypatch.setenv("HIPENGINE_PREFILL_MCUMODE", "1")
    with_mcumode = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="prefill",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )
    decode = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="decode",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )

    assert default.cache_key != with_mcumode.cache_key
    assert "-mcumode" not in default.flags
    assert with_mcumode.flags[-1] == "-mcumode"
    assert decode.flags.count("-mcumode") == 1


def test_build_hip_dry_run_does_not_create_cache_or_run_compiler(tmp_path: Path) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")

    artifact = build_hip(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler="definitely-not-a-real-hipcc",
        dry_run=True,
        load=False,
    )

    assert artifact.command[0] == "definitely-not-a-real-hipcc"
    assert artifact.profile.name == "baseline"
    assert artifact.flags == ()
    assert not artifact.cache_dir.exists()


def test_resolve_compiler_version_explicit_becomes_process_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit version becomes the process default for per-call loads.

    The per-call launch path calls ``build_X(load=True)`` with
    ``compiler_version=None``. When a session pins an explicit version, the
    loaded-library cache would be missed (the None path resolving a different
    version), re-running build_hip machinery on every launch. Cache the explicit
    version so later None calls resolve to it and hit ``_LOADED_LIB_CACHE``.
    """

    monkeypatch.setattr(build_module, "_COMPILER_VERSION_CACHE", {})
    resolved = build_module._resolve_compiler_version(
        compiler="hipcc", compiler_version="v-pinned 1", dry_run=False
    )
    assert resolved == "v-pinned 1"
    assert build_module._COMPILER_VERSION_CACHE == {"hipcc": "v-pinned 1"}

    # A later per-call (compiler_version=None) resolves to the pinned version
    # instead of probing a potentially different installed compiler.
    resolved_none = build_module._resolve_compiler_version(
        compiler="hipcc", compiler_version=None, dry_run=False
    )
    assert resolved_none == "v-pinned 1"


def test_resolve_compiler_version_explicit_strip_and_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_module, "_COMPILER_VERSION_CACHE", {})
    resolved = build_module._resolve_compiler_version(
        compiler="hipcc", compiler_version="  v-pinned 2\n", dry_run=False
    )
    assert resolved == "v-pinned 2"
    assert build_module._COMPILER_VERSION_CACHE == {"hipcc": "v-pinned 2"}
    # dry_run must not touch the cache (no side effect for planning).
    build_module._resolve_compiler_version(
        compiler="clang", compiler_version="dry", dry_run=True
    )
    assert "clang" not in build_module._COMPILER_VERSION_CACHE


def test_build_hip_uses_version_file_for_cached_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    version = "hipcc cached test version"
    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text(version + "\n")
    expected = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler="definitely-not-a-real-hipcc",
        compiler_version=version,
    )
    expected.cache_dir.mkdir(parents=True)
    expected.output_path.write_bytes(b"not a real shared object")
    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_FILE", str(version_file))

    artifact = build_hip(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler="definitely-not-a-real-hipcc",
        load=False,
        require_cached=True,
    )

    assert artifact.cache_key == expected.cache_key
    assert artifact.output_path == expected.output_path
    assert artifact.compiler_version == version


def test_build_hip_loaded_cache_distinguishes_environment_target_arch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    version = "hipcc cached test version"
    artifacts = {
        target_arch: plan_hip_build(
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-hipcc",
            compiler_version=version,
            target_arch=target_arch,
        )
        for target_arch in ("gfx1100", "gfx1151")
    }
    for artifact in artifacts.values():
        artifact.cache_dir.mkdir(parents=True)
        artifact.output_path.write_bytes(b"not a real shared object")

    monkeypatch.setattr(build_module, "_LOADED_LIB_CACHE", {})
    monkeypatch.setattr(build_module.ctypes, "CDLL", lambda path: Path(path))

    loaded = {}
    for target_arch in ("gfx1100", "gfx1151"):
        monkeypatch.setenv("HIPENGINE_HIP_ARCH", target_arch)
        loaded[target_arch] = build_hip(
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-hipcc",
            compiler_version=version,
            load=True,
            require_cached=True,
        )

    assert loaded == {
        target_arch: artifact.output_path for target_arch, artifact in artifacts.items()
    }


def test_build_hip_require_cached_rejects_missing_artifact_without_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    version_file = tmp_path / "hipcc-version.txt"
    version_file.write_text("hipcc cached test version\n")
    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_FILE", str(version_file))

    with pytest.raises(FileNotFoundError, match="cached build artifact missing"):
        build_hip(
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-hipcc",
            load=False,
            require_cached=True,
        )


def test_plan_hip_build_rejects_bad_profile_and_missing_source(tmp_path: Path) -> None:
    source = write_source(tmp_path / "smoke.hip", "// ok\n")

    with pytest.raises(ValueError, match="unknown build profile"):
        plan_hip_build(
            sources=[source],
            family="smoke",
            profile="bogus",  # type: ignore[arg-type]
            compiler_version="hipcc test version",
        )

    with pytest.raises(FileNotFoundError):
        plan_hip_build(
            sources=[tmp_path / "missing.hip"],
            family="smoke",
            compiler_version="hipcc test version",
        )
