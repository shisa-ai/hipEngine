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


def test_plan_hip_build_uses_isolated_environment_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    isolated = tmp_path / "isolated-cache"
    monkeypatch.setenv("HIPENGINE_BUILD_CACHE_ROOT", str(isolated))

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        compiler_version="hipcc test version",
    )

    assert artifact.cache_dir.parent == isolated
    assert not isolated.exists()


def test_build_hip_environment_require_cached_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    monkeypatch.setenv("HIPENGINE_BUILD_CACHE_ROOT", str(tmp_path / "isolated-cache"))
    monkeypatch.setenv("HIPENGINE_REQUIRE_CACHED_BUILD", "1")
    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_TEXT", "hipcc test version")

    with pytest.raises(FileNotFoundError, match="cached build artifact missing"):
        build_hip(
            sources=[source],
            family="smoke",
            compiler="definitely-not-a-real-hipcc",
            load=False,
        )


def test_build_hip_dry_run_does_not_create_cache_or_run_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    monkeypatch.delenv("HIPENGINE_HIP_ARCH", raising=False)
    monkeypatch.delenv("HIPENGINE_HIP_OFFLOAD_ARCH", raising=False)

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


def _write_fake_cached_artifact(
    tmp_path: Path, source: Path, *, target_arch: str | None = None, cache_root: Path | None = None
) -> Path:
    """Materialise the cached ``.so`` the build machinery expects to find."""

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=cache_root or tmp_path / "cache",
        compiler="definitely-not-a-real-hipcc",
        compiler_version="hipcc cached test version",
        target_arch=target_arch,
    )
    artifact.cache_dir.mkdir(parents=True, exist_ok=True)
    artifact.output_path.write_bytes(b"not a real shared object")
    return artifact.output_path


def test_build_hip_per_launch_load_does_not_rederive_its_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated per-launch ``build_hip(load=True)`` must not re-derive its key.

    The kernel launchers resolve their library on every launch (they do not pass
    ``library=``), so the compiler-version probe, the target-arch read and the
    cache-root resolution run once per *kernel launch*: measured at 8.0 us of an
    11.8 us host launch cost, against 1.8 us for the bare ctypes call. The fast
    path has to short-circuit on the raw call inputs plus the environment values
    those derivations read, so the derivation runs once per distinct request.
    """

    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    _write_fake_cached_artifact(tmp_path, source)
    monkeypatch.setattr(build_module, "_LOADED_LIB_CACHE", {})
    monkeypatch.setattr(build_module, "_FAST_PATH_CACHE", {})
    monkeypatch.setattr(build_module.ctypes, "CDLL", lambda path: Path(path))

    calls = {"n": 0}
    real_resolve = build_module._resolve_compiler_version

    def counting_resolve(*args: object, **kwargs: object) -> str:
        calls["n"] += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(build_module, "_resolve_compiler_version", counting_resolve)

    def load() -> object:
        return build_hip(
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-hipcc",
            compiler_version="hipcc cached test version",
            load=True,
            require_cached=True,
        )

    first = load()
    for _ in range(5):
        assert load() == first

    assert calls["n"] == 1, (
        "build_hip re-derived its key on a repeated per-launch load; "
        f"expected 1 derivation, saw {calls['n']}"
    )


def test_build_hip_fast_path_honours_environment_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-process environment change must select a fresh resolution.

    The fast path is keyed on the environment values the key derivation reads,
    so pointing ``HIPENGINE_BUILD_CACHE_ROOT`` at a different cache must not
    return the previously loaded library.
    """

    source = write_source(tmp_path / "smoke.hip", "extern \"C\" void smoke_host() {}\n")
    root_a = tmp_path / "cache-a"
    root_b = tmp_path / "cache-b"
    path_a = _write_fake_cached_artifact(tmp_path, source, cache_root=root_a)
    path_b = _write_fake_cached_artifact(tmp_path, source, cache_root=root_b)
    monkeypatch.setattr(build_module, "_LOADED_LIB_CACHE", {})
    monkeypatch.setattr(build_module, "_FAST_PATH_CACHE", {})
    monkeypatch.setattr(build_module.ctypes, "CDLL", lambda path: Path(path))
    monkeypatch.delenv("HIPENGINE_HIP_ARCH", raising=False)
    monkeypatch.delenv("HIPENGINE_HIP_OFFLOAD_ARCH", raising=False)

    def load() -> object:
        return build_hip(
            sources=[source],
            family="smoke",
            profile="baseline",
            compiler="definitely-not-a-real-hipcc",
            compiler_version="hipcc cached test version",
            load=True,
            require_cached=True,
        )

    monkeypatch.setenv("HIPENGINE_BUILD_CACHE_ROOT", str(root_a))
    assert load() == path_a
    monkeypatch.setenv("HIPENGINE_BUILD_CACHE_ROOT", str(root_b))
    assert load() == path_b


def _fast_key(compiler: str = "hipcc", source: str = "a.hip") -> tuple | None:
    return build_module._build_fast_key(
        family="smoke",
        profile="baseline",
        output_name=None,
        sources=[source],
        cache_root=None,
        compiler=compiler,
        compiler_version=None,
        target_arch=None,
        include_dirs=[],
        extra_flags=[],
    )


def test_build_fast_key_changes_with_every_build_environment_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each listed knob must actually move the key.

    ``_build_fast_key`` treats an unlisted environment variable as constant for
    the process lifetime, so a knob that changes what a build resolves to has to
    change the key. This catches a rename or a typo in ``_BUILD_ENV_KEYS``.
    """

    for name in build_module._BUILD_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HIPENGINE_COMPILER_VERSION_TEXT", raising=False)
    monkeypatch.delenv("HIPENGINE_COMPILER_VERSION_FILE", raising=False)
    baseline = _fast_key()
    assert baseline is not None

    for name in build_module._BUILD_ENV_KEYS:
        monkeypatch.setenv(name, "fast-key-probe")
        assert _fast_key() != baseline, f"{name} does not affect _build_fast_key"
        monkeypatch.delenv(name)

    monkeypatch.setenv("HIPENGINE_COMPILER_VERSION_TEXT", "v-probe")
    assert _fast_key() != baseline, "the version override does not affect _build_fast_key"


def test_build_fast_key_lists_every_environment_knob_the_build_path_reads() -> None:
    """No environment knob may be read by the build path without being listed.

    The fast path assumes an unlisted variable cannot change a build. A new
    ``HIPENGINE_*`` read anywhere in this module would silently break that, so
    the module's own source is the oracle.
    """

    import re

    source = Path(build_module.__file__).read_text()
    read_names = set(re.findall(r'"(HIPENGINE_[A-Z0-9_]+|HIP_DEVICE_LIB_PATH)"', source))
    # The compiler-version overrides are dynamic per compiler
    # (``HIPENGINE_<PREFIX>_VERSION_TEXT``), so they are covered as a family by
    # ``_environment_version_identity`` rather than named in ``_BUILD_ENV_KEYS``.
    covered_by_identity = {
        "HIPENGINE_COMPILER_VERSION_TEXT",
        "HIPENGINE_COMPILER_VERSION_FILE",
    }
    # CUDA-path knobs: read only by ``build_cuda``, which has no fast path, so
    # they cannot make a HIP key stale.
    covered_elsewhere = {
        "HIPENGINE_CUDA_ARCH",
        "HIPENGINE_CUDA_TARGET_ARCH",
    }
    unlisted = (
        read_names
        - set(build_module._BUILD_ENV_KEYS)
        - covered_by_identity
        - covered_elsewhere
    )
    assert not unlisted, (
        "these environment knobs are read by hipengine/core/build.py but are not in "
        f"_BUILD_ENV_KEYS, so _build_fast_key would treat them as constant: {sorted(unlisted)}"
    )


def test_env_get_falls_back_when_environ_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast env read must stay correct if ``os.environ`` is swapped out.

    ``_env_get`` reads the raw dict behind ``os.environ`` to avoid its per-access
    cost. A caller that replaces ``os.environ`` with a different object (tests do
    this) must still be read correctly, not through the stale dict.
    """

    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx-from-real-environ")
    assert build_module._env_get("HIPENGINE_HIP_ARCH") == "gfx-from-real-environ"

    monkeypatch.setattr(build_module.os, "environ", {"HIPENGINE_HIP_ARCH": "gfx-from-replacement"})
    assert build_module._env_get("HIPENGINE_HIP_ARCH") == "gfx-from-replacement"
    assert build_module._env_get("HIPENGINE_BUILD_CACHE_ROOT") == ""
