"""Launch-library handle lifetime and build-identity separation (R0446).

``hipengine/core/build.py`` keeps three process-level caches in front of the HIP
build: ``_FAST_PATH_CACHE``, ``_RESOLUTION_CACHE`` and ``_LOADED_LIB_CACHE``.
``tests/test_unit_build.py`` already pins warm identity, environment
observation, explicit invalidation, missing cached artifacts and
environment-derived target-arch separation. This module covers the remaining
lifetime contracts R0446 names, using a fake ``ctypes.CDLL`` loader so no GPU,
compiler, or real shared object is involved:

- explicitly parameterized builds of one family stay distinct through
  interleaved warm calls (variant flags, compiler version, arch, profile);
- a handle the caller already holds stays usable after invalidation, while the
  next call resolves a fresh handle;
- invalidation does not resurrect a handle whose artifact has disappeared, and
  the ``require_cached`` refusal still fires after it;
- an explicit ``target_arch`` pins identity across an environment arch change;
- invalidation drops handle ownership without dropping the process-pinned
  compiler version;
- HIP and CUDA resolutions of one family do not share a handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core import build as build_module
from hipengine.core.build import (
    BuildArtifact,
    build_cuda,
    build_hip,
    plan_cuda_build,
    plan_hip_build,
)

COMPILER = "definitely-not-a-real-hipcc"
VERSION = "hipcc cached test version"


class FakeLibrary:
    """Stand-in for ``ctypes.CDLL`` that records which path was loaded."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.exports = {"smoke_host": object()}
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _write_source(tmp_path: Path) -> Path:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" void smoke_host() {}\n')
    return source


def _artifact(
    tmp_path: Path,
    source: Path,
    *,
    profile: str = "baseline",
    arch: str | None = "gfx1100",
    extra_flags: tuple[str, ...] = (),
    compiler_version: str = VERSION,
) -> BuildArtifact:
    """Materialise the cached ``.so`` a warm ``build_hip`` expects to find."""

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile=profile,  # type: ignore[arg-type]
        cache_root=tmp_path / "cache",
        compiler=COMPILER,
        compiler_version=compiler_version,
        target_arch=arch,
        extra_flags=extra_flags,
    )
    artifact.cache_dir.mkdir(parents=True, exist_ok=True)
    artifact.output_path.write_bytes(b"not a real shared object")
    return artifact


def _build(
    tmp_path: Path,
    source: Path,
    *,
    profile: str = "baseline",
    arch: str | None = "gfx1100",
    extra_flags: tuple[str, ...] = (),
    compiler_version: str | None = VERSION,
    require_cached: bool = True,
) -> FakeLibrary:
    return build_hip(  # type: ignore[return-value]
        sources=[source],
        family="smoke",
        profile=profile,  # type: ignore[arg-type]
        cache_root=tmp_path / "cache",
        compiler=COMPILER,
        compiler_version=compiler_version,
        target_arch=arch,
        extra_flags=extra_flags,
        load=True,
        require_cached=require_cached,
    )


@pytest.fixture(autouse=True)
def _isolated_build_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_FAST_PATH_CACHE",
        "_RESOLUTION_CACHE",
        "_LOADED_LIB_CACHE",
        "_COMPILER_VERSION_CACHE",
        "_ENV_VERSION_CACHE",
    ):
        monkeypatch.setattr(build_module, name, {})


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[FakeLibrary]:
    loaded: list[FakeLibrary] = []

    def fake_cdll(path: str) -> FakeLibrary:
        library = FakeLibrary(str(path))
        loaded.append(library)
        return library

    monkeypatch.setattr(build_module.ctypes, "CDLL", fake_cdll)
    return loaded


def test_interleaved_explicit_variants_keep_distinct_handles(
    tmp_path: Path,
    loads: list[FakeLibrary],
) -> None:
    """Variant builds of one family must not alias through interleaved warm calls.

    ``build_hip`` keys its caches on the full parameterization precisely so two
    builds of one family (WMMA tile variants, a pinned compiler version) do not
    collide. Interleaving them is the case a single-identity warm test cannot
    reach.
    """

    source = _write_source(tmp_path)
    tile16 = _artifact(tmp_path, source, extra_flags=("-DTEST_TILE=16",))
    tile32 = _artifact(tmp_path, source, extra_flags=("-DTEST_TILE=32",))
    other_version = _artifact(tmp_path, source, compiler_version="hipcc other test version")
    assert len({tile16.output_path, tile32.output_path, other_version.output_path}) == 3

    requests = [
        (tile16.output_path, {"extra_flags": ("-DTEST_TILE=16",)}),
        (tile32.output_path, {"extra_flags": ("-DTEST_TILE=32",)}),
        (other_version.output_path, {"compiler_version": "hipcc other test version"}),
        (tile16.output_path, {"extra_flags": ("-DTEST_TILE=16",)}),
        (other_version.output_path, {"compiler_version": "hipcc other test version"}),
        (tile32.output_path, {"extra_flags": ("-DTEST_TILE=32",)}),
        (tile16.output_path, {"extra_flags": ("-DTEST_TILE=16",)}),
    ]
    first_seen: dict[str, FakeLibrary] = {}
    for expected_path, kwargs in requests:
        library = _build(tmp_path, source, **kwargs)
        assert library.path == str(expected_path)
        assert first_seen.setdefault(str(expected_path), library) is library, (
            "an interleaved warm call returned another variant's handle"
        )

    assert len(loads) == 3
    assert len({library.path for library in loads}) == 3


def test_interleaved_arch_and_profile_identities_stay_distinct(
    tmp_path: Path,
    loads: list[FakeLibrary],
) -> None:
    """Explicit arch and profile are identity, not just cache-key components."""

    source = _write_source(tmp_path)
    combos = (
        ("gfx1100", "baseline"),
        ("gfx1151", "baseline"),
        ("gfx1100", "decode"),
        ("gfx1151", "decode"),
    )
    artifacts = {
        combo: _artifact(tmp_path, source, arch=combo[0], profile=combo[1])
        for combo in combos
    }
    assert len({artifact.output_path for artifact in artifacts.values()}) == len(combos)

    first_seen: dict[int, FakeLibrary] = {}
    for index in (0, 1, 3, 2, 0, 3, 1, 2):
        arch, profile = combos[index]
        library = _build(tmp_path, source, arch=arch, profile=profile)
        assert library.path == str(artifacts[(arch, profile)].output_path)
        assert first_seen.setdefault(index, library) is library, (
            f"interleaved ({arch}, {profile}) call returned another build's handle"
        )

    assert len(loads) == len(combos)


def test_caller_held_handle_survives_invalidation(
    tmp_path: Path,
    loads: list[FakeLibrary],
) -> None:
    """Invalidation releases cache ownership, not the caller's loaded handle."""

    source = _write_source(tmp_path)
    artifact = _artifact(tmp_path, source)
    held = _build(tmp_path, source)
    assert held.path == str(artifact.output_path)

    build_module.invalidate_build_resolution()

    # The caller still owns its handle: clearing the caches must not close it
    # out from under a live kernel-launch path.
    assert held.closed is False
    assert held.exports["smoke_host"] is not None

    fresh = _build(tmp_path, source)
    assert fresh is not held
    assert fresh.path == held.path
    assert fresh.closed is False
    assert [library.path for library in loads] == [str(artifact.output_path)] * 2

    # The fresh resolution is memoized again, so a repeat call is a warm hit.
    assert _build(tmp_path, source) is fresh
    assert len(loads) == 2


def test_held_variants_survive_invalidation_and_re_resolve_distinctly(
    tmp_path: Path,
    loads: list[FakeLibrary],
) -> None:
    """Two held variant handles stay usable while a fresh pair re-resolves."""

    source = _write_source(tmp_path)
    variant_a = _artifact(tmp_path, source, extra_flags=("-DTEST_VARIANT=0",))
    variant_b = _artifact(tmp_path, source, extra_flags=("-DTEST_VARIANT=1",))
    held_a = _build(tmp_path, source, extra_flags=("-DTEST_VARIANT=0",))
    held_b = _build(tmp_path, source, extra_flags=("-DTEST_VARIANT=1",))
    assert held_a.path == str(variant_a.output_path)
    assert held_b.path == str(variant_b.output_path)

    build_module.invalidate_build_resolution()
    assert held_a.closed is False
    assert held_b.closed is False

    fresh_a = _build(tmp_path, source, extra_flags=("-DTEST_VARIANT=0",))
    fresh_b = _build(tmp_path, source, extra_flags=("-DTEST_VARIANT=1",))
    assert fresh_a is not held_a
    assert fresh_b is not held_b
    assert fresh_a is not fresh_b
    assert fresh_a.path == held_a.path
    assert fresh_b.path == held_b.path
    assert len(loads) == 4


def test_invalidation_does_not_resurrect_handle_for_removed_artifact(
    tmp_path: Path,
    loads: list[FakeLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm handle outlives the file, but invalidation must re-fail closed."""

    source = _write_source(tmp_path)
    artifact = _artifact(tmp_path, source)
    runs: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], check: bool = False, **kwargs: object) -> None:
        assert check is True
        runs.append(tuple(command))
        Path(command[-1]).write_bytes(b"rebuilt fake shared object")

    monkeypatch.setattr(build_module.subprocess, "run", fake_run)

    warm = _build(tmp_path, source)
    assert warm.path == str(artifact.output_path)
    assert runs == []

    # A loaded handle survives independently of the artifact on disk: the warm
    # hit does not re-stat the cache directory or invoke the compiler.
    artifact.output_path.unlink()
    assert _build(tmp_path, source) is warm
    assert runs == []
    assert len(loads) == 1

    build_module.invalidate_build_resolution()
    with pytest.raises(FileNotFoundError, match="cached build artifact missing"):
        _build(tmp_path, source, require_cached=True)
    assert runs == []

    rebuilt = _build(tmp_path, source, require_cached=False)
    assert rebuilt is not warm
    assert rebuilt.path == str(artifact.output_path)
    assert len(runs) == 1
    assert artifact.output_path.exists()
    assert len(loads) == 2


def test_invalidation_observes_changed_source_bytes_at_the_same_path(
    tmp_path: Path,
    loads: list[FakeLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm resolution is path-based; changed source bytes need invalidation."""

    source = _write_source(tmp_path)
    original = _artifact(tmp_path, source)
    runs: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], check: bool = False, **kwargs: object) -> None:
        assert check is True
        runs.append(tuple(command))
        Path(command[-1]).write_bytes(b"rebuilt fake shared object")

    monkeypatch.setattr(build_module.subprocess, "run", fake_run)

    warm = _build(tmp_path, source)
    assert warm.path == str(original.output_path)

    # Source bytes are hashed only when the build is planned, and a warm hit
    # plans nothing, so the previous artifact stays selected until the caches
    # are dropped.
    source.write_text('extern "C" void smoke_host_v2() {}\n')
    assert _build(tmp_path, source) is warm
    assert runs == []

    build_module.invalidate_build_resolution()
    rebuilt = _build(tmp_path, source, require_cached=False)
    assert rebuilt is not warm
    assert rebuilt.path != str(original.output_path)
    assert len(runs) == 1
    assert Path(rebuilt.path).exists()
    assert len(loads) == 2


def test_explicit_target_arch_pins_identity_across_environment_arch_change(
    tmp_path: Path,
    loads: list[FakeLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``target_arch`` outranks a changed environment default."""

    source = _write_source(tmp_path)
    artifact = _artifact(tmp_path, source, arch="gfx1100")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1100")
    pinned = _build(tmp_path, source, arch="gfx1100")
    assert pinned.path == str(artifact.output_path)

    # No gfx1151 artifact exists, so a resolution that let the environment win
    # would fail closed instead of quietly returning the gfx1100 library.
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    assert _build(tmp_path, source, arch="gfx1100") is pinned
    assert len(loads) == 1

    # Without the explicit parameter the environment default is still observed.
    with pytest.raises(FileNotFoundError, match="cached build artifact missing"):
        _build(tmp_path, source, arch=None, require_cached=True)


def test_invalidation_keeps_process_pinned_compiler_version(
    tmp_path: Path,
    loads: list[FakeLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalidation drops handles, not the explicit per-process version pin."""

    source = _write_source(tmp_path)
    pinned_version = "hipcc pinned test version"
    for name in (
        "HIPENGINE_COMPILER_VERSION_TEXT",
        "HIPENGINE_COMPILER_VERSION_FILE",
        "HIPENGINE_HIPCC_VERSION_TEXT",
        "HIPENGINE_HIPCC_VERSION_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    artifact = _artifact(tmp_path, source, compiler_version=pinned_version)

    _build(tmp_path, source, compiler_version=pinned_version)
    assert build_module._COMPILER_VERSION_CACHE[COMPILER] == pinned_version

    build_module.invalidate_build_resolution()

    # A per-launch call (``compiler_version=None``) must still resolve the
    # pinned build. If invalidation dropped the pin, this would probe the fake
    # compiler and fail instead.
    per_launch = _build(tmp_path, source, compiler_version=None)
    assert per_launch.path == str(artifact.output_path)
    assert build_module._COMPILER_VERSION_CACHE[COMPILER] == pinned_version
    assert len(loads) == 2


def test_pin_change_is_observed_after_explicit_invalidation(
    tmp_path: Path,
    loads: list[FakeLibrary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed process version default needs explicit invalidation.

    The implicit memo holds ``compiler_version=None`` plus environment inputs,
    not the separately pinned default. Updating that default does not replace
    a cached implicit resolution until invalidation. The pin itself survives.
    """

    source = _write_source(tmp_path)
    for name in (
        "HIPENGINE_COMPILER_VERSION_TEXT",
        "HIPENGINE_COMPILER_VERSION_FILE",
        "HIPENGINE_HIPCC_VERSION_TEXT",
        "HIPENGINE_HIPCC_VERSION_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    pinned_a = "hipcc pinned version a"
    pinned_b = "hipcc pinned version b"
    artifact_a = _artifact(tmp_path, source, compiler_version=pinned_a)
    artifact_b = _artifact(tmp_path, source, compiler_version=pinned_b)

    _build(tmp_path, source, compiler_version=pinned_a)
    build_module.invalidate_build_resolution()
    implicit_a = _build(tmp_path, source, compiler_version=None)
    assert implicit_a.path == str(artifact_a.output_path)

    _build(tmp_path, source, compiler_version=pinned_b)
    assert build_module._COMPILER_VERSION_CACHE[COMPILER] == pinned_b

    warm = _build(tmp_path, source, compiler_version=None)
    assert warm is implicit_a
    assert warm.path == str(artifact_a.output_path)
    assert len(loads) == 3

    build_module.invalidate_build_resolution()
    fresh = _build(tmp_path, source, compiler_version=None)
    assert fresh.path == str(artifact_b.output_path)
    assert len(loads) == 4


def test_hip_and_cuda_resolutions_of_one_family_do_not_share_a_handle(
    tmp_path: Path,
    loads: list[FakeLibrary],
) -> None:
    """The shared handle store separates backend identity."""

    source = _write_source(tmp_path)
    cuda_version = "nvcc test version"
    cuda_artifact = plan_cuda_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler="definitely-not-a-real-nvcc",
        compiler_version=cuda_version,
        target_arch="sm_86",
    )
    cuda_artifact.cache_dir.mkdir(parents=True, exist_ok=True)
    cuda_artifact.output_path.write_bytes(b"not a real shared object")
    hip_artifact = _artifact(tmp_path, source, arch="gfx1100")

    def cuda() -> FakeLibrary:
        return build_cuda(  # type: ignore[return-value]
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-nvcc",
            compiler_version=cuda_version,
            target_arch="sm_86",
            load=True,
            require_cached=True,
        )

    first_cuda = cuda()
    hip = _build(tmp_path, source, arch="gfx1100")
    assert first_cuda.path == str(cuda_artifact.output_path)
    assert hip.path == str(hip_artifact.output_path)
    assert first_cuda is not hip
    assert cuda() is first_cuda
    assert len(loads) == 2
