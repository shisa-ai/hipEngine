from __future__ import annotations

from pathlib import Path

import pytest

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
    assert "-mcumode" in artifact_a.flags
    assert "-amdgpu-unroll-threshold-local=600" in artifact_a.flags
    assert artifact_a.profile.wavefront == 64


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
