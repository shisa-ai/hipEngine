"""Lazy deterministic build plan for the minimal ROCr/PM4 native core."""

from __future__ import annotations

from pathlib import Path

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build

_SOURCE = Path(__file__).with_name("native.cpp")


def plan_pm4_native_build(
    *,
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    target_arch: str = "gfx1100",
) -> BuildArtifact:
    return plan_hip_build(
        sources=(_SOURCE,),
        family="pm4-native",
        profile="baseline",
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=compiler_version,
        extra_flags=("-std=c++17", "-lhsa-runtime64"),
        target_arch=target_arch,
        output_name="pm4_native.so",
    )


def build_pm4_native(
    *,
    cache_root: str | Path | None = None,
    compiler: str = "hipcc",
    compiler_version: str | None = None,
    target_arch: str = "gfx1100",
    force: bool = False,
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
):
    """Build/load the native core without import-time compiler or GPU access."""

    return build_hip(
        sources=(_SOURCE,),
        family="pm4-native",
        profile="baseline",
        cache_root=cache_root,
        compiler=compiler,
        compiler_version=compiler_version,
        extra_flags=("-std=c++17", "-lhsa-runtime64"),
        target_arch=target_arch,
        output_name="pm4_native.so",
        force=force,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )
