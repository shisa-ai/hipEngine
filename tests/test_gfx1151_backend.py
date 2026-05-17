from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.build import plan_hip_build
from hipengine.generation import register_builtin_generators, resolve_text_generator
from hipengine.kernels.backends import (
    CPU_BACKEND,
    hip_target_arch_for_backend,
    resolve_backend,
    select_backend,
)
from hipengine.kernels.hip_gfx1100.norm import (
    paro_rmsnorm_out_fp16,
    register_qwen35_rmsnorm_kernels,
)
from hipengine.kernels.hip_gfx1151 import TARGET_ARCH, register_gfx1151_kernels
from hipengine.kernels.registry import resolve


def test_auto_backend_selects_supported_hip_arches() -> None:
    assert select_backend("auto", detected_arches=["gfx1100"]).backend == "hip_gfx1100"
    assert (
        select_backend("auto", detected_arches=["gfx1151:sramecc+:xnack-"]).backend
        == "hip_gfx1151"
    )


def test_auto_backend_honors_force_env_override() -> None:
    selection = select_backend(
        "auto",
        detected_arches=["gfx1151"],
        env={"HIPENGINE_BACKEND": "hip_gfx1100"},
    )

    assert selection.backend == "hip_gfx1100"
    assert selection.source == "HIPENGINE_BACKEND"


def test_auto_backend_warns_and_falls_back_for_unknown_arch() -> None:
    selection = select_backend("auto", detected_arches=["gfx1102"], env={})

    assert selection.backend == CPU_BACKEND
    assert selection.detected_arches == ("gfx1102",)
    assert selection.warning is not None
    assert "gfx1102" in selection.warning
    assert "HIPENGINE_BACKEND=hip_gfx1100" in selection.warning

    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=["gfx1102"], env={}) == CPU_BACKEND


def test_explicit_backend_is_not_autodetected() -> None:
    selection = select_backend("custom_backend", detected_arches=["gfx1151"], env={})

    assert selection.backend == "custom_backend"
    assert selection.source == "explicit"
    assert selection.detected_arches == ()


def test_gfx1151_backend_aliases_gfx1100_kernel_keys() -> None:
    register_qwen35_rmsnorm_kernels()
    register_gfx1151_kernels()

    assert TARGET_ARCH == "gfx1151"
    assert hip_target_arch_for_backend("hip_gfx1151") == "gfx1151"
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="rmsnorm",
            quant="w4_paro",
            variant="paro_out_fp16",
        )
        is paro_rmsnorm_out_fp16
    )


def test_plan_hip_build_target_arch_is_in_flags_and_cache_key(tmp_path: Path) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')

    gfx1100 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1100",
    )
    gfx1151 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert gfx1100.cache_key != gfx1151.cache_key
    assert gfx1100.target_arch == "gfx1100"
    assert gfx1151.target_arch == "gfx1151"
    assert "--offload-arch=gfx1100" in gfx1100.flags
    assert "--offload-arch=gfx1151" in gfx1151.flags
    assert "--offload-arch=gfx1151" in gfx1151.command


def test_plan_hip_build_reads_target_arch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )

    assert artifact.target_arch == "gfx1151"
    assert "--offload-arch=gfx1151" in artifact.flags


def test_plan_hip_build_includes_device_lib_path_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    device_lib_path = tmp_path / "amdgcn" / "bitcode"
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", str(device_lib_path))

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.flags
    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.command


def test_qwen35_paro_gfx1151_generation_factory_sets_backend() -> None:
    register_builtin_generators()
    factory = resolve_text_generator(
        model="qwen3_5_moe_paro",
        backend="hip_gfx1151",
        quant="w4_paro",
    )

    generator = factory(model_path="/tmp/fake", weight_index=object(), model_plugin=object())

    assert getattr(generator, "backend") == "hip_gfx1151"
