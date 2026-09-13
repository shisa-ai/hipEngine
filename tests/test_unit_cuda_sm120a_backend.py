from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.build import build_cuda, plan_cuda_build
from hipengine.core.device import Device
from hipengine.kernels.backends import (
    cuda_target_arch_for_backend,
    load_backend_kernel_package,
    select_backend,
)
from hipengine.kernels.registry import (
    KernelKey,
    clear_registry_for_tests,
    is_registered,
    resolve,
)


def setup_function() -> None:
    clear_registry_for_tests()


def _write_source(path: Path) -> Path:
    path.write_text('extern "C" __global__ void smoke() {}\n')
    return path


def test_cuda_sm120a_backend_metadata_and_explicit_selection() -> None:
    assert Device("cuda", 0).kind == "cuda"
    assert cuda_target_arch_for_backend("cuda_sm120a") == "sm_120a"
    selection = select_backend("cuda_sm120a", detected_arches=[], env={})
    assert selection.backend == "cuda_sm120a"
    assert selection.source == "explicit"


def test_plan_cuda_build_separates_arch_source_flags_and_compiler_version(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "smoke.cu")
    sm120a = plan_cuda_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="nvcc test version",
        target_arch="sm_120a",
    )
    sm120 = plan_cuda_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="nvcc test version",
        target_arch="sm_120",
    )
    other_version = plan_cuda_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="different nvcc",
        target_arch="sm_120a",
    )

    assert sm120a.cache_key != sm120.cache_key
    assert sm120a.cache_key != other_version.cache_key
    assert sm120a.target_arch == "sm_120a"
    assert sm120a.flags == ("-arch=sm_120a",)
    assert sm120a.command[:5] == (
        "nvcc",
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler=-fPIC",
    )
    assert "-arch=sm_120a" in sm120a.command
    assert "--offload-arch=sm_120a" not in sm120a.command


def test_plan_cuda_build_uses_cuda_arch_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "smoke.cu")
    monkeypatch.setenv("HIPENGINE_CUDA_ARCH", "sm_120a")

    artifact = plan_cuda_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="nvcc test version",
    )

    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)


def test_build_cuda_dry_run_and_require_cached_are_safe(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "smoke.cu")
    artifact = build_cuda(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler="definitely-not-a-real-nvcc",
        compiler_version="nvcc test version",
        target_arch="sm_120a",
        dry_run=True,
        load=False,
    )

    assert artifact.command[0] == "definitely-not-a-real-nvcc"
    assert artifact.profile.name == "baseline"
    assert not artifact.cache_dir.exists()

    with pytest.raises(FileNotFoundError, match="cached build artifact missing"):
        build_cuda(
            sources=[source],
            family="smoke",
            profile="baseline",
            cache_root=tmp_path / "cache",
            compiler="definitely-not-a-real-nvcc",
            compiler_version="nvcc test version",
            target_arch="sm_120a",
            load=False,
            require_cached=True,
        )


def test_cuda_sm120a_package_reregisters_all_implemented_kernel_families() -> None:
    smoke_key = KernelKey("cuda_sm120a", "smoke_add", "fp32")
    encoder_key = KernelKey(
        "cuda_sm120a", "moonshine_conv1_tanh", "fp16", "strided_valid"
    )
    aot_attention_key = KernelKey(
        "cuda_sm120a", "moonshine_self_attention", "fp16", "aot_cutlass"
    )
    assert not is_registered(smoke_key)
    assert not is_registered(encoder_key)
    assert not is_registered(aot_attention_key)

    module = load_backend_kernel_package("cuda_sm120a")

    assert module.BACKEND == "cuda_sm120a"
    assert module.TARGET_ARCH == "sm_120a"
    assert callable(resolve(backend="cuda_sm120a", layer="smoke_add", quant="fp32"))
    assert callable(
        resolve(
            backend="cuda_sm120a",
            layer="moonshine_conv1_tanh",
            quant="fp16",
            variant="strided_valid",
        )
    )
    assert callable(
        resolve(
            backend="cuda_sm120a",
            layer="moonshine_self_attention",
            quant="fp16",
            variant="aot_cutlass",
        )
    )
