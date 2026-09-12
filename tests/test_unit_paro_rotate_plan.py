from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.rotary import (
    paro_rotate1_bf16_gate_fp16,
    paro_rotate1_f32_to_fp16,
    paro_rotate1_fp16,
    paro_rotate2_bf16,
    paro_rotate2_fp16,
    paro_rotate3_bf16,
    paro_rotate3_fp16,
    plan_paro_rotate_build,
    register_paro_rotate_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_rotate_registers_pairwise_variants() -> None:
    register_paro_rotate_kernels()

    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate2", quant="w4_paro", variant="bf16")
        is paro_rotate2_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate3", quant="w4_paro", variant="bf16")
        is paro_rotate3_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate1", quant="w4_paro", variant="fp16")
        is paro_rotate1_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate1", quant="w4_paro", variant="f32_to_fp16")
        is paro_rotate1_f32_to_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="paro_rotate1",
            quant="w4_paro",
            variant="bf16_gate_fp16",
        )
        is paro_rotate1_bf16_gate_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate2", quant="w4_paro", variant="fp16")
        is paro_rotate2_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="paro_rotate3", quant="w4_paro", variant="fp16")
        is paro_rotate3_fp16
    )


def test_paro_rotate_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_rotate_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro rotate test version",
    )

    assert artifact.family == "paro_rotate"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_rotate.so"
    assert artifact.compiler_version == "hipcc paro rotate test version"
    assert any(str(path).endswith("paro_rotate.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_rotate_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        paro_rotate2_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 1)
    with pytest.raises(ValueError, match="group_size must be even"):
        paro_rotate2_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 3, 1)
    with pytest.raises(ValueError, match="hidden must be divisible"):
        paro_rotate3_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 8, 1)
    with pytest.raises(ValueError, match="krot must be non-negative"):
        paro_rotate3_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 8, -1)
    with pytest.raises(ValueError, match="tokens must be positive"):
        paro_rotate1_fp16(0, 0, 0, 0, 0, 0, 8, 8, 1)
    with pytest.raises(ValueError, match="tokens must be positive"):
        paro_rotate1_f32_to_fp16(0, 0, 0, 0, 0, 0, 8, 8, 1)
    with pytest.raises(ValueError, match="tokens must be positive"):
        paro_rotate1_bf16_gate_fp16(0, 0, 0, 0, 0, 0, 0, 8, 8, 1)
    with pytest.raises(ValueError, match="group_size must be even"):
        paro_rotate2_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 3, 1)
    with pytest.raises(ValueError, match="krot must be non-negative"):
        paro_rotate3_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 8, -1)
