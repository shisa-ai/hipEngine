from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.fused import (
    plan_paro_combine_build,
    register_paro_combine_kernels,
    shared_gate_combine_out_bf16,
    shared_gate_combine_out_fp16,
    shared_gate_combine_residual_batch_out_bf16,
    shared_gate_combine_residual_batch_out_fp16,
    shared_gate_combine_residual_out_bf16,
    shared_gate_combine_residual_out_fp16,
    weighted_lanes_sum_out_bf16_f32w,
    weighted_lanes_sum_out_fp16_f32w,
    weighted_sum_out_bf16_f32w,
    weighted_sum_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_fp16_f32w,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_combine_registers_bf16_fp16_and_w4_paro_variants() -> None:
    register_paro_combine_kernels()

    for quant in ("bf16", "w4_paro"):
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_lanes_sum", quant=quant, variant="out")
            is weighted_lanes_sum_out_bf16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_lanes_sum", quant=quant, variant="out_fp16")
            is weighted_lanes_sum_out_fp16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_sum", quant=quant, variant="out")
            is weighted_sum_out_bf16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_sum", quant=quant, variant="out_fp16")
            is weighted_sum_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out",
            )
            is weighted_sum_shared_gate_combine_residual_out_bf16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out_fp16",
            )
            is weighted_sum_shared_gate_combine_residual_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out_fp16",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine",
                quant=quant,
                variant="out",
            )
            is shared_gate_combine_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine",
                quant=quant,
                variant="out_fp16",
            )
            is shared_gate_combine_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="out",
            )
            is shared_gate_combine_residual_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="out_fp16",
            )
            is shared_gate_combine_residual_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="batch_out",
            )
            is shared_gate_combine_residual_batch_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="batch_out_fp16",
            )
            is shared_gate_combine_residual_batch_out_fp16
        )
    assert resolve(backend="hip_gfx1100", layer="weighted_sum", quant="fp16", variant="out") is weighted_sum_out_fp16_f32w


def test_paro_combine_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_combine_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro combine test version",
    )

    assert artifact.family == "paro_combine"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_combine.so"
    assert artifact.compiler_version == "hipcc paro combine test version"
    assert any(str(path).endswith("paro_combine.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_combine_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        weighted_lanes_sum_out_bf16_f32w(0, 0, 0, 0, 0, 0, 8, 16)
    with pytest.raises(ValueError, match="top_k must be positive"):
        weighted_lanes_sum_out_fp16_f32w(0, 0, 0, 0, 0, 2, 0, 16)
    with pytest.raises(ValueError, match="rows must be positive"):
        weighted_sum_out_bf16_f32w(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="features must be positive"):
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w(0, 0, 0, 0, 0, 0, 2, 0)
    with pytest.raises(ValueError, match="threads must be one of"):
        shared_gate_combine_out_bf16(0, 0, 0, 0, 8, threads=32)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)
    with pytest.raises(ValueError, match="features must be positive"):
        shared_gate_combine_residual_out_bf16(0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        shared_gate_combine_residual_batch_out_bf16(0, 0, 0, 0, 0, 2, 16, 0)
    with pytest.raises(ValueError, match="tokens must be positive"):
        shared_gate_combine_residual_batch_out_fp16(0, 0, 0, 0, 0, 0, 16, 129)
    with pytest.raises(ValueError, match="rows must be positive"):
        weighted_sum_out_fp16_f32w(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)
