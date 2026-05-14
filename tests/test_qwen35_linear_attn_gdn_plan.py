from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear_attn import (
    plan_qwen35_linear_attn_gdn_build,
    qwen35_gdn_prefill_recurrent_f32,
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_linear_attn_gdn_registers_decode_and_prefill_variants() -> None:
    register_qwen35_linear_attn_gdn_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_recurrent_rmsnorm_gate",
            quant="w4_paro",
            variant="bf16_lowp",
        )
        is qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="w4_paro",
            variant="f32",
        )
        is qwen35_gdn_prefill_recurrent_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="w4_paro",
            variant="f32_k2",
        )
        is qwen35_gdn_prefill_recurrent_k2_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_prefill_prepare",
            quant="w4_paro",
            variant="f32_bf16",
        )
        is qwen35_linear_attn_prefill_prepare_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_rmsnorm_gate",
            quant="w4_paro",
            variant="bf16",
        )
        is qwen35_gdn_prefill_rmsnorm_gate_bf16
    )


def test_qwen35_linear_attn_gdn_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_linear_attn_gdn_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 linear attn gdn test version",
    )

    assert artifact.family == "qwen35_linear_attn_gdn"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 64
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_linear_attn_gdn.so"
    assert artifact.compiler_version == "hipcc qwen35 linear attn gdn test version"
    assert any(str(path).endswith("gdn.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_linear_attn_gdn_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="num_k_heads must be positive"):
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 0, 2, 8, 4
        )
    with pytest.raises(ValueError, match="num_v_heads must be divisible"):
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 2, 3, 8, 4
        )
    with pytest.raises(ValueError, match="head_v_dim must be <= 128"):
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 2, 8, 129
        )
    with pytest.raises(ValueError, match="head_k_dim must be 128"):
        qwen35_gdn_prefill_recurrent_f32(0, 0, 0, 0, 0, 0, 0, 1, 2, 64, 4)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_gdn_prefill_recurrent_k2_f32(0, 0, 0, 0, 0, 0, 0, 0, 2, 128, 4)
    with pytest.raises(ValueError, match="num_v_heads must be divisible"):
        qwen35_linear_attn_prefill_prepare_f32_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 128, 4)
    with pytest.raises(ValueError, match="head_v_dim must be positive"):
        qwen35_gdn_prefill_rmsnorm_gate_bf16(0, 0, 0, 0, 1.0e-6, 1, 2, 0)
