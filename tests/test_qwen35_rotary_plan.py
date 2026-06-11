from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.rotary import (
    plan_qwen35_rotary_build,
    qwen35_head_rmsnorm_partial_rotary_f32_bf16,
    qwen35_head_rmsnorm_partial_rotary_position_f32_bf16,
    qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16,
    qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32,
    qwen35_partial_rotary_f32,
    qwen35_split_qgate_bf16,
    qwen35_split_qgate_fp16,
    qwen35_split_qgate_fp16_key_f32,
    register_qwen35_rotary_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_rotary_registers_full_attention_prelude_variants() -> None:
    register_qwen35_rotary_kernels()

    assert (
        resolve(backend="hip_gfx1100", layer="split_qgate", quant="w4_paro", variant="qwen35_bf16")
        is qwen35_split_qgate_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="split_qgate", quant="w4_paro", variant="qwen35_fp16")
        is qwen35_split_qgate_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="split_qgate+key_cast",
            quant="w4_paro",
            variant="qwen35_fp16_key_f32",
        )
        is qwen35_split_qgate_fp16_key_f32
    )
    assert (
        resolve(backend="hip_gfx1100", layer="partial_rotary", quant="w4_paro", variant="qwen35_f32")
        is qwen35_partial_rotary_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary",
            quant="w4_paro",
            variant="qwen35_f32_bf16",
        )
        is qwen35_head_rmsnorm_partial_rotary_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary",
            quant="w4_paro",
            variant="qwen35_position_f32_bf16",
        )
        is qwen35_head_rmsnorm_partial_rotary_position_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary",
            quant="w4_paro",
            variant="qwen35_positions_f32_bf16",
        )
        is qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary",
            quant="w4_paro",
            variant="qwen35_positions_q_bf16_key_f32",
        )
        is qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32
    )


def test_qwen35_rotary_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_rotary_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 rotary test version",
    )

    assert artifact.family == "qwen35_rotary"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_rotary.so"
    assert artifact.compiler_version == "hipcc qwen35 rotary test version"
    assert any(str(path).endswith("qwen35_rotary.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_rotary_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_split_qgate_bf16(0, 0, 0, 0, 1, 8)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_split_qgate_fp16(0, 0, 0, 0, 1, 8)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_split_qgate_fp16_key_f32(0, 0, 0, 0, 0, 0, 1, 1, 8)
    with pytest.raises(ValueError, match="num_kv_heads must be positive"):
        qwen35_split_qgate_fp16_key_f32(0, 0, 0, 0, 0, 1, 1, 0, 8)
    with pytest.raises(ValueError, match="num_q_heads must be positive"):
        qwen35_partial_rotary_f32(0, 0, 0, 0, 0, 0, 0, 1, 8, 4)
    with pytest.raises(ValueError, match="rotary_dim must be <= head_dim"):
        qwen35_head_rmsnorm_partial_rotary_f32_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 1, 8, 16
        )
    with pytest.raises(ValueError, match="rotary_dim must be even"):
        qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 1, 8, 3, 2
        )
    with pytest.raises(ValueError, match="max_positions must be positive"):
        qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 1, 1, 8, 4, 0
        )
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 0, 1, 1, 8, 4, 2
        )
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0e-6, 0, 1, 1, 8, 4, 2
        )
