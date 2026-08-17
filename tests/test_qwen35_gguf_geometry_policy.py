from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import (
    GGUFModelGeometry,
    QWEN35_DENSE_H1024_GEOMETRY,
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_mtp import _resolve_gguf_verifier_backend

_Q4_K_M = "MOSTLY_Q4_K_M"
_Q4_K_S = "MOSTLY_Q4_K_S"


def _dense_config() -> SimpleNamespace:
    return SimpleNamespace(
        architecture="qwen35",
        block_count=64,
        hidden_size=5_120,
        vocab_size=248_320,
        feed_forward_length=17_408,
        head_count=24,
        head_count_kv=4,
        key_length=256,
        value_length=256,
        full_attention_interval=4,
        layer_types=tuple(
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(64)
        ),
        ssm_inner_size=6_144,
        ssm_group_count=16,
        ssm_state_size=128,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        expert_count=0,
        expert_used_count=0,
        expert_feed_forward_length=0,
        expert_shared_feed_forward_length=0,
        is_moe=False,
    )


def _dense_runner(*, model_name: str = "arbitrary-finetune") -> SimpleNamespace:
    config = _dense_config()
    return SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(
            config=config,
            geometry=GGUFModelGeometry.from_config(config),
            model_name=model_name,
            file_type_name=_Q4_K_M,
        ),
    )


def test_qwen35_geometry_keys_describe_architecture_not_model_names() -> None:
    geometry = GGUFModelGeometry.from_config(_dense_config())

    assert geometry == QWEN35_DENSE_H5120_GEOMETRY
    assert geometry != QWEN35_MOE_H2048_E256_GEOMETRY
    assert replace(geometry, hidden_size=geometry.hidden_size + 1) != geometry
    assert "Qwen3.6" not in repr(geometry)
    assert "Qwen3.8" not in repr(geometry)


def test_dense_backend_policies_are_geometry_keyed() -> None:
    identity = (QWEN35_DENSE_H5120_GEOMETRY, _Q4_K_M)
    for capability in (
        "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES",
        "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES",
        "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES",
        "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
        "GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES",
        "GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES",
        "GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES",
    ):
        policies = backend_package_capability("hip_gfx1100", capability, {})
        assert identity in policies
        assert all(not isinstance(key[0], str) for key in policies)


def test_gfx1151_qwen38_memory_policies_are_geometry_and_quant_scoped() -> None:
    runner = _dense_runner(model_name="community/Qwen3.8-27B-finetune")
    runner.backend = "hip_gfx1151"
    runner.weights.file_type_name = _Q4_K_S
    identity = (QWEN35_DENSE_H5120_GEOMETRY, _Q4_K_S)

    for capability in (
        "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES",
        "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES",
        "GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES",
    ):
        policies = backend_package_capability("hip_gfx1151", capability, {})
        assert identity in policies
        assert all(not isinstance(key[0], str) for key in policies)

    assert gguf_runner._resolve_gguf_private_c1_small_weight_arena(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == (True, "private_c1_selective")
    assert gguf_runner._resolve_gguf_private_c1_weight_arena_max_allocation_bytes(
        backend=runner.backend,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == 80 * 1024 * 1024
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == (True, "private_c1_geometry_policy")
    assert gguf_runner._gguf_dense_prefill_scratch_policy(runner) is None
    assert gguf_runner._gguf_dense_prefill_scratch_row_cap(
        runner,
        capacity=4_352,
    ) == 4_096
    assert gguf_runner._gguf_dense_prefill_scratch_row_cap(
        runner,
        capacity=8_192,
    ) == 1_024
    assert gguf_runner._gguf_dense_prefill_scratch_row_cap(
        runner,
        capacity=1_280,
    ) is None
    assert gguf_runner._resolve_gguf_token_embedding_placement(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q4_K",
    ) == ("host", "mapped_host_private_c1_auto")
    assert backend_package_capability(
        runner.backend,
        "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_COPY",
        False,
    ) is True

    runner.weights.file_type_name = _Q4_K_M
    assert gguf_runner._gguf_dense_prefill_scratch_policy(runner) is None
    assert gguf_runner._gguf_dense_prefill_scratch_row_cap(
        runner,
        capacity=4_352,
    ) is None
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == (False, "backend_capability_fallback")


def test_08b_decode_policies_are_geometry_keyed() -> None:
    identity = (QWEN35_DENSE_H1024_GEOMETRY, _Q4_K_M)
    for capability in (
        "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
        "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES",
        "GGUF_NORM_RESIDUAL_DECODE_POLICIES",
    ):
        policies = backend_package_capability("hip_gfx1151", capability, {})
        assert identity in policies
        assert all(not isinstance(key[0], str) for key in policies)


def test_dense_policy_admission_ignores_finetune_name_but_rejects_geometry_drift() -> None:
    runner = _dense_runner(model_name="community/Qwen3.8-27B-finetune")

    assert gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    assert gguf_runner._gguf_t16_f16_rocblas_prefill_policy(runner) is not None
    assert gguf_runner._gguf_dense_prefill_scratch_policy(runner) is not None
    assert gguf_runner._resolve_gguf_private_c1_small_weight_arena(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == (True, "private_c1_selective")
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        backend=runner.backend,
        max_batch_size=1,
        has_shared_runner=False,
        geometry=runner.weights.geometry,
        file_type_name=runner.weights.file_type_name,
    ) == (True, "private_c1_geometry_policy")
    assert gguf_runner._gguf_dense_pair_silu_decode_variant(
        runner,
        rows=1,
        in_features=5_120,
        out_features=17_408,
    ) == "dense_dual_local32_bf16_bf16_out"

    runner.weights.model_name = "totally-different-name"
    assert gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    runner.weights.geometry = replace(runner.weights.geometry, head_count=23)
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    assert gguf_runner._gguf_t16_f16_rocblas_prefill_policy(runner) is None
    assert gguf_runner._gguf_dense_prefill_scratch_policy(runner) is None
    assert gguf_runner._gguf_dense_pair_silu_decode_variant(
        runner,
        rows=1,
        in_features=5_120,
        out_features=17_408,
    ) is None


def test_verifier_inherits_target_backend_and_rejects_cross_backend_override() -> None:
    target = SimpleNamespace(backend="hip_gfx1151")

    assert _resolve_gguf_verifier_backend(target, None) == "hip_gfx1151"
    assert _resolve_gguf_verifier_backend(target, "auto") == "hip_gfx1151"
    assert _resolve_gguf_verifier_backend(target, "hip_gfx1151") == "hip_gfx1151"
    with pytest.raises(ValueError, match="does not match target backend"):
        _resolve_gguf_verifier_backend(target, "hip_gfx1100")
