from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.speculative import TargetCommitPlan, TargetStateCommitBuffers
from hipengine.kernels.hip_gfx1100.linear import (
    argmax_f32_rows_i32,
    lm_head_fp16_argmax_bf16_rows_i32,
    plan_lm_head_build,
    register_lm_head_kernels,
    topk_f32_rows_i32,
)
from hipengine.kernels.hip_gfx1100.speculative import (
    dflash_accept_chain_i32,
    dflash_accept_chain_i32_packed,
    dflash_accept_chain_i32_packed_update_state,
    dflash_add_bf16,
    dflash_commit_chain_i32,
    dflash_concat_rows_bf16,
    dflash_concat_rows_f32,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_head_rmsnorm_rotary_f32,
    dflash_head_rmsnorm_rotary_indexed_key_f32,
    dflash_key_rmsnorm_rotary_f32,
    dflash_prepare_noise_inputs_bf16_i32,
    dflash_prepare_noise_inputs_f16_to_bf16_i32,
    dflash_qkv_proj_bf16_mixed_indexed_v,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
    dflash_update_kv_metadata_i32,
    plan_dflash_accept_build,
    plan_dflash_commit_build,
    plan_dflash_drafter_build,
    register_dflash_accept_kernels,
    register_dflash_commit_kernels,
    register_dflash_drafter_kernels,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import resolve


def test_dflash_accept_and_row_argmax_build_plans_include_native_arch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    lm_head = plan_lm_head_build(compiler_version="hipcc:test")
    accept = plan_dflash_accept_build(compiler_version="hipcc:test")
    commit = plan_dflash_commit_build(compiler_version="hipcc:test")
    drafter = plan_dflash_drafter_build(compiler_version="hipcc:test")

    assert "--offload-arch=gfx1151" in lm_head.command
    assert "--offload-arch=gfx1151" in accept.command
    assert "--offload-arch=gfx1151" in commit.command
    assert "--offload-arch=gfx1151" in drafter.command
    assert lm_head.target_arch == "gfx1151"
    assert accept.target_arch == "gfx1151"
    assert commit.target_arch == "gfx1151"
    assert drafter.target_arch == "gfx1151"


def test_dflash_accept_and_row_argmax_register_for_gfx1151_aliases() -> None:
    register_lm_head_kernels(replace=True)
    register_dflash_accept_kernels(replace=True)
    register_dflash_commit_kernels(replace=True)
    register_dflash_drafter_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert (
        resolve(backend="hip_gfx1151", layer="argmax", quant="w4_paro", variant="f32_rows_i32")
        is argmax_f32_rows_i32
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="lm_head",
            quant="w4_paro",
            variant="fp16_argmax_bf16_rows_i32",
        )
        is lm_head_fp16_argmax_bf16_rows_i32
    )
    assert resolve(backend="hip_gfx1151", layer="topk", quant="w4_paro", variant="f32_rows_i32") is topk_f32_rows_i32
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_accept_chain", quant="w4_paro", variant="i32")
        is dflash_accept_chain_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_commit_chain", quant="w4_paro", variant="i32")
        is dflash_commit_chain_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_prepare_noise_inputs", quant="w4_paro", variant="bf16_i32")
        is dflash_prepare_noise_inputs_bf16_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_prepare_noise_inputs", quant="w4_paro", variant="f16_to_bf16_i32")
        is dflash_prepare_noise_inputs_f16_to_bf16_i32
    )
    assert resolve(backend="hip_gfx1151", layer="dflash_add", quant="w4_paro", variant="bf16") is dflash_add_bf16
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_concat_rows", quant="w4_paro", variant="f32")
        is dflash_concat_rows_f32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_concat_rows", quant="w4_paro", variant="bf16")
        is dflash_concat_rows_bf16
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_rmsnorm", quant="w4_paro", variant="bf16")
        is dflash_rmsnorm_bf16
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_silu_mul", quant="w4_paro", variant="bf16")
        is dflash_silu_mul_bf16
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_dense", quant="w4_paro", variant="bf16_to_bf16")
        is dflash_dense_bf16_to_bf16
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_dense", quant="w4_paro", variant="bf16_to_f32")
        is dflash_dense_bf16_to_f32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_head_rmsnorm_rotary", quant="w4_paro", variant="f32_bf16")
        is dflash_head_rmsnorm_rotary_f32
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="dflash_head_rmsnorm_rotary",
            quant="w4_paro",
            variant="f32_bf16_indexed_key",
        )
        is dflash_head_rmsnorm_rotary_indexed_key_f32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_qkv_proj", quant="w4_paro", variant="bf16_mixed_indexed_v")
        is dflash_qkv_proj_bf16_mixed_indexed_v
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_key_rmsnorm_rotary", quant="w4_paro", variant="f32_bf16")
        is dflash_key_rmsnorm_rotary_f32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_update_kv_metadata", quant="w4_paro", variant="i32")
        is dflash_update_kv_metadata_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_gqa_attention", quant="w4_paro", variant="f32_bf16")
        is dflash_gqa_attention_f32_bf16
    )


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def test_row_argmax_and_dflash_accept_wrappers_validate_shapes_before_loading_hip() -> None:
    with pytest.raises(ValueError, match="rows"):
        argmax_f32_rows_i32(0, 0, 0, 0, None, rows=0, vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        lm_head_fp16_argmax_bf16_rows_i32(0, 0, 0, 0, 0, 0, None, rows=1, hidden_size=8, vocab_size=0)
    with pytest.raises(ValueError, match="top_k"):
        topk_f32_rows_i32(0, None, 0, rows=1, vocab_size=16, top_k=9)
    with pytest.raises(ValueError, match="elements"):
        dflash_add_bf16(0, 0, 0, elements=0)
    with pytest.raises(ValueError, match="features"):
        dflash_concat_rows_bf16(0, 0, 0, batch_size=1, context_len=1, query_len=1, features=0)
    with pytest.raises(ValueError, match="hidden_size"):
        dflash_rmsnorm_bf16(
            0,
            0,
            0,
            rows=1,
            hidden_size=0,
        )
    with pytest.raises(ValueError, match="elements"):
        dflash_silu_mul_bf16(0, 0, 0, elements=0)
    with pytest.raises(ValueError, match="out_features"):
        dflash_dense_bf16_to_bf16(0, 0, 0, rows=1, in_features=4, out_features=0)
    with pytest.raises(ValueError, match="in_features"):
        dflash_dense_bf16_to_f32(
            0,
            0,
            0,
            rows=1,
            in_features=0,
            out_features=4,
        )
    with pytest.raises(ValueError, match="rotary_dim"):
        dflash_head_rmsnorm_rotary_f32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            batch_size=1,
            query_len=1,
            kv_len=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rotary_dim=9,
            max_positions=16,
        )
    with pytest.raises(ValueError, match="batch_size=1"):
        dflash_head_rmsnorm_rotary_indexed_key_f32(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            batch_size=2,
            query_len=1,
            kv_len=1,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rotary_dim=8,
            max_positions=16,
        )
    with pytest.raises(ValueError, match="cache_rows"):
        dflash_qkv_proj_bf16_mixed_indexed_v(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=1,
            in_features=4,
            q_features=8,
            kv_features=2,
        )
    with pytest.raises(ValueError, match="rotary_dim"):
        dflash_key_rmsnorm_rotary_f32(
            0,
            0,
            0,
            0,
            0,
            0,
            rows=1,
            num_kv_heads=2,
            head_dim=8,
            rotary_dim=9,
            max_positions=16,
        )
    with pytest.raises(ValueError, match="end"):
        dflash_update_kv_metadata_i32(0, 0, 0, start=2, count=1, end=1)
    with pytest.raises(ValueError, match="num_q_heads"):
        dflash_gqa_attention_f32_bf16(
            0,
            0,
            0,
            0,
            batch_size=1,
            query_len=1,
            kv_len=1,
            num_q_heads=3,
            num_kv_heads=2,
            head_dim=8,
        )
    with pytest.raises(ValueError, match="mask_token_id"):
        dflash_prepare_noise_inputs_bf16_i32(
            0,
            0,
            0,
            0,
            0,
            0,
            request_count=1,
            block_size=4,
            hidden_size=16,
            vocab_size=10,
            mask_token_id=10,
        )
    with pytest.raises(ValueError, match="request_count"):
        dflash_accept_chain_i32(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=2,
            request_count=3,
            output_stride=2,
        )
    with pytest.raises(ValueError, match="output_stride"):
        dflash_accept_chain_i32(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=2,
            request_count=1,
            output_stride=0,
        )
    with pytest.raises(ValueError, match="packed_payload_i32_ptr"):
        dflash_accept_chain_i32_packed(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=2,
            request_count=1,
            output_stride=2,
        )
    with pytest.raises(ValueError, match="resident_positions_i64_ptr"):
        dflash_accept_chain_i32_packed_update_state(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0x1000,
            0,
            0x2000,
            rows=2,
            request_count=1,
            output_stride=2,
        )
    with pytest.raises(ValueError, match="resident_contexts_i64_ptr"):
        dflash_accept_chain_i32_packed_update_state(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0x1000,
            0x2000,
            0,
            rows=2,
            request_count=1,
            output_stride=2,
        )


def test_dflash_commit_wrapper_validates_shape_before_loading_hip() -> None:
    plan = TargetCommitPlan(
        transaction_id=4,
        request_ids=(1,),
        accepted_counts=(1,),
        commit_rows=(1,),
        commit_tokens=(11,),
        commit_positions=(6,),
        candidate_counts=(1,),
    )
    buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x1000, (1,), "int32"),
        commit_rows=_tensor(0x1100, (1,), "int32"),
        commit_positions=_tensor(0x1200, (1,), "int32"),
        parent_rows=_tensor(0x1300, (2,), "int32"),
        linear_state_src=_tensor(0x1400, (2, 4), "bf16"),
        linear_state_dst=_tensor(0x1500, (1, 4), "bf16"),
    )

    with pytest.raises(ValueError, match="target_rows"):
        dflash_commit_chain_i32(buffers, target_rows=0)
    with pytest.raises(ValueError, match="parent_rows"):
        dflash_commit_chain_i32(buffers, target_rows=3)
    int64_summary = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x1600, (1,), "int64"),
        commit_rows=_tensor(0x1700, (1,), "int32"),
        commit_positions=_tensor(0x1800, (1,), "int32"),
        linear_state_src=_tensor(0x1900, (2, 4), "bf16"),
        linear_state_dst=_tensor(0x1A00, (1, 4), "bf16"),
    )
    with pytest.raises(ValueError, match="accepted_counts"):
        dflash_commit_chain_i32(int64_summary, target_rows=2)
