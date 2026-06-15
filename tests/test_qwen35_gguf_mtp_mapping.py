from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    build_qwen35_gguf_mtp_block_maps,
    build_qwen35_gguf_mtp_draft_specs,
    build_qwen35_gguf_mtp_draft_tensor_plans,
    build_qwen35_gguf_tensor_map,
    qwen35_gguf_mtp_block_inventories,
    required_qwen35_gguf_tensor_names,
    validate_qwen35_gguf_mtp_blocks,
    validate_qwen35_gguf_tensor_map,
)


def test_qwen35moe_gguf_map_ignores_trailing_mtp_nextn_block() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    validation = validate_qwen35_gguf_tensor_map(info)
    assert validation.passed
    assert validation.config.declared_block_count == 3
    assert validation.config.block_count == 2
    assert validation.config.ignored_block_ids == (2,)
    assert validation.config.layer_types == (LINEAR_ATTENTION, FULL_ATTENTION)
    assert validation.ignored
    assert all(name.startswith("blk.2.") for name in validation.ignored)

    model_map = build_qwen35_gguf_tensor_map(info)
    assert model_map.validation.passed
    assert len(model_map.layers) == 2
    assert not any(name.startswith("blk.2.") for name in model_map.tensor_names)
    assert set(required_qwen35_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)
    assert len(model_map.mtp_blocks) == 1
    assert model_map.mtp_blocks[0].layer_id == 2


def test_qwen35moe_gguf_mtp_inventory_reports_required_optional_and_fallbacks() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    (block,) = qwen35_gguf_mtp_block_inventories(info)

    assert block.passed
    assert block.layer_id == 2
    assert len(block.tensor_names) == 20
    assert len(block.nextn_tensor_names) == 4
    assert block.missing_required_tensor_names == ()
    assert block.optional_tensor_names == ("blk.2.nextn.shared_head_norm.weight",)
    assert block.missing_optional_tensor_names == (
        "blk.2.nextn.embed_tokens.weight",
        "blk.2.nextn.shared_head_head.weight",
    )
    assert dict(block.optional_fallback_tensor_names) == {
        "nextn.embed_tokens": "token_embd.weight",
        "nextn.shared_head_head": "output.weight",
    }
    assert block.unexpected_tensor_names == ()


def test_qwen35moe_gguf_mtp_inventory_detects_missing_required_nextn() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        drop_tensors={"blk.2.nextn.hnorm.weight"},
    )

    (block,) = qwen35_gguf_mtp_block_inventories(info)

    assert not block.passed
    assert block.missing_required_tensor_names == ("blk.2.nextn.hnorm.weight",)


def test_qwen35moe_gguf_mtp_block_validation_passes_complete_inventory() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    (block,) = validate_qwen35_gguf_mtp_blocks(info)

    assert block.layer_id == 2
    assert block.passed


def test_qwen35moe_gguf_mtp_block_map_resolves_target_fallbacks() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    (block_map,) = build_qwen35_gguf_mtp_block_maps(info)

    assert block_map.layer_id == 2
    assert block_map.tensor("attn_q").name == "blk.2.attn_q.weight"
    assert block_map.tensor("ffn_gate_exps").name == "blk.2.ffn_gate_exps.weight"
    assert block_map.tensor("nextn.eh_proj").name == "blk.2.nextn.eh_proj.weight"
    assert (
        block_map.tensor("nextn.shared_head_norm").name
        == "blk.2.nextn.shared_head_norm.weight"
    )
    assert block_map.tensor("nextn.embed_tokens").name == "token_embd.weight"
    assert block_map.tensor("nextn.shared_head_head").name == "output.weight"
    assert dict(block_map.fallback_slots) == {
        "nextn.embed_tokens": "token_embedding",
        "nextn.shared_head_head": "lm_head",
    }
    assert "token_embd.weight" in block_map.tensor_names
    assert "output.weight" in block_map.tensor_names


def test_qwen35moe_gguf_mtp_block_map_prefers_present_optional_tensors() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[
            _tensor("blk.2.nextn.embed_tokens.weight", (11, 8)),
            _tensor("blk.2.nextn.shared_head_head.weight", (11, 8)),
        ],
    )

    (block_map,) = build_qwen35_gguf_mtp_block_maps(info)

    assert (
        block_map.tensor("nextn.embed_tokens").name
        == "blk.2.nextn.embed_tokens.weight"
    )
    assert (
        block_map.tensor("nextn.shared_head_head").name
        == "blk.2.nextn.shared_head_head.weight"
    )
    assert dict(block_map.fallback_slots) == {}


def test_qwen35moe_gguf_mtp_draft_spec_uses_target_fallback_shapes() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    (spec,) = build_qwen35_gguf_mtp_draft_specs(info)

    assert spec.layer_id == 2
    assert spec.hidden_size == 8
    assert spec.vocab_size == 11
    assert spec.eh_proj_shape == (8, 16)
    assert dict(spec.tensor_shapes) == {
        "attn_norm": (8,),
        "post_attention_norm": (8,),
        "attn_q": (16, 8),
        "attn_k": (4, 8),
        "attn_v": (4, 8),
        "attn_output": (8, 8),
        "attn_q_norm": (4,),
        "attn_k_norm": (4,),
        "nextn.eh_proj": (8, 16),
        "nextn.enorm": (8,),
        "nextn.hnorm": (8,),
        "nextn.shared_head_norm": (8,),
        "nextn.embed_tokens": (11, 8),
        "nextn.shared_head_head": (11, 8),
        "ffn_gate_inp": (3, 8),
        "ffn_gate_inp_shexp": (8,),
        "ffn_gate_exps": (3, 5, 8),
        "ffn_up_exps": (3, 5, 8),
        "ffn_down_exps": (3, 8, 5),
        "ffn_gate_shexp": (6, 8),
        "ffn_up_shexp": (6, 8),
        "ffn_down_shexp": (8, 6),
    }
    assert spec.as_dict()["tensor_shapes"]["attn_output"] == [8, 8]
    assert spec.embed_tokens_tensor == "token_embd.weight"
    assert spec.shared_head_tensor == "output.weight"
    assert spec.shared_head_norm_tensor == "blk.2.nextn.shared_head_norm.weight"
    assert dict(spec.fallback_slots) == {
        "nextn.embed_tokens": "token_embedding",
        "nextn.shared_head_head": "lm_head",
    }


def test_qwen35moe_gguf_mtp_draft_spec_prefers_present_optional_tensors() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[
            _tensor("blk.2.nextn.embed_tokens.weight", (11, 8)),
            _tensor("blk.2.nextn.shared_head_head.weight", (11, 8)),
        ],
    )

    (spec,) = build_qwen35_gguf_mtp_draft_specs(info)

    assert spec.embed_tokens_tensor == "blk.2.nextn.embed_tokens.weight"
    assert spec.shared_head_tensor == "blk.2.nextn.shared_head_head.weight"
    assert dict(spec.fallback_slots) == {}


def test_qwen35moe_gguf_mtp_draft_tensor_plan_orders_cpu_oracle_slots() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    (plan,) = build_qwen35_gguf_mtp_draft_tensor_plans(info)

    assert plan.layer_id == 2
    assert plan.hidden_size == 8
    assert plan.vocab_size == 11
    assert plan.cpu_reference_kernel == (
        "cpu_reference",
        "mtp_nextn_layer",
        "gguf_moe",
        "qwen35_dense_logits",
    )
    assert plan.num_heads == 2
    assert plan.num_kv_heads == 1
    assert plan.qk_head_dim == 4
    assert plan.value_head_dim == 4
    assert plan.attention_width == 8
    assert plan.experts_used == 1
    assert plan.expert_weights_scale == 0.0
    assert plan.rms_norm_eps == 1.0e-6
    assert plan.rotary_dim == 4
    assert plan.rope_freq_base == 10000000.0
    assert plan.rope_dimension_sections == ()
    assert plan.attention_scale == 0.5
    assert dict(plan.kernel_kwargs) == {
        "num_heads": 2,
        "num_kv_heads": 1,
        "experts_used": 1,
        "rotary_dim": 4,
        "scale": 0.5,
        "expert_weights_scale": 0.0,
        "eps": 1.0e-6,
    }
    assert [slot.slot for slot in plan.slots] == [
        "nextn.embed_tokens",
        "nextn.eh_proj",
        "nextn.hnorm",
        "nextn.enorm",
        "attn_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_q_norm",
        "attn_k_norm",
        "post_attention_norm",
        "ffn_gate_inp",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_inp_shexp",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "nextn.shared_head_norm",
        "nextn.shared_head_head",
    ]
    assert plan.slot("nextn.embed_tokens").tensor_name == "token_embd.weight"
    assert plan.slot("nextn.embed_tokens").fallback_slot == "token_embedding"
    assert plan.slot("nextn.shared_head_head").tensor_name == "output.weight"
    assert plan.slot("nextn.shared_head_head").fallback_slot == "lm_head"
    assert plan.slot("ffn_gate_exps").shape == (3, 5, 8)
    assert plan.slot("ffn_gate_exps").ggml_type_name == "F32"
    plan_dict = plan.as_dict()
    assert plan_dict["kernel_kwargs"] == {
        "num_heads": 2,
        "num_kv_heads": 1,
        "experts_used": 1,
        "rotary_dim": 4,
        "scale": 0.5,
        "expert_weights_scale": 0.0,
        "eps": 1.0e-6,
    }
    assert plan_dict["slots"][0] == {
        "slot": "nextn.embed_tokens",
        "tensor_name": "token_embd.weight",
        "shape": [11, 8],
        "ggml_type_name": "F32",
        "fallback_slot": "token_embedding",
    }


def test_qwen35moe_gguf_mtp_draft_tensor_plan_prefers_present_optional_tensors() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[
            _tensor("blk.2.nextn.embed_tokens.weight", (11, 8)),
            _tensor("blk.2.nextn.shared_head_head.weight", (11, 8)),
        ],
    )

    (plan,) = build_qwen35_gguf_mtp_draft_tensor_plans(info)

    assert plan.slot("nextn.embed_tokens").tensor_name == "blk.2.nextn.embed_tokens.weight"
    assert plan.slot("nextn.embed_tokens").fallback_slot is None
    assert plan.slot("nextn.shared_head_head").tensor_name == "blk.2.nextn.shared_head_head.weight"
    assert plan.slot("nextn.shared_head_head").fallback_slot is None
    assert dict(plan.fallback_slots) == {}


def test_qwen35moe_gguf_mtp_draft_spec_rejects_bad_nextn_shape() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[_tensor("blk.2.nextn.eh_proj.weight", (8, 8))],
    )

    with pytest.raises(
        MissingGGUFTensorError,
        match="nextn\\.eh_proj has shape \\(8, 8\\), expected \\(8, 16\\)",
    ):
        build_qwen35_gguf_mtp_draft_specs(info)


def test_qwen35moe_gguf_mtp_draft_spec_rejects_bad_attention_shape() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[_tensor("blk.2.attn_output.weight", (8, 16))],
    )

    with pytest.raises(
        MissingGGUFTensorError,
        match="attn_output has shape \\(8, 16\\), expected \\(8, 8\\)",
    ):
        build_qwen35_gguf_mtp_draft_specs(info)


def test_qwen35moe_gguf_mtp_block_validation_fails_missing_required_nextn() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        drop_tensors={"blk.2.nextn.hnorm.weight"},
    )

    with pytest.raises(
        MissingGGUFTensorError,
        match="MTP block 2 missing required tensors: blk\\.2\\.nextn\\.hnorm\\.weight",
    ):
        validate_qwen35_gguf_mtp_blocks(info)


def test_qwen35moe_gguf_mtp_block_validation_fails_unexpected_trailing_tensor() -> None:
    info = _synthetic_qwen35moe_mtp_info(
        extra_tensors=[_tensor("blk.2.nextn.surprise.weight", (8,))],
    )

    with pytest.raises(
        MissingGGUFTensorError,
        match="MTP block 2 unexpected tensors: blk\\.2\\.nextn\\.surprise\\.weight",
    ):
        validate_qwen35_gguf_mtp_blocks(info)


def _synthetic_qwen35moe_mtp_info(
    *,
    drop_tensors: set[str] | None = None,
    extra_tensors: list[GGUFTensorInfo] | None = None,
) -> GGUFModelInfo:
    metadata = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 3,
        "qwen35moe.embedding_length": 8,
        "qwen35moe.context_length": 128,
        "qwen35moe.attention.head_count": 2,
        "qwen35moe.attention.head_count_kv": 1,
        "qwen35moe.attention.key_length": 4,
        "qwen35moe.attention.value_length": 4,
        "qwen35moe.full_attention_interval": 2,
        "qwen35moe.rope.dimension_count": 4,
        "qwen35moe.rope.dimension_sections": (),
        "qwen35moe.ssm.inner_size": 16,
        "qwen35moe.ssm.group_count": 2,
        "qwen35moe.ssm.state_size": 3,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.time_step_rank": 2,
        "qwen35moe.expert_count": 3,
        "qwen35moe.expert_used_count": 1,
        "qwen35moe.expert_feed_forward_length": 5,
        "qwen35moe.expert_shared_feed_forward_length": 6,
    }
    tensors = [
        _tensor("token_embd.weight", (11, 8)),
        _tensor("output_norm.weight", (8,)),
        _tensor("output.weight", (11, 8)),
    ]
    tensors.extend(_qwen35moe_common_mlp_tensors(0))
    tensors.extend(
        [
            _tensor("blk.0.attn_gate.weight", (16, 8)),
            _tensor("blk.0.attn_qkv.weight", (28, 8)),
            _tensor("blk.0.ssm_a", (2,)),
            _tensor("blk.0.ssm_alpha.weight", (2, 8)),
            _tensor("blk.0.ssm_beta.weight", (2, 8)),
            _tensor("blk.0.ssm_conv1d.weight", (28, 4)),
            _tensor("blk.0.ssm_dt.bias", (2,)),
            _tensor("blk.0.ssm_norm.weight", (3,)),
            _tensor("blk.0.ssm_out.weight", (8, 16)),
        ]
    )
    tensors.extend(_qwen35moe_common_mlp_tensors(1))
    tensors.extend(_full_attention_tensors(1))
    tensors.extend(_qwen35moe_common_mlp_tensors(2))
    tensors.extend(_full_attention_tensors(2))
    tensors.extend(
        [
            _tensor("blk.2.nextn.eh_proj.weight", (8, 16)),
            _tensor("blk.2.nextn.enorm.weight", (8,)),
            _tensor("blk.2.nextn.hnorm.weight", (8,)),
            _tensor("blk.2.nextn.shared_head_norm.weight", (8,)),
        ]
    )
    if extra_tensors:
        tensors.extend(extra_tensors)
    if drop_tensors:
        tensors = [tensor for tensor in tensors if tensor.name not in drop_tensors]
    return GGUFModelInfo(
        path=Path("synthetic-qwen35moe-mtp.gguf"),
        version=3,
        alignment=32,
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )


def _qwen35moe_common_mlp_tensors(layer_id: int) -> list[GGUFTensorInfo]:
    prefix = f"blk.{layer_id}"
    return [
        _tensor(f"{prefix}.attn_norm.weight", (8,)),
        _tensor(f"{prefix}.post_attention_norm.weight", (8,)),
        _tensor(f"{prefix}.ffn_gate_inp.weight", (3, 8)),
        _tensor(f"{prefix}.ffn_gate_inp_shexp.weight", (8,)),
        _tensor(f"{prefix}.ffn_gate_exps.weight", (3, 5, 8)),
        _tensor(f"{prefix}.ffn_up_exps.weight", (3, 5, 8)),
        _tensor(f"{prefix}.ffn_down_exps.weight", (3, 8, 5)),
        _tensor(f"{prefix}.ffn_gate_shexp.weight", (6, 8)),
        _tensor(f"{prefix}.ffn_up_shexp.weight", (6, 8)),
        _tensor(f"{prefix}.ffn_down_shexp.weight", (8, 6)),
    ]


def _full_attention_tensors(layer_id: int) -> list[GGUFTensorInfo]:
    prefix = f"blk.{layer_id}"
    return [
        _tensor(f"{prefix}.attn_q.weight", (16, 8)),
        _tensor(f"{prefix}.attn_k.weight", (4, 8)),
        _tensor(f"{prefix}.attn_v.weight", (4, 8)),
        _tensor(f"{prefix}.attn_output.weight", (8, 8)),
        _tensor(f"{prefix}.attn_q_norm.weight", (4,)),
        _tensor(f"{prefix}.attn_k_norm.weight", (4,)),
    ]


def _tensor(name: str, shape: tuple[int, ...]) -> GGUFTensorInfo:
    n_elements = int(prod(shape))
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=0,
        ggml_type_name="F32",
        n_elements=n_elements,
        nbytes=n_elements * 4,
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )
