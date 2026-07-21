from __future__ import annotations

from math import prod
from pathlib import Path

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    build_qwen35_gguf_tensor_map,
    required_qwen35_gguf_tensor_names,
    validate_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_nextn import (
    build_qwen35_gguf_nextn_tensor_map,
    required_qwen35_gguf_nextn_tensor_names,
    validate_qwen35_gguf_nextn_tensor_map,
)
from hipengine.loading.qwen35_gguf_nextn_materialize import plan_qwen35_gguf_nextn_materialization
from hipengine.quant.gguf import GGMLQuantizationType


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


def test_qwen35moe_gguf_builds_separate_nextn_map_with_target_fallbacks() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    validation = validate_qwen35_gguf_nextn_tensor_map(info)
    assert validation.passed
    assert validation.block_id == 2
    assert validation.present == tuple(sorted(name for name in required_qwen35_gguf_nextn_tensor_names(2)))
    assert validation.embedding_fallback == "token_embd.weight"
    assert validation.head_fallback == "output.weight"
    assert validation.head_norm_source == "blk.2.nextn.shared_head_norm.weight"

    draft_map = build_qwen35_gguf_nextn_tensor_map(info)
    assert draft_map.block_id == 2
    assert len(draft_map.tensor_names) == 20
    assert set(draft_map.tensor_names) == set(required_qwen35_gguf_nextn_tensor_names(2))
    assert draft_map.tensor("ffn_gate_exps").ggml_type_name == "Q3_K"
    assert draft_map.tensor("ffn_up_exps").ggml_type_name == "Q3_K"
    assert draft_map.tensor("ffn_down_exps").ggml_type_name == "Q4_K"
    assert draft_map.fallback("token_embedding").name == "token_embd.weight"
    assert draft_map.fallback("lm_head").name == "output.weight"

    ar_map = build_qwen35_gguf_tensor_map(info)
    assert ar_map.config.block_count == 2
    assert len(ar_map.layers) == 2
    assert not any(name.startswith("blk.2.") for name in ar_map.tensor_names)


def test_qwen35moe_gguf_nextn_plan_keeps_raw_quant_and_records_fallbacks() -> None:
    draft_map = build_qwen35_gguf_nextn_tensor_map(_synthetic_qwen35moe_mtp_info())
    plan = plan_qwen35_gguf_nextn_materialization(draft_map)

    assert len(plan.draft_specs) == 20
    assert plan.layer_specs["ffn_gate_exps"].quant_key == "gguf_q3_k"
    assert plan.layer_specs["ffn_gate_exps"].layout == "raw_gguf"
    assert plan.layer_specs["ffn_up_exps"].quant_key == "gguf_q3_k"
    assert plan.layer_specs["ffn_down_exps"].quant_key == "gguf_q4_k"
    assert plan.nextn_specs["eh_proj"].quant_key == "gguf_q8_0"
    assert plan.fallback_specs["token_embedding"].source.name == "token_embd.weight"
    assert plan.fallback_specs["lm_head"].source.name == "output.weight"
    assert plan.fallback_specs["output_norm"].source.name == "blk.2.nextn.shared_head_norm.weight"


def _synthetic_qwen35moe_mtp_info() -> GGUFModelInfo:
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
    tensors.extend(_qwen35moe_common_mlp_tensors(2, nextn=True))
    tensors.extend(_full_attention_tensors(2, nextn=True))
    tensors.extend(
        [
            _tensor("blk.2.nextn.eh_proj.weight", (8, 16), GGMLQuantizationType.Q8_0),
            _tensor("blk.2.nextn.enorm.weight", (8,)),
            _tensor("blk.2.nextn.hnorm.weight", (8,)),
            _tensor("blk.2.nextn.shared_head_norm.weight", (8,)),
        ]
    )
    return GGUFModelInfo(
        path=Path("synthetic-qwen35moe-mtp.gguf"),
        version=3,
        alignment=32,
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )


def _qwen35moe_common_mlp_tensors(layer_id: int, *, nextn: bool = False) -> list[GGUFTensorInfo]:
    prefix = f"blk.{layer_id}"
    return [
        _tensor(f"{prefix}.attn_norm.weight", (8,)),
        _tensor(f"{prefix}.post_attention_norm.weight", (8,)),
        _tensor(f"{prefix}.ffn_gate_inp.weight", (3, 8), GGMLQuantizationType.BF16 if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_gate_inp_shexp.weight", (8,), GGMLQuantizationType.BF16 if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_gate_exps.weight", (3, 5, 8), GGMLQuantizationType.Q3_K if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_up_exps.weight", (3, 5, 8), GGMLQuantizationType.Q3_K if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_down_exps.weight", (3, 8, 5), GGMLQuantizationType.Q4_K if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_gate_shexp.weight", (6, 8), GGMLQuantizationType.Q8_0 if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_up_shexp.weight", (6, 8), GGMLQuantizationType.Q8_0 if nextn else GGMLQuantizationType.F32),
        _tensor(f"{prefix}.ffn_down_shexp.weight", (8, 6), GGMLQuantizationType.Q8_0 if nextn else GGMLQuantizationType.F32),
    ]


def _full_attention_tensors(layer_id: int, *, nextn: bool = False) -> list[GGUFTensorInfo]:
    prefix = f"blk.{layer_id}"
    qtype = GGMLQuantizationType.Q8_0 if nextn else GGMLQuantizationType.F32
    return [
        _tensor(f"{prefix}.attn_q.weight", (16, 8), qtype),
        _tensor(f"{prefix}.attn_k.weight", (4, 8), qtype),
        _tensor(f"{prefix}.attn_v.weight", (4, 8), qtype),
        _tensor(f"{prefix}.attn_output.weight", (8, 8 if nextn else 16), qtype),
        _tensor(f"{prefix}.attn_q_norm.weight", (4,)),
        _tensor(f"{prefix}.attn_k_norm.weight", (4,)),
    ]


def _tensor(
    name: str,
    shape: tuple[int, ...],
    qtype: GGMLQuantizationType = GGMLQuantizationType.F32,
) -> GGUFTensorInfo:
    n_elements = int(prod(shape))
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=n_elements,
        nbytes=n_elements * (4 if qtype == GGMLQuantizationType.F32 else 2),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )
