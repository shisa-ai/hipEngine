from __future__ import annotations

from pathlib import Path
from typing import Any

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    ggml_type_name,
    nbytes_for_shape,
    quant_shape_to_byte_shape,
)


def laguna_metadata() -> dict[str, Any]:
    return {
        "general.architecture": "laguna",
        "general.file_type": 15,
        "general.quantization_version": 2,
        "laguna.attention.head_count": [48, 72, 72, 72] * 12,
        "laguna.attention.head_count_kv": 8,
        "laguna.attention.key_length": 128,
        "laguna.attention.layer_norm_rms_epsilon": 1.0e-6,
        "laguna.attention.sliding_window": 512,
        "laguna.attention.value_length": 128,
        "laguna.block_count": 48,
        "laguna.context_length": 262_144,
        "laguna.embedding_length": 3_072,
        "laguna.expert_count": 256,
        "laguna.expert_feed_forward_length": 1_024,
        "laguna.expert_gating_func": 2,
        "laguna.expert_shared_feed_forward_length": 1_024,
        "laguna.expert_used_count": 10,
        "laguna.expert_weights_norm": True,
        "laguna.expert_weights_scale": 2.5,
        "laguna.feed_forward_length": 12_288,
        "laguna.leading_dense_block_count": 1,
        "laguna.rope.dimension_count": 64,
        "laguna.rope.dimension_count_swa": 128,
        "laguna.rope.freq_base": 500_000.0,
        "laguna.rope.freq_base_swa": 10_000.0,
        "laguna.rope.scaling.factor": 32.0,
        "laguna.rope.scaling.original_context_length": 8_192,
        "laguna.rope.scaling.type": "yarn",
        "laguna.rope.scaling.yarn_attn_factor": 1.0,
        "laguna.rope.scaling.yarn_beta_fast": 32.0,
        "laguna.rope.scaling.yarn_beta_slow": 1.0,
        "laguna.vocab_size": 100_352,
    }


def make_laguna_info(
    *,
    metadata: dict[str, Any] | None = None,
    tensors: tuple[GGUFTensorInfo, ...] = (),
) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("/synthetic/laguna-s-2.1-Q4_K_M.gguf"),
        version=3,
        alignment=32,
        metadata=laguna_metadata() if metadata is None else metadata,
        tensors=tensors,
        tensor_data_offset=0,
    )


def laguna_tensors() -> tuple[GGUFTensorInfo, ...]:
    tensors = [
        tensor_info("token_embd.weight", (100_352, 3_072), GGMLQuantizationType.Q4_K),
        tensor_info("output_norm.weight", (3_072,), GGMLQuantizationType.F32),
        tensor_info("output.weight", (100_352, 3_072), GGMLQuantizationType.Q6_K),
    ]
    for layer_id, heads in enumerate([48, 72, 72, 72] * 12):
        prefix = f"blk.{layer_id}"
        tensors.extend(
            (
                tensor_info(f"{prefix}.attn_norm.weight", (3_072,), GGMLQuantizationType.F32),
                tensor_info(f"{prefix}.attn_q.weight", (heads * 128, 3_072), GGMLQuantizationType.F16),
                tensor_info(f"{prefix}.attn_k.weight", (1_024, 3_072), GGMLQuantizationType.F16),
                tensor_info(f"{prefix}.attn_v.weight", (1_024, 3_072), GGMLQuantizationType.F16),
                tensor_info(f"{prefix}.attn_gate.weight", (heads, 3_072), GGMLQuantizationType.F16),
                tensor_info(f"{prefix}.attn_q_norm.weight", (128,), GGMLQuantizationType.F32),
                tensor_info(f"{prefix}.attn_k_norm.weight", (128,), GGMLQuantizationType.F32),
                tensor_info(f"{prefix}.attn_output.weight", (3_072, heads * 128), GGMLQuantizationType.F16),
                tensor_info(f"{prefix}.ffn_norm.weight", (3_072,), GGMLQuantizationType.F32),
            )
        )
        if layer_id == 0:
            tensors.extend(
                (
                    tensor_info(f"{prefix}.ffn_gate.weight", (12_288, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_up.weight", (12_288, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_down.weight", (3_072, 12_288), GGMLQuantizationType.Q6_K),
                )
            )
        else:
            tensors.extend(
                (
                    tensor_info(f"{prefix}.ffn_gate_inp.weight", (256, 3_072), GGMLQuantizationType.F32),
                    tensor_info(f"{prefix}.exp_probs_b.bias", (256,), GGMLQuantizationType.F32),
                    tensor_info(f"{prefix}.ffn_gate_exps.weight", (256, 1_024, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_up_exps.weight", (256, 1_024, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_down_exps.weight", (256, 3_072, 1_024), GGMLQuantizationType.Q6_K),
                    tensor_info(f"{prefix}.ffn_gate_shexp.weight", (1_024, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_up_shexp.weight", (1_024, 3_072), GGMLQuantizationType.Q4_K),
                    tensor_info(f"{prefix}.ffn_down_shexp.weight", (3_072, 1_024), GGMLQuantizationType.Q6_K),
                )
            )
    return tuple(tensors)


def laguna_q2_xl_tensors() -> tuple[GGUFTensorInfo, ...]:
    """Exact tensor-type recipe from Laguna-S-2.1-UD-Q2_K_XL."""

    result: list[GGUFTensorInfo] = []
    for tensor in laguna_tensors():
        name = tensor.name
        qtype = GGMLQuantizationType(tensor.ggml_type)
        if name == "token_embd.weight":
            qtype = GGMLQuantizationType.Q5_K
        elif name == "output.weight":
            qtype = GGMLQuantizationType.Q4_K
        elif name.startswith("blk."):
            layer_id = int(name.split(".", 2)[1])
            suffix = name.split(".", 2)[2]
            if suffix in {"attn_q.weight", "attn_gate.weight", "attn_output.weight"}:
                qtype = (
                    GGMLQuantizationType.Q6_K
                    if layer_id == 47
                    else GGMLQuantizationType.Q5_K
                )
            elif suffix in {"attn_k.weight", "attn_v.weight"}:
                qtype = (
                    GGMLQuantizationType.Q8_0
                    if layer_id == 47
                    else GGMLQuantizationType.Q6_K
                )
            elif suffix in {"ffn_gate.weight", "ffn_up.weight"}:
                qtype = GGMLQuantizationType.Q5_K
            elif suffix == "ffn_gate_exps.weight" or suffix == "ffn_up_exps.weight":
                qtype = (
                    GGMLQuantizationType.IQ3_XXS
                    if layer_id == 47
                    else GGMLQuantizationType.IQ2_XS
                )
            elif suffix == "ffn_down_exps.weight":
                qtype = (
                    GGMLQuantizationType.IQ4_XS
                    if layer_id >= 46
                    else GGMLQuantizationType.IQ3_XXS
                )
            elif suffix in {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"}:
                qtype = (
                    GGMLQuantizationType.Q6_K
                    if layer_id == 47
                    else GGMLQuantizationType.Q5_K
                )
            elif suffix == "ffn_down_shexp.weight":
                qtype = (
                    GGMLQuantizationType.Q8_0
                    if layer_id == 47
                    else GGMLQuantizationType.Q6_K
                )
        result.append(tensor_info(name, tensor.shape, qtype))
    return tuple(result)


def tensor_info(
    name: str,
    shape: tuple[int, ...],
    qtype: GGMLQuantizationType,
) -> GGUFTensorInfo:
    ggml_shape = tuple(reversed(shape))
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=ggml_shape,
        ggml_type=int(qtype),
        ggml_type_name=ggml_type_name(qtype),
        n_elements=1,
        nbytes=nbytes_for_shape(shape, qtype),
        offset=0,
        data_offset=0,
        byte_shape=quant_shape_to_byte_shape(shape, qtype),
    )
