from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.loading.gguf import MissingGGUFTensorError
from hipengine.loading.laguna_gguf import (
    DENSE_MLP,
    FULL_ATTENTION,
    PER_ELEMENT_GATE,
    PER_HEAD_GATE,
    SPARSE_MOE,
    SLIDING_ATTENTION,
    build_laguna_gguf_tensor_map,
    required_laguna_gguf_tensor_names,
    validate_laguna_gguf_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._laguna_synthetic import (
    laguna_q2_xl_tensors,
    laguna_tensors,
    make_laguna_info,
    tensor_info,
)


def _info():
    return make_laguna_info(tensors=laguna_tensors())


def test_laguna_gguf_tensor_map_covers_production_inventory() -> None:
    info = _info()
    model_map = build_laguna_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert len(model_map.tensor_names) == 814
    assert set(model_map.tensor_names) == {tensor.name for tensor in info.tensors}
    assert set(required_laguna_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)

    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("token_embedding").ggml_type_name == "Q4_K"
    assert model_map.root("lm_head").name == "output.weight"
    assert model_map.root("lm_head").ggml_type_name == "Q6_K"
    assert model_map.root("output_norm").ggml_type_name == "F32"

    layer0 = model_map.layer(0)
    assert layer0.attention_type == FULL_ATTENTION
    assert layer0.mlp_type == DENSE_MLP
    assert layer0.attention_gate_type == PER_HEAD_GATE
    assert layer0.tensor("attn_gate").shape == (48, 3_072)
    assert layer0.tensor("attn_q").shape == (6_144, 3_072)
    assert layer0.tensor("attn_output").shape == (3_072, 6_144)
    assert layer0.tensor("ffn_down").shape == (3_072, 12_288)
    with pytest.raises(MissingGGUFTensorError, match="ffn_gate_inp"):
        layer0.tensor("ffn_gate_inp")

    layer1 = model_map.layer(1)
    assert layer1.attention_type == SLIDING_ATTENTION
    assert layer1.mlp_type == SPARSE_MOE
    assert layer1.attention_gate_type == PER_HEAD_GATE
    assert layer1.tensor("attn_gate").shape == (72, 3_072)
    assert layer1.tensor("attn_q").shape == (9_216, 3_072)
    assert layer1.tensor("exp_probs_b").name == "blk.1.exp_probs_b.bias"
    assert layer1.tensor("ffn_gate_inp").shape == (256, 3_072)
    assert layer1.tensor("ffn_gate_exps").shape == (256, 1_024, 3_072)
    assert layer1.tensor("ffn_down_exps").shape == (256, 3_072, 1_024)
    assert layer1.tensor("ffn_gate_shexp").shape == (1_024, 3_072)
    assert layer1.tensor("ffn_down_shexp").shape == (3_072, 1_024)


def test_laguna_q2_xl_tensor_map_accepts_exact_quant_recipe() -> None:
    info = make_laguna_info(tensors=laguna_q2_xl_tensors())

    model_map = build_laguna_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert len(model_map.tensor_names) == 814
    assert model_map.root("token_embedding").ggml_type_name == "Q5_K"
    assert model_map.root("lm_head").ggml_type_name == "Q4_K"
    assert model_map.layer(1).tensor("ffn_gate_exps").ggml_type_name == "IQ2_XS"
    assert model_map.layer(1).tensor("ffn_down_exps").ggml_type_name == "IQ3_XXS"
    assert model_map.layer(46).tensor("ffn_down_exps").ggml_type_name == "IQ4_XS"
    assert model_map.layer(47).tensor("ffn_gate_exps").ggml_type_name == "IQ3_XXS"
    assert model_map.layer(47).tensor("attn_k").ggml_type_name == "Q8_0"


def test_laguna_gguf_tensor_map_accepts_per_element_attention_gate() -> None:
    info = _replace_tensor(
        _info(),
        "blk.1.attn_gate.weight",
        replacement=tensor_info(
            "blk.1.attn_gate.weight",
            (9_216, 3_072),
            GGMLQuantizationType.F16,
        ),
    )

    model_map = build_laguna_gguf_tensor_map(info)

    assert model_map.layer(1).attention_gate_type == PER_ELEMENT_GATE


def test_laguna_gguf_tensor_map_reports_missing_correction_bias() -> None:
    info = _without_tensor(_info(), "blk.1.exp_probs_b.bias")

    validation = validate_laguna_gguf_tensor_map(info)
    assert "blk.1.exp_probs_b.bias" in validation.missing
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="blk.1.exp_probs_b.bias"):
        build_laguna_gguf_tensor_map(info)


def test_laguna_gguf_tensor_map_requires_untied_output() -> None:
    info = _without_tensor(_info(), "output.weight")

    validation = validate_laguna_gguf_tensor_map(info)
    assert "output.weight" in validation.missing
    with pytest.raises(MissingGGUFTensorError, match="output.weight"):
        build_laguna_gguf_tensor_map(info)


def test_laguna_gguf_tensor_map_reports_unexpected_tensor() -> None:
    extra = replace(_info().tensors[0], name="blk.0.unexpected.weight")
    info = replace(_info(), tensors=_info().tensors + (extra,))

    validation = validate_laguna_gguf_tensor_map(info)
    assert "blk.0.unexpected.weight" in validation.unexpected
    with pytest.raises(MissingGGUFTensorError, match="unexpected tensors"):
        build_laguna_gguf_tensor_map(info)


def test_laguna_gguf_tensor_map_reports_variable_q_shape_error() -> None:
    info = _replace_tensor(
        _info(),
        "blk.1.attn_q.weight",
        replacement=tensor_info(
            "blk.1.attn_q.weight",
            (6_144, 3_072),
            GGMLQuantizationType.F16,
        ),
    )

    validation = validate_laguna_gguf_tensor_map(info)
    assert any("blk.1.attn_q.weight" in item for item in validation.shape_errors)
    with pytest.raises(MissingGGUFTensorError, match="shape errors"):
        build_laguna_gguf_tensor_map(info)


def test_laguna_gguf_tensor_map_rejects_invalid_attention_gate_width() -> None:
    info = _replace_tensor(
        _info(),
        "blk.1.attn_gate.weight",
        replacement=tensor_info(
            "blk.1.attn_gate.weight",
            (71, 3_072),
            GGMLQuantizationType.F16,
        ),
    )

    validation = validate_laguna_gguf_tensor_map(info)
    assert any("blk.1.attn_gate.weight" in item for item in validation.shape_errors)
    with pytest.raises(MissingGGUFTensorError, match="shape errors"):
        build_laguna_gguf_tensor_map(info)


def test_laguna_gguf_tensor_map_reports_source_type_error() -> None:
    info = _replace_tensor(
        _info(),
        "blk.1.attn_q.weight",
        replacement=tensor_info(
            "blk.1.attn_q.weight",
            (9_216, 3_072),
            GGMLQuantizationType.BF16,
        ),
    )

    validation = validate_laguna_gguf_tensor_map(info)
    assert any("blk.1.attn_q.weight" in item for item in validation.type_errors)
    with pytest.raises(MissingGGUFTensorError, match="type errors"):
        build_laguna_gguf_tensor_map(info)


def _without_tensor(info, name: str):
    return replace(info, tensors=tuple(tensor for tensor in info.tensors if tensor.name != name))


def _replace_tensor(info, name: str, *, replacement):
    return replace(
        info,
        tensors=tuple(replacement if tensor.name == name else tensor for tensor in info.tensors),
    )
