from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFModelInfo, GGUFReader, MissingGGUFTensorError
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    build_qwen35_gguf_tensor_map,
    required_qwen35_gguf_tensor_names,
    validate_qwen35_gguf_tensor_map,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def _info() -> GGUFModelInfo:
    return GGUFReader(MODEL).info


def test_qwen35_gguf_tensor_map_covers_local_inventory() -> None:
    info = _info()
    model_map = build_qwen35_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert model_map.config.architecture == "qwen35"
    assert model_map.config.block_count == 24
    assert model_map.config.hidden_size == 1024
    assert model_map.config.vocab_size == 248320
    assert model_map.config.layer_types.count(FULL_ATTENTION) == 6
    assert model_map.config.layer_types.count(LINEAR_ATTENTION) == 18
    assert len(model_map.layers) == 24
    assert set(model_map.tensor_names) == {tensor.name for tensor in info.tensors}
    assert len(model_map.tensor_names) == len(info.tensors)
    assert set(required_qwen35_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)

    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("token_embedding").ggml_type_name == "Q6_K"
    assert model_map.root("lm_head").name == "token_embd.weight"
    assert model_map.root("output_norm").shape == (1024,)

    layer0 = model_map.layer(0)
    assert layer0.layer_type == LINEAR_ATTENTION
    assert layer0.tensor("attn_qkv").name == "blk.0.attn_qkv.weight"
    assert layer0.tensor("attn_qkv").ggml_type_name == "Q5_K"
    assert layer0.tensor("attn_gate").ggml_type_name == "Q4_K"
    assert layer0.tensor("ssm_out").ggml_type_name == "Q5_K"
    assert layer0.tensor("ssm_alpha").ggml_type_name == "Q8_0"

    layer3 = model_map.layer(3)
    assert layer3.layer_type == FULL_ATTENTION
    assert layer3.tensor("attn_q").name == "blk.3.attn_q.weight"
    assert layer3.tensor("attn_q").shape == (4096, 1024)
    assert layer3.tensor("attn_k").shape == (512, 1024)
    assert layer3.tensor("attn_v").ggml_type_name == "Q6_K"
    assert layer3.tensor("attn_output").ggml_type_name == "Q4_K"


def test_qwen35moe_gguf_tensor_map_covers_local_inventory() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    info = GGUFReader(MOE_MODEL).info
    model_map = build_qwen35_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert model_map.config.architecture == "qwen35moe"
    assert model_map.config.block_count == 40
    assert model_map.config.hidden_size == 2048
    assert model_map.config.vocab_size == 248320
    assert model_map.config.expert_count == 256
    assert model_map.config.expert_used_count == 8
    assert model_map.config.expert_feed_forward_length == 512
    assert model_map.config.expert_shared_feed_forward_length == 512
    assert model_map.config.layer_types.count(FULL_ATTENTION) == 10
    assert model_map.config.layer_types.count(LINEAR_ATTENTION) == 30
    assert len(model_map.layers) == 40
    assert set(model_map.tensor_names) == {tensor.name for tensor in info.tensors}
    assert len(model_map.tensor_names) == len(info.tensors)
    assert set(required_qwen35_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)

    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("lm_head").name == "output.weight"
    assert model_map.root("lm_head").ggml_type_name == "Q6_K"

    layer0 = model_map.layer(0)
    assert layer0.layer_type == LINEAR_ATTENTION
    assert layer0.tensor("ffn_gate_inp").shape == (256, 2048)
    assert layer0.tensor("ffn_gate_exps").shape == (256, 512, 2048)
    assert layer0.tensor("ffn_down_exps").shape == (256, 2048, 512)
    assert layer0.tensor("ffn_gate_shexp").shape == (512, 2048)


def test_qwen35_gguf_tensor_map_reports_missing_tensor() -> None:
    info = _info()
    broken = _without_tensor(info, "blk.0.attn_qkv.weight")

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert "blk.0.attn_qkv.weight" in validation.missing
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="blk.0.attn_qkv.weight"):
        build_qwen35_gguf_tensor_map(broken)


def test_qwen35_gguf_tensor_map_reports_unexpected_tensor() -> None:
    info = _info()
    extra = replace(info.tensors[0], name="blk.0.unexpected.weight")
    broken = replace(info, tensors=info.tensors + (extra,))

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert "blk.0.unexpected.weight" in validation.unexpected
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="unexpected tensors"):
        build_qwen35_gguf_tensor_map(broken)


def test_qwen35_gguf_tensor_map_reports_shape_error() -> None:
    info = _info()
    broken = _replace_tensor(info, "blk.0.ssm_norm.weight", shape=(129,))

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert any("blk.0.ssm_norm.weight" in item for item in validation.shape_errors)
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="shape errors"):
        build_qwen35_gguf_tensor_map(broken)


def _without_tensor(info: GGUFModelInfo, name: str) -> GGUFModelInfo:
    return replace(info, tensors=tuple(tensor for tensor in info.tensors if tensor.name != name))


def _replace_tensor(info: GGUFModelInfo, name: str, **updates) -> GGUFModelInfo:
    tensors = tuple(replace(tensor, **updates) if tensor.name == name else tensor for tensor in info.tensors)
    return replace(info, tensors=tensors)
