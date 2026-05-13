from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.loading import (
    MissingTensorError,
    load_weight_index,
    materialize_qwen35_paro_full_attention_moe_c1_prepared_layer,
    materialize_qwen35_paro_full_attention_moe_c1_layer,
    materialize_qwen35_paro_moe_c1_layer,
    normalize_qwen35_weight_name,
    prepare_qwen35_paro_moe_c1_host_tensors,
    prepared_moe_c1_tensor_names,
    qwen35_paro_config_from_hf,
    required_full_attention_c1_tensor_names,
    required_full_attention_moe_c1_tensor_names,
    required_moe_c1_tensor_names,
    validate_qwen35_paro_full_attention_moe_c1_layout,
    validate_qwen35_paro_moe_c1_layout,
)
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x4000
        self.buffers: dict[int, bytearray] = {}
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.buffers[ptr] = bytearray(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.buffers.pop(ptr, None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.buffers[dst][:count] = ctypes.string_at(src, count)


def _write_config(path, *, quant_method: str = "paroquant") -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "num_experts": 2,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 3,
                "shared_expert_intermediate_size": 3,
                "layer_types": ["full_attention"],
                "quantization_config": {"quant_method": quant_method},
            }
        ),
        encoding="utf-8",
    )


def _valid_attention_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float16),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float16),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float16),
    }
    for proj in ("q_proj", "k_proj", "v_proj"):
        base = f"model.layers.0.self_attn.{proj}"
        tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
        tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
        tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
        tensors[f"{base}.theta"] = np.zeros((1, 2), dtype=np.float16)
        tensors[f"{base}.pairs"] = np.zeros((1, 4), dtype=np.int16)
        tensors[f"{base}.channel_scales"] = np.zeros((4,), dtype=np.float16)
    base = "model.layers.0.self_attn.o_proj"
    tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
    tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
    tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
    return tensors


def _valid_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float16),
        "model.layers.0.mlp.gate.weight": np.zeros((2, 4), dtype=np.float16),
        "model.layers.0.mlp.shared_expert_gate.weight": np.zeros((1, 4), dtype=np.float16),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": np.zeros((3, 4), dtype=np.float16),
        "model.layers.0.mlp.shared_expert.up_proj.weight": np.zeros((3, 4), dtype=np.float16),
        "model.layers.0.mlp.shared_expert.down_proj.weight": np.zeros((4, 3), dtype=np.float16),
        "model.layers.0.mlp.experts.gate_up_weight_theta": np.zeros((1, 2), dtype=np.float16),
        "model.layers.0.mlp.experts.gate_up_weight_pairs": np.zeros((1, 4), dtype=np.int16),
        "model.layers.0.mlp.experts.gate_up_weight_channel_scales": np.zeros((4,), dtype=np.float16),
        "model.layers.0.mlp.experts.down_weight_theta": np.zeros((1, 2), dtype=np.float16),
        "model.layers.0.mlp.experts.down_weight_pairs": np.zeros((1, 4), dtype=np.int16),
        "model.layers.0.mlp.experts.down_weight_channel_scales": np.zeros((4,), dtype=np.float16),
    }
    for expert in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.0.mlp.experts.{expert}.{proj}"
            tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
            tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
            tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
    return tensors


def test_qwen35_paro_config_and_weight_name_normalization() -> None:
    config = qwen35_paro_config_from_hf(
        {
            "model_type": "qwen3_5_moe",
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 8,
            "quantization_config": {"quant_method": "paroquant"},
        }
    )

    assert config.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert config.layer_types == ("full_attention", "full_attention")
    assert config.num_attention_heads == 2
    assert config.num_key_value_heads == 1
    assert config.head_dim == 4
    assert config.quant_method == "paroquant"
    assert normalize_qwen35_weight_name("model.layers.0.mlp.gate.weight") == "layers.0.mlp.gate.weight"
    assert normalize_qwen35_weight_name("language_model.layers.0.x") == "layers.0.x"


def test_required_moe_c1_names_include_all_expert_triples() -> None:
    names = required_moe_c1_tensor_names(layer_id=3, num_experts=2)

    assert "layers.3.mlp.gate.weight" in names
    assert "layers.3.mlp.experts.gate_up_weight_theta" in names
    assert "layers.3.mlp.experts.1.down_proj.scales" in names
    assert sum(name.endswith(".qweight") for name in names) == 6


def test_required_full_attention_names_include_rotated_qkv_and_o_proj() -> None:
    names = required_full_attention_c1_tensor_names(layer_id=3)
    combined = required_full_attention_moe_c1_tensor_names(layer_id=3, num_experts=2)

    assert "layers.3.input_layernorm.weight" in names
    assert "layers.3.self_attn.q_norm.weight" in names
    assert "layers.3.self_attn.q_proj.theta" in names
    assert "layers.3.self_attn.v_proj.channel_scales" in names
    assert "layers.3.self_attn.o_proj.qweight" in names
    assert "layers.3.mlp.experts.1.down_proj.scales" in combined


def test_validate_qwen35_paro_moe_c1_layout_passes(tmp_path) -> None:
    _write_config(tmp_path)
    save_file(_valid_tensors(), tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    result = validate_qwen35_paro_moe_c1_layout(index)

    assert result.passed
    assert result.config.num_experts == 2
    assert not result.missing
    assert not result.shape_errors


def test_materialize_qwen35_paro_moe_c1_layer_uses_normalized_device_names(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = _valid_tensors()
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    layer = materialize_qwen35_paro_moe_c1_layer(index, device=Device("hip", 0), runtime=runtime)

    qweight_name = "layers.0.mlp.experts.1.down_proj.qweight"
    prefixed_name = f"model.{qweight_name}"
    assert layer.config.hidden_size == 4
    assert layer.layer_id == 0
    assert layer.tensor(prefixed_name) == layer.tensor(qweight_name)
    assert layer.tensor(qweight_name).dtype is DType.INT32
    assert layer.tensor(qweight_name).shape == tensors[prefixed_name].shape
    qweight_alloc = layer.allocation(qweight_name)
    assert qweight_alloc.name == qweight_name
    assert qweight_alloc.source.name == prefixed_name
    assert bytes(runtime.buffers[qweight_alloc.buffer.ptr]) == tensors[prefixed_name].tobytes()

    pairs = layer.tensor("layers.0.mlp.experts.gate_up_weight_pairs")
    assert pairs.dtype is DType.INT16
    assert pairs.shape == (1, 4)
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(required_moe_c1_tensor_names(layer_id=0, num_experts=2))


def test_materialize_qwen35_paro_full_attention_moe_c1_layer(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_valid_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    result = validate_qwen35_paro_full_attention_moe_c1_layout(index)
    layer = materialize_qwen35_paro_full_attention_moe_c1_layer(index, runtime=runtime)

    assert result.passed
    assert layer.tensor("layers.0.self_attn.q_norm.weight").shape == (2,)
    assert layer.tensor("model.layers.0.self_attn.q_proj.pairs").dtype is DType.INT16
    o_proj_name = "layers.0.self_attn.o_proj.qweight"
    o_proj_prefixed = f"model.{o_proj_name}"
    assert bytes(runtime.buffers[layer.allocation(o_proj_name).buffer.ptr]) == tensors[o_proj_prefixed].tobytes()
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(required_full_attention_moe_c1_tensor_names(layer_id=0, num_experts=2))


def test_prepare_qwen35_paro_moe_c1_host_tensors_matches_parent_stacking(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_valid_tensors()}
    tensors["model.layers.0.mlp.gate.weight"] = np.arange(8, dtype=np.float16).reshape(2, 4)
    tensors["model.layers.0.mlp.shared_expert_gate.weight"] = np.arange(4, dtype=np.float16).reshape(1, 4) + 100
    for expert in range(2):
        base = f"model.layers.0.mlp.experts.{expert}.gate_proj.qweight"
        tensors[base] = (np.arange(4, dtype=np.int32).reshape(4, 1) + expert * 10)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    prepared = prepare_qwen35_paro_moe_c1_host_tensors(index)

    assert set(prepared_moe_c1_tensor_names(layer_id=0)) == set(prepared)
    combined = prepared["layers.0.mlp.router_shared_gate.weight"]
    assert combined.shape == (3, 4)
    np.testing.assert_array_equal(combined[:2], tensors["model.layers.0.mlp.gate.weight"])
    np.testing.assert_array_equal(combined[2:], tensors["model.layers.0.mlp.shared_expert_gate.weight"])
    stacked = prepared["layers.0.mlp.experts.stacked_gate_qweight"]
    transposed = prepared["layers.0.mlp.experts.stacked_gate_qweight_pack8_decode"]
    assert stacked.shape == (2, 4, 1)
    assert transposed.shape == (2, 1, 4)
    np.testing.assert_array_equal(transposed, np.swapaxes(stacked, 1, 2))


def test_materialize_qwen35_paro_full_attention_moe_c1_prepared_layer(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_valid_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    layer = materialize_qwen35_paro_full_attention_moe_c1_prepared_layer(index, runtime=runtime)

    prepared_name = "layers.0.mlp.experts.stacked_gate_qweight_pack8_decode"
    assert layer.tensor(prepared_name).shape == (2, 1, 4)
    assert layer.tensor(prepared_name).dtype is DType.INT32
    assert layer.tensor("layers.0.mlp.router_shared_gate.weight").shape == (3, 4)
    expected_count = len(required_full_attention_moe_c1_tensor_names(layer_id=0, num_experts=2)) + len(
        prepared_moe_c1_tensor_names(layer_id=0)
    )
    layer.free(runtime=runtime)
    assert len(runtime.freed) == expected_count


def test_validate_qwen35_paro_moe_c1_layout_reports_missing_and_shapes(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = _valid_tensors()
    del tensors["model.layers.0.mlp.experts.1.down_proj.scales"]
    tensors["model.layers.0.mlp.gate.weight"] = np.zeros((3, 4), dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    result = validate_qwen35_paro_moe_c1_layout(index)

    assert not result.passed
    assert result.missing == ("layers.0.mlp.experts.1.down_proj.scales",)
    assert result.shape_errors == ("layers.0.mlp.gate.weight: expected (2, 4), got (3, 4)",)
    with pytest.raises(MissingTensorError, match="missing tensors"):
        validate_qwen35_paro_moe_c1_layout(index, raise_on_error=True)


def test_validate_qwen35_paro_rejects_wrong_quant_method(tmp_path) -> None:
    _write_config(tmp_path, quant_method="awq")
    save_file(_valid_tensors(), tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    with pytest.raises(ValueError, match="quant_method='paroquant'"):
        validate_qwen35_paro_moe_c1_layout(index)
