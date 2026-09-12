from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.loading import (
    MissingTensorError,
    float_array_to_bf16_bits,
    load_weight_index,
    materialize_qwen35_paro_full_attention_dense_c1_runtime_layer,
    materialize_qwen35_paro_full_attention_moe_c1_prepared_layer,
    materialize_qwen35_paro_full_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_dense_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_full_attention_moe_c1_layer,
    materialize_qwen35_paro_moe_c1_layer,
    normalize_qwen35_weight_name,
    prepare_qwen35_paro_moe_c1_host_tensors,
    prepare_qwen35_paro_moe_c1_runtime_host_tensors,
    prepared_moe_c1_tensor_names,
    runtime_full_attention_dense_c1_tensor_names,
    runtime_full_attention_moe_c1_tensor_names,
    runtime_linear_attention_dense_c1_tensor_names,
    runtime_linear_attention_moe_c1_tensor_names,
    runtime_prepared_moe_c1_tensor_names,
    qwen35_paro_config_from_hf,
    required_full_attention_c1_tensor_names,
    required_full_attention_moe_c1_tensor_names,
    required_linear_attention_c1_tensor_names,
    required_linear_attention_moe_c1_tensor_names,
    required_moe_c1_tensor_names,
    validate_qwen35_paro_full_attention_dense_c1_layout,
    validate_qwen35_paro_full_attention_moe_c1_layout,
    validate_qwen35_paro_linear_attention_dense_c1_layout,
    validate_qwen35_paro_linear_attention_moe_c1_layout,
    validate_qwen35_paro_moe_c1_layout,
)
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.qwen35_paro import _qwen_head_norm_offset_bf16_bits


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


def _write_config(path, *, quant_method: str = "paroquant", layer_types: list[str] | None = None) -> None:
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
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
                "vocab_size": 16,
                "rms_norm_eps": 1.0e-6,
                "layer_types": layer_types or ["full_attention"],
                "quantization_config": {"quant_method": quant_method},
            }
        ),
        encoding="utf-8",
    )


def _write_dense_config(path, *, layer_types: list[str] | None = None) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "intermediate_size": 3,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5},
                "vocab_size": 16,
                "rms_norm_eps": 1.0e-6,
                "layer_types": layer_types or ["linear_attention"],
                "quantization_config": {"quant_method": "paroquant"},
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
    tensors[f"{base}.theta"] = np.zeros((1, 2), dtype=np.float16)
    tensors[f"{base}.pairs"] = np.zeros((1, 4), dtype=np.int16)
    tensors[f"{base}.channel_scales"] = np.zeros((4,), dtype=np.float16)
    return tensors


def _valid_linear_attention_tensors() -> dict[str, np.ndarray]:
    prefix = "model.layers.0.linear_attn"
    tensors: dict[str, np.ndarray] = {"model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float16)}
    for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
        base = f"{prefix}.{proj}"
        tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
        tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
        tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
        tensors[f"{base}.theta"] = np.zeros((1, 2), dtype=np.float16)
        tensors[f"{base}.pairs"] = np.zeros((1, 4), dtype=np.int16)
        tensors[f"{base}.channel_scales"] = np.zeros((4,), dtype=np.float16)
    tensors[f"{prefix}.in_proj_a.weight"] = np.zeros((1, 4), dtype=np.float16)
    tensors[f"{prefix}.in_proj_b.weight"] = np.zeros((1, 4), dtype=np.float16)
    tensors[f"{prefix}.conv1d.weight"] = np.zeros((8, 1, 4), dtype=np.float16)
    tensors[f"{prefix}.A_log"] = np.zeros((1,), dtype=np.float16)
    tensors[f"{prefix}.dt_bias"] = np.zeros((1,), dtype=np.float16)
    tensors[f"{prefix}.norm.weight"] = np.zeros((4,), dtype=np.float16)
    return tensors


def _valid_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float16),
        "model.layers.0.mlp.gate.weight": np.zeros((2, 4), dtype=np.float16),
        "model.layers.0.mlp.shared_expert_gate.weight": np.zeros((1, 4), dtype=np.float16),
        "model.layers.0.mlp.experts.gate_up_weight_theta": np.zeros((1, 2), dtype=np.float16),
        "model.layers.0.mlp.experts.gate_up_weight_pairs": np.zeros((1, 4), dtype=np.int16),
        "model.layers.0.mlp.experts.gate_up_weight_channel_scales": np.zeros((4,), dtype=np.float16),
        "model.layers.0.mlp.experts.down_weight_theta": np.zeros((1, 2), dtype=np.float16),
        "model.layers.0.mlp.experts.down_weight_pairs": np.zeros((1, 4), dtype=np.int16),
        "model.layers.0.mlp.experts.down_weight_channel_scales": np.zeros((4,), dtype=np.float16),
    }
    # Packed shared-expert PARO tensors (one independent rotation per projection,
    # mirrors the dense-attention layout). hidden_size=4, shared_int=3, group_size=4.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.0.mlp.shared_expert.{proj}"
        tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
        tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
        tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
        tensors[f"{base}.theta"] = np.zeros((1, 2), dtype=np.float16)
        tensors[f"{base}.pairs"] = np.zeros((1, 4), dtype=np.int16)
        tensors[f"{base}.channel_scales"] = np.zeros((4,), dtype=np.float16)
    for expert in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.0.mlp.experts.{expert}.{proj}"
            tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
            tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
            tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
    return tensors


def _valid_dense_mlp_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float16),
    }
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.0.mlp.{proj}"
        tensors[f"{base}.qweight"] = np.zeros((4, 1), dtype=np.int32)
        tensors[f"{base}.qzeros"] = np.zeros((1, 1), dtype=np.int32)
        tensors[f"{base}.scales"] = np.zeros((1, 8), dtype=np.float16)
        tensors[f"{base}.theta"] = np.zeros((1, 2), dtype=np.float16)
        tensors[f"{base}.pairs"] = np.zeros((1, 4), dtype=np.int16)
        tensors[f"{base}.channel_scales"] = np.zeros((4,), dtype=np.float16)
    return tensors


def _legacy_shared_expert_tensors() -> dict[str, np.ndarray]:
    tensors = _valid_tensors()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.0.mlp.shared_expert.{proj}"
        for suffix in ("qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"):
            tensors.pop(f"{base}.{suffix}")
    tensors["model.layers.0.mlp.shared_expert.gate_proj.weight"] = np.arange(12, dtype=np.float16).reshape(3, 4)
    tensors["model.layers.0.mlp.shared_expert.up_proj.weight"] = (np.arange(12, dtype=np.float16).reshape(3, 4) + 20)
    tensors["model.layers.0.mlp.shared_expert.down_proj.weight"] = (np.arange(12, dtype=np.float16).reshape(4, 3) + 40)
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
    assert config.rotary_dim == 4
    assert config.linear_num_key_heads == 0
    assert config.rms_norm_eps == 1.0e-6
    assert config.quant_method == "paroquant"
    assert normalize_qwen35_weight_name("model.layers.0.mlp.gate.weight") == "layers.0.mlp.gate.weight"
    assert normalize_qwen35_weight_name("language_model.layers.0.x") == "layers.0.x"


def test_required_moe_c1_names_include_all_expert_triples_and_packed_shared_expert() -> None:
    names = required_moe_c1_tensor_names(layer_id=3, num_experts=2)

    assert "layers.3.mlp.gate.weight" in names
    assert "layers.3.mlp.experts.gate_up_weight_theta" in names
    assert "layers.3.mlp.experts.1.down_proj.scales" in names
    # Packed shared-expert PARO tensors are the default required-family because
    # that is the new compact format.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"):
            assert f"layers.3.mlp.shared_expert.{proj}.{suffix}" in names
        assert f"layers.3.mlp.shared_expert.{proj}.weight" not in names
    # 6 routed-expert qweight (2 experts × 3 projs) + 3 packed shared-expert qweight.
    assert sum(name.endswith(".qweight") for name in names) == 9

    legacy = required_moe_c1_tensor_names(layer_id=3, num_experts=2, shared_expert_format="legacy_fp16")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"layers.3.mlp.shared_expert.{proj}.weight" in legacy
        assert f"layers.3.mlp.shared_expert.{proj}.qweight" not in legacy


def test_required_full_attention_names_include_rotated_qkv_and_o_proj() -> None:
    names = required_full_attention_c1_tensor_names(layer_id=3)
    combined = required_full_attention_moe_c1_tensor_names(layer_id=3, num_experts=2)

    assert "layers.3.input_layernorm.weight" in names
    assert "layers.3.self_attn.q_norm.weight" in names
    assert "layers.3.self_attn.q_proj.theta" in names
    assert "layers.3.self_attn.v_proj.channel_scales" in names
    assert "layers.3.self_attn.o_proj.qweight" in names
    assert "layers.3.self_attn.o_proj.theta" in names
    assert "layers.3.mlp.experts.1.down_proj.scales" in combined


def test_required_linear_attention_names_include_rotated_projections_and_state() -> None:
    names = required_linear_attention_c1_tensor_names(layer_id=0)
    combined = required_linear_attention_moe_c1_tensor_names(layer_id=0, num_experts=2)

    assert "layers.0.linear_attn.in_proj_qkv.theta" in names
    assert "layers.0.linear_attn.in_proj_z.channel_scales" in names
    assert "layers.0.linear_attn.out_proj.qweight" in names
    assert "layers.0.linear_attn.in_proj_a.weight" in names
    assert "layers.0.linear_attn.conv1d.weight" in names
    assert "layers.0.linear_attn.A_log" in names
    assert "layers.0.mlp.experts.1.up_proj.qweight" in combined


def test_runtime_dense_attention_names_include_dense_mlp_sidecars() -> None:
    linear = runtime_linear_attention_dense_c1_tensor_names(layer_id=0)
    full = runtime_full_attention_dense_c1_tensor_names(layer_id=3)

    assert "layers.0.linear_attn.in_proj_qkv.qweight" in linear
    assert "layers.0.mlp.gate_proj.theta" in linear
    assert "layers.0.mlp.up_proj.qweight" in linear
    assert "layers.3.self_attn.q_proj.qweight" in full
    assert "layers.3.mlp.down_proj.channel_scales" in full
    assert all("shared_expert" not in name and ".experts." not in name for name in linear + full)


def test_validate_and_materialize_dense_linear_attention_runtime_layer(tmp_path) -> None:
    _write_dense_config(tmp_path, layer_types=["linear_attention"])
    tensors = {**_valid_linear_attention_tensors(), **_valid_dense_mlp_tensors()}
    tensors["model.layers.0.mlp.gate_proj.qweight"] = np.arange(4, dtype=np.int32).reshape(4, 1)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    validation = validate_qwen35_paro_linear_attention_dense_c1_layout(index)
    layer = materialize_qwen35_paro_linear_attention_dense_c1_runtime_layer(index, runtime=runtime)

    assert validation.passed
    assert validation.config.num_experts == 0
    assert layer.tensor("layers.0.mlp.gate_proj.qweight_pack8_decode").shape == (1, 4)
    assert layer.tensor("layers.0.mlp.gate_proj.qweight_pack8_decode").dtype is DType.INT32
    layer.free(runtime=runtime)


def test_validate_and_materialize_dense_full_attention_runtime_layer(tmp_path) -> None:
    _write_dense_config(tmp_path, layer_types=["full_attention"])
    tensors = {**_valid_attention_tensors(), **_valid_dense_mlp_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    validation = validate_qwen35_paro_full_attention_dense_c1_layout(index)
    layer = materialize_qwen35_paro_full_attention_dense_c1_runtime_layer(index, runtime=runtime)

    assert validation.passed
    assert layer.tensor("layers.0.self_attn.q_proj.qweight_pack8_decode").shape == (1, 4)
    assert layer.tensor("layers.0.mlp.down_proj.qweight_pack8_decode").shape == (1, 4)
    layer.free(runtime=runtime)


def test_validate_qwen35_paro_moe_c1_layout_passes(tmp_path) -> None:
    _write_config(tmp_path)
    save_file(_valid_tensors(), tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    result = validate_qwen35_paro_moe_c1_layout(index)

    assert result.passed
    assert result.config.num_experts == 2
    assert result.shared_expert_format == "packed_paro_w4"
    assert not result.missing
    assert not result.shape_errors


def test_validate_qwen35_paro_moe_c1_layout_passes_for_legacy_shared_expert(tmp_path) -> None:
    _write_config(tmp_path)
    save_file(_legacy_shared_expert_tensors(), tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    result = validate_qwen35_paro_moe_c1_layout(index)

    assert result.passed
    assert result.shared_expert_format == "legacy_fp16"
    assert not result.missing
    assert not result.shape_errors


def test_validate_qwen35_paro_moe_c1_layout_can_force_legacy_when_both_formats_exist(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = _valid_tensors()
    legacy = _legacy_shared_expert_tensors()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.0.mlp.shared_expert.{proj}.weight"
        tensors[key] = legacy[key]
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    auto = validate_qwen35_paro_moe_c1_layout(index)
    forced_legacy = validate_qwen35_paro_moe_c1_layout(index, shared_expert_format="legacy_fp16")
    forced_packed = validate_qwen35_paro_moe_c1_layout(index, shared_expert_format="packed_paro_w4")

    assert auto.passed
    assert auto.shared_expert_format == "packed_paro_w4"
    assert forced_legacy.passed
    assert forced_legacy.shared_expert_format == "legacy_fp16"
    assert forced_packed.passed
    assert forced_packed.shared_expert_format == "packed_paro_w4"


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
    # Distinct shared-expert qweight per projection so we can check the transpose.
    for offset, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
        base = f"model.layers.0.mlp.shared_expert.{proj}.qweight"
        tensors[base] = (np.arange(4, dtype=np.int32).reshape(4, 1) + offset * 100)
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
    # Packed shared-expert qweight_pack8_decode is the rank-2 transpose [N/8, K]
    # of the raw [K, N/8] qweight; raw qweight/qzeros/scales/theta/pairs/channel_scales
    # are kept in the checkpoint, not duplicated into the prepared map.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        decode = prepared[f"layers.0.mlp.shared_expert.{proj}.qweight_pack8_decode"]
        raw = tensors[f"model.layers.0.mlp.shared_expert.{proj}.qweight"]
        assert decode.dtype == np.int32
        assert decode.shape == (1, 4)
        np.testing.assert_array_equal(decode, np.ascontiguousarray(raw.T))
    for legacy in (
        "layers.0.mlp.shared_expert.gate_up_weight_w8a16",
        "layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale",
        "layers.0.mlp.shared_expert.down_weight_w8a16",
        "layers.0.mlp.shared_expert.down_weight_w8a16_scale",
    ):
        assert legacy not in prepared


def test_prepare_qwen35_paro_moe_c1_host_tensors_supports_legacy_shared_expert_w8a16(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_legacy_shared_expert_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    prepared = prepare_qwen35_paro_moe_c1_host_tensors(index)

    assert set(prepared_moe_c1_tensor_names(layer_id=0, shared_expert_format="legacy_fp16")) == set(prepared)
    gate_up = prepared["layers.0.mlp.shared_expert.gate_up_weight_w8a16"]
    gate_up_scale = prepared["layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale"]
    down = prepared["layers.0.mlp.shared_expert.down_weight_w8a16"]
    down_scale = prepared["layers.0.mlp.shared_expert.down_weight_w8a16_scale"]
    assert gate_up.dtype == np.int8
    assert gate_up.shape == (6, 4)
    assert gate_up_scale.dtype == np.float32
    assert gate_up_scale.shape == (6,)
    assert down.dtype == np.int8
    assert down.shape == (4, 3)
    assert down_scale.dtype == np.float32
    assert down_scale.shape == (4,)
    assert "layers.0.mlp.shared_expert.gate_proj.qweight_pack8_decode" not in prepared


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
    for proj in ("gate_proj", "up_proj", "down_proj"):
        decode = layer.tensor(f"layers.0.mlp.shared_expert.{proj}.qweight_pack8_decode")
        assert decode.dtype is DType.INT32
        assert decode.shape == (1, 4)
    expected_count = len(required_full_attention_moe_c1_tensor_names(layer_id=0, num_experts=2)) + len(
        prepared_moe_c1_tensor_names(layer_id=0)
    )
    layer.free(runtime=runtime)
    assert len(runtime.freed) == expected_count


def test_prepare_qwen35_paro_moe_c1_runtime_host_tensors_uses_parent_mixed_dtypes(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_valid_tensors()}
    tensors["model.layers.0.mlp.gate.weight"] = np.asarray(
        [[1.0, -2.5, 0.5, 0.0], [3.0, 4.0, -1.0, -0.25]], dtype=np.float16
    )
    tensors["model.layers.0.mlp.shared_expert_gate.weight"] = np.asarray([[2.0, 0.25, -0.5, 1.5]], dtype=np.float16)
    for expert in range(2):
        tensors[f"model.layers.0.mlp.experts.{expert}.gate_proj.scales"] = np.full((1, 8), 1.0 + expert, dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    prepared = prepare_qwen35_paro_moe_c1_runtime_host_tensors(index)

    router = prepared["layers.0.mlp.router_shared_gate.weight"]
    expected_router = np.concatenate(
        (tensors["model.layers.0.mlp.gate.weight"], tensors["model.layers.0.mlp.shared_expert_gate.weight"]),
        axis=0,
    )
    assert router.dtype == np.uint16
    np.testing.assert_array_equal(router, float_array_to_bf16_bits(expected_router))
    scales = prepared["layers.0.mlp.experts.stacked_gate_scales"]
    assert scales.dtype == np.float16
    assert prepared["layers.0.mlp.experts.stacked_gate_qweight_pack8_decode"].dtype == np.int32
    # Packed shared-expert qweight_pack8_decode lands as raw int32 in the
    # runtime-prepared map (no W8A16 quantization).
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert prepared[f"layers.0.mlp.shared_expert.{proj}.qweight_pack8_decode"].dtype == np.int32
    assert "layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale" not in prepared


def test_prepare_qwen35_paro_moe_c1_runtime_host_tensors_supports_legacy_shared_expert(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_legacy_shared_expert_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    prepared = prepare_qwen35_paro_moe_c1_runtime_host_tensors(index)

    assert set(runtime_prepared_moe_c1_tensor_names(layer_id=0, shared_expert_format="legacy_fp16")) == set(prepared)
    assert prepared["layers.0.mlp.shared_expert.gate_up_weight_w8a16"].dtype == np.int8
    assert prepared["layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale"].dtype == np.float32
    assert prepared["layers.0.mlp.shared_expert.down_weight_w8a16"].dtype == np.int8
    assert prepared["layers.0.mlp.shared_expert.down_weight_w8a16_scale"].dtype == np.float32
    assert "layers.0.mlp.shared_expert.gate_proj.qweight_pack8_decode" not in prepared


def test_materialize_qwen35_paro_full_attention_moe_c1_runtime_layer_uses_parent_mixed_dtypes(tmp_path) -> None:
    _write_config(tmp_path)
    tensors = {**_valid_attention_tensors(), **_valid_tensors()}
    tensors["model.layers.0.self_attn.q_proj.scales"] = np.full((1, 8), 1.0, dtype=np.float16)
    tensors["model.layers.0.self_attn.q_proj.theta"] = np.full((1, 2), 0.5, dtype=np.float16)
    tensors["model.layers.0.self_attn.q_norm.weight"] = np.asarray([0.01, 0.10], dtype=np.float16)
    tensors["model.layers.0.self_attn.k_norm.weight"] = np.asarray([0.02, 0.03], dtype=np.float16)
    tensors["model.layers.0.mlp.gate.weight"] = np.ones((2, 4), dtype=np.float16)
    tensors["model.layers.0.mlp.shared_expert_gate.weight"] = np.full((1, 4), 2.0, dtype=np.float16)
    tensors["model.layers.0.mlp.experts.down_weight_theta"] = np.full((1, 2), -1.0, dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    layer = materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(index, runtime=runtime)

    names = set(layer.weights.tensors)
    expected_names = set(runtime_full_attention_moe_c1_tensor_names(layer_id=0))
    expected_names.update(
        {
            "layers.0.self_attn.q_proj.qweight_pack8_decode",
            "layers.0.self_attn.k_proj.qweight_pack8_decode",
        }
    )
    assert names == expected_names
    assert "layers.0.mlp.experts.0.gate_proj.qweight" not in names
    assert layer.tensor("layers.0.self_attn.q_proj.scales").dtype is DType.FP16
    assert layer.tensor("layers.0.self_attn.q_proj.theta").dtype is DType.FP16
    assert layer.tensor("layers.0.self_attn.q_proj.qweight").dtype is DType.INT32
    assert layer.tensor("layers.0.mlp.router_shared_gate.weight").dtype is DType.BF16
    assert layer.tensor("layers.0.mlp.experts.stacked_gate_scales").dtype is DType.FP16
    assert layer.tensor("layers.0.mlp.experts.down_weight_theta").dtype is DType.FP16
    assert layer.tensor("layers.0.mlp.shared_expert.gate_proj.pairs").dtype is DType.INT16
    assert layer.tensor("layers.0.mlp.shared_expert.gate_proj.theta").dtype is DType.FP16
    assert layer.tensor("layers.0.mlp.shared_expert.gate_proj.channel_scales").dtype is DType.FP16
    assert layer.tensor("layers.0.mlp.shared_expert.gate_proj.qzeros").dtype is DType.INT32
    assert layer.tensor("layers.0.mlp.shared_expert.gate_proj.scales").dtype is DType.FP16
    assert "layers.0.mlp.shared_expert.gate_proj.qweight" not in names
    q_scales = layer.allocation("layers.0.self_attn.q_proj.scales")
    assert bytes(runtime.buffers[q_scales.buffer.ptr]) == tensors[
        "model.layers.0.self_attn.q_proj.scales"
    ].astype(np.float16).tobytes()
    q_norm = layer.allocation("layers.0.self_attn.q_norm.weight")
    assert bytes(runtime.buffers[q_norm.buffer.ptr]) == _qwen_head_norm_offset_bf16_bits(
        tensors["model.layers.0.self_attn.q_norm.weight"]
    ).tobytes()
    input_norm = layer.allocation("layers.0.input_layernorm.weight")
    assert layer.tensor("layers.0.input_layernorm.weight").dtype is DType.FP16
    assert bytes(runtime.buffers[input_norm.buffer.ptr]) == (
        tensors["model.layers.0.input_layernorm.weight"].astype(np.float32) + np.float32(1.0)
    ).astype(np.float16).tobytes()
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(expected_names)


def test_materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer_uses_state_dtypes(tmp_path) -> None:
    _write_config(tmp_path, layer_types=["linear_attention"])
    tensors = {**_valid_linear_attention_tensors(), **_valid_tensors()}
    tensors["model.layers.0.linear_attn.conv1d.weight"] = np.full((8, 1, 4), 0.5, dtype=np.float16)
    tensors["model.layers.0.linear_attn.A_log"] = np.full((1,), -2.0, dtype=np.float16)
    tensors["model.layers.0.linear_attn.dt_bias"] = np.full((1,), 1.0, dtype=np.float16)
    tensors["model.layers.0.linear_attn.norm.weight"] = np.full((4,), 0.25, dtype=np.float16)
    tensors["model.layers.0.linear_attn.in_proj_a.weight"] = np.full((1, 4), 2.0, dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    validation = validate_qwen35_paro_linear_attention_moe_c1_layout(index)
    layer = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(index, runtime=runtime)

    assert validation.passed
    expected_names = set(runtime_linear_attention_moe_c1_tensor_names(layer_id=0))
    expected_names.update(
        {
            "layers.0.linear_attn.in_proj_qkv.qweight_pack8_decode",
            "layers.0.linear_attn.in_proj_z.qweight_pack8_decode",
        }
    )
    assert set(layer.weights.tensors) == expected_names
    assert layer.config.linear_num_key_heads == 1
    assert layer.config.linear_value_head_dim == 4
    assert layer.tensor("layers.0.linear_attn.conv1d.weight").dtype is DType.FP32
    assert layer.tensor("layers.0.linear_attn.A_log").dtype is DType.FP32
    assert layer.tensor("layers.0.linear_attn.dt_bias").dtype is DType.FP32
    assert layer.tensor("layers.0.linear_attn.norm.weight").dtype is DType.FP32
    assert layer.tensor("layers.0.linear_attn.in_proj_a.weight").dtype is DType.FP16
    assert layer.tensor("layers.0.linear_attn.in_proj_qkv.scales").dtype is DType.FP16
    assert layer.tensor("layers.0.mlp.router_shared_gate.weight").dtype is DType.BF16
    assert layer.tensor("layers.0.mlp.shared_expert.down_proj.pairs").dtype is DType.INT16
    assert layer.tensor("layers.0.mlp.shared_expert.down_proj.theta").dtype is DType.FP16
    assert "layers.0.mlp.shared_expert.down_proj.qweight" not in layer.weights.tensors
    conv = layer.allocation("layers.0.linear_attn.conv1d.weight")
    assert bytes(runtime.buffers[conv.buffer.ptr]) == tensors["model.layers.0.linear_attn.conv1d.weight"].astype(np.float32).tobytes()
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(expected_names)


def test_materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer_supports_legacy_shared_expert(tmp_path) -> None:
    _write_config(tmp_path, layer_types=["linear_attention"])
    tensors = {**_valid_linear_attention_tensors(), **_legacy_shared_expert_tensors()}
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    validation = validate_qwen35_paro_linear_attention_moe_c1_layout(index)
    layer = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(index, runtime=runtime)

    assert validation.passed
    assert validation.shared_expert_format == "legacy_fp16"
    expected_names = set(runtime_linear_attention_moe_c1_tensor_names(layer_id=0, shared_expert_format="legacy_fp16"))
    expected_names.update(
        {
            "layers.0.linear_attn.in_proj_qkv.qweight_pack8_decode",
            "layers.0.linear_attn.in_proj_z.qweight_pack8_decode",
        }
    )
    assert set(layer.weights.tensors) == expected_names
    assert layer.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16").dtype is DType.INT8
    assert layer.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16_scale").dtype is DType.FP32
    assert layer.tensor("layers.0.mlp.shared_expert.down_weight_w8a16").dtype is DType.INT8
    assert layer.tensor("layers.0.mlp.shared_expert.down_weight_w8a16_scale").dtype is DType.FP32
    assert "layers.0.mlp.shared_expert.gate_proj.qweight_pack8_decode" not in layer.weights.tensors
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(expected_names)


def test_materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer_can_force_legacy_shared_expert(tmp_path) -> None:
    _write_config(tmp_path, layer_types=["linear_attention"])
    tensors = {**_valid_linear_attention_tensors(), **_valid_tensors()}
    legacy = _legacy_shared_expert_tensors()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.0.mlp.shared_expert.{proj}.weight"
        tensors[key] = legacy[key]
    save_file(tensors, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    auto = validate_qwen35_paro_linear_attention_moe_c1_layout(index)
    forced = validate_qwen35_paro_linear_attention_moe_c1_layout(index, shared_expert_format="legacy_fp16")
    layer = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
        index,
        runtime=runtime,
        shared_expert_format="legacy_fp16",
    )

    assert auto.passed
    assert auto.shared_expert_format == "packed_paro_w4"
    assert forced.passed
    assert forced.shared_expert_format == "legacy_fp16"
    expected_names = set(runtime_linear_attention_moe_c1_tensor_names(layer_id=0, shared_expert_format="legacy_fp16"))
    expected_names.update(
        {
            "layers.0.linear_attn.in_proj_qkv.qweight_pack8_decode",
            "layers.0.linear_attn.in_proj_z.qweight_pack8_decode",
        }
    )
    assert set(layer.weights.tensors) == expected_names
    assert layer.tensor("layers.0.mlp.shared_expert.gate_up_weight_w8a16").dtype is DType.INT8
    assert "layers.0.mlp.shared_expert.gate_proj.qweight_pack8_decode" not in layer.weights.tensors
    layer.free(runtime=runtime)
    assert len(runtime.freed) == len(expected_names)


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
