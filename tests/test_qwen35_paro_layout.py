from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.loading import (
    MissingTensorError,
    load_weight_index,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
    required_moe_c1_tensor_names,
    validate_qwen35_paro_moe_c1_layout,
)


def _write_config(path, *, quant_method: str = "paroquant") -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 1,
                "hidden_size": 4,
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
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 8,
            "quantization_config": {"quant_method": "paroquant"},
        }
    )

    assert config.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert config.layer_types == ("full_attention", "full_attention")
    assert config.quant_method == "paroquant"
    assert normalize_qwen35_weight_name("model.layers.0.mlp.gate.weight") == "layers.0.mlp.gate.weight"
    assert normalize_qwen35_weight_name("language_model.layers.0.x") == "layers.0.x"


def test_required_moe_c1_names_include_all_expert_triples() -> None:
    names = required_moe_c1_tensor_names(layer_id=3, num_experts=2)

    assert "layers.3.mlp.gate.weight" in names
    assert "layers.3.mlp.experts.gate_up_weight_theta" in names
    assert "layers.3.mlp.experts.1.down_proj.scales" in names
    assert sum(name.endswith(".qweight") for name in names) == 6


def test_validate_qwen35_paro_moe_c1_layout_passes(tmp_path) -> None:
    _write_config(tmp_path)
    save_file(_valid_tensors(), tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    result = validate_qwen35_paro_moe_c1_layout(index)

    assert result.passed
    assert result.config.num_experts == 2
    assert not result.missing
    assert not result.shape_errors


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
