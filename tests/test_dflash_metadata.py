from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from hipengine.loading.dflash import (
    DFLASH_DRAFTER_MODEL,
    DFLASH_PACKED_TARGET_MODEL,
    dflash_draft_config_from_hf,
    dflash_drafter_runtime_tensor_names,
    validate_dflash_drafter_metadata,
    validate_dflash_target_metadata,
)
from hipengine.loading.qwen35_paro import (
    runtime_full_attention_dense_c1_tensor_names,
    runtime_linear_attention_dense_c1_tensor_names,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index, read_tensor_storage_bytes


LOCAL_TARGET = Path(
    "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)
LOCAL_DRAFTER = Path(
    "/models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/"
    "snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719"
)


def test_dflash_draft_config_captures_required_fields() -> None:
    cfg = dflash_draft_config_from_hf(_draft_config())

    assert cfg.architecture == "DFlashDraftModel"
    assert cfg.block_size == 4
    assert cfg.mask_token_id == 99
    assert cfg.target_layer_ids == (1, 3)
    assert cfg.target_hidden_concat_size == 32
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 4
    assert cfg.rope_theta == 10_000_000.0
    assert cfg.vocab_size == 100


def test_dflash_drafter_metadata_validation_reports_missing_dtype_and_shape() -> None:
    index = _draft_index()
    tensors = dict(index.tensors)
    tensors.pop("norm.weight")
    tensors["fc.weight"] = _tensor("fc.weight", "BF16", (16, 31))
    tensors["layers.0.self_attn.q_proj.weight"] = _tensor("layers.0.self_attn.q_proj.weight", "F16", (16, 16))
    bad = WeightIndex(index.model_path, index.config, tensors, index.shards)

    result = validate_dflash_drafter_metadata(bad)

    assert result.passed is False
    assert "norm.weight" in result.missing
    assert any("q_proj" in err and "expected dtype BF16" in err for err in result.dtype_errors)
    assert any("fc.weight" in err and "expected shape (16, 32)" in err for err in result.shape_errors)
    with pytest.raises(KeyError, match="missing tensors: norm.weight"):
        result.raise_for_errors()


def test_dflash_drafter_metadata_validation_passes_fake_manifest() -> None:
    result = validate_dflash_drafter_metadata(_draft_index())

    assert result.passed is True
    assert result.config.target_layer_ids == (1, 3)
    assert result.config.hidden_size == 16
    assert result.config.num_hidden_layers == 2
    assert len(result.present) == 25


def test_dflash_drafter_runtime_names_support_layer_limit() -> None:
    config = dflash_draft_config_from_hf(_draft_config())

    names = dflash_drafter_runtime_tensor_names(config, layer_limit=1)

    assert names[:2] == ("fc.weight", "hidden_norm.weight")
    assert "layers.0.self_attn.q_proj.weight" in names
    assert "layers.1.self_attn.q_proj.weight" not in names
    assert names[-1] == "norm.weight"
    with pytest.raises(ValueError, match="layer_limit"):
        dflash_drafter_runtime_tensor_names(config, layer_limit=3)


def test_read_tensor_storage_bytes_handles_bf16_payload(tmp_path: Path) -> None:
    payload = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    header = {"x": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    info = TensorInfo(name="x", shard_path=path, dtype="BF16", shape=(2, 2))

    assert read_tensor_storage_bytes(info) == payload


def test_dflash_target_metadata_validation_passes_fake_packed_manifest() -> None:
    result = validate_dflash_target_metadata(_target_index())

    assert result.passed is True
    assert result.config.shared_expert_format == "packed_paro_w4"
    assert result.config.quant_method == "paroquant"
    assert result.config.num_hidden_layers == 4
    assert len(result.present) == 2 + 4 * 3 * 6


def test_dflash_target_metadata_validation_accepts_dense_paro_manifest() -> None:
    result = validate_dflash_target_metadata(_dense_target_index())

    assert result.passed is True
    assert result.config.shared_expert_format == "dense_paro_w4"
    assert result.config.num_experts == 0
    assert "layers.0.mlp.gate_proj.qweight" in result.present
    assert "layers.3.self_attn.q_proj.qweight" in result.present


def test_dflash_defaults_do_not_reference_old_quark_mtp_artifact() -> None:
    assert DFLASH_PACKED_TARGET_MODEL == "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed"
    assert DFLASH_DRAFTER_MODEL == "z-lab/Qwen3.6-35B-A3B-DFlash"
    script = Path("scripts/dflash_speculative_bench.py").read_text()
    assert "Qwen3.6-35B-A3B-Quark-W8A8-INT8-MTP-BF16" not in script


@pytest.mark.skipif(not (LOCAL_TARGET.exists() and LOCAL_DRAFTER.exists()), reason="local DFlash artifacts not cached")
def test_local_cached_dflash_artifact_metadata_offline() -> None:
    target = validate_dflash_target_metadata(load_weight_index(LOCAL_TARGET))
    drafter = validate_dflash_drafter_metadata(load_weight_index(LOCAL_DRAFTER), target_config=target.config)

    assert target.passed, target.to_json_dict()
    assert drafter.passed, drafter.to_json_dict()
    assert drafter.config.target_layer_ids == (1, 10, 19, 28, 37)
    assert drafter.config.block_size == 16
    assert drafter.config.mask_token_id == 248070
    assert drafter.config.target_hidden_concat_size == 5 * target.config.hidden_size
    assert drafter.config.vocab_size == target.config.vocab_size


def _draft_config() -> dict:
    return {
        "architectures": ["DFlashDraftModel"],
        "block_size": 4,
        "dflash_config": {"mask_token_id": 99, "target_layer_ids": [1, 3]},
        "dtype": "bfloat16",
        "head_dim": 4,
        "hidden_size": 16,
        "intermediate_size": 32,
        "layer_types": ["full_attention", "full_attention"],
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_target_layers": 4,
        "rope_theta": 10_000_000.0,
        "vocab_size": 100,
    }


def _target_config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "vocab_size": 100,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
        },
        "quantization_config": {"quant_method": "paroquant", "bits": 4},
    }


def _dense_target_config() -> dict:
    cfg = _target_config()
    cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    cfg["model_type"] = "qwen3_5"
    removed_dense_fields = {
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    }
    cfg["text_config"] = {
        key: value
        for key, value in cfg["text_config"].items()
        if key not in removed_dense_fields
    }
    cfg["text_config"]["model_type"] = "qwen3_5_text"
    cfg["text_config"]["intermediate_size"] = 32
    return cfg


def _draft_index() -> WeightIndex:
    tensors = {
        "fc.weight": _tensor("fc.weight", "BF16", (16, 32)),
        "hidden_norm.weight": _tensor("hidden_norm.weight", "BF16", (16,)),
        "norm.weight": _tensor("norm.weight", "BF16", (16,)),
    }
    for layer in range(2):
        prefix = f"layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor(f"{prefix}.input_layernorm.weight", "BF16", (16,)),
                f"{prefix}.post_attention_layernorm.weight": _tensor(f"{prefix}.post_attention_layernorm.weight", "BF16", (16,)),
                f"{prefix}.self_attn.q_proj.weight": _tensor(f"{prefix}.self_attn.q_proj.weight", "BF16", (16, 16)),
                f"{prefix}.self_attn.k_proj.weight": _tensor(f"{prefix}.self_attn.k_proj.weight", "BF16", (8, 16)),
                f"{prefix}.self_attn.v_proj.weight": _tensor(f"{prefix}.self_attn.v_proj.weight", "BF16", (8, 16)),
                f"{prefix}.self_attn.o_proj.weight": _tensor(f"{prefix}.self_attn.o_proj.weight", "BF16", (16, 16)),
                f"{prefix}.self_attn.q_norm.weight": _tensor(f"{prefix}.self_attn.q_norm.weight", "BF16", (4,)),
                f"{prefix}.self_attn.k_norm.weight": _tensor(f"{prefix}.self_attn.k_norm.weight", "BF16", (4,)),
                f"{prefix}.mlp.gate_proj.weight": _tensor(f"{prefix}.mlp.gate_proj.weight", "BF16", (32, 16)),
                f"{prefix}.mlp.up_proj.weight": _tensor(f"{prefix}.mlp.up_proj.weight", "BF16", (32, 16)),
                f"{prefix}.mlp.down_proj.weight": _tensor(f"{prefix}.mlp.down_proj.weight", "BF16", (16, 32)),
            }
        )
    return WeightIndex(Path("/fake/drafter"), _draft_config(), tensors, (Path("fake.safetensors"),))


def _target_index() -> WeightIndex:
    tensors = {
        "model.language_model.embed_tokens.weight": _tensor("model.language_model.embed_tokens.weight", "F16", (100, 16)),
        "lm_head.weight": _tensor("lm_head.weight", "F16", (100, 16)),
    }
    for layer in range(4):
        shared = f"model.language_model.layers.{layer}.mlp.shared_expert"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{shared}.{proj}"
            tensors.update(
                {
                    f"{base}.qweight": _tensor(f"{base}.qweight", "I32", (16, 4)),
                    f"{base}.qzeros": _tensor(f"{base}.qzeros", "I32", (1, 4)),
                    f"{base}.scales": _tensor(f"{base}.scales", "F16", (1, 16)),
                    f"{base}.theta": _tensor(f"{base}.theta", "F16", (8, 16)),
                    f"{base}.pairs": _tensor(f"{base}.pairs", "I16", (8, 16)),
                    f"{base}.channel_scales": _tensor(f"{base}.channel_scales", "F16", (1, 16)),
                }
            )
    return WeightIndex(Path("/fake/target"), _target_config(), tensors, (Path("fake.safetensors"),))


def _dense_target_index() -> WeightIndex:
    tensors = {
        "model.language_model.embed_tokens.weight": _tensor("model.language_model.embed_tokens.weight", "F16", (100, 16)),
        "lm_head.weight": _tensor("lm_head.weight", "F16", (100, 16)),
    }
    layer_types = _dense_target_config()["text_config"]["layer_types"]
    for layer, layer_type in enumerate(layer_types):
        names = (
            runtime_full_attention_dense_c1_tensor_names(layer_id=layer)
            if layer_type == "full_attention"
            else runtime_linear_attention_dense_c1_tensor_names(layer_id=layer)
        )
        for name in names:
            if name.endswith(".qweight") or name.endswith(".qzeros"):
                tensors[f"model.language_model.{name}"] = _tensor(f"model.language_model.{name}", "I32", (1, 1))
            elif name.endswith(".pairs"):
                tensors[f"model.language_model.{name}"] = _tensor(f"model.language_model.{name}", "I16", (1, 1))
            else:
                tensors[f"model.language_model.{name}"] = _tensor(f"model.language_model.{name}", "F16", (1,))
    return WeightIndex(Path("/fake/dense-target"), _dense_target_config(), tensors, (Path("fake.safetensors"),))


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, shard_path=Path("fake.safetensors"), dtype=dtype, shape=shape)
