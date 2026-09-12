from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.loading.dflash import (
    DFlashDrafterDeviceWeights,
    dflash_draft_config_from_hf,
    dflash_drafter_runtime_tensor_names,
    validate_dflash_drafter_metadata,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index


LOCAL_LAGUNA_DRAFTER = Path(
    "/home/lhl/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash/"
    "snapshots/b0486d1586daa0d56435c508108171fc1c8daff9"
)


def test_laguna_dflash_config_normalizes_nested_schema_and_capture_depths() -> None:
    config = dflash_draft_config_from_hf(_laguna_config())

    assert config.architecture == "DFlashLagunaForCausalLM"
    assert config.decoder_arch == "laguna"
    assert config.block_size == 16
    assert config.mask_token_id == 12
    assert config.num_target_layers == 48
    assert config.target_layer_ids == (1, 10, 19, 29, 38, 47)
    assert config.target_capture_depths == (2, 11, 20, 30, 39, 48)
    assert config.target_hidden_concat_size == 6 * 3072
    assert config.rms_norm_eps == pytest.approx(1.0e-6)
    assert config.max_position_embeddings == 1_048_576
    assert config.sliding_windows == (512,) * 6
    assert config.attention_gate_type == "per_head"
    assert config.qkv_layout == "fused_qkv"
    assert config.aux_hidden_norm_count == 6
    assert config.causal is True
    assert config.dtype == "bfloat16"


def test_laguna_dflash_metadata_accepts_exact_69_tensor_contract() -> None:
    result = validate_dflash_drafter_metadata(_laguna_index())

    assert result.passed, result.to_json_dict()
    assert len(result.present) == 69
    assert result.config.architecture == "DFlashLagunaForCausalLM"
    names = dflash_drafter_runtime_tensor_names(result.config)
    assert len(names) == 69
    assert names[:3] == (
        "aux_hidden_norms.0.weight",
        "aux_hidden_norms.1.weight",
        "aux_hidden_norms.2.weight",
    )
    assert "layers.0.self_attn.qkv_proj.weight" in names
    assert "layers.0.self_attn.g_proj.weight" in names
    assert "layers.0.self_attn.q_proj.weight" not in names


def test_laguna_dflash_metadata_rejects_off_by_one_capture_and_missing_gate() -> None:
    index = _laguna_index()
    config = dict(index.config)
    config["eagle_aux_hidden_state_layer_ids"] = [1, 10, 19, 29, 38, 47]
    tensors = dict(index.tensors)
    tensors.pop("layers.2.self_attn.g_proj.weight")

    result = validate_dflash_drafter_metadata(
        WeightIndex(index.model_path, config, tensors, index.shards)
    )

    assert result.passed is False
    assert "layers.2.self_attn.g_proj.weight" in result.missing
    assert any("capture depths" in error for error in result.config_errors)


def test_laguna_fused_qkv_exposes_zero_copy_row_views() -> None:
    config = dflash_draft_config_from_hf(_laguna_config())
    qkv = Tensor.from_handle(
        0x1000,
        (config.qkv_features, config.hidden_size),
        "bf16",
        Device("hip", 0),
    )
    weights = DFlashDrafterDeviceWeights(
        config=config,
        weights=_FakeWeightMap({"layers.0.self_attn.qkv_proj.weight": qkv}),  # type: ignore[arg-type]
        layer_limit=1,
    )

    q = weights.tensor("layers.0.self_attn.q_proj.weight")
    k = weights.tensor("layers.0.self_attn.k_proj.weight")
    v = weights.tensor("layers.0.self_attn.v_proj.weight")

    row_bytes = config.hidden_size * 2
    assert q.ptr == qkv.ptr
    assert q.shape == (config.q_features, config.hidden_size)
    assert k.ptr == qkv.ptr + config.q_features * row_bytes
    assert k.shape == (config.kv_features, config.hidden_size)
    assert v.ptr == qkv.ptr + (config.q_features + config.kv_features) * row_bytes
    assert v.shape == (config.kv_features, config.hidden_size)


def test_unknown_dflash_architecture_fails_closed() -> None:
    config = _laguna_config()
    config["architectures"] = ["UnknownDFlashForCausalLM"]

    with pytest.raises(ValueError, match="unregistered DFlash architecture"):
        dflash_draft_config_from_hf(config)


@pytest.mark.skipif(not LOCAL_LAGUNA_DRAFTER.exists(), reason="local Poolside Laguna DFlash artifact not cached")
def test_local_poolside_laguna_dflash_metadata_offline() -> None:
    index = load_weight_index(LOCAL_LAGUNA_DRAFTER)
    result = validate_dflash_drafter_metadata(index)

    assert result.passed, result.to_json_dict()
    assert len(index.tensors) == 69
    assert sum(tensor.nbytes or 0 for tensor in index.tensors.values()) == 2_229_955_584
    assert result.config.target_capture_depths == (2, 11, 20, 30, 39, 48)


class _FakeWeightMap:
    def __init__(self, tensors: dict[str, Tensor]) -> None:
        self.tensors = tensors

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def free(self, *, runtime=None) -> None:
        del runtime


def _laguna_config() -> dict:
    return {
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "max_position_embeddings": 1048576,
        "model_type": "laguna",
        "num_attention_heads": 72,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "sliding_window": 512,
        "vocab_size": 100352,
        "layer_types": ["sliding_attention"] * 6,
        "rope_theta": 500000.0,
        "gating": "per-head",
        "architectures": ["DFlashLagunaForCausalLM"],
        "num_experts": 0,
        "sliding_windows": [512] * 6,
        "draft_vocab_size": 100352,
        "torch_dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [2, 11, 20, 30, 39, 48],
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 48,
            "target_layer_ids": [1, 10, 19, 29, 38, 47],
            "causal": True,
        },
    }


def _laguna_index() -> WeightIndex:
    config = dflash_draft_config_from_hf(_laguna_config())
    tensors: dict[str, TensorInfo] = {
        f"aux_hidden_norms.{index}.weight": _tensor(
            f"aux_hidden_norms.{index}.weight", (config.hidden_size,)
        )
        for index in range(config.aux_hidden_norm_count)
    }
    tensors.update(
        {
            "fc.weight": _tensor(
                "fc.weight", (config.hidden_size, config.target_hidden_concat_size)
            ),
            "hidden_norm.weight": _tensor("hidden_norm.weight", (config.hidden_size,)),
            "norm.weight": _tensor("norm.weight", (config.hidden_size,)),
        }
    )
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": _tensor(
                    f"{prefix}.input_layernorm.weight", (config.hidden_size,)
                ),
                f"{prefix}.post_attention_layernorm.weight": _tensor(
                    f"{prefix}.post_attention_layernorm.weight", (config.hidden_size,)
                ),
                f"{prefix}.self_attn.qkv_proj.weight": _tensor(
                    f"{prefix}.self_attn.qkv_proj.weight",
                    (config.qkv_features, config.hidden_size),
                ),
                f"{prefix}.self_attn.g_proj.weight": _tensor(
                    f"{prefix}.self_attn.g_proj.weight",
                    (config.num_attention_heads, config.hidden_size),
                ),
                f"{prefix}.self_attn.o_proj.weight": _tensor(
                    f"{prefix}.self_attn.o_proj.weight",
                    (config.hidden_size, config.q_features),
                ),
                f"{prefix}.self_attn.q_norm.weight": _tensor(
                    f"{prefix}.self_attn.q_norm.weight", (config.head_dim,)
                ),
                f"{prefix}.self_attn.k_norm.weight": _tensor(
                    f"{prefix}.self_attn.k_norm.weight", (config.head_dim,)
                ),
                f"{prefix}.mlp.gate_proj.weight": _tensor(
                    f"{prefix}.mlp.gate_proj.weight",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}.mlp.up_proj.weight": _tensor(
                    f"{prefix}.mlp.up_proj.weight",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}.mlp.down_proj.weight": _tensor(
                    f"{prefix}.mlp.down_proj.weight",
                    (config.hidden_size, config.intermediate_size),
                ),
            }
        )
    return WeightIndex(
        Path("/fake/laguna-dflash"),
        _laguna_config(),
        tensors,
        (Path("fake.safetensors"),),
    )


def _tensor(name: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, shard_path=Path("fake.safetensors"), dtype="BF16", shape=shape)
