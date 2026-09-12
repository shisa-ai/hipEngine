from __future__ import annotations

import pytest

from hipengine.models.evie import (
    EVIE_ARCHITECTURE,
    EVIE_DEFAULT_HEAD,
    EvieModel,
    expected_evie_weight_shapes,
    parse_evie_model_spec,
    validate_evie_weight_index,
)
from hipengine.models.registry import resolve_model


def test_plugin_registered() -> None:
    plugin = resolve_model(EVIE_ARCHITECTURE)
    assert isinstance(plugin, EvieModel)
    assert plugin.name == "evie_4p5b"
    assert plugin.default_quant == "fp32"


def _evie_config() -> dict:
    return {
        "anchor_dim": 128,
        "architectures": ["ColQwen3_5"],
        "dim": 2048,
        "head_dims": [64, 128, 256, 512, 1024, 2048],
        "image_token_id": 248056,
        "mrl_prefix": True,
        "model_type": "qwen3_5",
        "text_config": {
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": (
                ["linear_attention"] * 3 + ["full_attention"]
            )
            * 8,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
        },
    }


def test_parse_spec_rejects_drift() -> None:
    spec = parse_evie_model_spec(_evie_config())
    assert spec.num_layers == 32
    assert [i for i in range(32) if spec.is_full_attention(i)] == [
        3, 7, 11, 15, 19, 23, 27, 31,
    ]
    assert spec.default_head == EVIE_DEFAULT_HEAD == 128

    bad = _evie_config()
    bad["text_config"]["hidden_size"] = 2048
    with pytest.raises(ValueError, match="drifted"):
        parse_evie_model_spec(bad)

    bad = _evie_config()
    bad["text_config"]["rope_parameters"]["mrope_interleaved"] = False
    with pytest.raises(ValueError, match="mrope_interleaved"):
        parse_evie_model_spec(bad)

    bad = _evie_config()
    bad["architectures"] = ["SomethingElse"]
    with pytest.raises(ValueError, match="ColQwen3_5"):
        parse_evie_model_spec(bad)


def test_expected_weight_shapes_cover_all_layers() -> None:
    spec = parse_evie_model_spec(_evie_config())
    shapes = expected_evie_weight_shapes(spec)
    # text stack: 32 layers x (2 norms + 3 mlp) + 8 attn sets + 24 gdn sets
    text = [n for n in shapes if n.startswith("language_model.")]
    assert sum(n.endswith("input_layernorm.weight") for n in text) == 32
    assert sum(n.endswith("mlp.down_proj.weight") for n in text) == 32
    assert sum(".self_attn.o_proj.weight" in n for n in text) == 8
    assert sum(".linear_attn.out_proj.weight" in n for n in text) == 24
    vision = [n for n in shapes if n.startswith("visual.")]
    assert sum(n.endswith("attn.qkv.weight") for n in vision) == 24
    assert shapes["custom_text_proj.weight"] == (2048, 2560)


def test_validate_against_cached_snapshot() -> None:
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.safetensors import load_weight_index

    try:
        path = resolve_model_path("tencent/EVIE-4.5B")
    except Exception:
        pytest.skip("tencent/EVIE-4.5B not in local HF cache")
    if not path.is_dir():
        pytest.skip("tencent/EVIE-4.5B not in local HF cache")
    index = load_weight_index(path)
    spec = parse_evie_model_spec(index.config)
    validate_evie_weight_index(index, spec)
