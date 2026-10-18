from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.models.registry import resolve_model
from hipengine.models.surya import (
    SURYA_ARCHITECTURE,
    SURYA_EOS_TOKEN_ID,
    SuryaModel,
    expected_surya_weight_shapes,
    parse_surya_model_spec,
    validate_surya_weight_index,
)


def test_plugin_registered() -> None:
    plugin = resolve_model(SURYA_ARCHITECTURE)
    assert isinstance(plugin, SuryaModel)
    assert plugin.name == "surya_ocr2"
    assert plugin.default_quant == "fp32"


def _surya_config() -> dict:
    """Faithful copy of the pinned datalab-to/surya-ocr-2 config.json
    (revision 3b3d4cdf), including the stale text_config.eos_token_id."""

    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 11,
        "model_type": "qwen3_5",
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": True,
        "video_token_id": 12,
        "vision_end_token_id": 10,
        "vision_start_token_id": 9,
        "text_config": {
            "attention_bias": False,
            "attn_output_gate": True,
            "eos_token_id": 248044,  # stale out-of-vocab metadata (trap 1)
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "num_nextn_predict_layers": 1,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "vocab_size": 65425,
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 12,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 768,
            "in_channels": 3,
            "intermediate_size": 3072,
            "model_type": "qwen3_5",
            "num_heads": 12,
            "num_position_embeddings": 2304,
            "out_hidden_size": 1024,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def test_parse_pinned_contract() -> None:
    spec = parse_surya_model_spec(_surya_config())
    assert spec.hidden_size == 1024
    assert spec.num_layers == 24
    assert spec.layer_types.count("linear_attention") == 18
    assert spec.layer_types.count("full_attention") == 6
    assert spec.num_attention_heads == 8
    assert spec.num_key_value_heads == 2
    assert spec.head_dim == 256
    assert spec.gdn_num_key_heads == 16
    assert spec.gdn_inner_size == 2048
    assert spec.gdn_time_step_rank == 16
    assert spec.vision_hidden_size == 768
    assert spec.vision_depth == 12
    assert spec.vision_out_hidden_size == 1024
    assert spec.mtp_num_layers == 1
    assert spec.tie_word_embeddings
    assert spec.is_full_attention(3) and spec.is_full_attention(23)
    assert not spec.is_full_attention(2)


def test_effective_eos_ignores_stale_text_config_value() -> None:
    """Regression: text_config.eos_token_id=248044 is stale out-of-vocab
    metadata; the effective EOS must stay the tokenizer's 2."""

    config = _surya_config()
    assert config["text_config"]["eos_token_id"] == 248044
    spec = parse_surya_model_spec(config)
    assert spec.eos_token_id == SURYA_EOS_TOKEN_ID == 2
    assert spec.eos_token_id != 248044
    assert spec.pad_token_id == 0
    assert spec.image_token_id == 11
    assert spec.vision_start_token_id == 9
    assert spec.vision_end_token_id == 10


def test_generation_config_crosscheck() -> None:
    spec = parse_surya_model_spec(
        _surya_config(), {"eos_token_id": 2, "pad_token_id": 0}
    )
    assert spec.eos_token_id == 2
    with pytest.raises(ValueError, match="contradicts"):
        parse_surya_model_spec(_surya_config(), {"eos_token_id": 151645})


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c["architectures"].__setitem__(0, "Qwen3_5ForCausalLM"), "not a Surya"),
        (lambda c: c.__setitem__("model_type", "qwen3"), "model_type"),
        (lambda c: c["text_config"].__setitem__("hidden_size", 2048), "drifted"),
        (
            lambda c: (
                c["text_config"].__setitem__("num_hidden_layers", 28),
                c["text_config"].__setitem__(
                    "layer_types", (["linear_attention"] * 3 + ["full_attention"]) * 7
                ),
            ),
            "drifted",
        ),
        (lambda c: c["text_config"].__setitem__("vocab_size", 151936), "drifted"),
        (lambda c: c["text_config"].__setitem__("linear_num_key_heads", 32), "drifted"),
        (lambda c: c["text_config"].__setitem__("tie_word_embeddings", False), "tied"),
        (
            lambda c: c["text_config"]["rope_parameters"].__setitem__(
                "mrope_section", [16, 16, 32]
            ),
            "mrope_section",
        ),
        (
            lambda c: c["text_config"]["rope_parameters"].__setitem__(
                "rope_theta", 1000000.0
            ),
            "rotary",
        ),
        (
            lambda c: c["text_config"]["rope_parameters"].__setitem__(
                "mrope_interleaved", False
            ),
            "mrope_interleaved",
        ),
        (lambda c: c["vision_config"].__setitem__("depth", 24), "drifted"),
        (lambda c: c["vision_config"].__setitem__("hidden_size", 1152), "drifted"),
    ],
)
def test_drift_rejection(mutate: object, match: str) -> None:
    config = _surya_config()
    mutate(config)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        parse_surya_model_spec(config)


def test_layer_schedule_rejection() -> None:
    config = _surya_config()
    # 4 GDN then 1 full: right counts, wrong cadence
    config["text_config"]["layer_types"] = (
        ["linear_attention"] * 4 + ["full_attention"]
    ) * 4 + ["linear_attention"] * 4
    with pytest.raises(ValueError, match="schedule"):
        parse_surya_model_spec(config)


def test_expected_weight_manifest_shapes() -> None:
    spec = parse_surya_model_spec(_surya_config())
    shapes = expected_surya_weight_shapes(spec)
    # measured checkpoint: 488 tensors, 686.2M params
    assert len(shapes) == 488
    assert shapes["model.language_model.embed_tokens.weight"] == (65425, 1024)
    assert (
        shapes["model.language_model.layers.3.self_attn.q_proj.weight"]
        == (4096, 1024)  # 8 heads x 256 x 2 (sigmoid output gate)
    )
    assert (
        shapes["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
        == (6144, 1024)
    )
    assert (
        shapes["model.language_model.layers.0.linear_attn.conv1d.weight"]
        == (6144, 1, 4)
    )
    assert shapes["model.language_model.layers.0.linear_attn.A_log"] == (16,)
    assert shapes["model.visual.merger.linear_fc1.weight"] == (3072, 3072)
    assert shapes["model.visual.patch_embed.proj.weight"] == (768, 3, 2, 16, 16)
    assert shapes["mtp.fc.weight"] == (1024, 2048)
    # full-attention and GDN layer key sets are disjoint
    full_keys = {k for k in shapes if ".layers.3." in k}
    gdn_keys = {k for k in shapes if ".layers.0." in k}
    assert full_keys.isdisjoint(gdn_keys)
    assert sum(1 for k in shapes if k.startswith("model.visual.blocks.")) == 144
    assert sum(1 for k in shapes if k.startswith("mtp.layers.")) == 11


class _FakeIndex:
    """Minimal WeightIndex stand-in for synthetic validation tests."""

    def __init__(self, tensors: dict[str, TensorInfo]) -> None:
        self.tensors = tensors


def _synthetic_index(spec: object) -> _FakeIndex:
    shapes = expected_surya_weight_shapes(spec)  # type: ignore[arg-type]
    return _FakeIndex(
        {
            name: TensorInfo(
                name=name,
                shard_path=Path("/fake/model.safetensors"),
                dtype="BF16",
                shape=shape,
            )
            for name, shape in shapes.items()
        }
    )


def test_validate_synthetic_index_passes() -> None:
    spec = parse_surya_model_spec(_surya_config())
    validate_surya_weight_index(_synthetic_index(spec), spec)  # type: ignore[arg-type]


def test_validate_missing_tensor() -> None:
    spec = parse_surya_model_spec(_surya_config())
    index = _synthetic_index(spec)
    del index.tensors["model.visual.merger.linear_fc2.bias"]
    with pytest.raises(ValueError, match="missing tensors"):
        validate_surya_weight_index(index, spec)  # type: ignore[arg-type]


def test_validate_extra_tensor() -> None:
    spec = parse_surya_model_spec(_surya_config())
    index = _synthetic_index(spec)
    index.tensors["lm_head.weight"] = TensorInfo(
        name="lm_head.weight",
        shard_path=Path("/fake/model.safetensors"),
        dtype="BF16",
        shape=(65425, 1024),
    )
    with pytest.raises(ValueError, match="unexpected tensors"):
        validate_surya_weight_index(index, spec)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field, value",
    [("shape", (1024, 65425)), ("dtype", "F16")],
)
def test_validate_tensor_drift(field: str, value: object) -> None:
    spec = parse_surya_model_spec(_surya_config())
    index = _synthetic_index(spec)
    name = "model.language_model.norm.weight"
    info = index.tensors[name]
    mutated = TensorInfo(
        name=info.name,
        shard_path=info.shard_path,
        dtype=value if field == "dtype" else info.dtype,
        shape=value if field == "shape" else info.shape,  # type: ignore[arg-type]
    )
    index.tensors[name] = mutated
    with pytest.raises(ValueError):
        validate_surya_weight_index(index, spec)  # type: ignore[arg-type]


def test_validate_against_cached_snapshot() -> None:
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.safetensors import load_weight_index

    try:
        path = resolve_model_path("datalab-to/surya-ocr-2")
    except Exception:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    if not path.is_dir():
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    index = load_weight_index(path)
    assert len(index.tensors) == 488
    assert not any("lm_head" in name for name in index.tensors), (
        "Surya OCR 2 must keep tied embeddings (no lm_head tensor)"
    )
    generation_config: dict = {}
    gen_path = path / "generation_config.json"
    if gen_path.exists():
        generation_config = json.loads(gen_path.read_text())
    spec = parse_surya_model_spec(index.config, generation_config)
    assert spec.eos_token_id == 2
    validate_surya_weight_index(index, spec)
