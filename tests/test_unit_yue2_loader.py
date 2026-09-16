"""Loader gate: validated inventory, tensor mapping, and weight-norm folding.

These tests need the checkpoints in the local Hugging Face cache (or the paths in
``YUE2_MODEL_DIR`` / ``YUE2_VAE_DIR``); they skip cleanly when absent. No torch
import and no GPU allocation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.yue2 import (
    PINNED_CONFIG,
    PINNED_MODEL_SHA256,
    YuE2Config,
    YuE2WeightError,
    expected_transformer_tensors,
    expected_vae_decoder_tensors,
    fold_weight_norm,
    load_yue2_vae_decoder,
    load_yue2_weights,
)

CACHE = Path.home() / ".cache/huggingface/hub"


def _resolve(env: str, pattern: str) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override)
    for directory in sorted(CACHE.glob(pattern)):
        if (directory / "model.safetensors").is_file():
            return directory
    pytest.skip(f"checkpoint for {pattern} not present in the local cache")


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return _resolve("YUE2_MODEL_DIR", "models--m-a-p--YuE2-3B/snapshots/*")


@pytest.fixture(scope="module")
def vae_dir() -> Path:
    return _resolve("YUE2_VAE_DIR", "models--m-a-p--YuE2-Vae/snapshots/*")


@pytest.fixture(scope="module")
def weights(model_dir):
    return load_yue2_weights(model_dir)


@pytest.fixture(scope="module")
def decoder(vae_dir):
    return load_yue2_vae_decoder(vae_dir)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_config_pins_every_geometry_field():
    config = YuE2Config.from_dict(dict(PINNED_CONFIG))
    config.validate()
    assert config.q_width == 2048 and config.kv_width == 1024
    with pytest.raises(YuE2WeightError):
        YuE2Config.from_dict({**PINNED_CONFIG, "hidden_size": 1024}).validate()
    with pytest.raises(YuE2WeightError):
        YuE2Config.from_dict({**PINNED_CONFIG, "latent_dim": 32}).validate()
    with pytest.raises(YuE2WeightError):
        YuE2Config.from_dict({**PINNED_CONFIG, "surprise": 1})


def test_config_accepts_huggingface_metadata_keys():
    config = YuE2Config.from_dict(
        {
            **PINNED_CONFIG,
            "model_type": "yue2",
            "architectures": ["YuE2ForCausalLM"],
            "auto_map": {"AutoConfig": "modeling_yue2.YuE2Config"},
            "transformers_version": "4.57.6",
            "dtype": "bfloat16",
            "tie_word_embeddings": False,
            "bos_token_id": None,
        }
    )
    assert config.hidden_size == 2048


def test_expected_tensor_sets_are_sized():
    assert len(expected_transformer_tensors(YuE2Config())) == 628
    assert len(expected_vae_decoder_tensors()) == 217


# ---------------------------------------------------------------------------
# transformer weights
# ---------------------------------------------------------------------------


def test_transformer_inventory_and_identity(weights, model_dir):
    assert weights.config.num_hidden_layers == 28
    assert len(weights.layers) == 28
    assert weights.embed_tokens.shape == (184704, 2048)
    assert weights.embed_tokens.dtype == np.uint16
    assert weights.lm_head.shape == (184704, 2048)
    assert weights.latent_pos_embed.shape == (24576, 2048)
    assert weights.latent_pos_embed.dtype == np.uint16
    assert weights.identity["tensor_count"] == 628
    assert weights.identity["files"]["config.json"]["bytes"] > 0
    assert weights.identity["shards"][0]["bytes"] == 7261441640
    # Host footprint is the checkpoint payload, not a widened copy.
    assert weights.bytes == 7261368448


def test_layer_shapes_and_dual_paths(weights):
    for layer in weights.layers:
        assert layer.self_attn.q.shape == (2048, 2048)
        assert layer.self_attn.k.shape == (1024, 2048)
        assert layer.self_attn.v.shape == (1024, 2048)
        assert layer.self_attn.o.shape == (2048, 2048)
        assert layer.self_attn.q_norm.shape == (128,)
        assert layer.self_attn.k_norm.shape == (128,)
        assert layer.mlp.gate.shape == (6144, 2048)
        assert layer.mlp.down.shape == (2048, 6144)
        # The NAR path has its own projections, norms and MLP.
        assert layer.nar_self_attn.q.shape == (2048, 2048)
        assert layer.nar_mlp.gate.shape == (6144, 2048)
        assert not np.array_equal(layer.self_attn.q, layer.nar_self_attn.q)
        assert not np.array_equal(layer.mlp.gate, layer.nar_mlp.gate)


def test_weights_are_bf16_bit_patterns(weights):
    """BF16 payloads must survive the load unchanged (no widening, no rounding)."""
    sample = weights.layers[0].self_attn.q[:8, :8]
    assert sample.dtype == np.uint16
    # 0x0000/0x8000 would be zero/signed zero; a real checkpoint has neither
    # everywhere, and every value must be a finite BF16 pattern.
    finite = (sample & np.uint16(0x7F80)) != np.uint16(0x7F80)
    assert finite.all()


def test_head_norms_are_trained_and_per_head(weights):
    # Q/K head norms are RMSNorm(128) parameters that the release trained; a
    # loader that mis-mapped them would show ones or duplicated rows here.
    for scope in ("self_attn", "nar_self_attn"):
        q_norm = getattr(weights.layers[0], scope).q_norm
        k_norm = getattr(weights.layers[0], scope).k_norm
        assert q_norm.shape == (128,) and q_norm.dtype == np.uint16
        assert not np.array_equal(q_norm, np.full(128, 0x3F80, dtype=np.uint16))
        assert not np.array_equal(q_norm, k_norm)
        values = (q_norm.astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(values).all()
        assert values.std() > 0.05


def test_load_rejects_a_wrong_directory(tmp_path):
    with pytest.raises((FileNotFoundError, YuE2WeightError, KeyError, ValueError)):
        load_yue2_weights(tmp_path)


# ---------------------------------------------------------------------------
# VAE decoder
# ---------------------------------------------------------------------------


def test_vae_decoder_inventory(decoder):
    assert len(decoder.blocks) == 6
    assert decoder.input_conv.weight.shape == (2048, 64, 7)
    assert decoder.output_conv.weight.shape == (2, 64, 7)
    assert decoder.output_conv.bias is None
    assert all(block.upsample.transposed for block in decoder.blocks)
    # The released decoder applies the strides in reverse: 6x first, 2x last.
    assert [block.upsample.stride for block in decoder.blocks] == [6, 5, 4, 4, 2, 2]
    assert [block.upsample.weight.shape[-1] for block in decoder.blocks] == [12, 10, 8, 8, 4, 4]
    assert [block.upsample.weight.shape[:2] for block in decoder.blocks] == [
        (2048, 1024),
        (1024, 512),
        (512, 256),
        (256, 128),
        (128, 64),
        (64, 64),
    ]
    assert [unit.conv.dilation for unit in decoder.blocks[0].residual_units] == [1, 3, 9]
    assert decoder.all_convs().__len__() == 1 + 6 * (1 + 3 * 2) + 1
    assert all(conv.weight.dtype == np.float32 for conv in decoder.all_convs())
    assert all(conv.bias is None or conv.bias.dtype == np.float32 for conv in decoder.all_convs())


def test_vae_natural_length_matches_release_formula(decoder):
    for frames in (1, 2, 3, 16, 64, 1024):
        assert decoder.natural_output_length(frames) == 1920 * frames - 64


def test_vae_stride_product(decoder):
    product = 1
    for block in decoder.blocks:
        product *= block.upsample.stride
    assert product == decoder.downsampling_ratio == 1920


def test_vae_identity_and_release_variant(decoder):
    assert decoder.release_variant == "standard"
    assert decoder.identity["shards"][0]["bytes"] == 530512720
    assert decoder.sample_rate == 48000 and decoder.latent_dim == 64


def test_vae_load_rejects_a_wrong_directory(tmp_path):
    with pytest.raises((FileNotFoundError, YuE2WeightError, KeyError, ValueError)):
        load_yue2_vae_decoder(tmp_path)


# ---------------------------------------------------------------------------
# weight-norm folding
# ---------------------------------------------------------------------------


def test_fold_weight_norm_matches_definition():
    rng = np.random.default_rng(0)
    weight_v = rng.standard_normal((4, 6, 3)).astype(np.float32)
    weight_g = (rng.random((4, 1, 1)).astype(np.float32) + 0.5)
    folded = fold_weight_norm(weight_g, weight_v)
    norm = np.sqrt(np.sum(weight_v.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(folded, weight_g / norm * weight_v, rtol=1e-6, atol=0)
    # Row norms equal weight_g after folding.
    np.testing.assert_allclose(
        np.sqrt(np.sum(folded.astype(np.float64) ** 2, axis=(1, 2))), weight_g.reshape(-1), rtol=1e-6
    )


def test_fold_weight_norm_rejects_bad_shapes_and_zero_norm():
    with pytest.raises(YuE2WeightError):
        fold_weight_norm(np.ones((4, 1), dtype=np.float32), np.ones((4, 6, 3), dtype=np.float32))
    with pytest.raises(YuE2WeightError):
        fold_weight_norm(np.ones((4, 1, 1), dtype=np.float32), np.zeros((4, 6, 3), dtype=np.float32))


def test_pinned_sha256_documented():
    """The pinned model payload hash is recorded in the loader, not inferred."""
    assert len(PINNED_MODEL_SHA256) == 64


def test_oracle_environment_record_matches_the_loader_pins():
    """The frozen oracle environment must agree with the loader's pins."""
    from hipengine.loading.yue2 import PINNED_VAE_REVISION

    path = Path(__file__).resolve().parents[1] / "tests/fixtures/yue2/oracle_env.json"
    record = json.loads(path.read_text())
    assert record["model"]["tensor_count"] == len(expected_transformer_tensors(YuE2Config()))
    # The VAE checkpoint ships encoder and decoder; the loader maps the 217
    # decoder tensors and ignores the encoder half.
    assert record["vae"]["tensor_count"] == 435
    assert len(expected_vae_decoder_tensors()) == 217
    assert record["model"]["shards"][0]["name"] == PINNED_MODEL_SHA256
    assert record["model"]["shards"][0]["bytes"] == 7261441640
    assert record["vae"]["revision"] == PINNED_VAE_REVISION
    assert record["vae"]["shards"][0]["bytes"] == 530512720
    # Fixtures were generated on a ROCm torch build; the architecture is part of
    # the evidence, not an assumption.
    assert record["torch"].startswith("2.") and record["torch_hip"].startswith("7.")
    assert record["gcn_arch"].startswith("gfx")
    assert record["shootout_wheel"]["sha256"] == (
        "8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641"
    )
    assert set(record["source"]["files"]) >= {
        "protocol.py",
        "sampling.py",
        "modeling_yue2.py",
        "nar.py",
        "modeling_vae.py",
    }
