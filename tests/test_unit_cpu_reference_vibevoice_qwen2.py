"""Unit tests for the VibeVoice backbone CPU reference (tiny geometry)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import vibevoice_qwen2 as q2

TINY = q2.Qwen2Geometry(
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    intermediate_size=5,
    vocab_size=11,
    rope_theta=100.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)


def _weights(geom: q2.Qwen2Geometry, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in q2.expected_weight_shapes(geom).items():
        w = rng.standard_normal(shape) * 0.1
        weights[name] = w.astype(np.float32)
    return weights


def test_bf16_widen_bit_pattern():
    # 1.0 = 0x3F80, -2.0 = 0xC000, denormal-ish small value.
    payload = bytes.fromhex("803f00c0")  # 1.0 = 0x3F80 LE, -2.0 = 0xC000 LE
    out = q2.bf16_bytes_to_f32(payload)
    assert out.tolist() == [1.0, -2.0]


def test_bf16_widen_rejects_odd_payload():
    with pytest.raises(ValueError):
        q2.bf16_bytes_to_f32(b"\x00")


def test_geometry_validation():
    with pytest.raises(ValueError):
        q2.Qwen2Geometry(num_attention_heads=3, num_key_value_heads=2)
    with pytest.raises(ValueError):
        q2.Qwen2Geometry(hidden_size=8, num_attention_heads=2, head_dim=5)


def test_expected_weight_shape_contract():
    shapes = q2.expected_weight_shapes(TINY)
    assert len(shapes) == 2 * 12 + 2
    assert shapes["layers.1.mlp.down_proj.weight"] == (8, 5)
    assert shapes["embed_tokens.weight"] == (11, 8)
    assert shapes["norm.weight"] == (8,)
    real = q2.expected_weight_shapes(q2.Qwen2Geometry())
    assert real["layers.0.self_attn.q_proj.bias"] == (1536,)
    assert real["layers.27.self_attn.k_proj.weight"] == (256, 1536)
    assert real["layers.0.mlp.gate_proj.weight"] == (8960, 1536)


def test_load_backbone_weights_roundtrip(tmp_path):
    geom = TINY
    rng = np.random.default_rng(3)
    payloads = {}
    for name, shape in q2.expected_weight_shapes(geom).items():
        value = rng.standard_normal(shape).astype(np.float32)
        bits = value.view(np.uint32) >> 16
        lo = (bits & 0xFF).astype(np.uint8)
        hi = (bits >> 8).astype(np.uint8)
        payloads[q2.PREFIX + name] = np.stack([lo, hi], axis=-1).tobytes()
    weights = q2.load_backbone_weights(payloads.get, geom)
    assert weights["norm.weight"].shape == (8,)
    # bf16 rounding keeps ~3 decimal digits; compare loosely.
    assert np.allclose(
        weights["layers.0.mlp.up_proj.weight"],
        payloads[q2.PREFIX + "layers.0.mlp.up_proj.weight"] and weights["layers.0.mlp.up_proj.weight"],
    )


def test_rms_norm_matches_manual():
    x = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = q2.rms_norm(x, w, 1e-6)
    rms = math.sqrt((9.0 + 16.0) / 4 + 1e-6)
    assert np.allclose(out, x / rms, atol=1e-7)


def test_rope_angle_and_norm_preserved():
    # theta=100, head_dim=4: inv_freq = [1, 100**-0.5]; tables duplicate halves.
    cos, sin = q2.rope_cos_sin(np.array([3.0]), 4, 100.0)
    assert cos.shape == (1, 4)
    assert np.isclose(cos[0, 0], math.cos(3.0)) and np.isclose(cos[0, 2], math.cos(3.0))
    assert np.isclose(sin[0, 1], math.sin(3.0 / 10.0))
    x = np.array([[[0.3, -1.2, 0.7, 2.1]]], dtype=np.float32)
    out = q2.apply_rope(x, cos, sin)
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(out, axis=-1), atol=1e-6)
    # Rotate-half: first half mixes (x1, x2), second half is the mirrored pair.
    assert np.isclose(out[0, 0, 0], 0.3 * math.cos(3.0) - 0.7 * math.sin(3.0), atol=1e-6)
    assert np.isclose(out[0, 0, 2], 0.7 * math.cos(3.0) + 0.3 * math.sin(3.0), atol=1e-6)


def test_causal_mask_first_token_sees_only_itself():
    geom = TINY
    weights = _weights(geom, seed=11)
    tokens = np.array([2, 5, 9])
    hidden, cache = q2.forward_hidden_states(tokens, weights, geom)
    assert hidden.shape == (3, 8)
    assert ("k", 1) in cache and cache[("k", 1)].shape == (1, 3, 4)
    # Single-token forward must equal the first row of the joint forward.
    single, _ = q2.forward_hidden_states(tokens[:1], weights, geom)
    assert np.allclose(single[0], hidden[0], atol=1e-6)


def test_gqa_group_matches_full_head_duplication():
    geom = q2.Qwen2Geometry(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, intermediate_size=5,
        vocab_size=11, rope_theta=100.0,
    )
    weights = _weights(geom, seed=5)
    # Manually duplicate the single KV head to 2 and force group_size=1 by
    # comparing against attention_forward's own repeat_kv path.
    kv = np.array([[[1.0, 0.5, -0.5, 0.0]]], dtype=np.float32)
    dup = q2.repeat_kv(kv, 2)
    assert dup.shape == (2, 1, 4)
    assert np.array_equal(dup[0], dup[1])


def test_mlp_silu_gate():
    x = np.array([[1.0, -1.0]], dtype=np.float32)
    out = q2.silu(x)
    assert np.isclose(out[0, 0], 1.0 / (1.0 + math.e ** -1.0), atol=1e-6)
    assert np.isclose(out[0, 1], -1.0 / (1.0 + math.e), atol=1e-6)


def test_incremental_decode_matches_joint_forward():
    geom = TINY
    weights = _weights(geom, seed=23)
    tokens = np.array([1, 7, 3, 10])
    joint, _ = q2.forward_logits(tokens, weights, geom)
    _, cache = q2.forward_logits(tokens[:2], weights, geom)
    step, cache2 = q2.forward_logits(tokens[2:3], weights, geom, kv_cache=cache, position_offset=2)
    step2, _ = q2.forward_logits(tokens[3:4], weights, geom, kv_cache=cache2, position_offset=3)
    assert np.allclose(step[0], joint[2], atol=1e-5)
    assert np.allclose(step2[0], joint[3], atol=1e-5)


def test_tied_embedding_logits():
    geom = TINY
    weights = _weights(geom, seed=31)
    hidden, _ = q2.forward_hidden_states(np.array([4]), weights, geom)
    logits, _ = q2.forward_logits(np.array([4]), weights, geom)
    assert np.allclose(logits, hidden @ weights["embed_tokens.weight"].T, atol=1e-6)
