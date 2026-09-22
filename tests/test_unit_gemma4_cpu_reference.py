"""Gemma 4 text CPU reference gated against the HF transformers oracle.

The fixture under ``tests/fixtures/gemma4`` is produced by
``scripts/gemma4_hf_tiny_oracle.py`` from a tiny randomly-initialized
``Gemma4ForCausalLM``. It exercises every structural feature the real
``google/gemma-4-26B-A4B-it`` text tower uses, so a passing forward here is
evidence that the reference implements Gemma 4 and not an adjacent model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    gemma4_attention_mask,
    gemma4_gelu_tanh,
    gemma4_rmsnorm,
    gemma4_rope_tables,
    gemma4_router_topk,
    gemma4_text_config_from_hf,
    gemma4_text_forward,
    gemma4_text_weights_from_hf,
    register_gemma4_cpu_reference_kernels,
)
from hipengine.kernels.registry import resolve

FIXTURE = Path(__file__).parent / "fixtures" / "gemma4"

# The reference is float32 NumPy against a float32 torch eager forward. Observed
# worst-case deviation is 1.4e-6 on logits; the gate is tight enough that a
# wrong activation, a missing norm, or a head-dimension mix-up cannot pass.
_LOGIT_ATOL = 1.0e-5
_STAGE_ATOL = 5.0e-5


def _load() -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    meta = json.loads((FIXTURE / "config.json").read_text())
    weights = dict(np.load(FIXTURE / "weights.npz"))
    activations = dict(np.load(FIXTURE / "activations.npz"))
    return meta, weights, activations


def test_fixture_weight_digest_is_pinned() -> None:
    """Fail loudly when the committed fixture is regenerated without review."""

    meta, weights, _ = _load()
    digest = hashlib.sha256()
    for name in sorted(weights):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(weights[name]).tobytes())
    assert digest.hexdigest() == meta["weight_digest"]


def test_config_normalization_recovers_dual_attention_geometry() -> None:
    meta, _, _ = _load()
    config = gemma4_text_config_from_hf(meta["tiny_config"])

    assert config.num_hidden_layers == 4
    assert config.embed_scale == pytest.approx(8.0)

    sliding = config.geometry(0)
    assert sliding.layer_type == SLIDING_ATTENTION
    assert (sliding.num_heads, sliding.num_kv_heads, sliding.head_dim) == (4, 2, 32)
    assert sliding.sliding_window == 4
    assert sliding.k_eq_v is False
    assert sliding.rope.rope_angles == 16  # full rotation on sliding layers

    global_layer = config.geometry(3)
    assert global_layer.layer_type == FULL_ATTENTION
    assert (global_layer.num_heads, global_layer.num_kv_heads, global_layer.head_dim) == (4, 1, 64)
    assert global_layer.sliding_window is None
    assert global_layer.k_eq_v is True
    # partial_rotary_factor 0.25 over a 64-wide head rotates 8 pairs.
    assert global_layer.rope.rope_angles == 8
    assert global_layer.rope.rope_theta == pytest.approx(1_000_000.0)
    assert sliding.rope.rope_theta == pytest.approx(10_000.0)


def test_attention_scaling_is_one_for_both_layer_types() -> None:
    """Gemma 4 uses ``scaling = 1.0``; ``head_dim**-0.5`` would be silently wrong."""

    for head_dim in (32, 64, 256, 512):
        geometry = Gemma4AttentionGeometry(
            layer_type=SLIDING_ATTENTION,
            num_heads=4,
            num_kv_heads=2,
            head_dim=head_dim,
            rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=head_dim, rope_angles=head_dim // 2),
        )
        assert geometry.scale == 1.0


def test_proportional_rope_leaves_unrotated_pairs_untouched() -> None:
    """Pairs past the rotated span carry zero inverse frequency."""

    rope = Gemma4RopeConfig(rope_theta=1_000_000.0, head_dim=64, rope_angles=8)
    frequencies = rope.inverse_frequencies
    assert frequencies.shape == (32,)
    assert np.all(frequencies[:8] > 0.0)
    assert np.all(frequencies[8:] == 0.0)

    cos, sin = gemma4_rope_tables(rope, np.array([0, 1, 7, 100], dtype=np.int64))
    assert cos.shape == (4, 32)
    # Angle zero: cosine is one and sine is zero, so those pairs are the identity.
    assert np.allclose(cos[:, 8:], 1.0)
    assert np.allclose(sin[:, 8:], 0.0)


def test_full_rotation_layers_have_no_unrotated_pairs() -> None:
    rope = Gemma4RopeConfig(rope_theta=10_000.0, head_dim=32, rope_angles=16)
    assert np.all(rope.inverse_frequencies > 0.0)


def test_sliding_window_mask_is_causal_and_bounded() -> None:
    geometry = Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=32, rope_angles=16),
        sliding_window=4,
    )
    positions = np.arange(6)
    mask = gemma4_attention_mask(geometry, positions, positions)
    for query in range(6):
        visible = np.flatnonzero(mask[query])
        # The window is inclusive of the query and 4 tokens wide.
        assert visible.tolist() == list(range(max(0, query - 3), query + 1))


def test_global_layers_are_fully_causal() -> None:
    geometry = Gemma4AttentionGeometry(
        layer_type=FULL_ATTENTION,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
        rope=Gemma4RopeConfig(rope_theta=1_000_000.0, head_dim=64, rope_angles=8),
        k_eq_v=True,
    )
    mask = gemma4_attention_mask(geometry, np.arange(5), np.arange(5))
    assert mask.tolist() == np.tril(np.ones((5, 5), dtype=bool)).tolist()


def test_gelu_tanh_is_not_silu() -> None:
    """A SiLU substitution is the single easiest way to get a plausible wrong answer."""

    x = np.array([-6.0, -1.0, -0.25, 0.0, 0.5, 3.0], dtype=np.float32)
    got = gemma4_gelu_tanh(x)
    expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    assert np.allclose(got, expected, atol=1e-6)
    silu = x / (1.0 + np.exp(-x))
    assert np.abs(got - silu).max() > 1e-2


def test_weightless_rmsnorm_matches_scaled_norm() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    got = gemma4_rmsnorm(x, None, 1e-6)
    expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    assert np.allclose(got, expected, atol=1e-6)
    weighted = gemma4_rmsnorm(x, np.full(8, 2.0, dtype=np.float32), 1e-6)
    assert np.allclose(weighted, expected * 2.0, atol=1e-6)


def test_router_renormalizes_then_applies_per_expert_scale() -> None:
    rng = np.random.default_rng(11)
    hidden = rng.standard_normal((5, 16)).astype(np.float32)
    scale = rng.standard_normal(16).astype(np.float32)
    proj = rng.standard_normal((6, 16)).astype(np.float32)
    per_expert = np.arange(1, 7, dtype=np.float32)

    probabilities, weights, index = gemma4_router_topk(
        hidden,
        norm_weight=None,
        scale=scale,
        proj_weight=proj,
        per_expert_scale=per_expert,
        top_k=2,
        scalar_root_size=16**-0.5,
        eps=1e-6,
    )

    assert probabilities.shape == (5, 6)
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
    assert weights.shape == (5, 2) and index.shape == (5, 2)

    for row in range(5):
        selected = index[row]
        # Descending probability order.
        assert probabilities[row, selected[0]] >= probabilities[row, selected[1]]
        # Renormalized weights sum to one *before* the per-expert scale.
        unscaled = weights[row] / per_expert[selected]
        assert unscaled.sum() == pytest.approx(1.0, abs=1e-5)


def test_router_uses_the_weightless_norm() -> None:
    """The router norm is weightless; the learned ``scale`` is the only scale."""

    rng = np.random.default_rng(13)
    hidden = rng.standard_normal((2, 8)).astype(np.float32)
    scale = rng.standard_normal(8).astype(np.float32)
    proj = rng.standard_normal((4, 8)).astype(np.float32)
    per_expert = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)

    probabilities, _, _ = gemma4_router_topk(
        hidden,
        norm_weight=None,
        scale=scale,
        proj_weight=proj,
        per_expert_scale=per_expert,
        top_k=2,
        scalar_root_size=8**-0.5,
        eps=1e-6,
    )

    # Rebuild the documented pipeline by hand: weightless RMS norm, then
    # ``scale * hidden_size**-0.5``, then the projection and a full softmax.
    normalized = hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-6)
    expected_logits = (normalized * scale * np.float32(8**-0.5)) @ proj.T
    shifted = np.exp(expected_logits - expected_logits.max(axis=-1, keepdims=True))
    expected = shifted / shifted.sum(axis=-1, keepdims=True)
    assert np.allclose(probabilities, expected, atol=1e-6)


def test_hf_oracle_stage_boundaries_match() -> None:
    meta, weights_raw, activations = _load()
    config = gemma4_text_config_from_hf(meta["tiny_config"])
    weights = gemma4_text_weights_from_hf(weights_raw, config)

    result = gemma4_text_forward(
        weights,
        config,
        meta["prompt"],
        capture_hidden_states=True,
    )

    assert np.abs(result.hidden_states[0] - activations["embedding"]).max() < _STAGE_ATOL
    for layer in range(config.num_hidden_layers):
        deviation = np.abs(
            result.hidden_states[layer + 1] - activations[f"layer.{layer}"]
        ).max()
        assert deviation < _STAGE_ATOL, f"layer {layer} deviation {deviation}"

    final = gemma4_rmsnorm(
        result.hidden_states[-1], weights.final_norm, config.rms_norm_eps
    )
    assert np.abs(final - activations["final_norm"]).max() < _STAGE_ATOL


def test_hf_oracle_logits_match() -> None:
    meta, weights_raw, activations = _load()
    config = gemma4_text_config_from_hf(meta["tiny_config"])
    weights = gemma4_text_weights_from_hf(weights_raw, config)

    result = gemma4_text_forward(weights, config, meta["prompt"])

    assert result.logits.shape == activations["logits"].shape
    deviation = np.abs(result.logits - activations["logits"]).max()
    assert deviation < _LOGIT_ATOL, f"logit deviation {deviation}"
    assert np.array_equal(result.logits.argmax(axis=-1), activations["logits"].argmax(axis=-1))


def test_hf_adapter_rejects_v_proj_on_k_eq_v_layers() -> None:
    meta, weights_raw, _ = _load()
    config = gemma4_text_config_from_hf(meta["tiny_config"])
    poisoned = dict(weights_raw)
    poisoned["model.layers.3.self_attn.v_proj.weight"] = np.zeros(
        (64, 64), dtype=np.float32
    )
    with pytest.raises(ValueError, match="attention_k_eq_v"):
        gemma4_text_weights_from_hf(poisoned, config)


def test_softcapping_is_applied_to_logits() -> None:
    """Without the softcap the logits would not satisfy the tanh envelope."""

    meta, weights_raw, _ = _load()
    config = gemma4_text_config_from_hf(meta["tiny_config"])
    weights = gemma4_text_weights_from_hf(weights_raw, config)

    capped = gemma4_text_forward(weights, config, meta["prompt"]).logits
    assert np.abs(capped).max() <= config.final_logit_softcapping + 1e-5

    uncapped_config = gemma4_text_config_from_hf(
        {**meta["tiny_config"], "final_logit_softcapping": None}
    )
    uncapped = gemma4_text_forward(weights, uncapped_config, meta["prompt"]).logits
    assert np.abs(uncapped).max() > 0.0
    cap = np.float32(30.0)
    assert np.allclose(capped, np.tanh(uncapped / cap) * cap, atol=1e-6)
    # The softcap never grows a logit. The slack absorbs float32 rounding at
    # this fixture's small logit magnitudes, where tanh(x/30)*30 can round back
    # to exactly x.
    assert np.all(np.abs(capped) <= np.abs(uncapped) + 1e-6)


def test_reference_registers_under_the_cpu_backend() -> None:
    register_gemma4_cpu_reference_kernels(replace=True)
    kernel = resolve(
        backend="cpu_reference", layer="gemma4_text", quant="fp32", variant="reference_forward"
    )
    assert callable(kernel)
    assert (
        resolve(backend="cpu_reference", layer="gemma4_text", quant="fp32", variant="gelu_tanh")
        is gemma4_gelu_tanh
    )
