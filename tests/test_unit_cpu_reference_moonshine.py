from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_apply_partial_rope,
    moonshine_attention,
    moonshine_decoder_mlp,
    moonshine_fixed_cache_read,
    moonshine_fixed_cache_write,
    moonshine_layernorm,
    moonshine_lm_head_argmax,
    moonshine_projection,
    moonshine_residual,
    moonshine_rope_tables,
    moonshine_stable_argmax,
    moonshine_tied_lm_logits,
    moonshine_triple_projection,
)
from hipengine.kernels.registry import resolve

FIXTURE = Path(__file__).parent / "fixtures/cpu_reference/moonshine/moonshine_primitives_v1.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def f16(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float16)


def test_moonshine_layernorm_matches_hand_fixture_with_fp32_statistics() -> None:
    fixture = load_fixture()["layernorm"]
    actual = moonshine_layernorm(f16(fixture["x"]), f16(fixture["weight"]), eps=fixture["eps"])
    np.testing.assert_array_equal(actual, f16(fixture["expected"]))
    assert actual.dtype == np.float16

    x32 = np.asarray(fixture["x"], dtype=np.float32)
    mean = np.mean(x32, axis=-1, keepdims=True, dtype=np.float32)
    centered = (x32 - mean).astype(np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    independent = (
        centered
        * np.reciprocal(np.sqrt(variance + np.float32(fixture["eps"])))
        * np.asarray(fixture["weight"], dtype=np.float32)
    ).astype(np.float16)
    np.testing.assert_array_equal(actual, independent)


def test_moonshine_partial_rope_is_interleaved_and_preserves_pass_dimensions() -> None:
    fixture = load_fixture()["rope"]
    query, key = moonshine_apply_partial_rope(
        f16(fixture["query"]),
        f16(fixture["key"]),
        f16(fixture["cos"]),
        f16(fixture["sin"]),
        position_ids=np.asarray(fixture["position_ids"], dtype=np.int64),
        rotary_dim=fixture["rotary_dim"],
    )
    np.testing.assert_array_equal(query, f16(fixture["expected_query"]))
    np.testing.assert_array_equal(key, f16(fixture["expected_key"]))
    np.testing.assert_array_equal(query[..., 4:], f16(fixture["query"])[..., 4:])
    np.testing.assert_array_equal(key[..., 4:], f16(fixture["key"])[..., 4:])


def test_moonshine_rope_tables_cover_required_positions_and_geometry() -> None:
    cos, sin = moonshine_rope_tables(194, rotary_dim=32, theta=10_000.0)
    assert cos.shape == sin.shape == (194, 16)
    assert cos.dtype == sin.dtype == np.float16
    np.testing.assert_array_equal(cos[0], np.ones(16, dtype=np.float16))
    np.testing.assert_array_equal(sin[0], np.zeros(16, dtype=np.float16))

    q = np.arange(1 * 8 * 4 * 52, dtype=np.float16).reshape(1, 8, 4, 52) / 100
    positions = np.asarray([[0, 1, 63, 193]], dtype=np.int64)
    rotated, _ = moonshine_apply_partial_rope(q, q, cos, sin, position_ids=positions, rotary_dim=32)
    assert rotated.shape == q.shape
    assert np.isfinite(rotated).all()
    np.testing.assert_array_equal(rotated[..., 32:], q[..., 32:])


def test_moonshine_fixed_cache_write_read_uses_visible_prefix_only() -> None:
    key_cache = np.zeros((1, 2, 4, 3), dtype=np.float16)
    value_cache = np.zeros_like(key_cache)
    key0 = np.arange(6, dtype=np.float16).reshape(1, 2, 1, 3)
    value0 = key0 + np.float16(10)
    visible_key, visible_value = moonshine_fixed_cache_write(
        key_cache, value_cache, key0, value0, position=0
    )
    np.testing.assert_array_equal(visible_key, key0)
    np.testing.assert_array_equal(visible_value, value0)

    key2 = key0 + np.float16(20)
    value2 = value0 + np.float16(20)
    moonshine_fixed_cache_write(key_cache, value_cache, key2, value2, position=2)
    read_key, read_value = moonshine_fixed_cache_read(key_cache, value_cache, visible_length=3)
    np.testing.assert_array_equal(read_key[:, :, 0:1], key0)
    np.testing.assert_array_equal(read_key[:, :, 1:2], np.zeros_like(key0))
    np.testing.assert_array_equal(read_key[:, :, 2:3], key2)
    np.testing.assert_array_equal(read_value[:, :, 2:3], value2)
    assert moonshine_fixed_cache_read(key_cache, value_cache, visible_length=0)[0].shape[2] == 0


def test_moonshine_fixed_cache_accepts_position_193_and_rejects_oob() -> None:
    key_cache = np.zeros((1, 8, 194, 52), dtype=np.float16)
    value_cache = np.zeros_like(key_cache)
    row = np.ones((1, 8, 1, 52), dtype=np.float16)
    visible, _ = moonshine_fixed_cache_write(key_cache, value_cache, row, row, position=193)
    assert visible.shape == (1, 8, 194, 52)
    with pytest.raises(ValueError, match="position"):
        moonshine_fixed_cache_write(key_cache, value_cache, row, row, position=194)
    with pytest.raises(ValueError, match="visible_length"):
        moonshine_fixed_cache_read(key_cache, value_cache, visible_length=195)


def test_moonshine_attention_matches_masked_hand_fixture() -> None:
    fixture = load_fixture()["attention"]
    actual = moonshine_attention(
        f16(fixture["query"]),
        f16(fixture["key"]),
        f16(fixture["value"]),
        mask=np.asarray(fixture["mask"], dtype=bool),
        scale=fixture["scale"],
    )
    np.testing.assert_array_equal(actual, f16(fixture["expected"]))


def test_moonshine_attention_supports_self_causal_and_cross_gqa() -> None:
    query = np.asarray([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]]], dtype=np.float16)
    key = np.asarray([[[[1, 0], [0, 1], [1, 1]]]], dtype=np.float16)
    value = np.asarray([[[[1, 2], [3, 4], [5, 6]]]], dtype=np.float16)
    cross = moonshine_attention(query, key, value, scale=1.0)
    assert cross.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(cross[:, 0], cross[:, 1])

    self_key = key[:, :, :2]
    self_value = value[:, :, :2]
    causal = moonshine_attention(query, self_key, self_value, scale=1.0, causal=True)
    np.testing.assert_array_equal(causal[:, :, 0], np.broadcast_to(value[:, :, 0], (1, 2, 2)))
    assert np.isfinite(causal).all()


@pytest.mark.parametrize("cache_length", [1, 2, 7, 32])
def test_moonshine_decode_attention_is_finite_at_non_power_of_two_lengths(cache_length: int) -> None:
    rng = np.random.default_rng(100 + cache_length)
    query = rng.normal(size=(1, 8, 1, 52)).astype(np.float16)
    key = rng.normal(size=(1, 8, cache_length, 52)).astype(np.float16)
    value = rng.normal(size=(1, 8, cache_length, 52)).astype(np.float16)
    output = moonshine_attention(query, key, value)
    assert output.shape == query.shape
    assert np.isfinite(output).all()


def test_moonshine_triple_projection_matches_hand_fixture() -> None:
    fixture = load_fixture()["triple_projection"]
    query, key, value = moonshine_triple_projection(
        f16(fixture["x"]),
        f16(fixture["q_weight"]),
        f16(fixture["k_weight"]),
        f16(fixture["v_weight"]),
    )
    np.testing.assert_array_equal(query, f16(fixture["expected_q"]))
    np.testing.assert_array_equal(key, f16(fixture["expected_k"]))
    np.testing.assert_array_equal(value, f16(fixture["expected_v"]))
    np.testing.assert_array_equal(
        moonshine_projection(f16(fixture["x"]), f16(fixture["q_weight"])),
        query,
    )


def test_moonshine_decoder_gated_mlp_matches_bias_and_split_order_fixture() -> None:
    fixture = load_fixture()["decoder_mlp"]
    actual = moonshine_decoder_mlp(
        f16(fixture["x"]),
        f16(fixture["fc1_weight"]),
        f16(fixture["fc1_bias"]),
        f16(fixture["fc2_weight"]),
        f16(fixture["fc2_bias"]),
    )
    np.testing.assert_array_equal(actual, f16(fixture["expected"]))


def test_moonshine_residual_rounds_at_fp16_boundary() -> None:
    left = np.asarray([[65_504.0, 1.0]], dtype=np.float16)
    right = np.asarray([[-65_504.0, 0.0003]], dtype=np.float16)
    actual = moonshine_residual(left, right)
    expected = np.add(left, right, dtype=np.float16)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float16


def test_moonshine_tied_lm_projection_and_lowest_id_argmax_match_fixture() -> None:
    fixture = load_fixture()["lm_head"]
    logits = moonshine_tied_lm_logits(
        f16(fixture["hidden"]), f16(fixture["tied_embedding_weight"])
    )
    np.testing.assert_array_equal(logits, f16(fixture["expected_logits"]))
    token = moonshine_stable_argmax(logits)
    np.testing.assert_array_equal(token, np.asarray(fixture["expected_token"], dtype=np.int64))
    fused_logits, fused_token = moonshine_lm_head_argmax(
        f16(fixture["hidden"]), f16(fixture["tied_embedding_weight"])
    )
    np.testing.assert_array_equal(fused_logits, logits)
    np.testing.assert_array_equal(fused_token, token)


def test_moonshine_oracles_reject_invalid_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="weight"):
        moonshine_layernorm(np.ones((1, 4), np.float16), np.ones(3, np.float16))
    with pytest.raises(ValueError, match="rotary_dim"):
        moonshine_apply_partial_rope(
            np.ones((1, 1, 1, 6), np.float16),
            np.ones((1, 1, 1, 6), np.float16),
            np.ones((1, 2), np.float16),
            np.zeros((1, 2), np.float16),
            position_ids=np.asarray([[0]]),
            rotary_dim=3,
        )
    with pytest.raises(ValueError, match="key length"):
        moonshine_attention(
            np.ones((1, 1, 1, 2), np.float16),
            np.empty((1, 1, 0, 2), np.float16),
            np.empty((1, 1, 0, 2), np.float16),
        )
    with pytest.raises(ValueError, match="finite"):
        moonshine_stable_argmax(np.asarray([[1.0, np.nan]], dtype=np.float32))


def test_moonshine_cpu_reference_registry_keys_resolve() -> None:
    cases = [
        ("moonshine_layernorm", "fp32_stats", "moonshine_layernorm"),
        ("moonshine_partial_rope", "interleaved", "moonshine_apply_partial_rope"),
        ("moonshine_self_cache", "fixed", "moonshine_fixed_cache_write"),
        ("moonshine_attention", "logical_head_dim", "moonshine_attention"),
        ("moonshine_projection", "fp32_accum", "moonshine_projection"),
        ("moonshine_qkv_proj", "triple", "moonshine_triple_projection"),
        ("moonshine_decoder_mlp", "gated_silu", "moonshine_decoder_mlp"),
        ("moonshine_residual", "rounded", "moonshine_residual"),
        ("moonshine_lm_head", "tied", "moonshine_tied_lm_logits"),
        ("moonshine_argmax", "lowest_id", "moonshine_stable_argmax"),
    ]
    for layer, variant, function_name in cases:
        kernel = resolve(
            backend="cpu_reference",
            layer=layer,
            quant="fp16",
            variant=variant,
        )
        assert kernel.__name__ == function_name
